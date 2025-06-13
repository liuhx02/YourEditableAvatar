from dataclasses import dataclass, field
import torch
import kaolin as kal
import tetgs_spatial
from tetgs_spatial.models.isosurface import (
    IsosurfaceHelper,
    MarchingTetrahedraHelper,
)
from tetgs_spatial.models.mesh import Mesh
from tetgs_spatial.utils.base import BaseModule
from tetgs_spatial.utils.ops import chunk_batch, scale_tensor
from tetgs_spatial.utils.typing import *

def contract_to_unisphere(
    x: Float[Tensor, "... 3"], bbox: Float[Tensor, "2 3"], unbounded: bool = False
) -> Float[Tensor, "... 3"]:
    if unbounded:
        x = scale_tensor(x, bbox, (0, 1))
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = x.norm(dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    else:
        x = scale_tensor(x, bbox, (0, 1))
    return x


class BaseGeometry(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pass
    cfg: Config

    @staticmethod
    def create_from(
        other: "BaseGeometry", cfg: Optional[Union[dict, DictConfig]] = None, **kwargs
    ) -> "BaseGeometry":
        raise TypeError(
            f"Cannot create {BaseGeometry.__name__} from {other.__class__.__name__}"
        )

    def export(self, *args, **kwargs) -> Dict[str, Any]:
        return {}


class BaseImplicitGeometry(BaseGeometry):
    @dataclass
    class Config(BaseGeometry.Config):
        radius: float = 1.0
        isosurface: bool = True
        isosurface_method: str = "mt"
        isosurface_resolution: int = 512
        isosurface_threshold: Union[float, str] = 0.0
        isosurface_chunk: int = 0
        isosurface_deformable_grid: bool = False
        use_sdf_loss: bool = False
        start_sdf_loss_step: int = 3000
        isosurface_subdive: bool = True
        isosurface_num_subdiv: int = 1
        per_update_compact_sdf: int = 250
    cfg: Config

    def configure(self) -> None:
        self.bbox: Float[Tensor, "2 3"]
        self.register_buffer(
            "bbox",
            torch.as_tensor(
                [
                    [-self.cfg.radius, -self.cfg.radius, -self.cfg.radius],
                    [self.cfg.radius, self.cfg.radius, self.cfg.radius],
                ],
                dtype=torch.float32,
                device=self.device
            ))
        self.isosurface_helper: Optional[IsosurfaceHelper] = None
        self.unbounded: bool = False

    def _initilize_isosurface_helper(self):
        if self.cfg.isosurface and self.isosurface_helper is None:
            if self.cfg.isosurface_method == "mt":
                self.isosurface_helper = MarchingTetrahedraHelper(
                    self.cfg.isosurface_resolution,
                    f"load/tets/{self.cfg.isosurface_resolution}_tets.npz").to(self.device)
            else:
                raise AttributeError("Unknown isosurface method {self.cfg.isosurface_method}")

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        raise NotImplementedError

    def forward_field(
        self, points: Float[Tensor, "*N Di"], sdf_type="full", frozen_sdf=None
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]], Optional[Float[Tensor, "*N 1"]]]:
        # return the value of the implicit field, could be density / signed distance
        # sdf_type: "full", "recon"
        raise NotImplementedError

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        # return the value of the implicit field, where the zero level set represents the surface
        raise NotImplementedError

    def _isosurface(self, bbox: Float[Tensor, "2 3"], fine_stage: bool = False) -> Mesh:
        def batch_func(x, sdf_type="full"):
            field, deformation, sdf_loss = self.forward_field(
                scale_tensor(x.to(bbox.device), self.isosurface_helper.points_range, bbox).to(self.device),
                sdf_type=sdf_type)
            field = field.to(x.device)
            if deformation is not None:
                deformation = deformation.to(x.device)
            return field, deformation, sdf_loss

        assert self.isosurface_helper is not None
        field, deformation, sdf_loss = chunk_batch(
            batch_func,
            self.cfg.isosurface_chunk,
            self.isosurface_helper.grid_vertices,
            "full",
        )

        threshold: float
        if isinstance(self.cfg.isosurface_threshold, float):
            threshold = self.cfg.isosurface_threshold
        elif self.cfg.isosurface_threshold == "auto":
            eps = 1.0e-5
            threshold = field[field > eps].mean().item()
            tetgs_spatial.info(f"Automatically determined isosurface threshold: {threshold}")
        else:
            raise TypeError(f"Unknown isosurface_threshold {self.cfg.isosurface_threshold}")

        level = self.forward_level(field, threshold)
        mesh: Mesh = self.isosurface_helper(level, deformation=deformation)
        mesh.v_pos = scale_tensor(
            mesh.v_pos, self.isosurface_helper.points_range, bbox
        )  # scale to bbox as the grid vertices are in [0, 1]
        mesh.add_extra("bbox", bbox)

        return mesh, sdf_loss
    
    def _isosurface_subdiv(self, bbox: Float[Tensor, "2 3"], num_subdiv=1):
        def batch_func(x, sdf_type="full"):
            field, deformation, sdf_loss = self.forward_field(
                scale_tensor(x.to(bbox.device), self.isosurface_helper.points_range, bbox),
                sdf_type=sdf_type)
            field = field.to(x.device)
            if deformation is not None:
                deformation = deformation.to(x.device)
            return field, deformation, sdf_loss
        
        assert self.isosurface_helper is not None
        
        # coarse
        field, deformation, sdf_loss = chunk_batch(
            batch_func,
            self.cfg.isosurface_chunk,
            self.isosurface_helper.grid_vertices,
            "full",
        )
        
        threshold: float
        if isinstance(self.cfg.isosurface_threshold, float):
            threshold = self.cfg.isosurface_threshold
        elif self.cfg.isosurface_threshold == "auto":
            eps = 1.0e-5
            threshold = field[field > eps].mean().item()
            tetgs_spatial.info(f"Automatically determined isosurface threshold: {threshold}")
        else:
            raise TypeError(f"Unknown isosurface_threshold {self.cfg.isosurface_threshold}")

        sdf = self.forward_level(field, threshold)
        
        if deformation is None:
            verts_deformed = self.isosurface_helper.grid_vertices
        tet = self.isosurface_helper.indices
        for i in range(num_subdiv):
            verts_deformed, _, tet, _, new_tet_idx_to_old = self.isosurface_helper.compact_tets(
                verts_deformed, 
                sdf,
                tet
            )
            # subdivide tets
            verts_deformed, tet, _, sub_tet_to_parent_idx = self.isosurface_helper.batch_subdivide_volume(
                verts_deformed.unsqueeze(0),  # [1, N, 3]
                tet.unsqueeze(0)   # [1, f, 4]
            )
            verts_deformed = verts_deformed[0]
            tet = tet[0]
            field, deformation, sdf_loss = chunk_batch(
                batch_func,
                self.cfg.isosurface_chunk,
                verts_deformed,
                "full",
            )
            sdf = self.forward_level(field, threshold)
            
        # extract mesh
        outputs = self.isosurface_helper._forward(verts_deformed, sdf, tet)
        mesh_verts, mesh_faces = outputs["verts"], outputs["faces"]
        face_to_sub_tet_idx = outputs["face_to_tet_idx"]
        face_to_tet_idx = sub_tet_to_parent_idx[face_to_sub_tet_idx]
        face_to_global_tet_idx = new_tet_idx_to_old[face_to_tet_idx]
        
        mesh = Mesh(
            v_pos=mesh_verts,
            t_pos_idx=mesh_faces,
            grid_vertices=verts_deformed,
            grid_level=sdf,
            grid_deformation=deformation,
            face_to_global_tet_idx=face_to_global_tet_idx
        )
        mesh.v_pos = scale_tensor(mesh.v_pos, self.isosurface_helper.points_range, bbox)
        mesh.add_extra("bbox", bbox)
        return mesh, sdf_loss  
    
    def isosurface(self) -> Mesh:
        if not self.cfg.isosurface:
            raise NotImplementedError("Isosurface is not enabled in the current configuration")
        self._initilize_isosurface_helper()
        if self.cfg.isosurface_subdive:
            mesh, sdf_loss = self._isosurface_subdiv(self.bbox, self.cfg.isosurface_num_subdiv)
        else:
            mesh, sdf_loss = self._isosurface(self.bbox)
        if self.cfg.use_sdf_loss:
            return mesh, sdf_loss
        else:
            return mesh
        
    def _part_isosurface_init(self, mask, bbox: Float[Tensor, "2 3"]):
        def batch_func(x, sdf_type="full"):
            field, deformation, sdf_loss = self.forward_field(
                scale_tensor(x.to(bbox.device), self.isosurface_helper.points_range, bbox
                ).to(self.device),
                sdf_type=sdf_type)
            field = field.to(x.device)  
            if deformation is not None:
                deformation = deformation.to(x.device)
            return field, deformation, sdf_loss
        
        def prev_batch_func(x, sdf_type="full"):
            field, deformation, sdf_loss = self.prev_geometry.forward_field(
                scale_tensor(x.to(bbox.device), self.isosurface_helper.points_range, bbox
                ).to(self.device),
                sdf_type=sdf_type)
            field = field.to(x.device)  
            if deformation is not None:
                deformation = deformation.to(x.device)
            return field, deformation, sdf_loss
        
        assert self.isosurface_helper is not None
        
        # query reconstructed sdf for irrelevant part
        field, deformation, _ = chunk_batch(
            prev_batch_func,
            self.cfg.isosurface_chunk,
            self.isosurface_helper.grid_vertices,
            "full",
        )
        threshold: float
        if isinstance(self.cfg.isosurface_threshold, float):
            threshold = self.cfg.isosurface_threshold
        elif self.cfg.isosurface_threshold == "auto":
            eps = 1.0e-5
            threshold = field[field > eps].mean().item()
            tetgs_spatial.info(f"Automatically determined isosurface threshold: {threshold}")
        else:
            raise TypeError(f"Unknown isosurface_threshold {self.cfg.isosurface_threshold}")
        sdf = self.forward_level(field, threshold)
        
        # devide tets into editing group and non-editing group
        outputs = self.isosurface_helper.mark_part_tets(
            deformation=deformation,
            level=sdf,
            edit_mask=mask,
        )
        keep_pos = outputs["keep_pos"]
        keep_tets = outputs["keep_tets"]
        part_outputs = {}
        part_outputs.update({"grid_outputs": outputs})
        
        # -------------- prepare non-editing info for faster query during editing ------------------
        # subdivide to capture high-frequency details
        verts_deformed, tet, _, _ = self.isosurface_helper.batch_subdivide_volume(
            keep_pos.unsqueeze(0),  # [1, N, 3]
            keep_tets.unsqueeze(0),  # [1, f, 4]
        )
        verts_deformed = verts_deformed[0]
        tet = tet[0]
        # fine
        fine_field, _, _ = chunk_batch(
            batch_func,
            self.cfg.isosurface_chunk,
            verts_deformed,
            "recon",
        )
        keep_fine_sdf = self.forward_level(fine_field, threshold)
        # save useful informations of non-editing part in keep_info
        keep_info = {}
        keep_info['verts_deformed'] = verts_deformed 
        keep_info['fine_sdf'] = keep_fine_sdf 
        keep_info['tet'] = tet
        part_outputs.update({"keep_info": keep_info})
        
        return part_outputs

    def _part_isosurface(self, keep_info, grid_outputs, bbox: Float[Tensor, "2 3"], subdivision=True, new_sdf=None, control_sdf=None):
        def batch_func(x, sdf_type="full", frozen_sdf=None):
            field, deformation, sdf_loss = self.forward_field(
                scale_tensor(x.to(bbox.device), self.isosurface_helper.points_range, bbox),
                sdf_type=sdf_type, frozen_sdf=frozen_sdf)
            field = field.to(x.device)
            if deformation is not None:
                deformation = deformation.to(x.device)
            return field, deformation, sdf_loss
        
        def prev_batch_func(x, sdf_type="full"):
            field, deformation, sdf_loss = self.prev_geometry.forward_field(
                scale_tensor(x.to(bbox.device), self.isosurface_helper.points_range, bbox).to(self.device),
                sdf_type=sdf_type)
            field = field.to(x.device)
            if deformation is not None:
                deformation = deformation.to(x.device)
            return field, deformation, sdf_loss
        
        assert self.isosurface_helper is not None
        threshold: float
        if isinstance(self.cfg.isosurface_threshold, float):
            threshold = self.cfg.isosurface_threshold
        elif self.cfg.isosurface_threshold == "auto":
            eps = 1.0e-5
            threshold = field[field > eps].mean().item()
            tetgs_spatial.info(f"Automatically determined isosurface threshold: {threshold}")
        else:
            raise TypeError(f"Unknown isosurface_threshold {self.cfg.isosurface_threshold}")
            
        # keep info to query non-editing surface
        keep_verts_deformed = keep_info['verts_deformed']
        keep_fine_sdf = keep_info['fine_sdf']
        keep_tet = keep_info['tet']
        # update info to query editing surface
        new_pos = grid_outputs["new_pos"]
        if new_sdf is None:
            new_sdf = grid_outputs["new_sdf"]
        new_tets = grid_outputs["new_tets"]
        mask_keep_part_in_new_pos = grid_outputs["mask_keep_part_in_new_pos"]
        
        # non-editing surface extraction
        keep_mesh_verts, keep_mesh_faces = kal.ops.conversions.marching_tetrahedra(
            vertices=keep_verts_deformed.unsqueeze(0),
            tets=keep_tet,
            sdf=keep_fine_sdf.unsqueeze(0)
        )
        keep_mesh_verts, keep_mesh_faces = keep_mesh_verts[0], keep_mesh_faces[0]
        keep_mesh = Mesh(
            v_pos=keep_mesh_verts,
            t_pos_idx=keep_mesh_faces,
            grid_vertices=keep_verts_deformed,
            grid_level=keep_fine_sdf,
            grid_deformation=None,
            face_to_global_tet_idx=None
        )
        keep_mesh.v_pos = scale_tensor(keep_mesh.v_pos, self.isosurface_helper.points_range, bbox)
        keep_mesh.add_extra("bbox", bbox)
        # calculate surface-aware loss for non-editing tets
        _, _, keep_sdf_loss = chunk_batch(
            batch_func,
            self.cfg.isosurface_chunk,
            keep_verts_deformed,
            "recon",
            keep_info['fine_sdf']
        )
        
        # editing surface extraction
        update_verts_coarse = grid_outputs["new_pos"]
        tet_coarse = grid_outputs["new_tets"]
        mask = mask_keep_part_in_new_pos
        sdf_values_coarse = torch.full((update_verts_coarse.shape[0], 1), float('nan'), dtype=torch.float32, device=update_verts_coarse.device)
        deformation_values_coarse = torch.full((update_verts_coarse.shape[0], 1), float(0.0), dtype=torch.float32, device=update_verts_coarse.device)
        batch_func_idx = (mask == 0)
        prev_idx = (mask == 1)
        
        if batch_func_idx.any():
            batch_func_verts = update_verts_coarse
            # Add a local sdf_loss following HUMANNORM[CVPR 2024] for better shape control
            new_field, _, new_sdf_loss = chunk_batch(
                batch_func,
                self.cfg.isosurface_chunk,
                batch_func_verts,
                "full",
                control_sdf
            )
            new_sdf_coarse = self.forward_level(new_field, threshold)
            sdf_values_coarse[batch_func_idx[..., 0], :] = new_sdf_coarse[batch_func_idx[..., 0], :]
            
        if prev_idx.any():
            with torch.no_grad():
                prev_verts = update_verts_coarse[prev_idx[..., 0], :]
                prev_field, _, _ = chunk_batch(
                    prev_batch_func,
                    self.cfg.isosurface_chunk,
                    prev_verts,
                    "full",
                )
                prev_sdf = self.prev_geometry.forward_level(prev_field, threshold)
                sdf_values_coarse[prev_idx[..., 0], :] = prev_sdf
        # update new_sdf for compact_tets
        new_sdf = sdf_values_coarse
        
        if subdivision:   
            # compact tets for effcient tetrahedron subdivision
            new_pos, _, new_tets, new_mask, new_tet_idx_to_old = self.isosurface_helper.compact_tets(
                pos_nx3=new_pos,
                sdf_n=new_sdf, 
                tet_fx4=new_tets,
                mask_keep_part_in_new_pos=mask_keep_part_in_new_pos,
            )
            # subdivision
            update_verts_deformed, tet, new_mask_sub, _ = self.isosurface_helper.batch_subdivide_volume(
                tet_pos_bxnx3=new_pos.unsqueeze(0), 
                tet_bxfx4=new_tets.unsqueeze(0), 
                mask_keep_part_in_new_pos=new_mask.unsqueeze(0),
            )
            update_verts_deformed = update_verts_deformed[0]
            tet = tet[0]
            new_mask_sub = new_mask_sub[0]
            # special operation where updatable_verts overlap with keep_verts
            # mask==0: query current mlp; mask==1: query reconstructed mlp
            sdf_values = torch.full((update_verts_deformed.shape[0], 1), float('nan'), dtype=torch.float32, device=update_verts_deformed.device)
            deformation_values = None
            batch_func_idx = (new_mask_sub == 0)
            prev_idx = (new_mask_sub == 1)
            
            if batch_func_idx.any():
                batch_func_verts = update_verts_deformed[batch_func_idx[..., 0], :]
                new_field, _, _ = chunk_batch(
                    batch_func,
                    self.cfg.isosurface_chunk,
                    batch_func_verts,
                    "full",
                )
                fine_sdf = self.forward_level(new_field, threshold)
                sdf_values[batch_func_idx[..., 0], :] = fine_sdf
            
            if prev_idx.any():
                with torch.no_grad():
                    prev_verts = update_verts_deformed[prev_idx[..., 0], :]
                    prev_field, _, _ = chunk_batch(
                        prev_batch_func,
                        self.cfg.isosurface_chunk,
                        prev_verts,
                        "full",
                    )
                    prev_sdf = self.prev_geometry.forward_level(prev_field, threshold)
                    sdf_values[prev_idx[..., 0], :] = prev_sdf
        
        if not subdivision:
            update_verts_deformed = update_verts_coarse
            tet = tet_coarse
            sdf_values = sdf_values_coarse
            deformation_values = deformation_values_coarse
        
        # extract editing surface
        update_mesh_verts, update_mesh_faces = kal.ops.conversions.marching_tetrahedra(
            vertices=update_verts_deformed.unsqueeze(0),
            tets=tet,
            sdf=sdf_values.unsqueeze(0),
        )
        update_mesh_verts, update_mesh_faces = update_mesh_verts[0], update_mesh_faces[0]
        update_mesh = Mesh(
            v_pos=update_mesh_verts,
            t_pos_idx=update_mesh_faces,
            grid_vertices=update_verts_deformed,
            grid_level=sdf_values,
            grid_deformation=deformation_values
        )
        update_mesh.v_pos = scale_tensor(
            update_mesh.v_pos,
            self.isosurface_helper.points_range,
            bbox
        )
        update_mesh.add_extra("bbox", bbox)
        
        return keep_mesh, update_mesh, new_sdf_loss, keep_sdf_loss, new_sdf
    
    def part_isosurface_init(self, mask):
        if not self.cfg.isosurface:
            raise NotImplementedError("Isosurface is not enabled in the current configuration")
        self._initilize_isosurface_helper()
        outputs = self._part_isosurface_init(mask, self.bbox)
        return outputs
    
    def part_isosurface(self, keep_info, grid_outputs, subdivision, update_new_sdf=None, control_sdf=None):
        if not self.cfg.isosurface:
            raise NotImplementedError("Isosurface is not enabled in the current configuration")
        self._initilize_isosurface_helper()
        keep_mesh, update_mesh, update_sdf_loss, keep_sdf_loss, new_sdf = self._part_isosurface(
            keep_info=keep_info,
            grid_outputs=grid_outputs,
            bbox=self.bbox,
            subdivision=subdivision,
            new_sdf=update_new_sdf,
            control_sdf=control_sdf
        )
        if self.cfg.use_sdf_loss:
            return keep_mesh, update_mesh, update_sdf_loss, keep_sdf_loss, new_sdf
        else:
            return keep_mesh, update_mesh


class BaseExplicitGeometry(BaseGeometry):
    @dataclass
    class Config(BaseGeometry.Config):
        radius: float = 1.0
    cfg: Config

    def configure(self) -> None:
        self.bbox: Float[Tensor, "2 3"]
        self.register_buffer(
            "bbox",
            torch.as_tensor(
                [
                    [-self.cfg.radius, -self.cfg.radius, -self.cfg.radius],
                    [self.cfg.radius, self.cfg.radius, self.cfg.radius],
                ],
                dtype=torch.float32))