from dataclasses import dataclass
import numpy as np
import torch
import trimesh
import pymeshlab as pml
import tetgs_spatial
from tetgs_spatial.models.mesh import Mesh
from tetgs_spatial.models.exporters.base import Exporter, ExporterOutput
from tetgs_spatial.models.geometry.implicit_sdf import ImplicitSDF
from tetgs_spatial.utils.typing import *
from tetgs_spatial.utils.ops import chunk_batch, scale_tensor


@tetgs_spatial.register("mesh-exporter-part")
class MeshExporterGeoOnlyNew(Exporter):
    @dataclass
    class Config(Exporter.Config):
        context_type: str = "cuda"
        
    cfg: Config
    
    def configure(
        self,
        geometry: ImplicitSDF,
        mask_npy_path: str="",
    ) -> None:
        @dataclass
        class SubModules:
            geometry: ImplicitSDF
            mask_npy_path: str
            
        self.sub_modules = SubModules(
            geometry,
            mask_npy_path
        )
        self.prev_geometry: ImplicitSDF = self.geometry.prev_geometry
        
        self.mask_npy_path = mask_npy_path
        mask_info = np.load(self.mask_npy_path, allow_pickle=True).item()
        face_mask = mask_info['editing_mask_faces']
        vert_mask = mask_info['editing_mask']
        self.mask = torch.from_numpy(face_mask).to(self.device)
        self.vert_mask = torch.from_numpy(vert_mask).to(self.device)
        
        # hard-coded
        self.radius = 1.0
        self.bbox = torch.as_tensor(
            [
                [-self.radius, -self.radius, -self.radius],
                [self.radius, self.radius, self.radius],
            ],
            dtype=torch.float32,
        ).to(self.device)
        
        
    def __call__(self) -> List[ExporterOutput]:
        def prev_batch_func(x, sdf_type="full"):
            field, deformation, _ = self.prev_geometry.forward_field(
                scale_tensor(x.to(self.bbox.device), self.prev_geometry.isosurface_helper.points_range, self.bbox), 
                sdf_type=sdf_type,
            )
            field = field.to(x.device)
            if deformation is not None:
                deformation = deformation.to(x.device)
            return field
        
        self.prev_geometry._initilize_isosurface_helper()
        field = chunk_batch(
            prev_batch_func,
            self.prev_geometry.cfg.isosurface_chunk,
            self.prev_geometry.isosurface_helper.grid_vertices,
            "full",
        )
        threshold: float
        if isinstance(self.prev_geometry.cfg.isosurface_threshold, float):
            threshold = self.prev_geometry.cfg.isosurface_threshold
        elif self.prev_geometry.cfg.isosurface_threshold == "auto":
            eps = 1.0e-5
            threshold = field[field > eps].mean().item()
            tetgs_spatial.info(
                f"Automatically determined isosurface threshold: {threshold}"
            )
        else:
            raise TypeError(
                f"Unknown isosurface_threshold {self.prev_geometry.cfg.isosurface_threshold}"
            )
        prev_sdf = self.prev_geometry.forward_level(field, threshold)
        
        # grid_outputs
        grid_outputs = self.prev_geometry.isosurface_helper.mark_part_tets(
            deformation=None,
            level=prev_sdf,
            edit_mask=self.mask,
        )
        keep_pos = grid_outputs["keep_pos"]
        keep_tets = grid_outputs["keep_tets"]
        keep_tet_idx = grid_outputs["keep_tet_idx"]

        # keep_info
        verts_deformed, tet, _, sub_tet_to_parent_idx = self.prev_geometry.isosurface_helper.batch_subdivide_volume(
            keep_pos.unsqueeze(0),  # [1, N, 3]
            keep_tets.unsqueeze(0),  # [1, f, 4]
        )
        verts_deformed = verts_deformed[0]
        tet = tet[0]
        keep_fine_field = chunk_batch(
            prev_batch_func,
            self.prev_geometry.cfg.isosurface_chunk,
            verts_deformed,
            "recon",
        )
        keep_fine_sdf = self.prev_geometry.forward_level(keep_fine_field, threshold)
        keep_info = {}
        keep_info['verts_deformed'] = verts_deformed
        keep_info['fine_sdf'] = keep_fine_sdf
        keep_info['tet'] = tet
        keep_outputs = self.prev_geometry.isosurface_helper._forward(verts_deformed, keep_fine_sdf, tet)
        keep_mesh_verts, keep_mesh_faces = keep_outputs["verts"], keep_outputs["faces"]
        keep_mesh_verts = scale_tensor(keep_mesh_verts, self.prev_geometry.isosurface_helper.points_range, self.prev_geometry.bbox)
        face_to_sub_tet_idx = keep_outputs["face_to_tet_idx"]
        face_to_tet_idx = sub_tet_to_parent_idx[face_to_sub_tet_idx]
        face_to_global_tet_idx = keep_tet_idx[face_to_tet_idx]
        
        mesh = Mesh(
            v_pos=keep_mesh_verts,
            t_pos_idx=keep_mesh_faces,
            grid_vertices=verts_deformed,
            grid_level=keep_fine_sdf,
            grid_deformation=None,
            face_to_global_tet_idx=face_to_global_tet_idx
        )
        mesh.v_pos = scale_tensor(mesh.v_pos, self.prev_geometry.isosurface_helper.points_range, self.prev_geometry.bbox)
        
        # extract mesh after editing
        self.geometry.isosurface_helper = self.prev_geometry.isosurface_helper
        if self.geometry.cfg.use_sdf_loss:
            _, edit_mesh, _, _, _ = self.geometry.part_isosurface(
                keep_info=keep_info,
                grid_outputs=grid_outputs,
                subdivision=True,
            )
        else:
            _, edit_mesh = self.geometry.part_isosurface(
                keep_info=keep_info,
                grid_outputs=grid_outputs,
                subdivision=True,
            )
            
        keep_vertices = keep_mesh_verts.contiguous().float().to(self.device)
        edit_vertices = edit_mesh.v_pos.contiguous().float().to(self.device)
        keep_faces = keep_mesh_faces.contiguous().int().to(self.device)
        edit_faces = edit_mesh.t_pos_idx.contiguous().int().to(self.device)
        # concat keep mesh and edit mesh
        full_vertices = torch.cat((keep_vertices, edit_vertices), dim=0)
        full_faces = torch.cat((keep_faces, edit_faces + len(keep_vertices)), dim=0)
        full_edit_mesh = trimesh.Trimesh(
            full_vertices.detach().cpu().numpy(),
            full_faces.detach().cpu().numpy(),
            process=False
        )
        full_edit_mesh = self.geometry.convert_mesh(full_edit_mesh)
        
        # 清理edit_vertices中的漂浮物
        m = pml.Mesh(edit_vertices.detach().cpu().numpy(), edit_faces.detach().cpu().numpy())
        ms = pml.MeshSet()
        ms.add_mesh(m, 'mesh')
        face_nums = edit_faces.shape[0]
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=int(face_nums * 0.1))
        pymesh = ms.current_mesh()
        masked_vertices_set = set(map(tuple, pymesh.vertex_matrix()))
        edit_mask = np.array([tuple(vertex) in masked_vertices_set for vertex in full_vertices.detach().cpu().numpy()])
        edit_mask = torch.from_numpy(edit_mask).int()

        full_mesh_data = {
            'vertices': full_edit_mesh.vertices,
            'faces': full_edit_mesh.faces,
            'face_to_global_tet_idx': face_to_global_tet_idx,
            'keep_vertices_num': keep_vertices.shape[0],
            'keep_faces_num': keep_faces.shape[0],
            'editing_mask': edit_mask
        }
        params = {
            "mesh": full_mesh_data,
        }
        
        comm_cf = self.geometry.comm_cfg
        edit_save_name = comm_cf["edit_mesh"]
        return [
            ExporterOutput(
                save_name=f"{edit_save_name}.npy", save_type="npy", params=params
            )
        ]