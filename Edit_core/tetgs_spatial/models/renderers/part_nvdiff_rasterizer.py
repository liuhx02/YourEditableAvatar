from dataclasses import dataclass
import torch
import torch.nn.functional as F
import numpy as np
import tetgs_spatial
from tetgs_spatial.models.geometry.base import BaseImplicitGeometry
from tetgs_spatial.models.renderers.base import Renderer
from tetgs_spatial.utils.misc import get_device
from tetgs_spatial.utils.rasterize import NVDiffRasterizerContext
from tetgs_spatial.utils.typing import *
from tetgs_spatial.models.renderers.nvdiff_rasterize_utils import compute_normal


@tetgs_spatial.register("part-nvdiff-rasterizer")
class PartNVDiffRasterizer(Renderer):
    @dataclass
    class Config(Renderer.Config):
        context_type: str = "cuda"
        normal_type: str = 'camera'
        use_sdf_loss: bool = False
        use_additional_input: bool = False
        
    cfg: Config
    
    def configure(
        self,
        geometry: BaseImplicitGeometry
    ) -> None:
        super().configure(geometry)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())
        
    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        # camera_positions: Float[Tensor, "B 3"],
        # light_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        mvp_mtx_global: Float[Tensor, "B 4 4"] = None,
        c2w_global: Float[Tensor, "B 4 4"] = None,
        keep_info: Dict[str, Any] = None,
        grid_outputs: Dict[str, Any] = None,
        subdivision: bool = True,
        flag: int = 0,
        new_sdf = None,
        control_sdf=None,
        **kwargs
    ) -> Dict[str, Any]:
        
        if keep_info is not None and grid_outputs is not None:
            if self.cfg.use_sdf_loss:
                keep_mesh, update_mesh, update_sdf_loss, keep_sdf_loss, update_new_sdf = self.geometry.part_isosurface(
                    keep_info=keep_info,
                    grid_outputs=grid_outputs,
                    subdivision=subdivision,
                    update_new_sdf=new_sdf,
                    control_sdf=control_sdf
                )
            else:
                keep_mesh, update_mesh = self.geometry.part_isosurface(
                    keep_info=keep_info,
                    grid_outputs=grid_outputs,
                    subdivision=subdivision,
                    update_new_sdf=None,
                    control_sdf=None
                )
                update_sdf_loss = None
                keep_sdf_loss = None
                update_new_sdf = None
        else:
            if self.cfg.use_sdf_loss:
                update_mesh, sdf_loss = self.geometry.isosurface()
                keep_mesh = None
                update_sdf_loss = None
                keep_sdf_loss = None
                update_new_sdf = None
            else:
                update_mesh = self.geometry.isosurface()
                sdf_loss = None
                keep_mesh = None
                update_sdf_loss = None
                keep_sdf_loss = None
                update_new_sdf = None
        
        if keep_mesh is not None:
            keep_mesh_verts = keep_mesh.v_pos.detach()
            keep_mesh_faces = keep_mesh.t_pos_idx.detach()
            update_mesh_verts = update_mesh.v_pos
            update_mesh_faces = update_mesh.t_pos_idx
            # concat [keep, update]
            global_vertices = torch.cat((keep_mesh_verts, update_mesh_verts), dim=0)
            global_triangles = torch.cat((keep_mesh_faces, update_mesh_faces + len(keep_mesh_verts)), dim=0)
        else:
            global_vertices = update_mesh.v_pos
            global_triangles = update_mesh.t_pos_idx
            update_mesh_verts = global_vertices
            update_mesh_faces = global_triangles
        
        # ----------------- local update mesh -------------------------
        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            verts=update_mesh_verts,
            mvp_mtx=mvp_mtx,
        )
        update_faces = update_mesh_faces
        rast, _ = self.ctx.rasterize(v_pos_clip, update_faces, (height, width))
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, update_faces)
        
        out = {"opacity": mask_aa, "keep_mesh": keep_mesh, "update_mesh": update_mesh, "update_sdf_loss": update_sdf_loss, "keep_sdf_loss": keep_sdf_loss, "update_new_sdf": update_new_sdf}
        
        global_vn, _ = compute_normal(update_mesh_verts, update_faces)
        gb_normal, _ = self.ctx.interpolate_one(global_vn, rast, update_faces)
        
        if self.cfg.normal_type == 'world':
            gb_normal = F.normalize(gb_normal, dim=-1)
            gb_normal = torch.cat([gb_normal[:,:,:,1:2], gb_normal[:,:,:,2:3], gb_normal[:,:,:,0:1]], -1)
            gb_normal_aa = torch.lerp(
                torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
            )
        elif self.cfg.normal_type == 'camera':
            # world coord to cam coord
            gb_normal = gb_normal.view(-1, height*width, 3)
            gb_normal = torch.matmul(torch.linalg.inv(c2w[:,:3,:3]), gb_normal[0][:,:,None])

            gb_normal = torch.where(gb_normal[:, 2:3] < 0, -gb_normal, gb_normal)
            gb_normal = gb_normal.view(-1, height, width, 3)
            gb_normal = F.normalize(gb_normal, dim=-1)
            bg_normal = torch.zeros_like(gb_normal)
            gb_normal_aa = torch.lerp(
                bg_normal, (gb_normal + 1.0) / 2.0, mask.float()
            )
        else:
            raise ValueError(f"Unknown normal type: {self.cfg.normal_type}")
        
        gb_normal_aa = self.ctx.antialias(
            gb_normal_aa, rast, v_pos_clip, update_faces
        )
        out.update({"comp_normal": gb_normal_aa})  # in [0, 1]
        
        # depth
        gb_depth, _ = self.ctx.interpolate_one(v_pos_clip[0,:, :3].contiguous(), rast, update_faces)
        gb_depth = 1./(gb_depth[..., 2:3] + 1e-7)
        max_depth = torch.max(gb_depth[mask[..., 0]])
        min_depth = torch.min(gb_depth[mask[..., 0]])
        gb_depth_aa = torch.lerp(
                torch.zeros_like(gb_depth), (gb_depth - min_depth) / (max_depth - min_depth + 1e-7), mask.float()
            )
        gb_depth_aa = self.ctx.antialias(
            gb_depth_aa, rast, v_pos_clip, update_faces
        )
        out.update({"comp_depth": gb_depth_aa})  # in [0, 1]
        
        # ----------------- global mesh (keep + update) --------------------------
        if mvp_mtx_global is not None and c2w_global is not None and keep_mesh is not None:
            if isinstance(global_vertices, np.ndarray):
                global_vertices = torch.from_numpy(global_vertices).contiguous().float().to(self.device)
            if isinstance(global_triangles, np.ndarray):
                global_triangles = torch.from_numpy(global_triangles).contiguous().int().to(self.device)
                
            v_pos_clip_global: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
                global_vertices, mvp_mtx_global
            )
            rast_global, _ = self.ctx.rasterize(v_pos_clip_global, global_triangles, (height, width))
            mask_global = rast_global[..., 3:] > 0
            mask_global_aa = self.ctx.antialias(mask_global.float(), rast_global, v_pos_clip_global, global_triangles)
            out.update({"global_opacity": mask_global_aa})
            
            global_vn, _ = compute_normal(global_vertices, global_triangles)
            global_gb_normal, _ = self.ctx.interpolate_one(global_vn, rast_global, global_triangles)
            
            if self.cfg.normal_type == 'world':
                global_gb_normal = F.normalize(global_gb_normal, dim=-1)
                global_gb_normal = torch.cat([global_gb_normal[:,:,:,1:2], global_gb_normal[:,:,:,2:3], global_gb_normal[:,:,:,0:1]], -1)
                global_gb_normal_aa = torch.lerp(
                    torch.zeros_like(global_gb_normal), (global_gb_normal + 1.0) / 2.0, mask_global.float()
                )
            elif self.cfg.normal_type == 'camera':
                # world coord to cam coord
                global_gb_normal = global_gb_normal.view(-1, height * width, 3)
                global_gb_normal = torch.matmul(torch.linalg.inv(c2w_global[:, :3, :3]), global_gb_normal[0][:, :, None])
                global_gb_normal = global_gb_normal.view(-1, height, width, 3)
                global_gb_normal = F.normalize(global_gb_normal, dim=-1)
                bg_normal = torch.zeros_like(global_gb_normal)
                global_gb_normal_aa = torch.lerp(
                    bg_normal, (global_gb_normal + 1.0) / 2.0, mask_global.float()
                )
            else:
                raise ValueError(f"Unknown normal type: {self.cfg.normal_type}")
            
            global_gb_normal_aa = self.ctx.antialias(
                global_gb_normal_aa, rast_global, v_pos_clip_global, global_triangles
            )
            out.update({"global_comp_normal": global_gb_normal_aa})
            
        else:
            out.update({"global_opacity": mask_aa})
            out.update({"global_comp_normal": gb_normal_aa})

        # additional inputs such as masks & normals
        if self.cfg.use_additional_input:
            out.update({"input_opacity": None})
            out.update({"input_normal": None})
        else:
            out.update({"input_opacity": None})
            out.update({"input_normal": None})
        
        return out