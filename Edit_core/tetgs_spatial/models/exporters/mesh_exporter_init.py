from dataclasses import dataclass
import torch
import trimesh
import tetgs_spatial
from tetgs_spatial.models.exporters.base import Exporter, ExporterOutput
from tetgs_spatial.models.geometry.base import BaseImplicitGeometry
from tetgs_spatial.models.geometry.implicit_sdf import ImplicitSDF
from tetgs_spatial.utils.typing import *


@tetgs_spatial.register("mesh-exporter-init")
class MeshExporterGeoOnly(Exporter):
    @dataclass
    class Config(Exporter.Config):
        context_type: str = "cuda"
        
    cfg: Config
    
    def configure(
        self,
        geometry: BaseImplicitGeometry,
        mask_npy_path: str="",
    ) -> None:
        @dataclass
        class SubModules:
            geometry: BaseImplicitGeometry
            mask_npy_path: str
        
        self.sub_modules = SubModules(
            geometry,
            mask_npy_path
        )
        
        # hard-coded
        self.radius = 1.0
        self.bbox = torch.as_tensor(
            [
                [-self.radius, -self.radius, -self.radius],
                [self.radius, self.radius, self.radius],
            ],
            dtype=torch.float32,
        ).to(self.device)
        
        self.prev_geometry: ImplicitSDF = self.geometry.prev_geometry
        
        
    def __call__(self) -> List[ExporterOutput]:
        self.prev_geometry._initilize_isosurface_helper()
        full_mesh_save = self.prev_geometry.isosurface()
        full_mesh = trimesh.Trimesh(
            full_mesh_save.v_pos.detach().cpu().numpy(),
            full_mesh_save.t_pos_idx.detach().cpu().numpy(),
            process=False,
        )
        full_mesh = self.geometry.convert_mesh(full_mesh)
        full_mesh_data = {
            'vertices': full_mesh.vertices,
            'faces': full_mesh.faces,
            'face_to_global_tet_idx': full_mesh_save.face_to_global_tet_idx.detach().cpu().numpy()
        }
        params = {
            "mesh": full_mesh_data
        }
        
        coarse_mesh_save, _ = self.prev_geometry._isosurface(self.bbox)
        coarse_mesh = trimesh.Trimesh(
            coarse_mesh_save.v_pos.detach().cpu().numpy(),
            coarse_mesh_save.t_pos_idx.detach().cpu().numpy(),
            process=False
        )
        coarse_mesh = self.geometry.convert_mesh(coarse_mesh)
        params_coarse = {
            "mesh": coarse_mesh
        }
        
        comm_cf = self.geometry.comm_cfg
        full_save_name = comm_cf["init_mesh"]
        coarse_save_name = comm_cf["init_mesh_coarse"]
        return [
            ExporterOutput(
                save_name=f"{coarse_save_name}.ply", save_type="ply", params=params_coarse
            ),
            ExporterOutput(
                save_name=f"{full_save_name}.npy", save_type="npy", params=params
            ),
        ]