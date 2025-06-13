import os
import random
from dataclasses import dataclass, field
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import trimesh
import tetgs_spatial
from tetgs_spatial.models.geometry.base import BaseImplicitGeometry, contract_to_unisphere
from tetgs_spatial.models.networks import get_encoding, get_mlp
from tetgs_spatial.utils.misc import broadcast, get_rank
from tetgs_spatial.utils.typing import *
from tetgs_spatial.utils.rasterize import NVDiffRasterizerContext
from tetgs_spatial.models.renderers.nvdiff_rasterize_utils import compute_normal
from tetgs_spatial.utils.cameras import gen_camera_params


@tetgs_spatial.register("implicit-sdf")
class ImplicitSDF(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 0
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.381912879967776,
                "start_level": 8 })
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1 })
        normal_type: Optional[
            str
        ] = "finite_difference"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']
        finite_difference_normal_eps: Union[
            float, str
        ] = 0.01  # in [float, "progressive"]
        shape_init: Optional[str] = None
        force_shape_init: bool = False
        sdf_bias: Union[float, str] = 0.0
        sdf_bias_params: Optional[Any] = None
        init_per_val: int = 100
        init_debug: bool = False
        comm_cfg: str = "../comm_config.yaml"
        
        # for normal shape init
        height: Any = 1024
        width: Any = 1024
        elevation_range: Tuple[float, float] = (-10, 10)
        azimuth_range: Tuple[float, float] = (-180, 180)
        camera_distance_range: Tuple[float, float] = (3, 3.3)
        fovy_range: Tuple[float, float] = (40, 45)

    cfg: Config
    
    def configure(
        self,
        prev_geometry: BaseImplicitGeometry = None,
    ) -> None:
        
        super().configure()
        
        @dataclass
        class SubModules:
            prev_geometry: BaseImplicitGeometry
        self.sub_modules = SubModules(prev_geometry)
        
        if prev_geometry is None:
            self.prev_geometry = None
        else:
            self.prev_geometry = self.sub_modules.prev_geometry
        self.encoding = get_encoding(
            self.cfg.n_input_dims, self.cfg.pos_encoding_config)
        self.sdf_network = get_mlp(
            self.encoding.n_output_dims, 1, self.cfg.mlp_network_config
        ).to(self.device)

        if self.cfg.n_feature_dims > 0:
            self.feature_network = get_mlp(
                self.encoding.n_output_dims,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            )

        if self.cfg.normal_type == "pred":
            self.normal_network = get_mlp(
                self.encoding.n_output_dims, 3, self.cfg.mlp_network_config
            )
        if self.cfg.isosurface_deformable_grid:
            assert (
                self.cfg.isosurface_method == "mt"
            ), "isosurface_deformable_grid only works with mt"
            self.deformation_network = get_mlp(
                self.encoding.n_output_dims, 3, self.cfg.mlp_network_config
            )

        self.finite_difference_normal_eps: Optional[float] = None
        self.start_sdf_recon = False
        self.start_control_sdf = False
        self.comm_cfg = OmegaConf.load(self.cfg.comm_cfg)
        
    def convert_mesh(self, res_mesh: trimesh.Trimesh) -> None:
        assert isinstance(self.comm_cfg.shape_init_params, float)
        mesh_path = self.cfg.shape_init
        anchor_path = self.comm_cfg.anchor_path

        if not os.path.exists(mesh_path):
            raise ValueError(f"Mesh file {mesh_path} does not exist.")
        scene = trimesh.load(mesh_path)
        anchor_scene = trimesh.load(anchor_path)
        scene.invert()
        if isinstance(scene, trimesh.Trimesh):
            mesh = scene
        elif isinstance(scene, trimesh.scene.Scene):
            mesh = trimesh.Trimesh()
            for obj in scene.geometry.values():
                mesh = trimesh.util.concatenate([mesh, obj])
        else:
            raise ValueError(f"Unknown mesh type at {mesh_path}.")
        
        # move to center
        centroid = anchor_scene.vertices.mean(0)
        mesh.vertices = mesh.vertices - centroid
        # adjust the position of mesh
        mesh.vertices[:,1] = mesh.vertices[:,1] + 0.3
        # calculate scale
        scale = np.abs(mesh.vertices).max()
        
        res_verts = res_mesh.vertices
        res_faces = res_mesh.faces
        res_verts = res_verts * scale / self.comm_cfg.shape_init_params
        
        dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
        dir2vec = {
            "+x": np.array([1, 0, 0]),
            "+y": np.array([0, 1, 0]),
            "+z": np.array([0, 0, 1]),
            "-x": np.array([-1, 0, 0]),
            "-y": np.array([0, -1, 0]),
            "-z": np.array([0, 0, -1]),}
        if (self.comm_cfg.shape_init_mesh_up not in dirs
            or self.comm_cfg.shape_init_mesh_front not in dirs):
            raise ValueError(f"shape_init_mesh_up and shape_init_mesh_front must be one of {dirs}.")
        if self.comm_cfg.shape_init_mesh_up[1] == self.comm_cfg.shape_init_mesh_front[1]:
            raise ValueError("shape_init_mesh_up and shape_init_mesh_front must be orthogonal.")
        z_, x_ = (dir2vec[self.comm_cfg.shape_init_mesh_up],
                    dir2vec[self.comm_cfg.shape_init_mesh_front],)
        y_ = np.cross(z_, x_)
        std2mesh = np.stack([x_, y_, z_], axis=0).T
        
        res_verts = np.dot(std2mesh, res_verts.T).T
        res_verts[:,1] = res_verts[:,1] - 0.3
        res_verts = res_verts + centroid
            
        res_mesh = trimesh.Trimesh(
            vertices=res_verts,
            faces=res_faces,
            process=False,
        )
        return res_mesh
    
    def initialize_shape(self) -> None:
        if self.cfg.shape_init is None and not self.cfg.force_shape_init:
            return
        if self.cfg.weights is not None and not self.cfg.force_shape_init:
            return
        if self.cfg.sdf_bias != 0.0:
            tetgs_spatial.warn("shape_init and sdf_bias are both specified, which may lead to unexpected results.")

        get_gt_sdf: Callable[[Float[Tensor, "N 3"]], Float[Tensor, "N 1"]]
        assert isinstance(self.cfg.shape_init, str)
        assert isinstance(self.comm_cfg.shape_init_params, float)
        mesh_path = self.cfg.shape_init
        anchor_path = self.comm_cfg.anchor_path
        if not os.path.exists(mesh_path):
            raise ValueError(f"Mesh file {mesh_path} does not exist.")

        import trimesh
        scene = trimesh.load(mesh_path)
        anchor_scene = trimesh.load(anchor_path)
        scene.invert()
        if isinstance(scene, trimesh.Trimesh):
            mesh = scene
        elif isinstance(scene, trimesh.scene.Scene):
            mesh = trimesh.Trimesh()
            for obj in scene.geometry.values():
                mesh = trimesh.util.concatenate([mesh, obj])
        else:
            raise ValueError(f"Unknown mesh type at {mesh_path}.")

        # move to center
        centroid = anchor_scene.vertices.mean(0)
        mesh.vertices = mesh.vertices - centroid
        # adjust the position of mesh
        mesh.vertices[:,1] = mesh.vertices[:,1] + 0.3
        # align to up-z and front-x
        dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
        dir2vec = {
            "+x": np.array([1, 0, 0]),
            "+y": np.array([0, 1, 0]),
            "+z": np.array([0, 0, 1]),
            "-x": np.array([-1, 0, 0]),
            "-y": np.array([0, -1, 0]),
            "-z": np.array([0, 0, -1]),}
        if (self.comm_cfg.shape_init_mesh_up not in dirs
            or self.comm_cfg.shape_init_mesh_front not in dirs):
            raise ValueError(f"shape_init_mesh_up and shape_init_mesh_front must be one of {dirs}.")
        if self.comm_cfg.shape_init_mesh_up[1] == self.comm_cfg.shape_init_mesh_front[1]:
            raise ValueError("shape_init_mesh_up and shape_init_mesh_front must be orthogonal.")
        z_, x_ = (dir2vec[self.comm_cfg.shape_init_mesh_up],
                  dir2vec[self.comm_cfg.shape_init_mesh_front],)
        y_ = np.cross(z_, x_)
        std2mesh = np.stack([x_, y_, z_], axis=0).T
        mesh2std = np.linalg.inv(std2mesh)

        # scaling
        scale = np.abs(mesh.vertices).max()
        mesh.vertices = mesh.vertices / scale * self.comm_cfg.shape_init_params
        mesh.vertices = np.dot(mesh2std, mesh.vertices.T).T

        from pysdf import SDF
        sdf = SDF(mesh.vertices, mesh.faces)
        def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
            # add a negative signed here
            # as in pysdf the inside of the shape has positive signed distance
            return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(
                points_rand
            )[..., None]
        get_gt_sdf = func
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        from tqdm import tqdm
        for _ in tqdm(
            range(15000),
            desc=f"Initializing SDF to a(n) {self.cfg.shape_init}:",
            disable=get_rank() != 0,
        ):
            points_rand = (torch.rand((40000, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0)
            sdf_gt = get_gt_sdf(points_rand)
            sdf_pred = self.forward_sdf(points_rand)
            loss = F.mse_loss(sdf_pred, sdf_gt)
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        # explicit broadcast to ensure param consistency across ranks
        for param in self.parameters():
            broadcast(param, src=0)
        
        if self.cfg.use_sdf_loss:
            pred_mesh, _ = self.isosurface()
        else:
            pred_mesh = self.isosurface()

        '''normal shape initialization'''        
        optim_n = torch.optim.Adam(self.parameters(), lr=5e-5)  # 1e-4
        ctx = NVDiffRasterizerContext("cuda", self.device)
        if self.cfg.init_debug:
            path_save = "./init_debug"
            os.makedirs(path_save, exist_ok=True)
        for step in tqdm(
            range(501),
            desc=f"Initializing SDF to a(n) {self.cfg.shape_init}:",
            disable=get_rank() != 0,
        ):
            camera_outputs = gen_camera_params(
                self.cfg.elevation_range,
                self.cfg.azimuth_range,
                self.cfg.camera_distance_range,
                self.cfg.fovy_range,
                self.cfg.height, 
                self.cfg.width)
            h = camera_outputs["height"]
            w = camera_outputs["width"]
            mvp_mtx = camera_outputs["mvp_mtx"].to(self.device)
            
            if self.cfg.use_sdf_loss:
                pred_mesh, _ = self.isosurface()
            else:
                pred_mesh = self.isosurface()
            
            v_pos_clip = ctx.vertex_transform(pred_mesh.v_pos, mvp_mtx)
            rast, _ = ctx.rasterize(v_pos_clip, pred_mesh.t_pos_idx, (h, w))
            mask = rast[..., 3:] > 0
            mask_aa = ctx.antialias(mask.float(), rast, v_pos_clip, pred_mesh.t_pos_idx)
            gb_normal, _ = ctx.interpolate_one(pred_mesh.v_nrm, rast, pred_mesh.t_pos_idx)
            gb_normal = F.normalize(gb_normal, dim=-1)
            gb_normal = torch.cat([gb_normal[:,:,:,1:2], gb_normal[:,:,:,2:3], gb_normal[:,:,:,0:1]], -1)
            gb_normal_aa = torch.lerp(torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float())
            gb_normal_aa = ctx.antialias(gb_normal_aa, rast, v_pos_clip, pred_mesh.t_pos_idx) # [0, 1]
            
            # gt normal
            gt_vertices = mesh.vertices.astype(np.float32)
            gt_faces = mesh.faces.astype(np.int32)
            if isinstance(gt_vertices, np.ndarray):
                gt_vertices = torch.from_numpy(gt_vertices).contiguous().float().to(self.device)
            if isinstance(gt_faces, np.ndarray):
                gt_faces = torch.from_numpy(gt_faces).contiguous().int().to(self.device)
            gt_pos_clip = ctx.vertex_transform(gt_vertices, mvp_mtx)
            gt_rast, _ = ctx.rasterize(gt_pos_clip, gt_faces, (h, w))
            gt_mask = gt_rast[..., 3:] > 0
            gt_mask_aa = ctx.antialias(gt_mask.float(), gt_rast, gt_pos_clip, gt_faces)
            gt_vn, _ = compute_normal(gt_vertices, gt_faces)
            gt_normal, _ = ctx.interpolate_one(gt_vn, gt_rast, gt_faces)
            gt_normal = F.normalize(gt_normal, dim=-1)
            gt_normal = torch.cat([gt_normal[:,:,:,1:2], gt_normal[:,:,:,2:3], gt_normal[:,:,:,0:1]], -1)
            gt_normal_aa = torch.lerp(torch.zeros_like(gt_normal), (gt_normal + 1.0) / 2.0, gt_mask.float())
            gt_normal_aa = ctx.antialias(gt_normal_aa, gt_rast, gt_pos_clip, gt_faces)
            
            criterion = torch.nn.L1Loss()
            loss = criterion(gb_normal_aa, gt_normal_aa)
            
            points_rand = (torch.rand((40000, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0)
            sdf_gt = get_gt_sdf(points_rand)
            sdf_pred = self.forward_sdf(points_rand)
            loss_sdf = F.mse_loss(sdf_pred, sdf_gt)
            loss = loss + 10000 * loss_sdf
            
            optim_n.zero_grad()
            loss.backward()
            optim_n.step()
            
            if step % self.cfg.init_per_val == 0 and self.cfg.init_debug:
                gb_normal_aa = (gb_normal_aa[0] * 255.0).detach().cpu().numpy().astype(np.uint8)
                gb_normal_aa = cv2.cvtColor(gb_normal_aa, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(path_save, f"{step}-pred-normal.jpg"), gb_normal_aa)

        # explicit broadcast to ensure param consistency across ranks
        for param in self.parameters():
            broadcast(param, src=0)
        
        if self.cfg.init_debug:
            # add one more isosurface after the whole initialization
            if self.cfg.use_sdf_loss:
                pred_mesh, _ = self.isosurface()
            else:
                pred_mesh = self.isosurface()
                
            mesh_export = trimesh.Trimesh(
                vertices=pred_mesh.v_pos.detach().cpu().numpy(),
                faces=pred_mesh.t_pos_idx.detach().cpu().numpy(),
                process=False)
            mesh_export = self.convert_mesh(mesh_export)
            mesh_export.export(os.path.join(path_save, "output.ply"))
            
            coarse_mesh, _ = self._isosurface(self.bbox, fine_stage=False)
            mesh_coarse_export = trimesh.Trimesh(
                vertices=coarse_mesh.v_pos.detach().cpu().numpy(),
                faces=coarse_mesh.t_pos_idx.detach().cpu().numpy(),
                process=False)
            mesh_coarse_export = self.convert_mesh(mesh_coarse_export)
            mesh_coarse_export.export(os.path.join(path_save, "output_coarse.ply"))

    def get_shifted_sdf(
        self, points: Float[Tensor, "*N Di"], sdf: Float[Tensor, "*N 1"]
    ) -> Float[Tensor, "*N 1"]:
        sdf_bias: Union[float, Float[Tensor, "*N 1"]]
        if isinstance(self.cfg.sdf_bias, float):
            sdf_bias = self.cfg.sdf_bias
        else:
            raise ValueError(f"Unknown sdf bias {self.cfg.sdf_bias}")
        return sdf + sdf_bias

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        grad_enabled = torch.is_grad_enabled()

        if output_normal and self.cfg.normal_type == "analytic":
            torch.set_grad_enabled(True)
            points.requires_grad_(True)

        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(
            points, self.bbox, self.unbounded
        )  # points normalized to (0, 1)

        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        sdf = self.sdf_network(enc).view(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        output = {"sdf": sdf}

        if self.cfg.n_feature_dims > 0:
            features = self.feature_network(enc).view(
                *points.shape[:-1], self.cfg.n_feature_dims
            )
            output.update({"features": features})

        if output_normal:
            if (self.cfg.normal_type == "finite_difference"
                or self.cfg.normal_type == "finite_difference_laplacian"
            ):
                assert self.finite_difference_normal_eps is not None
                eps: float = self.finite_difference_normal_eps
                if self.cfg.normal_type == "finite_difference_laplacian":
                    offsets: Float[Tensor, "6 3"] = torch.as_tensor(
                        [[eps, 0.0, 0.0],
                         [-eps, 0.0, 0.0],
                         [0.0, eps, 0.0],
                         [0.0, -eps, 0.0],
                         [0.0, 0.0, eps],
                         [0.0, 0.0, -eps],]).to(points_unscaled)
                    points_offset: Float[Tensor, "... 6 3"] = (points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    sdf_offset: Float[Tensor, "... 6 1"] = self.forward_sdf(points_offset)
                    sdf_grad = (0.5 * (sdf_offset[..., 0::2, 0] - sdf_offset[..., 1::2, 0]) / eps)
                else:
                    offsets: Float[Tensor, "3 3"] = torch.as_tensor(
                        [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 3 3"] = (points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    sdf_offset: Float[Tensor, "... 3 1"] = self.forward_sdf(points_offset)
                    sdf_grad = (sdf_offset[..., 0::1, 0] - sdf) / eps
                normal = F.normalize(sdf_grad, dim=-1)
            elif self.cfg.normal_type == "pred":
                normal = self.normal_network(enc).view(*points.shape[:-1], 3)
                normal = F.normalize(normal, dim=-1)
                sdf_grad = normal
            elif self.cfg.normal_type == "analytic":
                sdf_grad = -torch.autograd.grad(
                    sdf,
                    points_unscaled,
                    grad_outputs=torch.ones_like(sdf),
                    create_graph=True)[0]
                normal = F.normalize(sdf_grad, dim=-1)
                if not grad_enabled:
                    sdf_grad = sdf_grad.detach()
                    normal = normal.detach()
            else:
                raise AttributeError(f"Unknown normal type {self.cfg.normal_type}")
            output.update(
                {"normal": normal, "shading_normal": normal, "sdf_grad": sdf_grad}
            )
        return output

    def forward_sdf(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        sdf = self.sdf_network(
            self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        ).reshape(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        return sdf

    def forward_field(
        self, points: Float[Tensor, "*N Di"], sdf_type="full", frozen_sdf=None
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        sdf = self.sdf_network(enc).reshape(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        deformation: Optional[Float[Tensor, "*N 3"]] = None
        if self.cfg.isosurface_deformable_grid:
            deformation = self.deformation_network(enc).reshape(*points.shape[:-1], 3)

        sdf_loss: Optional[Float[Tensor, "*N 1"]] = None
        if sdf_type == "full": 
            if self.cfg.use_sdf_loss and self.start_control_sdf:
                selected_points_idx = torch.LongTensor(random.sample(range(points_unscaled.shape[0]), 30000))
                if frozen_sdf is not None:
                    gt_sdf = frozen_sdf[selected_points_idx].clone().detach()
                    sdf_loss = F.mse_loss(gt_sdf, sdf[selected_points_idx], reduction='sum')
                else:
                    sdf_loss = 0.
        elif sdf_type == "recon":
            if self.start_sdf_recon:
                selected_points_idx = torch.LongTensor(random.sample(range(points_unscaled.shape[0]), min(100000, points_unscaled.shape[0]))).to(self.device)
                if frozen_sdf is not None:
                    gt_sdf = frozen_sdf[selected_points_idx].clone().detach()
                    sdf_loss = F.mse_loss(gt_sdf, sdf[selected_points_idx], reduction='sum')
                else:
                    sdf_loss = 0.
        elif sdf_type == "none":
            pass
        
        return sdf, deformation, sdf_loss

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        return field - threshold

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(*points.shape[:-1], self.cfg.n_feature_dims)
        out.update({"features": features,})
        return out
    
    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # cache frozen sdf for surface-aware loss
        if global_step > 0 and self.start_sdf_recon is False:     
            self.start_sdf_recon = True
        # control sdf for edit part following HUMANNORM
        if global_step >= (self.cfg.start_sdf_loss_step + 1) and self.start_control_sdf is False and self.cfg.use_sdf_loss:
            self.start_control_sdf = True

        if (self.cfg.normal_type == "finite_difference"
            or self.cfg.normal_type == "finite_difference_laplacian"
        ):
            if isinstance(self.cfg.finite_difference_normal_eps, float):
                self.finite_difference_normal_eps = (self.cfg.finite_difference_normal_eps)
            elif self.cfg.finite_difference_normal_eps == "progressive":
                # progressive finite difference eps from Neuralangelo
                # https://arxiv.org/abs/2306.03092
                hg_conf: Any = self.cfg.pos_encoding_config
                assert (
                    hg_conf.otype == "ProgressiveBandHashGrid"
                ), "finite_difference_normal_eps=progressive only works with ProgressiveBandHashGrid"
                current_level = min(
                    hg_conf.start_level
                    + max(global_step - hg_conf.start_step, 0) // hg_conf.update_steps,
                    hg_conf.n_levels,)
                grid_res = hg_conf.base_resolution * hg_conf.per_level_scale ** (current_level - 1)
                grid_size = 2 * self.cfg.radius / grid_res
                if grid_size != self.finite_difference_normal_eps:
                    tetgs_spatial.info(f"Update finite_difference_normal_eps to {grid_size}")
                self.finite_difference_normal_eps = grid_size
            else:
                raise ValueError(f"Unknown finite_difference_normal_eps={self.cfg.finite_difference_normal_eps}")