import bisect
import math
import random
from dataclasses import dataclass, field
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import tetgs_spatial
from tetgs_spatial import register
from tetgs_spatial.utils.base import Updateable
from tetgs_spatial.utils.config import parse_structured
from tetgs_spatial.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from tetgs_spatial.utils.typing import *

def convert_proj(K, H, W, near, far):
    return [
        [2 * K[0, 0] / W, -2 * K[0, 1] / W, (W - 2 * K[0, 2]) / W, 0],
        [0, -2 * K[1, 1] / H, (H - 2 * K[1, 2]) / H, 0],
        [0, 0, (-far - near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0],
    ]

@dataclass
class RandomCameraDataModuleConfig:
    height: Any = 512
    width: Any = 512
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = 1024
    eval_width: int = 1024
    eval_batch_size: int = 1
    n_val_views: int = 60
    n_test_views: int = 60
    elevation_range: Tuple[float, float] = (-10, 90)
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (1, 1.5)
    fovy_range: Tuple[float, float] = (40, 70,)  # in degrees, in vertical direction (along height)
    batch_uniform_azimuth: bool = True
    progressive_until: int = 0
    random_test: bool = False
    sampling_type: str = "global_local"
    global_focal_scale: float = 1.4  # 1.4
    local_focal_scale: float = 2.2  # 1.8
    global_center_perturb: float = -0.05
    local_type: str = "full" # "full" or "upper" or "lower"
    local_center_perturb: List[float] = field(default_factory=lambda: [-0.05, 0.3, -0.35])
    test_save_path: str = "./.tetgs_spatial_cache"
    comm_cfg: str = "../comm_config.yaml"


class RandomCameraLocalGlobalIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        self.resolution_milestones: List[int]
        if (
            len(self.heights) == 1
            and len(self.widths) == 1
            and len(self.batch_sizes) == 1
        ):
            if len(self.cfg.resolution_milestones) > 0:
                tetgs_spatial.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.elevation_range = self.cfg.elevation_range
        self.azimuth_range = self.cfg.azimuth_range
        self.camera_distance_range = self.cfg.camera_distance_range
        self.fovy_range = self.cfg.fovy_range
        self.comm_cfg = OmegaConf.load(self.cfg.comm_cfg)
        
    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        self.width = self.widths[size_ind]
        self.batch_size = self.batch_sizes[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        tetgs_spatial.debug(
            f"Training height: {self.height}, width: {self.width}, batch_size: {self.batch_size}"
        )
        # progressive view
        self.progressive_view(global_step)
        
    def __iter__(self):
        while True:
            yield {}

    def progressive_view(self, global_step):
        r = min(1.0, global_step / (self.cfg.progressive_until + 1))
        self.elevation_range = [
            (1 - r) * self.comm_cfg.eval_elevation_deg + r * self.cfg.elevation_range[0],
            (1 - r) * self.comm_cfg.eval_elevation_deg + r * self.cfg.elevation_range[1],
        ]
        self.azimuth_range = [
            (1 - r) * 0.0 + r * self.cfg.azimuth_range[0],
            (1 - r) * 0.0 + r * self.cfg.azimuth_range[1],
        ]
        
    def collate(self, batch) -> Dict[str, Any]:
        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(self.batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            )
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(self.batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            )
            elevation_deg = elevation / math.pi * 180.0
        
        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        if self.cfg.batch_uniform_azimuth:
            # ensures sampled azimuth angles in a batch cover the whole range
            azimuth_deg = (
                torch.rand(self.batch_size) + torch.arange(self.batch_size)
            ) / self.batch_size * (
                self.azimuth_range[1] - self.azimuth_range[0]
            ) + self.azimuth_range[
                0
            ]
        else:
            # simple random sampling
            azimuth_deg = (
                torch.rand(self.batch_size)
                * (self.azimuth_range[1] - self.azimuth_range[0])
                + self.azimuth_range[0]
            )
        azimuth = azimuth_deg * math.pi / 180
        
        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(self.batch_size)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        )
        
        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )
        
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)
        
        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(self.batch_size) * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        )
        fovy = fovy_deg * math.pi / 180
            
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        if self.cfg.sampling_type == "global_local":
            # global: full body
            focal_scale_global = torch.full_like(focal_length, self.cfg.global_focal_scale) 
            focal_length_global = focal_length * focal_scale_global
            cx_global = torch.full_like(focal_length_global, self.cfg.width / 2)
            cy_global = torch.full_like(focal_length_global, self.cfg.height / 2)
            global_center = torch.zeros_like(camera_positions)
            global_center[:, 2] = self.cfg.global_center_perturb
            
            # local: only clothes
            focal_scale_local = torch.full_like(focal_length, self.cfg.local_focal_scale)
            focal_length_local = focal_length * focal_scale_local
            cx = torch.full_like(focal_length_local, self.cfg.width / 2)
            cy = torch.full_like(focal_length_local, self.cfg.height / 2)
            local_center = torch.zeros_like(camera_positions)
            if self.cfg.local_type == "full":
                local_center[:,2] = self.cfg.local_center_perturb[0]
            elif self.cfg.local_type == "upper":
                local_center[:,2] = self.cfg.local_center_perturb[1]
            elif self.cfg.local_type == "lower":
                local_center[:,2] = self.cfg.local_center_perturb[2]
        else:
            raise NotImplementedError
        
        # local
        intrinsic: Float[Tensor, "B 4 4"] = torch.eye(4)[None, :,:].repeat(self.batch_size, 1, 1)
        intrinsic[:, 0, 0] = focal_length_local
        intrinsic[:, 1, 1] = focal_length_local
        intrinsic[:, 0, 2] = cx
        intrinsic[:, 1, 2] = cy
        # global
        intrinsic_global: Float[Tensor, "B 4 4"] = torch.eye(4)[None, :,:].repeat(self.batch_size, 1, 1)
        intrinsic_global[:, 0, 0] = focal_length_global
        intrinsic_global[:, 1, 1] = focal_length_global
        intrinsic_global[:, 0, 2] = cx_global
        intrinsic_global[:, 1, 2] = cy_global
        
        proj_mtx = []
        directions = []
        proj_mtx_global = []
        for i in range(self.batch_size):
            direction: Float[Tensor, "H W 3"] = get_ray_directions(
                self.cfg.height,
                self.cfg.width,
                (intrinsic[i, 0, 0], intrinsic[i, 1, 1]),
                (intrinsic[i, 0, 2], intrinsic[i, 1, 2]),
                use_pixel_centers=False,
            )
            directions.append(direction)
            
            # local
            proj = convert_proj(intrinsic[i], self.cfg.height, self.cfg.width, 0.1, 1000.0)
            proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
            proj_mtx.append(proj)
            # global
            proj_global = convert_proj(intrinsic_global[i], self.cfg.height, self.cfg.width, 0.1, 1000.0)
            proj_global: Float[Tensor, "4 4"] = torch.FloatTensor(proj_global)
            proj_mtx_global.append(proj_global)
            
        proj_mtx: Float[Tensor, "B 4 4"] = torch.stack(proj_mtx, dim=0)
        directions: Float[Tensor, "B H W 3"] = torch.stack(directions, dim=0)
        proj_mtx_global: Float[Tensor, "B 4 4"] = torch.stack(proj_mtx_global, dim=0)
        
        def get_c2w_mvp(center, proj_mtx):
            cam_lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
            cam_right: Float[Tensor, "B 3"] = F.normalize(torch.cross(cam_lookat, up), dim=-1)
            cam_up = F.normalize(torch.cross(cam_right, cam_lookat), dim=-1)
            c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
                [torch.stack([cam_right, cam_up, -cam_lookat], dim=-1), camera_positions[:, :, None]],
                dim=-1
            )
            c2w: Float[Tensor, "B 4 4"] = torch.cat(
                [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
            )
            c2w[:, 3, 3] = 1.0
            mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)   
            return c2w, mvp_mtx
        
        # global paramters
        c2w_global, mvp_mtx_global = get_c2w_mvp(global_center, proj_mtx_global)
        # local paramters
        c2w, mvp_mtx = get_c2w_mvp(local_center, proj_mtx)
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        
        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
            "focal": focal_length_local,
            "human_part": torch.tensor(1, dtype=torch.long),
            "mvp_mtx_global": mvp_mtx_global,
            "c2w_global": c2w_global
        }


class RandomCameraLocalGlobalDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.split = split
        self.comm_cfg = OmegaConf.load("../comm_config.yaml")
        
        if split == "val":
            self.n_views = self.cfg.n_val_views
        else:
            self.n_views = self.cfg.n_test_views
            
        if self.cfg.random_test:
            if self.cfg.sampling_type == "global_local":
                n_views_part = self.n_views
            else:
                raise NotImplementedError
        
        azimuth_deg: Float[Tensor, "B"]
        if self.cfg.random_test:
            azimuth_deg = torch.linspace(0, 360.0, n_views_part)
            azimuth_deg = azimuth_deg.repeat(self.n_views//n_views_part)
        else:
            azimuth_deg = torch.linspace(0, 360.0, self.n_views)
        
        if self.cfg.random_test and self.split == "test":
            # randomly select ele and radius
            # only for test, not val
            elevation_deg: Float[Tensor, "B"] = (
                torch.rand(self.n_views)
                * (self.cfg.elevation_range[1] - self.cfg.elevation_range[0])
                + self.cfg.elevation_range[0]
            )

            camera_distances: Float[Tensor, "B"] = (
                torch.rand(self.n_views)
                * (self.cfg.camera_distance_range[1] - self.cfg.camera_distance_range[0])
                + self.cfg.camera_distance_range[0]
            )
        else:
            elevation_deg: Float[Tensor, "B"] = torch.full_like(
                azimuth_deg, self.comm_cfg.eval_elevation_deg
            )
            camera_distances: Float[Tensor, "B"] = torch.full_like(
                elevation_deg, self.comm_cfg.eval_camera_distance
            )
            
        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )
        
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.cfg.eval_batch_size, 1)
        fovy_deg: Float[Tensor, "B"] = torch.full_like(elevation_deg, self.comm_cfg.eval_fovy_deg)
        fovy = fovy_deg * math.pi / 180
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
        )
        
        if self.cfg.random_test:
            if self.cfg.sampling_type == "global_local":
                focal_scale = torch.full([n_views_part], self.cfg.global_focal_scale)
                focal_length *= focal_scale
                
                cx = torch.full_like(focal_length, self.cfg.eval_width / 2)
                cy = torch.full_like(focal_length, self.cfg.eval_height / 2)

                center[:, 2] = center[:, 2] + self.cfg.global_center_perturb
            else:
                raise NotImplementedError
            
            intrinsic: Float[Tensor, "B 4 4"] = torch.eye(4)[None, :,:].repeat(self.n_views, 1, 1)
            intrinsic[:, 0, 0] = focal_length
            intrinsic[:, 1, 1] = focal_length
            intrinsic[:, 0, 2] = cx
            intrinsic[:, 1, 2] = cy
            
            proj_mtx = []
            directions = []
            for i in range(self.n_views):
                proj = convert_proj(intrinsic[i], self.cfg.eval_height, self.cfg.eval_width, 0.1, 1000.0)
                proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
                proj_mtx.append(proj)
                
                direction: Float[Tensor, "H W 3"] = get_ray_directions(
                    self.cfg.eval_height,
                    self.cfg.eval_width,
                    (intrinsic[i, 0, 0], intrinsic[i, 1, 1]),
                    (intrinsic[i, 0, 2], intrinsic[i, 1, 2]),
                    use_pixel_centers=False,
                )
                directions.append(direction)
                
            proj_mtx: Float[Tensor, "B 4 4"] = torch.stack(proj_mtx, dim=0)
            directions: Float[Tensor, "B H W 3"] = torch.stack(directions, dim=0)
        
        # global sampling
        # only scale the focal length by 1.0
        else:
            cx: Float[Tensor, "B"] = torch.full_like(focal_length, self.cfg.eval_width / 2)
            cy: Float[Tensor, "B"] = torch.full_like(focal_length, self.cfg.eval_height / 2)
            directions_unit_focal = get_ray_directions(
                H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0
            )
            directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
                None, :, :, :
            ].repeat(self.n_views, 1, 1, 1)
            proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
                fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
            ) 
            directions[:, :, :, :2] = (
                directions[:, :, :, :2] / focal_length[:, None, None, None]
            )
        
        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
        self.c2w = c2w
        self.camera_positions = camera_positions
        self.elevation, self.azimuth = elevation, azimuth
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distances = camera_distances
        self.focal_length = focal_length
        self.cx = cx
        self.cy = cy
        self.mvp_mtx_global = mvp_mtx
        self.c2w_global = c2w
        
    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        return {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.c2w[index],
            "camera_positions": self.camera_positions[index],
            "elevation": self.elevation_deg[index],
            "azimuth": self.azimuth_deg[index],
            "camera_distances": self.camera_distances[index],
            "height": self.cfg.eval_height,
            "width": self.cfg.eval_width,
            "focal": self.focal_length[index],
            "cx": self.cx[index],
            "cy": self.cy[index],
            "n_views": self.n_views,
            "test_save_path": self.cfg.test_save_path,
            "mvp_mtx_global": self.mvp_mtx_global[index],
            "c2w_global": self.c2w_global[index],
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch


@register("random-camera-global-local-datamodule")
class RandomCameraLocalGlobalDataModule(pl.LightningDataModule):
    cfg: RandomCameraDataModuleConfig
    
    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomCameraDataModuleConfig, cfg)
    
    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = RandomCameraLocalGlobalIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraLocalGlobalDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraLocalGlobalDataset(self.cfg, "test")
            
    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        
    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )
        
    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )