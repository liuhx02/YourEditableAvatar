import numpy as np
import torch
import torch.nn.functional as F
import math

from tetgs_spatial.utils.typing import *
from tetgs_spatial.data.uncond import convert_proj
from tetgs_spatial.utils.ops import get_mvp_matrix

def gen_camera_params(elevation_range, azimuth_range, camera_distance_range, fovy_range, height, width):
        
        elevation_deg = (
            torch.rand(1)
            * (elevation_range[1] - elevation_range[0])
            + elevation_range[0]
        )
        elevation = elevation_deg * math.pi / 180
        
        azimuth_deg = (
            torch.rand(1)
            * (azimuth_range[1] - azimuth_range[0])
            + azimuth_range[0]
        )
        azimuth = azimuth_deg * math.pi / 180
        
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(1)
            * (camera_distance_range[1] - camera_distance_range[0])
            + camera_distance_range[0]
        )
        
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )
        
        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(1, 1)

        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(1) * (fovy_range[1] - fovy_range[0])
            + fovy_range[0]
        )
        fovy = fovy_deg * math.pi / 180
        focal_length: Float[Tensor, "B"] = 0.5 * height / torch.tan(0.5 * fovy)
        
        human_part = np.random.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])
        if human_part == 0: # full
            cx = torch.full_like(focal_length, width / 2)
            cy = torch.full_like(focal_length, height / 2)
        elif human_part == 1: # head
            focal_scale = 2
            focal_scale = torch.full_like(focal_length, focal_scale)
            focal_length *= focal_scale
            cx = torch.full_like(focal_length, width / 2)
            cy = torch.full_like(focal_length, height / 2)
            center[:,2] += 0.6
        elif human_part == 2: # upper body
            focal_scale = 2.0
            focal_scale = torch.full_like(focal_length, focal_scale)
            focal_length *= focal_scale
            cx = torch.full_like(focal_length, width / 2)
            cy = torch.full_like(focal_length, height / 2)
            center[:,2] += 0.3
        elif human_part == 3: # lower body
            focal_scale = 2.0
            focal_scale = torch.full_like(focal_length, focal_scale)
            focal_length *= focal_scale
            cx = torch.full_like(focal_length, width / 2)
            cy = torch.full_like(focal_length, height / 2)
            center[:,2] -= 0.5
            
        intrinsic: Float[Tensor, "B 4 4"] = torch.eye(4)[None, :,:].repeat(1, 1, 1)
        intrinsic[:, 0, 0] = focal_length
        intrinsic[:, 1, 1] = focal_length
        intrinsic[:, 0, 2] = cx
        intrinsic[:, 1, 2] = cy

        proj_mtx = []
        for i in range(1):
            proj = convert_proj(intrinsic[i], height, width, 0.1, 1000.0)
            proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
            proj_mtx.append(proj)
        proj_mtx: Float[Tensor, "B 4 4"] = torch.stack(proj_mtx, dim=0)
        
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
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)
        
        outputs = {
            "height": height,
            "width": width,
            "mvp_mtx": mvp_mtx,
            "c2w": c2w,
        }
        
        return outputs