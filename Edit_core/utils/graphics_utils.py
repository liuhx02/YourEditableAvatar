#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple
from pytorch3d.transforms import matrix_to_quaternion

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

# def getWorld2View(R, t):
#     Rt = np.zeros((4, 4))
#     Rt[:3, :3] = R.transpose()
#     Rt[:3, 3] = t
#     Rt[3, 3] = 1.0
#     return np.float32(Rt)

def getWorld2View(R, t, tensor=False):
    if tensor:
        Rt = torch.zeros(4, 4, device=R.device)
        Rt[..., :3, :3] = R.transpose(-1, -2)
        Rt[..., :3, 3] = t
        Rt[..., 3, 3] = 1.0
        return Rt
    else:
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0
        return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

scale_activation = torch.exp
scale_inverse_activation = torch.log
use_old_method = False

def radius_fn(radiuses:torch.Tensor, max_value=0.2):
    scales = scale_activation(radiuses[..., 4:])
    return (scales.abs().clamp(max=max_value).max(dim=-1, keepdim=True)[0])
    
def inverse_radius_fn(radiuses:torch.Tensor):
    scales = scale_inverse_activation(radiuses.expand(-1, -1, 3).clone())
    quaternions = matrix_to_quaternion(
        torch.eye(3)[None, None].repeat(1, radiuses.shape[1], 1, 1).to(radiuses.device)
        )
    return torch.cat([quaternions, scales], dim=-1)

def circumcircle_radius(A, B, C):
    a = torch.norm(B - C, dim=1)
    b = torch.norm(A - C, dim=1)
    c = torch.norm(A - B, dim=1)
    s = (a + b + c) / 2
    K = torch.sqrt(s * (s - a) * (s - b) * (s - c))
    R = (a * b * c) / (4 * K)
    return R

def triangle_area(A, B, C):
    AB = B - A
    AC = C - A
    cross_product = torch.cross(AB, AC)
    area = 0.5 * torch.norm(cross_product, dim=1, keepdim=True)  # [N, 1]
    return area

def calculate_distances(points, A, B, C):
    A = A.reshape(-1, 3)
    B = B.reshape(-1, 3)
    C = C.reshape(-1, 3)
    dA = torch.linalg.norm(points - A, axis=1)  
    dB = torch.linalg.norm(points - B, axis=1)  
    dC = torch.linalg.norm(points - C, axis=1)  
    
    distances = torch.minimum(torch.minimum(dA, dB), dC)
    print(distances.shape)
    distances = distances.reshape(-1, 1)
    
    return distances