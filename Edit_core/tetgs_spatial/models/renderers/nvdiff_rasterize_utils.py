import torch
import torch.nn.functional as F
from tetgs_spatial.utils.typing import *


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, -1, keepdim=True)

def compute_normal(vertices, faces):
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.as_tensor(vertices).float()
    if not isinstance(faces, torch.Tensor):
        faces = torch.as_tensor(faces).long()

    i0, i1, i2 = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()

    v0, v1, v2 = vertices[i0, :], vertices[i1, :], vertices[i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    vn = torch.zeros_like(vertices)
    vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    vn = torch.where(dot(vn, vn) > 1e-20, vn, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device))
    vn = safe_normalize(vn)

    face_normals = safe_normalize(face_normals)
    return vn, faces

def compute_vertex_normal(vertices, faces):
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.as_tensor(vertices).float()
    if not isinstance(faces, torch.Tensor):
        faces = torch.as_tensor(faces).long()
        
    i0 = faces[:, 0].to(torch.int64)
    i1 = faces[:, 1].to(torch.int64)
    i2 = faces[:, 2].to(torch.int64)

    v0 = vertices[i0, :]
    v1 = vertices[i1, :]
    v2 = vertices[i2, :]
    
    face_normals = torch.cross(v1 - v0, v2 - v0)
    
    # Splat face normals to vertices
    v_nrm = torch.zeros_like(vertices)
    v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)
    
    # Normalize, replace zero (degenerated) normals with some default value
    v_nrm = torch.where(
            dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm)
        )
    v_nrm = F.normalize(v_nrm, dim=1)
    
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_nrm))

    return v_nrm