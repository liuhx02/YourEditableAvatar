import os
import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import nvdiffrast.torch as dr
import torch.nn.functional as F
from mesh_localization import generate_back_projection_rays, refine_region
from tetgs_inpainter.cameras.cameras import Cameras

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

class TexturedMeshModel(nn.Module):
    def __init__(
        self, 
        render_angle_thres=75,    # 68
        render_angle_thres_later=68,   # 78
        verts=None, faces=None, 
        device=torch.device('cpu')
    ):
        super().__init__()
        self.device = device
        self.render_angle_thres = torch.tensor(render_angle_thres).to(self.device)
        self.render_angle_thres_later = torch.tensor(render_angle_thres_later).to(self.device)
        self.verts = torch.tensor(verts).to(self.device)
        self.faces = torch.tensor(faces).to(self.device)
        
        self.glctx = dr.RasterizeCudaContext(device=self.device)

    def forward(self):
        raise NotImplementedError
    
    def mask_mesh(
        self,
        verts_mask,  # [N, 1]
        idx,
        sample_cameras: Cameras,
        validate_mask_folder,
        use_normal_mask=True,
    ):
        verts = self.verts
        faces = self.faces
        
        if verts_mask is None:
            verts_mask = torch.ones_like(verts[:, :1]).to(self.device)
            verts_mask = verts_mask.contiguous().float()
        
        # nvdiffrast mask
        mvps = sample_cameras.mvps
        mvp = mvps[idx]
        c2ws = sample_cameras.camera_to_worlds
        c2w = c2ws[idx].unsqueeze(0)
        h0 = sample_cameras[idx].height[0].item()
        w0 = sample_cameras[idx].width[0].item()
        
        if isinstance(verts, np.ndarray):
            verts = torch.from_numpy(verts).contiguous().float().to(self.device)
        else:
            verts = verts.contiguous().float().to(self.device)
        if isinstance(faces, np.ndarray):
            faces = torch.from_numpy(faces).contiguous().to(torch.int32).to(self.device)
        else:
            faces = faces.contiguous().to(torch.int32).to(self.device)
        
        vertices_clip = torch.matmul(F.pad(verts, pad=(0, 1), mode='constant', value=1.0), torch.transpose(mvp, 0, 1)).float().unsqueeze(0)
        rast, _ = dr.rasterize(self.glctx, vertices_clip, faces, (h0, w0))
        
        xyzs, _ = dr.interpolate(verts.unsqueeze(0), rast, faces)
        mask, _ = dr.interpolate(verts_mask.unsqueeze(0), rast, faces)
        mask_aa = dr.antialias(mask.float(), rast, vertices_clip, faces)
        
        mask_aa_save = mask_aa[0]
        mask_aa_save[mask_aa_save > 0.0] = 1.0
        save_mask_aa_path = os.path.join(validate_mask_folder, f'{idx}_aa.png')
        cv2.imwrite(save_mask_aa_path, ((mask_aa_save.cpu().numpy() * 255).astype(np.uint8)).squeeze())
        
        # compute normal and filter mask
        verts_normal, _ = compute_normal(verts, faces)
        normal, _ = dr.interpolate(verts_normal, rast, faces)
        normal = normal.reshape(-1, h0*w0, 3)
        normal = torch.matmul(torch.linalg.inv(c2w[:,:3,:3]), normal[0][:,:,None])
        normal = normal[:, 2:3, :].reshape(1, h0, w0, 1)
        if idx < 8:
            render_angle_thres = self.render_angle_thres
        else:
            render_angle_thres = self.render_angle_thres_later
        cos_thres = torch.cos(render_angle_thres / 180 * torch.pi)
        normal[normal < cos_thres] = 0
        normal_mask = normal.clone()
        normal_mask[normal_mask > 0] = 1
        normal_mask = normal_mask[0]
        if use_normal_mask:
            mask_aa_and_normal = torch.logical_and(mask_aa_save, normal_mask)
        else:
            mask_aa_and_normal = mask_aa_save
        
        # render normal for inpainting
        gb_normal, _ = dr.interpolate(verts_normal, rast, faces)
        gb_normal = gb_normal.view(-1, h0*w0, 3)
        gb_normal = torch.matmul(torch.linalg.inv(c2w[:,:3,:3]), gb_normal[0][:,:,None])
        gb_normal = torch.where(gb_normal[:, 2:3] < 0, -gb_normal, gb_normal)
        gb_normal = gb_normal.view(-1, h0, w0, 3)
        gb_normal = F.normalize(gb_normal, dim=-1)
        bg_normal = torch.ones_like(gb_normal)
        full_verts_mask = torch.ones_like(verts[:, :1]).to(self.device)
        mask_full, _ = dr.interpolate(full_verts_mask.unsqueeze(0), rast, faces)
        # [0, 1]
        gb_normal_aa = torch.lerp(
            bg_normal, (gb_normal + 1.0) / 2.0, mask_full.float()
        )
        gb_normal_aa = dr.antialias(
            gb_normal_aa, rast, vertices_clip, faces
        )[0]
        
        outputs = {
            "mask_aa_save": ((mask_aa_save.cpu().numpy() * 255).astype(np.uint8)).squeeze(),
            "save_mask_aa_path": save_mask_aa_path,
            "normal_mask": normal_mask,
            "gb_normal_aa": ((gb_normal_aa.cpu().numpy() * 255).astype(np.uint8)).squeeze(),
            "mask_aa_and_normal": mask_aa_and_normal,
            "verts_mask": verts_mask,
        }
        return outputs
    
    def prepare_mask_proj(
        self,
        idx,
        sample_cameras: Cameras,
        save_mask_aa_path,
        normal_mask,
        keep_vertices_num,
        use_normal_mask=True,
    ):
        mask_aa = cv2.imread(save_mask_aa_path) 
        # 1. eorde
        mask_aa = cv2.erode(mask_aa, np.ones((2, 2), np.uint8))
        # 2. dilate and blur
        blur_kernel = 20   # 15
        mask_blur = cv2.GaussianBlur(cv2.dilate(mask_aa, np.ones((blur_kernel, blur_kernel), np.uint8)).astype(np.float32), (21, 21), 0)
        # 3. binarize mask_blur
        mask_blur_binary = mask_blur
        mask_blur_binary[mask_blur_binary > 0] = 1 
        mask_blur_binary = torch.tensor(mask_blur_binary[:, :, :1]).to(self.device)
        # 4. normal_mask
        if use_normal_mask:
            mask_blur_binary = torch.logical_and(mask_blur_binary, normal_mask)
        # 5. edit_mask
        mvps = sample_cameras.mvps
        mvp = mvps[idx]
        h0 = sample_cameras[idx].height[0].item()
        w0 = sample_cameras[idx].width[0].item()
        verts = self.verts.contiguous().float().to(self.device)
        faces = self.faces.contiguous().to(torch.int32).to(self.device)
        vertices_clip = torch.matmul(F.pad(verts, pad=(0, 1), mode='constant', value=1.0), torch.transpose(mvp, 0, 1)).float().unsqueeze(0)
        rast, _ = dr.rasterize(self.glctx, vertices_clip, faces, (h0, w0))
        edit_verts_mask = torch.zeros_like(verts[:, :1]).to(self.device)
        edit_verts_mask = edit_verts_mask.contiguous().float()
        edit_verts_mask[keep_vertices_num:, :] = 1
        edit_mask, _ = dr.interpolate(edit_verts_mask.unsqueeze(0), rast, faces)
        edit_mask = dr.antialias(edit_mask.float(), rast, vertices_clip, faces)
        edit_mask = edit_mask[0]
        edit_mask[edit_mask > 0] = 1
        if idx < 8:
            mask_proj = torch.logical_and(mask_blur_binary, edit_mask)
        else:
            mask_proj_aa = cv2.dilate(mask_aa, np.ones((5, 5), np.uint8)).astype(np.float32)
            mask_proj_aa[mask_proj_aa > 0] = 1
            mask_proj_aa = torch.tensor(mask_proj_aa[:, :, :1]).to(self.device)
            if use_normal_mask:
                mask_proj_aa = torch.logical_and(mask_proj_aa, normal_mask)
            mask_proj = torch.logical_and(mask_proj_aa, edit_mask)
        
        outputs = {
            "mask_blur": mask_blur,
            "mask_proj": mask_proj,
            "mask_blur_binary": mask_blur_binary,
            "edit_mask": edit_mask,
        }
        return outputs
    
    def back_project(
        self,
        idx,
        sample_cameras: Cameras,
        mask_proj,
        verts_mask,
    ):
        verts = self.verts.contiguous().float().to(self.device)
        faces = self.faces.contiguous().to(torch.int32).to(self.device)
        
        # back project mask
        mask_cc = mask_proj.reshape(-1).cpu().numpy()
        # edit mask_idx
        mask_idx = np.where(mask_cc == 1.0)[0]
        rays = generate_back_projection_rays(sample_cameras, idx, mask_idx)
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts.cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)
        
        hit_tri_id_Set = set()
        ans = scene.cast_rays(rays)
        hit_ids = ans['primitive_ids'].numpy()
        valid_hit_ids = hit_ids[hit_ids < 4294967295]
        for hit_id in valid_hit_ids:
            hit_tri_id_Set.add(hit_id)
             
        # update verts_mask
        if idx < 8:
            masked_verts, masked_faces = refine_region(mesh, hit_tri_id_Set, dilate_iter=2, erode_iter=5)
        else:
            masked_verts, masked_faces = refine_region(mesh, hit_tri_id_Set)

        mask_mesh = o3d.geometry.TriangleMesh()
        mask_mesh.vertices = o3d.utility.Vector3dVector(masked_verts)
        mask_mesh.triangles = o3d.utility.Vector3iVector(masked_faces)
        edit_vertices = np.asarray(mask_mesh.vertices)
        masked_vertices_tensor = torch.from_numpy(edit_vertices)  # [M, 3]
        all_vertices_tensor = torch.from_numpy(np.asarray(mesh.vertices))  # [N, 3]
        masked_vertices_set = set(map(tuple, masked_vertices_tensor.numpy()))
        mask = np.array([tuple(vertex) in masked_vertices_set for vertex in all_vertices_tensor.numpy()])
        mask = torch.from_numpy(mask).int()
        # 1: verts to be updated, 0: keep verts
        mask_dilate = (1 - mask).reshape(-1, 1).contiguous().float()
   
        mask_new = torch.zeros_like(verts_mask)
        # 1: uncolored verts, 0: newly painted verts
        mask = (1 - mask)
        mask_new = mask_new.view(-1)
        mask_new[:mask.shape[0]] = mask 
        verts_mask = torch.logical_and(verts_mask.view(-1), mask_new.to(self.device))
        verts_mask = verts_mask.reshape(-1, 1).contiguous().float()
        
        outputs = {
            "verts_mask": verts_mask,
            "mesh": mesh,
            "mask_dilate": mask_dilate,
        }
        return outputs
    
    def prepare_mask_normal_for_inpainting(
        self,
        verts_mask,
        cam_idx,
        sample_cameras: Cameras,
        validate_mask_folder,
        keep_vertices_num,
    ):
        mask_mesh_outputs = self.mask_mesh(
            verts_mask=verts_mask,
            idx=cam_idx,
            sample_cameras=sample_cameras,
            validate_mask_folder=validate_mask_folder
        )
        save_mask_aa_path = mask_mesh_outputs["save_mask_aa_path"]
        mask_aa_save = mask_mesh_outputs["mask_aa_save"]
        normal_mask = mask_mesh_outputs["normal_mask"]
        gb_normal_aa = mask_mesh_outputs["gb_normal_aa"]
        mask_aa_and_normal = mask_mesh_outputs["mask_aa_and_normal"]
        verts_mask = mask_mesh_outputs["verts_mask"]
        
        if cam_idx < 8:
            mask_proj_outputs = self.prepare_mask_proj(
                idx=cam_idx,
                sample_cameras=sample_cameras,
                save_mask_aa_path=save_mask_aa_path,
                normal_mask=normal_mask,
                keep_vertices_num=keep_vertices_num,
                use_normal_mask=True
            )
        else:
            mask_proj_outputs = self.prepare_mask_proj(
                idx=cam_idx,
                sample_cameras=sample_cameras,
                save_mask_aa_path=save_mask_aa_path,
                normal_mask=normal_mask,
                keep_vertices_num=keep_vertices_num,
                use_normal_mask=True
            )
        mask_blur = mask_proj_outputs["mask_blur"]
        mask_proj = mask_proj_outputs["mask_proj"]
        
        back_project_outputs = self.back_project(
            idx=cam_idx,
            sample_cameras=sample_cameras,
            mask_proj=mask_proj,
            verts_mask=verts_mask,
        )
        verts_mask = back_project_outputs["verts_mask"]
        mask_dilate = back_project_outputs["mask_dilate"]
        mesh = back_project_outputs["mesh"]
        
        outputs = {
            "verts_mask": verts_mask,
            "mesh": mesh,
            "mask_dilate": mask_dilate,
            "mask_blur": mask_blur,
            "mask_proj": mask_proj,
            "gb_normal": gb_normal_aa,
            "save_mask_aa_path": save_mask_aa_path,
            "mask_aa_save": mask_aa_save,
            "mask_aa_and_normal": mask_aa_and_normal,
        }
        return outputs
    
    def get_concat_mask(
        self,
        idx,
        sample_cameras: Cameras,
        verts_mask
    ):
        mvps = sample_cameras.mvps
        mvp = mvps[idx]
        h0 = sample_cameras[idx].height[0].item()
        w0 = sample_cameras[idx].width[0].item()
        verts = self.verts.contiguous().float().to(self.device)
        faces = self.faces.contiguous().to(torch.int32).to(self.device)
        vertices_clip = torch.matmul(F.pad(verts, pad=(0, 1), mode='constant', value=1.0), torch.transpose(mvp, 0, 1)).float().unsqueeze(0)
        rast, _ = dr.rasterize(self.glctx, vertices_clip, faces, (h0, w0))
        # edit_verts_mask
        edit_verts_mask = torch.zeros_like(verts[:, :1]).to(self.device)
        edit_verts_mask = edit_verts_mask.contiguous().float()
        edit_verts_mask = verts_mask
        edit_mask, _ = dr.interpolate(edit_verts_mask.unsqueeze(0), rast, faces)
        edit_mask = dr.antialias(edit_mask.float(), rast, vertices_clip, faces)
        edit_mask = edit_mask[0]
        edit_mask[edit_mask > 0] = 1  # [H, W, 1]
        # keep_verts_mask
        keep_verts_mask = torch.zeros_like(verts[:, :1]).to(self.device)
        keep_verts_mask = keep_verts_mask.contiguous().float()
        keep_verts_mask = 1 - verts_mask
        keep_mask, _ = dr.interpolate(keep_verts_mask.unsqueeze(0), rast, faces)
        keep_mask = dr.antialias(keep_mask.float(), rast, vertices_clip, faces)
        keep_mask = keep_mask[0]
        keep_mask[keep_mask > 0] = 1  # [H, W, 1]
        
        # blur keep_mask
        keep_mask = keep_mask.repeat(1, 1, 3)
        keep_mask = keep_mask.cpu().numpy()
        keep_mask_blur = cv2.GaussianBlur(keep_mask.astype(np.float32), (3, 3), 0)
        keep_mask_blur = torch.tensor(keep_mask_blur[:, :, :1]).to(self.device)
        edit_mask_rest = 1 - keep_mask_blur
        
        # # blur edit_mask
        # edit_mask = edit_mask.repeat(1, 1, 3)
        # edit_mask = edit_mask.cpu().numpy()
        # edit_mask_blur = cv2.dilate(edit_mask.astype(np.float32), np.ones((10, 10), np.uint8))
        # edit_mask_blur = cv2.GaussianBlur(edit_mask.astype(np.float32), (3, 3), 0)
        # edit_mask_blur = torch.tensor(edit_mask_blur[:, :, :1]).to(self.device)
        # keep_mask_rest = 1 - edit_mask_blur
        
        # outputs = {
        #     "edit_mask": edit_mask_blur,
        #     "keep_mask": keep_mask_rest
        # }
        outputs = {
            "edit_mask": edit_mask_rest,
            "keep_mask": keep_mask_blur
        }
        return outputs