"""
Localization on mesh surface using sampled rendered TetGS images
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import os
from PIL import Image
import open3d as o3d
from rich.console import Console
import pymeshlab as pml
import torch
import trimesh
from tqdm import tqdm
from lang_sam import LangSAM
import sys
sys.path.append("../")
from tetgs_inpainter.camera_sample_views_uncond import UncondSampleViews
CONSOLE = Console(width=120)

def generate_back_projection_rays(sample_cameras, cam_idx, mask_idx):
    ray_bundle = sample_cameras.generate_rays(camera_indices=cam_idx)
    rays_d = ray_bundle.directions # [H, W, 3]
    rays_o = ray_bundle.origins # [H, W, 3]
    origins = rays_o.cpu().numpy().reshape(-1, 3)
    directions = rays_d.cpu().numpy().reshape(-1, 3)
    
    pixel_oris = origins[mask_idx].reshape(-1, 3)
    pixel_dirs = directions[mask_idx].reshape(-1, 3)
    rays_core = np.concatenate([pixel_oris, pixel_dirs], axis=1)
    rays = o3d.core.Tensor([rays_core], dtype=o3d.core.Dtype.Float32)
    return rays

def refine_region(mesh, hit_tri_id_all_set, dilate_iter=1, erode_iter=2):
    chosen_vec_set = set()
    mesh_tri_np = np.asarray(mesh.triangles)
    np_vec = np.asarray(mesh.vertices)
    
    if hit_tri_id_all_set:
        all_vec_ids = mesh_tri_np[list(hit_tri_id_all_set)].flatten()
        chosen_vec_set.update(set(all_vec_ids))
    chosen_vec_list = list(chosen_vec_set)
    chosen_vec_list = sorted(chosen_vec_list)
    
    tri_np_mask = np.ones(len(mesh_tri_np))
    # mask: 0 == keep, 1 == remove
    for hit_tri_id in list(hit_tri_id_all_set):
        tri_np_mask[hit_tri_id] = 0
    
    # dilate undelete mask 
    m = pml.Mesh(np_vec, mesh_tri_np, f_scalar_array=tri_np_mask)   
    ms = pml.MeshSet()   
    ms.add_mesh(m, 'mesh')
    ms.compute_selection_by_condition_per_face(condselect='fq == 0') # select kept faces
    # dilate to aviod holes...
    for _ in range(dilate_iter):
        ms.apply_selection_dilatation()
    for _ in range(erode_iter):
        ms.apply_selection_erosion()
    ms.apply_selection_inverse(invfaces=True) # invert
    ms.meshing_remove_selected_faces() # delete faces
    ms.meshing_remove_unreferenced_vertices() # clean unref verts
    m = ms.current_mesh()  # extract
    verts = m.vertex_matrix()
    faces = m.face_matrix()
    mesh = trimesh.Trimesh(verts, faces, process=False)
    return mesh.vertices, mesh.faces

@dataclass
class LocalMeshEditingModel(torch.nn.Module):
    def __init__(self, cam_pack, image_folder, comm_cfg, args):
        super().__init__()
        self.comm_cfg = comm_cfg
        self.radius_range = cam_pack["radius_range"]
        self.fovy_range = cam_pack["fovy_range"]
        self.theta_list = cam_pack["theta_list"]
        self.phi_list = cam_pack["phi_list"]
        self.H = cam_pack["H"]
        self.W = cam_pack["W"]
        self.image_folder = image_folder
        self.output_root = args.output_dir
        self.prompt = args.seg_prompt
        self.mesh_path = args.seg_mesh_path
        self.source_path = args.scene_path
        self.threshold = 0.5
        self.torch_precision = "high"
        self.model = LangSAM("vit_h")
    
    def back_projection(self, scene):
        hit_tri_id_Set = set()
        # mesh cameras
        Camera_Sample_Views = UncondSampleViews(
            device=self.device,
            radius_range=self.radius_range,
            fovy_range=self.fovy_range,
            phi_list=self.phi_list,
            theta_list=self.theta_list,
            shape_init_params=self.comm_cfg.shape_init_params,
            anchor_path=self.comm_cfg.anchor_path,
            mesh_path=os.path.join(self.source_path, self.comm_cfg.mesh_name),
            H=self.H, W=self.W
        )
        sample_cameras = Camera_Sample_Views.generate_sample_views()
        sample_cameras = sample_cameras.to(self.device)
        
        phi_1d_list = Camera_Sample_Views.phi_1d_list
        theta_1d_list = Camera_Sample_Views.theta_1d_list
        size = Camera_Sample_Views.size
        for idx in tqdm(range(size)):
            # load tetgs image using theta and phi
            image_filepath = os.path.join(self.image_folder, f"{phi_1d_list[idx]}_{theta_1d_list[idx]}.png")
            tetgs_image = Image.open(image_filepath).convert('RGB')
            # segment mask
            mask, _, _, _ = self.model.predict(tetgs_image, self.prompt)
            if mask.ndim == 3:
                mask = mask[0:1].cpu().numpy().astype(np.float32)
            else:
                print(f"None {self.prompt} Detected")
                mask = torch.zeros_like(tetgs_image[0:1]).cpu().numpy().astype(np.float32)
            mask[mask > self.threshold * 1.] = 1.
            mask = mask.reshape(-1)
            mask_idx = np.where(mask == 1.)[0]  # edit part

            rays = generate_back_projection_rays(sample_cameras, idx, mask_idx)
            ans = scene.cast_rays(rays)
            hit_ids = ans['primitive_ids'].numpy()
            valid_hit_ids = hit_ids[hit_ids < 4294967295]
            for hit_id in valid_hit_ids:
                hit_tri_id_Set.add(hit_id)
                
        return hit_tri_id_Set
    
    def gen_editing_region_mask(self, mesh, hit_tri_id_all_set):
        verts, faces = refine_region(mesh, hit_tri_id_all_set, dilate_iter=8, erode_iter=10)
        pymesh = pml.Mesh(verts, faces)
        num_faces = faces.shape[0]
        ms = pml.MeshSet()   
        ms.add_mesh(pymesh)
        # remove floaters
        ms.meshing_remove_duplicate_faces() # faces defined by the same verts
        ms.meshing_remove_null_faces() # faces with area == 0
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=int(num_faces * 0.25))
        pymesh = ms.current_mesh()
        mask_mesh = o3d.geometry.TriangleMesh()
        mask_mesh.vertices = o3d.utility.Vector3dVector(pymesh.vertex_matrix())
        mask_mesh.triangles = o3d.utility.Vector3iVector(pymesh.face_matrix())
        o3d.io.write_triangle_mesh(os.path.join(self.output_root, 'editing_region.ply'), mask_mesh)
        return mask_mesh
        
    def mesh_localization(self):
        torch.set_float32_matmul_precision(self.torch_precision)
        self.device = f"cuda:{0}"
        
        # back projection to mesh
        print(f"back projection to mesh")
        trimesh_mesh = trimesh.load(self.mesh_path, process=False)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)   
        mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)   
        np_tri = np.asarray(mesh.triangles)
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)
        hit_tri_id_Set = self.back_projection(scene=scene)
        
        # refine_region
        mask_mesh = self.gen_editing_region_mask(mesh, hit_tri_id_Set)
        
        # save as numpy information
        all_vertices = np.asarray(mesh.vertices)
        edit_vertices = np.asarray(mask_mesh.vertices)
        edit_vertices_tensor = torch.from_numpy(edit_vertices) # [M, 3]
        all_vertices_tensor = torch.from_numpy(all_vertices)  # [N, 3]
        # matches: [N, M, 3], [N, 1, 3] vs [1, M, 3]
        matches = (all_vertices_tensor.unsqueeze(1) == edit_vertices_tensor.unsqueeze(0))
        matches = matches.all(dim=2) # check x, y, z
        mask = matches.any(dim=1) # [N]
        mask = mask.int()
        assert mask.shape[0] == all_vertices_tensor.shape[0]
        assert torch.sum(mask) == edit_vertices_tensor.shape[0]
        
        mask_np = mask.detach().cpu().numpy()
        # 0: keep, 1: edit
        hit_tri_id_keep_set = np.where((mask_np[np_tri[:, 0]] == 0) &
                                       (mask_np[np_tri[:, 1]] == 0) &
                                       (mask_np[np_tri[:, 2]] == 0))[0]
        # vertices mask
        edit_mask_np = np.ones_like(mask_np)
        edit_mask_0_indices = np.array([[idx for idx in np_tri[tri_id]] for tri_id in hit_tri_id_keep_set])
        edit_mask_0_indices = edit_mask_0_indices.flatten()
        edit_mask_np[edit_mask_0_indices] = 0
        # faces mask
        edit_mask_faces = np.ones((np_tri.shape[0]))
        edit_mask_faces[hit_tri_id_keep_set] = 0
        # save npy
        mesh_info = {}
        mesh_info['editing_mask'] = edit_mask_np
        mesh_info['editing_mask_faces'] = edit_mask_faces
        np.save(os.path.join(self.output_root, 'editing_region_info.npy'), mesh_info)