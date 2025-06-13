import torch
import numpy as np
import json
from jaxtyping import Float
from torch import Tensor
from tetgs_scene.cameras import convert_mesh_init, gen_tet_camera
from tetgs_inpainter.cameras.cameras import Cameras

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def convert_proj(K, H, W, near, far):
    return [
        [2 * K[0, 0] / W, -2 * K[0, 1] / W, (W - 2 * K[0, 2]) / W, 0],
        [0, -2 * K[1, 1] / H, (H - 2 * K[1, 2]) / H, 0],
        [0, 0, (-far - near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0]]
    
def get_mvp_matrix(
    c2w: Float[Tensor, "B 4 4"], proj_mtx: Float[Tensor, "B 4 4"]
) -> Float[Tensor, "B 4 4"]:
    mvp_mtx = proj_mtx @ torch.inverse(c2w)
    return mvp_mtx

class UncondSampleViews:
    def __init__(
        self,
        device,
        radius_range=[3.0, 3.0], 
        fovy_range=[45, 45], 
        phi_list=[[0, 180, 90, 270, 45, 135, 225, 315],
                [0, 180, 90, 270, 30, 60, 120, 150, 210, 240, 300, 330],
                [10, 40, 70, 100, 130, 160, 190, 220, 250, 280, 310, 340]], 
        theta_list=[5, -15, 25], 
        shape_init_params=0.9,
        anchor_path=None,
        mesh_path=None, 
        H=512, W=512,
        R_path=None,
        metadata_path=None,
        sample_type="full",
    ):
        self.device = device
        self.R_path = R_path
        self.H = H
        self.W = W
        self.radius_range = radius_range
        self.fovy_range = fovy_range
        self.phi_list = phi_list
        self.theta_list = theta_list

        num_turns = len(phi_list)
        self.size = 0
        self.phi_1d_list = []
        self.theta_1d_list = []
        for idx in range(num_turns):
            self.size += len(phi_list[idx])
            num_angles = len(phi_list[idx])
            for j in range(num_angles):
                self.phi_1d_list.append(phi_list[idx][j])
                self.theta_1d_list.append(theta_list[idx])
        print("phi_list shape: ", len(self.phi_1d_list))
        print("size: ", self.size)
        
        self.cx = self.W / 2
        self.cy = self.H / 2
        self.shape_init_params = shape_init_params
        self.anchor_path = anchor_path
        self.mesh_path = mesh_path
        self.metadata_path = metadata_path
        self.sample_type = sample_type
        
    def gen_input_camera(
        self, 
        idx, 
        phi, 
        theta, 
        radius, 
        fovy_range,
    ):
        c2w, focal_length = gen_tet_camera(idx, radius, theta, phi, fovy_range, self.H, self.sample_type, self.device)
        
        cx = torch.tensor(self.W / 2).to(self.device)
        cy = torch.tensor(self.H / 2).to(self.device)
        intrinsic: Float[Tensor, "B 4 4"] = torch.eye(4)[None, :,:].repeat(1, 1, 1)
        intrinsic[:, 0, 0] = focal_length
        intrinsic[:, 1, 1] = focal_length
        intrinsic[:, 0, 2] = cx
        intrinsic[:, 1, 2] = cy
        
        proj_mtx = []
        proj = convert_proj(intrinsic[0], self.H, self.W, 0.1, 1000.0)
        proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
        proj_mtx.append(proj)
        proj_mtx: Float[Tensor, "B 4 4"] = torch.stack(proj_mtx, dim=0).to(self.device)
        
        # trans camera pose to sample around 3D Gaussians
        c2w = convert_mesh_init(c2w, mesh_path=self.mesh_path, 
                                    shape_init_params=self.shape_init_params, 
                                    anchor_path=self.anchor_path, 
                                    device=self.device)
        if self.R_path is not None:
            R = np.load(self.R_path)
            R = np.linalg.inv(R)
            c2w = np.dot(R, c2w[0].cpu().numpy())  
        if self.metadata_path is not None:
            metadata = json.load(open(self.metadata_path))
            worldtogt = np.array(metadata['worldtogt'])
            sdfstudio_to_colmap = np.array([
                [-0.,  1.,  0.,  0.],
                [ 1.,  0., -0., -0.],
                [-0., -0., -1.,  0.],
                [ 0.,  0.,  0.,  1.]])
            c2w[:4, 3] = np.dot(c2w[:4, 3], worldtogt.T)
            c2w[:4, 3] = np.dot(c2w[:4, 3], sdfstudio_to_colmap)
            c2w[:4, :3] = (np.dot(c2w[:4, :3].T, sdfstudio_to_colmap)).T
            c2w = torch.tensor(c2w.reshape(4, 4)).to(self.device).float()
            c2w = c2w.unsqueeze(0)         
        
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)
        input_camera = {
            "eval_height": self.H,
            "eval_width": self.W,
            "c2w": c2w,
            "mvp_mtx": mvp_mtx,
            "focal": focal_length,
            "cx": cx,
            "cy": cy,
        }
        return input_camera
    
    def generate_sample_views(self) -> Cameras:
        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []
        mvps = []
        
        fov = (self.fovy_range[0] + self.fovy_range[1]) / 2
        for idx in range(self.size):
            # phi = self.phi_list[idx % len(self.phi_list)]
            # theta = self.theta_list[int(idx / len(self.phi_list))]
            theta = self.theta_1d_list[idx]
            phi = self.phi_1d_list[idx]
            input_camera = self.gen_input_camera(
                idx=idx,
                phi=phi,
                theta=theta,
                radius=self.radius_range[0],
                fovy_range=fov,
            )
            fx.append(input_camera["focal"])
            fy.append(input_camera["focal"])
            cx.append(input_camera["cx"])
            cy.append(input_camera["cy"])
            camera_to_worlds.append(input_camera["c2w"])
            mvps.append(input_camera["mvp_mtx"])
        
        fx = torch.stack(fx).reshape(-1, 1).to(self.device) # [B, 1]
        fy = torch.stack(fy).reshape(-1, 1).to(self.device) # [B, 1]
        cx = torch.stack(cx).reshape(-1, 1).to(self.device) # [B, 1]
        cy = torch.stack(cy).reshape(-1, 1).to(self.device) # [B, 1]
        camera_to_worlds = torch.stack(camera_to_worlds).reshape(-1, 4, 4).to(self.device) # [B, 4, 4]
        mvps = torch.stack(mvps).reshape(-1, 4, 4).to(self.device)
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=self.H,
            width=self.W,
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            mvps=mvps,
        )
        return cameras
    
    def generate_spherical_sample_views(self, theta=5, radius=3.0, fovy = 45):
        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []
        mvps = []

        phi_list = torch.linspace(0, 360, 60)
        size = len(phi_list)
        for idx in range(size):
            phi = phi_list[idx]
            input_camera = self.gen_input_camera(
                idx=idx,
                phi=phi,
                theta=theta,
                radius=radius,
                fovy_range=fovy,
            )
            fx.append(input_camera["focal"])
            fy.append(input_camera["focal"])
            cx.append(input_camera["cx"])
            cy.append(input_camera["cy"])
            camera_to_worlds.append(input_camera["c2w"])
            mvps.append(input_camera["mvp_mtx"])
            
        fx = torch.stack(fx).reshape(-1, 1).to(self.device) # [B, 1]
        fy = torch.stack(fy).reshape(-1, 1).to(self.device) # [B, 1]
        cx = torch.stack(cx).reshape(-1, 1).to(self.device) # [B, 1]
        cy = torch.stack(cy).reshape(-1, 1).to(self.device) # [B, 1]
        camera_to_worlds = torch.stack(camera_to_worlds).reshape(-1, 4, 4).to(self.device) # [B, 4, 4]
        mvps = torch.stack(mvps).reshape(-1, 4, 4).to(self.device)
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=self.H,
            width=self.W,
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            mvps=mvps,
        )
        return cameras, phi_list, theta, radius, fovy