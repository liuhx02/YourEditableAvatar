import os
import json
import numpy as np
import torch
from torch import nn
from PIL import Image
import random
import math
import torch.nn.functional as F
from pytorch3d.renderer import FoVPerspectiveCameras as P3DCameras
from pytorch3d.renderer.cameras import _get_sfm_calibration_matrix
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
from tetgs_scene.dataset_readers import sceneLoadTypeCallbacks


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def load_gs_cameras(source_path, image_resolution=1, 
                    load_gt_images=True, load_mask_images=False, max_img_size=1920, white_background=False,
                    remove_indices=[]):
    """Loads Gaussian Splatting camera parameters from a COLMAP reconstruction.

    Args:
        source_path (str): Path to the source data.
        gs_output_path (str): Path to the Gaussian Splatting output.
        image_resolution (int, optional): Factor by which to downscale the images. Defaults to 1.
        load_gt_images (bool, optional): If True, loads the ground truth images. Defaults to True.
        max_img_size (int, optional): Maximum size of the images. Defaults to 1920.
        white_background (bool, optional): If True, uses a white background. Defaults to False.
        remove_indices (list, optional): List of indices to remove. Defaults to [].

    Returns:
        List of GSCameras: List of Gaussian Splatting cameras.
    """
    image_dir = os.path.join(source_path, 'images')
    mask_dir = os.path.join(source_path, "mask")
    
    if os.path.exists(os.path.join(source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](source_path)
    json_cams = []
    camlist = []
    if scene_info.test_cameras:
        camlist.extend(scene_info.test_cameras)
    if scene_info.train_cameras:
        camlist.extend(scene_info.train_cameras)
    for id, cam in enumerate(camlist):
        json_cams.append(camera_to_JSON(id, cam))
    unsorted_camera_transforms = json_cams
        
    # Remove indices
    if len(remove_indices) > 0:
        print("Removing cameras with indices:", remove_indices, sep="\n")
        new_unsorted_camera_transforms = []
        for i in range(len(unsorted_camera_transforms)):
            if i not in remove_indices:
                new_unsorted_camera_transforms.append(unsorted_camera_transforms[i])
        unsorted_camera_transforms = new_unsorted_camera_transforms
        
    # Removing cameras with same image name
    error_names_list = []
    camera_dict = {}
    for i in range(len(unsorted_camera_transforms)):
        name = unsorted_camera_transforms[i]['img_name']
        if name in camera_dict:
            error_names_list.append(name)
        camera_dict[name] = unsorted_camera_transforms[i]
    if len(error_names_list) > 0:
        print("Warning: Found multiple cameras with same GT image name:", error_names_list, sep="\n")
        print("For each GT image, only the last camera will be kept.")
        new_unsorted_camera_transforms = []
        for name in camera_dict:
            new_unsorted_camera_transforms.append(camera_dict[name])
        unsorted_camera_transforms = new_unsorted_camera_transforms
    
    camera_transforms = sorted(unsorted_camera_transforms.copy(), key = lambda x : x['img_name'])

    cam_list = []
    poses_list = []
    extension = '.' + os.listdir(image_dir)[0].split('.')[-1]
    if extension not in ['.jpg', '.png', '.JPG', '.PNG']:
        print(f"Warning: image extension {extension} not supported.")
    else:
        print(f"Found image extension {extension}")
    
    for cam_idx in range(len(camera_transforms)):
        camera_transform = camera_transforms[cam_idx]
        
        # Extrinsics
        rot = np.array(camera_transform['rotation'])
        pos = np.array(camera_transform['position'])
        W2C = np.zeros((4,4))
        W2C[:3, :3] = rot
        W2C[:3, 3] = pos
        W2C[3,3] = 1
        Rt = np.linalg.inv(W2C)
        T = Rt[:3, 3]
        R = Rt[:3, :3].transpose()
        
        # Intrinsics
        width = camera_transform['width']
        height = camera_transform['height']
        fy = camera_transform['fy']
        fx = camera_transform['fx']
        fov_y = focal2fov(fy, height)
        fov_x = focal2fov(fx, width)
        
        # GT data
        id = camera_transform['id']
        name = camera_transform['img_name']
        image_path = os.path.join(image_dir,  name + extension)
        mask_path = os.path.join(mask_dir, name + extension)
        
        if load_gt_images:
            image = Image.open(image_path)
            if white_background:
                im_data = np.array(image.convert("RGBA"))
                bg = np.array([1,1,1])
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            orig_w, orig_h = image.size
            downscale_factor = 1
            if image_resolution in [1, 2, 4, 8]:
                downscale_factor = image_resolution
            if max(orig_h, orig_w) > max_img_size:
                additional_downscale_factor = max(orig_h, orig_w) / max_img_size
                downscale_factor = additional_downscale_factor * downscale_factor
            resolution = round(orig_w/(downscale_factor)), round(orig_h/(downscale_factor))
            resized_image_rgb = PILtoTorch(image, resolution)
            gt_image = resized_image_rgb[:3, ...]
            
            image_height, image_width = None, None
            if load_mask_images:
                mask = Image.open(mask_path)
                resized_mask = PILtoTorch(mask, resolution)
                gt_mask = resized_mask[:1, ...]
            else:
                gt_mask = None
        else:
            gt_image = None
            gt_mask = None
            if image_resolution in [1, 2, 4, 8]:
                downscale_factor = image_resolution
            if max(height, width) > max_img_size:
                additional_downscale_factor = max(height, width) / max_img_size
                downscale_factor = additional_downscale_factor * downscale_factor
            image_height, image_width = round(height/downscale_factor), round(width/downscale_factor)
        
        gs_camera = GSCamera(
            colmap_id=id, image=gt_image, gt_alpha_mask=gt_mask,
            R=R, T=T, FoVx=fov_x, FoVy=fov_y,
            image_name=name, uid=id,
            image_height=image_height, image_width=image_width,)
        cam_list.append(gs_camera)
        poses_list.append(T)

    return cam_list

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def convert_mesh_init(c2w, mesh_path, shape_init_params, anchor_path, device):
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
        "-z": np.array([0, 0, -1]),
    }
    shape_init_mesh_up = "+y"
    shape_init_mesh_front = "+z"
    z_, x_ = (
        dir2vec[shape_init_mesh_up],
        dir2vec[shape_init_mesh_front],
    )
    y_ = np.cross(z_, x_)
    std2mesh = np.stack([x_, y_, z_], axis=0).T
    mesh2std = np.linalg.inv(std2mesh)
    
    # scaling
    scale = np.abs(mesh.vertices).max()
    mesh.vertices = mesh.vertices / scale * shape_init_params
    mesh.vertices = np.dot(mesh2std, mesh.vertices.T).T
    
    pose = c2w[:, :3, :4].detach().cpu().numpy()
    pose = std2mesh @ pose
    c2w[:, :3, :4] = torch.from_numpy(pose).to(device)
    c2w[:, :3, :4] = c2w[:, :3, :4] * scale / shape_init_params 
    c2w[:, :3, 3] = c2w[:, :3, 3] + torch.from_numpy(centroid).to(device)
    c2w[:, 1, 3] = c2w[:, 1, 3] - 0.3
    
    return c2w
    
def gen_tet_camera(idx, radius, theta, phi, fov, H, sample_type, device):
    azimuth_deg = torch.tensor(phi, dtype=torch.float32)
    elevation_deg = torch.full_like(
        azimuth_deg, theta
    )
    camera_distances = torch.full_like(
        elevation_deg, radius
    )
    elevation = elevation_deg * math.pi / 180
    azimuth = azimuth_deg * math.pi / 180
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    ).reshape(-1, 3)
    
    # default scene center at origin
    center = torch.zeros_like(camera_positions)
    # default camera up direction as +z
    up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
        None, :
    ].repeat(1, 1)
    fovy_deg = torch.full_like(
        elevation_deg, fov
    )
    fovy = fovy_deg * math.pi / 180
    
    eval_height = H
    focal_length = (
        0.5 * eval_height / torch.tan(0.5 * fovy)
    )

    if idx < 2:
        focal_scale = 1.4
        center[0:1, 2] -= 0.05
    else:
        if sample_type == "full":
            focal_scale = 1.4 
            center[0:1, 2] -= 0.05
        elif sample_type == "upper":
            focal_scale = 2.2
            center[0:1, 2] += 0.3
        elif sample_type == "lower":
            focal_scale = 1.8
            center[0:1, 2] -= 0.3
    focal_length = focal_scale * focal_length
    
    # c2w
    lookat = F.normalize(center - camera_positions, dim=-1)
    right = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w = torch.cat(
        [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
    )
    c2w[:, 3, 3] = 1.0
    c2w = c2w.to(device)
    
    return c2w, focal_length

'''
相机旋转过程
1. 标准绕y轴相机 ——> 绕tet mesh相机: 同sample_views_uncond文件
2. 绕tet mesh相机 ——> 绕重建mesh相机: 通过npy文件
2. 绕重建mesh相机 ——> 绕点云相机: 同mesh到点云的旋转过程
'''
def sample_gs_cameras(
    theta_list,
    fovy_range,
    radius_range,
    phi_list,
    H=1920, W=1080,
    R_path=None,
    metadata_path=None,
    mesh_path=None,
    shape_init_params=0.9,
    anchor_path=None,
    sample_type="full",
):
    width = W
    height = H
    device = torch.device("cuda")
    
    # fixed focal
    fov = (fovy_range[0] + fovy_range[1]) / 2.

    num_turns = len(phi_list)
    size = 0
    phi_1d_list = []
    theta_1d_list = []
    for idx in range(num_turns):
        size += len(phi_list[idx])
        num_angles = len(phi_list[idx])
        for j in range(num_angles):
            phi_1d_list.append(phi_list[idx][j])
            theta_1d_list.append(theta_list[idx])
    radius = random.uniform(radius_range[0], radius_range[1])
    
    cam_list = []
    poses_list = []
    for idx in range(size):
        theta = theta_1d_list[idx]
        phi = phi_1d_list[idx]
        
        c2w, focal_length = gen_tet_camera(
            idx=idx,
            radius=radius,
            theta=theta,
            phi=phi,
            fov=fov,
            H=H,
            sample_type=sample_type,
            device=device
        )
        cy = focal2fov(focal_length, height)
        cx = focal2fov(focal_length, width)
        
        c2w = convert_mesh_init(c2w, mesh_path, shape_init_params, anchor_path, device)
        if R_path is not None:
            R = np.load(R_path)
            R = torch.tensor(R).to(device).float()
            R = torch.inverse(R)
            poses = R @ c2w
        poses = poses.cpu().numpy()
        poses = poses.reshape(4, 4)
        # mesh to sfm
        if metadata_path is not None:
            metadata = json.load(open(metadata_path))
            worldtogt = np.array(metadata['worldtogt'])
            sdfstudio_to_colmap = np.array([
                [-0.,  1.,  0.,  0.],
                [ 1.,  0., -0., -0.],
                [-0., -0., -1.,  0.],
                [ 0.,  0.,  0.,  1.]]
            )
            poses[:4, 3] = np.dot(poses[:4, 3], worldtogt.T)
            poses[:4, 3] = np.dot(poses[:4, 3], sdfstudio_to_colmap)
            poses[:4, :3] = (np.dot(poses[:4, :3].T, sdfstudio_to_colmap)).T
            poses = poses.reshape(4, 4)
            
        T = poses[:3, 3]
        R = poses[:3, :3] # .transpose()
        gs_camera = GSCamera(
            colmap_id=idx, 
            image=None,
            gt_alpha_mask=None,
            R=R, T=T,
            FoVx=cx, FoVy=cy,
            image_name='', uid=idx,
            image_height=height, image_width=width,
        )
        cam_list.append(gs_camera)
        poses_list.append(T)
    return cam_list


def sample_circle_gs_cameras(
    theta_range,
    fovy_range,
    radius_range,
    phi_range,
    H=1920, W=1080,
    R_path=None,
    metadata_path=None,
    mesh_path=None,
    shape_init_params=0.9,
    anchor_path=None,
    num_views=60,
    sample_type="full"
):
    width = W
    height = H
    device = torch.device("cuda")   
    
    # fixed focal
    fov = (fovy_range[0] + fovy_range[1]) / 2.
    
    # random sampled camera
    phi_list = torch.linspace(phi_range[0], phi_range[1], num_views)  # [0, 360]
    theta = random.uniform(theta_range[0], theta_range[1])
    size = len(phi_list)
    radius = random.uniform(radius_range[0], radius_range[1])
    
    cam_list = []
    poses_list = []
    for idx in range(size):
        phi = phi_list[idx % phi_list.shape[0]]
        
        c2w, focal_length = gen_tet_camera(
            idx=idx,
            radius=radius,
            theta=theta,
            phi=phi,
            fov=fov,
            H=H,
            sample_type=sample_type,
            device=device
        )
        cy = focal2fov(focal_length, height)
        cx = focal2fov(focal_length, width)

        c2w = convert_mesh_init(c2w, mesh_path, shape_init_params, anchor_path, device)
        if R_path is not None:
            R = np.load(R_path)
            R = torch.tensor(R).to(device).float()
            R = torch.inverse(R)
            poses = R @ c2w
        poses = poses.cpu().numpy()
        poses = poses.reshape(4, 4)
        
        # mesh to sfm
        if metadata_path is not None:
            metadata = json.load(open(metadata_path))
            worldtogt = np.array(metadata['worldtogt'])
            
            sdfstudio_to_colmap = np.array([
                [-0.,  1.,  0.,  0.],
                [ 1.,  0., -0., -0.],
                [-0., -0., -1.,  0.],
                [ 0.,  0.,  0.,  1.]]
            )
            poses[:4, 3] = np.dot(poses[:4, 3], worldtogt.T)
            poses[:4, 3] = np.dot(poses[:4, 3], sdfstudio_to_colmap)
            poses[:4, :3] = (np.dot(poses[:4, :3].T, sdfstudio_to_colmap)).T
            poses = poses.reshape(4, 4)
            
        T = poses[:3, 3]
        R = poses[:3, :3] # .transpose()
        gs_camera = GSCamera(
            colmap_id=idx, 
            image=None,
            gt_alpha_mask=None,
            R=R, T=T,
            FoVx=cx, FoVy=cy,
            image_name='', uid=idx,
            image_height=height, image_width=width,
        )
        cam_list.append(gs_camera)
        poses_list.append(T)
    return cam_list


def sample_random_gs_cameras(
    image_root,
    H=1920, W=1080,
    R_path=None,
    metadata_path=None,
    mesh_path=None,
    shape_init_params=0.9,
    anchor_path=None,
    load_gt_images=True,
    num_views=60,
    sample_type="full"
):
    width = W
    height = H
    device = torch.device("cuda")
    name_list = os.listdir(image_root)
    size = len(name_list) 
    
    cam_list = []
    poses_list = []
    phi_list = torch.linspace(0, 360, num_views)
    
    for idx in range(size):
        name = name_list[idx]
        theta, cam_idx, radius, fov = name.replace('.png', '').split('_')
        theta = float(theta)
        radius = float(radius)
        fov = float(fov)
        phi = phi_list[int(cam_idx)]
        
        c2w, focal_length = gen_tet_camera(
            idx=idx,
            radius=radius,
            theta=theta,
            phi=phi,
            fov=fov,
            H=H,
            sample_type=sample_type,
            device=device
        )
        cy = focal2fov(focal_length, height)
        cx = focal2fov(focal_length, width)
        
        c2w = convert_mesh_init(c2w, mesh_path, shape_init_params, anchor_path, device)
        if R_path is not None:
            R = np.load(R_path)
            R = torch.tensor(R).to(device).float()
            R = torch.inverse(R)
            poses = R @ c2w
        poses = poses.cpu().numpy()
        poses = poses.reshape(4, 4)
        # mesh to sfm
        if metadata_path is not None:
            metadata = json.load(open(metadata_path))
            worldtogt = np.array(metadata['worldtogt'])
            sdfstudio_to_colmap = np.array([
                [-0.,  1.,  0.,  0.],
                [ 1.,  0., -0., -0.],
                [-0., -0., -1.,  0.],
                [ 0.,  0.,  0.,  1.]]
            )
            poses[:4, 3] = np.dot(poses[:4, 3], worldtogt.T)
            poses[:4, 3] = np.dot(poses[:4, 3], sdfstudio_to_colmap)
            poses[:4, :3] = (np.dot(poses[:4, :3].T, sdfstudio_to_colmap)).T
            poses = poses.reshape(4, 4)

        T = poses[:3, 3]
        R = poses[:3, :3] # .transpose()
        
        # GT data
        image_path = os.path.join(image_root, name)
        gt_image = None
        if load_gt_images:
            image = Image.open(image_path)
            resolution = height, width
            resized_image_rgb = PILtoTorch(image, resolution)
            gt_image = resized_image_rgb[:3, ...]  
        
        gs_camera = GSCamera(
            colmap_id=idx,
            image=gt_image,
            gt_alpha_mask=None,
            R=R, T=T,
            FoVx=cx, FoVy=cy,
            image_name=name,
            uid=cam_idx,
            image_height=height, image_width=width,
        )
        cam_list.append(gs_camera)
        poses_list.append(T)  
    return cam_list
    

class GSCamera(torch.nn.Module):
    """Class to store Gaussian Splatting camera parameters."""
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 image_height=None, image_width=None,
                 ):
        super(GSCamera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if image is None:
            if image_height is None or image_width is None:
                raise ValueError("Either image or image_height and image_width must be specified")
            else:
                self.image_height = image_height
                self.image_width = image_width
        else:        
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]

            if gt_alpha_mask is not None:
                # self.original_image *= gt_alpha_mask.to(self.data_device)
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
                self.original_mask = gt_alpha_mask.clamp(0.0, 1.0).to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
                self.original_mask = None

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
    @property
    def device(self):
        return self.world_view_transform.device
    
    def to(self, device):
        self.world_view_transform = self.world_view_transform.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)
        return self

    
def create_p3d_cameras(R=None, T=None, K=None, znear=0.0001):
    if R is None:
        R = torch.eye(3)[None]
    if T is None:
        T = torch.zeros(3)[None]
        
    if K is not None:
        p3d_cameras = P3DCameras(R=R, T=T, K=K, znear=0.0001)
    else:
        p3d_cameras = P3DCameras(R=R, T=T, znear=0.0001)
        p3d_cameras.K = p3d_cameras.get_projection_transform().get_matrix().transpose(-1, -2)
        
    return p3d_cameras


def convert_camera_from_gs_to_pytorch3d(gs_cameras, device='cuda'):
    N = len(gs_cameras)
    R = torch.Tensor(np.array([gs_camera.R for gs_camera in gs_cameras])).to(device)
    T = torch.Tensor(np.array([gs_camera.T for gs_camera in gs_cameras])).to(device)
    fx = torch.Tensor(np.array([fov2focal(gs_camera.FoVx, gs_camera.image_width) for gs_camera in gs_cameras])).to(device)
    fy = torch.Tensor(np.array([fov2focal(gs_camera.FoVy, gs_camera.image_height) for gs_camera in gs_cameras])).to(device)
    image_height = torch.tensor(np.array([gs_camera.image_height for gs_camera in gs_cameras]), dtype=torch.int).to(device)
    image_width = torch.tensor(np.array([gs_camera.image_width for gs_camera in gs_cameras]), dtype=torch.int).to(device)
    cx = image_width / 2.  # torch.zeros_like(fx).to(device)
    cy = image_height / 2.  # torch.zeros_like(fy).to(device)
    
    w2c = torch.zeros(N, 4, 4).to(device)
    w2c[:, :3, :3] = R.transpose(-1, -2)
    w2c[:, :3, 3] = T
    w2c[:, 3, 3] = 1
    
    c2w = w2c.inverse()
    c2w[:, :3, 1:3] *= -1
    c2w = c2w[:, :3, :]

    # Pytorch3d-compatible camera matrices
    # Intrinsics
    image_size = torch.Tensor(
        [image_width[0], image_height[0]],
    )[
        None
    ].to(device)
    scale = image_size.min(dim=1, keepdim=True)[0] / 2.0
    c0 = image_size / 2.0
    p0_pytorch3d = (
        -(
            torch.Tensor(
                (cx[0], cy[0]),
            )[
                None
            ].to(device)
            - c0
        )
        / scale
    )
    focal_pytorch3d = (
        torch.Tensor([fx[0], fy[0]])[None].to(device) / scale
    )
    K = _get_sfm_calibration_matrix(
        1, "cpu", focal_pytorch3d, p0_pytorch3d, orthographic=False
    )
    K = K.expand(N, -1, -1)

    # Extrinsics
    line = torch.Tensor([[0.0, 0.0, 0.0, 1.0]]).to(device).expand(N, -1, -1)
    cam2world = torch.cat([c2w, line], dim=1)
    world2cam = cam2world.inverse()
    R, T = world2cam.split([3, 1], dim=-1)
    R = R[:, :3].transpose(1, 2) * torch.Tensor([-1.0, 1.0, -1]).to(device)
    T = T.squeeze(2)[:, :3] * torch.Tensor([-1.0, 1.0, -1]).to(device)

    p3d_cameras = P3DCameras(device=device, R=R, T=T, K=K, znear=0.0001)

    return p3d_cameras

class CamerasWrapper:
    """Class to wrap Gaussian Splatting camera parameters 
    and facilitates both usage and integration with PyTorch3D.
    """
    def __init__(
        self,
        gs_cameras,
        p3d_cameras=None,
        p3d_cameras_computed=False,
        validate=False,
    ) -> None:
        self.gs_cameras = gs_cameras
        self._p3d_cameras = p3d_cameras
        self._p3d_cameras_computed = p3d_cameras_computed
        
        device = gs_cameras[0].device        
        N = len(gs_cameras)
        R = torch.Tensor(np.array([gs_camera.R for gs_camera in gs_cameras])).to(device)
        T = torch.Tensor(np.array([gs_camera.T for gs_camera in gs_cameras])).to(device)
        self.fx = torch.Tensor(np.array([fov2focal(gs_camera.FoVx, gs_camera.image_width) for gs_camera in gs_cameras])).to(device)
        self.fy = torch.Tensor(np.array([fov2focal(gs_camera.FoVy, gs_camera.image_height) for gs_camera in gs_cameras])).to(device)
        self.height = torch.tensor(np.array([gs_camera.image_height for gs_camera in gs_cameras]), dtype=torch.int).to(device)
        self.width = torch.tensor(np.array([gs_camera.image_width for gs_camera in gs_cameras]), dtype=torch.int).to(device)
        self.cx = self.width / 2. 
        self.cy = self.height / 2.
        
        if not validate:
            w2c = torch.zeros(N, 4, 4).to(device)
            w2c[:, :3, :3] = R.transpose(-1, -2)
            w2c[:, :3, 3] = T
            w2c[:, 3, 3] = 1
            c2w = w2c.inverse()
            c2w[:, :3, 1:3] *= -1
            c2w = c2w[:, :3, :]
        else:
            c2w = torch.zeros(N, 4, 4).to(device)
            c2w[:, :3, :3] = R
            c2w[:, :3, 3] = T
            c2w[:, 3, 3] = 1
            c2w = c2w[:, :3, :]
        
        self.camera_to_worlds = c2w

    @property
    def device(self):
        return self.camera_to_worlds.device

    @property
    def p3d_cameras(self):
        if not self._p3d_cameras_computed:
            self._p3d_cameras = convert_camera_from_gs_to_pytorch3d(
                self.gs_cameras,
            )
            self._p3d_cameras_computed = True

        return self._p3d_cameras

    def __len__(self):
        return len(self.gs_cameras)

    def to(self, device):
        self.camera_to_worlds = self.camera_to_worlds.to(device)
        self.fx = self.fx.to(device)
        self.fy = self.fy.to(device)
        self.cx = self.cx.to(device)
        self.cy = self.cy.to(device)
        self.width = self.width.to(device)
        self.height = self.height.to(device)
        
        for gs_camera in self.gs_cameras:
            gs_camera.to(device)

        if self._p3d_cameras_computed:
            self._p3d_cameras = self._p3d_cameras.to(device)

        return self
        
    def get_spatial_extent(self):
        """Returns the spatial extent of the cameras, computed as 
        the extent of the bounding box containing all camera centers."""
        camera_centers = self.p3d_cameras.get_camera_center()
        avg_camera_center = camera_centers.mean(dim=0, keepdim=True)
        half_diagonal = torch.norm(camera_centers - avg_camera_center, dim=-1).max().item()

        radius = 1.1 * half_diagonal
        return radius