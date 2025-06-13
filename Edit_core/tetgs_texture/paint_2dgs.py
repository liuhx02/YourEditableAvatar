import os
import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
import cv2
from rich.console import Console
import time
import random
from tqdm import tqdm
from tetgs_scene.gs_model import GaussianSplattingWrapper
from tetgs_scene.tetgs_edit_2d import EditTetGS, load_inpainted_model
from tetgs_scene.tetgs_model import load_init_model, convert_refined_tetgs_into_masked_gaussians
from tetgs_scene.tetgs_optimizer import OptimizationParams, EditTetGSOptimizer
from utils.loss_utils import ssim, l1_loss, l2_loss
from tetgs_scene.cameras import CamerasWrapper, sample_gs_cameras, sample_circle_gs_cameras
from tetgs_inpainter.mask_mesh_0822 import TexturedMeshModel
from tetgs_inpainter.camera_sample_views_uncond import UncondSampleViews
from tetgs_inpainter.inpaint_utils import (
    normal_based_inpaint,
    prepare_image_guidance,
    prepare_inpainting_image_guidance,
    prepare_fb_inputs,
    prepare_fb_image_guidance
)
from utils.general_utils import trans_gs_mesh, transfer_pcd_color
from tetgs_inpainter.sdxl_tile_refiner import sdxl_refiner
from diffusers import EulerAncestralDiscreteScheduler, AutoencoderKL
from tetgs_inpainter.models.controlnet_union import ControlNetModel_Union
from tetgs_inpainter.pipeline.pipeline_controlnet_union_multi_inpaint_sd_xl import StableDiffusionXLControlNetUnionMultiInpaintPipeline
from tetgs_inpainter.pipeline.pipeline_controlnet_union_sd_xl_img2img import StableDiffusionXLControlNetUnionImg2ImgPipeline

class TetGS_Inpaint(nn.Module):
    def __init__(self, args, comm_cfg):
        super().__init__()
        self.CONSOLE = Console(width=120)
        self.init_params(args)
        self.init_nerfmodel()
        self.init_bind_mesh(comm_cfg)
        self.init_inpainting_model()
        
    def init_params(self, args):
        # ====================Parameters====================
        self.num_device = args.gpu
        self.detect_anomaly = False
        # -----Data parameters-----
        self.downscale_resolution_factor = 1 
        # -----Model parameters-----
        self.use_eval_split = False
        self.n_skip_images_for_eval_split = 8
        self.freeze_gaussians = False
        self.no_rendering = self.freeze_gaussians
        self.n_points_at_start = None  # If None, takes all points in the SfM point cloud
        self.learnable_positions = False  # True in 3DGS
        self.sh_levels = 1
        # -----Mesh render angle threshold-----
        self.render_angle_thres_begin = 70 # 74
        self.render_angle_thres_later = 68   # 68  
        # -----Optimization parameters-----
        # Learning rates and scheduling
        self.num_iterations_start = 1000  # 1200
        self.num_iterations_later = 800
        self.num_iterations_end = 400  # 700
        self.spatial_lr_scale = None
        self.position_lr_init=0.00016
        self.position_lr_final=0.0000016
        self.position_lr_delay_mult=0.01
        self.position_lr_max_steps=30_000
        self.feature_lr=0.0025
        self.opacity_lr=0.05
        self.scaling_lr=0.005
        self.rotation_lr=0.001
        # Loss functions
        self.loss_function = 'l1+dssim'  # 'l1' or 'l2' or 'l1+dssim'
        self.dssim_factor = 0.2
        # bind mesh
        self.bind_to_surface_mesh = True
        if self.bind_to_surface_mesh:
            self.learn_surface_mesh_positions = False
            self.learn_surface_mesh_opacity = False
            self.learn_surface_mesh_scales = False
        # sh 
        self.do_sh_warmup = True
        if self.do_sh_warmup:
            self.sh_warmup_every = 1000
            self.current_sh_levels = 1
        else:
            self.current_sh_levels = self.sh_levels
        # -----Log and save-----
        self.print_loss_every_n_iterations = 100
        # -----Inpaint-----
        self.prompt = args.prompt
        self.seed = args.seed
        self.upscale_to_2048 = args.upscale_to_2048
        if self.seed is None or self.seed < 0:
            self.seed = random.randint(0, 2147483647)
        print(f"Seed: {self.seed}")
        self.dilate_kernel = 30  # 30
        self.sample_type = args.sample_type
        # ====================End of parameters====================
        
        self.output_root = args.output_dir
        self.output_dir = os.path.join(args.output_dir, "tetgs_inpaint")
        self.source_path = args.scene_path
        self.surface_mesh_to_bind_path = args.mesh_path
        self.mesh_name = self.surface_mesh_to_bind_path.split("/")[-1].split(".")[0]
        self.reconstruction_tetgs_path = os.path.join(self.output_dir, "../tetgs_init", 'last.pt')
        self.tetgs_checkpoint_path = self.output_dir
        
        self.use_white_background = args.white_background

        # Bounding box
        if args.bboxmin is None:
            self.use_custom_bbox = False
        else:
            if args.bboxmax is None:
                raise ValueError("You need to specify both bboxmin and bboxmax.")
            self.use_custom_bbox = True
            # Parse bboxmin
            if args.bboxmin[0] == '(':
                args.bboxmin = args.bboxmin[1:]
            if args.bboxmin[-1] == ')':
                args.bboxmin = args.bboxmin[:-1]
            args.bboxmin = tuple([float(x) for x in args.bboxmin.split(",")])
            # Parse bboxmax
            if args.bboxmax[0] == '(':
                args.bboxmax = args.bboxmax[1:]
            if args.bboxmax[-1] == ')':
                args.bboxmax = args.bboxmax[:-1]
            args.bboxmax = tuple([float(x) for x in args.bboxmax.split(",")])
        if self.use_custom_bbox:
            self.fg_bbox_min = args.bboxmin
            self.fg_bbox_max = args.bboxmax
        
        self.CONSOLE.print("-----Parsed parameters-----")
        self.CONSOLE.print("Source path:", self.source_path)
        self.CONSOLE.print("   > Content:", len(os.listdir(self.source_path)))
        self.CONSOLE.print("SUGAR checkpoint path:", self.tetgs_checkpoint_path)
        self.CONSOLE.print("Surface mesh to bind to:", self.surface_mesh_to_bind_path)
        if self.use_custom_bbox:
            self.CONSOLE.print("Foreground bounding box min:", self.fg_bbox_min)
            self.CONSOLE.print("Foreground bounding box max:", self.fg_bbox_max)
        self.CONSOLE.print("Use eval split:", self.use_eval_split)
        self.CONSOLE.print("Use white background:", self.use_white_background)
        self.CONSOLE.print("----------------------------")
        
        # Setup device
        torch.cuda.set_device(self.num_device)
        self.CONSOLE.print("Using device:", self.num_device)
        self.device = torch.device(f'cuda:{self.num_device}')
        self.CONSOLE.print(torch.cuda.memory_summary())
        
        torch.autograd.set_detect_anomaly(self.detect_anomaly)
        
        # Creates save directory if it does not exist
        os.makedirs(self.tetgs_checkpoint_path, exist_ok=True)
        
        # camera parameters
        self.H = 2048
        self.W = 2048
        self.radius_range = [3.0, 3.0]  
        self.fovy_range = [45, 45]
        self.phi_list = [[0, 180, 90, 270, 45, 135, 225, 315],
                    [0, 180, 90, 270, 30, 60, 120, 150, 210, 240, 300, 330],
                    [10, 40, 70, 100, 130, 160, 190, 220, 250, 280, 310, 340]]
        self.theta_list = [5, -15, 25]
    
    def init_bind_mesh(self, comm_cfg):
        # Mesh to bind
        if self.bind_to_surface_mesh:
            self.CONSOLE.print(f'\nLoading mesh to bind to: {self.surface_mesh_to_bind_path}...')
            self.surface_mesh_data = np.load(self.surface_mesh_to_bind_path, allow_pickle=True).item()
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(self.surface_mesh_data["vertices"])
            o3d_mesh.triangles = o3d.utility.Vector3iVector(self.surface_mesh_data["faces"])
            o3d_mesh = trans_gs_mesh(o3d_mesh, os.path.join(self.source_path, comm_cfg.metadata_name), os.path.join(self.source_path, comm_cfg.R_name))
            # initializing mesh color using SFM point cloud
            o3d_mesh = transfer_pcd_color(os.path.join(self.source_path, "sparse/0/points3D.ply"), o3d_mesh)
            self.o3d_mesh = o3d_mesh
            self.CONSOLE.print("Mesh to bind to loaded.")
        
    def init_nerfmodel(self):
        # Load Gaussian Splatting checkpoint 
        self.nerfmodel = GaussianSplattingWrapper(
            source_path=self.source_path,
            load_gt_images=False,
            eval_split=self.use_eval_split,
            eval_split_interval=self.n_skip_images_for_eval_split,
            white_background=self.use_white_background,
        )
        
    def init_inpainting_model(self):
        # ====================Load Inpainting Model================================
        eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("../pretrained_models/stable-diffusion-xl-base-1.0", subfolder="scheduler")
        vae = AutoencoderKL.from_pretrained("../pretrained_models/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        from huggingface_hub import snapshot_download
        controlnet_model = ControlNetModel_Union.from_pretrained("../pretrained_models/controlnet-union-sdxl-1.0-promax", torch_dtype=torch.float16, use_safetensors=True)
        sdxl_base_model = "../pretrained_models/stable-diffusion-xl-base-1.0"
        pipe = StableDiffusionXLControlNetUnionMultiInpaintPipeline.from_pretrained(
            sdxl_base_model, 
            controlnet=controlnet_model, 
            vae=vae,
            torch_dtype=torch.float16,
            scheduler=eulera_scheduler,
        )
        self.pipe = pipe.to(self.device)
        
        img2img_pipe = StableDiffusionXLControlNetUnionImg2ImgPipeline.from_pretrained(
            sdxl_base_model, 
            controlnet=controlnet_model, 
            vae=vae,
            torch_dtype=torch.float16,
            scheduler=eulera_scheduler,
        )
        self.img2img_pipe = img2img_pipe.to(self.device)

    def inpaint_training(self, comm_cfg):
        # Point cloud
        points = torch.randn(1000, 3, device=self.nerfmodel.device)
        colors = torch.rand(1000, 3, device=self.nerfmodel.device)
        
        edit_face_to_tet_idx = self.surface_mesh_data['face_to_global_tet_idx']
        keep_faces_num = self.surface_mesh_data['keep_faces_num']
        keep_vertices_num = self.surface_mesh_data['keep_vertices_num']
            
        # Background tensor if needed
        if self.use_white_background:
            bg_tensor = torch.ones(3, dtype=torch.float, device=self.nerfmodel.device)
        else:
            bg_tensor = None
        
        R_path = os.path.join(self.source_path, comm_cfg.R_name)
        metadata_path = os.path.join(self.source_path, comm_cfg.metadata_name)
        mesh_path = os.path.join(self.source_path, comm_cfg.mesh_name)
        
        # gs camera
        cameras_sample_list = sample_gs_cameras(
            radius_range=self.radius_range,
            fovy_range=self.fovy_range,
            phi_list=self.phi_list,
            theta_list=self.theta_list,
            H=self.H, W=self.W,
            R_path=R_path,
            metadata_path=metadata_path,
            mesh_path=mesh_path,
            shape_init_params=comm_cfg.shape_init_params,
            anchor_path=comm_cfg.anchor_path,
            sample_type=self.sample_type,
        )
        cameras_to_use = CamerasWrapper(cameras_sample_list, validate=True)
        
        # load TexturedMeshModel for mesh mask projection
        mesh_model = TexturedMeshModel(
            render_angle_thres=self.render_angle_thres_begin,
            render_angle_thres_later=self.render_angle_thres_later,
            verts=np.asarray(self.o3d_mesh.vertices), faces=np.asarray(self.o3d_mesh.triangles),
            device=self.nerfmodel.device
        ).to(self.nerfmodel.device)
        
        # mesh camera
        mesh_camera_sample_views = UncondSampleViews(
            device=self.nerfmodel.device,
            radius_range=self.radius_range,
            fovy_range=self.fovy_range,
            phi_list=self.phi_list,
            theta_list=self.theta_list,
            shape_init_params=comm_cfg.shape_init_params,
            anchor_path=comm_cfg.anchor_path,
            H=self.H, W=self.W,
            mesh_path=mesh_path,
            R_path=R_path,
            metadata_path=metadata_path,
            sample_type=self.sample_type
        )
        mesh_sample_cameras = mesh_camera_sample_views.generate_sample_views()
        mesh_sample_cameras = mesh_sample_cameras.to(self.nerfmodel.device)
        assert mesh_sample_cameras.shape[0] == len(cameras_sample_list)
            
        # ====================Initialize TetGS model====================
        # load reconstruction tetgs
        reconstruction_tetgs = load_init_model(
            refined_tetgs_path=self.reconstruction_tetgs_path,
            nerfmodel=self.nerfmodel
        )
        reconstruction_tetgs.adapt_to_cameras(cameras_to_use)
        keep_gaussians = convert_refined_tetgs_into_masked_gaussians(
            refined_tetgs=reconstruction_tetgs,
            edit_face_to_global_tet_idx=edit_face_to_tet_idx
        )
        
        # Construct EditTetGS model
        tetgs = EditTetGS(
            nerfmodel=self.nerfmodel,
            points=points,
            colors=colors,
            keep_gaussians=keep_gaussians,
            keep_faces_num=keep_faces_num,
            keep_vertices_num=keep_vertices_num,
            nerf_cameras=cameras_to_use,
            initialize=False,
            sh_levels=self.sh_levels,
            learnable_positions=self.learnable_positions,
            freeze_gaussians=self.freeze_gaussians,
            surface_mesh_to_bind=self.o3d_mesh,
            surface_mesh_thickness=None,
            learn_surface_mesh_positions=self.learn_surface_mesh_positions,
            learn_surface_mesh_opacity=self.learn_surface_mesh_opacity,
            learn_surface_mesh_scales=self.learn_surface_mesh_scales
        )
        self.CONSOLE.print(f'\nTetGS model has been initialized.')
        self.CONSOLE.print(tetgs)
        self.CONSOLE.print(f'Number of parameters: {sum(p.numel() for p in tetgs.parameters() if p.requires_grad)}')
        self.CONSOLE.print(f'Checkpoints will be saved in {self.tetgs_checkpoint_path}')
        self.CONSOLE.print("\nModel parameters:")
        for name, param in tetgs.named_parameters():
            self.CONSOLE.print(name, param.shape, param.requires_grad)
            
        torch.cuda.empty_cache()
        
        # Compute scene extent
        cameras_spatial_extent = tetgs.get_cameras_spatial_extent(
            nerf_cameras=cameras_to_use
        )
        if self.use_custom_bbox:
            bbox_radius = ((torch.tensor(self.fg_bbox_max) - torch.tensor(self.fg_bbox_min)).norm(dim=-1) / 2.).item()
        else:
            bbox_radius = cameras_spatial_extent   
        spatial_lr_scale = 1.0
        print("Using as spatial_lr_scale:", spatial_lr_scale, "with bbox_radius:", bbox_radius)
        
        # prepare inpainting verts mask
        verts_mask = self.surface_mesh_data['editing_mask']
        verts_mask = torch.tensor(verts_mask).to(self.nerfmodel.device)
        verts_mask = verts_mask.reshape(-1, 1).contiguous().float()
        mask_outputs = None
        epoch = 0
        validate_mask_folder = os.path.join(self.tetgs_checkpoint_path, "validate_mask")
        os.makedirs(validate_mask_folder, exist_ok=True)
        
        # ====================Loss function====================
        if self.loss_function == 'l1':
            loss_fn = l1_loss
        elif self.loss_function == 'l2':
            loss_fn = l2_loss
        elif self.loss_function == 'l1+dssim':
            def loss_fn(pred_rgb, gt_rgb):
                return (1.0 - self.dssim_factor) * l1_loss(pred_rgb, gt_rgb) + self.dssim_factor * (1.0 - ssim(pred_rgb, gt_rgb))
        self.CONSOLE.print(f'Using loss function: {self.loss_function}')
        
        for cam_idx in range(len(cameras_sample_list)):
            print(f"Rendering image with index {cam_idx}.")
            if cam_idx < 4:
                num_iterations = self.num_iterations_start
            elif cam_idx < 8:
                num_iterations = self.num_iterations_later
            else:
                num_iterations = self.num_iterations_end
            
            # ====================Initialize optimizer====================
            opt_params = OptimizationParams(
                iterations=num_iterations,
                position_lr_init=self.position_lr_init,
                position_lr_final=self.position_lr_final,
                position_lr_delay_mult=self.position_lr_delay_mult,
                position_lr_max_steps=self.position_lr_max_steps,
                feature_lr=self.feature_lr,
                opacity_lr=self.opacity_lr,
                scaling_lr=self.scaling_lr,
                rotation_lr=self.rotation_lr,
            )
            optimizer = EditTetGSOptimizer(tetgs, opt_params, spatial_lr_scale=spatial_lr_scale)
            self.CONSOLE.print("Optimizer initialized.")
            self.CONSOLE.print("Optimizable parameters:")
            for param_group in optimizer.optimizer.param_groups:
                self.CONSOLE.print(param_group['name'], param_group['lr'])
            
            # generate front&back images
            if cam_idx == 0:
                save_mask1, save_mask2, ori_image_path1, ori_image_path2, normal1, normal2 = \
                    prepare_fb_inputs(
                        mesh_model, tetgs,
                        verts_mask, mesh_sample_cameras, validate_mask_folder, keep_vertices_num,
                        cameras_to_use, bg_tensor, self.current_sh_levels
                    )
                prepare_fb_image_guidance(
                    self.pipe,
                    save_mask1[:, :, None], save_mask2[:, :, None],
                    ori_image_path1, ori_image_path2,
                    normal1, normal2,
                    self.prompt, self.seed,
                )
                
            # ====================Start training====================
            tetgs.train()
            iteration = 0
            train_losses = []
            t0 = time.time()
            
            last_sh = tetgs._edit_sh_coordinates_dc.detach().clone()
            last_densities = tetgs.all_edit_densities.detach().clone()
            
            # newly inpainted triangles: [M, 1]
            mask_outputs = mesh_model.prepare_mask_normal_for_inpainting(
                verts_mask=verts_mask,
                cam_idx=cam_idx,
                sample_cameras=mesh_sample_cameras,
                validate_mask_folder=validate_mask_folder,
                keep_vertices_num=keep_vertices_num,
            )
            verts_mask = mask_outputs["verts_mask"]
            mesh = mask_outputs["mesh"]
            diff_verts_mask_dilate = mask_outputs["mask_dilate"]
            mask_blur = mask_outputs["mask_blur"]
            mask_proj = mask_outputs["mask_proj"]
            gb_normal = mask_outputs["gb_normal"]
            save_mask_aa_path = mask_outputs["save_mask_aa_path"]
            mask_aa_and_normal = mask_outputs["mask_aa_and_normal"]
            # save mask_proj
            save_mask_proj_path = os.path.join(validate_mask_folder, f'{cam_idx}_proj.png')
            cv2.imwrite(save_mask_proj_path, ((mask_proj.cpu().numpy() * 255).astype(np.uint8)).squeeze())
            
            np_tri = np.asarray(mesh.triangles)
            diff_verts_mask_dilate_np = diff_verts_mask_dilate.cpu().numpy()
            # 0: keep, 1: edit
            hit_tri_id_keep_set = np.where((diff_verts_mask_dilate_np[np_tri[:, 0]] == 0) &
                                            (diff_verts_mask_dilate_np[np_tri[:, 1]] == 0) &
                                            (diff_verts_mask_dilate_np[np_tri[:, 2]] == 0))[0]
            faces_mask_np = np.zeros((np_tri.shape[0], 1), dtype=np.uint8)
            faces_mask_np[hit_tri_id_keep_set] = 1
            faces_mask = torch.tensor(faces_mask_np).to(self.nerfmodel.device)
            update_gaussian_mask = faces_mask[tetgs._edit_face_indices + keep_faces_num].reshape(-1)
            
            # prepare inpainted image as guidance
            initial_outputs = tetgs.render_image_gaussian_rasterizer(
                nerf_cameras=cameras_to_use,
                camera_indices=cam_idx,
                verbose=False,
                bg_color=bg_tensor,
                sh_deg=self.current_sh_levels-1,
                sh_rotations=None,
                compute_color_in_rasterizer=False,
                compute_covariance_in_rasterizer=True, 
                return_2d_radii=False,
                quaternions=None
            )
            tetgs_image = initial_outputs.detach().clone()
            tetgs_image = tetgs_image.nan_to_num().clamp(min=0, max=1)
            tetgs_image = (tetgs_image * 255.0).cpu().numpy()
            tetgs_image = cv2.cvtColor(tetgs_image[..., :3], cv2.COLOR_RGB2BGR)
            tetgs_image_path = os.path.join(validate_mask_folder, f'{cam_idx}_render.png')
            cv2.imwrite(tetgs_image_path, tetgs_image) 
            
            if cam_idx < 2:
                gt_rgb = prepare_image_guidance(
                    tetgs=tetgs,
                    cam_idx=cam_idx,
                    image_folder=validate_mask_folder,
                    initial_outputs=initial_outputs,
                    mask_bb=mask_aa_and_normal,
                    device=self.nerfmodel.device,
                    downscale_resolution_factor=self.downscale_resolution_factor
                ) 
            else:
                # normal_img = gb_normal.detach().clone()
                # normal_img = normal_img.nan_to_num().clamp(min=0, max=1)
                # normal_img = (normal_img * 255.0).cpu().numpy().astype(np.uint8)
                
                inpainting_img, mask_blur = normal_based_inpaint(
                    pipe=self.pipe,
                    mask_path=save_mask_aa_path,
                    tetgs_image_path=tetgs_image_path,
                    normal_img=gb_normal,
                    prompt=self.prompt, 
                    seed=self.seed,
                    dilate_kernel=self.dilate_kernel,   # 30 # TODO: add background
                )
                gt_rgb = prepare_inpainting_image_guidance(
                    tetgs=tetgs,
                    cam_idx=cam_idx,
                    save_folder=validate_mask_folder,
                    initial_outputs=initial_outputs,
                    inpainting_img=inpainting_img,
                    mask_blur=mask_blur,
                    mask_proj=mask_proj,
                    device=self.nerfmodel.device,
                    downscale_resolution_factor=self.downscale_resolution_factor
                )
            
            for batch in range(9_999_999):
                if iteration >= num_iterations:
                    break
                iteration += 1
                optimizer.update_learning_rate(iteration)
                
                # Computing rgb predictions
                if not self.no_rendering:
                    outputs = tetgs.render_image_gaussian_rasterizer(
                        nerf_cameras=cameras_to_use,
                        camera_indices=cam_idx,
                        verbose=False,
                        bg_color=bg_tensor,
                        sh_deg=self.current_sh_levels-1,
                        sh_rotations=None,
                        compute_color_in_rasterizer=False,
                        compute_covariance_in_rasterizer=True, 
                        return_2d_radii=False,
                        quaternions=None
                    )
                    pred_rgb = outputs.view(-1, tetgs.image_height, tetgs.image_width, 3)
                    pred_rgb = pred_rgb.transpose(-1, -2).transpose(-2, -3)
                    # Compute loss 
                    loss = loss_fn(pred_rgb, gt_rgb)
                    
                # Update parameters
                loss.backward()
                # Optimization step
                optimizer.step()
                optimizer.zero_grad(set_to_none = True)
                
                # Print loss
                if iteration==1 or iteration % self.print_loss_every_n_iterations == 0:
                    self.CONSOLE.print(f'\n-------------------\nIteration: {iteration}')
                    train_losses.append(loss.detach().item())
                    self.CONSOLE.print(f"loss: {loss:>7f}  [{iteration:>5d}/{num_iterations:>5d}]",
                        "computed in", (time.time() - t0) / 60., "minutes.")
                    with torch.no_grad():
                        self.CONSOLE.print("------Stats-----")
                        self.CONSOLE.print("---Min, Max, Mean, Std")
                        self.CONSOLE.print("Points:", tetgs.points.min().item(), tetgs.points.max().item(), tetgs.points.mean().item(), tetgs.points.std().item(), sep='   ')
                        self.CONSOLE.print("Scaling factors:", tetgs.scaling.min().item(), tetgs.scaling.max().item(), tetgs.scaling.mean().item(), tetgs.scaling.std().item(), sep='   ')
                        self.CONSOLE.print("Quaternions:", tetgs.quaternions.min().item(), tetgs.quaternions.max().item(), tetgs.quaternions.mean().item(), tetgs.quaternions.std().item(), sep='   ')
                        self.CONSOLE.print("Sh coordinates dc:", tetgs._edit_sh_coordinates_dc.min().item(), tetgs._edit_sh_coordinates_dc.max().item(), tetgs._edit_sh_coordinates_dc.mean().item(), tetgs._edit_sh_coordinates_dc.std().item(), sep='   ')
                        if self.sh_levels > 1:
                            self.CONSOLE.print("Sh coordinates rest:", tetgs._edit_sh_coordinates_rest.min().item(), tetgs._edit_sh_coordinates_rest.max().item(), tetgs._edit_sh_coordinates_rest.mean().item(), tetgs._edit_sh_coordinates_rest.std().item(), sep='   ')
                        self.CONSOLE.print("Opacities:", tetgs.strengths.min().item(), tetgs.strengths.max().item(), tetgs.strengths.mean().item(), tetgs.strengths.std().item(), sep='   ')
                    t0 = time.time()
            
            # Reset attr by mask
            current_sh = tetgs._edit_sh_coordinates_dc.detach().clone()
            current_densities = tetgs.all_edit_densities.detach().clone()
            self.CONSOLE.print("Sh coordinates dc:", tetgs._edit_sh_coordinates_dc.min().item(), tetgs._edit_sh_coordinates_dc.max().item(), tetgs._edit_sh_coordinates_dc.mean().item(), tetgs._edit_sh_coordinates_dc.std().item(), sep='   ')
            
            # edit gaussians中的keep gaussians
            keep_gaussian_indices = torch.where(update_gaussian_mask == 0)[0]
            current_sh[keep_gaussian_indices, :] = last_sh[keep_gaussian_indices, :]
            current_densities[keep_gaussian_indices, :] = last_densities[keep_gaussian_indices, :]
            
            # new parameters
            tetgs._edit_sh_coordinates_dc = torch.nn.Parameter(
                current_sh,
                requires_grad=True and (not self.freeze_gaussians)
            ).to(self.nerfmodel.device)
            tetgs.all_edit_densities = torch.nn.Parameter(
                current_densities, 
                requires_grad=tetgs.learn_opacities
            ).to(self.nerfmodel.device)
            self.CONSOLE.print("Sh coordinates dc:", tetgs._edit_sh_coordinates_dc.min().item(), tetgs._edit_sh_coordinates_dc.max().item(), tetgs._edit_sh_coordinates_dc.mean().item(), tetgs._edit_sh_coordinates_dc.std().item(), sep='   ')
        
        # save last model
        self.CONSOLE.print("Saving model...")
        model_path = os.path.join(self.tetgs_checkpoint_path, 'last.pt')
        tetgs.save_model(
            path=model_path,
            train_losses=train_losses,
            epoch=epoch,
            iteration=cam_idx,
            optimizer_state_dict=optimizer.state_dict(),
        )
        self.CONSOLE.print("Model saved.")
        self.CONSOLE.print(f"Training finished after {epoch} epochs with loss={loss.detach().item()}.")

    def validate(self, comm_cfg):
        inpainted_tetgs_ckpt_path = os.path.join(self.tetgs_checkpoint_path, "last.pt")
        print(f"\nLoading config {inpainted_tetgs_ckpt_path}...")
        tetgs_inpaint = load_inpainted_model(inpainted_tetgs_ckpt_path, self.nerfmodel)
        validate_output_folder = os.path.join(self.tetgs_checkpoint_path, "validation_inpaint")
        os.makedirs(validate_output_folder, exist_ok=True)
        
        cam_pack = {"radius": 3., "fovy": 45, "theta": 5, "H": 1024, "W": 1024}
        if self.upscale_to_2048:
            cam_pack = {"radius": 3., "fovy": 45, "theta": 5, "H": 2048, "W": 2048}
        cameras_sample_list = sample_circle_gs_cameras(
            radius_range=[cam_pack["radius"], cam_pack["radius"]],
            fovy_range=[cam_pack["fovy"], cam_pack["fovy"]],
            phi_range=[0, 360],
            theta_range=[cam_pack["theta"], cam_pack["theta"]],
            H=cam_pack["H"], W=cam_pack["W"],
            R_path=os.path.join(self.source_path, comm_cfg.R_name),
            metadata_path=os.path.join(self.source_path, comm_cfg.metadata_name),
            mesh_path=os.path.join(self.source_path, comm_cfg.mesh_name),
            shape_init_params=comm_cfg.shape_init_params,
            anchor_path=comm_cfg.anchor_path,
            
        )
        cameras_to_use = CamerasWrapper(cameras_sample_list, validate=True)
        
        tetgs_inpaint.eval()
        tetgs_inpaint.adapt_to_cameras(cameras_to_use)
        tetgs_images = []
        for cam_idx in tqdm(range(len(cameras_to_use.gs_cameras))):
            with torch.no_grad():
                tetgs_image = tetgs_inpaint.render_image_gaussian_rasterizer(
                    nerf_cameras=cameras_to_use,
                    camera_indices=cam_idx,
                    bg_color=1. * torch.Tensor([1.0, 1.0, 1.0]).to(tetgs_inpaint.device),
                    sh_deg=0,
                    compute_color_in_rasterizer=False,
                    compute_covariance_in_rasterizer=True
                ).nan_to_num().clamp(min=0, max=1)
                tetgs_image = (tetgs_image * 255.0).cpu().numpy()
                tetgs_image = cv2.cvtColor(tetgs_image[..., :3], cv2.COLOR_RGB2BGR)
                cv2.imwrite(validate_output_folder + "/frame%04d.png" % cam_idx, tetgs_image) 
                tetgs_images.append(tetgs_image)
        return tetgs_images, cam_pack, cameras_to_use
                
    def prepare_refine_guidance(self, comm_cfg):
        edit_images_inpainted, cam_pack, cameras_to_use = self.validate(comm_cfg)
        edit_images_refined = sdxl_refiner(
            self.img2img_pipe,
            edit_images_inpainted,
            input_prompt=self.prompt,
            upscale_to_2048=self.upscale_to_2048,
        )
        recon_tetgs_ckpt_path = self.reconstruction_tetgs_path
        print(f"\nLoading config {recon_tetgs_ckpt_path}...")
        verts_mask = self.surface_mesh_data['editing_mask']
        verts_mask = torch.tensor(verts_mask).to(self.nerfmodel.device)
        verts_mask = verts_mask.reshape(-1, 1).contiguous().float()
        
        mesh_camera_sample_views = UncondSampleViews(
            device=self.nerfmodel.device,
            H=cam_pack["H"], W=cam_pack["W"],
            R_path=os.path.join(self.source_path, comm_cfg.R_name),
            metadata_path=os.path.join(self.source_path, comm_cfg.metadata_name),
            mesh_path=os.path.join(self.source_path, comm_cfg.mesh_name),
            shape_init_params=comm_cfg.shape_init_params,
            anchor_path=comm_cfg.anchor_path
        )
        mesh_spherical_cameras, phi_list, theta, radius, fovy = mesh_camera_sample_views.generate_spherical_sample_views(
            theta=cam_pack["theta"], radius=cam_pack["radius"], fovy=cam_pack["fovy"])
        mesh_spherical_cameras = mesh_spherical_cameras.to(self.nerfmodel.device)
        assert mesh_spherical_cameras.shape[0] == len(cameras_to_use.gs_cameras)
        
        # mesh model for generating edit_mask
        mesh_model = TexturedMeshModel(
            render_angle_thres=self.render_angle_thres_begin,
            render_angle_thres_later=self.render_angle_thres_later,
            verts=np.asarray(self.o3d_mesh.vertices), faces=np.asarray(self.o3d_mesh.triangles),
            device=self.nerfmodel.device).to(self.nerfmodel.device)
        recon_tetgs = load_init_model(recon_tetgs_ckpt_path, self.nerfmodel)
        recon_tetgs.eval()
        recon_tetgs.adapt_to_cameras(cameras_to_use)
        blend_image_root = os.path.join(self.output_root, comm_cfg.blend_image_root)
        os.makedirs(blend_image_root, exist_ok=True)
        
        for cam_idx in range(len(cameras_to_use.gs_cameras)):
            concat_mask_outputs = mesh_model.get_concat_mask(
                idx=cam_idx,
                sample_cameras=mesh_spherical_cameras,
                verts_mask=verts_mask
            )
            edit_mask = concat_mask_outputs["edit_mask"].view(-1, cam_pack["H"], cam_pack["W"], 1).repeat(1, 1, 1, 3)
            keep_mask = concat_mask_outputs["keep_mask"].view(-1, cam_pack["H"], cam_pack["W"], 1).repeat(1, 1, 1, 3)
            
            # render recon image
            recon_tetgs_image = recon_tetgs.render_image_gaussian_rasterizer(
                nerf_cameras=cameras_to_use,
                camera_indices=cam_idx,
                bg_color=1. * torch.Tensor([1.0, 1.0, 1.0]).to(recon_tetgs.device),
                sh_deg=0,
                compute_color_in_rasterizer=False,
                compute_covariance_in_rasterizer=True
            ).nan_to_num().clamp(min=0, max=1)
            recon_tetgs_image = recon_tetgs_image.view(-1, cam_pack["H"], cam_pack["W"], 3).detach().clone()
            
            # read edit_refined_image
            edit_image = torch.from_numpy(np.array(edit_images_refined[cam_idx])) / 255.0
            if len(edit_image.shape) == 3:
                edit_image = edit_image.permute(2, 0, 1)
            else:
                edit_image = edit_image.unsqueeze(dim=-1).permute(2, 0, 1)
            edit_image = edit_image[:3, ...].clamp(0.0, 1.0).to(recon_tetgs.device).permute(1, 2, 0).view(-1, cam_pack["H"], cam_pack["W"], 3)
            concat_image = recon_tetgs_image * keep_mask + edit_image * edit_mask
            concat_image = concat_image.reshape(cam_pack["H"], cam_pack["W"], 3)
            concat_image = (concat_image * 255.0).cpu().numpy().astype(np.uint8)
            concat_image = cv2.cvtColor(concat_image, cv2.COLOR_RGB2BGR)
            
            image_name = str(theta) + "_" + str(cam_idx) + "_" + str(radius) + "_" + str(fovy) + ".png"
            cv2.imwrite(os.path.join(blend_image_root, image_name), concat_image)     