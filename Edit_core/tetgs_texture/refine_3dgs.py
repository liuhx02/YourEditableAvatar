import os
import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
from rich.console import Console
import time
from tqdm import tqdm
import cv2
from tetgs_scene.gs_model import GaussianSplattingWrapper
from tetgs_scene.tetgs_edit_3d import Edit3DTetGS, load_refined_model
from tetgs_scene.tetgs_edit_2d import load_inpainted_model, convert_edited_tetgs_into_masked_gaussians
from tetgs_scene.tetgs_optimizer import OptimizationParams, EditTetGSOptimizer
from utils.loss_utils import ssim, l1_loss, l2_loss
from utils.general_utils import trans_gs_mesh, transfer_pcd_color
from tetgs_scene.cameras import CamerasWrapper, sample_random_gs_cameras, sample_circle_gs_cameras

def get_gt_image(cam_list, camera_indices, to_cuda=False):
    gt_image = cam_list[camera_indices].original_image
    if to_cuda:
        gt_image = gt_image.cuda()
    return gt_image.permute(1, 2, 0)

class TetGS_Refine(nn.Module):
    def __init__(self, args, comm_cfg):
        super().__init__()
        self.CONSOLE = Console(width=120)
        self.init_params(args, comm_cfg)
        self.init_bind_mesh(comm_cfg)
        self.init_nerfmodel()
        
    def init_params(self, args, comm_cfg):
        # ====================Parameters====================
        self.num_device = args.gpu
        self.detect_anomaly = False
        # -----Model parameters-----
        self.use_eval_split = False
        self.n_skip_images_for_eval_split = 8
        self.freeze_gaussians = False
        self.no_rendering = self.freeze_gaussians
        self.n_points_at_start = None  # If None, takes all points in the SfM point cloud
        self.learnable_positions = True  # True for refined_editing stage
        self.use_same_scale_in_all_directions = False  
        self.sh_levels = 4
        # scaling regularization
        self.scaling_reg = True
        # -----Optimization parameters-----
        # Learning rates and scheduling
        self.num_iterations = args.refinement_iterations
        self.spatial_lr_scale = None
        self.position_lr_init=0.00016
        self.position_lr_final=0.0000016
        self.position_lr_delay_mult=0.01
        self.position_lr_max_steps=30_000
        self.feature_lr=0.0025
        self.opacity_lr=0.05
        self.scaling_lr=0.005
        self.rotation_lr=0.001
        self.train_num_images_per_batch = 1  # 1 for full images
        # Loss functions
        self.loss_function = 'l1+dssim'  # 'l1' or 'l2' or 'l1+dssim'
        self.dssim_factor = 0.2
        # Regularization
        self.enforce_entropy_regularization = False
        self.regularize_sdf = False
        self.bind_to_surface_mesh = True
        if self.bind_to_surface_mesh:
            self.learn_surface_mesh_positions = True  
            self.learn_surface_mesh_opacity = True 
            self.learn_surface_mesh_scales = True
        # sh
        self.do_sh_warmup = True  # Should be True
        if self.do_sh_warmup:
            self.sh_warmup_every = 1000
            self.current_sh_levels = 1
        else:
            self.current_sh_levels = self.sh_levels
        # -----Log and save-----
        self.print_loss_every_n_iterations = 50
        self.save_model_every_n_iterations = 1_000_000
        self.save_milestones = [2000, 4_000, 5_000, 7_000]
        # ====================End of parameters====================
        
        self.output_root = args.output_dir
        self.output_dir = os.path.join(args.output_dir, "refined")
        self.edited_2dgs_path = os.path.join(self.output_dir, "../tetgs_inpaint", 'last.pt')
        self.source_path = args.scene_path
        self.surface_mesh_to_bind_path = args.mesh_path
        self.mesh_name = self.surface_mesh_to_bind_path.split("/")[-1].split(".")[0] 
        self.blend_image_root = os.path.join(self.output_root, comm_cfg.blend_image_root)
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
        device = torch.device(f'cuda:{self.num_device}')
        self.CONSOLE.print(torch.cuda.memory_summary())
        
        torch.autograd.set_detect_anomaly(self.detect_anomaly)
        
        # Creates save directory if it does not exist
        os.makedirs(self.tetgs_checkpoint_path, exist_ok=True)
        
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
        self.nerfmodel = GaussianSplattingWrapper(
            source_path=self.source_path,
            load_gt_images=False,
            eval_split=self.use_eval_split,
            eval_split_interval=self.n_skip_images_for_eval_split,
            white_background=self.use_white_background,
        )
        
    def refined_editing(self, comm_cfg):
        # camera parameters
        H = 2048
        W = 2048
        R_path = os.path.join(self.source_path, comm_cfg.R_name)
        metadata_path = os.path.join(self.source_path, comm_cfg.metadata_name)
        mesh_path = os.path.join(self.source_path, comm_cfg.mesh_name)
        blend_image_root = os.path.join(self.output_root, comm_cfg.blend_image_root)
        
        # gs camera
        cameras_sample_list = sample_random_gs_cameras(
            image_root=blend_image_root,
            H=H, W=W,
            R_path=R_path,
            metadata_path=metadata_path,
            mesh_path=mesh_path,
            shape_init_params=comm_cfg.shape_init_params,
            anchor_path=comm_cfg.anchor_path,
            load_gt_images=True
        )
        cameras_to_use = CamerasWrapper(cameras_sample_list, validate=True)
        
        # Point cloud
        points = torch.randn(1000, 3, device=self.nerfmodel.device)
        colors = torch.rand(1000, 3, device=self.nerfmodel.device)
            
        # Background tensor if needed
        if self.use_white_background:
            bg_tensor = torch.ones(3, dtype=torch.float, device=self.nerfmodel.device)
        else:
            bg_tensor = None
        
        # ====================Initialize TetGS model====================
        # Construct TetGS model
        edited_2dgs_tetgs = load_inpainted_model(
            self.edited_2dgs_path,
            self.nerfmodel
        )
        edited_2dgs_tetgs.adapt_to_cameras(cameras_to_use)
        
        keep_faces_num = self.surface_mesh_data['keep_faces_num']
        keep_vertices_num = self.surface_mesh_data['keep_vertices_num']
        edited_all_gaussians = convert_edited_tetgs_into_masked_gaussians(
            edited_tetgs=edited_2dgs_tetgs
        )
        
        # Construct Edit3DTetGS model
        tetgs = Edit3DTetGS(
            nerfmodel=self.nerfmodel,
            points=points,
            colors=colors,
            edited_all_gaussians=edited_all_gaussians,
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
        
        # ====================Initialize optimizer====================
        opt_params = OptimizationParams(
            iterations=self.num_iterations,
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
        self.CONSOLE.print("Optimization parameters:")
        self.CONSOLE.print(opt_params)
        self.CONSOLE.print("Optimizable parameters:")
        for param_group in optimizer.optimizer.param_groups:
            self.CONSOLE.print(param_group['name'], param_group['lr'])
            
        # ====================Loss function====================
            if self.loss_function == 'l1':
                loss_fn = l1_loss
            elif self.loss_function == 'l2':
                loss_fn = l2_loss
            elif self.loss_function == 'l1+dssim':
                def loss_fn(pred_rgb, gt_rgb):
                    return (1.0 - self.dssim_factor) * l1_loss(pred_rgb, gt_rgb) + self.dssim_factor * (1.0 - ssim(pred_rgb, gt_rgb))
            self.CONSOLE.print(f'Using loss function: {self.loss_function}')
        
        # ====================Start training====================
        tetgs.train()
        epoch = 0
        iteration = 0
        train_losses = []
        t0 = time.time()
            
        for batch in range(9_999_999):
            if iteration >= self.num_iterations:
                break
            
            # Shuffle images
            shuffled_idx = torch.randperm(len(cameras_to_use))
            train_num_images = len(shuffled_idx)
            
            # We iterate on images
            for i in range(0, train_num_images, self.train_num_images_per_batch):
                iteration += 1
                
                # Update learning rates
                optimizer.update_learning_rate(iteration)
                
                start_idx = i
                end_idx = min(i + self.train_num_images_per_batch, train_num_images)
                
                camera_indices = shuffled_idx[start_idx:end_idx]
                
                # Computing rgb predictions
                if not self.no_rendering:
                    outputs = tetgs.render_image_gaussian_rasterizer(
                        camera_indices=camera_indices.item(),
                        verbose=False,
                        bg_color=bg_tensor,
                        sh_deg=self.current_sh_levels-1,
                        sh_rotations=None,
                        compute_color_in_rasterizer=False,
                        compute_covariance_in_rasterizer=True, 
                        return_2d_radii=False,
                        quaternions=None,
                        use_same_scale_in_all_directions=self.use_same_scale_in_all_directions,
                        return_opacities=self.enforce_entropy_regularization,
                    )
                    pred_rgb = outputs.view(-1, tetgs.image_height, tetgs.image_width, 3)
                    pred_rgb = pred_rgb.transpose(-1, -2).transpose(-2, -3)
                    # Gather rgb ground truth
                    gt_image = get_gt_image(cam_list=cameras_sample_list,
                                            camera_indices=camera_indices)
                    gt_rgb = gt_image.view(-1, tetgs.image_height, tetgs.image_width, 3)
                    gt_rgb = gt_rgb.transpose(-1, -2).transpose(-2, -3)
                    # Compute loss 
                    loss = loss_fn(pred_rgb, gt_rgb)
                    # more weight for front & back & side views
                    if camera_indices == 0 or camera_indices == 14 or camera_indices == 29 or camera_indices == 44:
                        loss = loss * 10
                else:
                    loss = 0.
                    
                # scaling regularization
                if self.scaling_reg:
                    radii = tetgs.radii
                    thresh_scaling_max = radii * 1.0
                    thresh_scaling_ratio = 10.0
                    max_vals, _ = torch.max(tetgs.edit_scaling, dim=-1)
                    min_vals, _ = torch.min(tetgs.edit_scaling, dim=-1)
                    ratio = max_vals / min_vals
                    thresh_idxs = (max_vals > thresh_scaling_max) & (ratio > thresh_scaling_ratio)
                    if thresh_idxs.sum() > 0:
                        loss_scaling = max_vals[thresh_idxs].mean() * 1.0
                        loss = loss + loss_scaling
                        
                # Update parameters
                loss.backward()
                # Optimization step
                optimizer.step()
                optimizer.zero_grad(set_to_none = True)
                
                # Print loss
                if iteration == 1 or iteration % self.print_loss_every_n_iterations == 0:
                    self.CONSOLE.print(f'\n-------------------\nIteration: {iteration}')
                    train_losses.append(loss.detach().item())
                    self.CONSOLE.print(f"loss: {loss:>7f}  [{iteration:>5d}/{self.num_iterations:>5d}]",
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
                
                # Save model
                if (iteration % self.save_model_every_n_iterations == 0) or (iteration in self.save_milestones):
                    self.CONSOLE.print("Saving model...")
                    model_path = os.path.join(self.tetgs_checkpoint_path, f'{iteration}.pt')
                    tetgs.save_model(path=model_path,
                                    train_losses=train_losses,
                                    epoch=epoch,
                                    iteration=iteration,
                                    optimizer_state_dict=optimizer.state_dict(),
                                    )
                    self.CONSOLE.print("Model saved.")

                if iteration >= self.num_iterations:
                    break
                
                if self.do_sh_warmup and (iteration > 0) and (self.current_sh_levels < self.sh_levels) and (iteration % self.sh_warmup_every == 0):
                    self.current_sh_levels += 1
                    self.CONSOLE.print("Increasing number of spherical harmonics levels to", self.current_sh_levels)
            epoch += 1
            
        self.CONSOLE.print(f"Training finished after {self.num_iterations} iterations with loss={loss.detach().item()}.")
        self.CONSOLE.print("Saving final model...")
        model_path = os.path.join(self.tetgs_checkpoint_path, 'last.pt')
        tetgs.save_model(path=model_path,
                        train_losses=train_losses,
                        epoch=epoch,
                        iteration=iteration,
                        optimizer_state_dict=optimizer.state_dict(),
                        )

        self.CONSOLE.print("Final model saved.")
        return model_path
    
    def validate(self, comm_cfg):
        refined_tetgs_ckpt_path = os.path.join(self.tetgs_checkpoint_path, "last.pt")
        print(f"\nLoading config {refined_tetgs_ckpt_path}...")
        tetgs_refined = load_refined_model(
            refined_tetgs_path=refined_tetgs_ckpt_path,
            refined_tetgs_2dgs_path=self.edited_2dgs_path,
            nerfmodel=self.nerfmodel,
            edit_sh_levels=self.sh_levels
        )
        validate_output_folder = os.path.join(self.tetgs_checkpoint_path, "validation_refine")
        os.makedirs(validate_output_folder, exist_ok=True)
        
        cameras_sample_list = sample_circle_gs_cameras(
            radius_range=[3., 3.],
            fovy_range=[45, 45],
            phi_range=[0, 360],
            theta_range=[5, 5],
            H=1024, W=1024,
            R_path=os.path.join(self.source_path, comm_cfg.R_name),
            metadata_path=os.path.join(self.source_path, comm_cfg.metadata_name),
            mesh_path=os.path.join(self.source_path, comm_cfg.mesh_name),
            shape_init_params=comm_cfg.shape_init_params,
            anchor_path=comm_cfg.anchor_path
        )
        cameras_to_use = CamerasWrapper(cameras_sample_list, validate=True)
        
        tetgs_refined.eval()
        tetgs_refined.adapt_to_cameras(cameras_to_use)
        for cam_idx in tqdm(range(len(cameras_to_use.gs_cameras))):
            with torch.no_grad():
                tetgs_image = tetgs_refined.render_image_gaussian_rasterizer(
                    nerf_cameras=cameras_to_use,
                    camera_indices=cam_idx,
                    bg_color=1. * torch.Tensor([1.0, 1.0, 1.0]).to(tetgs_refined.device),
                    sh_deg=self.sh_levels - 1,
                    compute_color_in_rasterizer=False,
                    compute_covariance_in_rasterizer=True
                ).nan_to_num().clamp(min=0, max=1)
                tetgs_image = (tetgs_image * 255.0).cpu().numpy()
                tetgs_image = cv2.cvtColor(tetgs_image[..., :3], cv2.COLOR_RGB2BGR)
                cv2.imwrite(validate_output_folder + "/frame%04d.png" % cam_idx, tetgs_image) 