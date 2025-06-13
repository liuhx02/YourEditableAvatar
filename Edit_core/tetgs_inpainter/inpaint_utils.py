import os
import cv2
import copy
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from lang_sam import LangSAM
from tetgs_scene.tetgs_edit_2d import EditTetGS
from tetgs_inpainter.mask_mesh_0822 import TexturedMeshModel
from utils.general_utils import PILtoTorch
import sys
sys.path.append("../")
negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

def normal_based_inpaint(
    pipe,
    mask_path,
    tetgs_image_path,
    normal_img,  # cv2 format, RGB
    prompt,
    seed,
    dilate_kernel=30
):
    prompt = prompt + " realistic, best quality, highres, 8k, real picture"
    
    mask = cv2.imread(mask_path)
    height, width, _  = mask.shape
    ratio = np.sqrt(1024. * 1024. / (width * height))
    W, H = int(width * ratio) // 8 * 8, int(height * ratio) // 8 * 8
    # process mask
    mask = cv2.erode(mask, np.ones((2, 2), np.uint8))
    # mask_blur
    blur_dilate_kernel = dilate_kernel  # 35
    mask_blur = cv2.GaussianBlur(cv2.dilate(mask, np.ones((blur_dilate_kernel, blur_dilate_kernel), np.uint8)).astype(np.float32), (21, 21), 0)
    mask_blur = cv2.resize(mask_blur, (W, H))
    # mask_inpaint
    mask = cv2.dilate(mask, np.ones((dilate_kernel, dilate_kernel), np.uint8))
    mask = cv2.resize(mask, (W, H))
    
    # process ori_img and normal_img
    original_img = cv2.imread(tetgs_image_path)
    original_img = cv2.resize(original_img, (W, H))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    normal_img = cv2.resize(normal_img, (W, H))

    controlnet_img = copy.deepcopy(original_img)
    controlnet_img[mask.squeeze() > 0.0] = 0
    controlnet_img = Image.fromarray(controlnet_img)
    original_img = Image.fromarray(original_img)
    mask = Image.fromarray(mask)
    normal_img = Image.fromarray(normal_img)
    
    width, height = W, H
    generator = torch.Generator('cuda').manual_seed(seed)
    # 0-openpose, 1-depth,2-hed/pidi/scribble/ted, 3-canny/lineart/anime_lineart/mlsd, 4-normal, 5-segment, 6-tile, 7-repaint
    images = pipe(
        prompt=[prompt]*1,
        image=original_img,
        mask_image=mask,
        strength=0.99,
        controlnet_conditioning_scale=1.0,
        control_image_list_1=[0, 0, 0, 0, normal_img, 0, 0, 0], 
        control_image_list_2=[0, 0, 0, 0, 0, 0, 0, controlnet_img],
        negative_prompt=[negative_prompt]*1,
        generator=generator,
        width=width, 
        height=height,
        num_inference_steps=30,
        union_control=True,
        union_control_type_1=torch.Tensor([0, 0, 0, 0, 1, 0, 0, 0]),
        union_control_type_2=torch.Tensor([0, 0, 0, 0, 0, 0, 0, 1]),
        crops_coords_top_left=(0, 0),
        target_size=(width, height),
        original_size=(width * 2, height * 2),
    ).images
    return images[0], mask_blur


def prepare_fb_inputs(
    mesh_model: TexturedMeshModel, 
    tetgs: EditTetGS, 
    verts_mask, mesh_sample_cameras, validate_mask_folder, keep_vertices_num, 
    cameras_to_use, bg_tensor, sh_deg
):
    # prepare front and back renderings&masks
    save_mask1, save_mask2 = None, None
    ori_image_path1, ori_image_path2 = "", ""
    normal1, normal2 = None, None
    for cam_idx in range(2):
        mask_outputs = mesh_model.prepare_mask_normal_for_inpainting(
            verts_mask=verts_mask,
            cam_idx=cam_idx,
            sample_cameras=mesh_sample_cameras,
            validate_mask_folder=validate_mask_folder,
            keep_vertices_num=keep_vertices_num
        )
        mask_aa = mask_outputs["mask_aa_save"]
        gb_normal = mask_outputs["gb_normal"]
        initial_outputs = tetgs.render_image_gaussian_rasterizer(
            nerf_cameras=cameras_to_use,
            camera_indices=cam_idx,
            verbose=False,
            bg_color=bg_tensor,
            sh_deg=sh_deg-1,
            sh_rotations=None,
            compute_color_in_rasterizer=False,
            compute_covariance_in_rasterizer=True, 
            return_2d_radii=False,
            quaternions=None,
        )
        tetgs_image = initial_outputs.detach().clone().nan_to_num().clamp(min=0, max=1)
        tetgs_image = (tetgs_image * 255.0).cpu().numpy()
        tetgs_image = cv2.cvtColor(tetgs_image[..., :3], cv2.COLOR_RGB2BGR)
        tetgs_image_path = os.path.join(validate_mask_folder, f'{cam_idx}_render.png')
        cv2.imwrite(tetgs_image_path, tetgs_image) 
        if cam_idx == 0:
            save_mask1 = mask_aa
            ori_image_path1 = tetgs_image_path
            normal1 = gb_normal
        else:
            save_mask2 = mask_aa
            ori_image_path2 = tetgs_image_path
            normal2 = gb_normal
    return save_mask1, save_mask2, ori_image_path1, ori_image_path2, normal1, normal2


def prepare_fb_image_guidance(
    pipe, 
    mask_front, mask_back,
    ori_image_path1, ori_image_path2,
    normal_front, normal_back,
    prompt, seed,
    dilate_kernel=25
):
    generator = torch.Generator('cuda').manual_seed(seed)
    prompt = prompt + " realistic, best quality, highres, 8k, real picture"
    # read inputs
    img_front, img_back = cv2.imread(ori_image_path1), cv2.imread(ori_image_path2)
    height, width, _ = mask_front.shape
    ratio = np.sqrt(1024. * 1024. / (width * height))
    W, H = int(width * ratio) // 8 * 8, int(height * ratio) // 8 * 8
    img_front, img_back = cv2.cvtColor(cv2.resize(img_front, (W,H)), cv2.COLOR_BGR2RGB), cv2.cvtColor(cv2.resize(img_back, (W,H)), cv2.COLOR_BGR2RGB)
    mask_front = cv2.resize(cv2.dilate(mask_front, np.ones((dilate_kernel, dilate_kernel), np.uint8)), (W,H))
    mask_back = cv2.resize(cv2.dilate(mask_back, np.ones((dilate_kernel, dilate_kernel), np.uint8)), (W,H))
    normal_front, normal_back = cv2.resize(normal_front, (W,H)), cv2.resize(normal_back, (W,H))
    # concat front and back view
    img_front, img_back = Image.fromarray(img_front[:, 256:768, :]), Image.fromarray(img_back[:, 256:768, :])
    mask_front, mask_back = Image.fromarray(mask_front[:, 256:768]), Image.fromarray(mask_back[:, 256:768])
    normal_front, normal_back = Image.fromarray(normal_front[:, 256:768, :]), Image.fromarray(normal_back[:, 256:768, :])
    concat_img, concat_mask, concat_normal = Image.new('RGB', (1024, 1024)), Image.new('RGB', (1024, 1024)), Image.new('RGB', (1024, 1024))
    concat_img.paste(img_front, (0, 0)), concat_img.paste(img_back, (512, 0))
    concat_mask.paste(mask_front, (0, 0)), concat_mask.paste(mask_back, (512, 0))
    concat_normal.paste(normal_front, (0, 0)), concat_normal.paste(normal_back, (512, 0))
    controlnet_img = copy.deepcopy(np.asarray(concat_img))
    controlnet_img[np.asarray(concat_mask) > 0.0] = 0
    controlnet_img = Image.fromarray(controlnet_img)
    
    # inpaint front and back view
    images = pipe(
        prompt=[prompt]*1,
        image=concat_img,
        mask_image=concat_mask,
        strength=0.99,
        controlnet_conditioning_scale=1.0,
        control_image_list_1=[0, 0, 0, 0, concat_normal, 0, 0, 0], 
        control_image_list_2=[0, 0, 0, 0, 0, 0, 0, controlnet_img],
        negative_prompt=[negative_prompt]*1,
        generator=generator,
        width=W, height=H,
        num_inference_steps=30,
        union_control=True,
        union_control_type_1=torch.Tensor([0, 0, 0, 0, 1, 0, 0, 0]),
        union_control_type_2=torch.Tensor([0, 0, 0, 0, 0, 0, 0, 1]),
        crops_coords_top_left=(0, 0),
        target_size=(W, H),
        original_size=(W * 2, H * 2),
    ).images
    (filepath, _) = os.path.split(ori_image_path1)  
    save_path_front, save_path_back = os.path.join(filepath, "front.png"), os.path.join(filepath, "back.png")
    front_inpainting, back_inpainting = np.ones((1024, 1024, 3)) * 255, np.ones((1024, 1024, 3)) * 255
    front_inpainting[:, 256:768, :] = np.asarray(images[0])[:, 0:512, :]
    back_inpainting[:, 256:768, :] = np.asarray(images[0])[:, 512:1024, :]
    front_inpainting, back_inpainting = Image.fromarray(np.uint8(front_inpainting)), Image.fromarray(np.uint8(back_inpainting))
    front_inpainting.save(save_path_front)
    back_inpainting.save(save_path_back)


def prepare_image_guidance(tetgs: EditTetGS, cam_idx, image_folder, initial_outputs, mask_bb, device, downscale_resolution_factor=1):
    if cam_idx == 0:
        image_path = os.path.join(image_folder, "front.png")
    else:
        image_path = os.path.join(image_folder, "back.png")
    image = Image.open(image_path)
    orig_w, orig_h = tetgs.image_height, tetgs.image_width
    resolution = round(orig_w/(downscale_resolution_factor)), round(orig_h/(downscale_resolution_factor))
    resized_image_rgb = PILtoTorch(image, resolution)
    gt_image = resized_image_rgb[:3, ...]
    gt_image = gt_image.clamp(0.0, 1.0).to(device)
    gt_image = gt_image.permute(1, 2, 0)
    gt_rgb = gt_image.view(-1, tetgs.image_height, tetgs.image_width, 3)
    gt_rgb = gt_rgb.transpose(-1, -2).transpose(-2, -3)
    
    initial_rgb = initial_outputs.view(-1, tetgs.image_height, tetgs.image_width, 3)
    initial_rgb = initial_rgb.transpose(-1, -2).transpose(-2, -3)
    
    # alleviate edge misalignment
    mask_foreground, _, _, _ = LangSAM("vit_h").predict(image.resize(resolution), "person")
    mask_foreground = mask_foreground.transpose(-2, -3).transpose(-1, -2).to(device)
    mask_foreground[mask_foreground < 1.0] = 0.
    mask_bb = torch.logical_and(mask_bb, mask_foreground)
    
    mask_bb = mask_bb.float()
    kernel_size = 15
    mask_bb = F.max_pool2d(mask_bb, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    mask_bb = mask_bb.view(-1, tetgs.image_height, tetgs.image_width, 1).repeat(1, 1, 1, 3)
    mask_bb = mask_bb.transpose(-1, -2).transpose(-2, -3)
    gt_rgb = gt_rgb * mask_bb + initial_rgb * (1 - mask_bb)
    gt_rgb = gt_rgb.detach().clone()
    return gt_rgb


def prepare_inpainting_image_guidance(
    tetgs: EditTetGS,
    cam_idx,
    save_folder,
    initial_outputs,
    inpainting_img,  # Image
    mask_blur,
    mask_proj,
    device,
    downscale_resolution_factor=1
):
    # resize mask_blur
    W, H = inpainting_img.size
    # get original_img
    original_img = initial_outputs.detach().clone()
    original_img = original_img.nan_to_num().clamp(min=0, max=1)
    original_img = (original_img * 255.0).cpu().numpy().astype(np.uint8)
    original_img = cv2.resize(original_img, (H, W))
    # get blurred inpaiting_img
    inpainting_img_blurred = np.array(mask_blur) / 255. * np.array(inpainting_img) + (1 - np.array(mask_blur) / 255.) * np.array(original_img)
    inpainting_img_blurred = Image.fromarray(np.uint8(inpainting_img_blurred))
    save_path = os.path.join(save_folder, f'{cam_idx}_full.png')
    inpainting_img_blurred.save(save_path)
    
    # resize inpainting_img
    orig_w, orig_h = tetgs.image_height, tetgs.image_width
    resolution = round(orig_w/(downscale_resolution_factor)), round(orig_h/(downscale_resolution_factor))
    resized_inpainting_img = PILtoTorch(inpainting_img_blurred, resolution)
    gt_image = resized_inpainting_img[:3, ...]
    gt_image = gt_image.clamp(0.0, 1.0).to(device)
    gt_image = gt_image.permute(1, 2, 0)
    gt_rgb = gt_image.view(-1, tetgs.image_height, tetgs.image_width, 3)
    gt_rgb = gt_rgb.transpose(-1, -2).transpose(-2, -3)
    
    initial_rgb = initial_outputs.view(-1, tetgs.image_height, tetgs.image_width, 3)
    initial_rgb = initial_rgb.transpose(-1, -2).transpose(-2, -3)
    mask_proj = mask_proj.view(-1, tetgs.image_height, tetgs.image_width, 1).repeat(1, 1, 1, 3)
    mask_proj = mask_proj.transpose(-1, -2).transpose(-2, -3)
    gt_rgb = gt_rgb * mask_proj + initial_rgb * (~mask_proj)
    gt_rgb = gt_rgb.detach().clone()
        
    return gt_rgb