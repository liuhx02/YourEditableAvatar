import os
import sys
sys.path.append('..')
import cv2
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

def sdxl_refiner(
    pipe,
    ori_images,
    input_prompt,
    seed=None,
    upscale_to_2048=False
):
    if seed is None:
        seed = random.randint(0, 2147483647)
    generator = torch.Generator('cuda').manual_seed(seed)
    prompt = "8K, ultra-detailed full-body" + input_prompt + " realistic, best quality, highres, real picture"
    negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    refined_images = []
    
    for controlnet_img in tqdm(ori_images, desc="Processing images"):
        height, width, _  = controlnet_img.shape
        ratio = np.sqrt(1024. * 1024. / (width * height))
        # 3 * 3 upscale correspond to 16 * 3 multiply, 2 * 2 correspond to 16 * 2 multiply and so on.
        W, H = int(width * ratio) // 32 * 32, int(height * ratio) // 32 * 32
        controlnet_img = cv2.resize(controlnet_img, (W, H))
        controlnet_img = cv2.cvtColor(controlnet_img, cv2.COLOR_BGR2RGB)
        controlnet_img = Image.fromarray(np.uint8(controlnet_img))
        
        if upscale_to_2048:
            target_width = W // 2
            target_height = H // 2
        else:
            target_width = W
            target_height = H
        images = []
        crops_coords_list = [(0, 0), (0, width // 2), (height // 2, 0), (width // 2, height // 2), 0, 0, 0, 0, 0]
        num_iters = 1
        if upscale_to_2048:
            num_iters = 2
        for i in range(num_iters):  # 2
            for j in range(num_iters):  # 2
                left = j * target_width
                top = i * target_height
                right = left + target_width
                bottom = top + target_height
                cropped_image = controlnet_img.crop((left, top, right, bottom))
                cropped_image = cropped_image.resize((W, H))
                images.append(cropped_image)
        # 6 -- tile
        # 7 -- repaint
        result_images = []
        for sub_img, crops_coords in zip(images, crops_coords_list):
            new_width, new_height = W, H
            out = pipe(prompt=[prompt]*1,
                        image=sub_img, 
                        control_image_list=[0, 0, 0, 0, 0, 0, sub_img, 0],
                        negative_prompt=[negative_prompt]*1,
                        generator=generator,
                        width=new_width, 
                        height=new_height,
                        strength=0.4,  # 0.5
                        num_inference_steps=30,
                        crops_coords_top_left=(W, H),
                        target_size=(W, H),
                        original_size=(W * 2, H * 2),
                        union_control=True,
                        union_control_type=torch.Tensor([0, 0, 0, 0, 0, 0, 1, 0]),
                    )
            result_images.append(out.images[0])

        new_img = Image.new('RGB', (new_width, new_height))
        if upscale_to_2048:
            new_img = Image.new('RGB', (new_width * 2, new_height * 2))
        
        new_img.paste(result_images[0], (0, 0))  
        if upscale_to_2048:
            new_img.paste(result_images[1], (new_width, 0))
            new_img.paste(result_images[2], (0, new_height))
            new_img.paste(result_images[3], (new_width, new_height))

        refined_images.append(new_img)
    return refined_images