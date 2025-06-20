import os
import argparse
from omegaconf import OmegaConf
from utils.general_utils import str2bool
from tetgs_texture.refine_3dgs import TetGS_Refine
from tetgs_texture.paint_2dgs import TetGS_Inpaint

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Script to refine a SuGaR model.')
    parser.add_argument('-s', '--scene_path',
                        type=str, 
                        help='path to the scene data to use.')  
    parser.add_argument('-m', '--mesh_path', 
                        type=str, 
                        help='Path to the extracted mesh file to use for refinement.')  
    parser.add_argument('-o', '--output_dir',
                        type=str, default=None, 
                        help='path to the output directory.')  
    # iteration to refine
    parser.add_argument('-f', '--refinement_iterations', type=int, default=4_000, 
                        help='Number of refinement iterations.') 
    parser.add_argument('-b', '--bboxmin', type=str, default=None, 
                        help='Min coordinates to use for foreground.')  
    parser.add_argument('-B', '--bboxmax', type=str, default=None, 
                        help='Max coordinates to use for foreground.')
    parser.add_argument('--eval', type=str2bool, default=True, help='Use eval split.')
    parser.add_argument('--white_background', type=str2bool, default=False, help='Use a white background instead of black.')
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')
    # inpainting
    parser.add_argument('--prompt', type=str, default=None, help='inpainting prompt')
    parser.add_argument('--seed', type=int, default=None, help='inpainting seed')
    parser.add_argument('--sample_type', type=str, default='full', help='inpainting sampling type')
    # refinement
    parser.add_argument('--upscale_to_2048', type=str2bool, default=False,
                        help='Whether to upscale the texture to 2048x2048 for texture enhancement.')
    
    args = parser.parse_args()
    comm_cfg = OmegaConf.load("../comm_config.yaml")
    
    # Call function
    # 1. inpaint stage
    trainer_tetgs_inpaint = TetGS_Inpaint(args, comm_cfg)
    trainer_tetgs_inpaint.inpaint_training(comm_cfg)
    
    # 2. refine stage
    # 2.1 prepare guidance image
    trainer_tetgs_inpaint.prepare_refine_guidance(comm_cfg)
    
    # 2.2 refine tetgs
    trainer_tetgs_refine = TetGS_Refine(args,  comm_cfg)
    trainer_tetgs_refine.refined_editing(comm_cfg)
    trainer_tetgs_refine.validate(comm_cfg)
    
'''

python train_edit_texture.py \
    -s ../data/model_man \
    -m ./outputs/model_man_seg/stage1-geometry-edit/save/edit_mesh.npy \
    -o ./outputs/model_man_seg/ \
    --white_background True \
    --refinement_iterations 2000 \
    --prompt "A photo of a man wearing a classic brown biker leather jacket, full body" \
    --sample_type "upper" \
    --seed -1 \
    --gpu 3
'''