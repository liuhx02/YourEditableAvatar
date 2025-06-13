import os
from omegaconf import OmegaConf
import argparse
from utils.general_utils import str2bool
from tetgs_texture.refine import TetGS_Init
from mesh_localization import LocalMeshEditingModel

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
                        type=str, default="../outputs/000/", 
                        help='path to the output directory.')  
    # iteration to train
    parser.add_argument('-f', '--refinement_iterations', type=int, default=4_000, 
                        help='Number of refinement iterations.') 
    parser.add_argument('-b', '--bboxmin', type=str, default=None, 
                        help='Min coordinates to use for foreground.')  
    parser.add_argument('-B', '--bboxmax', type=str, default=None, 
                        help='Max coordinates to use for foreground.')
    parser.add_argument('--eval', type=str2bool, default=True, help='Use eval split.')
    parser.add_argument('--white_background', type=str2bool, default=False, help='Use a white background instead of black.')
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')
    # region to edit
    parser.add_argument('--seg_prompt', type=str, default='shirt')
    parser.add_argument('--seg_mesh_path', type=str, default=None)
    
    args = parser.parse_args()
    comm_cfg = OmegaConf.load("../comm_config.yaml")
    
    # init tetgs texture
    trainer_tetgs_init = TetGS_Init(args)
    trainer_tetgs_init.init_training(comm_cfg)
    cam_pack, validate_output_folder = trainer_tetgs_init.validate(comm_cfg, phi_list=[0,120,240], theta_list=[5,20,-10])
    # mesh surface localization
    localization_model = LocalMeshEditingModel(cam_pack, validate_output_folder, comm_cfg, args)
    localization_model.mesh_localization()