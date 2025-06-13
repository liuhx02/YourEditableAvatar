import os
import torch
from .cameras import CamerasWrapper, load_gs_cameras


class ModelParams(): 
    """Parameters of the Gaussian Splatting model.
    Largely inspired by the original implementation of the 3D Gaussian Splatting paper:
    https://github.com/graphdeco-inria/gaussian-splatting
    """
    def __init__(self):
        self.sh_degree = 3
        self.source_path = ""
        self.model_path = ""
        self.images = "images"
        self.resolution = -1
        self.white_background = False
        self.data_device = "cuda"
        self.eval = False
    
        
class PipelineParams():
    """Parameters of the Gaussian Splatting pipeline.
    Largely inspired by the original implementation of the 3D Gaussian Splatting paper:
    https://github.com/graphdeco-inria/gaussian-splatting
    """
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


class OptimizationParams():
    """Parameters of the Gaussian Splatting optimization.
    Largely inspired by the original implementation of the 3D Gaussian Splatting paper:
    https://github.com/graphdeco-inria/gaussian-splatting
    """
    def __init__(self):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002


class GaussianSplattingWrapper:
    """Class to wrap original Gaussian Splatting models and facilitates both usage and integration with PyTorch3D.
    """
    def __init__(self, 
                 source_path: str,
                 model_params: ModelParams=None,
                 pipeline_params: PipelineParams=None,
                 opt_params: OptimizationParams=None,
                 load_gt_images=True,
                 eval_split=False,
                 eval_split_interval=8,
                 background=[0., 0., 0.],
                 white_background=False,
                 remove_camera_indices=[],
                 ) -> None:
        self.source_path = source_path
        self.device = "cuda"
        
        if os.path.basename(source_path) in ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']:
            if len(remove_camera_indices) == 0.:
                remove_camera_indices = [i for i in range(0, 200)]
        
        if model_params is None:
            model_params = ModelParams()
        if pipeline_params is None:
            pipeline_params = PipelineParams()
        if opt_params is None:
            opt_params = OptimizationParams()
        
        self.model_params = model_params
        self.pipeline_params = pipeline_params
        self.opt_params = opt_params
        
        if white_background:
            background = [1., 1., 1.]
        
        self._C0 = 0.28209479177387814
        
        cam_list = load_gs_cameras(
            source_path=source_path,
            load_gt_images=load_gt_images,
            white_background=white_background,
            remove_indices=remove_camera_indices,
            )
        
        if eval_split:
            self.cam_list = []
            self.test_cam_list = []
            for i, cam in enumerate(cam_list):
                if i % eval_split_interval == 0:
                    self.test_cam_list.append(cam)
                else:
                    self.cam_list.append(cam)
            self.test_cameras = CamerasWrapper(self.test_cam_list)
        else:
            self.cam_list = cam_list
            self.test_cam_list = None
            self.test_cameras = None 
        self.training_cameras = CamerasWrapper(self.cam_list)

        self.background = torch.tensor(background, device=self.device, dtype=torch.float32)
    
    @property
    def image_height(self):
        return self.cam_list[0].image_height
    
    @property
    def image_width(self):
        return self.cam_list[0].image_width
    
    def get_gt_image(self, camera_indices:int, to_cuda=False):
        gt_image = self.cam_list[camera_indices].original_image
        if to_cuda:
            gt_image = gt_image.cuda()
        return gt_image.permute(1, 2, 0)
    
    def get_test_gt_image(self, camera_indices:int, to_cuda=False):
        gt_image = self.test_cam_list[camera_indices].original_image
        if to_cuda:
            gt_image = gt_image.cuda()
        return gt_image.permute(1, 2, 0)