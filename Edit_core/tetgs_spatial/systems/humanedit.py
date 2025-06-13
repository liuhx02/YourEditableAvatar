import numpy as np
from PIL import Image
import random
from dataclasses import dataclass, field
import torch
from tetgs_spatial.utils.perceptual import PerceptualLoss
import tetgs_spatial
from tetgs_spatial.models.exporters.base import Exporter, ExporterOutput
from tetgs_spatial.systems.base import BaseLift3DSystem
from tetgs_spatial.systems.humaninit import get_prev_geometry_ckpt
from tetgs_spatial.systems.utils import parse_optimizer, parse_scheduler
from tetgs_spatial.utils.typing import *


def silhouette_loss(alpha, gt_mask, loss_mask=None, l2_weight=0.01):
    sil_l2loss = (gt_mask - alpha) ** 2
    if loss_mask is not None:
        sil_l2loss = sil_l2loss * loss_mask
    return sil_l2loss.sum() * l2_weight 

def get_input_mask(mask_path, device):
    input_mask = np.array(Image.open(mask_path))
    input_mask = input_mask.astype(np.float32) / 255.0  # (h, w)
    input_mask = torch.from_numpy(input_mask)
    input_mask = input_mask[:, :, 0].to(device).unsqueeze(-1)
    
def get_input_normal(normal_path, device):
    input_normal = np.array(Image.open(normal_path))
    input_normal = input_normal.astype(np.float32) / 255.0   # (0, 1)
    input_normal = torch.from_numpy(input_normal)
    input_normal = input_normal[:, :, :3].to(device)


@tetgs_spatial.register("humanedit-system")
class HumanEdit(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        texture: bool = False
        start_sdf_loss_step: int = -1
        sub_step: int = 3000
        # additional input for image-guided editing
        use_additional_input: bool = False
        additional_input_path_mask: str = ""
        additional_input_path_normal: str = ""
        additional_input_path_mask_back: str = ""
        additional_input_path_normal_back: str = ""   
    cfg: Config
    
    def configure(self):
        # create geometry, renderer
        super().configure()
        self.perceptual_loss = PerceptualLoss().to(self.device)
        self.frames = []
        self.transforms = {
                "camera_model": "OPENCV",
                "orientation_override": "none",
            }
        self.keep_info = None
        self.grid_outputs = None
        self.new_sdf = None
        self.control_sdf = None
        
    def configure_optimizers(self):
        optim = parse_optimizer(self.cfg.optimizer, self)
        ret = {
            "optimizer": optim,
        }
        if self.cfg.scheduler is not None:
            ret.update(
                {
                    "lr_scheduler": parse_scheduler(self.cfg.scheduler, optim),
                }
            )
        return ret    
        
    def forward(self, batch: Dict[str, Any], subdivision=True, flag=0, update_new_sdf=None, control_sdf=None) -> Dict[str, Any]:
        if not self.cfg.shape_init:
            render_out = self.renderer(
                **batch,
                keep_info=self.keep_info,
                grid_outputs=self.grid_outputs,
                subdivision=subdivision,
                flag=flag,
                new_sdf=update_new_sdf,
                control_sdf=control_sdf
            )
        else:
            render_out = self.renderer(
                **batch,
                keep_info=None,
                grid_outputs=None,
                subdivision=True,
                flag=flag,
                new_sdf=update_new_sdf,
                control_sdf=control_sdf
            )
        return {
            **render_out,
        }
        
    def on_fit_start(self) -> None:
        super().on_fit_start()
        
        self.prev_geometry = get_prev_geometry_ckpt(self.cfg.prev_checkpoint)
        '''only override self.geometry.prev_geometry'''
        self.geometry.prev_geometry = self.prev_geometry

        self.renderer = tetgs_spatial.find(self.cfg.renderer_type)(
            self.cfg.renderer,
            geometry=self.geometry
        )
        
        # only used in training
        # local & global prompt processor
        self.prompt_processor_local = tetgs_spatial.find(self.cfg.prompt_processor_local_type)(
            self.cfg.prompt_processor_local
        )
        self.prompt_processor_global = tetgs_spatial.find(self.cfg.prompt_processor_global_type)(
            self.cfg.prompt_processor_global
        )
        # humannorm guidance
        self.guidance_normal = None
        if len(self.cfg.guidance_type_normal) > 0:
            self.guidance_normal = tetgs_spatial.find(self.cfg.guidance_type_normal)(
                self.cfg.guidance_normal
            )

        # initialize SDF
        if self.cfg.shape_init:
            self.geometry.initialize_shape()
        else:
            # mask on mesh surface
            mask_info = np.load(self.cfg.mask_npy_path, allow_pickle=True).item()
            mask = mask_info['editing_mask_faces']
            self.mask = torch.from_numpy(mask).to(self.device)
            
            # prepare keep_info & grid_outputs
            part_outputs = self.geometry.part_isosurface_init(mask=self.mask)
            keep_info = part_outputs["keep_info"]
            grid_outputs = part_outputs["grid_outputs"]
            self.keep_info = keep_info
            self.grid_outputs = grid_outputs
                
        if self.cfg.use_additional_input:
            # front mask & normal
            self.input_mask_front = get_input_mask(self.cfg.additional_input_path_mask, self.device)
            self.input_normal_front = get_input_normal(self.cfg.additional_input_path_normal, self.device)
            # back mask & normal
            self.input_mask_back = get_input_mask(self.cfg.additional_input_path_mask_back, self.device)
            self.input_normal_back = get_input_normal(self.cfg.additional_input_path_normal_back, self.device)
                
    def training_step(self, batch, batch_idx):
        sub_step = self.cfg.sub_step
        loss = 0.0
        
        if self.true_global_step > sub_step:
            subdivision = True
        else:
            subdivision = False
        
        # randomly choose the front or back guidance view for image-guided editing
        flag = random.randint(0, 1)
        out = self(batch, subdivision, flag, self.new_sdf, self.control_sdf)
        self.new_sdf = out["update_new_sdf"]

        prompt_utils_local = self.prompt_processor_local()
        prompt_utils_global = self.prompt_processor_global()
         
        # ADD: sdf loss following HUMANNORM [CVPR 2024] for better shape control
        if self.true_global_step == self.cfg.start_sdf_loss_step:
            self.control_sdf = out["update_new_sdf"]

        guidance_out = {}
        weight_local = self.C(self.cfg.loss.lambda_sds)
        # choose randomly from local and global sds for faster training
        choose = random.choices([0, 1], weights=[weight_local, 1 - weight_local])[0]
        
        # local normal SDS loss
        if self.C(self.cfg.loss.lambda_sds) > 0 and choose == 0:
            guidance_inp = out["comp_normal"]
            guidance_out_local = self.guidance_normal(
                guidance_inp, prompt_utils_local, **batch, rgb_as_latents=False
            )
            guidance_out.update({
                "loss_sds": guidance_out_local["loss_sds"]
            })
        
        # global normal SDS loss
        if self.prompt_processor_global is not None and self.C(self.cfg.loss.lambda_sds_global) > 0 and choose == 1:
            guidance_inp = out["global_comp_normal"]
            guidance_out_global = self.guidance_normal(
                guidance_inp, prompt_utils_global, **batch, rgb_as_latents=False
            )
            guidance_out.update({
                "loss_sds_global": guidance_out_global["loss_sds"]
            })
        
        # Surface-aware SDF loss 
        if out['keep_sdf_loss'] is not None:
            guidance_out.update({"loss_sdf_recon": out['keep_sdf_loss']})
        # ADD: overall SDF loss following HUMANNORM [CVPR 2024] for better shape control
        if out['update_sdf_loss'] is not None:
            guidance_out.update({"loss_sdf": out['update_sdf_loss']})
        
        # normal consistency loss (can also be replace by mesh.laplacian(), which gets similar effect)
        if self.C(self.cfg.loss.lambda_normal_consistency) > 0:
            loss_normal_consistency = out["update_mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            if self.true_global_step < sub_step:
                loss += loss_normal_consistency * self.C(
                    self.cfg.loss.lambda_normal_consistency
                )
            else:
                loss += loss_normal_consistency * self.C(
                    self.cfg.loss.lambda_normal_consistency_sub
                )
            
        # additional inputs including mask & normal loss for image-guided editing
        if out["input_opacity"] is not None and self.cfg.use_additional_input:
            input_mask_unkeep = out["input_mask_unkeep"][0].float()
            # pred mask
            unkeep_opacity = input_mask_unkeep * out["input_opacity"][0].clamp(1.0e-5, 1.0 - 1.0e-5)
            # pred normal
            unkeep_normal = input_mask_unkeep * out["input_normal"][0] + 0.5 * (input_mask_unkeep == 0).float()

            if flag == 0:  # front side
                unkeep_normal_gt = input_mask_unkeep * self.input_normal_front.float()
            elif flag == 1:  # back side
                unkeep_normal_gt = input_mask_unkeep * self.input_normal_back.float()
            unkeep_normal_gt = unkeep_normal_gt + 0.5 * (input_mask_unkeep == 0).float()
            H = unkeep_normal.shape[0]
            W = unkeep_normal.shape[1]
            unkeep_normal = unkeep_normal.reshape(1, 3, H, W).contiguous()
            unkeep_normal_gt = unkeep_normal_gt.reshape(1, 3, H, W).contiguous()
            
            criterion = torch.nn.MSELoss(reduction='sum')
            # image-guided normal loss
            loss_normal = criterion(
                unkeep_normal,
                unkeep_normal_gt,
            )
            self.log("train/loss_normal", loss_normal)
            if self.true_global_step < sub_step:
                loss += loss_normal * self.C(self.cfg.loss.lambda_normal)
            else:
                loss += loss_normal * self.C(self.cfg.loss.lambda_normal_sub)
            
            # image-guided mask loss (only front view)
            if flag == 0:
                loss_mask = silhouette_loss(unkeep_opacity.reshape(1, 1, H, W), (input_mask_unkeep * self.input_mask.float()).reshape(1, 1, H, W), l2_weight=1., edge_weight=0.)
            elif flag == 1:
                loss_mask = 0.0
            self.log("train/loss_mask", loss_mask)
            loss += loss_mask * self.C(self.cfg.loss.lambda_mask)
            
            # save image-guided local normal
            if self.cfg.verbose and self.true_global_step % 250 == 0:
                self.save_image_grid(
                    f"train-normal/it{self.true_global_step}-local-normal.jpg",
                    [
                        {
                            "type": "rgb",
                            "img": out["input_normal"][0].reshape(H, W, 3),
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        },
                    ],
                    name="validation_step",
                    step=self.true_global_step,
                )
                
        # save global & local normal
        if self.cfg.verbose and self.true_global_step % 250 == 0:
            self.save_image_grid(
                f"train-normal/it{self.true_global_step}-global.jpg",
                [{
                    "type": "rgb",
                    "img": out["global_comp_normal"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                }],
                name="validation_step",
                step=self.true_global_step,
            )
            self.save_image_grid(
                f"train-normal/it{self.true_global_step}-local.jpg",
                [{
                    "type": "rgb",
                    "img": out["comp_normal"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                }],
                name="validation_step",
                step=self.true_global_step,
            )    
        
        # loss  
        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                # print(name, ": ", value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")]))
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
                
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
            
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        out = self(batch, update_new_sdf=self.new_sdf, control_sdf=self.control_sdf)
        if "global_comp_normal" in out:
            self.save_image_grid(
                f"it{self.true_global_step}-val-normal/{batch['index'][0]:05d}.jpg",
                (
                    [{
                        "type": "rgb",
                        "img": out["global_comp_normal"][0] + (1 - out["global_opacity"][0, :, :, :]),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }]
                    if "global_comp_normal" in out
                    else []
                ),
                name="validation_step",
                step=self.true_global_step,
            )
    
    def on_validation_epoch_end(self):
        pass   
    
    def test_step(self, batch, batch_idx):
        out = self(batch, update_new_sdf=self.new_sdf, control_sdf=self.control_sdf)
        # local
        self.save_image_grid(
            f"it{self.true_global_step}-test-normal/{batch['index'][0]}.png",
            (
                [{
                    "type": "rgb",
                    "img": out["comp_normal"][0] + (1 - out["opacity"][0, :, :, :]),
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                }]
                if "comp_normal" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )
        # global
        self.save_image_grid(
            f"it{self.true_global_step}-test-normal/{batch['index'][0]}-global.png",
            (
                [{
                    "type": "rgb",
                    "img": out["global_comp_normal"][0] + (1 - out["global_opacity"][0, :, :, :]),
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                }]
                if "global_comp_normal" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )
        
    def on_test_epoch_end(self):
        pass
    
    def on_predict_start(self) -> None:
        self.prev_geometry = get_prev_geometry_ckpt(self.cfg.prev_checkpoint)
        '''only override self.geometry.prev_geometry'''
        self.geometry.prev_geometry = self.prev_geometry
        # export edited geometry
        self.exporter_part: Exporter = tetgs_spatial.find(self.cfg.exporter_part_type)(
            self.cfg.exporter_part,
            geometry=self.geometry,
            mask_npy_path=self.cfg.mask_npy_path,
        )
        
    def on_predict_epoch_end(self) -> None:
        exporter_output: List[ExporterOutput] = self.exporter_part()
        for out in exporter_output:
            save_func_name = f"save_{out.save_type}"
            if not hasattr(self, save_func_name):
                raise ValueError(f"{save_func_name} not supported by the SaverMixin")
            save_func = getattr(self, save_func_name)
            save_func(f"{out.save_name}", **out.params) 