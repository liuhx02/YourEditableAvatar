from dataclasses import dataclass, field
import os
from tetgs_spatial.utils.perceptual import PerceptualLoss
import tetgs_spatial
from tetgs_spatial.models.exporters.base import Exporter, ExporterOutput
from tetgs_spatial.systems.base import BaseLift3DSystem, BaseSystem
from tetgs_spatial.systems.utils import parse_optimizer, parse_scheduler
from tetgs_spatial.utils.typing import *


def silhouette_loss(alpha, gt_mask, edt=None, loss_mask=None, kernel_size=7, edt_power=0.25, l2_weight=0.01, edge_weight=0.01):
    sil_l2loss = (gt_mask - alpha) ** 2
    if loss_mask is not None:
        sil_l2loss = sil_l2loss * loss_mask

    return sil_l2loss.sum() * l2_weight 

def get_prev_geometry_ckpt(prev_ckpt):
    from tetgs_spatial.utils.config import load_config, parse_structured
    from tetgs_spatial.utils.misc import load_module_weights, get_device
    
    prev_cfg = load_config(
        os.path.join(
            os.path.dirname(prev_ckpt),
            "../configs/parsed.yaml",
        )
    )
    prev_system_cfg: HumanInit.Config = parse_structured(
        HumanInit.Config, prev_cfg.system
    )
    prev_geometry_cfg = prev_system_cfg.geometry
    prev_geometry = tetgs_spatial.find(prev_system_cfg.geometry_type)(
        prev_geometry_cfg
    )
    state_dict, epoch, global_step = load_module_weights(
        prev_ckpt,
        module_name="geometry",
        map_location="cpu",
    )
    prev_geometry.load_state_dict(state_dict, strict=False)
    for param in prev_geometry.parameters():
        param.requires_grad = False
    # restore step-dependent states
    prev_geometry.do_update_step(epoch, global_step, on_load_weights=True)
    prev_geometry = prev_geometry.to(get_device())
    
    return prev_geometry


@tetgs_spatial.register("humaninit-system")
class HumanInit(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        texture: bool = False
        start_sdf_loss_step: int = -1
        
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
        
    def forward(self, batch: Dict[str, Any], subdivision=True, flag=0) -> Dict[str, Any]:
        if self.cfg.shape_init:
            render_out = self.renderer(
                **batch,
                keep_info=None,
                grid_outputs=None,
                subdivision=True,
                flag=flag,
            )
        else:
            raise NotImplementedError
        return {
            **render_out,
        }  
        
    def on_fit_start(self) -> None:
        super().on_fit_start()
        # initialize SDF
        if self.cfg.shape_init:
            self.geometry.initialize_shape()
                       
    def training_step(self, batch, batch_idx):
        pass
    
    def validation_step(self, batch, batch_idx):
        pass
    
    def on_validation_epoch_end(self):
        pass 
    
    def on_predict_start(self) -> None:
        self.prev_geometry = get_prev_geometry_ckpt(self.cfg.prev_checkpoint)
        '''only override self.geometry.prev_geometry'''
        self.geometry.prev_geometry = self.prev_geometry 
        
        self.exporter_part: Exporter = tetgs_spatial.find(self.cfg.exporter_part_type)(
            self.cfg.exporter_part,
            geometry=self.geometry,
        )
        
    def on_predict_epoch_end(self) -> None:
        exporter_output: List[ExporterOutput] = self.exporter_part()
        for out in exporter_output:
            save_func_name = f"save_{out.save_type}"
            if not hasattr(self, save_func_name):
                raise ValueError(f"{save_func_name} not supported by the SaverMixin")
            save_func = getattr(self, save_func_name)
            save_func(f"it{self.true_global_step}-export/{out.save_name}", **out.params)   