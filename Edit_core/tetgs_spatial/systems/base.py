import os
from dataclasses import dataclass, field
import pytorch_lightning as pl
import torch.nn.functional as F
import tetgs_spatial
from tetgs_spatial.models.exporters.base import Exporter, ExporterOutput
from tetgs_spatial.systems.utils import parse_optimizer, parse_scheduler
from tetgs_spatial.utils.base import Updateable, update_if_possible
from tetgs_spatial.utils.config import parse_structured
from tetgs_spatial.utils.misc import C, cleanup, get_device, load_module_weights
from tetgs_spatial.utils.saving import SaverMixin
from tetgs_spatial.utils.typing import *


class BaseSystem(pl.LightningModule, Updateable, SaverMixin):
    @dataclass
    class Config:
        loggers: dict = field(default_factory=dict)
        loss: dict = field(default_factory=dict)
        optimizer: dict = field(default_factory=dict)
        scheduler: Optional[dict] = None
        weights: Optional[str] = None
        weights_ignore_modules: Optional[List[str]] = None
        cleanup_after_validation_step: bool = False
        cleanup_after_test_step: bool = False
        prev_checkpoint: str = ""

    cfg: Config

    def __init__(self, cfg, resumed=False) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self._save_dir: Optional[str] = None
        self._resumed: bool = resumed
        self._resumed_eval: bool = False
        self._resumed_eval_status: dict = {"global_step": 0, "current_epoch": 0}
        if "loggers" in cfg:
            self.create_loggers(cfg.loggers)

        self.configure()
        if self.cfg.weights is not None:
            self.load_weights(self.cfg.weights, self.cfg.weights_ignore_modules)
        self.post_configure()

    def load_weights(self, weights: str, ignore_modules: Optional[List[str]] = None):
        state_dict, epoch, global_step = load_module_weights(
            weights, ignore_modules=ignore_modules, map_location="cpu"
        )
        self.load_state_dict(state_dict, strict=False)
        # restore step-dependent states
        self.do_update_step(epoch, global_step, on_load_weights=True)

    def set_resume_status(self, current_epoch: int, global_step: int):
        # restore correct epoch and global step in eval
        self._resumed_eval = True
        self._resumed_eval_status["current_epoch"] = current_epoch
        self._resumed_eval_status["global_step"] = global_step

    @property
    def resumed(self):
        # whether from resumed checkpoint
        return self._resumed

    @property
    def true_global_step(self):
        if self._resumed_eval:
            return self._resumed_eval_status["global_step"]
        else:
            return self.global_step

    @property
    def true_current_epoch(self):
        if self._resumed_eval:
            return self._resumed_eval_status["current_epoch"]
        else:
            return self.current_epoch

    def configure(self) -> None:
        pass

    def post_configure(self) -> None:
        """
        executed after weights are loaded
        """
        pass

    def C(self, value: Any) -> float:
        return C(value, self.true_current_epoch, self.true_global_step)

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

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        if self.cfg.cleanup_after_validation_step:
            # cleanup to save vram
            cleanup()

    def on_validation_epoch_end(self):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_test_batch_end(self, outputs, batch, batch_idx):
        if self.cfg.cleanup_after_test_step:
            # cleanup to save vram
            cleanup()

    def on_test_epoch_end(self):
        pass

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_predict_batch_end(self, outputs, batch, batch_idx):
        if self.cfg.cleanup_after_test_step:
            # cleanup to save vram
            cleanup()

    def on_predict_epoch_end(self):
        pass

    def preprocess_data(self, batch, stage):
        pass

    """
    Implementing on_after_batch_transfer of DataModule does the same.
    But on_after_batch_transfer does not support DP.
    """

    def on_train_batch_start(self, batch, batch_idx, unused=0):
        self.preprocess_data(batch, "train")
        self.dataset = self.trainer.train_dataloader.dataset
        update_if_possible(self.dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.preprocess_data(batch, "validation")
        self.dataset = self.trainer.val_dataloaders.dataset
        update_if_possible(self.dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.preprocess_data(batch, "test")
        self.dataset = self.trainer.test_dataloaders.dataset
        update_if_possible(self.dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.preprocess_data(batch, "predict")
        self.dataset = self.trainer.predict_dataloaders.dataset
        update_if_possible(self.dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        pass

    def on_before_optimizer_step(self, optimizer):
        pass


class BaseLift3DSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        geometry_type: str = "implicit-sdf"
        geometry: dict = field(default_factory=dict)
        geometry_convert_from: Optional[str] = None
        geometry_convert_inherit_texture: bool = False
        geometry_convert_override: dict = field(default_factory=dict)

        renderer_type: str = "part-nvdiff-rasterizer"
        renderer: dict = field(default_factory=dict)
        
        guidance_type_normal: str = ""
        guidance_normal: dict = field(default_factory=dict)
        
        prompt_processor_local_type: str = "stable-diffusion-prompt-processor"
        prompt_processor_local: dict = field(default_factory=dict)
        
        prompt_processor_global_type: str = "stable-diffusion-prompt-processor"
        prompt_processor_global: dict = field(default_factory=dict)

        # geometry export configurations, no need to specify in training
        exporter_part_type: str = "mesh-exporter-part"
        exporter_part: dict = field(default_factory=dict)

        mask_npy_path: str = ""
        shape_init: bool = True  # shape init or shape edit
        prev_checkpoint: str = ""  # initial checkpoint path
        verbose: bool = True

    cfg: Config

    def configure(self) -> None:
        if (
            self.cfg.geometry_convert_from  # from_coarse must be specified
            and not self.cfg.weights  # not initialized from coarse when weights are specified
            and not self.resumed  # not initialized from coarse when resumed from checkpoints
        ):
            tetgs_spatial.info("Initializing geometry from a given checkpoint ...")
            from tetgs_spatial.utils.config import load_config, parse_structured

            prev_cfg = load_config(
                os.path.join(
                    os.path.dirname(self.cfg.geometry_convert_from),
                    "../configs/parsed.yaml",
                )
            )  # TODO: hard-coded relative path
            prev_system_cfg: BaseLift3DSystem.Config = parse_structured(
                self.Config, prev_cfg.system
            )
            prev_geometry_cfg = prev_system_cfg.geometry
            prev_geometry_cfg.update(self.cfg.geometry_convert_override)
            prev_geometry = tetgs_spatial.find(prev_system_cfg.geometry_type)(
                prev_geometry_cfg
            )
            state_dict, epoch, global_step = load_module_weights(
                self.cfg.geometry_convert_from,
                module_name="geometry",
                map_location="cpu",
            )
            prev_geometry.load_state_dict(state_dict, strict=False)
            # restore step-dependent states
            prev_geometry.do_update_step(epoch, global_step, on_load_weights=True)
            # convert from coarse stage geometry
            prev_geometry = prev_geometry.to(get_device())
            self.geometry = tetgs_spatial.find(self.cfg.geometry_type).create_from(
                prev_geometry,
                self.cfg.geometry,
                copy_net=self.cfg.geometry_convert_inherit_texture,
            )
            del prev_geometry
            cleanup()
        else:
            self.geometry = tetgs_spatial.find(self.cfg.geometry_type)(self.cfg.geometry)

        self.renderer = tetgs_spatial.find(self.cfg.renderer_type)(
            self.cfg.renderer,
            geometry=self.geometry
        )

    def on_fit_start(self) -> None:
        if self._save_dir is not None:
            tetgs_spatial.info(f"Validation results will be saved to {self._save_dir}")
        else:
            tetgs_spatial.warn(
                f"Saving directory not set for the system, visualization results will not be saved"
            )

    def on_test_end(self) -> None:
        if self._save_dir is not None:
            tetgs_spatial.info(f"Test results saved to {self._save_dir}")

    def on_predict_start(self) -> None:
        self.exporter_part: Exporter = tetgs_spatial.find(self.cfg.exporter_part_type)(
            self.cfg.exporter_part,
            geometry=self.geometry,
            mask_npy_path=self.cfg.mask_npy_path,
        )
        
    def predict_step(self, batch, batch_idx):
        pass

    def on_predict_epoch_end(self) -> None:
        exporter_output: List[ExporterOutput] = self.exporter_part()
        for out in exporter_output:
            save_func_name = f"save_{out.save_type}"
            if not hasattr(self, save_func_name):
                raise ValueError(f"{save_func_name} not supported by the SaverMixin")
            save_func = getattr(self, save_func_name)
            save_func(f"it{self.true_global_step}-export/{out.save_name}", **out.params)  
        
    def on_predict_end(self) -> None:
        if self._save_dir is not None:
            tetgs_spatial.info(f"Export assets saved to {self._save_dir}")

    def guidance_evaluation_save(self, comp_rgb, guidance_eval_out):
        B, size = comp_rgb.shape[:2]
        resize = lambda x: F.interpolate(
            x.permute(0, 3, 1, 2), (size, size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        filename = f"it{self.true_global_step}-train.png"

        def merge12(x):
            return x.reshape(-1, *x.shape[2:])

        self.save_image_grid(
            filename,
            [{
                "type": "rgb",
                "img": merge12(comp_rgb),
                "kwargs": {"data_format": "HWC"},
            }]
            + (
                [{
                    "type": "rgb",
                    "img": merge12(resize(guidance_eval_out["imgs_noisy"])),
                    "kwargs": {"data_format": "HWC"},
                }]
            )
            + (
                [{
                    "type": "rgb",
                    "img": merge12(resize(guidance_eval_out["imgs_1step"])),
                    "kwargs": {"data_format": "HWC"},
                }]
            )
            + (
                [{
                    "type": "rgb",
                    "img": merge12(resize(guidance_eval_out["imgs_1orig"])),
                    "kwargs": {"data_format": "HWC"},
                }]
            )
            + (
                [{
                    "type": "rgb",
                    "img": merge12(resize(guidance_eval_out["imgs_final"])),
                    "kwargs": {"data_format": "HWC"},
                }]
            ),
            name="train_step",
            step=self.true_global_step,
            texts=guidance_eval_out["texts"],
        )