from dataclasses import dataclass
import torch
from tetgs_spatial.models.geometry.base import BaseImplicitGeometry
from tetgs_spatial.utils.base import BaseModule
from tetgs_spatial.utils.typing import *


class Renderer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        radius: float = 1.0

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry
    ) -> None:
        # keep references to submodules using namedtuple, avoid being registered as modules
        @dataclass
        class SubModules:
            geometry: BaseImplicitGeometry

        self.sub_modules = SubModules(geometry)

        # set up bounding box
        self.bbox: Float[Tensor, "2 3"]
        self.register_buffer(
            "bbox",
            torch.as_tensor(
                [
                    [-self.cfg.radius, -self.cfg.radius, -self.cfg.radius],
                    [self.cfg.radius, self.cfg.radius, self.cfg.radius],
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def geometry(self) -> BaseImplicitGeometry:
        return self.sub_modules.geometry

    def set_geometry(self, geometry: BaseImplicitGeometry) -> None:
        self.sub_modules.geometry = geometry