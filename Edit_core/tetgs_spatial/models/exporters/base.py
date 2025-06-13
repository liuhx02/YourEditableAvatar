from dataclasses import dataclass
import tetgs_spatial
from tetgs_spatial.models.geometry.base import BaseImplicitGeometry
from tetgs_spatial.utils.base import BaseObject
from tetgs_spatial.utils.typing import *


@dataclass
class ExporterOutput:
    save_name: str
    save_type: str
    params: Dict[str, Any]


class Exporter(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        save_video: bool = False

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry
    ) -> None:
        @dataclass
        class SubModules:
            geometry: BaseImplicitGeometry

        self.sub_modules = SubModules(geometry)

    @property
    def geometry(self) -> BaseImplicitGeometry:
        return self.sub_modules.geometry

    def __call__(self, *args, **kwargs) -> List[ExporterOutput]:
        raise NotImplementedError


@tetgs_spatial.register("dummy-exporter")
class DummyExporter(Exporter):
    def __call__(self, *args, **kwargs) -> List[ExporterOutput]:
        # DummyExporter does not export anything
        return []
