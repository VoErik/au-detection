from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass
class PainfaceReaderConfig:
    root: str
    view: Literal["frontal", "lateral"] = "frontal"

    @property
    def videos(self) -> str:
        return str(Path(self.root) / "videos_all_full-length_n168")

    @property
    def annotations(self) -> str:
        return str(Path(self.root) / "au_annotations_n40")

@dataclass 
class DISFAConfig:
    root: str


@dataclass 
class BP40DConfig:
    root: str

@dataclass
class AUDetectionConfig:
    # datasets
    painfacereader: PainfaceReaderConfig
    disfa: DISFAConfig
    bp40d: BP40DConfig

    # general
    img_size: int = 224

    # model
    model: str = "resnet50"

    # training
    n_folds: int = 5
    n_steps: int = 8_000
    optimizer: str = "adam"


    # eval

    @property
    def mode(self) -> str:
        if self.model in ["resnet50", "densenet121"]:
            return "frame"
        if self.model in ["v-jepa"]:
            return "video"
        raise ValueError(f"Model: {self.model} not recognized.")


