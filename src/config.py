from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal, Optional


@dataclass(frozen=True)
class PainfaceReaderConfig:
    root: str = "/home/voigt/data/Datensatz_MSE_Hauptstudie_Loy"
    view: Literal["frontal", "lateral"] = "frontal"

    @property
    def videos(self) -> str:
        return str(Path(self.root) / "videos_all_full-length_n168")

    @property
    def annotations(self) -> str:
        return str(Path(self.root) / "au_annotations_n40")

    @property
    def crops_dir(self) -> str:
        return str(Path(self.root) / "crops")


@dataclass(frozen=True)
class DISFAConfig:
    root: str
    camera: Literal["left", "right"] = "left"

    @property
    def videos(self) -> str:                     # Video_LeftCamera/ | Video_RightCamera/
        return str(Path(self.root) / f"Video_{self.camera.capitalize()}Camera")

    @property
    def labels(self) -> str:                     # ActionUnit_Labels/SN0xx/SN0xx_auN.txt
        return str(Path(self.root) / "ActionUnit_Labels")

    @property
    def crops_dir(self) -> str:
        return str(Path(self.root) / "crops")


@dataclass(frozen=True)
class BP40DConfig:
    root: str

    @property
    def images(self) -> str:                     # <subject>/<task>/<frame>.jpg live under root
        return str(Path(self.root))

    @property
    def au_coding(self) -> str:                  # AUCoding/<subject>_<task>.csv
        return str(Path(self.root) / "AUCoding")

    @property
    def crops_dir(self) -> str:
        return str(Path(self.root) / "crops")


@dataclass
class AUDetectionConfig:
    task: Literal["painfacereader", "disfa", "bp4d"] = "painfacereader"
    # datasets
    painfacereader: PainfaceReaderConfig = PainfaceReaderConfig()
    disfa: Optional[DISFAConfig] = None
    bp4d: Optional[BP40DConfig] = None

    # general
    img_size: int = 224
    crop_margin: float = 0.15
    device: Literal["cpu", "cuda", "mps"] = "cuda"
    window_len: int = 1
    window_mode: str = "causal"
    unit_type: str = "frame"

    # AU selection / folds
    aus: Optional[list[str]] = None            # explicit AU set; None -> filter_aus picks
    min_subjects: int = 5
    min_carrier_frames: int = 25
    min_prevalence: float = 0.02
    folds_iter: int = 5_000
    folds_n_restart: int = 640
    folds_t0: float = 0.05
    folds_tmin: float = 1e-4

    # model
    model: str = "densenet121"

    # training
    n_folds: int = 5
    n_steps: int = 8_000
    eval_every: int = 1_500
    optimizer: str = "adam"
    batch_size: int = 64
    lr: float = 3e-4
    n_workers: int = 8

    # imbalance handling
    sampler: Literal["neutral_active", "none", "block"] = "neutral_active"
    block_size: int = 8                        # for sampler == "block"
    pos_weight_clip: float = 50.0              # BCE pos_weight clipped to [1, this]; <=1 disables

    @property
    def dataset(self):
        """The dataset config for the active task (painfacereader/disfa/bp4d)."""
        cfgs = {"painfacereader": self.painfacereader, "disfa": self.disfa, "bp4d": self.bp4d}
        d = cfgs[self.task]
        if d is None:
            raise ValueError(f"config.{self.task} is None; set it to run task={self.task!r}")
        return d

    @property
    def mode(self) -> str:
        if self.model in ["resnet50", "densenet121", "mae_face"]:
            return "frame"
        if self.model.endswith("_tcn") or self.model in ["v-jepa"]:
            return "video"
        raise ValueError(f"Model: {self.model} not recognized.")

    # -- yaml -------------------------------------------------------------- #
    @classmethod
    def from_yaml(cls, path) -> "AUDetectionConfig":
        """Build a config from a yaml file. Values present in the file override the
        class defaults; anything absent keeps its default."""
        import yaml
        data = yaml.safe_load(Path(path).read_text()) or {}
        nested = {"painfacereader": PainfaceReaderConfig,
                  "disfa": DISFAConfig, "bp4d": BP40DConfig}
        kwargs = {}
        for key, value in data.items():
            if key in nested and value is not None:
                kwargs[key] = nested[key](**value)
            else:
                kwargs[key] = value
        valid = {f.name for f in fields(cls)}
        unknown = set(kwargs) - valid
        if unknown:
            print(f"[config] ignoring unknown yaml keys: {sorted(unknown)}")
            kwargs = {k: v for k, v in kwargs.items() if k in valid}
        return cls(**kwargs)

    def to_yaml(self, path) -> None:
        """Dump the full config (nested dataclasses included) to a yaml file."""
        import yaml
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))

