
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Type, Union
import yaml
import dacite
from enum import Enum

from .scorers.huggingface import CafeAesthetic, CafeStyle, CafeWaifu
from .scorers.sd_chat import SacLogosAva1


class Scorer(Enum):
    CafeAesthetic = CafeAesthetic
    CafeStyle = CafeStyle
    CafeWaifu = CafeWaifu
    SacLogosAva1 = SacLogosAva1

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str) and value in cls.__members__:
            return cls[value]
        return None


@dataclass
class Prompt():
    prompt: str
    negative_prompt: str = ""
    seed: Optional[int] = None


@dataclass
class Config():
    clip_model: str = "ViT-L/14"
    scheduler: str = "EulerAncestralDiscreteScheduler"
    scorers: List[str] = field(default_factory=list)
    prompts: List[Prompt] = field(default_factory=list)
    modules: List[Union[
        Literal["unet"],
        Literal["text_encoder"],
        Literal["vae"],
    ]] = field(default_factory=list)

    num_inference_steps: int = 20
    guidance_scale: int = 7
    width: int = 512
    height: int = 512

    @classmethod
    def from_dict(cls, dict):
        return dacite.from_dict(data_class=cls, data=dict, config=dacite.Config(strict=True, check_types=True))

    @classmethod
    def from_file(cls, file: str):
        with open(file, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))
