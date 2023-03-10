from dataclasses import dataclass
from functools import cached_property
from typing import List
from bmbw.scorers.scorer import Scorer
from PIL import Image
from transformers import pipeline


@dataclass
class HuggingFaceScorer(Scorer):
    model: str

    @cached_property
    def pipeline(self):
        return pipeline("image-classification", model=self.model, device=self.optimizer.device)

    def batch_score(self, images: List[Image.Image]) -> List[float]:
        return [
            result[0]["score"] - result[1]["score"]
            for result in self.pipeline(images)
        ]


class CafeWaifu(HuggingFaceScorer):
    def __init__(self, optimizer):
        super().__init__(optimizer, "cafeai/cafe_waifu")


class CafeAesthetic(HuggingFaceScorer):
    def __init__(self, optimizer):
        super().__init__(optimizer, "cafeai/cafe_aesthetic")


class CafeStyle(HuggingFaceScorer):
    def __init__(self, optimizer):
        super().__init__(optimizer, "cafeai/cafe_style")
