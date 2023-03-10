from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Type
from numpy import mean
from bmbw.scorers.scorer import Scorer
from PIL import Image


@dataclass
class MultiScorer(Scorer):
    scorer_classes: List[Type[Scorer]] = field(default_factory=list)

    @cached_property
    def scorers(self):
        return [scorer_class(self.optimizer) for scorer_class in self.scorer_classes]

    def batch_score(self, images: List[Image.Image]) -> List[float]:
        scores = [scorer.batch_score(images) for scorer in self.scorers]
        return [mean(score) for score in zip(*scores)]  # type: ignore
