
from dataclasses import dataclass
from typing import List
from PIL import Image

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from bmbw.optimizer import Optimizer

@dataclass
class Scorer():
    optimizer: "Optimizer"

    def batch_score(self, images: List[Image.Image]) -> List[float]:
        raise NotImplementedError()
