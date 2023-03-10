from bmbw.scorers.scorer import Scorer
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from PIL import Image
from typing import List
import requests
import torch
import torch.nn as nn


# from https://github.com/grexzen/SD-Chad
class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


@dataclass
class SDChadScorer(Scorer):
    checkpoint_name: str

    @cached_property
    def checkpoint_path(self):
        path = Path("models", self.checkpoint_name)
        if not path.is_file():
            url = f"https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/{self.checkpoint_name}?raw=true"
            r = requests.get(url)
            with open(path, "wb") as f:
                print(f"saved into {self.checkpoint_path}")
                f.write(r.content)

        return path

    @cached_property
    def model(self):
        pt_state = torch.load(self.checkpoint_path,
                              map_location=self.optimizer.device)
        model = AestheticPredictor(768)
        model.load_state_dict(pt_state)
        model.to(self.optimizer.device)
        model.eval()

        return model

    def get_image_features(self, image: Image.Image) -> torch.Tensor:
        clip_model, clip_processor = self.optimizer.clip
        image = clip_processor(
            image).unsqueeze(0).to(self.optimizer.device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().detach().numpy()
        return image_features

    def score(self, image: Image.Image) -> float:
        image_features = self.get_image_features(image)
        score = self.model(torch.from_numpy(
            image_features).to(self.optimizer.device).float())
        return score.item()

    def batch_score(self, images: List[Image.Image]) -> List[float]:
        return [self.score(img) / 10 for img in images]


class SacLogosAva1(SDChadScorer):
    def __init__(self, optimizer):
        super().__init__(optimizer, "sac+logos+ava1-l14-linearMSE.pth")
