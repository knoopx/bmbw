from bayes_opt import BayesianOptimization, Events
from dataclasses import dataclass
from functools import cached_property
from numpy import mean
from stqdm import stqdm
from typing import Optional
import clip
import streamlit as st

from .logger import Logger
from .config import Config, Scorer
from .merger import Merger
from .sampler import Sampler
from .scorers.multi_scorer import MultiScorer


@dataclass
class Optimizer:
    model_a: str
    model_b: str
    config_path: str
    init_points: int
    n_iters: int
    device: str
    output_dir: Optional[str] = None

    def __post_init__(self):
        assert len(self.config.prompts), "no prompts provided"

    @cached_property
    def merger(self):
        return Merger(self.model_a, self.model_b, self.device)

    @cached_property
    def sampler(self):
        return Sampler(self.config)

    @cached_property
    def config(self):
        return Config.from_file(self.config_path)

    @cached_property
    def progress_bar(self):
        return stqdm(range(self.n_iters))

    @cached_property
    def scorer(self):
        return MultiScorer(self, [Scorer(scorer).value for scorer in self.config.scorers])

    @cached_property
    def clip(self):
        return clip.load(self.config.clip_model, device=self.device)

    @cached_property
    def logger(self):
        return Logger()

    def sd_target_function(self, **params):
        if self.progress_bar.n < self.n_iters:
            self.progress_bar.update(1)

        weights = [params[f"block_{i}"] for i in range(25)]
        base_alpha = params["base_alpha"]

        self.pipeline = self.merger.merge(
            weights,
            base_alpha,
        ).to(self.device)

        images = self.sampler.sample_with(self.pipeline)
        assert isinstance(images, list)

        scores = self.scorer.batch_score(images)
        score = mean(scores)
        st.title(f"Iteration {self.progress_bar.n}")
        st.subheader(f"Score: {score}")
        st.image(images, [str(score) for score in scores])

        return score

    def start_optimization(self) -> None:
        # TODO: what if we want to optimise only certain blocks?
        pbounds = {f"block_{i}": (0.0, 1.0) for i in range(25)}
        pbounds["base_alpha"] = (0.0, 1.0)

        # TODO: fork bayesian-optimisation and add LHS
        self.optimizer = BayesianOptimization(
            f=self.sd_target_function,
            pbounds=pbounds,
            random_state=1,
        )

        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, self.logger)

        self.optimizer.maximize(
            init_points=self.init_points,
            n_iter=self.n_iters,
        )

        for i, res in enumerate(self.optimizer.res):
            print(f"Iteration {i}: \n\t{res}")

        print(self.optimizer.max)
        if self.output_dir:
            self.pipeline.save_pretrained(self.output_dir)
