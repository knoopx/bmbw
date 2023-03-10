from dataclasses import dataclass
from functools import cached_property
import random
import torch
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionPipeline
from diffusers.utils.dynamic_modules_utils import get_class_from_dynamic_module
from diffusers.utils.import_utils import is_xformers_available

from bmbw.config import Config


def compatible_schedulers(pipeline):
    return {
        scheduler.__name__: scheduler.from_config(pipeline.scheduler.config) for scheduler in pipeline.scheduler.compatibles
    }


def get_scheduler(pipeline, name):
    schedulers = compatible_schedulers(pipeline)
    try:
        return schedulers[name]
    except KeyError:
        print(
            f"Scheduler {name} not available, available schedulers: {schedulers.keys()}")

    return EulerAncestralDiscreteScheduler


@dataclass
class Sampler():
    config: Config

    @cached_property
    def pipeline_class(self) -> StableDiffusionPipeline:
        return get_class_from_dynamic_module("lpw_stable_diffusion", module_file="pipeline.py")

    def configure_pipeline(self, pipeline: StableDiffusionPipeline):
        pipeline = self.pipeline_class(**pipeline.components)
        pipeline.scheduler = get_scheduler(pipeline, self.config.scheduler),

        pipeline.enable_vae_slicing()
        if is_xformers_available():
            pipeline.enable_xformers_memory_efficient_attention()
        else:
            pipeline.enable_memory_efficient_attention()

    def sample_with(self, pipeline: StableDiffusionPipeline):
        self.configure_pipeline(pipeline)

        seeds = [
            prompt.seed or random.randint(0, 2 ** 32 - 1)
            for prompt in self.config.prompts
        ]

        generator = [
            torch.Generator(device="cuda").manual_seed(seed)
            for seed in seeds
        ]

        prompts = [prompt.prompt for prompt in self.config.prompts]
        negative_prompts = [
            prompt.negative_prompt for prompt in self.config.prompts]

        return pipeline(prompt=prompts, negative_prompt=negative_prompts, generator=generator).images
