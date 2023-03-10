# adapted from
# bbc-mc/sdweb-merge-block-weighted-gui/scripts/mbw/merge_block_weighted.py

from dataclasses import dataclass
from functools import cached_property
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import load_pipeline_from_original_stable_diffusion_ckpt
from typing import List
import re
import torch

NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS

KEY_POSITION_IDS = ".".join(
    [
        "cond_stage_model",
        "transformer",
        "text_model",
        "embeddings",
        "position_ids",
    ]
)


def get_module_state_dict(pipeline, module_key, device=None, dtype=None):
    sd = getattr(pipeline, module_key).state_dict()
    for key in list(sd.keys()):
        if isinstance(sd[key], torch.Tensor):
            sd[key] = sd[key].to(device, dtype=dtype)
    return sd


def load_checkpoint(checkpoint_path, **kwargs):
    print(f"Loading checkpoint from {checkpoint_path}...")
    from_safetensors = checkpoint_path.lower().endswith(".safetensors")
    pipeline = load_pipeline_from_original_stable_diffusion_ckpt(checkpoint_path=checkpoint_path,
                                                                 from_safetensors=from_safetensors, **kwargs)
    pipeline.register_modules(safety_checker=None, feature_extractor=None)

    return pipeline


@dataclass
class Merger:
    model_a: str
    model_b: str
    device: str
    skip_position_ids: int = 0

    @cached_property
    def target_pipeline(self):
        return load_checkpoint(self.model_b)

    def merge(
        self,
        weights: List[float],
        base_alpha: int,
        modules=None,
    ):
        if len(weights) != NUM_TOTAL_BLOCKS:
            raise ValueError(f"weights value must be {NUM_TOTAL_BLOCKS}")

        pipeline = load_checkpoint(self.model_a)

        merged_count = 0
        skipped_count = 0
        for module_key, module in pipeline.components.items():
            if not hasattr(module, "state_dict"):
                continue

            if modules is not None and module_key not in modules:
                skipped_count += 1
                continue

            theta_0 = module.state_dict()
            theta_1 = get_module_state_dict(
                self.target_pipeline, module_key, device=pipeline.device)

            re_inp = re.compile(r"\.input_blocks\.(\d+)\.")  # 12
            re_mid = re.compile(r"\.middle_block\.(\d+)\.")  # 1
            re_out = re.compile(r"\.output_blocks\.(\d+)\.")  # 12

            for key in theta_0.keys():
                if "model" in key and key in theta_1:
                    if KEY_POSITION_IDS in key and self.skip_position_ids in [1, 2]:
                        if self.skip_position_ids == 2:
                            theta_0[key] = torch.tensor(
                                [list(range(77))], dtype=torch.int64
                            )
                        continue

                    c_alpha = base_alpha
                    if "model.diffusion_model." in key:
                        weight_index = -1

                        if "time_embed" in key:
                            weight_index = 0  # before input blocks
                        elif ".out." in key:
                            weight_index = NUM_TOTAL_BLOCKS - 1  # after output blocks
                        elif m := re_inp.search(key):
                            weight_index = int(m.groups()[0])
                        elif re_mid.search(key):
                            weight_index = NUM_INPUT_BLOCKS
                        elif m := re_out.search(key):
                            weight_index = (
                                NUM_INPUT_BLOCKS + NUM_MID_BLOCK +
                                int(m.groups()[0])
                            )

                        if weight_index >= NUM_TOTAL_BLOCKS:
                            raise ValueError(f"illegal block index {key}")

                        if weight_index >= 0:
                            c_alpha = weights[weight_index]

                    theta_0[key] = (1 - c_alpha) * theta_0[key] + \
                        c_alpha * theta_1[key]
                    theta_0[key] = theta_0[key].half()
                    merged_count += 1

            for key in theta_1.keys():
                if "model" in key and key not in theta_0:
                    if KEY_POSITION_IDS in key and self.skip_position_ids in [1, 2]:
                        if self.skip_position_ids == 2:
                            theta_1[key] = torch.tensor(
                                [list(range(77))], dtype=torch.int64
                            )
                        continue
                    theta_0.update({key: theta_1[key]})
                    theta_0[key] = theta_0[key].half()
                    merged_count += 1
            del theta_1

            module.load_state_dict(theta_0)
            del theta_0

        print(f"Merged {merged_count} params, skipped {skipped_count} params")
        return pipeline
