# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Inference-only configuration for nuScenes LoRA model.

This config loads the LoRA checkpoint without initializing kinematic modules,
which is appropriate for standard inference without kinematic conditioning.

Usage:
    torchrun --nproc_per_node=2 examples/inference.py \
      -i /path/to/input.json \
      -o outputs/nuscenes_inference \
      --checkpoint-path /path/to/model_ema_bf16.pt \
      --experiment predict2_lora_inference_2b_nuscenes \
      --disable-guardrails
"""

from hydra.core.config_store import ConfigStore
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

# Use the same base checkpoint as training
DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]

# Inference configuration - inherits from base experiment but overrides model config
predict2_lora_inference_2b_nuscenes = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="lora_nuscenes_inference",
        name="2b_nuscenes_lora_inference",
    ),
    model=dict(
        config=dict(
            # LoRA settings (must match training)
            use_lora=True,
            lora_rank=32,
            lora_alpha=32,
            lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
            
            # Standard inference settings for Video2World (2 conditional frames)
            min_num_conditional_frames=2,
            max_num_conditional_frames=2,
            conditional_frame_timestep=0.1,
            conditioning_strategy="frame_replace",
            denoise_replace_gt_frames=True,
            
            # Disable kinematic-related settings for standard inference
            # (kinematic modules will not be loaded/used)
        ),
    ),
)

cs = ConfigStore.instance()

# Register the inference configuration
cs.store(
    group="experiment",
    package="_global_",
    name="predict2_lora_inference_2b_nuscenes",
    node=predict2_lora_inference_2b_nuscenes,
)



