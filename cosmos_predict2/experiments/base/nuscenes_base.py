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
Base 2B NuScenes experiment configuration WITHOUT LoRA or kinematic modules.
Use this for inference with original Cosmos checkpoints.
"""

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]

# NuScenes dataset configuration
example_dataset_nuscenes = L(VideoDataset)(
    dataset_dir="/scratch/user/u.pt152369/WM/data/datasetWM",
    num_frames=29,
    video_size=(720, 1280),
    caption_format="json",
)

dataloader_train_nuscenes = L(get_generic_dataloader)(
    dataset=example_dataset_nuscenes,
    sampler=L(get_sampler)(dataset=example_dataset_nuscenes),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

# Base configuration (no LoRA, no kinematics)
_base_defaults = [
    f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
    {"override /data_train": "mock"},
    {"override /data_val": "mock"},
    "_self_",
]

_base_checkpoint = dict(
    load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
    load_from_object_store=dict(enabled=False),
    save_to_object_store=dict(enabled=False),
)

_base_model_config = dict(
    config=dict(
        # Disable LoRA
        use_lora=False,
        # Disable kinematic modules
        kinematic_loss_weight=0.0,
        # Standard conditional frame settings (2 latent frames = 5 pixel frames)
        min_num_conditional_frames=2,
        max_num_conditional_frames=2,
        conditional_frames_probs={2:1.0},
        conditional_frame_timestep=-1.0,  # Use clean frames for conditioning (no noise)
        conditioning_strategy="frame_replace",
        denoise_replace_gt_frames=True,
    ),
)

_base_model_parallel = dict(
    context_parallel_size=1,
)

# Main experiment configuration
predict2_2b_nuscenes = dict(
    defaults=_base_defaults,
    job=dict(
        project="cosmos_predict_v2p5",
        group="base_nuscenes",
        name="2b_nuscenes_base",
    ),
    dataloader_train=dataloader_train_nuscenes,
    checkpoint=_base_checkpoint,
    model=_base_model_config,
    model_parallel=_base_model_parallel,
)

cs = ConfigStore.instance()

# Register the configuration
cs.store(
    group="experiment",
    package="_global_",
    name="predict2_2b_nuscenes",
    node=predict2_2b_nuscenes,
)

