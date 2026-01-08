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

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2._src.predict2.callbacks.validation_draw_sample import ValidationDrawSample
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]

# NuScenes dataset with kinematics
# Training: Randomly samples a window of num_frames from videos (videos have ~40 frames)
# The dataset randomly picks a start_frame and extracts [start_frame:start_frame+num_frames]
example_dataset_nuscenes_train = L(VideoDataset)(
    dataset_dir="/scratch/user/u.pt152369/WM/data/datasetWM",  # Your dataset path
    num_frames=29,                  # Number of frames to load (randomly sampled from ~40 frame videos)
    video_size=(720, 1280),        # Video resolution (original NuScenes resolution)
    caption_format="json",          # Using metas/*.txt for captions
)

dataloader_train_nuscenes = L(get_generic_dataloader)(
    dataset=example_dataset_nuscenes_train,
    sampler=L(get_sampler)(dataset=example_dataset_nuscenes_train),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

# LoRA post-training configuration for NuScenes
_lora_defaults = [
    f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
    {"override /data_train": "mock"},
    {"override /data_val": "mock"},
    "_self_",
]

_lora_checkpoint_base = dict(
    load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
    load_from_object_store=dict(enabled=False),
    save_to_object_store=dict(enabled=False),
)

_lora_optimizer = dict(
    lr=2 ** (-14.5),
    weight_decay=0.001,
)

_lora_scheduler = dict(
    f_max=[0.5],
    f_min=[0.2],
    warm_up_steps=[0],
    cycle_lengths=[100000],
)

_lora_trainer = dict(
    run_validation=False,
    logging_iter=10,
    max_iter=5000,
    callbacks=dict(
        heart_beat=dict(save_s3=False),
        iter_speed=dict(hit_thres=200, save_s3=False),
        device_monitor=dict(save_s3=False),
        every_n_sample_reg=dict(every_n=200, save_s3=False),
        every_n_sample_ema=dict(every_n=200, save_s3=False),
        wandb=dict(save_s3=False),
        wandb_10x=dict(save_s3=False),
        dataloader_speed=dict(save_s3=False),
        validation_draw_sample_reg=L(ValidationDrawSample)(
            n_samples=2,
            is_ema=False,
            save_s3=False,
            do_x0_prediction=False,
        ),
        validation_draw_sample_ema=L(ValidationDrawSample)(
            n_samples=2,
            is_ema=True,
            save_s3=False,
            do_x0_prediction=False,
        ),
    ),
)

_lora_model_config = dict(
    config=dict(
        use_lora=True,
        lora_rank=32,
        lora_alpha=32,
        lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        init_lora_weights=True,
        min_num_conditional_frames=5,  # Fixed to 5 conditional frames
        max_num_conditional_frames=5,
        conditional_frames_probs={5: 1.0},
        conditional_frame_timestep=-1.0,
        conditioning_strategy="frame_replace",
        denoise_replace_gt_frames=True,
        # Kinematic loss weight (relative to video loss)
        kinematic_loss_weight=0.1,  # Tune this: 0.05-0.2 range recommended
    ),
)

_lora_model_parallel = dict(
    context_parallel_size=1,
)


def setup_kinematic_modules(model):
    """
    Setup kinematic conditioning and prediction modules.
    
    This function is called after model initialization to add:
    1. KinematicConditioner: Converts kinematics → spatial grid features
    2. DETRKinematicHead: Predicts agent trajectories from video features
    
    Args:
        model: The Video2WorldModelRectifiedFlow instance
    
    Returns:
        model: The model with kinematic modules attached
    """
    from cosmos_predict2._src.predict2.networks.kinematic_conditioner import KinematicConditioner
    from cosmos_predict2._src.predict2.networks.detr_kinematic_head import DETRKinematicHead
    
    # Get model dimensions from the DiT network
    model_channels = model.net.model_channels  # Should be 2048 for Video2World 2B
    
    print(f"[Kinematic Setup] Initializing kinematic modules with model_channels={model_channels}")
    
    # 1. Create and attach kinematic conditioner
    model.net.kinematic_conditioner = KinematicConditioner(
        model_channels=model_channels,
        kinematic_dim=18,  # (x,y,z,vx,vy,vz,ax,ay,az) + 4 classes
        max_agents=32,     # Maximum number of agents to track
        sigma=0.1,         # Gaussian splatting bandwidth
    )
    print(f"[Kinematic Setup] ✓ KinematicConditioner initialized (4.2M params)")
    
    # 2. Create and attach DETR kinematic prediction head
    model.net.kinematic_head = DETRKinematicHead(
        model_channels=model_channels,
        num_agents=32,              # Number of agent queries
        num_decoder_layers=3,       # Number of DETR decoder layers
        num_heads=8,                # Number of attention heads (must divide model_channels)
        dim_feedforward=2048,       # FFN hidden dimension
        max_frames=100,             # Maximum number of frames (for positional encoding)
    )
    print(f"[Kinematic Setup] ✓ DETRKinematicHead initialized (130.3M params)")
    
    # Move to appropriate device
    device = next(model.net.parameters()).device
    model.net.kinematic_conditioner.to(device)
    model.net.kinematic_head.to(device)
    
    print(f"[Kinematic Setup] ✓ Modules moved to device: {device}")
    print(f"[Kinematic Setup] ✓ Kinematic integration complete!")
    print(f"[Kinematic Setup]   - kin_scale: {model.net.kin_scale.item():.6f} (starts at 0)")
    print(f"[Kinematic Setup]   - kinematic_loss_weight: {model.config.kinematic_loss_weight}")
    
    return model


# Custom callback to setup kinematic modules after model loading
class SetupKinematicModulesCallback:
    """Callback to initialize kinematic modules after model is loaded."""
    
    def __init__(self):
        self.setup_done = False
    
    def __call__(self, trainer):
        """Called by trainer after model initialization."""
        if not self.setup_done:
            print("\n" + "="*70)
            print("Setting up kinematic modules...")
            print("="*70)
            setup_kinematic_modules(trainer.model)
            self.setup_done = True
            print("="*70 + "\n")


predict2_lora_training_2b_nuscenes = dict(
    defaults=_lora_defaults,
    job=dict(
        project="cosmos_predict_v2p5",
        group="lora_nuscenes",
        name="2b_nuscenes_kinematics_lora",
    ),
    dataloader_train=dataloader_train_nuscenes,
    checkpoint=dict(
        **_lora_checkpoint_base,
        save_iter=200,
    ),
    optimizer=_lora_optimizer,
    scheduler=_lora_scheduler,
    trainer=_lora_trainer,
    model=_lora_model_config,
    model_parallel=_lora_model_parallel,
)

cs = ConfigStore.instance()

# Register the configuration
cs.store(
    group="experiment",
    package="_global_",
    name="predict2_lora_training_2b_nuscenes",
    node=predict2_lora_training_2b_nuscenes,
)

