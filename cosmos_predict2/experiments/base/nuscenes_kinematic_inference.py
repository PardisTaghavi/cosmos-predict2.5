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
Kinematic-conditioned inference configuration for nuScenes LoRA model.

This config loads the LoRA checkpoint WITH kinematic modules for inference
that uses kinematic conditioning (agent trajectories).

Usage:
    torchrun --nproc_per_node=2 examples/inference.py \
      -i /path/to/input_with_kinematics.json \
      -o outputs/nuscenes_kinematic_inference \
      --checkpoint-path /path/to/model_ema_bf16.pt \
      --experiment predict2_lora_kinematic_inference_2b_nuscenes \
      --disable-guardrails

Input JSON should include kinematics_path:
    {
      "inference_type": "video2world",
      "name": "ego_car_with_kinematics",
      "prompt_path": "/path/to/prompt.txt",
      "input_path": "/path/to/input.mp4",
      "kinematics_path": "/path/to/kinematics.h5",
      "resolution": "480, 848",
      "guidance": 1
    }
"""

from hydra.core.config_store import ConfigStore
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

# Use the same base checkpoint as training
DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]


def setup_kinematic_modules(model):
    """
    Setup kinematic conditioning and prediction modules for inference.
    
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
    
    print(f"[Kinematic Inference Setup] Initializing kinematic modules with model_channels={model_channels}")
    
    # 1. Create and attach kinematic conditioner
    model.net.kinematic_conditioner = KinematicConditioner(
        model_channels=model_channels,
        kinematic_dim=18,  # (x,y,z,vx,vy,vz,ax,ay,az) + 4 classes + 5 agent types
        max_agents=32,     # Maximum number of agents to track
        sigma=0.1,         # Gaussian splatting bandwidth
    )
    print(f"[Kinematic Inference Setup] ✓ KinematicConditioner initialized")
    
    # 2. Create and attach DETR kinematic prediction head
    model.net.kinematic_head = DETRKinematicHead(
        model_channels=model_channels,
        num_agents=32,              # Number of agent queries
        num_decoder_layers=3,       # Number of DETR decoder layers
        num_heads=8,                # Number of attention heads
        dim_feedforward=2048,       # FFN hidden dimension
        max_frames=100,             # Maximum number of frames
    )
    print(f"[Kinematic Inference Setup] ✓ DETRKinematicHead initialized")
    
    # Move to appropriate device
    device = next(model.net.parameters()).device
    model.net.kinematic_conditioner.to(device)
    model.net.kinematic_head.to(device)
    
    print(f"[Kinematic Inference Setup] ✓ Modules moved to device: {device}")
    print(f"[Kinematic Inference Setup] ✓ Kinematic integration complete!")
    
    return model


# Custom callback to setup kinematic modules after model loading
class SetupKinematicModulesCallback:
    """Callback to initialize kinematic modules after model is loaded for inference."""
    
    def __init__(self):
        self.setup_done = False
    
    def __call__(self, trainer):
        """Called by trainer after model initialization."""
        if not self.setup_done:
            print("\n" + "="*70)
            print("Setting up kinematic modules for inference...")
            print("="*70)
            setup_kinematic_modules(trainer.model)
            self.setup_done = True
            print("="*70 + "\n")


# Inference configuration with kinematic modules
predict2_lora_kinematic_inference_2b_nuscenes = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="lora_nuscenes_kinematic_inference",
        name="2b_nuscenes_kinematic_lora_inference",
    ),
    model=dict(
        config=dict(
            # LoRA settings (must match training)
            use_lora=True,
            lora_rank=32,
            lora_alpha=32,
            lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
            init_lora_weights=True,
            
            # Kinematic-conditioned inference settings
            min_num_conditional_frames=5,  # Fixed to 5 conditional frames
            max_num_conditional_frames=5,
            conditional_frames_probs={5: 1.0},
            conditional_frame_timestep=-1.0,
            conditioning_strategy="frame_replace",
            denoise_replace_gt_frames=True,
            
            # Enable kinematic conditioning for inference
            # Set to non-zero to trigger kinematic module initialization
            kinematic_loss_weight=1.0,  # Triggers module initialization (loss not used in inference)
        ),
    ),
)

cs = ConfigStore.instance()

# Register the kinematic inference configuration
cs.store(
    group="experiment",
    package="_global_",
    name="predict2_lora_kinematic_inference_2b_nuscenes",
    node=predict2_lora_kinematic_inference_2b_nuscenes,
)

