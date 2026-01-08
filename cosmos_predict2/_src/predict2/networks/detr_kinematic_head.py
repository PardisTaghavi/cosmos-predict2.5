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
DETR-based head for predicting agent kinematics from DiT features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple

from .kinematic_matcher import KinematicHungarianMatcher


# Normalization constants for kinematic data (computed from dataset statistics)
# These map raw values to [-1, 1] range: normalized = raw / scale
# Using per-dimension scales for all kinematic quantities (x, y, z have different ranges)
KINEMATIC_NORMALIZATION = {
    'position': torch.tensor([80.0, 20.0, 5.0]),      # [x, y, z] scales in meters
                                                         # x: forward (up to ~80m)
                                                         # y: left/right (up to ~20m)
                                                         # z: up/down (up to ~5m)
    'velocity': torch.tensor([30.0, 10.0, 2.0]),       # [vx, vy, vz] scales in m/s
                                                         # vx: forward velocity (up to ~30 m/s)
                                                         # vy: lateral velocity (up to ~10 m/s)
                                                         # vz: vertical velocity (up to ~2 m/s)
    'acceleration': torch.tensor([5.0, 3.0, 2.0]),     # [ax, ay, az] scales in m/s²
                                                         # ax: forward acceleration (up to ~5 m/s²)
                                                         # ay: lateral acceleration (up to ~3 m/s²)
                                                         # az: vertical acceleration (up to ~2 m/s²)
}

def normalize_kinematics(
    position: torch.Tensor,
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize kinematic values to [-1, 1] range using per-dimension scaling.
    
    Args:
        position: [..., 3] position in meters (x, y, z)
        velocity: [..., 3] velocity in m/s (vx, vy, vz)
        acceleration: [..., 3] acceleration in m/s² (ax, ay, az)
    
    Returns:
        Normalized (position, velocity, acceleration) in [-1, 1] range
    """
    # Position: per-dimension normalization (x, y, z have different scales)
    pos_scale = KINEMATIC_NORMALIZATION['position'].to(position.device).to(position.dtype)
    pos_norm = position / pos_scale
    
    # Velocity: per-dimension normalization (vx, vy, vz have different scales)
    vel_scale = KINEMATIC_NORMALIZATION['velocity'].to(velocity.device).to(velocity.dtype)
    vel_norm = velocity / vel_scale
    
    # Acceleration: per-dimension normalization (ax, ay, az have different scales)
    acc_scale = KINEMATIC_NORMALIZATION['acceleration'].to(acceleration.device).to(acceleration.dtype)
    acc_norm = acceleration / acc_scale
    
    return pos_norm, vel_norm, acc_norm


def denormalize_kinematics(
    position: torch.Tensor,
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Denormalize kinematic values from [-1, 1] range back to original scale using per-dimension scaling.
    
    Args:
        position: [..., 3] normalized position in [-1, 1] (x, y, z)
        velocity: [..., 3] normalized velocity in [-1, 1] (vx, vy, vz)
        acceleration: [..., 3] normalized acceleration in [-1, 1] (ax, ay, az)
    
    Returns:
        Denormalized (position, velocity, acceleration) in original units
    """
    # Position: per-dimension denormalization (x, y, z have different scales)
    pos_scale = KINEMATIC_NORMALIZATION['position'].to(position.device).to(position.dtype)
    pos_denorm = position * pos_scale
    
    # Velocity: per-dimension denormalization (vx, vy, vz have different scales)
    vel_scale = KINEMATIC_NORMALIZATION['velocity'].to(velocity.device).to(velocity.dtype)
    vel_denorm = velocity * vel_scale
    
    # Acceleration: per-dimension denormalization (ax, ay, az have different scales)
    acc_scale = KINEMATIC_NORMALIZATION['acceleration'].to(acceleration.device).to(acceleration.dtype)
    acc_denorm = acceleration * acc_scale
    
    return pos_denorm, vel_denorm, acc_denorm


class DETRKinematicHead(nn.Module):
    """
    DETR-based head for predicting agent kinematics from DiT features.
    
    Given DiT output [B, T*H*W, D], predicts agent states [B, T, N, 14]:
    - Position: (x, y, z)
    - Velocity: (vx, vy, vz)
    - Acceleration: (ax, ay, az)
    - Class: 5 logits (no-object, ego, vehicle, person, bicycle)
    """
    
    def __init__(
        self,
        model_channels: int = 2048,
        num_agents: int = 32,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        max_frames: int = 100,
    ):
        super().__init__()
        
        self.model_channels = model_channels
        self.num_agents = num_agents
        self.max_frames = max_frames
        
        # Learnable object queries [N, D] - one per agent
        # These are learned "slots" that ask: "where are the agents?"
        self.object_queries = nn.Parameter(torch.randn(num_agents, model_channels))
        nn.init.normal_(self.object_queries, std=0.02)
        
        # Temporal positional encoding (added to pooled features per frame)
        self.temporal_pos_encoding = nn.Parameter(torch.randn(1, max_frames, model_channels))
        nn.init.normal_(self.temporal_pos_encoding, std=0.02)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_channels,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )
        
        # Prediction heads (separate MLPs for each output type)
        # Note: Removed LayerNorm for BFloat16 compatibility
        
        # Position head: predicts (x, y, z) in meters
        self.position_head = nn.Sequential(
            nn.Linear(model_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),  # (x, y, z)
        )
        
        # Velocity head: predicts (vx, vy, vz) in m/s
        self.velocity_head = nn.Sequential(
            nn.Linear(model_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),  # (vx, vy, vz)
        )
        
        # Acceleration head: predicts (ax, ay, az) in m/s^2
        self.acceleration_head = nn.Sequential(
            nn.Linear(model_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),  # (ax, ay, az)
        )
        
        # Class head: predicts 5 class logits (no-object, ego, vehicle, person, bicycle)
        self.class_head = nn.Sequential(
            nn.Linear(model_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 5),  # 5 classes: 0=no-object, 1=ego, 2=vehicle, 3=person, 4=bicycle
        )
    
    def pool_spatial_features(
        self,
        features_B_THW_D: torch.Tensor,
        spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Pool spatial features to get one feature vector per frame.
        
        Args:
            features_B_THW_D: DiT output [B, T*H*W, D]
            spatial_shape: (H, W)
        
        Returns:
            temporal_features: [B, T, D] - one feature per frame
        """
        B, THW, D = features_B_THW_D.shape
        H, W = spatial_shape
        T = THW // (H * W)
        
        # Reshape: [B, T*H*W, D] -> [B, T, H*W, D]
        features = rearrange(features_B_THW_D, "b (t h w) d -> b t (h w) d", t=T, h=H, w=W)
        
        # Global average pooling over spatial dimensions
        temporal_features = features.mean(dim=2)  # [B, T, D]
        
        return temporal_features
    
    def forward(
        self,
        dit_output_B_THW_D: torch.Tensor,
        spatial_shape: Tuple[int, int],
    ) -> dict:
        """
        Predict agent kinematics from DiT output.
        
        Args:
            dit_output_B_THW_D: DiT output features [B, T*H*W, D]
            spatial_shape: (H, W) spatial dimensions
        
        Returns:
            predictions: Dictionary containing:
                - 'position': [B, T, N, 3] - (x, y, z)
                - 'velocity': [B, T, N, 3] - (vx, vy, vz)
                - 'acceleration': [B, T, N, 3] - (ax, ay, az)
                - 'class_logits': [B, T, N, 5] - class logits (0=no-object, 1=ego, 2=vehicle, 3=person, 4=bicycle)
                - 'kinematics': [B, T, N, 14] - concatenated output
        """
        B, THW, D = dit_output_B_THW_D.shape
        H, W = spatial_shape
        T = THW // (H * W)
        N = self.num_agents
        
        # 1. Pool spatial features to get temporal features [B, T, D]
        temporal_features = self.pool_spatial_features(dit_output_B_THW_D, spatial_shape)
        
        # 2. Add temporal positional encoding
        temporal_pos = self.temporal_pos_encoding[:, :T, :]  # [1, T, D]
        temporal_features = temporal_features + temporal_pos  # [B, T, D]
        
        # 3. Prepare object queries for decoder
        # Object queries: [N, D] -> [B, T, N, D]
        # We want independent queries per frame (each frame predicts N agents)
        object_queries = repeat(self.object_queries, "n d -> b t n d", b=B, t=T)
        object_queries = rearrange(object_queries, "b t n d -> (b t) n d")  # [B*T, N, D]
        
        # Prepare memory (temporal features) for decoder
        memory = repeat(temporal_features, "b t d -> (b t) 1 d")  # [B*T, 1, D]
        
        # 4. Run DETR decoder
        # Each agent query cross-attends to the frame features
        decoded_queries = self.decoder(
            tgt=object_queries,  # [B*T, N, D] - what we want to predict
            memory=memory,        # [B*T, 1, D] - what we condition on
        )  # Output: [B*T, N, D]
        
        # Reshape back: [B*T, N, D] -> [B, T, N, D]
        decoded_queries = rearrange(decoded_queries, "(b t) n d -> b t n d", b=B, t=T)
        
        # 5. Apply prediction heads
        # Reshape for head processing: [B, T, N, D] -> [B*T*N, D]
        decoded_flat = rearrange(decoded_queries, "b t n d -> (b t n) d")
        
        # Predict each component
        position = self.position_head(decoded_flat)        # [B*T*N, 3]
        velocity = self.velocity_head(decoded_flat)        # [B*T*N, 3]
        acceleration = self.acceleration_head(decoded_flat) # [B*T*N, 3]
        class_logits = self.class_head(decoded_flat)       # [B*T*N, 5]
        
        # Reshape back: [B*T*N, F] -> [B, T, N, F]
        position = rearrange(position, "(b t n) f -> b t n f", b=B, t=T, n=N)
        velocity = rearrange(velocity, "(b t n) f -> b t n f", b=B, t=T, n=N)
        acceleration = rearrange(acceleration, "(b t n) f -> b t n f", b=B, t=T, n=N)
        class_logits = rearrange(class_logits, "(b t n) f -> b t n f", b=B, t=T, n=N)
        
        # 6. Concatenate into full kinematic output [B, T, N, 14]
        kinematics = torch.cat([
            position,       # 3
            velocity,       # 3
            acceleration,   # 3
            class_logits,   # 5
        ], dim=-1)  # [B, T, N, 14]
        
        return {
            'position': position,
            'velocity': velocity,
            'acceleration': acceleration,
            'class_logits': class_logits,
            'kinematics': kinematics,
        }


def compute_kinematic_loss(
    predictions: dict,
    target_kinematics: torch.Tensor,
    matcher: Optional[KinematicHungarianMatcher] = None,
    loss_weights: Optional[dict] = None,
    eos_coef: float = 0.1,
) -> dict:
    """
    Compute kinematic prediction loss with Hungarian matching.
    
    Design:
        - Query 0: Reserved for ego vehicle (supervised directly, no matching)
          * Velocity loss (L1)
          * Acceleration loss (L1)
          * Class loss (should predict class=1 for ego)
          * NO position loss (ego position is always [0,0,0])
        
        - Queries 1-31: Matched to visible agents via Hungarian algorithm
          * Position loss (L1) on matched pairs
          * Velocity loss (L1) on matched pairs
          * Acceleration loss (L1) on matched pairs
          * Class loss (cross-entropy) on all queries (matched + unmatched)
    
    Args:
        predictions: dict with 'position', 'velocity', 'acceleration', 'class_logits'
            - 'position': [B, T, N_queries, 3] (query 0 = ego)
            - 'velocity': [B, T, N_queries, 3] (query 0 = ego)
            - 'acceleration': [B, T, N_queries, 3] (query 0 = ego)
            - 'class_logits': [B, T, N_queries, 5] - (0=no-object, 1=ego, 2=vehicle, 3=person, 4=bicycle)
        target_kinematics: [B, T, N_targets, 18] ground truth kinematics
            - [:, :, :, 0:3]: (x, y, z) position
            - [:, :, :, 3:6]: (vx, vy, vz) velocity
            - [:, :, :, 6:9]: (ax, ay, az) acceleration
            - [:, :, :, 9:12]: (l, w, h) object dimensions
            - [:, :, :, 12]: yaw angle
            - [:, :, :, 13]: tracking_id
            - [:, :, :, 14:18]: one-hot class labels (4 classes: ego, vehicle, person, bicycle)
        matcher: Hungarian matcher (if None, uses default)
        loss_weights: dict of loss weights (default: {'pos': 5.0, 'vel': 2.0, 'acc': 1.0, 'cls': 2.0, 'ego': 2.0})
        eos_coef: Weight for no-object class (default: 0.1)
    
    Returns:
        losses: dict with individual and total losses
    """
    if matcher is None:
        matcher = KinematicHungarianMatcher(
            cost_class=1.0,
            cost_position=5.0,
            cost_velocity=2.0,
        )
    
    # Default loss weights
    default_loss_weights = {'pos': 5.0, 'vel': 2.0, 'acc': 1.0, 'cls': 2.0, 'ego': 2.0}
    if loss_weights is None:
        loss_weights = default_loss_weights
    else:
        # Merge with defaults to ensure all keys are present
        loss_weights = {**default_loss_weights, **loss_weights}
    
    B, T, N_queries, _ = predictions['position'].shape
    device = predictions['position'].device
    
    # ============================================================================
    # Part 1: Dedicated ego loss for query 0
    # ============================================================================
    # Find ego target (class 0 in 4-class = ego, index 14 in one-hot)
    target_cls_onehot_4 = target_kinematics[..., 14:18]  # [B, T, N_targets, 4]
    is_ego_target = target_cls_onehot_4[..., 0] == 1.0  # [B, T, N_targets]
    
    # DEBUG: Print ego detection statistics (every 100 iterations)
    if not hasattr(compute_kinematic_loss, '_debug_iter'):
        compute_kinematic_loss._debug_iter = 0
    compute_kinematic_loss._debug_iter += 1
    
    if compute_kinematic_loss._debug_iter % 100 == 0:
        # Check slot 0 class distribution
        slot0_class_onehot = target_cls_onehot_4[:, :, 0, :]  # [B, T, 4]
        slot0_class_idx = slot0_class_onehot.argmax(dim=-1)  # [B, T]
        slot0_is_ego = (slot0_class_onehot[:, :, 0] == 1.0).sum().item()
        slot0_is_no_object = (slot0_class_onehot.sum(dim=-1) == 0).sum().item()
        slot0_is_vehicle = (slot0_class_onehot[:, :, 1] == 1.0).sum().item()
        
        # Check where ego is found
        ego_found_at_slot0 = is_ego_target[:, :, 0].sum().item()
        ego_found_total = is_ego_target.sum().item()
        ego_found_at_other_slots = ego_found_total - ego_found_at_slot0
        
        print(f"\n[Ego Detection Debug] iter={compute_kinematic_loss._debug_iter}")
        print(f"  Slot 0 one-hot vector (first frame example): {slot0_class_onehot[0, 0].cpu().tolist()}")
        print(f"    Format: [ego, vehicle, person, bicycle]")
        print(f"    Expected for ego: [1, 0, 0, 0]")
        print(f"  Slot 0 class distribution (4-class indices):")
        print(f"    - Ego [1,0,0,0] → 4-class=0: {slot0_is_ego} frames")
        print(f"    - No-object [0,0,0,0] → 4-class=0 (but sum=0): {slot0_is_no_object} frames")
        print(f"    - Vehicle [0,1,0,0] → 4-class=1: {slot0_is_vehicle} frames")
        print(f"    - 4-class index distribution: {torch.unique(slot0_class_idx, return_counts=True)}")
        
        # Convert slot 0 to 5-class for clarity
        slot0_sum = slot0_class_onehot.sum(dim=-1)  # [B, T]
        slot0_class_5 = torch.where(
            slot0_sum == 0,
            torch.tensor(0, device=slot0_class_idx.device),  # no-object
            slot0_class_idx + 1  # shift: 0->1, 1->2, 2->3, 3->4
        )
        slot0_class_5_unique, slot0_class_5_counts = torch.unique(slot0_class_5, return_counts=True)
        slot0_class_5_dist = {int(c.item()): int(count.item()) for c, count in zip(slot0_class_5_unique, slot0_class_5_counts)}
        print(f"    - 5-class index distribution: {slot0_class_5_dist}  # 0=no-object, 1=ego, 2=vehicle, 3=person, 4=bicycle")
        print(f"    - Expected: 5-class=1 (ego) for ALL frames")
        
        print(f"  Ego found across all slots:")
        print(f"    - At slot 0: {ego_found_at_slot0} frames")
        print(f"    - At other slots: {ego_found_at_other_slots} frames")
        print(f"    - Total ego frames: {ego_found_total} / {B*T} total")
        
        # Check first few slots with one-hot vectors
        print(f"  First 5 slots one-hot vectors (first frame):")
        for slot_idx in range(min(5, target_kinematics.shape[2])):
            slot_class_onehot = target_cls_onehot_4[:, :, slot_idx, :]
            slot_onehot_example = slot_class_onehot[0, 0].cpu().tolist()  # First frame example
            slot_class_idx = slot_class_onehot.argmax(dim=-1)
            slot_is_ego = (slot_class_onehot[:, :, 0] == 1.0).sum().item()
            slot_is_no_object = (slot_class_onehot.sum(dim=-1) == 0).sum().item()
            slot_class_4_dist = torch.unique(slot_class_idx, return_counts=True)
            print(f"    Slot {slot_idx}: one-hot={slot_onehot_example}, ego={slot_is_ego}, no-obj={slot_is_no_object}, 4-class={slot_class_4_dist}")
    
    # Extract ego targets (should be at index 0, but search to be safe)
    ego_target_vel = []
    ego_target_acc = []
    has_ego = []
    ego_slot_indices = []  # Track which slot ego was found at
    
    for b in range(B):
        for t in range(T):
            ego_mask = is_ego_target[b, t]
            if ego_mask.any():
                ego_idx = torch.where(ego_mask)[0][0]  # Get first ego (should be only one)
                ego_target_vel.append(target_kinematics[b, t, ego_idx, 3:6])
                ego_target_acc.append(target_kinematics[b, t, ego_idx, 6:9])
                has_ego.append(True)
                ego_slot_indices.append(ego_idx.item())
            else:
                # No ego in this frame (shouldn't happen, but handle gracefully)
                ego_target_vel.append(torch.zeros(3, device=device))
                ego_target_acc.append(torch.zeros(3, device=device))
                has_ego.append(False)
                ego_slot_indices.append(-1)
    
    # DEBUG: Print ego slot distribution
    if compute_kinematic_loss._debug_iter % 100 == 0:
        ego_slot_tensor = torch.tensor(ego_slot_indices).reshape(B, T)
        unique_slots, counts = torch.unique(ego_slot_tensor[ego_slot_tensor >= 0], return_counts=True)
        print(f"  Ego found at slots: {dict(zip(unique_slots.tolist(), counts.tolist()))}")
        print(f"  Frames without ego: {(ego_slot_tensor < 0).sum().item()}")
    
    ego_target_vel = torch.stack(ego_target_vel).reshape(B, T, 3)  # [B, T, 3]
    ego_target_acc = torch.stack(ego_target_acc).reshape(B, T, 3)  # [B, T, 3]
    has_ego_tensor = torch.tensor(has_ego, device=device).reshape(B, T)  # [B, T]
    
    # Normalize ego targets to [-1, 1] range
    _, ego_target_vel_norm, ego_target_acc_norm = normalize_kinematics(
        torch.zeros_like(ego_target_vel),  # position not used for ego
        ego_target_vel,
        ego_target_acc,
    )
    
    # Normalize ego predictions to [-1, 1] range
    _, ego_pred_vel_norm, ego_pred_acc_norm = normalize_kinematics(
        torch.zeros_like(predictions['velocity'][:, :, 0, :]),  # position not used for ego
        predictions['velocity'][:, :, 0, :],
        predictions['acceleration'][:, :, 0, :],
    )
    
    # Compute ego losses for query 0 (using normalized values)
    # IMPORTANT: Query 0 should ALWAYS predict ego (class 1) if ego exists in the scene
    # Even if ego is not found in targets, we still supervise query 0 to predict ego
    # This ensures query 0 learns to always represent ego
    
    # Get query 0 predictions
    ego_pred_logits = predictions['class_logits'][:, :, 0, :].reshape(-1, 5).float()  # [B*T, 5]
    
    if has_ego_tensor.any():
        # Ego exists: supervise query 0 with ego targets
        # Velocity loss for ego (query 0) - normalized
        ego_loss_vel = F.l1_loss(
            ego_pred_vel_norm[has_ego_tensor],
            ego_target_vel_norm[has_ego_tensor],
            reduction='mean'
        )
        
        # Acceleration loss for ego (query 0) - normalized
        ego_loss_acc = F.l1_loss(
            ego_pred_acc_norm[has_ego_tensor],
            ego_target_acc_norm[has_ego_tensor],
            reduction='mean'
        )
        
        # Class loss for ego (query 0 should predict class=1 for ego)
        # Supervise ALL frames where ego exists (not just matched ones)
        # IMPORTANT: Query 0 is ONLY supervised via this ego_loss_cls, NOT via general class loss
        ego_target_cls = torch.full((B, T), 1, dtype=torch.int64, device=device)  # class 1 = ego
        ego_loss_cls = F.cross_entropy(
            ego_pred_logits,
            ego_target_cls.reshape(-1),
            reduction='mean'
        )
        
        # DEBUG: Print query 0 class predictions and loss
        if compute_kinematic_loss._debug_iter % 100 == 0:
            query0_pred_class = ego_pred_logits.argmax(dim=-1).reshape(B, T)
            query0_pred_ego = (query0_pred_class == 1).sum().item()
            query0_pred_no_object = (query0_pred_class == 0).sum().item()
            query0_pred_vehicle = (query0_pred_class == 2).sum().item()
            query0_pred_person = (query0_pred_class == 3).sum().item()
            query0_pred_bicycle = (query0_pred_class == 4).sum().item()
            
            # Get class logits for query 0 to see confidence
            query0_logits_mean = ego_pred_logits.reshape(B, T, 5).mean(dim=(0, 1))  # [5]
            
            print(f"  Query 0 predictions (supervised ONLY via ego_loss_cls):")
            print(f"    - Predicts ego (class 1): {query0_pred_ego} / {B*T} frames ({100*query0_pred_ego/(B*T):.1f}%)")
            print(f"    - Predicts no-object (class 0): {query0_pred_no_object} / {B*T} frames ({100*query0_pred_no_object/(B*T):.1f}%)")
            print(f"    - Predicts vehicle (class 2): {query0_pred_vehicle} / {B*T} frames")
            print(f"    - Predicts person (class 3): {query0_pred_person} / {B*T} frames")
            print(f"    - Predicts bicycle (class 4): {query0_pred_bicycle} / {B*T} frames")
            print(f"    - Average logits: no-obj={query0_logits_mean[0]:.3f}, ego={query0_logits_mean[1]:.3f}, vehicle={query0_logits_mean[2]:.3f}, person={query0_logits_mean[3]:.3f}, bicycle={query0_logits_mean[4]:.3f}")
            print(f"    - ego_loss_cls: {ego_loss_cls.item():.4f} (weight: {loss_weights['ego']}, weighted: {loss_weights['ego'] * ego_loss_cls.item():.4f})")
            print(f"    - Expected: ALWAYS predict ego (class 1)")
            print()
    else:
        # No ego found in targets - this shouldn't happen for NuScenes, but handle gracefully
        # Still supervise query 0 to predict ego (class 1) - ego should always exist
        ego_loss_vel = torch.tensor(0.0, device=device)
        ego_loss_acc = torch.tensor(0.0, device=device)
        
        # Supervise query 0 to predict ego even if not found (assume ego exists)
        # IMPORTANT: Query 0 should ALWAYS predict ego, even if ego not found in targets
        ego_target_cls = torch.full((B, T), 1, dtype=torch.int64, device=device)  # class 1 = ego (ALWAYS)
        ego_loss_cls = F.cross_entropy(
            ego_pred_logits,
            ego_target_cls.reshape(-1),
            reduction='mean'
        )
        
        # DEBUG: Print warning when ego not found
        if compute_kinematic_loss._debug_iter % 100 == 0:
            query0_pred_class = ego_pred_logits.argmax(dim=-1).reshape(B, T)
            print(f"  ⚠️  WARNING: No ego found in targets for {B*T} frames!")
            print(f"  Query 0 predictions: {torch.unique(query0_pred_class, return_counts=True)}")
            print(f"  Query 0 is still supervised to predict ego (class 1)")
            print()
        
        # DEBUG: Warn if no ego found
        if compute_kinematic_loss._debug_iter % 100 == 0:
            print(f"  WARNING: No ego found in targets for {B*T} frames!")
            print(f"    - Still supervising query 0 to predict ego (class 1)")
    
    # ============================================================================
    # Part 2: Hungarian matching for queries 1-31
    # ============================================================================
    # Perform Hungarian matching (matcher handles query 0 exclusion internally)
    indices = matcher(predictions, target_kinematics)
    
    # Extract targets
    target_pos = target_kinematics[..., :3]  # [B, T, N_targets, 3]
    target_vel = target_kinematics[..., 3:6]  # [B, T, N_targets, 3]
    target_acc = target_kinematics[..., 6:9]  # [B, T, N_targets, 3]
    target_cls_onehot_4 = target_kinematics[..., 14:18]  # [B, T, N_targets, 4]
    
    # Convert 4-class targets to 5-class indices
    target_cls_sum = target_cls_onehot_4.sum(dim=-1)  # [B, T, N_targets]
    target_cls_idx_4 = target_cls_onehot_4.argmax(dim=-1)  # [B, T, N_targets]
    target_cls_idx_5 = torch.where(
        target_cls_sum == 0,
        torch.tensor(0, device=device, dtype=target_cls_idx_4.dtype),
        target_cls_idx_4 + 1  # shift: 0->1, 1->2, 2->3, 3->4
    )  # [B, T, N_targets]
    
    # Collect matched predictions and targets
    matched_pred_pos = []
    matched_pred_vel = []
    matched_pred_acc = []
    matched_tgt_pos = []
    matched_tgt_vel = []
    matched_tgt_acc = []
    
    idx = 0
    for b in range(B):
        for t in range(T):
            pred_idx, tgt_idx = indices[idx]
            idx += 1
            
            if len(pred_idx) > 0:
                matched_pred_pos.append(predictions['position'][b, t, pred_idx])
                matched_pred_vel.append(predictions['velocity'][b, t, pred_idx])
                matched_pred_acc.append(predictions['acceleration'][b, t, pred_idx])
                matched_tgt_pos.append(target_pos[b, t, tgt_idx])
                matched_tgt_vel.append(target_vel[b, t, tgt_idx])
                matched_tgt_acc.append(target_acc[b, t, tgt_idx])
    
    # Compute regression losses (L1) on matched pairs
    if len(matched_pred_pos) > 0:
        matched_pred_pos = torch.cat(matched_pred_pos, dim=0)  # [num_matched, 3]
        matched_pred_vel = torch.cat(matched_pred_vel, dim=0)  # [num_matched, 3]
        matched_pred_acc = torch.cat(matched_pred_acc, dim=0)  # [num_matched, 3]
        matched_tgt_pos = torch.cat(matched_tgt_pos, dim=0)  # [num_matched, 3]
        matched_tgt_vel = torch.cat(matched_tgt_vel, dim=0)  # [num_matched, 3]
        matched_tgt_acc = torch.cat(matched_tgt_acc, dim=0)  # [num_matched, 3]
        
        # Normalize matched targets and predictions to [-1, 1] range
        matched_pred_pos_norm, matched_pred_vel_norm, matched_pred_acc_norm = normalize_kinematics(
            matched_pred_pos, matched_pred_vel, matched_pred_acc
        )
        matched_tgt_pos_norm, matched_tgt_vel_norm, matched_tgt_acc_norm = normalize_kinematics(
            matched_tgt_pos, matched_tgt_vel, matched_tgt_acc
        )
        
        # Compute losses on normalized values
        loss_pos = F.l1_loss(matched_pred_pos_norm, matched_tgt_pos_norm, reduction='mean')
        loss_vel = F.l1_loss(matched_pred_vel_norm, matched_tgt_vel_norm, reduction='mean')
        loss_acc = F.l1_loss(matched_pred_acc_norm, matched_tgt_acc_norm, reduction='mean')
    else:
        loss_pos = torch.tensor(0.0, device=device)
        loss_vel = torch.tensor(0.0, device=device)
        loss_acc = torch.tensor(0.0, device=device)
    
    # Compute classification loss for ALL queries
    # Create target class tensor: matched queries get their target class, unmatched get no-object (0)
    target_classes = torch.full(
        (B, T, N_queries), 0, dtype=torch.int64, device=device
    )  # Initialize all to no-object
    
    idx = 0
    for b in range(B):
        for t in range(T):
            pred_idx, tgt_idx = indices[idx]
            idx += 1
            
            if len(pred_idx) > 0:
                # IMPORTANT: The matcher already excludes query 0 (returns queries 1-31)
                # But add safety check to ensure query 0 is never matched
                # Query 0 is reserved for ego and handled separately
                mask = pred_idx != 0  # Filter out query 0 if somehow present
                pred_idx_filtered = pred_idx[mask]
                tgt_idx_filtered = tgt_idx[mask]
                
                if len(pred_idx_filtered) > 0:
                    target_classes[b, t, pred_idx_filtered] = target_cls_idx_5[b, t, tgt_idx_filtered]
    
    # CRITICAL: Query 0 should ALWAYS be ego (class 1), regardless of matching
    # This overrides any assignment from Hungarian matching
    target_classes[:, :, 0] = 1  # class 1 = ego (ALWAYS)
    
    # DEBUG: Verify query 0 target classes
    if compute_kinematic_loss._debug_iter % 100 == 0:
        query0_target_class = target_classes[:, :, 0]  # [B, T]
        print(f"  Query 0 target class (should be 1=ego for all): {torch.unique(query0_target_class, return_counts=True)}")
        print(f"  Note: Query 0 is supervised ONLY via ego_loss_cls, NOT via general class loss")
    
    # Apply class weights (lower weight for no-object)
    # Convert logits to float32 for cross-entropy (cross-entropy doesn't work well with BFloat16)
    # IMPORTANT: Exclude query 0 from general class loss - it's handled separately via ego_loss_cls
    # Query 0 is reserved for ego and should only be supervised through ego loss
    pred_logits_all = predictions['class_logits'].reshape(B, T, N_queries, 5).float()  # [B, T, N_queries, 5]
    pred_logits_queries_1_31 = pred_logits_all[:, :, 1:, :].reshape(-1, 5)  # [B*T*(N_queries-1), 5] - exclude query 0
    target_classes_queries_1_31 = target_classes[:, :, 1:].reshape(-1)  # [B*T*(N_queries-1)] - exclude query 0
    
    # Class weights must match logits dtype (float32)
    class_weights = torch.ones(5, device=device, dtype=pred_logits_queries_1_31.dtype)
    class_weights[0] = eos_coef  # Lower weight for no-object class
    
    # Compute class loss only for queries 1-31 (query 0 handled separately via ego_loss_cls)
    # Classification loss breakdown:
    # - Query 0: Supervised via ego_loss_cls (target: class 1 = ego, weight: loss_weights['ego'])
    # - Queries 1-31: Supervised via loss_cls (targets from Hungarian matching, weight: loss_weights['cls'])
    #   - Matched queries: Predict assigned class (ego=1, vehicle=2, person=3, bicycle=4)
    #   - Unmatched queries: Predict no-object (class 0, lower weight via eos_coef=0.1)
    if pred_logits_queries_1_31.numel() > 0:
        loss_cls = F.cross_entropy(
            pred_logits_queries_1_31,
            target_classes_queries_1_31,
            weight=class_weights,
            reduction='mean'
        )
    else:
        loss_cls = torch.tensor(0.0, device=device)
    
    # DEBUG: Print general class loss info
    if compute_kinematic_loss._debug_iter % 100 == 0:
        matched_queries = (target_classes_queries_1_31 > 0).sum().item()
        unmatched_queries = (target_classes_queries_1_31 == 0).sum().item()
        print(f"  General class loss (queries 1-31 only, query 0 excluded):")
        print(f"    - Matched queries (predict class 1-4): {matched_queries}")
        print(f"    - Unmatched queries (predict no-object, class 0): {unmatched_queries}")
        print(f"    - loss_cls: {loss_cls.item():.4f} (weight: {loss_weights['cls']}, weighted: {loss_weights['cls'] * loss_cls.item():.4f})")
        print()
    
    # Weighted combination (including ego losses)
    total_loss = (
        loss_weights['pos'] * loss_pos +
        loss_weights['vel'] * loss_vel +
        loss_weights['acc'] * loss_acc +
        loss_weights['cls'] * loss_cls +
        loss_weights['ego'] * (ego_loss_vel + ego_loss_acc + ego_loss_cls)
    )
    
    return {
        'kinematic_loss': total_loss,
        'kinematic_loss_position': loss_pos,
        'kinematic_loss_velocity': loss_vel,
        'kinematic_loss_acceleration': loss_acc,
        'kinematic_loss_class': loss_cls,
        'kinematic_loss_ego_velocity': ego_loss_vel,
        'kinematic_loss_ego_acceleration': ego_loss_acc,
        'kinematic_loss_ego_class': ego_loss_cls,
    }


if __name__ == "__main__":
    """Test the DETR kinematic head."""
    print("=" * 70)
    print("Testing DETR Kinematic Head")
    print("=" * 70)
    
    # Dummy DiT output
    B = 2
    T = 10
    H = 16
    W = 32
    D = 2048
    N = 32
    
    dit_output = torch.randn(B, T*H*W, D)
    
    print(f"\nInput shape: {dit_output.shape}")
    
    # Create head
    head = DETRKinematicHead(
        model_channels=D,
        num_agents=N,
        num_decoder_layers=3,
        num_heads=8,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in head.parameters()) / 1e6:.2f}M")
    
    # Forward pass
    print("\nRunning forward pass...")
    predictions = head(dit_output, spatial_shape=(H, W))
    
    print(f"\nOutput shapes:")
    print(f"  position: {predictions['position'].shape}")
    print(f"  velocity: {predictions['velocity'].shape}")
    print(f"  acceleration: {predictions['acceleration'].shape}")
    print(f"  class_logits: {predictions['class_logits'].shape}")
    print(f"  kinematics: {predictions['kinematics'].shape}")
    
    # Test loss computation with Hungarian matching
    print("\nTesting loss computation with Hungarian matching...")
    # Create target_kinematics in 18D format
    target_kinematics = torch.zeros(B, T, N, 18)
    
    # Add ego vehicle (slot 0, class 0 in 4-class = ego)
    target_kinematics[:, :, 0, 14] = 1.0  # ego class (one-hot index 14)
    target_kinematics[:, :, 0, 3:6] = torch.randn(B, T, 3) * 0.1  # ego velocity
    
    # Add 5 vehicles (slots 1-5, class 1 in 4-class = vehicle)
    target_kinematics[:, :, 1:6, :3] = torch.randn(B, T, 5, 3) * 10  # positions
    target_kinematics[:, :, 1:6, 3:6] = torch.randn(B, T, 5, 3) * 2  # velocities
    target_kinematics[:, :, 1:6, 6:9] = torch.randn(B, T, 5, 3) * 0.5  # accelerations
    target_kinematics[:, :, 1:6, 15] = 1.0  # vehicle class (one-hot index 15)
    
    # Add 3 pedestrians (slots 6-8, class 2 in 4-class = person)
    target_kinematics[:, :, 6:9, :3] = torch.randn(B, T, 3, 3) * 10
    target_kinematics[:, :, 6:9, 3:6] = torch.randn(B, T, 3, 3) * 1
    target_kinematics[:, :, 6:9, 16] = 1.0  # person class (one-hot index 16)
    
    # Rest remain zeros (no-object/empty slots)
    
    # Create matcher
    matcher = KinematicHungarianMatcher(
        cost_class=1.0,
        cost_position=5.0,
        cost_velocity=2.0,
    )
    
    # Compute losses
    losses = compute_kinematic_loss(predictions, target_kinematics, matcher=matcher)
    
    print(f"\nLoss values:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.6f}")
    
    # Verify ego losses are computed
    print(f"\nEgo-specific losses:")
    print(f"  ego_velocity: {losses['kinematic_loss_ego_velocity'].item():.6f}")
    print(f"  ego_acceleration: {losses['kinematic_loss_ego_acceleration'].item():.6f}")
    print(f"  ego_class: {losses['kinematic_loss_ego_class'].item():.6f}")
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    losses['kinematic_loss'].backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    print(f"Gradients flowing: {'✓' if has_grad else '✗'}")
    
    # Test matcher directly
    print("\nTesting Hungarian matcher (queries 1-31 only)...")
    with torch.no_grad():
        indices = matcher(predictions, target_kinematics)
        print(f"Number of matches: {len(indices)} (should be B*T = {B*T})")
        total_matched = sum(len(pred_idx) for pred_idx, _ in indices)
        print(f"Total matched pairs: {total_matched}")
        # Should match 8 agents per frame (5 vehicles + 3 pedestrians, excluding ego)
        print(f"Expected matches per frame: 8 (5 vehicles + 3 pedestrians)")
        
        # Verify query 0 is never in matched indices
        all_pred_indices = torch.cat([pred_idx for pred_idx, _ in indices])
        has_query_0 = (all_pred_indices == 0).any()
        print(f"Query 0 in matches: {'✗ ERROR' if has_query_0 else '✓ Correctly excluded'}")
        
        # Verify matched queries are in range [1, 31]
        if len(all_pred_indices) > 0:
            min_idx = all_pred_indices.min().item()
            max_idx = all_pred_indices.max().item()
            print(f"Matched query range: [{min_idx}, {max_idx}] (should be [1, 31])")
    
    assert predictions['kinematics'].shape == (B, T, N, 14)
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)

