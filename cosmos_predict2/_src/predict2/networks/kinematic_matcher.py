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
Hungarian matcher for kinematic detection.
Simplified from DETR matcher for agent detection with position, velocity, and class.
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


class KinematicHungarianMatcher(nn.Module):
    """
    Hungarian matcher for kinematic agent detection.
    
    Computes optimal assignment between predicted queries and ground truth agents
    based on position, velocity, and class costs.
    
    Design:
        - Query 0 is RESERVED for ego vehicle (not matched)
        - Queries 1-31 are matched to visible agents (vehicles, pedestrians, cyclists)
        - Ego is always excluded from matching (not visible in RGB image)
    """
    
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_position: float = 5.0,
        cost_velocity: float = 2.0,
    ):
        """
        Args:
            cost_class: Weight for classification cost
            cost_position: Weight for position L1 distance cost
            cost_velocity: Weight for velocity L1 distance cost
        
        Note: Ego vehicle is always excluded from matching (query 0 is reserved for ego).
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_position = cost_position
        self.cost_velocity = cost_velocity
        
        assert cost_class != 0 or cost_position != 0 or cost_velocity != 0, \
            "At least one cost must be non-zero"
    
    @torch.no_grad()
    def forward(self, predictions: dict, targets: torch.Tensor):
        """
        Performs Hungarian matching between predictions and targets.
        
        Design:
            - Query 0 is RESERVED for ego (not included in matching)
            - Only queries 1-31 are matched to visible agents
            - Ego targets are excluded from matching
        
        Args:
            predictions: dict with keys:
                - 'position': [B, T, N_queries, 3] - predicted positions (query 0 = ego)
                - 'velocity': [B, T, N_queries, 3] - predicted velocities (query 0 = ego)
                - 'class_logits': [B, T, N_queries, 5] - class logits (0=no-object, 1=ego, 2-4=classes)
            targets: [B, T, N_targets, 18] - ground truth kinematics
                - [:, :, :, 0:3]: position
                - [:, :, :, 3:6]: velocity
                - [:, :, :, 14:18]: one-hot class labels (4 classes: ego, vehicle, person, bicycle)
        
        Returns:
            List of tuples (pred_indices, target_indices) for each (batch, time) pair.
            Length = B * T, each tuple contains matched indices for queries 1-31.
            Query 0 is NOT included in the returned indices.
        """
        B, T, N_queries, _ = predictions['position'].shape
        
        # Extract predictions for queries 1-31 only (skip query 0 which is reserved for ego)
        pred_pos = predictions['position'][:, :, 1:, :]  # [B, T, N_queries-1, 3]
        pred_vel = predictions['velocity'][:, :, 1:, :]  # [B, T, N_queries-1, 3]
        pred_logits = predictions['class_logits'][:, :, 1:, :]  # [B, T, N_queries-1, 5]
        
        # Extract targets
        target_pos = targets[..., :3]  # [B, T, N_targets, 3]
        target_vel = targets[..., 3:6]  # [B, T, N_targets, 3]
        target_cls_onehot_4 = targets[..., 14:18]  # [B, T, N_targets, 4]
        
        # Convert 4-class targets to 5-class indices
        # Target 4 classes: 0=ego, 1=vehicle, 2=person, 3=bicycle
        # Prediction 5 classes: 0=no-object, 1=ego, 2=vehicle, 3=person, 4=bicycle
        target_cls_sum = target_cls_onehot_4.sum(dim=-1)  # [B, T, N_targets]
        target_cls_idx_4 = target_cls_onehot_4.argmax(dim=-1)  # [B, T, N_targets]
        target_cls_idx_5 = torch.where(
            target_cls_sum == 0,
            torch.tensor(0, device=target_cls_sum.device, dtype=target_cls_idx_4.dtype),
            target_cls_idx_4 + 1  # shift: 0->1, 1->2, 2->3, 3->4
        )  # [B, T, N_targets]
        
        # Create valid mask (non-empty targets, excluding ego)
        valid_targets = target_cls_sum > 0  # [B, T, N_targets]
        
        # Always exclude ego vehicle (class 1 in 5-class system) from matching
        # Ego is handled separately via query 0
        is_ego = target_cls_idx_5 == 1  # [B, T, N_targets]
        valid_targets = valid_targets & ~is_ego  # Exclude ego from matching
        
        indices = []
        
        # Match each (batch, time) independently
        for b in range(B):
            for t in range(T):
                # Get valid targets for this frame
                valid_mask_t = valid_targets[b, t]  # [N_targets]
                n_valid = valid_mask_t.sum().item()
                
                if n_valid == 0:
                    # No valid targets - all queries are unmatched
                    indices.append((
                        torch.tensor([], dtype=torch.int64),
                        torch.tensor([], dtype=torch.int64)
                    ))
                    continue
                
                # Filter to valid targets only
                valid_idx = torch.where(valid_mask_t)[0]  # Indices of valid targets
                tgt_pos_t = target_pos[b, t, valid_mask_t]  # [n_valid, 3]
                tgt_vel_t = target_vel[b, t, valid_mask_t]  # [n_valid, 3]
                tgt_cls_t = target_cls_idx_5[b, t, valid_mask_t]  # [n_valid]
                
                # Get predictions for this frame (convert to float32 for cdist - cdist doesn't support BFloat16)
                pred_pos_t = pred_pos[b, t].float()  # [N_queries, 3]
                pred_vel_t = pred_vel[b, t].float()  # [N_queries, 3]
                pred_logits_t = pred_logits[b, t].float()  # [N_queries, 5] - convert to float32 for softmax
                
                # Convert targets to float32 for cdist (explicit conversion to ensure float32)
                tgt_pos_t_float = tgt_pos_t.float()  # [n_valid, 3]
                tgt_vel_t_float = tgt_vel_t.float()  # [n_valid, 3]
                
                # Compute class cost: negative log-probability of target class
                pred_probs_t = pred_logits_t.softmax(dim=-1)  # [N_queries, 5]
                cost_class = -pred_probs_t[:, tgt_cls_t]  # [N_queries, n_valid]
                
                # Compute position cost: L1 distance (cdist requires float32, not BFloat16)
                cost_position = torch.cdist(pred_pos_t, tgt_pos_t_float, p=1)  # [N_queries, n_valid]
                
                # Compute velocity cost: L1 distance (cdist requires float32, not BFloat16)
                cost_velocity = torch.cdist(pred_vel_t, tgt_vel_t_float, p=1)  # [N_queries, n_valid]
                
                # Final cost matrix
                C = (
                    self.cost_class * cost_class +
                    self.cost_position * cost_position +
                    self.cost_velocity * cost_velocity
                )  # [N_queries, n_valid]
                
                # Solve Hungarian matching
                C_cpu = C.cpu().numpy()
                pred_idx, tgt_idx = linear_sum_assignment(C_cpu)
                
                # Map prediction indices back to original query indices (add 1 to skip query 0)
                pred_idx_original = pred_idx + 1  # Shift from [0, N_queries-2] to [1, N_queries-1]
                
                # Map target indices back to original indices (before filtering)
                tgt_idx_original = valid_idx[tgt_idx].cpu()
                
                indices.append((
                    torch.as_tensor(pred_idx_original, dtype=torch.int64),
                    torch.as_tensor(tgt_idx_original, dtype=torch.int64)
                ))
        
        return indices
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"cost_class={self.cost_class}, "
                f"cost_position={self.cost_position}, "
                f"cost_velocity={self.cost_velocity}, "
                f"ego_query=0)")

