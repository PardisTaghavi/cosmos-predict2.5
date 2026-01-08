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
Kinematic Conditioner module for 3D kinematic conditioning in video generation.
Simple direct injection approach: encodes kinematics and splats to spatial grid.
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import Tuple


class KinematicConditioner(nn.Module):
    """
    Simple Kinematic Conditioner using direct injection.
    
    Processes kinematic data [B, T, N, D] where:
    - B: batch size
    - T: temporal frames
    - N: number of agents (32)
    - D: features per agent (18)
        - [0:3]: x, y, z (position in meters)
        - [3:6]: vx, vy, vz (velocity)
        - [6:9]: ax, ay, az (acceleration)
        - [9:12]: l, w, h (object dimensions)
        - [12]: yaw (rotation angle)
        - [13]: tracking_id
        - [14:18]: 4 one-hot class labels
    
    Approach:
    1. Encode kinematic features (position + velocity + acceleration + class) → embeddings
    2. Project 3D world positions (x,y,z) to 2D image coordinates (u,v)
    3. Splat kinematic embeddings to spatial grid using Gaussian splatting
    4. Return grid-aligned kinematic conditioning [B, T*H*W, D_model]
    
    Coordinate system:
    - World space (NuScenes): x=forward, y=left, z=up (meters)
    - Image space: u=horizontal (width), v=vertical (height), normalized [0,1]
    """
    
    def __init__(
        self,
        model_channels: int = 2048,
        kinematic_dim: int = 13,
        max_agents: int = 32,
        sigma: float = 0.1,  # Gaussian splatting bandwidth
    ):
        """
        Args:
            model_channels: DiT model dimension (2048 for Video2World 2B)
            kinematic_dim: Input kinematic features per agent (18)
            max_agents: Maximum number of agents to track (32)
            sigma: Gaussian kernel bandwidth for splatting (controls spatial spread)
        """
        super().__init__()
        
        self.model_channels = model_channels
        self.kinematic_dim = kinematic_dim
        self.max_agents = max_agents
        self.sigma = sigma
        
        # Encode all kinematic features (13D) to model dimension
        self.kinematic_encoder = nn.Sequential(
            nn.Linear(kinematic_dim, model_channels),
            nn.GELU(),
            nn.Linear(model_channels, model_channels),
        )
        
        # Project 3D world positions (x,y,z) to 2D image coordinates (u,v)
        # This learns the camera projection (world → image space)
        self.position_projector = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 2),  # Output: (u, v) in image space
        )
        
        # Temporal windowing to match VAE compression (factor of 4)
        # VAE pattern: frame 0 alone, then windows of 4 frames
        self.temporal_window = 4
        
        # Temporal aggregation for windows (learnable pooling)
        # Aggregates frames within each window to produce 1 embedding per window
        self.temporal_aggregator = nn.Sequential(
            nn.Linear(model_channels, model_channels),
            nn.GELU(),
            nn.Linear(model_channels, model_channels),
        )
        
        # Final projection (no normalization to avoid BFloat16 issues)
        self.output_proj = nn.Linear(model_channels, model_channels)
        
    def splat_to_grid(
        self,
        embeddings: torch.Tensor,  # [B, T*N, D]
        positions_2d: torch.Tensor,  # [B, T*N, 2] - (u, v) in [0, 1]
        valid_mask: torch.Tensor,  # [B, T*N] - True for valid agents
        spatial_shape: Tuple[int, int],  # (H, W)
    ) -> torch.Tensor:
        """
        Splat agent embeddings to spatial grid using Gaussian splatting.
        
        Args:
            embeddings: Encoded kinematic features [B, T*N, D]
            positions_2d: 2D image coordinates [B, T*N, 2] normalized to [0, 1]
            valid_mask: Boolean mask for valid agents [B, T*N]
            spatial_shape: Output grid size (H, W)
        
        Returns:
            grid: Splatted features [B, T*H*W, D]
        """
        B, TN, D = embeddings.shape
        H, W = spatial_shape
        device = embeddings.device
        dtype = embeddings.dtype
        
        # Create spatial grid: [H, W, 2]
        v_coords = torch.linspace(0, 1, H, device=device, dtype=dtype)
        u_coords = torch.linspace(0, 1, W, device=device, dtype=dtype)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing='ij')
        grid_uv = torch.stack([u_grid, v_grid], dim=-1)  # [H, W, 2]
        
        # Expand for batch: [B, H, W, 2]
        grid_uv = grid_uv.unsqueeze(0).expand(B, -1, -1, -1)
        
        # Expand positions for broadcasting: [B, T*N, 1, 1, 2]
        positions_2d_expanded = positions_2d.view(B, TN, 1, 1, 2)
        
        # Expand grid for broadcasting: [B, 1, H, W, 2]
        grid_uv_expanded = grid_uv.unsqueeze(1)
        
        # Compute squared distances: [B, T*N, H, W]
        # Distance between each agent position and each grid cell
        distances_sq = ((positions_2d_expanded - grid_uv_expanded) ** 2).sum(dim=-1)
        
        # Gaussian kernel: exp(-d^2 / (2 * sigma^2))
        weights = torch.exp(-distances_sq / (2 * self.sigma ** 2))  # [B, T*N, H, W]
        
        # Mask out invalid agents
        valid_mask_expanded = valid_mask.view(B, TN, 1, 1)  # [B, T*N, 1, 1]
        weights = weights * valid_mask_expanded.float()
        
        # Normalize weights per grid cell (sum over agents)
        # [B, 1, H, W] - sum of weights per pixel
        weight_sum = weights.sum(dim=1, keepdim=True) + 1e-8  # Add epsilon for stability
        weights_normalized = weights / weight_sum  # [B, T*N, H, W]
        
        # Splat embeddings to grid: weighted sum
        # embeddings: [B, T*N, D]
        # weights_normalized: [B, T*N, H, W]
        # Output: [B, D, H, W]
        embeddings_expanded = embeddings.view(B, TN, D, 1, 1)  # [B, T*N, D, 1, 1]
        weights_expanded = weights_normalized.unsqueeze(2)  # [B, T*N, 1, H, W]
        
        # Weighted sum: [B, T*N, D, H, W] → [B, D, H, W]
        grid_features = (embeddings_expanded * weights_expanded).sum(dim=1)  # [B, D, H, W]
        
        # Reshape to [B, H*W, D]
        grid_features = rearrange(grid_features, "b d h w -> b (h w) d")
        
        return grid_features
    
    def _apply_temporal_windowing(
        self,
        embeddings_B_T_N_C: torch.Tensor,  # [B, T_pixel, N, C]
        positions_2d_B_T_N_2: torch.Tensor,  # [B, T_pixel, N, 2]
        valid_mask_B_T_N: torch.Tensor,  # [B, T_pixel, N]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply temporal windowing matching VAE's temporal compression pattern.
        
        VAE pattern:
        - Frame 0: processed alone → 1 latent
        - Frames 1-4: processed as window → 1 latent
        - Frames 5-8: processed as window → 1 latent
        - etc.
        
        Args:
            embeddings_B_T_N_C: Encoded embeddings [B, T_pixel, N, C]
            positions_2d_B_T_N_2: 2D positions [B, T_pixel, N, 2]
            valid_mask_B_T_N: Valid mask [B, T_pixel, N]
        
        Returns:
            windowed_embeddings: [B, T_latent, N, C] where T_latent = 1 + (T_pixel - 1) // 4
            windowed_positions: [B, T_latent, N, 2]
            windowed_valid_mask: [B, T_latent, N]
        """
        B, T_pixel, N, C = embeddings_B_T_N_C.shape
        temporal_window = self.temporal_window
        
        # Match VAE's exact windowing pattern
        # VAE formula: T_latent = 1 + (T_pixel - 1) // 4 (this is what the model expects)
        # VAE code processes: frame 0, then windows [1:5], [5:9], ..., and MAY add remainder
        # However, the model's T comes from actual VAE encoding, which may not always add remainder
        # So we use the formula-based T_latent to ensure we match what the model expects
        T_latent_expected = 1 + (T_pixel - 1) // temporal_window
        iter_ = 1 + (T_pixel - 1) // temporal_window
        
        # Initialize output tensors
        windowed_embeddings_list = []
        windowed_positions_list = []
        windowed_valid_mask_list = []
        
        # Frame 0: process alone (matches VAE's _i0_encode, i=0)
        windowed_embeddings_list.append(embeddings_B_T_N_C[:, 0:1, :, :])  # [B, 1, N, C]
        windowed_positions_list.append(positions_2d_B_T_N_2[:, 0:1, :, :])  # [B, 1, N, 2]
        windowed_valid_mask_list.append(valid_mask_B_T_N[:, 0:1, :])  # [B, 1, N]
        
        # Process windows: frames [1:5], [5:9], [9:13], ... (matches VAE's loop i=1 to iter_-1)
        # We process (iter_ - 1) windows to get exactly T_latent_expected frames total
        for i in range(1, iter_):
            start_idx = 1 + temporal_window * (i - 1)
            end_idx = 1 + temporal_window * i
            
            # Extract window: [B, window_size, N, C]
            window_embeddings = embeddings_B_T_N_C[:, start_idx:end_idx, :, :]
            window_positions = positions_2d_B_T_N_2[:, start_idx:end_idx, :, :]
            window_valid = valid_mask_B_T_N[:, start_idx:end_idx, :]
            
            # Aggregate window: mean pooling across temporal dimension
            # [B, window_size, N, C] -> [B, 1, N, C]
            window_mean = window_embeddings.mean(dim=1, keepdim=True)  # [B, 1, N, C]
            window_aggregated = self.temporal_aggregator(
                rearrange(window_mean, "b 1 n c -> (b n) c")
            )  # [B*N, C]
            window_aggregated = rearrange(window_aggregated, "(b n) c -> b 1 n c", b=B, n=N)
            
            windowed_embeddings_list.append(window_aggregated)
            
            # For positions: use mean of window
            window_pos_mean = window_positions.mean(dim=1, keepdim=True)  # [B, 1, N, 2]
            windowed_positions_list.append(window_pos_mean)
            
            # For valid mask: any agent valid in window is valid
            window_valid_any = window_valid.any(dim=1, keepdim=True)  # [B, 1, N]
            windowed_valid_mask_list.append(window_valid_any)
        
        # Note: We do NOT add remainder frames here, as the model expects exactly T_latent_expected frames
        # The VAE may or may not add remainder frames depending on implementation details,
        # but the model's T is determined by the formula T_latent = 1 + (T_pixel - 1) // 4
        
        # Concatenate all windows
        windowed_embeddings = torch.cat(windowed_embeddings_list, dim=1)
        windowed_positions = torch.cat(windowed_positions_list, dim=1)
        windowed_valid_mask = torch.cat(windowed_valid_mask_list, dim=1)
        
        # Verify output shape matches expected T_latent
        T_latent_actual = windowed_embeddings.shape[1]
        assert T_latent_actual == T_latent_expected, (
            f"Temporal windowing mismatch: expected T_latent={T_latent_expected} "
            f"(from formula 1 + (T_pixel - 1) // {temporal_window} with T_pixel={T_pixel}), "
            f"got {T_latent_actual}"
        )
        
        return windowed_embeddings, windowed_positions, windowed_valid_mask
    
    def forward(
        self,
        kinematics_B_T_N_D: torch.Tensor,  # [B, T_pixel, N, 18] - INPUT at pixel frame rate
        spatial_shape: Tuple[int, int],  # (H, W)
    ) -> torch.Tensor:
        """
        Process kinematic data with temporal windowing matching VAE compression.
        
        Args:
            kinematics_B_T_N_D: Kinematic data [B, T_pixel, N, 18] at pixel frame rate
                - [:, :, :, 0:3]: (x, y, z) world coordinates in meters
                - [:, :, :, 3:6]: (vx, vy, vz) velocities
                - [:, :, :, 6:9]: (ax, ay, az) accelerations
                - [:, :, :, 9:12]: (l, w, h) object dimensions
                - [:, :, :, 12]: yaw angle
                - [:, :, :, 13]: tracking_id
                - [:, :, :, 14:18]: one-hot class labels (4 classes)
            spatial_shape: Output spatial grid size (H, W)
        
        Returns:
            conditioning: Kinematic conditioning [B, T_latent*H*W, model_channels]
                where T_latent = 1 + (T_pixel - 1) // 4 (matches VAE temporal compression)
        """
        B, T_pixel, N, D_feat = kinematics_B_T_N_D.shape
        H, W = spatial_shape
        
        # Match dtype to model parameters
        model_dtype = next(self.parameters()).dtype
        kinematics_B_T_N_D = kinematics_B_T_N_D.to(dtype=model_dtype)
        
        # ============================================================================
        # Step 1: Create valid agent mask (agents with non-zero class labels)
        # ============================================================================
        onehot_classes = kinematics_B_T_N_D[..., 14:18]  # [B, T_pixel, N, 4]
        valid_mask_B_T_N = onehot_classes.sum(dim=-1) > 0  # [B, T_pixel, N]
        
        # ============================================================================
        # Step 2: Encode kinematic features (all frames at pixel rate)
        # ============================================================================
        # Reshape for encoding: [B*T*N, D_feat]
        kinematics_BTN_D = rearrange(kinematics_B_T_N_D, "b t n d -> (b t n) d")
        
        # Encode: [B*T*N, model_channels]
        embeddings_BTN_C = self.kinematic_encoder(kinematics_BTN_D)
        
        # Reshape back: [B, T_pixel, N, model_channels]
        embeddings_B_T_N_C = rearrange(
            embeddings_BTN_C, "(b t n) c -> b t n c", b=B, t=T_pixel, n=N
        )
        
        # ============================================================================
        # Step 3: Project 3D positions to 2D image space (all frames)
        # ============================================================================
        # Extract 3D positions: [B, T_pixel, N, 3]
        positions_3d = kinematics_B_T_N_D[..., :3]
        
        # Reshape for projection: [B*T*N, 3]
        positions_3d_BTN = rearrange(positions_3d, "b t n d -> (b t n) d")
        
        # Project to 2D: [B*T*N, 2]
        positions_2d_BTN = self.position_projector(positions_3d_BTN)
        positions_2d_BTN = torch.sigmoid(positions_2d_BTN)  # Normalize to [0, 1]
        
        # Reshape back: [B, T_pixel, N, 2]
        positions_2d_B_T_N = rearrange(
            positions_2d_BTN, "(b t n) d -> b t n d", b=B, t=T_pixel, n=N
        )
        
        # ============================================================================
        # Step 4: Apply temporal windowing (matches VAE compression)
        # ============================================================================
        # Process frames in windows: frame 0 alone, then windows of 4 frames
        # Output: [B, T_latent, N, C] where T_latent = 1 + (T_pixel - 1) // 4
        embeddings_T_latent, positions_T_latent, valid_mask_T_latent = self._apply_temporal_windowing(
            embeddings_B_T_N_C,
            positions_2d_B_T_N,
            valid_mask_B_T_N,
        )
        
        T_latent = embeddings_T_latent.shape[1]
        
        # ============================================================================
        # Step 5: Splat windowed embeddings to spatial grid
        # ============================================================================
        # Reshape for splatting: [B, T_latent, N, C] → [B*T_latent, N, C]
        embeddings_BT_N_C = rearrange(embeddings_T_latent, "b t n c -> (b t) n c")
        positions_2d_BT_N_2 = rearrange(positions_T_latent, "b t n d -> (b t) n d")
        valid_mask_BT_N = rearrange(valid_mask_T_latent, "b t n -> (b t) n")
        
        # Splat to grid: [B*T_latent, H*W, C]
        grid_features_BT_HW_C = self.splat_to_grid(
            embeddings=embeddings_BT_N_C,
            positions_2d=positions_2d_BT_N_2,
            valid_mask=valid_mask_BT_N,
            spatial_shape=spatial_shape,
        )
        
        # Reshape back: [B, T_latent*H*W, C]
        grid_features_B_THW_C = rearrange(
            grid_features_BT_HW_C, "(b t) hw c -> b (t hw) c", b=B, t=T_latent
        )
        
        # Ensure dtype matches model parameters before final projection
        grid_features_B_THW_C = grid_features_B_THW_C.to(dtype=model_dtype)
        
        # ============================================================================
        # Step 6: Final projection
        # ============================================================================
        output = self.output_proj(grid_features_B_THW_C)
        
        return output


if __name__ == "__main__":
    """Test the simplified kinematic conditioner."""
    print("=" * 70)
    print("Testing SimpleKinematicConditioner")
    print("=" * 70)
    
    # Test parameters
    B = 2  # batch size
    T = 10  # temporal frames (after VAE temporal compression)
    N = 32  # number of agents
    D = 13  # kinematic features per agent
    H = 16  # spatial height
    W = 32  # spatial width
    model_channels = 512
    
    # Create dummy kinematics
    kinematics_B_T_N_D = torch.randn(B, T, N, D)
    
    # Set realistic values
    kinematics_B_T_N_D[..., :3] = torch.randn(B, T, N, 3) * 10  # xyz in meters
    kinematics_B_T_N_D[..., 3:9] = torch.randn(B, T, N, 6) * 2  # velocities/accelerations
    kinematics_B_T_N_D[..., 9:] = torch.zeros(B, T, N, 4)  # one-hot classes
    
    # Set some agents as valid (first 10 agents per frame)
    kinematics_B_T_N_D[:, :, :10, 9] = 1.0  # Class 1
    kinematics_B_T_N_D[:, :, 10:15, 10] = 1.0  # Class 2
    
    print(f"\nInput shapes:")
    print(f"  kinematics: {kinematics_B_T_N_D.shape}")
    print(f"  spatial_shape: ({H}, {W})")
    print(f"  valid agents: {(kinematics_B_T_N_D[..., 9:].sum(-1) > 0).float().mean().item():.2%}")
    
    # Create conditioner
    conditioner = KinematicConditioner(
        model_channels=model_channels,
        kinematic_dim=D,
        max_agents=N,
        sigma=0.1,
    )
    
    # Forward pass
    print("\nRunning forward pass...")
    output = conditioner(
        kinematics_B_T_N_D=kinematics_B_T_N_D,
        spatial_shape=(H, W),
    )
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected shape: [B, T*H*W, model_channels] = [{B}, {T*H*W}, {model_channels}]")
    print(f"Output magnitude: {output.abs().mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")
    
    # Verify output shape
    assert output.shape == (B, T * H * W, model_channels), \
        f"Shape mismatch! Got {output.shape}, expected ({B}, {T*H*W}, {model_channels})"
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in conditioner.parameters())
    print(f"Gradients flowing: {'✓' if has_grad else '✗'}")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)

