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

"""Generic video dataset loader for Cosmos Predict2."""

import json
import os
import random
import h5py
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from decord import VideoReader, cpu
from megatron.core import parallel_state
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms as T

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import ResizePreprocess, ToTensorVideo


class VideoDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        prompt_type: str | None = None,  # "long", "short", "medium", or None for auto
        caption_format: str = "auto",  # "text", "json", or "auto"
        video_paths: Optional[list[str]] = None,
    ) -> None:
        """Dataset class for loading image-text-to-video generation data.

        Args:
            dataset_dir (str): Base path to the dataset directory
            num_frames (int): Number of frames to load per sequence
            video_size (tuple[int, int]): Target size (H,W) for video frames
            prompt_type (str | None): Which prompt to use from JSON ("long", "short", "medium").
                                     If None, uses the first available prompt type.
                                     Only applicable when using JSON format.
            caption_format (str): Caption format - "text", "json", or "auto" to detect automatically

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - video_name: Dict with episode/frame metadata
        """

        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.prompt_type = prompt_type
        self.caption_format = caption_format

        # Determine caption format and directory
        self._setup_caption_format()

        video_dir = os.path.join(self.dataset_dir, "videos")
        self.kinematics_dir = os.path.join(self.dataset_dir, "kinematics")


        if video_paths is None:
            self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
            self.video_paths = sorted(self.video_paths)
        else:
            self.video_paths = video_paths
        print(f"[VideoDataset] {len(self.video_paths)} videos in total")
        
        # Verify kinematics directory exists
        if not os.path.exists(self.kinematics_dir):
            raise FileNotFoundError(
                f"Kinematics directory not found: {self.kinematics_dir}\n"
                f"Expected structure:\n"
                f"  {self.dataset_dir}/\n"
                f"    ├── videos/\n"
                f"    ├── metas/\n"
                f"    └── kinematics/  ← MISSING\n"
                f"\nPlease ensure your dataset has a 'kinematics' subdirectory with .h5 files."
            )
        
        # Count available kinematics files
        kin_files = [f for f in os.listdir(self.kinematics_dir) if f.endswith('.h5')]
        print(f"[VideoDataset] {len(kin_files)} kinematic files found in {self.kinematics_dir}")
        if len(kin_files) == 0:
            print(f"[VideoDataset] WARNING: No .h5 kinematics files found in {self.kinematics_dir}")

        self.num_failed_loads = 0
        self.preprocess = T.Compose([ToTensorVideo(), ResizePreprocess((video_size[0], video_size[1]))])

    def __str__(self) -> str:
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.video_paths)

    def _load_video(self, video_path: str) -> tuple[np.ndarray, float, int]:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)
        if total_frames < self.sequence_length:
            raise ValueError(
                f"Video {video_path} has only {total_frames} frames, "
                f"at least {self.sequence_length} frames are required."
            )

        # randomly sample a sequence of frames
        max_start_idx = total_frames - self.sequence_length
        if max_start_idx > 0:
            start_frame = np.random.randint(0, max_start_idx)
        else:
            # Video has exactly sequence_length frames, use all of them
            start_frame = 0
        end_frame = start_frame + self.sequence_length
        frame_ids = np.arange(start_frame, end_frame).tolist()

        frame_data = vr.get_batch(frame_ids).asnumpy()
        vr.seek(0)  # set video reader point back to 0 to clean up cache

        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS, assume it is 16
            fps = 16
        del vr  # delete the reader to avoid memory leak
        return frame_data, fps, start_frame

    def _setup_caption_format(self) -> None:
        """Determine the caption format and set up the caption directory."""
        metas_dir = os.path.join(self.dataset_dir, "metas")
        captions_dir = os.path.join(self.dataset_dir, "captions")

        if self.caption_format == "auto":
            # Auto-detect based on directory existence
            if os.path.exists(captions_dir) and any(f.endswith(".json") for f in os.listdir(captions_dir)):
                self.caption_format = "json"
                self.caption_dir = captions_dir
            elif os.path.exists(metas_dir) and any(f.endswith(".txt") for f in os.listdir(metas_dir)):
                self.caption_format = "text"
                self.caption_dir = metas_dir
            else:
                raise ValueError(
                    f"Could not auto-detect caption format. Neither 'metas/*.txt' nor 'captions/*.json' found in {self.dataset_dir}"
                )
        elif self.caption_format == "json":
            if not os.path.exists(captions_dir):
                raise ValueError(f"JSON format specified but 'captions' directory not found in {self.dataset_dir}")
            self.caption_dir = captions_dir
        elif self.caption_format == "text":
            if not os.path.exists(metas_dir):
                raise ValueError(f"Text format specified but 'metas' directory not found in {self.dataset_dir}")
            self.caption_dir = metas_dir
        else:
            raise ValueError(f"Invalid caption_format: {self.caption_format}. Must be 'text', 'json', or 'auto'")

    def _load_text(self, text_source: Path) -> str:
        """Load text caption from file."""
        try:
            return text_source.read_text().strip()
        except Exception as e:
            print(f"[VideoDataset] WARNING: Failed to read caption file {text_source}: {e}")
            return ""

    def _load_json_caption(self, json_path: Path) -> str:
        """Load caption from JSON file, concatenating all clip captions for full video."""
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            # Check if it's the new format with "captions" array (for 5-second clip captions)
            if "captions" in data and isinstance(data["captions"], list):
                # Concatenate all captions for the full 20-second video
                captions = data["captions"]
                # Filter out empty captions and join with separator
                valid_captions = [cap.strip() for cap in captions if cap.strip()]
                if valid_captions:
                    # Join with double newline to separate clip descriptions
                    return "\n\n".join(valid_captions)
                return ""
            
            # Fallback: old format (for backward compatibility)
            # Handle JSON that might not have top-level object
            if not isinstance(data, dict) or not data:
                return ""
            
            # Get the first model's captions (e.g., "qwen3_vl_30b_a3b")
            model_key = next(iter(data.keys()))
            captions = data[model_key]

            if self.prompt_type:
                # Use specified prompt type
                if self.prompt_type in captions:
                    return captions[self.prompt_type]
                else:
                    print(
                        f"[VideoDataset] WARNING: Prompt type '{self.prompt_type}' not found in {json_path}. "
                        f"Available: {list(captions.keys())}. Using first available."
                    )

            # Use first available prompt type
            first_prompt = next(iter(captions.values()))
            return first_prompt

        except Exception as e:
            print(f"[VideoDataset] WARNING: Failed to read JSON caption file {json_path}: {e}")
            return ""

    def _get_frames(self, video_path: str) -> tuple[torch.Tensor, float, int]:
        frames, fps, start_frame = self._load_video(video_path)
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, fps, start_frame
    
    def _load_kinematics(self, video_basename: str, start_frame: int) -> torch.Tensor:
        """Load kinematics from .h5 file at full pixel frame rate.
        
        Kinematics are loaded at pixel frame rate (same as video frames).
        Temporal windowing and downsampling to match VAE compression is handled
        by the KinematicConditioner module, which processes frames in windows
        matching the VAE's temporal compression pattern.
        """
        kinematics_path = os.path.join(self.kinematics_dir, f"{video_basename}.h5")
        
        if not os.path.exists(kinematics_path):
            raise FileNotFoundError(
                f"Kinematics file not found: {kinematics_path}\n"
                f"Expected kinematics_dir: {self.kinematics_dir}\n"
                f"Video basename: {video_basename}"
            )
        
        with h5py.File(kinematics_path, 'r') as f:
            # Load kinematics data: shape (T, num_objects, features)
            if 'frames' not in f:
                raise KeyError(f"'frames' dataset not found in {kinematics_path}. Available keys: {list(f.keys())}")
            
            kin_data = f['frames'][:]
            
            # Sample same temporal range as video for alignment
            # Load at full pixel frame rate - temporal windowing handled by conditioner
            end_frame = start_frame + self.sequence_length
            kin_full = kin_data[start_frame:end_frame]  # [T_pixel, N, D]
            
            return torch.from_numpy(kin_full).float()
                
        
    def __getitem__(self, index: int) -> dict | Any:
        try:
            data = dict()
            video, fps, start_frame = self._get_frames(self.video_paths[index])
            video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]

            # Load caption based on format
            video_path = self.video_paths[index]
            video_basename = os.path.basename(video_path).replace(".mp4", "")

            if self.caption_format == "json":
                caption_path = os.path.join(self.caption_dir, f"{video_basename}.json")
                caption = self._load_json_caption(Path(caption_path))
            else:  # text format
                caption_path = os.path.join(self.caption_dir, f"{video_basename}.txt")
                caption = self._load_text(Path(caption_path))

            data["video"] = video
            data["ai_caption"] = caption

            # Load kinematics with same temporal range as video
            kinematics = self._load_kinematics(video_basename, start_frame)
            data["kinematics"] = kinematics  # (T, num_objects, features)


            _, _, h, w = video.shape

            data["fps"] = fps
            data["image_size"] = torch.tensor([h, w, h, w])
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, h, w)

            return data
        except Exception as e:
            self.num_failed_loads += 1
            print(
                f"[VideoDataset] WARNING: Failed to load video {self.video_paths[index]} "
                f"(total failures: {self.num_failed_loads}): {e}\n"
                f"{traceback.format_exc()}"
            )
            # Randomly sample another video
            return self[np.random.randint(len(self.video_paths))]


def get_generic_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    sampler: Optional[Any] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable] = None,
    **kwargs,  # Ignore extra arguments
) -> DataLoader:
    """Create DataLoader with commonly used parameters.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        sampler: Optional sampler for data loading
        num_workers: Number of worker processes
        pin_memory: Pin memory for CUDA transfer
        drop_last: Drop incomplete last batch
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        collate_fn: Custom collate function
        **kwargs: Extra arguments (ignored)

    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # False when using sampler
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )


def get_sampler(dataset) -> DistributedSampler:
    """Create a distributed sampler for the dataset."""
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


def get_train_val_dataloaders(
    dataset_path: str, val_percentage: float, seed: int, video_size: tuple[int, int] = (704, 1280)
):
    video_dir = os.path.join(dataset_path, "videos")
    if not os.path.exists(video_dir):
        print(f"[VideoDataset] Dataset path {dataset_path} does not exist, returning empty dataloaders")
        return dict(), dict()
    video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
    random.seed(seed)
    random.shuffle(video_paths)

    cutoff = int(len(video_paths) * val_percentage)
    val_video_paths = video_paths[:cutoff]
    train_video_paths = video_paths[cutoff:]

    def get_dataset(video_paths):
        return L(VideoDataset)(
            video_paths=video_paths,
            num_frames=93,
            video_size=video_size,
            dataset_dir=dataset_path,
        )

    ipn_hand_train_dataset = get_dataset(train_video_paths)
    ipn_hand_val_dataset = get_dataset(val_video_paths)

    def get_dataloader(dataset):
        return L(get_generic_dataloader)(
            dataset=dataset,
            sampler=L(get_sampler)(dataset=dataset),
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        )

    return get_dataloader(ipn_hand_train_dataset), get_dataloader(ipn_hand_val_dataset)
