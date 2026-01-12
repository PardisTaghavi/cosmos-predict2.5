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
# Script for generating I2W videos in s3
PYTHONPATH=. python cosmos_predict2/_src/predict2/inference/video2world.py --experiment=Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4 --ckpt_path s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000 --save_root results/cli_debug_from_s3 --input_root /project/cosmos/ybalaji/data/internal_val_set_clean

# Script for text2world generation
export EXPERIMENT=Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/inference/video2world.py \
--experiment=${EXPERIMENT} \
--ckpt_path s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/${EXPERIMENT}/checkpoints/iter_000025000 \
--save_root results/base_model/${EXPERIMENT}_025k_seed0_t2w \
--num_latent_conditional_frames=0 --seed=0 \
--input_root /project/cosmos/fangyinw/data/pbench/v0

# I2W with context parallel with 8 GPUs:
PYTHONPATH=. torchrun --nproc_per_node=8 cosmos_predict2/_src/predict2/inference/video2world.py --experiment=Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4 --ckpt_path s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000 --save_root results/cli_debug_from_s3 --input_root /project/cosmos/ybalaji/data/internal_val_set_clean --context_parallel_size 8

# V2W with context parallel with 8 GPUs:
PYTHONPATH=. torchrun --nproc_per_node=8 cosmos_predict2/_src/predict2/inference/video2world.py --experiment=Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4 --ckpt_path s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000 --save_root results/cli_debug_from_s3 --input_root pbench_upsampled_prompts --num_latent_conditional_frames=2 --context_parallel_size=8


Folder structure:
We assume the input root contains images and prompts in the following format:
input_root/
 ├── image_1.jpg
 ├── image_1.txt
 ├── image_2.jpg
 └── image_2.txt
 └── ...

or videos and prompts in the following format:
input_root/
 ├── video_1.mp4
 ├── video_1.txt
 ├── video_2.mp4
 └── video_2.txt
 └── ...
"""

import math
import os
from typing import TYPE_CHECKING

import numpy as np
import torch
import torchvision
from megatron.core import parallel_state
from PIL import Image

from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.inference.get_t5_emb import get_text_embedding
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint

_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]
_VIDEO_EXTENSIONS = [".mp4"]

_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


def resize_input(video: torch.Tensor, resolution: list[int]):
    r"""
    Resizes and crops the input video tensor while preserving aspect ratio.

    The video is first resized so that the smaller dimension matches the target resolution,
    preserving the aspect ratio. Then, it's center-cropped to the target resolution.

    Args:
        video (torch.Tensor): Input video tensor of shape (T, C, H, W).
        resolution (list[int]): Target resolution [H, W].

    Returns:
        torch.Tensor: Resized and cropped video tensor of shape (T, C, target_H, target_W).
    """

    orig_h, orig_w = video.shape[2], video.shape[3]
    target_h, target_w = resolution

    scaling_ratio = max((target_w / orig_w), (target_h / orig_h))
    resizing_shape = (int(math.ceil(scaling_ratio * orig_h)), int(math.ceil(scaling_ratio * orig_w)))
    video_resized = torchvision.transforms.functional.resize(video, resizing_shape)
    video_cropped = torchvision.transforms.functional.center_crop(video_resized, resolution)
    return video_cropped


def read_and_process_image(img_path: str, resolution: list[int], num_video_frames: int, resize: bool = True):
    """
    Reads an image, converts it to a video tensor, and processes it for model input.

    The image is loaded, converted to a tensor, and replicated to match the
    `num_video_frames`. It's then optionally resized and permuted to the
    standard video format (B, C, T, H, W).

    Args:
        img_path (str): Path to the input image file.
        resolution (list[int]): Target resolution [H, W] for resizing.
        num_video_frames (int): The number of frames the output video tensor should have.
        resize (bool, optional): Whether to resize the image to the target resolution. Defaults to True.

    Returns:
        torch.Tensor: Processed video tensor of shape (1, C, T, H, W).

    Raises:
        ValueError: If the image extension is not one of the supported types.
    """
    ext = os.path.splitext(img_path)[1]
    if ext not in _IMAGE_EXTENSIONS:
        raise ValueError(f"Invalid image extension: {ext}")

    # Read the image
    img = Image.open(img_path)

    # Convert to tensor
    img = torchvision.transforms.functional.to_tensor(img)
    # Create a video tensor by repeating the first frame
    vid_input = img.unsqueeze(0)  # Add temporal dimension T=1

    # Repeat the first frame to match the desired number of video frames
    # Note: The actual content for frames > 0 will be generated by the model.
    vid_input = torch.cat([vid_input, torch.zeros_like(vid_input).repeat(num_video_frames - 1, 1, 1, 1)], dim=0)
    vid_input = (vid_input * 255.0).to(torch.uint8)  # Convert to uint8 range if needed (might depend on model)
    if resize:
        # Resize and crop to the target resolution
        vid_input = resize_input(vid_input, resolution)

    # Convert to {B, C, T, H, W} format expected by the model
    vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)  # Add batch dim B=1 and permute
    return vid_input


def read_and_process_video(
    video_path: str,
    resolution: list[int],
    num_video_frames: int,
    num_latent_conditional_frames: int = 2,
    resize: bool = True,
):
    """
    Reads a video, processes it for model input.

    The video is loaded using easy_io, and uses the last 4x(num_latent_conditional_frames - 1) + 1 from the video.
    If the video is shorter than num_video_frames, it pads with the last frame repeated.
    The first num_latent_conditional_frames are marked as conditioning frames.

    Args:
        video_path (str): Path to the input video file.
        resolution (list[int]): Target resolution [H, W] for resizing.
        num_video_frames (int): Number of frames needed by the model (should equal model.tokenizer.get_pixel_num_frames(model.config.state_t)).
        num_latent_conditional_frames (int): Number of latent conditional frames from the input video (1 or 2).
        resize (bool, optional): Whether to resize the video to the target resolution. Defaults to True.

    Returns:
        torch.Tensor: Processed video tensor of shape (1, C, T, H, W) where T equals num_video_frames.

    Raises:
        ValueError: If the video extension is not supported or other validation errors.

    Note:
        Uses the last 4x(num_latent_conditional_frames - 1) + 1 frames from the video. If video is shorter, pads with last frame repeated.
    """
    ext = os.path.splitext(video_path)[1]
    if ext.lower() not in _VIDEO_EXTENSIONS:
        raise ValueError(f"Invalid video extension: {ext}")

    # Load video using easy_io
    try:
        video_frames, video_metadata = easy_io.load(video_path)  # Returns (T, H, W, C) numpy array
        log.info(f"Loaded video with shape {video_frames.shape}, metadata: {video_metadata}")
    except Exception as e:
        raise ValueError(f"Failed to load video {video_path}: {e}")

    # Convert numpy array to tensor and rearrange dimensions
    video_tensor = torch.from_numpy(video_frames).float() / 255.0  # Convert to [0, 1] range
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

    available_frames = video_tensor.shape[1]

    # Calculate how many frames to extract from input video
    frames_to_extract = 4 * (num_latent_conditional_frames - 1) + 1
    log.info(f"Will extract {frames_to_extract} frames from input video and pad to {num_video_frames}")

    # Validate num_latent_conditional_frames
    if num_latent_conditional_frames not in [1, 2]:
        raise ValueError(f"num_latent_conditional_frames must be 1 or 2, but got {num_latent_conditional_frames}")

    # Create output tensor with exact num_video_frames
    C, _, H, W = video_tensor.shape
    full_video = torch.zeros(C, num_video_frames, H, W)

    if available_frames < frames_to_extract:
        raise ValueError(
            f"Video has only {available_frames} frames but needs at least {frames_to_extract} frames for num_latent_conditional_frames={num_latent_conditional_frames}"
        )

    # Extract the last frames_to_extract from input video
    start_idx = available_frames - frames_to_extract
    extracted_frames = video_tensor[:, start_idx:, :, :]
    full_video[:, :frames_to_extract, :, :] = extracted_frames
    log.info(f"Extracted last {frames_to_extract} frames from video (frames {start_idx} to {available_frames - 1})")

    # Pad remaining frames with the last extracted frame
    if frames_to_extract < num_video_frames:
        last_frame = extracted_frames[:, -1:, :, :]  # (C, 1, H, W)
        padding_frames = num_video_frames - frames_to_extract
        last_frame_repeated = last_frame.repeat(1, padding_frames, 1, 1)  # (C, padding_frames, H, W)
        full_video[:, frames_to_extract:, :, :] = last_frame_repeated
        log.info(f"Padded {padding_frames} frames with last extracted frame")

    # Convert to the format expected by the rest of the pipeline
    full_video = full_video.permute(1, 0, 2, 3)  # (C, T, H, W) -> (T, C, H, W)
    full_video = (full_video * 255.0).to(torch.uint8)  # Convert to uint8 range

    if resize:
        # Resize and crop to the target resolution
        full_video = resize_input(full_video, resolution)

    # Convert to {B, C, T, H, W} format expected by the model
    full_video = full_video.unsqueeze(0).permute(0, 2, 1, 3, 4)  # Add batch dim B=1 and permute
    return full_video


class Video2WorldInference:
    """
    Handles the Video2World inference process, including model loading, data preparation,
    and video generation from an image/video and text prompt. Now supports context parallelism.
    """

    def __init__(
        self,
        experiment_name: str,
        ckpt_path: str,
        s3_credential_path: str,
        context_parallel_size: int = 1,
        config_file: str = "cosmos_predict2/_src/predict2/configs/video2world/config.py",
        experiment_opts: list[str] | None = None,
        offload_diffusion_model: bool = False,
        offload_text_encoder: bool = False,
        offload_tokenizer: bool = False,
        quantize_8bit: bool = False,
    ):
        """
        Initializes the Video2WorldInference class.

        Loads the diffusion model and its configuration based on the provided
        experiment name and checkpoint path. Sets up distributed processing if needed.

        Args:
            experiment_name (str): Name of the experiment configuration.
            ckpt_path (str): Path to the model checkpoint (local or S3).
            s3_credential_path (str): Path to S3 credentials file (if loading from S3).
            context_parallel_size (int): Number of GPUs for context parallelism.
            quantize_8bit (bool): Whether to quantize DiT network to 8-bit.
        """
        self.experiment_name = experiment_name
        self.ckpt_path = ckpt_path
        self.s3_credential_path = s3_credential_path
        self.context_parallel_size = context_parallel_size
        self.process_group = None

        self.offload_diffusion_model = offload_diffusion_model
        self.offload_text_encoder = offload_text_encoder
        self.offload_tokenizer = offload_tokenizer
        self.quantize_8bit = quantize_8bit

        # If no offloading is specified, instruct model loader to move the model to GPU
        model_device = None if offload_diffusion_model else "cuda"

        # Initialize distributed processing if context parallel size > 1
        if self.context_parallel_size > 1:
            self._init_distributed()

        # Load the model and config
        if experiment_opts is None:
            experiment_opts = []
        if not INTERNAL:
            experiment_opts.append("~data_train")

        # LazyConfig interference is not available yet
        # Use envvar to control whether DiT should be offloaded immediately after ctor
        if self.offload_diffusion_model:
            os.environ["COSMOS_PREDICT2_OFFLOAD_DIT"] = "1"

        model, config = load_model_from_checkpoint(
            experiment_name=self.experiment_name,
            s3_checkpoint_dir=self.ckpt_path,
            config_file=config_file,
            load_ema_to_reg=True,
            experiment_opts=experiment_opts,
            to_device=model_device,
        )

        # By default, everything will be constructed directly on the GPU (except DiT)
        # Handle offloading options at inference entry

        # [On-entry offloading part 1]: DiT was offloaded as default by the lazy ctor
        # Offload or reload according to setup
        if self.offload_diffusion_model:
            log.info("[Memory Optimization] Offloading DiT conditioner to CPU")
            if hasattr(model, "conditioner") and model.conditioner is not None:
                model.conditioner = model.conditioner.to("cpu")
        else:
            # Move everything to the GPU (marginal overhead)
            model.net.to("cuda")

        # [Quantization]: Apply 8-bit quantization to DiT network if requested
        if self.quantize_8bit:
            log.info("[Memory Optimization] Applying 8-bit quantization to DiT network")
            model.net = self._quantize_dit_network(model.net)
            log.info("[Memory Optimization] DiT network quantized to 8-bit")

        # [On-entry offloading part 2]: Tokenizer
        if self.offload_tokenizer:
            log.info("[Memory Optimization] Offloading tokenizer encoder & decoder to CPU")
            if hasattr(model.tokenizer, "encoder") and model.tokenizer.encoder is not None:
                model.tokenizer.encoder = model.tokenizer.encoder.to("cpu")
            if hasattr(model.tokenizer, "decoder") and model.tokenizer.decoder is not None:
                model.tokenizer.decoder = model.tokenizer.decoder.to("cpu")
            torch.cuda.empty_cache()

        # [On-entry offloading part 3]: Text encoder
        if self.offload_text_encoder:
            # Text encoder is the first module in the pipeline.
            # Rather offload it **during** DiT run.
            pass

        if TYPE_CHECKING:
            from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
                Video2WorldModelRectifiedFlow,
            )

            model: Video2WorldModelRectifiedFlow = model

        # Print model temporal capacity information
        state_t = model.config.state_t
        pixel_num_frames = model.tokenizer.get_pixel_num_frames(state_t)
        print("="*70)
        print("Model Temporal Capacity Information:")
        print(f"  state_t (latent frames): {state_t}")
        print(f"  pixel_num_frames (native capacity): {pixel_num_frames}")
        print(f"  VAE temporal compression: 4x")
        print(f"  Formula: pixel_frames = (state_t - 1) * 4 + 1")
        print("="*70)

        # Enable context parallel on the model if using context parallelism
        if self.context_parallel_size > 1:
            model.net.enable_context_parallel(self.process_group)

        self.model = model
        self.config = config
        self.batch_size = 1
        self.neg_t5_embeddings = None

    def _quantize_dit_network(self, dit_network):
        """
        Apply 8-bit quantization to the DiT network using BitsAndBytes.
        
        Args:
            dit_network: The DiT network to quantize
            
        Returns:
            Quantized DiT network
        """
        try:
            import torch
            from torch import nn
            
            # Import bitsandbytes
            try:
                import bitsandbytes as bnb
            except ImportError:
                log.error("BitsAndBytes not installed. Install with: pip install bitsandbytes")
                log.error("Continuing without quantization...")
                return dit_network
            
            # Get device
            device = next(dit_network.parameters()).device
            dtype = next(dit_network.parameters()).dtype
            
            # Move to CPU for quantization if on GPU
            was_on_gpu = device.type == "cuda"
            if was_on_gpu:
                dit_network = dit_network.to("cpu")
            
            # Quantize all Linear layers in the DiT network
            def quantize_linear_layers(module):
                for name, child in module.named_children():
                    if isinstance(child, nn.Linear):
                        # Get layer parameters
                        in_features = child.in_features
                        out_features = child.out_features
                        bias = child.bias is not None
                        
                        # Create 8-bit linear layer
                        quantized_layer = bnb.nn.Linear8bitLt(
                            in_features,
                            out_features,
                            bias=bias,
                            has_fp16_weights=False,
                            threshold=6.0,
                        )
                        
                        # Copy weights
                        with torch.no_grad():
                            quantized_layer.weight.data = child.weight.data.clone()
                            if bias:
                                quantized_layer.bias.data = child.bias.data.clone()
                        
                        # Replace the layer
                        setattr(module, name, quantized_layer)
                        log.debug(f"Quantized layer: {name} ({in_features} -> {out_features})")
                    else:
                        # Recursively quantize child modules
                        quantize_linear_layers(child)
            
            log.info("Quantizing Linear layers in DiT network...")
            quantize_linear_layers(dit_network)
            
            # Move back to original device
            if was_on_gpu:
                dit_network = dit_network.to("cuda")
            
            log.info("DiT network quantization complete")
            return dit_network
            
        except Exception as e:
            log.error(f"Error during quantization: {e}")
            log.error("Continuing without quantization...")
            return dit_network

    def _init_distributed(self):
        """Initialize distributed processing for context parallelism."""

        # Initialize distributed environment
        distributed.init()

        # Initialize model parallel states
        parallel_state.initialize_model_parallel(
            context_parallel_size=self.context_parallel_size,
        )

        # Get the process group for context parallel
        self.process_group = parallel_state.get_context_parallel_group()

        log.info(f"Initialized context parallel with size {self.context_parallel_size}")
        log.info(f"Current rank: {distributed.get_rank()}, World size: {distributed.get_world_size()}")

    def _get_data_batch_input(
        self,
        video: torch.Tensor,
        prompt: str,
        num_conditional_frames: int = 1,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        use_neg_prompt: bool = True,
        camera: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        kinematics: torch.Tensor | None = None,
    ):
        """
        Prepares the input data batch for the diffusion model.

        Constructs a dictionary containing the video tensor, text embeddings,
        and other necessary metadata required by the model's forward pass.
        Optionally includes negative text embeddings.

        Args:
            video (torch.Tensor): The input video tensor (B, C, T, H, W).
            prompt (str): The text prompt for conditioning.
            num_conditional_frames (int): Number of conditional frames to use.
            negative_prompt (str, optional): Custom negative prompt.
            use_neg_prompt (bool, optional): Whether to include negative prompt embeddings. Defaults to True.
            camera: (torch.Tensor, optional) Target camera extrinsics and intrinsics for the K output videos, must be provided for camera conditioned model.
            action: (torch.Tensor, optional) Target robot action for the K output videos, must be provided for action conditioned model.

        Returns:
            dict: A dictionary containing the prepared data batch, moved to the correct device and dtype.
        """
        B, C, T, H, W = video.shape

        data_batch = {
            "dataset_name": "video_data",
            "video": video,
            "camera": camera,
            "action": action.unsqueeze(0) if action is not None else None,
            "fps": torch.tensor([2.0] * self.batch_size).float(),  # Set to 2Hz to match input video
            "padding_mask": torch.zeros(self.batch_size, 1, H, W),  # Padding mask (assumed no padding here)
            "num_conditional_frames": num_conditional_frames,  # Specify number of conditional frames
        }
        
        # Add kinematics if provided
        if kinematics is not None:
            data_batch["kinematics"] = kinematics

        if use_neg_prompt:
            assert negative_prompt is not None, "Negative prompt is required when use_neg_prompt is True"

        # Compute text embeddings
        if self.model.text_encoder is not None:
            data_batch["ai_caption"] = [prompt]
            data_batch["t5_text_embeddings"] = self.model.text_encoder.compute_text_embeddings_online(
                data_batch={"ai_caption": [prompt], "images": None},
                input_caption_key="ai_caption",
            )
            if use_neg_prompt:
                data_batch["neg_t5_text_embeddings"] = self.model.text_encoder.compute_text_embeddings_online(
                    data_batch={"ai_caption": [negative_prompt], "images": None},
                    input_caption_key="ai_caption",
                )
        else:
            data_batch["t5_text_embeddings"] = get_text_embedding(prompt)
            if use_neg_prompt:
                data_batch["neg_t5_text_embeddings"] = get_text_embedding(negative_prompt)

        # Move tensors to GPU and convert to bfloat16 if they are floating point
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

        return data_batch

    def _load_kinematics_from_file(
        self,
        kinematics_path: str,
        num_pixel_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Load kinematics from H5 or NPY file for inference.
        
        Args:
            kinematics_path: Path to .h5 or .npy file containing kinematics
            num_pixel_frames: Number of pixel frames to load (will take first N frames)
            device: Device to load kinematics on
        
        Returns:
            kinematics: [1, T_pixel, N, 18] tensor at pixel frame rate
        """
        import h5py
        import numpy as np
        
        if kinematics_path.endswith('.h5'):
            with h5py.File(kinematics_path, 'r') as f:
                if 'frames' not in f:
                    raise KeyError(f"'frames' dataset not found in {kinematics_path}. Available keys: {list(f.keys())}")
                kin_data = f['frames'][:]  # [T, N, 18]
        elif kinematics_path.endswith('.npy'):
            kin_data = np.load(kinematics_path)  # [T, N, 18]
        else:
            raise ValueError(f"Unsupported kinematics file format: {kinematics_path}. Expected .h5 or .npy")
        
        # Take first num_pixel_frames
        if kin_data.shape[0] < num_pixel_frames:
            log.warning(
                f"Kinematics file has {kin_data.shape[0]} frames but {num_pixel_frames} requested. "
                f"Padding with last frame."
            )
            # Pad with last frame
            last_frame = kin_data[-1:, :, :]  # [1, N, 18]
            padding_frames = num_pixel_frames - kin_data.shape[0]
            padding = np.repeat(last_frame, padding_frames, axis=0)
            kin_data = np.concatenate([kin_data, padding], axis=0)
        else:
            kin_data = kin_data[:num_pixel_frames]
        
        # Convert to tensor: [1, T_pixel, N, 18]
        kinematics = torch.from_numpy(kin_data).float().unsqueeze(0).to(device)
        return kinematics

    def _upsample_kinematics_to_pixel_rate(
        self,
        kinematic_predictions: dict,
        T_pixel_target: int,
    ) -> torch.Tensor:
        """
        Upsample kinematic predictions from latent rate to pixel rate.
        
        Kinematic predictions are at latent rate [B, T_latent, N, 14] (or 18 if full format).
        Need to upsample to pixel rate [B, T_pixel, N, 18] for next chunk conditioning.
        
        Args:
            kinematic_predictions: Dictionary with 'position', 'velocity', 'acceleration', 'class_logits'
                Each is [B, T_latent, N, F] where F is feature dimension
            T_pixel_target: Target number of pixel frames
        
        Returns:
            kinematics_pixel: [B, T_pixel, N, 18] tensor at pixel frame rate
        """
        # Extract components
        position = kinematic_predictions['position']  # [B, T_latent, N, 3]
        velocity = kinematic_predictions['velocity']  # [B, T_latent, N, 3]
        acceleration = kinematic_predictions['acceleration']  # [B, T_latent, N, 3]
        class_logits = kinematic_predictions['class_logits']  # [B, T_latent, N, 5]
        
        B, T_latent, N, _ = position.shape
        
        # Convert 5-class logits to 4-class one-hot for compatibility with 18D format
        # Prediction 5 classes: 0=no-object, 1=ego, 2=vehicle, 3=person, 4=bicycle
        # Target 4 classes: 0=ego, 1=vehicle, 2=person, 3=bicycle
        class_probs = torch.softmax(class_logits, dim=-1)  # [B, T_latent, N, 5]
        class_idx_5 = class_probs.argmax(dim=-1)  # [B, T_latent, N] - class indices 0-4
        
        # Convert to 4-class one-hot: if class_idx_5 == 0 (no-object) -> all zeros
        # else: class_idx_5 - 1 maps to 4-class index (1->0, 2->1, 3->2, 4->3)
        class_onehot_4 = torch.zeros(B, T_latent, N, 4, device=class_logits.device, dtype=class_logits.dtype)
        # Only set one-hot for non-no-object classes
        valid_mask = class_idx_5 > 0  # [B, T_latent, N] - true for objects
        class_idx_4 = class_idx_5[valid_mask] - 1  # [num_valid] - map 1->0, 2->1, 3->2, 4->3
        class_onehot_4[valid_mask].scatter_(-1, class_idx_4.unsqueeze(-1), 1.0)
        
        # Concatenate to [B, T_latent, N, 14]
        kinematics_latent = torch.cat([
            position,      # 3
            velocity,      # 3
            acceleration,  # 3
            class_logits,  # 5
        ], dim=-1)
        
        # Pad to 18D format: add zeros for dimensions, yaw, tracking_id
        # Format: [pos(3), vel(3), acc(3), dims(3), yaw(1), track_id(1), class(4)] = 18
        zeros_dims = torch.zeros(B, T_latent, N, 3, device=kinematics_latent.device, dtype=kinematics_latent.dtype)
        zeros_yaw = torch.zeros(B, T_latent, N, 1, device=kinematics_latent.device, dtype=kinematics_latent.dtype)
        zeros_track = torch.zeros(B, T_latent, N, 1, device=kinematics_latent.device, dtype=kinematics_latent.dtype)
        
        kinematics_18d = torch.cat([
            kinematics_latent[..., :3],   # position
            kinematics_latent[..., 3:6],  # velocity
            kinematics_latent[..., 6:9],  # acceleration
            zeros_dims,                   # dimensions (not predicted)
            zeros_yaw,                     # yaw (not predicted)
            zeros_track,                   # tracking_id (not predicted)
            class_onehot_4,                # class one-hot (4 classes)
        ], dim=-1)  # [B, T_latent, N, 18]
        
        # Upsample from latent rate to pixel rate using linear interpolation
        # VAE compression: T_latent = 1 + (T_pixel - 1) // 4
        # Inverse: T_pixel = (T_latent - 1) * 4 + 1
        # But we need to handle arbitrary T_pixel_target
        
        # Use linear interpolation along temporal dimension
        kinematics_18d_permuted = kinematics_18d.permute(0, 2, 1, 3)  # [B, N, T_latent, 18]
        kinematics_18d_reshaped = kinematics_18d_permuted.reshape(B * N, T_latent, 18)  # [B*N, T_latent, 18]
        
        # Create indices for interpolation
        indices = torch.linspace(0, T_latent - 1, T_pixel_target, device=kinematics_18d.device)
        indices_floor = indices.floor().long()
        indices_ceil = torch.clamp(indices_floor + 1, max=T_latent - 1)
        alpha = (indices - indices_floor).unsqueeze(-1)  # [T_pixel, 1]
        
        # Linear interpolation
        floor_values = kinematics_18d_reshaped[:, indices_floor, :]  # [B*N, T_pixel, 18]
        ceil_values = kinematics_18d_reshaped[:, indices_ceil, :]   # [B*N, T_pixel, 18]
        interpolated = floor_values * (1 - alpha) + ceil_values * alpha  # [B*N, T_pixel, 18]
        
        # Reshape back: [B*N, T_pixel, 18] -> [B, N, T_pixel, 18] -> [B, T_pixel, N, 18]
        kinematics_pixel = interpolated.reshape(B, N, T_pixel_target, 18).permute(0, 2, 1, 3)
        
        return kinematics_pixel

    def _print_kinematic_predictions(self, kinematic_predictions: dict, frame_offset: int = 0):
        """
        Print kinematic predictions for each frame with actual numerical values.
        
        Args:
            kinematic_predictions: Dictionary with kinematic predictions
            frame_offset: Offset to add to frame numbers (for autoregressive chunks)
        """
        if kinematic_predictions is None:
            return
        
        position = kinematic_predictions['position']  # [B, T_latent, N, 3]
        velocity = kinematic_predictions['velocity']  # [B, T_latent, N, 3]
        acceleration = kinematic_predictions['acceleration']  # [B, T_latent, N, 3]
        class_logits = kinematic_predictions['class_logits']  # [B, T_latent, N, 5]
        kinematics = kinematic_predictions['kinematics']  # [B, T_latent, N, 14]
        
        B, T_latent, N, _ = position.shape
        
        for t in range(T_latent):
            frame_num = frame_offset + t + 1
            print(f"\nframe {frame_num}:")
            print(f"   {{")
            
            # Convert tensors to numpy arrays (convert to float32 first to handle BFloat16)
            pos_np = position[0, t, :, :].detach().cpu().float().numpy()
            vel_np = velocity[0, t, :, :].detach().cpu().float().numpy()
            acc_np = acceleration[0, t, :, :].detach().cpu().float().numpy()
            cls_np = class_logits[0, t, :, :].detach().cpu().float().numpy()
            kin_np = kinematics[0, t, :, :].detach().cpu().float().numpy()
            
            # Convert class logits to class predictions
            # class_logits shape: [N, 5] where indices are [0=no-object, 1=ego, 2=vehicle, 3=person, 4=bicycle]
            # Use softmax + argmax to get class predictions
            cls_probs = np.exp(cls_np - np.max(cls_np, axis=-1, keepdims=True))  # [N, 5] - numerical stability
            cls_probs = cls_probs / cls_probs.sum(axis=-1, keepdims=True)  # normalize
            class_predictions = np.argmax(cls_probs, axis=-1)  # [N] - class indices 0-4
            
            # Print with actual values
            print(f"       'position': {pos_np.tolist()},      # (x, y, z) in meters")
            print(f"       'velocity': {vel_np.tolist()},     # (vx, vy, vz) in m/s")
            print(f"       'acceleration': {acc_np.tolist()}, # (ax, ay, az) in m/s²")
            print(f"       'class': {class_predictions.tolist()}, # 0=no-object, 1=ego, 2=vehicle, 3=person, 4=bicycle")
            print(f"       'kinematics': {kin_np.tolist()}   # concatenated [pos(3), vel(3), acc(3), cls(5)]")
            print(f"   }}")

    def generate_vid2world(
        self,
        prompt: str,
        input_path: str | torch.Tensor | None,
        guidance: int = 7,
        num_video_frames: int = 77,
        num_latent_conditional_frames: int = 1,
        num_input_video: int = 1,
        num_output_video: int = 1,
        resolution: str = "192,320",
        seed: int = 1,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        camera: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        kinematics: torch.Tensor | None = None,
        kinematics_path: str | None = None,
        num_steps: int = 35,
    ) -> tuple[torch.Tensor, dict | None]:
        """
        Generates a video based on an input image or video and text prompt.

        Processes the input, prepares the data batch, runs the diffusion
        model sampling, and decodes the result into a video tensor.

        Args:
            prompt: The text prompt describing the desired video content/style.
            input_path: Path to the input image or video file or a torch.Tensor.
            guidance: Classifier-free guidance scale. Defaults to 7.
            num_video_frames: Number of video frames to generate. Defaults to 77.
            num_latent_conditional_frames : Number of latent conditional frames. Defaults to 1.
            resolution: Target video resolution in "H,W" format. Defaults to "192,320".
            seed: Random seed for reproducibility. Defaults to 1.
            negative_prompt: Custom negative prompt. Defaults to the predefined default negative prompt.
            camera: Target camera extrinsics and intrinsics for the K output videos. Must be provided if model is camera conditioned.
            action: Target robot action for the K output videos. Must be provided if model is action conditioned.
            kinematics: Kinematic conditioning tensor [1, T_pixel, N, 18] at pixel frame rate. Optional.
            kinematics_path: Path to .h5 or .npy file containing kinematics. Optional. If provided, will load kinematics from file.
            num_steps: Number of generation steps. Defaults to 35.

        Returns:
            tuple: (video_tensor, kinematic_predictions)
                - video_tensor: The generated video tensor (B, C, T, H, W) in the range [-1, 1]
                - kinematic_predictions: Dictionary with kinematic predictions at latent rate, or None if not available
        """
        assert camera is not None or action is not None or num_input_video == 1 and num_output_video == 1, (
            "expected num_output_video==1 and num_output_video==1 for no camera conditioning or action conditioning"
        )

        # Parse resolution string into tuple of integers
        if resolution == "none":
            h, w = self.model.get_video_height_width()
            video_resolution = (h, w)
        else:
            video_resolution = resolution.split(",")
            video_resolution = tuple([int(x) for x in video_resolution])
            assert len(video_resolution) == 2, "Resolution must be in 'H,W' format"

        # Get the correct number of frames needed by the model
        model_required_frames = self.model.tokenizer.get_pixel_num_frames(self.model.config.state_t)

        # Load kinematics from file if path provided
        if kinematics is None and kinematics_path is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            kinematics = self._load_kinematics_from_file(kinematics_path, num_video_frames, device)
            log.info(f"Loaded kinematics from {kinematics_path}: shape {kinematics.shape}")

        # Determine if input is image or video and process accordingly
        if input_path is None or num_latent_conditional_frames == 0:
            vid_input = torch.zeros(1, 3, model_required_frames, video_resolution[0], video_resolution[1]).to(
                torch.uint8
            )
        elif isinstance(input_path, str):
            ext = os.path.splitext(input_path)[1].lower()
            if ext in _IMAGE_EXTENSIONS:
                log.info(f"Processing image input: {input_path}")
                vid_input = read_and_process_image(
                    img_path=input_path,
                    resolution=video_resolution,
                    num_video_frames=model_required_frames,
                    resize=True,
                )
            elif ext in _VIDEO_EXTENSIONS:
                log.info(f"Processing video input: {input_path}")
                vid_input = read_and_process_video(
                    video_path=input_path,
                    resolution=video_resolution,
                    num_video_frames=model_required_frames,
                    num_latent_conditional_frames=num_latent_conditional_frames,
                    resize=True,
                )
            else:
                raise ValueError(
                    f"Unsupported file extension: {ext}. Supported extensions: {_IMAGE_EXTENSIONS + _VIDEO_EXTENSIONS}"
                )
        elif isinstance(input_path, torch.Tensor):
            vid_input = input_path
        else:
            raise ValueError(f"Unsupported input_path type: {type(input_path)}")

        # Prepare the data batch with text embeddings
        # Note: TextEncoder.compute_text_embeddings_online() will automatically move its model to GPU
        data_batch = self._get_data_batch_input(
            video=vid_input,
            prompt=prompt,
            camera=camera,
            action=action,
            num_conditional_frames=num_latent_conditional_frames,
            negative_prompt=negative_prompt,
            use_neg_prompt=True,
            kinematics=kinematics,
        )

        mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        log.info(f"GPU memory usage after getting data_batch: {mem_bytes / (1024**3):.2f} GB")

        # Memory Optimization Step 1: Offload Text Encoder
        # Offload text encoder after computing embeddings to free memory
        if self.offload_text_encoder and self.model.text_encoder is not None:
            log.info("[Memory Optimization] Offloading text encoder to CPU")
            # TextEncoder is a wrapper class with self.model (the actual neural network)
            if hasattr(self.model.text_encoder, "model") and self.model.text_encoder.model is not None:
                self.model.text_encoder.model = self.model.text_encoder.model.to("cpu")
            torch.cuda.empty_cache()

        # Memory Optimization Step 2: Tokenizer Encoder
        # Load tokenizer encoder to GPU for encoding input video
        if self.offload_tokenizer:
            log.info("[Memory Optimization] Loading tokenizer encoder to GPU")
            if hasattr(self.model.tokenizer, "encoder") and self.model.tokenizer.encoder is not None:
                self.model.tokenizer.encoder = self.model.tokenizer.encoder.to("cuda")
            torch.cuda.empty_cache()

        # Memory Optimization Step 3: Diffusion Network
        # Load the main diffusion network to GPU for sampling
        if self.offload_diffusion_model:
            log.info("[Memory Optimization] Loading diffusion network to GPU")
            self.model.net = self.model.net.to("cuda")
            # Also load conditioner if it exists
            if hasattr(self.model, "conditioner") and self.model.conditioner is not None:
                self.model.conditioner = self.model.conditioner.to("cuda")
            torch.cuda.empty_cache()

        extra_kwargs = {}
        if camera is not None:
            extra_kwargs = {
                "num_input_video": num_input_video,
                "num_output_video": num_output_video,
            }

        # Generate latent samples using the diffusion model
        # Video should be of shape torch.Size([1, 3, 93, 192, 320]) # Note: Shape check comment
        log.info("[Memory Optimization] Starting latent sample generation")
        if self.model.config.use_lora:
            generate_samples = self.model.generate_samples_from_batch_lora
        else:
            generate_samples = self.model.generate_samples_from_batch
        
        # Extract kinematics from data_batch for velocity function
        kinematics_for_velocity = data_batch.get("kinematics", None)
        
        result = generate_samples(
            data_batch,
            n_sample=1,  # Generate one sample
            guidance=guidance,
            seed=seed,  # Fixed seed for reproducibility
            is_negative_prompt=True,  # Use classifier-free guidance
            num_steps=num_steps,
            kinematics=kinematics_for_velocity,
            **extra_kwargs,
        )
        
        # Handle return value: could be (sample,) or (sample, kinematic_predictions)
        if isinstance(result, tuple) and len(result) == 2:
            sample, kinematic_predictions = result
        elif isinstance(result, tuple) and len(result) == 1:
            sample = result[0]
            kinematic_predictions = None
        else:
            sample = result
            kinematic_predictions = None
        
        # Denormalize kinematic predictions from [-1, 1] back to original scale
        if kinematic_predictions is not None:
            from cosmos_predict2._src.predict2.networks.detr_kinematic_head import denormalize_kinematics
            
            # Denormalize position, velocity, acceleration
            pos_denorm, vel_denorm, acc_denorm = denormalize_kinematics(
                kinematic_predictions['position'],
                kinematic_predictions['velocity'],
                kinematic_predictions['acceleration'],
            )
            
            # Update predictions with denormalized values
            kinematic_predictions['position'] = pos_denorm
            kinematic_predictions['velocity'] = vel_denorm
            kinematic_predictions['acceleration'] = acc_denorm
            
            # Update concatenated kinematics tensor
            # kinematics = [pos(3), vel(3), acc(3), cls(5)]
            kinematic_predictions['kinematics'] = torch.cat([
                pos_denorm,      # 3
                vel_denorm,      # 3
                acc_denorm,      # 3
                kinematic_predictions['class_logits'],  # 5
            ], dim=-1)  # [B, T, N, 14]

        # Memory Optimization Step 4: Offload Diffusion Network
        # Offload diffusion network after sampling to make room for decoder
        if self.offload_diffusion_model:
            log.info("[Memory Optimization] Offloading diffusion network to CPU")
            self.model.net = self.model.net.to("cpu")
            if hasattr(self.model, "conditioner") and self.model.conditioner is not None:
                self.model.conditioner = self.model.conditioner.to("cpu")

        if self.offload_tokenizer:
            # Also offload encoder since we only need decoder now
            if hasattr(self.model.tokenizer, "encoder") and self.model.tokenizer.encoder is not None:
                self.model.tokenizer.encoder = self.model.tokenizer.encoder.to("cpu")
            torch.cuda.empty_cache()

        # Memory Optimization Step 5: Load Decoder
        # Load tokenizer decoder to GPU for decoding latents
        if self.offload_tokenizer:
            log.info("[Memory Optimization] Loading tokenizer decoder to GPU")
            if hasattr(self.model.tokenizer, "decoder") and self.model.tokenizer.decoder is not None:
                self.model.tokenizer.decoder = self.model.tokenizer.decoder.to("cuda")
            torch.cuda.empty_cache()

        # Decode the latent samples
        if isinstance(sample, list):
            # Decode the latent sample into a video tensor
            video_list = []
            for sample_chunk in sample:
                video_chunk = self.model.decode(sample_chunk)
                video_list.append(video_chunk)
            video = torch.cat(video_list, dim=3)
        else:
            # Decode the latent sample into a video tensor
            video = self.model.decode(sample)
        
        video = video.clip(min=-1, max=1)

        # Print kinematic predictions for each frame
        if kinematic_predictions is not None:
            print("\n=== Kinematic Predictions ===")
            self._print_kinematic_predictions(kinematic_predictions, frame_offset=0)

        # Memory Optimization Step 6: Final Cleanup
        # Offload decoder after decoding & reload the tokenizer for the next inference call
        if self.offload_tokenizer:
            log.info("[Memory Optimization] Offloading tokenizer decoder to CPU")
            if hasattr(self.model.tokenizer, "decoder") and self.model.tokenizer.decoder is not None:
                self.model.tokenizer.decoder = self.model.tokenizer.decoder.to("cpu")
            torch.cuda.empty_cache()

        if self.offload_text_encoder and self.model.text_encoder is not None:
            log.info("[Memory Optimization] Load text encoder to GPU")
            # TextEncoder is a wrapper class with self.model (the actual neural network)
            if hasattr(self.model.text_encoder, "model") and self.model.text_encoder.model is not None:
                self.model.text_encoder.model = self.model.text_encoder.model.to("cuda")
            torch.cuda.empty_cache()

        return video, kinematic_predictions

    def generate_autoregressive_from_batch(
        self,
        prompt: str,
        input_path: str | torch.Tensor | None,
        num_output_frames: int,
        chunk_size: int,
        chunk_overlap: int,
        guidance: int = 7,
        num_latent_conditional_frames: int = 1,
        resolution: str = "192,320",
        seed: int = 1,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        camera: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        kinematics: torch.Tensor | None = None,
        kinematics_path: str | None = None,
        num_steps: int = 35,
    ) -> torch.Tensor:
        """
        Generate video using autoregressive sliding window approach.

        Args:
            prompt: The text prompt describing the desired video content/style.
            input_path: Path to the input image or video file or a torch.Tensor.
            num_output_frames: Total number of frames to generate in the final output.
            chunk_size: Number of frames per chunk (model's native capacity).
            chunk_overlap: Number of overlapping frames between chunks.
            guidance: Classifier-free guidance scale.
            num_latent_conditional_frames: Number of latent conditional frames.
            resolution: Target video resolution in "H,W" format.
            seed: Random seed for reproducibility.
            negative_prompt: Custom negative prompt.
            camera: Target camera extrinsics and intrinsics for the K output videos.
            action: Target robot action for the K output videos.
            kinematics: Full kinematic sequence [1, T_total, N, 18] at pixel frame rate. Optional.
            kinematics_path: Path to .h5 or .npy file containing kinematics. Optional.
            num_steps: Number of generation steps.

        Returns:
            torch.Tensor: The generated video tensor (B, C, T, H, W) in the range [-1, 1].
        """
        # Parse resolution string into tuple of integers
        if resolution == "none":
            h, w = self.model.get_video_height_width()
            video_resolution = (h, w)
        else:
            video_resolution = resolution.split(",")
            video_resolution = tuple([int(x) for x in video_resolution])
            assert len(video_resolution) == 2, "Resolution must be in 'H,W' format"

        # Get the correct number of frames needed by the model
        model_required_frames = self.model.tokenizer.get_pixel_num_frames(self.model.config.state_t)

        # Load and process the full input video/image
        if input_path is None or num_latent_conditional_frames == 0:
            # For text2world, create a full length zero video
            full_input_video = torch.zeros(1, 3, num_output_frames, video_resolution[0], video_resolution[1]).to(
                torch.uint8
            )
        elif isinstance(input_path, str):
            ext = os.path.splitext(input_path)[1].lower()
            if ext in _IMAGE_EXTENSIONS:
                log.info(f"Processing image input for autoregressive: {input_path}")
                # For image input, create full video with first frame as image, rest zeros
                img = Image.open(input_path)
                img = torchvision.transforms.functional.to_tensor(img)
                img = img.unsqueeze(0)  # Add temporal dimension T=1
                img = (img * 255.0).to(torch.uint8)
                if video_resolution:
                    img = resize_input(img, video_resolution)
                # Create full length video with first frame as image
                full_input_video = torch.cat([img, torch.zeros_like(img).repeat(num_output_frames - 1, 1, 1, 1)], dim=0)
                full_input_video = full_input_video.unsqueeze(0).permute(0, 2, 1, 3, 4)
            elif ext in _VIDEO_EXTENSIONS:
                log.info(f"Processing video input for autoregressive: {input_path}")
                # Load video and extend to full length if needed
                video_frames, _ = easy_io.load(input_path)
                video_tensor = torch.from_numpy(video_frames).float() / 255.0
                video_tensor = video_tensor.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
                available_frames = video_tensor.shape[1]

                # Calculate frames to extract
                frames_to_extract = 4 * (num_latent_conditional_frames - 1) + 1
                if available_frames < frames_to_extract:
                    raise ValueError(f"Video has only {available_frames} frames but needs at least {frames_to_extract}")

                # Extract last frames_to_extract
                start_idx = available_frames - frames_to_extract
                extracted_frames = video_tensor[:, start_idx:, :, :]

                # Create full length tensor
                C, _, H, W = video_tensor.shape
                full_video = torch.zeros(C, num_output_frames, H, W)
                full_video[:, :frames_to_extract, :, :] = extracted_frames

                # Pad with last frame
                if frames_to_extract < num_output_frames:
                    last_frame = extracted_frames[:, -1:, :, :]
                    padding_frames = num_output_frames - frames_to_extract
                    last_frame_repeated = last_frame.repeat(1, padding_frames, 1, 1)
                    full_video[:, frames_to_extract:, :, :] = last_frame_repeated

                full_video = full_video.permute(1, 0, 2, 3)  # (C, T, H, W) -> (T, C, H, W)
                full_video = (full_video * 255.0).to(torch.uint8)
                if video_resolution:
                    full_video = resize_input(full_video, video_resolution)
                full_input_video = full_video.unsqueeze(0).permute(0, 2, 1, 3, 4)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        elif isinstance(input_path, torch.Tensor):
            # If tensor, extend to full length
            full_input_video = input_path
            if full_input_video.shape[2] < num_output_frames:
                # Pad with zeros
                padding_frames = num_output_frames - full_input_video.shape[2]
                padding = torch.zeros(
                    full_input_video.shape[0],
                    full_input_video.shape[1],
                    padding_frames,
                    full_input_video.shape[3],
                    full_input_video.shape[4],
                ).to(full_input_video.dtype)
                full_input_video = torch.cat([full_input_video, padding], dim=2)
        else:
            raise ValueError(f"Unsupported input_path type: {type(input_path)}")

        # Initialize output
        generated_chunks = []

        # Calculate number of chunks
        # Note: All chunks generate chunk_size frames, we store all of chunk 0 and (chunk_size - chunk_overlap) from others
        # Total stored = chunk_size + (num_chunks - 1) * (chunk_size - chunk_overlap) >= num_output_frames
        effective_chunk_size = chunk_size - chunk_overlap

        # Solve for num_chunks: chunk_size + (num_chunks - 1) * effective_chunk_size >= num_output_frames
        remaining_after_first = num_output_frames - chunk_size
        if remaining_after_first <= 0:
            num_chunks = 1
        else:
            # Ceiling division to ensure we have enough frames for the last chunk.
            num_chunks = 1 + (remaining_after_first + effective_chunk_size - 1) // effective_chunk_size

        log.info(
            f"Generating {num_chunks} chunks with chunk_size={chunk_size}, chunk_overlap={chunk_overlap} "
            f"for {num_output_frames} total frames"
        )

        # Load kinematics if path provided
        if kinematics is None and kinematics_path is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            kinematics = self._load_kinematics_from_file(kinematics_path, num_output_frames, device)
            log.info(f"Loaded kinematics from {kinematics_path}: shape {kinematics.shape}")

        # Generate chunks
        current_input_video = full_input_video.clone()
        predicted_kinematics_pixel = None  # Store predicted kinematics at pixel rate for next chunk

        for chunk_idx in range(num_chunks):
            # Calculate frame range for this chunk
            # All chunks are positioned with stride (chunk_size - chunk_overlap)
            start_frame = chunk_idx * effective_chunk_size
            end_frame = min(start_frame + chunk_size, num_output_frames)
            actual_chunk_size = end_frame - start_frame

            if start_frame >= num_output_frames:
                break

            log.info(f"Processing chunk {chunk_idx + 1}/{num_chunks}, frames {start_frame}-{end_frame}")

            # Extract chunk from current input
            chunk_input = current_input_video[:, :, start_frame:end_frame, :, :]

            # Pad to model_required_frames if needed
            if actual_chunk_size < model_required_frames:
                padding_frames = model_required_frames - actual_chunk_size
                padding = torch.zeros(
                    chunk_input.shape[0],
                    chunk_input.shape[1],
                    padding_frames,
                    chunk_input.shape[3],
                    chunk_input.shape[4],
                ).to(chunk_input.dtype)
                chunk_input = torch.cat([chunk_input, padding], dim=2)

            # Determine num_conditional_frames for this chunk
            if chunk_idx == 0:
                chunk_num_conditional = num_latent_conditional_frames
            else:
                chunk_num_conditional = chunk_overlap

            # Determine kinematic conditioning for this chunk
            chunk_kinematics = None
            if chunk_idx == 0:
                # First chunk: Use GT kinematics for conditioned frames only
                if kinematics is not None:
                    num_conditional_pixel_frames = 4 * (num_latent_conditional_frames - 1) + 1
                    chunk_kinematics = kinematics[:, :num_conditional_pixel_frames, :, :]  # [1, T_cond, N, 18]
                    log.info(
                        f"Chunk {chunk_idx + 1}: Using GT kinematics for {num_conditional_pixel_frames} conditioned frames"
                    )
            else:
                # Subsequent chunks: Use predicted kinematics from previous chunk
                if predicted_kinematics_pixel is not None:
                    # Extract overlap region from predicted kinematics
                    overlap_pixel_frames = 4 * (chunk_overlap - 1) + 1
                    chunk_kinematics = predicted_kinematics_pixel[:, -overlap_pixel_frames:, :, :]  # [1, T_overlap, N, 18]
                    log.info(
                        f"Chunk {chunk_idx + 1}: Using predicted kinematics for {overlap_pixel_frames} overlap frames"
                    )
                elif kinematics is not None:
                    # Fallback: Use GT kinematics for overlap if predictions not available
                    overlap_start = start_frame
                    overlap_pixel_frames = 4 * (chunk_overlap - 1) + 1
                    overlap_end = min(overlap_start + overlap_pixel_frames, num_output_frames)
                    chunk_kinematics = kinematics[:, overlap_start:overlap_end, :, :]
                    log.info(
                        f"Chunk {chunk_idx + 1}: Using GT kinematics for overlap (fallback): "
                        f"frames {overlap_start}-{overlap_end}"
                    )

            # Generate chunk with kinematic conditioning
            chunk_video, chunk_kinematic_predictions = self.generate_vid2world(
                prompt=prompt,
                input_path=chunk_input,
                guidance=guidance,
                num_video_frames=model_required_frames,
                num_latent_conditional_frames=chunk_num_conditional,
                resolution=resolution,
                seed=seed + chunk_idx,
                negative_prompt=negative_prompt,
                camera=camera,
                action=action,
                kinematics=chunk_kinematics,
                num_steps=num_steps,
            )  # Returns (video, kinematic_predictions)

            # Extract only the actual generated frames (remove padding)
            chunk_video = chunk_video[:, :, :actual_chunk_size, :, :]

            # Store generated chunk
            if chunk_idx == 0:
                generated_chunks.append(chunk_video)
            else:
                # Remove overlap frames from the beginning
                generated_chunks.append(chunk_video[:, :, chunk_overlap:, :, :])

            # Extract and upsample kinematic predictions for next chunk
            if chunk_kinematic_predictions is not None:
                # Print kinematic predictions for this chunk
                print(f"\n=== Chunk {chunk_idx + 1} Kinematic Predictions ===")
                frame_offset = start_frame  # Offset for frame numbering
                self._print_kinematic_predictions(chunk_kinematic_predictions, frame_offset=frame_offset)
                # Upsample from latent rate to pixel rate
                # chunk_kinematic_predictions is at latent rate [B, T_latent, N, 14]
                # Need to upsample to pixel rate for next chunk conditioning
                actual_chunk_pixel_frames = actual_chunk_size  # actual_chunk_size is already at pixel rate
                predicted_kinematics_pixel = self._upsample_kinematics_to_pixel_rate(
                    chunk_kinematic_predictions,
                    T_pixel_target=actual_chunk_pixel_frames,
                )
                log.info(
                    f"Chunk {chunk_idx + 1}: Extracted kinematic predictions "
                    f"(latent: {chunk_kinematic_predictions['position'].shape[1]}, "
                    f"pixel: {predicted_kinematics_pixel.shape[1]})"
                )
            else:
                predicted_kinematics_pixel = None

            # Update input for next iteration using generated frames
            if chunk_idx < num_chunks - 1:
                # Convert generated chunk from [-1, 1] to [0, 255] uint8 range
                chunk_video_uint8 = ((chunk_video / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
                # Update the input video with generated frames for conditioning next chunk
                update_start = start_frame + chunk_num_conditional
                update_end = end_frame
                current_input_video[:, :, update_start:update_end, :, :] = chunk_video_uint8[
                    :, :, chunk_num_conditional:, :, :
                ]

        # Concatenate all chunks along time dimension
        final_video = torch.cat(generated_chunks, dim=2)

        log.info(f"Generated final video with shape {final_video.shape}")
        return final_video

    def cleanup(self):
        """Clean up distributed resources."""
        if self.context_parallel_size > 1:
            import torch.distributed as dist
            from megatron.core import parallel_state

            if parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            dist.destroy_process_group()
