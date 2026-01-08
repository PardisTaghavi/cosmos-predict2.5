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
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pynvml
import torch

from cosmos_predict2._src.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets
from cosmos_predict2._src.imaginaire.flags import SMOKE
from cosmos_predict2._src.imaginaire.lazy_config.lazy import LazyConfig
from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from cosmos_predict2.config import InferenceArguments, SetupArguments, path_to_str


def get_gpu_stats() -> dict[str, float]:
    """Collect GPU and CPU statistics similar to DeviceMonitor callback."""
    stats = {}
    
    # CPU memory
    try:
        cur_process = psutil.Process(os.getpid())
        cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
        stats["cpu_mem_gb"] = cpu_memory_usage / (1024**3)
    except Exception:
        stats["cpu_mem_gb"] = 0.0
    
    if torch.cuda.is_available():
        # GPU memory stats
        stats["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        stats["peak_gpu_mem_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024**3)
        
        # GPU temperature, power, utilization, clock
        try:
            stats["temp"] = torch.cuda.temperature()
        except Exception:
            stats["temp"] = 0.0
        
        try:
            stats["power"] = torch.cuda.power_draw()
        except Exception:
            stats["power"] = 0.0
        
        try:
            stats["util"] = torch.cuda.utilization()
        except Exception:
            stats["util"] = 0.0
        
        try:
            stats["clock"] = torch.cuda.clock_rate()
        except Exception:
            stats["clock"] = 0.0
        
        # NVML stats
        try:
            pynvml.nvmlInit()
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            stats["nvml_used_gpu_mem_gb"] = memory_info.used / (1024**3)
            stats["nvml_free_gpu_mem_gb"] = memory_info.free / (1024**3)
        except Exception:
            stats["nvml_used_gpu_mem_gb"] = 0.0
            stats["nvml_free_gpu_mem_gb"] = 0.0
    else:
        # Set defaults if CUDA not available
        stats.update({
            "peak_gpu_mem_gb": 0.0,
            "peak_gpu_mem_reserved_gb": 0.0,
            "temp": 0.0,
            "power": 0.0,
            "util": 0.0,
            "clock": 0.0,
            "nvml_used_gpu_mem_gb": 0.0,
            "nvml_free_gpu_mem_gb": 0.0,
        })
    
    return stats


def print_gpu_stats_summary(stats: dict[str, float], label: str = "Inference"):
    """Print GPU stats in a formatted table similar to training output."""
    if not torch.cuda.is_available():
        log.info(f"{label} GPU Stats: CUDA not available")
        return
    
    # Create summary DataFrame with same format as training (Avg/Max/Min columns)
    metric_order = [
        "cpu_mem_gb",
        "peak_gpu_mem_gb",
        "peak_gpu_mem_reserved_gb",
        "nvml_used_gpu_mem_gb",
        "nvml_free_gpu_mem_gb",
        "temp",
        "power",
        "util",
        "clock",
    ]
    
    summary_data = {"Avg": [], "Max": [], "Min": []}
    metric_names = []
    
    for metric in metric_order:
        if metric in stats:
            value = stats[metric]
            summary_data["Avg"].append(value)
            summary_data["Max"].append(value)
            summary_data["Min"].append(value)
            metric_names.append(metric)
    
    if metric_names:
        df = pd.DataFrame(summary_data, index=metric_names)
        log.info(f"{label} GPU Stats:\n{df.to_string()}")


class Inference:
    def __init__(self, args: SetupArguments):
        log.debug(f"{args.__class__.__name__}({args})")

        torch.enable_grad(False)  # Disable gradient calculations for inference

        self.rank0 = distributed.is_rank0()
        self.setup_args = args
        experiment_opts = []
        if args.model_key.distilled:
            experiment_opts.append("model.config.init_student_with_teacher=False")
        self.pipe = Video2WorldInference(
            # pyrefly: ignore  # bad-argument-type
            experiment_name=args.experiment,
            # pyrefly: ignore  # bad-argument-type
            ckpt_path=args.checkpoint_path,
            s3_credential_path="",
            # pyrefly: ignore  # bad-argument-type
            context_parallel_size=args.context_parallel_size,
            config_file=args.config_file,
            experiment_opts=experiment_opts,
            offload_diffusion_model=args.offload_diffusion_model,
            offload_text_encoder=args.offload_text_encoder,
            offload_tokenizer=args.offload_tokenizer,
        )
        if self.rank0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            config_path = args.output_dir / "config.yaml"
            # pyrefly: ignore  # bad-argument-type
            LazyConfig.save_yaml(self.pipe.config, config_path)
            log.info(f"Saved config to {config_path}")

        self.guardrail_enabled = not args.disable_guardrails

        if self.rank0 and self.guardrail_enabled:
            self.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
            self.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
        else:
            # pyrefly: ignore  # bad-assignment
            self.text_guardrail_runner = None
            # pyrefly: ignore  # bad-assignment
            self.video_guardrail_runner = None

    def generate(self, samples: list[InferenceArguments], output_dir: Path) -> list[str]:
        if SMOKE:
            samples = samples[:1]

        sample_names = [sample.name for sample in samples]
        log.info(f"Generating {len(samples)} samples: {sample_names}")

        output_paths: list[str] = []
        for i_sample, sample in enumerate(samples):
            log.info(f"[{i_sample + 1}/{len(samples)}] Processing sample {sample.name}")
            output_path = self._generate_sample(sample, output_dir)
            if output_path is not None:
                output_paths.append(output_path)
        return output_paths

    def _generate_sample(self, sample: InferenceArguments, output_dir: Path) -> str | None:
        log.debug(f"{sample.__class__.__name__}({sample})")
        output_path = output_dir / sample.name

        if self.rank0:
            output_dir.mkdir(parents=True, exist_ok=True)
            open(f"{output_path}.json", "w").write(sample.model_dump_json())
            log.info(f"Saved arguments to {output_path}.json")

            # run text guardrail on the prompt
            if self.text_guardrail_runner is not None:
                if not guardrail_presets.run_text_guardrail(sample.prompt, self.text_guardrail_runner):
                    message = f"Guardrail blocked text2world generation. Prompt: {sample.prompt}"
                    log.critical(message)
                    if self.setup_args.keep_going:
                        return None
                    else:
                        raise Exception(message)
                else:
                    log.success("Passed guardrail on prompt")
            elif self.text_guardrail_runner is None:
                log.warning("Guardrail checks on prompt are disabled")

        # Reset peak memory stats before generation
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Choose generation mode based on autoregressive flag
        video: torch.Tensor
        if sample.enable_autoregressive:
            log.info(f"Generating video with autoregressive mode...")
            result = self.pipe.generate_autoregressive_from_batch(
                prompt=sample.prompt,
                input_path=path_to_str(sample.input_path),
                num_output_frames=sample.num_output_frames,
                chunk_size=sample.chunk_size,
                chunk_overlap=sample.chunk_overlap,
                guidance=sample.guidance,
                num_latent_conditional_frames=sample.num_input_frames,
                resolution=sample.resolution,
                seed=sample.seed,
                negative_prompt=sample.negative_prompt,
                kinematics_path=path_to_str(sample.kinematics_path) if sample.kinematics_path else None,
                num_steps=sample.num_steps,
            )
            # Handle return value: autoregressive returns video only
            video = result
        else:
            log.info(f"Generating video with standard mode...")
            result = self.pipe.generate_vid2world(
                prompt=sample.prompt,
                input_path=path_to_str(sample.input_path),
                guidance=sample.guidance,
                num_video_frames=sample.num_output_frames,
                num_latent_conditional_frames=sample.num_input_frames,
                resolution=sample.resolution,
                seed=sample.seed,
                negative_prompt=sample.negative_prompt,
                kinematics_path=path_to_str(sample.kinematics_path) if sample.kinematics_path else None,
                num_steps=sample.num_steps,
            )
            # Handle return value: could be (video,) or (video, kinematic_predictions)
            if isinstance(result, tuple):
                video = result[0]
            else:
                video = result
        
        # Print GPU stats after generation
        if self.rank0:
            stats = get_gpu_stats()
            print_gpu_stats_summary(stats, label=f"Inference [{sample.name}]")

        if self.rank0:
            video = (1.0 + video[0]) / 2

            # run video guardrail on the video
            if self.video_guardrail_runner is not None:
                log.info("Running guardrail check on video...")
                frames = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8)
                frames = frames.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)  # (T, H, W, C)
                processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
                if processed_frames is None:
                    message = "Guardrail blocked video2world generation."
                    log.critical(message)
                    if self.setup_args.keep_going:
                        return None
                    else:
                        raise Exception(message)
                else:
                    log.success("Passed guardrail on generated video")
                # Convert processed frames back to tensor format
                processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0
                video = processed_video.to(video.device, dtype=video.dtype)
            else:
                log.warning("Guardrail checks on video are disabled")

            save_img_or_video(video, str(output_path), fps=2)
            log.success(f"Saved video to {output_path}.mp4")
        return f"{output_path}.mp4"
