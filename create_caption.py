# Create videos from CAM_FRONT images and generate captions using QwenVL
import os
import json
import cv2
import imageio
import sys

# Unset VLLM_ATTENTION_BACKEND to allow automatic backend selection
# This enables FLASH_ATTN for text (head_dim=128) and XFORMERS for vision (head_dim=72)
if 'VLLM_ATTENTION_BACKEND' in os.environ:
    del os.environ['VLLM_ATTENTION_BACKEND']

# Initialize CUDA before any vLLM imports
import torch
assert torch.cuda.is_available(), "CUDA not available. Ensure you're running in a GPU allocation."
_ = torch.cuda.current_device()  # Force CUDA initialization

# Paths
path = "/scratch/user/u.pt152369/WM/data/datasetWM"
nuscenes_samples_root = "/scratch/user/u.pt152369/WM/data/v1.0-mini"
nuscenes_root = "/scratch/user/u.pt152369/WM/data/v1.0-mini/v1.0-mini"

video_path = os.path.join(path, "videos")
caption_path = os.path.join(path, "meta")

os.makedirs(video_path, exist_ok=True)
os.makedirs(caption_path, exist_ok=True)

fps = 2.0  # NuScenes dataset is 2 Hz

# Load NuScenes data
print("Loading NuScenes data...")
with open(os.path.join(nuscenes_root, "scene.json"), "r") as f:
    scenes = json.load(f)

with open(os.path.join(nuscenes_root, "sample.json"), "r") as f:
    samples = json.load(f)

with open(os.path.join(nuscenes_root, "sample_data.json"), "r") as f:
    sample_data = json.load(f)

# Create lookup dictionaries
scene_dict = {scene["token"]: scene for scene in scenes}
sample_dict = {sample["token"]: sample for sample in samples}
sample_data_dict = {sd["token"]: sd for sd in sample_data}

# Create mapping from sample_token to CAM_FRONT sample_data (key frames)
# Match exact "__CAM_FRONT__" pattern to exclude CAM_FRONT_LEFT, CAM_FRONT_RIGHT, etc.
cam_front_samples = [
    sd for sd in sample_data
    if "__CAM_FRONT__" in sd.get("filename", "") and sd.get("is_key_frame", False)
]

sample_to_cam_front_data = {}
for sd in cam_front_samples:
    sample_token = sd["sample_token"]
    sample_to_cam_front_data[sample_token] = sd

print(f"Found {len(scenes)} scenes")


def create_video_from_images(image_paths, output_path, fps=2.0, verbose=False):
    """Create video from list of image paths."""
    images = []
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)
        
        if verbose and (idx + 1) % 10 == 0:
            print(f"  Loaded {idx + 1}/{len(image_paths)} images")
    
    height, width, _ = images[0].shape
    
    if verbose:
        print(f"  Writing video: {output_path}")
        print(f"    FPS: {fps}, Resolution: {width}x{height}, Frames: {len(images)}")
    
    # Set macro_block_size to None to prevent automatic resizing
    imageio.mimwrite(output_path, images, fps=fps, codec='libx264', macro_block_size=None)
    
    if verbose:
        print(f"  ✅ Video saved: {output_path}")


# Global model instance (initialized once)
_qwen25vl_model = None

def get_qwen25vl_model():
    """Get or initialize Qwen2.5-VL-72B model (singleton pattern)."""
    global _qwen25vl_model
    if _qwen25vl_model is None:
        import transformers
        import vllm
        import logging
        import warnings
        
        # Suppress warnings and logging
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
        warnings.filterwarnings("ignore", message=".*torchvision.*")
        logging.getLogger("vllm").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        
        # Use FP8 quantized model from Neural Magic (optimized for 2x H100)
        model_name = "neuralmagic/Qwen2.5-VL-72B-Instruct-FP8-Dynamic"
        
        # Check GPU availability
        num_gpus = torch.cuda.device_count()
        print(f"  Detected {num_gpus} GPU(s)")
        if num_gpus < 2:
            raise RuntimeError(f"Need 2 GPUs for tensor parallelism, but only {num_gpus} GPU(s) detected. "
                             f"Set CUDA_VISIBLE_DEVICES=0,1 to use 2 GPUs.")
        
        print("  Initializing Qwen2.5-VL-72B (FP8) on 2x H100...")
        
        # Clear any existing CUDA cache
        torch.cuda.empty_cache()
        
        # Initialize vLLM with FP8 quantized model
        llm = vllm.LLM(
            model=model_name,
            tensor_parallel_size=2,
            max_model_len=16384,
            gpu_memory_utilization=0.75,
            max_num_seqs=256,
            limit_mm_per_prompt={"video": 1},
            trust_remote_code=True,
            disable_custom_all_reduce=True,
        )
        
        # Initialize processor (use base model name for processor)
        processor_model_name = "Qwen/Qwen2.5-VL-72B-Instruct"  # Base model for processor
        processor = transformers.AutoProcessor.from_pretrained(
            processor_model_name,
            use_fast=True,
            trust_remote_code=True,
        )
        
        # Sampling parameters
        sampling_params = vllm.SamplingParams(
            max_tokens=4096,
            n=1,
            temperature=0.01,
            seed=42,
        )
        
        # Store model components
        _qwen25vl_model = {
            "llm": llm,
            "processor": processor,
            "sampling_params": sampling_params,
        }
        
        print("  ✅ Qwen2.5-VL-72B model loaded successfully")
    return _qwen25vl_model

# Old 2Hz version - commented out, using 10Hz version below
'''
frequency = 2.0  # Hz
##########
# 2.0 version : 2Hz captioning with 5 seconds per clip
##########
def caption_video_with_qwen25vl(video_path):
    """Caption a video using Qwen2.5-VL-72B."""
    import qwen_vl_utils
    import contextlib
    import io
    
    model_dict = get_qwen25vl_model()
    llm = model_dict["llm"]
    processor = model_dict["processor"]
    sampling_params = model_dict["sampling_params"]
    
    prompt = """1. Agents & Traffic Elements: Identify all vehicles, pedestrians, and cyclists. Note traffic lights (including their state), signs, and road markings the ego-vehicle must account for. 2. Environmental Factors: Describe the weather (e.g., clear, foggy, snowy), time of day (daytime vs. nighttime), and road conditions (e.g., dry, wet, urban, rural, tunnel). 3. Ego-Vehicle Dynamics: Describe the ego-vehicle's speed qualitatively using descriptive terms (e.g., stationary, slow, moderate, fast, accelerating, decelerating, constant speed) - DO NOT provide numerical speed values. Describe maneuvers (e.g., sharp turn, slow curve, highway cruising, lane change). 4. Surrounding Dynamics: Describe the longitudinal and lateral meta-actions of surrounding vehicles, including their relative speeds and any state transitions. 5. Interactions: Detail the interactions between key objects and the ego-vehicle."""
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": 2, "total_pixels": 6422528},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    
    prompt_text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    
    image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
        conversation, return_video_kwargs=True
    )
    
    mm_data = {}
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    
    llm_inputs = {
        "prompt": prompt_text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }
    
    with contextlib.redirect_stderr(io.StringIO()):
        outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    
    caption = outputs[0].outputs[0].text
    return caption.strip()


if __name__ == '__main__':
    # Process each scene
    for scene in scenes:
        scene_name = scene["name"]
        scene_token = scene["token"]
        first_sample_token = scene["first_sample_token"]
        
        print(f"\n{'='*70}")
        print(f"Processing Scene: {scene_name}")
        print(f"{'='*70}")
        
        # Extract scene number (e.g., "scene-0061" -> "0061")
        scene_number = scene_name.split('-')[-1]
        video_filename = f"{scene_number}.mp4"
        caption_filename = f"{scene_number}.json"
        
        video_output_path = os.path.join(video_path, video_filename)
        caption_output_path = os.path.join(caption_path, caption_filename)
        
        # Collect CAM_FRONT images for this scene
        image_paths = []
        current_sample_token = first_sample_token
        
        while current_sample_token:
            sample = sample_dict[current_sample_token]
            
            # Get CAM_FRONT sample_data
            cam_front_sd = sample_to_cam_front_data[current_sample_token]
            
            # Get image filename
            filename = cam_front_sd["filename"]
            
            # Construct full image path (filename already includes subdirectory)
            image_path = os.path.join(nuscenes_samples_root, filename)
            image_paths.append(image_path)
            
            # Move to next sample
            current_sample_token = sample["next"]
        
        total_frames = len(image_paths)
        print(f"  Found {total_frames} images for scene {scene_name}")
        
        # Create full video
        print(f"  Creating full video...")
        create_video_from_images(image_paths, video_output_path, fps=fps, verbose=True)
        
        # Split into 4 clips (5 seconds each, ~20 seconds total)
        num_clips = 4
        frames_per_clip = total_frames // num_clips
        
        print(f"  Splitting into {num_clips} clips (~{frames_per_clip} frames per clip)")
        
        # Generate captions for each clip
        captions = []
        temp_clip_paths = []
        
        for clip_idx in range(num_clips):
            start_idx = clip_idx * frames_per_clip
            # For the last clip, include all remaining frames
            end_idx = total_frames if clip_idx == num_clips - 1 else (clip_idx + 1) * frames_per_clip
            
            clip_images = image_paths[start_idx:end_idx]
            
            if len(clip_images) == 0:
                print(f"    Warning: Clip {clip_idx + 1} has no frames, skipping")
                captions.append("")
                continue
            
            # Create temporary video file for this clip
            temp_clip_path = os.path.join(video_path, f"{scene_number}_clip_{clip_idx + 1}.mp4")
            temp_clip_paths.append(temp_clip_path)
            
            print(f"    Creating clip {clip_idx + 1}/{num_clips} ({len(clip_images)} frames)...")
            create_video_from_images(clip_images, temp_clip_path, fps=fps, verbose=False)
            
            # Generate caption for this clip
            print(f"    Generating caption for clip {clip_idx + 1}/{num_clips}...")
            caption = caption_video_with_qwen25vl(temp_clip_path)
            captions.append(caption)
            print(f"    ✅ Caption {clip_idx + 1}:")
            print(f"    {'-'*60}")
            print(f"    {caption}")
            print(f"    {'-'*60}")
        
        # Save all captions as JSON
        caption_data = {
            "scene_name": scene_name,
            "scene_number": scene_number,
            "total_frames": total_frames,
            "num_clips": num_clips,
            "fps": fps,
            "captions": captions
        }
        
        with open(caption_output_path, "w") as f:
            json.dump(caption_data, f, indent=2)
        
        print(f"  ✅ Captions saved: {caption_output_path}")
        
        # Clean up temporary clip files
        for temp_path in temp_clip_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    print(f"\n{'='*70}")
    print("✅ Processing complete!")
    print(f"{'='*70}")
'''



##########
# 10.0 version : 10Hz captioning, one clip around 4 seconds
##########

output_video_folder = "/scratch/user/u.pt152369/WM/data/datasetWM/videos_10Hz"
output_caption_folder = "/scratch/user/u.pt152369/WM/data/datasetWM/meta_10Hz"

frequency = 10.0  # Hz

os.makedirs(output_video_folder, exist_ok=True)
os.makedirs(output_caption_folder, exist_ok=True)

def strip_symbols(text):
    """Remove symbols like #%/, etc. from text."""
    import re
    return re.sub(r'[#%/\-]+', ' ', text)

def caption_video_with_qwen25vl(video_path):
    """Caption a video using Qwen2.5-VL-72B."""
    import qwen_vl_utils
    import contextlib
    import io
    
    model_dict = get_qwen25vl_model()
    llm = model_dict["llm"]
    processor = model_dict["processor"]
    sampling_params = model_dict["sampling_params"]
    
    prompt = """1. Agents & Traffic Elements: Identify all vehicles, pedestrians, and cyclists. Note traffic lights (including their state), signs, and road markings the ego-vehicle must account for. 2. Environmental Factors: Describe the weather (e.g., clear, foggy, snowy), time of day (daytime vs. nighttime), and road conditions (e.g., dry, wet, urban, rural, tunnel). 3. Ego-Vehicle Dynamics: Describe the ego-vehicle's speed qualitatively using descriptive terms (e.g., stationary, slow, moderate, fast, accelerating, decelerating, constant speed) - DO NOT provide numerical speed values. Describe maneuvers (e.g., sharp turn, slow curve, highway cruising, lane change). 4. Surrounding Dynamics: Describe the longitudinal and lateral meta-actions of surrounding vehicles, including their relative speeds and any state transitions. 5. Interactions: Detail the interactions between key objects and the ego-vehicle."""
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": int(frequency), "total_pixels": 6422528},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    
    prompt_text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    
    image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
        conversation, return_video_kwargs=True
    )
    
    mm_data = {}
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    
    llm_inputs = {
        "prompt": prompt_text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }
    
    with contextlib.redirect_stderr(io.StringIO()):
        outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    
    caption = outputs[0].outputs[0].text
    return caption.strip()

if __name__ == '__main__':
    for scene in scenes:
        scene_name = scene["name"]
        scene_token = scene["token"]
        first_sample_token = scene["first_sample_token"]
        
        print(f"\n{'='*70}")
        print(f"Processing Scene: {scene_name}")
        print(f"{'='*70}")
        
        scene_number = scene_name.split('-')[-1]
        video_filename = f"{scene_number}.mp4"
        caption_filename = f"{scene_number}.txt"
        
        video_output_path = os.path.join(output_video_folder, video_filename)
        caption_output_path = os.path.join(output_caption_folder, caption_filename)
        
        image_paths = []
        current_sample_token = first_sample_token
        
        while current_sample_token:
            sample = sample_dict[current_sample_token]
            cam_front_sd = sample_to_cam_front_data[current_sample_token]
            filename = cam_front_sd["filename"]
            image_path = os.path.join(nuscenes_samples_root, filename)
            image_paths.append(image_path)
            current_sample_token = sample["next"]
        
        total_frames = len(image_paths)
        video_duration = total_frames / frequency
        
        print(f"  Found {total_frames} images for scene {scene_name}")
        print(f"  Video duration: {video_duration:.2f} seconds ({total_frames} frames at {frequency} Hz)")
        
        print(f"  Creating 10Hz video...")
        create_video_from_images(image_paths, video_output_path, fps=frequency, verbose=True)
        
        print(f"  Generating caption...")
        caption = caption_video_with_qwen25vl(video_output_path)
        caption_cleaned = strip_symbols(caption)
        
        print(f"  ✅ Caption:")
        print(f"  {'-'*60}")
        print(f"  {caption_cleaned}")
        print(f"  {'-'*60}")
        
        with open(caption_output_path, "w") as f:
            f.write(caption_cleaned)
        
        print(f"  ✅ Video saved: {video_output_path}")
        print(f"  ✅ Caption saved: {caption_output_path}")

    print(f"\n{'='*70}")
    print("✅ Processing complete!")
    print(f"{'='*70}")


