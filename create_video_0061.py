# Extract video for scene-0061 from CAM_FRONT images

import os
import json
import numpy as np
import imageio
from PIL import Image

# Paths
path = "/Users/ptgh/Desktop/code/cosmos-predict2.5/data"
nuscenes_root = os.path.join(path, "v1.0-mini", "v1.0-mini")
cam_front_dir = os.path.join(path, "v1.0-mini", "samples", "CAM_FRONT")
output_video_dir = os.path.join(path, "videos")
os.makedirs(output_video_dir, exist_ok=True)

# Load scene data
with open(os.path.join(nuscenes_root, "scene.json"), "r") as f:
    scenes = json.load(f)

with open(os.path.join(nuscenes_root, "sample.json"), "r") as f:
    samples = json.load(f)

with open(os.path.join(nuscenes_root, "sample_data.json"), "r") as f:
    sample_data = json.load(f)

# Find scene-0061
scene_0061 = None
for scene in scenes:
    if scene["name"] == "scene-0061":
        scene_0061 = scene
        break

assert scene_0061 is not None, "Scene-0061 not found"

print(f"Found scene: {scene_0061['name']}")
print(f"Number of samples: {scene_0061['nbr_samples']}")

# Create lookup dictionaries
sample_dict = {s["token"]: s for s in samples}
sample_data_dict = {sd["token"]: sd for sd in sample_data}

# Collect CAM_FRONT images for this scene
first_sample_token = scene_0061["first_sample_token"]
current_sample_token = first_sample_token
image_paths = []
frame_count = 0

print("\nCollecting images...")
while current_sample_token:
    sample = sample_dict[current_sample_token]
    
    # Find CAM_FRONT sample_data for this sample
    cam_front_sd = None
    for sd_token, sd in sample_data_dict.items():
        if (sd["sample_token"] == current_sample_token and 
            "CAM_FRONT" in sd.get("filename", "") and 
            sd.get("is_key_frame", False)):
            cam_front_sd = sd
            break
    
    if cam_front_sd:
        filename = cam_front_sd["filename"]
        # Remove 'samples/' prefix if present
        if filename.startswith("samples/"):
            filename = filename[len("samples/"):]
        elif filename.startswith("sweeps/"):
            # Skip sweeps, only use key frames
            current_sample_token = sample["next"]
            continue
        
        image_path = os.path.join(cam_front_dir, os.path.basename(filename))
        if os.path.exists(image_path):
            image_paths.append(image_path)
            frame_count += 1
        else:
            print(f"  Warning: Image not found: {image_path}")
    
    current_sample_token = sample["next"]
    
    if frame_count >= 100:
        break

print(f"Collected {len(image_paths)} images")

# Read first image to get dimensions
if len(image_paths) == 0:
    print("No images found!")
    exit(1)

first_image = Image.open(image_paths[0])
width, height = first_image.size
print(f"Image dimensions: {width}x{height}")

# Create video using imageio
output_path = os.path.join(output_video_dir, "0061.mp4")
fps = 2.0  # NuScenes is 2 Hz

print(f"\nWriting video to: {output_path}")
print(f"FPS: {fps}")
print(f"Resolution: {width}x{height}")

# Read all images
images = []
for idx, image_path in enumerate(image_paths):
    img = Image.open(image_path)
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Convert to numpy array
    img_array = np.array(img)
    images.append(img_array)
    
    if (idx + 1) % 10 == 0:
        print(f"  Loaded {idx + 1}/{len(image_paths)} images")

# Write video using imageio with ffmpeg codec
print("Writing video file...")
imageio.mimwrite(output_path, images, fps=fps, codec='libx264', quality=8)

print(f"\n✅ Video saved successfully: {output_path}")
print(f"   Total frames: {len(image_paths)}")
print(f"   Duration: {len(image_paths) / fps:.1f} seconds")

