# Visualization script for kinematic trajectories
# Validates x,y,z trajectories of objects over frames
# Creates static plots and animated BEV video

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict
import imageio
from PIL import Image

# Paths
path = "/Users/ptgh/Desktop/code/cosmos-predict2.5/data"
nuscenes_root = os.path.join(path, "v1.0-mini", "v1.0-mini")
output_dir = "/Users/ptgh/Desktop/code/cosmos-predict2.5"
os.makedirs(output_dir, exist_ok=True)

# Load NuScenes JSON files
with open(os.path.join(nuscenes_root, "scene.json"), "r") as f:
    scenes = json.load(f)

with open(os.path.join(nuscenes_root, "sample.json"), "r") as f:
    samples = json.load(f)

with open(os.path.join(nuscenes_root, "sample_annotation.json"), "r") as f:
    annotations = json.load(f)

with open(os.path.join(nuscenes_root, "ego_pose.json"), "r") as f:
    ego_poses = json.load(f)

# Create lookup dictionaries
sample_dict = {s["token"]: s for s in samples}
annotation_dict = {a["token"]: a for a in annotations}
ego_pose_dict = {ep["token"]: ep for ep in ego_poses}

# Build sample -> annotations mapping
sample_to_annotations = {}
for ann in annotations:
    sample_token = ann["sample_token"]
    if sample_token not in sample_to_annotations:
        sample_to_annotations[sample_token] = []
    sample_to_annotations[sample_token].append(ann)

# Build sample -> ego_pose mapping (using CAM_FRONT key frames)
with open(os.path.join(nuscenes_root, "sample_data.json"), "r") as f:
    sample_data = json.load(f)

cam_front_samples = [
    sd for sd in sample_data
    if "CAM_FRONT" in sd.get("filename", "") and sd.get("is_key_frame", False)
]

sample_to_ego_pose = {}
for sd in cam_front_samples:
    sample_token = sd["sample_token"]
    ego_pose_token = sd["ego_pose_token"]
    sample_to_ego_pose[sample_token] = ego_pose_token

# Load category mapping for object types
with open(os.path.join(nuscenes_root, "category.json"), "r") as f:
    categories = json.load(f)
category_dict = {cat["token"]: cat["name"] for cat in categories}

# Load instance -> category mapping
with open(os.path.join(nuscenes_root, "instance.json"), "r") as f:
    instances = json.load(f)
instance_to_category = {inst["token"]: inst["category_token"] for inst in instances}

VISIBILITY_THRESHOLD_TOKEN = "4"  # v80-100%

# Extract trajectories for scene-0061
scene_name_target = "scene-0061"
scene = None
for s in scenes:
    if s["name"] == scene_name_target:
        scene = s
        break

if scene is None:
    scene = scenes[0]  # Fallback to first scene
    print(f"Warning: {scene_name_target} not found, using {scene['name']}")

scene_name = scene["name"]
first_sample_token = scene["first_sample_token"]

print(f"Extracting trajectories for scene: {scene_name}")

# Track trajectories: instance_token -> list of (frame_idx, x, y, z)
trajectories = defaultdict(list)
ego_trajectory = []

# Follow sample chain
current_sample_token = first_sample_token
frame_idx = 0

while current_sample_token:
    sample = sample_dict[current_sample_token]
    
    # Get ego pose
    ego_pose_token = sample_to_ego_pose.get(current_sample_token)
    if ego_pose_token:
        ego_pose = ego_pose_dict[ego_pose_token]
        ego_translation = ego_pose["translation"]
        ego_trajectory.append((frame_idx, ego_translation[0], ego_translation[1], ego_translation[2]))
    
    # Get visible annotations
    sample_annotations = sample_to_annotations.get(current_sample_token, [])
    visible_annotations = [
        ann for ann in sample_annotations
        if ann["visibility_token"] == VISIBILITY_THRESHOLD_TOKEN
    ]
    
    # Store trajectories by instance_token (for tracking)
    for ann in visible_annotations:
        instance_token = ann["instance_token"]
        translation = ann["translation"]
        trajectories[instance_token].append((frame_idx, translation[0], translation[1], translation[2]))
    
    current_sample_token = sample.get("next")
    frame_idx += 1
    
    # Limit to reasonable number of frames
    if frame_idx >= 100:
        break

print(f"Extracted {len(ego_trajectory)} frames")
print(f"Found {len(trajectories)} unique object instances")

# Convert to numpy arrays for plotting
if ego_trajectory:
    ego_frames, ego_x, ego_y, ego_z = zip(*ego_trajectory)
    ego_frames = np.array(ego_frames)
    ego_x = np.array(ego_x)
    ego_y = np.array(ego_y)
    ego_z = np.array(ego_z)
else:
    ego_frames = np.array([])
    ego_x = np.array([])
    ego_y = np.array([])
    ego_z = np.array([])

# Prepare object trajectories
object_trajectories = {}
for instance_token, traj in trajectories.items():
    if len(traj) > 1:  # Only objects with at least 2 frames
        frames, x, y, z = zip(*traj)
        object_trajectories[instance_token] = {
            "frames": np.array(frames),
            "x": np.array(x),
            "y": np.array(y),
            "z": np.array(z),
            "category": category_dict.get(instance_to_category.get(instance_token, ""), "unknown")
        }

print(f"Objects with valid trajectories (>1 frame): {len(object_trajectories)}")

# ============================================================================
# 1. CREATE STATIC PLOTS
# ============================================================================
print("\n" + "="*70)
print("Creating static plots...")
print("="*70)

fig = plt.figure(figsize=(16, 10))

# Plot 1: Top-down view (X-Y plane)
ax1 = fig.add_subplot(2, 2, 1)
if len(ego_x) > 0:
    ax1.plot(ego_x, ego_y, 'r-', linewidth=3, label='Ego Vehicle', alpha=0.7)
    ax1.scatter(ego_x[0], ego_y[0], c='red', s=100, marker='o', zorder=5, label='Start')
    ax1.scatter(ego_x[-1], ego_y[-1], c='darkred', s=100, marker='s', zorder=5, label='End')

# Plot object trajectories
colors = plt.cm.tab20(np.linspace(0, 1, len(object_trajectories)))
for idx, (instance_token, traj) in enumerate(object_trajectories.items()):
    ax1.plot(traj["x"], traj["y"], '-', color=colors[idx], alpha=0.5, linewidth=1)
    ax1.scatter(traj["x"][0], traj["y"][0], color=colors[idx], s=30, marker='o', alpha=0.7)
    ax1.scatter(traj["x"][-1], traj["y"][-1], color=colors[idx], s=30, marker='s', alpha=0.7)

ax1.set_xlabel('X (meters)', fontsize=12)
ax1.set_ylabel('Y (meters)', fontsize=12)
ax1.set_title('Top-Down View: X-Y Trajectories', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')
ax1.set_aspect('equal', adjustable='box')

# Plot 2: Height over time (Z vs Frame)
ax2 = fig.add_subplot(2, 2, 2)
if len(ego_z) > 0:
    ax2.plot(ego_frames, ego_z, 'r-', linewidth=3, label='Ego Vehicle', alpha=0.7)

for idx, (instance_token, traj) in enumerate(object_trajectories.items()):
    ax2.plot(traj["frames"], traj["z"], '-', color=colors[idx], alpha=0.5, linewidth=1)

ax2.set_xlabel('Frame Index', fontsize=12)
ax2.set_ylabel('Z (meters)', fontsize=12)
ax2.set_title('Height Over Time', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')

# Plot 3: 3D trajectory view
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
if len(ego_x) > 0:
    ax3.plot(ego_x, ego_y, ego_z, 'r-', linewidth=3, label='Ego Vehicle', alpha=0.7)
    ax3.scatter(ego_x[0], ego_y[0], ego_z[0], c='red', s=100, marker='o', zorder=5)
    ax3.scatter(ego_x[-1], ego_y[-1], ego_z[-1], c='darkred', s=100, marker='s', zorder=5)

for idx, (instance_token, traj) in enumerate(object_trajectories.items()):
    ax3.plot(traj["x"], traj["y"], traj["z"], '-', color=colors[idx], alpha=0.5, linewidth=1)

ax3.set_xlabel('X (meters)', fontsize=12)
ax3.set_ylabel('Y (meters)', fontsize=12)
ax3.set_zlabel('Z (meters)', fontsize=12)
ax3.set_title('3D Trajectories', fontsize=14, fontweight='bold')

# Plot 4: Distance from ego over time
ax4 = fig.add_subplot(2, 2, 4)

# Calculate distance from ego for each object
for idx, (instance_token, traj) in enumerate(object_trajectories.items()):
    distances = []
    valid_frames = []
    
    for i, frame_idx in enumerate(traj["frames"]):
        if frame_idx < len(ego_x):
            obj_x = traj["x"][i]
            obj_y = traj["y"][i]
            obj_z = traj["z"][i]
            
            ego_x_at_frame = ego_x[frame_idx]
            ego_y_at_frame = ego_y[frame_idx]
            ego_z_at_frame = ego_z[frame_idx]
            
            dist = np.sqrt((obj_x - ego_x_at_frame)**2 + 
                          (obj_y - ego_y_at_frame)**2 + 
                          (obj_z - ego_z_at_frame)**2)
            distances.append(dist)
            valid_frames.append(frame_idx)
    
    if distances:
        ax4.plot(valid_frames, distances, '-', color=colors[idx], alpha=0.5, linewidth=1)

ax4.set_xlabel('Frame Index', fontsize=12)
ax4.set_ylabel('Distance from Ego (meters)', fontsize=12)
ax4.set_title('Distance from Ego Over Time', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.suptitle(f'Kinematic Trajectories: {scene_name}', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
static_plot_path = os.path.join(output_dir, f'kinematic_trajectories_{scene_name}.png')
plt.savefig(static_plot_path, dpi=150, bbox_inches='tight')
print(f"✅ Static plot saved to: {static_plot_path}")
plt.close()

# ============================================================================
# 2. CREATE ANIMATED BEV VIDEO
# ============================================================================
print("\n" + "="*70)
print("Creating animated BEV video...")
print("="*70)

if len(ego_x) == 0:
    print("⚠️  No ego trajectory data, skipping BEV video")
else:
    # Determine plot bounds with some padding
    all_x_coords = np.concatenate([traj["x"] for traj in object_trajectories.values()] + [ego_x])
    all_y_coords = np.concatenate([traj["y"] for traj in object_trajectories.values()] + [ego_y])
    
    x_min, x_max = all_x_coords.min() - 10, all_x_coords.max() + 10
    y_min, y_max = all_y_coords.min() - 10, all_y_coords.max() + 10
    
    # Create figure for animation
    fig_bev, ax_bev = plt.subplots(figsize=(12, 12))
    ax_bev.set_xlim(x_min, x_max)
    ax_bev.set_ylim(y_min, y_max)
    ax_bev.set_aspect('equal')
    ax_bev.set_xlabel('X (meters)', fontsize=14)
    ax_bev.set_ylabel('Y (meters)', fontsize=14)
    ax_bev.set_title(f'Bird\'s Eye View - {scene_name}', fontsize=16, fontweight='bold')
    ax_bev.grid(True, alpha=0.3)
    
    # Store all trajectory points for drawing
    max_frames = len(ego_x)
    
    # Create color map for objects
    obj_colors = {inst_token: colors[i] for i, inst_token in enumerate(object_trajectories.keys())}
    
    # Prepare trajectory history (for trailing lines)
    trajectory_history = {inst_token: [] for inst_token in object_trajectories.keys()}
    ego_history = []
    
    # Category markers
    category_markers = {
        'vehicle': 's',  # square
        'car': 's',
        'truck': 's',
        'bus': 's',
        'person': 'o',  # circle
        'pedestrian': 'o',
        'bicycle': '^',  # triangle
        'bike': '^',
        'motorcycle': '^',
    }
    
    def get_marker(category):
        for key, marker in category_markers.items():
            if key.lower() in category.lower():
                return marker
        return 'o'  # default circle
    
    frames_to_render = []
    
    for frame_idx in range(max_frames):
        # Clear axis but keep grid
        ax_bev.clear()
        ax_bev.set_xlim(x_min, x_max)
        ax_bev.set_ylim(y_min, y_max)
        ax_bev.set_aspect('equal')
        ax_bev.set_xlabel('X (meters)', fontsize=14)
        ax_bev.set_ylabel('Y (meters)', fontsize=14)
        ax_bev.set_title(f'Bird\'s Eye View - {scene_name} | Frame {frame_idx}/{max_frames-1}', 
                        fontsize=16, fontweight='bold')
        ax_bev.grid(True, alpha=0.3)
        
        # Draw ego vehicle trajectory up to current frame
        if frame_idx < len(ego_x):
            ego_history.append((ego_x[frame_idx], ego_y[frame_idx]))
            if len(ego_history) > 1:
                ego_x_hist, ego_y_hist = zip(*ego_history)
                ax_bev.plot(ego_x_hist, ego_y_hist, 'r-', linewidth=4, alpha=0.6, label='Ego Path')
            # Current ego position
            ax_bev.scatter(ego_x[frame_idx], ego_y[frame_idx], c='red', s=200, 
                          marker='*', zorder=10, edgecolors='darkred', linewidths=2, label='Ego')
        
        # Draw object trajectories
        for inst_token, traj in object_trajectories.items():
            # Find frames where this object exists
            mask = traj["frames"] <= frame_idx
            if np.any(mask):
                obj_frames = traj["frames"][mask]
                obj_x = traj["x"][mask]
                obj_y = traj["y"][mask]
                
                # Update history
                if len(obj_frames) > 0:
                    last_idx = len(obj_frames) - 1
                    trajectory_history[inst_token].append((obj_x[last_idx], obj_y[last_idx]))
                    
                    # Draw trajectory line
                    if len(trajectory_history[inst_token]) > 1:
                        hist_x, hist_y = zip(*trajectory_history[inst_token])
                        ax_bev.plot(hist_x, hist_y, '-', color=obj_colors[inst_token], 
                                   linewidth=2, alpha=0.4)
                    
                    # Draw current position
                    category = traj["category"]
                    marker = get_marker(category)
                    ax_bev.scatter(obj_x[last_idx], obj_y[last_idx], 
                                 c=[obj_colors[inst_token]], s=100, marker=marker, 
                                 alpha=0.8, edgecolors='black', linewidths=1, zorder=5)
        
        # Add legend (only once)
        if frame_idx == 0:
            ax_bev.legend(loc='upper right', fontsize=10)
        
        # Convert to image using a simpler method
        fig_bev.canvas.draw()
        # Save to a temporary buffer and read it back
        from io import BytesIO
        buf = BytesIO()
        fig_bev.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)
        img_rgb = img.convert('RGB')
        frames_to_render.append(np.array(img_rgb))
        buf.close()
        
        if (frame_idx + 1) % 10 == 0:
            print(f"  Rendered {frame_idx + 1}/{max_frames} frames")
    
    plt.close(fig_bev)
    
    # Save as video
    output_video_path = os.path.join(output_dir, 'vis.mp4')
    print(f"\nSaving video to: {output_video_path}")
    imageio.mimwrite(output_video_path, frames_to_render, fps=2.0, codec='libx264', quality=8)
    print(f"✅ BEV video saved to: {output_video_path}")
    print(f"   Total frames: {len(frames_to_render)}")
    print(f"   Duration: {len(frames_to_render) / 2.0:.1f} seconds")

# ============================================================================
# 3. PRINT VALIDATION STATISTICS
# ============================================================================
print("\n" + "="*70)
print("Trajectory Validation Statistics")
print("="*70)

# Check for smooth trajectories (no large jumps)
print("\n1. Trajectory Smoothness Check:")
large_jumps = []
for instance_token, traj in object_trajectories.items():
    if len(traj["x"]) > 1:
        dx = np.diff(traj["x"])
        dy = np.diff(traj["y"])
        dz = np.diff(traj["z"])
        distances = np.sqrt(dx**2 + dy**2 + dz**2)
        
        max_jump = np.max(distances)
        if max_jump > 10.0:  # More than 10 meters per frame (unrealistic)
            large_jumps.append((instance_token[:16], max_jump, traj["category"]))

if large_jumps:
    print(f"  ⚠️  Found {len(large_jumps)} objects with large jumps (>10m/frame):")
    for inst_token, jump, cat in large_jumps[:5]:
        print(f"     Instance {inst_token}... ({cat}): {jump:.2f}m jump")
else:
    print("  ✅ All trajectories are smooth (no large jumps)")

# Check trajectory length
print("\n2. Trajectory Length Distribution:")
if object_trajectories:
    traj_lengths = [len(traj["frames"]) for traj in object_trajectories.values()]
    print(f"  Min length: {min(traj_lengths)} frames")
    print(f"  Max length: {max(traj_lengths)} frames")
    print(f"  Mean length: {np.mean(traj_lengths):.1f} frames")
    print(f"  Median length: {np.median(traj_lengths):.1f} frames")
else:
    print("  No object trajectories found")

# Check object categories
print("\n3. Object Categories:")
category_counts = defaultdict(int)
for traj in object_trajectories.values():
    category_counts[traj["category"]] += 1

for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
    print(f"  {category}: {count} objects")

# Check position range
print("\n4. Position Range:")
if object_trajectories and len(ego_x) > 0:
    all_x = np.concatenate([traj["x"] for traj in object_trajectories.values()] + [ego_x])
    all_y = np.concatenate([traj["y"] for traj in object_trajectories.values()] + [ego_y])
    all_z = np.concatenate([traj["z"] for traj in object_trajectories.values()] + [ego_z])
    
    print(f"  X range: [{all_x.min():.2f}, {all_x.max():.2f}] meters")
    print(f"  Y range: [{all_y.min():.2f}, {all_y.max():.2f}] meters")
    print(f"  Z range: [{all_z.min():.2f}, {all_z.max():.2f}] meters")

print(f"\n{'='*70}")
print("✅ Visualization complete!")
print(f"{'='*70}")
