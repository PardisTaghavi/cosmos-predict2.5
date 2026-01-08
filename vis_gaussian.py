# vis_gaussian.py
"""
Visualize Gaussian Splatting on GT Videos
Shows Gaussian kernels overlaid on video frames with kinematic data
"""

import os
import json
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import imageio
from pathlib import Path

# Configuration
VIDEO_DIR = "/Users/ptgh/Desktop/code/cosmos-predict2.5/data/datasetWM/videos"
KINEMATICS_DIR = "/Users/ptgh/Desktop/code/cosmos-predict2.5/data/datasetWM/kinematics"
NUSCENES_ROOT = "/Users/ptgh/Desktop/code/cosmos-predict2.5/data/v1.0-mini/v1.0-mini"
OUTPUT_DIR = "/Users/ptgh/Desktop/code/cosmos-predict2.5/gaussian_visualizations"
# Model uses sigma=0.1 in normalized [0,1] space
# For visualization in pixel space (1600x900), scale appropriately
# sigma=0.1 in normalized space ≈ 160 pixels for width, 90 pixels for height
# Use average: ~100 pixels for good visualization
SIGMA = 100.0  # Gaussian bandwidth in pixels (for visualization)

# Z adjustment for projection
# True: Adjust z coordinate for better alignment
# False: Use ground level (z as stored) - matches FOV check in create_kinematic.py
USE_VISUAL_CENTER_Z = True

# Z adjustment factor (only used if USE_VISUAL_CENTER_Z = True)
# Positive values: z + factor * h (moves point UP in 3D, appears HIGHER in image)
# Negative values: z - |factor| * h (moves point DOWN in 3D, appears LOWER in image)
# 0.5 = center of bounding box (z + h/2) - visual center
# 0.0 = ground level (z) - as stored in kinematic data
# With proper camera extrinsics, try 0.0 (ground) or 0.5 (visual center)
Z_ADJUSTMENT_FACTOR = 0.5  # Visual center (z + h/2) - adjust if needed

# Color mapping for agent classes
CLASS_COLORS = {
    0: (0, 0, 255),      # Ego - Blue
    1: (255, 0, 0),      # Vehicle - Red
    2: (0, 255, 0),      # Pedestrian - Green
    3: (255, 255, 0),    # Bicycle - Yellow
}

CLASS_NAMES = ['ego', 'vehicle', 'pedestrian', 'bicycle']


def load_video(video_path):
    """Load video frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)


def load_kinematics(h5_path):
    """Load kinematics from h5 file."""
    with h5py.File(h5_path, 'r') as f:
        kinematics = f['frames'][:]  # [T, N, 18]
    return kinematics


def load_nuscenes_calibration():
    """Load NuScenes camera calibration data (same logic as create_kinematic.py)."""
    with open(os.path.join(NUSCENES_ROOT, "calibrated_sensor.json"), "r") as f:
        calibrated_sensors = json.load(f)
    with open(os.path.join(NUSCENES_ROOT, "sample_data.json"), "r") as f:
        sample_data = json.load(f)
    
    # Create lookup dictionaries
    calibrated_sensor_dict = {cs["token"]: cs for cs in calibrated_sensors}
    
    # Find CAM_FRONT sample_data entries (same logic as create_kinematic.py)
    cam_front_samples = [
        sd for sd in sample_data
        if "CAM_FRONT" in sd.get("filename", "") and sd.get("is_key_frame", False)
    ]
    
    # Get camera intrinsics and extrinsics from first CAM_FRONT sample
    if cam_front_samples:
        cam_front_sd = cam_front_samples[0]
        calibrated_sensor_token = cam_front_sd["calibrated_sensor_token"]
        cam_calib = calibrated_sensor_dict[calibrated_sensor_token]
        cam_intrinsics = cam_calib["camera_intrinsic"]
        cam_translation = cam_calib.get("translation", [0.0, 0.0, 0.0])  # Camera position relative to ego
        cam_rotation = cam_calib.get("rotation", [1.0, 0.0, 0.0, 0.0])  # Camera rotation quaternion [w,x,y,z]
        return cam_intrinsics, cam_translation, cam_rotation
    
    return None, None, None


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to rotation matrix (same as create_kinematic.py)."""
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    return R


def project_3d_to_2d_cam_front(positions_3d, image_shape, cam_intrinsics, cam_translation=None, cam_rotation=None):
    """
    Project 3D ego-frame positions to 2D image coordinates using CAM_FRONT camera model.
    This matches the exact projection logic from create_kinematic.py.
    
    Note: create_kinematic.py doesn't use camera extrinsics (assumes camera at ego origin).
    However, NuScenes cameras have extrinsics. We'll match create_kinematic.py's behavior
    but can optionally apply extrinsics for more accurate projection.
    
    Args:
        positions_3d: [N, 3] - (x, y, z) in ego frame (x forward, y left, z up)
        image_shape: (H, W) - image dimensions
        cam_intrinsics: [3, 3] camera intrinsic matrix from NuScenes calibration
        cam_translation: Optional [3] camera translation relative to ego
        cam_rotation: Optional [4] camera rotation quaternion [w,x,y,z] relative to ego
    
    Returns:
        positions_2d: [N, 2] - (u, v) pixel coordinates
    """
    H, W = image_shape[:2]
    
    # Convert camera intrinsics to numpy array (same as create_kinematic.py)
    K = np.array(cam_intrinsics)
    if K.shape != (3, 3):
        raise ValueError(f"Camera intrinsics must be 3x3, got shape {K.shape}")
    
    # Extract positions
    x = positions_3d[:, 0]  # forward (ego frame)
    y = positions_3d[:, 1]  # left (ego frame)
    z = positions_3d[:, 2]  # up (ego frame)
    
    # Apply camera extrinsics if provided (transform from ego to camera frame)
    if cam_translation is not None and cam_rotation is not None:
        # Transform points from ego frame to camera frame
        # cam_pos = R_cam^T @ (ego_pos - cam_translation)
        cam_translation = np.array(cam_translation)
        R_cam = quaternion_to_rotation_matrix(cam_rotation)
        
        # Transform each point
        ego_positions = np.stack([x, y, z], axis=-1)  # [N, 3]
        relative_positions = ego_positions - cam_translation  # [N, 3]
        cam_frame_positions = (R_cam.T @ relative_positions.T).T  # [N, 3]
        
        # Extract camera frame coordinates
        cam_x = cam_frame_positions[:, 0]  # Right in image
        cam_y = cam_frame_positions[:, 1]  # Down in image
        cam_z = cam_frame_positions[:, 2]  # Forward (depth)
    else:
        # Simple transformation (matching create_kinematic.py - assumes camera at ego origin)
        # NuScenes ego frame: x forward, y left, z up
        # Standard camera frame: x right, y down, z forward
        # Transformation: camera_x = -ego_y, camera_y = -ego_z, camera_z = ego_x
        cam_x = -y   # Right in image (positive = right)
        cam_y = -z   # Down in image (positive = down)
        cam_z = x    # Forward (depth, positive = in front)
    
    # Filter out points behind camera
    valid = cam_z > 0
    
    # Project to image plane using pinhole camera model (EXACTLY as in create_kinematic.py)
    # u = fx * (X/Z) + cx
    # v = fy * (Y/Z) + cy
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # Avoid division by zero for points behind camera
    cam_z_safe = np.where(cam_z > 0, cam_z, 1.0)  # Use 1.0 for invalid points
    
    u = fx * (cam_x / cam_z_safe) + cx
    v = fy * (cam_y / cam_z_safe) + cy
    
    # Clip to image bounds
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)
    
    # Set invalid points (behind camera) to center
    u[~valid] = W / 2
    v[~valid] = H / 2
    
    return np.stack([u, v], axis=-1)  # [N, 2]


def compute_gaussian_weights(agent_positions_2d, image_shape, sigma=SIGMA):
    """
    Compute Gaussian weights for each pixel given agent positions.
    
    Args:
        agent_positions_2d: [N, 2] - (u, v) coordinates of agents in pixels
        image_shape: (H, W) - image dimensions
        sigma: Gaussian bandwidth in pixels
    
    Returns:
        weights: [N, H, W] - Gaussian weights for each agent at each pixel
    """
    H, W = image_shape
    N = len(agent_positions_2d)
    
    # Create pixel grid
    v_coords = np.linspace(0, H - 1, H)
    u_coords = np.linspace(0, W - 1, W)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)  # [H, W]
    
    # Initialize weights
    weights = np.zeros((N, H, W))
    
    for i, (u_agent, v_agent) in enumerate(agent_positions_2d):
        # Compute squared distances from agent to each pixel (in pixels)
        distances_sq = (u_grid - u_agent) ** 2 + (v_grid - v_agent) ** 2
        
        # Gaussian kernel: exp(-d² / (2σ²))
        # Note: sigma is in pixels, so distances_sq is in pixels²
        weights[i] = np.exp(-distances_sq / (2 * sigma ** 2))
    
    return weights


def visualize_gaussian_splatting(video_path, h5_path, output_path, frame_idx=0, cam_intrinsics=None, cam_translation=None, cam_rotation=None):
    """
    Visualize Gaussian splatting for a single frame.
    
    Args:
        video_path: Path to video file
        h5_path: Path to kinematics h5 file
        output_path: Path to save visualization
        frame_idx: Frame index to visualize
        cam_intrinsics: Camera intrinsics matrix (required)
        cam_translation: Optional camera translation relative to ego
        cam_rotation: Optional camera rotation quaternion relative to ego
    """
    if cam_intrinsics is None:
        raise ValueError("cam_intrinsics must be provided")
    
    # Load data
    video_frames = load_video(video_path)
    kinematics = load_kinematics(h5_path)
    
    if frame_idx >= len(video_frames) or frame_idx >= len(kinematics):
        print(f"Frame {frame_idx} out of range. Video: {len(video_frames)}, Kinematics: {len(kinematics)}")
        return
    
    frame = video_frames[frame_idx]
    H, W = frame.shape[:2]
    
    # Get kinematic data for this frame
    frame_kinematics = kinematics[frame_idx]  # [N, 18]
    
    # Extract positions, dimensions, and classes
    positions_3d = frame_kinematics[:, :3]  # [N, 3] - (x, y, z) - ground level position
    dimensions = frame_kinematics[:, 9:12]  # [N, 3] - (l, w, h) - length, width, height
    classes_onehot = frame_kinematics[:, 14:18]  # [N, 4] - one-hot classes
    
    # Find valid agents (non-zero class)
    valid_mask = classes_onehot.sum(axis=1) > 0
    
    # Filter out ego vehicle (position [0, 0, 0] or very close to it)
    # Ego vehicle cannot be seen in RGB image (it's the camera's own vehicle)
    ego_threshold = 0.1  # meters - anything closer than this is considered ego
    ego_mask = np.linalg.norm(positions_3d, axis=1) < ego_threshold
    valid_mask = valid_mask & ~ego_mask  # Remove ego from valid agents
    
    valid_positions = positions_3d[valid_mask]
    valid_dimensions = dimensions[valid_mask]
    valid_classes = classes_onehot[valid_mask]
    
    if len(valid_positions) == 0:
        print(f"  No valid non-ego agents found in frame {frame_idx}")
        return None
    
    # Adjust z position for projection
    valid_positions_adjusted = valid_positions.copy()
    if USE_VISUAL_CENTER_Z:
        # Adjust z by a fraction of height (default: 0.5 = center of bounding box)
        valid_positions_adjusted[:, 2] = valid_positions[:, 2] + valid_dimensions[:, 2] * Z_ADJUSTMENT_FACTOR
    # else: use ground level z as-is
    
    if len(valid_positions) == 0:
        print("No valid agents in this frame")
        return
    
    # Project 3D to 2D using CAM_FRONT camera model (with actual intrinsics AND extrinsics)
    # For accurate visualization, use camera extrinsics (even though create_kinematic.py doesn't)
    # create_kinematic.py stores 3D positions correctly; vis_gaussian.py needs accurate projection
    positions_2d = project_3d_to_2d_cam_front(
        valid_positions_adjusted, (H, W), cam_intrinsics, 
        cam_translation=cam_translation,  # USE extrinsics for accurate projection
        cam_rotation=cam_rotation
    )  # [N_valid, 2]
    
    # Debug: print projection info
    print(f"  Projected {len(positions_2d)} agents to 2D:")
    print(f"    U range: {positions_2d[:, 0].min():.1f} to {positions_2d[:, 0].max():.1f} (image width: {W})")
    print(f"    V range: {positions_2d[:, 1].min():.1f} to {positions_2d[:, 1].max():.1f} (image height: {H})")
    
    # Compute Gaussian weights
    weights = compute_gaussian_weights(positions_2d, (H, W), sigma=SIGMA)  # [N_valid, H, W]
    
    # Debug: print weight statistics
    total_weights = weights.sum(axis=0)
    print(f"    Total weights range: {total_weights.min():.4f} to {total_weights.max():.4f}")
    print(f"    Mean weight: {total_weights.mean():.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Original video frame with agent positions
    ax1 = axes[0, 0]
    ax1.imshow(frame)
    ax1.set_title(f"Frame {frame_idx}: Agent Positions", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Draw agent positions
    seen_classes = set()
    for i, (u, v) in enumerate(positions_2d):
        class_idx = np.argmax(valid_classes[i])
        color = np.array(CLASS_COLORS[class_idx]) / 255.0
        class_name = CLASS_NAMES[class_idx]
        label = class_name if class_name not in seen_classes else ""
        seen_classes.add(class_name)
        ax1.plot(u, v, 'o', color=color, markersize=10, label=label)
        ax1.text(u + 5, v + 5, f"A{i}", color='white', fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    if seen_classes:
        ax1.legend()
    
    # 2. Gaussian weights heatmap (sum of all agents)
    ax2 = axes[0, 1]
    total_weights = weights.sum(axis=0)  # [H, W] - sum over all agents
    
    # Normalize for better visualization if needed
    if total_weights.max() > 0:
        # Use a reasonable scale - show weights up to 3x (multiple agents can overlap)
        vmax = min(total_weights.max(), 3.0)
    else:
        vmax = 1.0
    
    im2 = ax2.imshow(total_weights, cmap='hot', alpha=0.8, vmin=0, vmax=vmax)
    ax2.set_title("Gaussian Splatting: Combined Weights (Bell Curves)", fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, label='Weight Intensity')
    
    # Overlay agent positions
    for i, (u, v) in enumerate(positions_2d):
        class_idx = np.argmax(valid_classes[i])
        color = np.array(CLASS_COLORS[class_idx]) / 255.0
        ax2.plot(u, v, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=1)
    
    # 3. Individual Gaussian kernel for first agent
    ax3 = axes[1, 0]
    if len(weights) > 0:
        single_agent_weights = weights[0]  # [H, W]
        # Use consistent colormap scaling
        im3 = ax3.imshow(single_agent_weights, cmap='viridis', alpha=0.9, vmin=0, vmax=1.0)
        ax3.set_title(f"Single Agent Gaussian Kernel (Agent 0: {CLASS_NAMES[np.argmax(valid_classes[0])]})", 
                      fontsize=14, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, label='Weight')
        
        # Mark agent position
        u, v = positions_2d[0]
        ax3.plot(u, v, 'rx', markersize=15, markeredgewidth=3, label='Agent Position')
        ax3.legend()
    
    # 4. Cross-section showing bell curve shape
    ax4 = axes[1, 1]
    if len(weights) > 0:
        # Extract horizontal cross-section through agent center
        u_center, v_center = int(positions_2d[0][0]), int(positions_2d[0][1])
        v_center = np.clip(v_center, 0, H - 1)
        
        cross_section = weights[0][v_center, :]  # [W] - horizontal slice
        
        ax4.plot(cross_section, 'b-', linewidth=2, label='Gaussian Bell Curve')
        ax4.axvline(u_center, color='r', linestyle='--', linewidth=2, label='Agent Position')
        ax4.set_xlabel('Pixel Position (u)', fontsize=12)
        ax4.set_ylabel('Weight', fontsize=12)
        ax4.set_title('Gaussian Bell Curve (Horizontal Cross-Section)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {output_path}")


def create_gaussian_video(video_path, h5_path, output_path, cam_intrinsics=None, cam_translation=None, cam_rotation=None):
    """
    Create video visualization showing Gaussian splatting over time.
    
    Args:
        video_path: Path to input video
        h5_path: Path to kinematics h5 file
        output_path: Path to save output video
        cam_intrinsics: Camera intrinsics matrix (required)
        cam_translation: Optional camera translation relative to ego
        cam_rotation: Optional camera rotation quaternion relative to ego
    """
    if cam_intrinsics is None:
        raise ValueError("cam_intrinsics must be provided")
    
    # Load data
    video_frames = load_video(video_path)
    kinematics = load_kinematics(h5_path)
    
    T = min(len(video_frames), len(kinematics))
    H, W = video_frames[0].shape[:2]
    
    output_frames = []
    
    print(f"Creating Gaussian splatting video: {T} frames")
    
    for frame_idx in range(T):
        frame = video_frames[frame_idx]
        frame_kinematics = kinematics[frame_idx]  # [N, 18]
        
        # Extract positions, dimensions, and classes
        positions_3d = frame_kinematics[:, :3]  # Ground level positions
        dimensions = frame_kinematics[:, 9:12]  # (l, w, h)
        classes_onehot = frame_kinematics[:, 14:18]
        
        # Find valid agents (non-zero class)
        valid_mask = classes_onehot.sum(axis=1) > 0
        
        # Filter out ego vehicle (position [0, 0, 0] or very close to it)
        # Ego vehicle cannot be seen in RGB image (it's the camera's own vehicle)
        ego_threshold = 0.1  # meters - anything closer than this is considered ego
        ego_mask = np.linalg.norm(positions_3d, axis=1) < ego_threshold
        valid_mask = valid_mask & ~ego_mask  # Remove ego from valid agents
        
        valid_positions = positions_3d[valid_mask]
        valid_dimensions = dimensions[valid_mask]
        valid_classes = classes_onehot[valid_mask]
        
        if len(valid_positions) == 0:
            output_frames.append(frame)
            continue
        
        # Adjust z position for projection
        valid_positions_adjusted = valid_positions.copy()
        if USE_VISUAL_CENTER_Z:
            # Adjust z by a fraction of height (default: 0.5 = center of bounding box)
            valid_positions_adjusted[:, 2] = valid_positions[:, 2] + valid_dimensions[:, 2] * Z_ADJUSTMENT_FACTOR
        # else: use ground level z as-is
        
        # Project 3D to 2D using CAM_FRONT camera model (with actual intrinsics AND extrinsics)
        # For accurate visualization, use camera extrinsics (even though create_kinematic.py doesn't)
        # create_kinematic.py stores 3D positions correctly; vis_gaussian.py needs accurate projection
        positions_2d = project_3d_to_2d_cam_front(
            valid_positions_adjusted, (H, W), cam_intrinsics,
            cam_translation=cam_translation,  # USE extrinsics for accurate projection
            cam_rotation=cam_rotation
        )
        
        # Compute Gaussian weights
        weights = compute_gaussian_weights(positions_2d, (H, W), sigma=SIGMA)
        
        # Create overlay: combine all Gaussian kernels
        total_weights = weights.sum(axis=0)  # [H, W]
        
        # Normalize weights to [0, 1] for visualization
        if total_weights.max() > 0:
            total_weights = total_weights / total_weights.max()
        
        # Create heatmap overlay
        heatmap = plt.cm.hot(total_weights)[:, :, :3]  # [H, W, 3] RGB
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Blend with original frame
        alpha = 0.5
        overlay_frame = (frame * (1 - alpha) + heatmap * alpha).astype(np.uint8)
        
        # Draw agent positions
        for i, (u, v) in enumerate(positions_2d):
            class_idx = np.argmax(valid_classes[i])
            color = CLASS_COLORS[class_idx]
            cv2.circle(overlay_frame, (int(u), int(v)), 5, color, -1)
            cv2.circle(overlay_frame, (int(u), int(v)), 8, (255, 255, 255), 2)
        
        # Add frame number
        cv2.putText(overlay_frame, f"Frame {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        output_frames.append(overlay_frame)
        
        if (frame_idx + 1) % 10 == 0:
            print(f"  Processed {frame_idx + 1}/{T} frames")
    
    # Save video
    print(f"Saving video to: {output_path}")
    imageio.mimwrite(output_path, output_frames, fps=2.0, codec='libx264')
    print(f"✓ Saved Gaussian splatting video: {output_path}")


def visualize_bell_curve_comparison(output_path):
    """
    Create a comparison plot showing Gaussian bell curves for different sigma values.
    Shows both normalized space (model) and pixel space (visualization).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 1D Gaussian curves in normalized space (model uses σ=0.1)
    ax1 = axes[0]
    distances_normalized = np.linspace(0, 0.5, 1000)
    sigmas_normalized = [0.05, 0.1, 0.2, 0.3]
    MODEL_SIGMA = 0.1  # Model uses σ=0.1 in normalized [0,1] space
    
    for sigma in sigmas_normalized:
        weights = np.exp(-distances_normalized ** 2 / (2 * sigma ** 2))
        ax1.plot(distances_normalized, weights, label=f'σ = {sigma}', linewidth=2)
    
    # Mark the model's sigma
    ax1.axvline(MODEL_SIGMA, color='r', linestyle='--', linewidth=2, 
                label=f'Model: σ = {MODEL_SIGMA} (normalized)', alpha=0.7)
    ax1.set_xlabel('Distance from Agent (normalized [0,1])', fontsize=12)
    ax1.set_ylabel('Gaussian Weight', fontsize=12)
    ax1.set_title('Gaussian Bell Curves (Normalized Space - Model)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 1.1])
    
    # 2. 2D Gaussian surface in normalized space (matching model)
    ax2 = axes[1]
    x = np.linspace(-0.5, 0.5, 100)
    y = np.linspace(-0.5, 0.5, 100)
    X, Y = np.meshgrid(x, y)
    distances_2d = np.sqrt(X ** 2 + Y ** 2)
    # Use model's sigma (normalized space)
    weights_2d = np.exp(-distances_2d ** 2 / (2 * MODEL_SIGMA ** 2))
    
    im = ax2.contourf(X, Y, weights_2d, levels=20, cmap='viridis')
    ax2.set_xlabel('Distance X (normalized)', fontsize=12)
    ax2.set_ylabel('Distance Y (normalized)', fontsize=12)
    ax2.set_title(f'2D Gaussian Kernel (σ = {MODEL_SIGMA} - Model)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Weight')
    ax2.plot(0, 0, 'rx', markersize=15, markeredgewidth=3, label='Agent Center')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved bell curve comparison to: {output_path}")
    print(f"  Note: Model uses σ={MODEL_SIGMA} in normalized space")
    print(f"        Visualization uses σ={SIGMA} pixels for display")


def main():
    """Main function to run visualizations."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load actual camera intrinsics from NuScenes
    print("Loading NuScenes camera calibration...")
    cam_intrinsics, cam_translation, cam_rotation = load_nuscenes_calibration()
    if cam_intrinsics is None:
        print("Warning: Could not load camera intrinsics, using defaults")
        # Fallback to approximate values
        cam_intrinsics = [[800.0, 0, 800.0], [0, 800.0, 450.0], [0, 0, 1]]
        cam_translation = None
        cam_rotation = None
    else:
        print(f"Loaded camera intrinsics: fx={cam_intrinsics[0][0]:.1f}, fy={cam_intrinsics[1][1]:.1f}, "
              f"cx={cam_intrinsics[0][2]:.1f}, cy={cam_intrinsics[1][2]:.1f}")
        if cam_translation:
            print(f"Using camera extrinsics for accurate projection:")
            print(f"  Translation: {cam_translation}")
            print(f"  Rotation: {cam_rotation}")
            print("Note: create_kinematic.py stores 3D positions correctly (ego frame)")
            print("      vis_gaussian.py uses extrinsics for accurate 2D projection")
        else:
            print("Warning: No extrinsics available, using simplified projection")
    
    # Find video files
    video_files = list(Path(VIDEO_DIR).glob("*.mp4"))
    
    if not video_files:
        print(f"No video files found in {VIDEO_DIR}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Visualize bell curve shapes
    bell_curve_path = os.path.join(OUTPUT_DIR, "gaussian_bell_curves.png")
    visualize_bell_curve_comparison(bell_curve_path)
    
    # Visualize first video
    video_path = str(video_files[0])
    video_basename = Path(video_path).stem
    h5_path = os.path.join(KINEMATICS_DIR, f"{video_basename}.h5")
    
    if not os.path.exists(h5_path):
        print(f"Kinematics file not found: {h5_path}")
        return
    
    # Single frame visualization - find first frame with valid agents
    kinematics = load_kinematics(h5_path)
    frame_idx = 0
    for i in range(min(len(kinematics), 10)):  # Check first 10 frames
        frame_kinematics = kinematics[i]
        classes_onehot = frame_kinematics[:, 14:18]
        if classes_onehot.sum(axis=1).sum() > 0:  # Has valid agents
            frame_idx = i
            break
    
    frame_viz_path = os.path.join(OUTPUT_DIR, f"{video_basename}_frame{frame_idx}_gaussian.png")
    visualize_gaussian_splatting(video_path, h5_path, frame_viz_path, frame_idx=frame_idx, 
                                cam_intrinsics=cam_intrinsics, cam_translation=cam_translation, cam_rotation=cam_rotation)
    
    # Video visualization
    video_viz_path = os.path.join(OUTPUT_DIR, f"{video_basename}_gaussian_splatting.mp4")
    create_gaussian_video(video_path, h5_path, video_viz_path, 
                         cam_intrinsics=cam_intrinsics, cam_translation=cam_translation, cam_rotation=cam_rotation)
    
    print("\n" + "="*70)
    print("Visualization Complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()

