#!/usr/bin/env python3
"""
Extract ego-pose sequences from NuScenes kinematic data for Cosmos 3.

This script converts your existing kinematic data [B, T, N, 18] format
to Cosmos 3's 9D ego-pose format: [tx, ty, tz, r1, r2, r3, r4, r5, r6]

Usage:
    python extract_ego_poses_for_cosmos3.py --input_dir /path/to/datasetWM --output_dir ./cosmos3_data
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch


def rotation_6d_from_yaw(yaw: np.ndarray) -> np.ndarray:
    """
    Convert yaw angle to 6D continuous rotation representation.
    
    6D rotation (from Zhou et al. "On the Continuity of Rotation Representations in Neural Networks"):
    Uses first two columns of rotation matrix (r1, r2, r3, r4, r5, r6)
    
    Args:
        yaw: [T] array of yaw angles in radians
    
    Returns:
        rotation_6d: [T, 6] array of 6D rotation representation
    """
    T = len(yaw)
    rotation_6d = np.zeros((T, 6), dtype=np.float32)
    
    # For 2D driving scenario, yaw rotation is around z-axis
    # Rotation matrix:
    # R = [ cos(yaw)  -sin(yaw)  0 ]
    #     [ sin(yaw)   cos(yaw)  0 ]
    #     [    0          0      1 ]
    #
    # First column: [cos(yaw), sin(yaw), 0]
    # Second column: [-sin(yaw), cos(yaw), 0]
    
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    # First column of rotation matrix
    rotation_6d[:, 0] = cos_yaw   # r1
    rotation_6d[:, 1] = sin_yaw   # r2
    rotation_6d[:, 2] = 0.0       # r3
    
    # Second column of rotation matrix
    rotation_6d[:, 3] = -sin_yaw  # r4
    rotation_6d[:, 4] = cos_yaw   # r5
    rotation_6d[:, 5] = 0.0       # r6
    
    return rotation_6d


def find_ego_vehicle_index(kinematics: np.ndarray, metadata: Dict = None) -> int:
    """
    Find the index of the ego vehicle in the agents array.
    
    Strategy:
    1. If metadata contains ego tracking_id, use it
    2. Otherwise, assume ego is agent with most consistent presence (fewest invalid frames)
    3. Fallback: agent closest to origin on average (camera is typically at ego)
    
    Args:
        kinematics: [T, N, D] kinematic data
        metadata: Optional metadata dict with ego info
    
    Returns:
        ego_idx: Index of ego vehicle
    """
    T, N, D = kinematics.shape
    
    # Check if tracking_id column exists (index 13)
    if D > 13 and metadata is not None and 'ego_tracking_id' in metadata:
        ego_id = metadata['ego_tracking_id']
        tracking_ids = kinematics[:, :, 13]  # [T, N]
        
        # Find agent with matching tracking_id (most frequent match)
        matches = (tracking_ids == ego_id).sum(axis=0)  # [N]
        ego_idx = int(np.argmax(matches))
        print(f"  Found ego vehicle using tracking_id={ego_id}: agent index {ego_idx}")
        return ego_idx
    
    # Strategy 2: Agent with most valid frames
    # Valid frame = position not all zeros
    positions = kinematics[:, :, 0:3]  # [T, N, 3]
    valid_frames = (np.abs(positions).sum(axis=-1) > 0.1).sum(axis=0)  # [N]
    
    ego_idx_candidate = int(np.argmax(valid_frames))
    max_valid = valid_frames[ego_idx_candidate]
    
    # If one agent has significantly more valid frames (>80% of video), use it
    if max_valid > 0.8 * T:
        print(f"  Found ego vehicle by consistency: agent index {ego_idx_candidate} ({max_valid}/{T} valid frames)")
        return ego_idx_candidate
    
    # Strategy 3: Agent closest to origin on average (camera at ego)
    distances = np.linalg.norm(positions, axis=-1)  # [T, N]
    avg_distance = distances.mean(axis=0)  # [N]
    ego_idx = int(np.argmin(avg_distance))
    
    print(f"  Found ego vehicle by proximity to origin: agent index {ego_idx} (avg distance: {avg_distance[ego_idx]:.2f}m)")
    return ego_idx


def extract_ego_pose_9d(kinematics: np.ndarray, ego_idx: int) -> np.ndarray:
    """
    Extract 9D ego-pose from kinematic data.
    
    Args:
        kinematics: [T, N, D] kinematic data where D >= 13
                    [:, :, 0:3] = (x, y, z) position
                    [:, :, 12] = yaw angle
        ego_idx: Index of ego vehicle
    
    Returns:
        ego_pose_9d: [T, 9] array: [tx, ty, tz, r1, r2, r3, r4, r5, r6]
    """
    T = kinematics.shape[0]
    
    # Extract ego position [T, 3]
    ego_position = kinematics[:, ego_idx, 0:3].astype(np.float32)
    
    # Extract ego yaw [T]
    ego_yaw = kinematics[:, ego_idx, 12].astype(np.float32)
    
    # Convert yaw to 6D rotation [T, 6]
    ego_rotation_6d = rotation_6d_from_yaw(ego_yaw)
    
    # Combine to 9D ego-pose [T, 9]
    ego_pose_9d = np.concatenate([ego_position, ego_rotation_6d], axis=-1)
    
    return ego_pose_9d


def process_sample(
    h5_path: Path,
    video_path: Path,
    output_dir: Path,
    sample_name: str
) -> Dict:
    """
    Process a single video sample and extract ego-pose.
    
    Args:
        h5_path: Path to kinematics .h5 file
        video_path: Path to corresponding video file
        output_dir: Output directory for ego-pose JSON
        sample_name: Base name for output files
    
    Returns:
        metadata: Dict with sample metadata
    """
    # Load kinematics from H5
    with h5py.File(h5_path, 'r') as f:
        # Assuming structure: f['kinematics'] = [T, N, D]
        if 'kinematics' in f:
            kinematics = f['kinematics'][:]
        elif 'kinematic' in f:
            kinematics = f['kinematic'][:]
        else:
            # Try to find the main dataset
            keys = list(f.keys())
            print(f"  Warning: No 'kinematics' key found. Available keys: {keys}")
            kinematics = f[keys[0]][:]
    
    T, N, D = kinematics.shape
    print(f"  Loaded kinematics: shape={kinematics.shape} (T={T} frames, N={N} agents, D={D} features)")
    
    # Find ego vehicle
    ego_idx = find_ego_vehicle_index(kinematics)
    
    # Extract 9D ego-pose
    ego_pose_9d = extract_ego_pose_9d(kinematics, ego_idx)
    
    # Save ego-pose as JSON (Cosmos 3 format)
    ego_pose_path = output_dir / f"{sample_name}_ego_pose.json"
    ego_pose_list = ego_pose_9d.tolist()  # Convert to list for JSON
    
    with open(ego_pose_path, 'w') as f:
        json.dump({
            'ego_pose': ego_pose_list,  # [T, 9]
            'num_frames': T,
            'ego_agent_index': ego_idx,
            'format': '[tx, ty, tz, r1, r2, r3, r4, r5, r6]',
            'units': 'meters (position), 6D rotation representation',
            'source_h5': str(h5_path),
            'video': str(video_path),
        }, f, indent=2)
    
    print(f"  ✓ Saved ego-pose to: {ego_pose_path}")
    
    # Print statistics
    ego_position = ego_pose_9d[:, 0:3]
    ego_displacement = np.linalg.norm(ego_position[-1] - ego_position[0])
    ego_speed_avg = np.linalg.norm(np.diff(ego_position, axis=0), axis=1).mean()
    
    print(f"  Ego statistics:")
    print(f"    - Total displacement: {ego_displacement:.2f}m")
    print(f"    - Average speed: {ego_speed_avg:.2f}m/frame")
    print(f"    - Position range: x=[{ego_position[:, 0].min():.1f}, {ego_position[:, 0].max():.1f}], "
          f"y=[{ego_position[:, 1].min():.1f}, {ego_position[:, 1].max():.1f}], "
          f"z=[{ego_position[:, 2].min():.1f}, {ego_position[:, 2].max():.1f}]")
    
    return {
        'sample_name': sample_name,
        'ego_pose_path': str(ego_pose_path),
        'video_path': str(video_path),
        'num_frames': T,
        'ego_idx': ego_idx,
    }


def main():
    parser = argparse.ArgumentParser(description='Extract ego-poses from NuScenes data for Cosmos 3')
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Path to dataset directory (e.g., /scratch/user/u.pt152369/WM/data/datasetWM)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./cosmos3_ego_poses',
        help='Output directory for ego-pose JSON files'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='Number of samples to process (default: 10, use -1 for all)'
    )
    parser.add_argument(
        '--video_ext',
        type=str,
        default='.mp4',
        help='Video file extension (default: .mp4)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting ego-poses from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Processing up to {args.num_samples} samples" if args.num_samples > 0 else "Processing all samples")
    print("=" * 70)
    
    # Find all H5 files
    h5_files = sorted(input_dir.rglob('*.h5'))
    
    if not h5_files:
        print(f"ERROR: No .h5 files found in {input_dir}")
        return
    
    print(f"Found {len(h5_files)} .h5 files")
    
    # Limit number of samples if specified
    if args.num_samples > 0:
        h5_files = h5_files[:args.num_samples]
    
    # Process each sample
    results = []
    for i, h5_path in enumerate(h5_files, 1):
        print(f"\n[{i}/{len(h5_files)}] Processing: {h5_path.name}")
        
        # Find corresponding video file
        video_path = h5_path.with_suffix(args.video_ext)
        if not video_path.exists():
            # Try alternative locations (e.g., videos/ subdirectory)
            video_path_alt = h5_path.parent / 'videos' / h5_path.with_suffix(args.video_ext).name
            if video_path_alt.exists():
                video_path = video_path_alt
            else:
                print(f"  Warning: Video not found at {video_path}, skipping")
                continue
        
        try:
            sample_name = h5_path.stem
            metadata = process_sample(h5_path, video_path, output_dir, sample_name)
            results.append(metadata)
        except Exception as e:
            print(f"  ERROR processing {h5_path.name}: {e}")
            continue
    
    # Save summary
    summary_path = output_dir / 'extraction_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'num_samples': len(results),
            'input_dir': str(input_dir),
            'output_dir': str(output_dir),
            'samples': results,
        }, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"✓ Extraction complete!")
    print(f"  Processed: {len(results)}/{len(h5_files)} samples")
    print(f"  Output directory: {output_dir}")
    print(f"  Summary: {summary_path}")
    print("\nNext steps:")
    print("  1. Review the extracted ego-poses in the output directory")
    print("  2. Test with Cosmos 3 inference:")
    print("     ```python")
    print("     import json")
    print("     from cosmos_framework import Cosmos3Client")
    print("     ")
    print("     # Load ego-pose")
    print(f"     with open('{output_dir}/SAMPLE_ego_pose.json') as f:")
    print("         data = json.load(f)")
    print("     ")
    print("     # Run forward dynamics")
    print("     client = Cosmos3Client()")
    print("     result = client.generate(")
    print("         video=data['video'],")
    print("         action=data['ego_pose'],")
    print("         embodiment='autonomous_vehicle'")
    print("     )")
    print("     ```")


if __name__ == '__main__':
    main()
