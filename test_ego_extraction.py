#!/usr/bin/env python3
"""
Quick test script to validate ego-pose extraction.

Run this first with a single sample to verify the extraction works:
    python test_ego_extraction.py
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np


def test_single_sample(h5_path: str = None):
    """Test ego-pose extraction on a single sample."""
    
    if h5_path is None:
        # Try to find a sample H5 file
        dataset_dir = Path("/scratch/user/u.pt152369/WM/data/datasetWM")
        if not dataset_dir.exists():
            print("ERROR: Dataset directory not found. Please provide path to a .h5 file:")
            print("  python test_ego_extraction.py /path/to/sample.h5")
            return
        
        h5_files = list(dataset_dir.rglob("*.h5"))
        if not h5_files:
            print(f"ERROR: No .h5 files found in {dataset_dir}")
            return
        
        h5_path = str(h5_files[0])
    
    print(f"Testing with: {h5_path}")
    print("=" * 70)
    
    # Load kinematics
    with h5py.File(h5_path, 'r') as f:
        print(f"H5 file keys: {list(f.keys())}")
        
        # Try different possible key names
        if 'kinematics' in f:
            kinematics = f['kinematics'][:]
            key_used = 'kinematics'
        elif 'kinematic' in f:
            kinematics = f['kinematic'][:]
            key_used = 'kinematic'
        else:
            key_used = list(f.keys())[0]
            kinematics = f[key_used][:]
        
        print(f"Loaded data from key: '{key_used}'")
        print(f"  Shape: {kinematics.shape}")
        print(f"  Dtype: {kinematics.dtype}")
    
    T, N, D = kinematics.shape
    print(f"\nData dimensions:")
    print(f"  T (frames): {T}")
    print(f"  N (agents): {N}")
    print(f"  D (features): {D}")
    
    # Expected format: D=18 with [x,y,z,vx,vy,vz,ax,ay,az,l,w,h,yaw,tracking_id,class1,class2,class3,class4]
    if D < 13:
        print(f"\nWARNING: Expected at least 13 features, got {D}")
        print("Expected format: [x,y,z,vx,vy,vz,ax,ay,az,l,w,h,yaw,...]")
        return
    
    # Analyze agent validity
    print(f"\nAgent analysis:")
    positions = kinematics[:, :, 0:3]  # [T, N, 3]
    valid_frames_per_agent = (np.abs(positions).sum(axis=-1) > 0.1).sum(axis=0)  # [N]
    
    print(f"  Valid frames per agent:")
    for i in range(min(N, 10)):  # Show first 10 agents
        print(f"    Agent {i}: {valid_frames_per_agent[i]}/{T} frames ({100*valid_frames_per_agent[i]/T:.1f}%)")
    
    if N > 10:
        print(f"    ... ({N-10} more agents)")
    
    # Find likely ego vehicle
    ego_idx = int(np.argmax(valid_frames_per_agent))
    print(f"\n  Likely ego vehicle: Agent {ego_idx} ({valid_frames_per_agent[ego_idx]}/{T} valid frames)")
    
    # Extract ego trajectory
    ego_position = kinematics[:, ego_idx, 0:3]
    ego_yaw = kinematics[:, ego_idx, 12] if D > 12 else np.zeros(T)
    
    print(f"\nEgo vehicle trajectory:")
    print(f"  Start position: ({ego_position[0, 0]:.2f}, {ego_position[0, 1]:.2f}, {ego_position[0, 2]:.2f})")
    print(f"  End position:   ({ego_position[-1, 0]:.2f}, {ego_position[-1, 1]:.2f}, {ego_position[-1, 2]:.2f})")
    
    displacement = np.linalg.norm(ego_position[-1] - ego_position[0])
    print(f"  Total displacement: {displacement:.2f} meters")
    
    # Calculate velocities
    if T > 1:
        ego_velocities = np.diff(ego_position, axis=0)
        ego_speeds = np.linalg.norm(ego_velocities, axis=1)
        print(f"  Average speed: {ego_speeds.mean():.2f} m/frame")
        print(f"  Max speed: {ego_speeds.max():.2f} m/frame")
    
    # Check yaw
    if D > 12:
        yaw_range = ego_yaw.max() - ego_yaw.min()
        print(f"  Yaw range: {np.rad2deg(yaw_range):.1f} degrees")
    
    print("\n" + "=" * 70)
    print("✓ Test successful! Data format looks compatible.")
    print("\nNext step: Run full extraction:")
    print(f"  python extract_ego_poses_for_cosmos3.py \\")
    print(f"    --input_dir /scratch/user/u.pt152369/WM/data/datasetWM \\")
    print(f"    --output_dir ./cosmos3_ego_poses \\")
    print(f"    --num_samples 10")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        test_single_sample(sys.argv[1])
    else:
        test_single_sample()
