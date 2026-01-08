#!/usr/bin/env python3
"""
Extract first N frames from a video and save as a new video at specified frame rate.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2


def extract_frames_to_video(input_path: str, output_path: str, num_frames: int = 5, fps: float = 2.0):
    """
    Extract first N frames from input video and save as a new video.
    Uses ffmpeg for reliable video encoding.
    
    Args:
        input_path: Path to input video file
        output_path: Path to save output video
        num_frames: Number of frames to extract (default: 5)
        fps: Frame rate for output video in Hz (default: 2.0)
    """
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        use_ffmpeg = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        use_ffmpeg = False
        print("Warning: ffmpeg not found, falling back to OpenCV")
    
    if use_ffmpeg:
        # Use ffmpeg for reliable video creation
        duration = num_frames / fps
        
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-t', str(duration),  # Duration in seconds
            '-r', str(fps),  # Output frame rate
            '-c:v', 'libx264',  # H.264 codec
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            '-y',  # Overwrite output file
            output_path
        ]
        
        print(f"Using ffmpeg to extract first {num_frames} frames...")
        print(f"Output video: {output_path}")
        print(f"Output FPS: {fps} Hz")
        print(f"Duration: {duration:.2f} seconds")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"\nSuccessfully created mini video:")
            print(f"  Output: {output_path}")
            print(f"  Frames: {num_frames}")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  FPS: {fps} Hz")
            return
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed: {e.stderr}")
            print("Falling back to OpenCV method...")
    
    # Fallback to OpenCV method
    # Open input video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {input_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {original_fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/original_fps:.2f} seconds")
    
    # Check if we have enough frames
    if total_frames < num_frames:
        print(f"Warning: Video has only {total_frames} frames, but {num_frames} requested.")
        num_frames = total_frames
    
    # Setup video writer with H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        # Try alternative codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise ValueError(f"Failed to create output video: {output_path}")
    
    print(f"\nExtracting first {num_frames} frames...")
    print(f"Output video: {output_path}")
    print(f"Output FPS: {fps} Hz")
    
    frames_written = 0
    
    while frames_written < num_frames:
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Reached end of video at frame {frames_written}")
            break
        
        # Ensure frame is in correct format (BGR for OpenCV)
        if frame is None:
            print(f"Warning: Got None frame at index {frames_written}")
            break
        
        # Verify frame shape matches expected dimensions
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        
        # Write frame to output video
        out.write(frame)
        frames_written += 1
        
        if frames_written % 10 == 0 or frames_written == num_frames:
            print(f"  Written {frames_written}/{num_frames} frames...")
    
    # Release resources
    cap.release()
    out.release()
    
    # Verify the video was created correctly
    verify_cap = cv2.VideoCapture(output_path)
    if verify_cap.isOpened():
        verify_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        verify_cap.release()
        if verify_frames == 0:
            print(f"Warning: Output video appears to be empty. Trying alternative method...")
            # Try using imageio or moviepy as fallback
            raise ValueError("OpenCV method failed to create valid video")
    
    print(f"\nSuccessfully created mini video:")
    print(f"  Output: {output_path}")
    print(f"  Frames: {frames_written}")
    print(f"  Duration: {frames_written/fps:.2f} seconds")
    print(f"  FPS: {fps} Hz")


def main():
    parser = argparse.ArgumentParser(description="Extract first N frames from a video")
    parser.add_argument("input_video", type=str, help="Path to input video file")
    parser.add_argument("output_video", type=str, help="Path to save output video")
    parser.add_argument("--num-frames", type=int, default=5, help="Number of frames to extract (default: 5)")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame rate for output video in Hz (default: 2.0)")
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    
    # Create output directory if needed
    output_path = Path(args.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    extract_frames_to_video(
        input_path=str(input_path),
        output_path=str(output_path),
        num_frames=args.num_frames,
        fps=args.fps
    )


if __name__ == "__main__":
    main()


Extract first N frames from a video and save as a new video at specified frame rate.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2


def extract_frames_to_video(input_path: str, output_path: str, num_frames: int = 5, fps: float = 2.0):
    """
    Extract first N frames from input video and save as a new video.
    Uses ffmpeg for reliable video encoding.
    
    Args:
        input_path: Path to input video file
        output_path: Path to save output video
        num_frames: Number of frames to extract (default: 5)
        fps: Frame rate for output video in Hz (default: 2.0)
    """
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        use_ffmpeg = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        use_ffmpeg = False
        print("Warning: ffmpeg not found, falling back to OpenCV")
    
    if use_ffmpeg:
        # Use ffmpeg for reliable video creation
        duration = num_frames / fps
        
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-t', str(duration),  # Duration in seconds
            '-r', str(fps),  # Output frame rate
            '-c:v', 'libx264',  # H.264 codec
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            '-y',  # Overwrite output file
            output_path
        ]
        
        print(f"Using ffmpeg to extract first {num_frames} frames...")
        print(f"Output video: {output_path}")
        print(f"Output FPS: {fps} Hz")
        print(f"Duration: {duration:.2f} seconds")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"\nSuccessfully created mini video:")
            print(f"  Output: {output_path}")
            print(f"  Frames: {num_frames}")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  FPS: {fps} Hz")
            return
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed: {e.stderr}")
            print("Falling back to OpenCV method...")
    
    # Fallback to OpenCV method
    # Open input video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {input_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {original_fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/original_fps:.2f} seconds")
    
    # Check if we have enough frames
    if total_frames < num_frames:
        print(f"Warning: Video has only {total_frames} frames, but {num_frames} requested.")
        num_frames = total_frames
    
    # Setup video writer with H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        # Try alternative codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise ValueError(f"Failed to create output video: {output_path}")
    
    print(f"\nExtracting first {num_frames} frames...")
    print(f"Output video: {output_path}")
    print(f"Output FPS: {fps} Hz")
    
    frames_written = 0
    
    while frames_written < num_frames:
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Reached end of video at frame {frames_written}")
            break
        
        # Ensure frame is in correct format (BGR for OpenCV)
        if frame is None:
            print(f"Warning: Got None frame at index {frames_written}")
            break
        
        # Verify frame shape matches expected dimensions
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        
        # Write frame to output video
        out.write(frame)
        frames_written += 1
        
        if frames_written % 10 == 0 or frames_written == num_frames:
            print(f"  Written {frames_written}/{num_frames} frames...")
    
    # Release resources
    cap.release()
    out.release()
    
    # Verify the video was created correctly
    verify_cap = cv2.VideoCapture(output_path)
    if verify_cap.isOpened():
        verify_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        verify_cap.release()
        if verify_frames == 0:
            print(f"Warning: Output video appears to be empty. Trying alternative method...")
            # Try using imageio or moviepy as fallback
            raise ValueError("OpenCV method failed to create valid video")
    
    print(f"\nSuccessfully created mini video:")
    print(f"  Output: {output_path}")
    print(f"  Frames: {frames_written}")
    print(f"  Duration: {frames_written/fps:.2f} seconds")
    print(f"  FPS: {fps} Hz")


def main():
    parser = argparse.ArgumentParser(description="Extract first N frames from a video")
    parser.add_argument("input_video", type=str, help="Path to input video file")
    parser.add_argument("output_video", type=str, help="Path to save output video")
    parser.add_argument("--num-frames", type=int, default=5, help="Number of frames to extract (default: 5)")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame rate for output video in Hz (default: 2.0)")
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    
    # Create output directory if needed
    output_path = Path(args.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    extract_frames_to_video(
        input_path=str(input_path),
        output_path=str(output_path),
        num_frames=args.num_frames,
        fps=args.fps
    )


if __name__ == "__main__":
    main()


Extract first N frames from a video and save as a new video at specified frame rate.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2


def extract_frames_to_video(input_path: str, output_path: str, num_frames: int = 5, fps: float = 2.0):
    """
    Extract first N frames from input video and save as a new video.
    Uses ffmpeg for reliable video encoding.
    
    Args:
        input_path: Path to input video file
        output_path: Path to save output video
        num_frames: Number of frames to extract (default: 5)
        fps: Frame rate for output video in Hz (default: 2.0)
    """
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        use_ffmpeg = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        use_ffmpeg = False
        print("Warning: ffmpeg not found, falling back to OpenCV")
    
    if use_ffmpeg:
        # Use ffmpeg for reliable video creation
        duration = num_frames / fps
        
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-t', str(duration),  # Duration in seconds
            '-r', str(fps),  # Output frame rate
            '-c:v', 'libx264',  # H.264 codec
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            '-y',  # Overwrite output file
            output_path
        ]
        
        print(f"Using ffmpeg to extract first {num_frames} frames...")
        print(f"Output video: {output_path}")
        print(f"Output FPS: {fps} Hz")
        print(f"Duration: {duration:.2f} seconds")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"\nSuccessfully created mini video:")
            print(f"  Output: {output_path}")
            print(f"  Frames: {num_frames}")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  FPS: {fps} Hz")
            return
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed: {e.stderr}")
            print("Falling back to OpenCV method...")
    
    # Fallback to OpenCV method
    # Open input video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {input_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {original_fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/original_fps:.2f} seconds")
    
    # Check if we have enough frames
    if total_frames < num_frames:
        print(f"Warning: Video has only {total_frames} frames, but {num_frames} requested.")
        num_frames = total_frames
    
    # Setup video writer with H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        # Try alternative codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise ValueError(f"Failed to create output video: {output_path}")
    
    print(f"\nExtracting first {num_frames} frames...")
    print(f"Output video: {output_path}")
    print(f"Output FPS: {fps} Hz")
    
    frames_written = 0
    
    while frames_written < num_frames:
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Reached end of video at frame {frames_written}")
            break
        
        # Ensure frame is in correct format (BGR for OpenCV)
        if frame is None:
            print(f"Warning: Got None frame at index {frames_written}")
            break
        
        # Verify frame shape matches expected dimensions
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        
        # Write frame to output video
        out.write(frame)
        frames_written += 1
        
        if frames_written % 10 == 0 or frames_written == num_frames:
            print(f"  Written {frames_written}/{num_frames} frames...")
    
    # Release resources
    cap.release()
    out.release()
    
    # Verify the video was created correctly
    verify_cap = cv2.VideoCapture(output_path)
    if verify_cap.isOpened():
        verify_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        verify_cap.release()
        if verify_frames == 0:
            print(f"Warning: Output video appears to be empty. Trying alternative method...")
            # Try using imageio or moviepy as fallback
            raise ValueError("OpenCV method failed to create valid video")
    
    print(f"\nSuccessfully created mini video:")
    print(f"  Output: {output_path}")
    print(f"  Frames: {frames_written}")
    print(f"  Duration: {frames_written/fps:.2f} seconds")
    print(f"  FPS: {fps} Hz")


def main():
    parser = argparse.ArgumentParser(description="Extract first N frames from a video")
    parser.add_argument("input_video", type=str, help="Path to input video file")
    parser.add_argument("output_video", type=str, help="Path to save output video")
    parser.add_argument("--num-frames", type=int, default=5, help="Number of frames to extract (default: 5)")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame rate for output video in Hz (default: 2.0)")
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    
    # Create output directory if needed
    output_path = Path(args.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    extract_frames_to_video(
        input_path=str(input_path),
        output_path=str(output_path),
        num_frames=args.num_frames,
        fps=args.fps
    )


if __name__ == "__main__":
    main()

