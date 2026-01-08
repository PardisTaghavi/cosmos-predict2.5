# Check and display H5 kinematic file
import h5py
import numpy as np

h5_file = "/Users/ptgh/Downloads/0061.h5"

with h5py.File(h5_file, 'r') as f:
    frames = f['frames'][:]
    
    print(f"Kinematic Data: {h5_file}")
    print(f"Shape: {frames.shape} [T frames, 32 agents, 18 features]")
    print(f"Total frames: {frames.shape[0]}\n")
    
    class_names = ['ego', 'vehicle', 'pedestrian', 'bicycle']
    
    # Print agent matrix for each frame
    for frame_idx in range(frames.shape[0]):
        print(f"{'='*130}")
        print(f"Frame {frame_idx}:")
        print(f"{'Agent':<6} {'x':<8} {'y':<8} {'z':<8} {'vx':<8} {'vy':<8} {'vz':<8} {'ax':<8} {'ay':<8} {'az':<8} {'l':<6} {'w':<6} {'h':<6} {'yaw':<7} {'ID':<4} {'Class':<15}")
        print("-"*130)
        
        frame_data = frames[frame_idx]
        for agent_idx in range(frames.shape[1]):
            agent = frame_data[agent_idx]
            
            # Check if agent is valid (has non-zero position or is ego)
            if agent_idx == 0 or np.any(agent[:3] != 0):
                class_vec = agent[14:18]  # Classes at indices 14-17
                class_name = class_names[np.argmax(class_vec)] if np.any(class_vec > 0) else 'empty'
                tracking_id = int(agent[13])  # tracking_id at index 13
                yaw_deg = np.degrees(agent[12])  # yaw at index 12
                
                print(f"{agent_idx:<6} "
                      f"{agent[0]:<8.2f} {agent[1]:<8.2f} {agent[2]:<8.2f} "
                      f"{agent[3]:<8.2f} {agent[4]:<8.2f} {agent[5]:<8.2f} "
                      f"{agent[6]:<8.2f} {agent[7]:<8.2f} {agent[8]:<8.2f} "
                      f"{agent[9]:<6.2f} {agent[10]:<6.2f} {agent[11]:<6.2f} "
                      f"{yaw_deg:<7.1f} {tracking_id:<4} {class_name:<15}")
        
        print()
