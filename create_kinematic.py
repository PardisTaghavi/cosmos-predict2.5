#create dataset
#Nuscenes dataset

import os
import json
import numpy as np
import h5py

path = "/Users/ptgh/Desktop/code/cosmos-predict2.5/data"
cam_front="/Users/ptgh/Desktop/code/cosmos-predict2.5/data/v1.0-mini/samples/CAM_FRONT"

#created dataset path 
new_path = "/Users/ptgh/Desktop/code/cosmos-predict2.5/data"
path_kinematic = os.path.join(new_path, "kinematics")
os.makedirs(path_kinematic, exist_ok=True)
path_video = os.path.join(new_path, "videos")
os.makedirs(path_video, exist_ok=True)

#--videos [save videos with original size high quality]
#-----0061.mp4
#-----0103.mp4
#-----etc.
#--kinematics
#-----0061.h5
#-----0103.h5
#-----etc.


#videos: T frames
#fps for Nuscenes dataset: 2 Hz
#num Agents: 32

'''
Kinematic info:

object class:
1) ego
2) vehicles
3) pedestrians
4) bicycles

only objects with visibility>80% and visible in front camera

for each frame: [32, 18]
[[0, 0, 0, vx, vy, vz, ax, ay, az, l, w, h, yaw, tracking_id, 4 class one-hot vector], -->ego
 [x, y, z, vx, vy, vz, ax, ay, az, l, w, h, yaw, tracking_id, 4 class one-hot vector], --> agent 1
 ...]

x,y,z in meters wrt ego vehicle (x forward, y left, z up)
l, w, h: object dimensions in meters (length, width, height)
yaw: rotation angle around z-axis (vertical) in radians, represents object heading/direction
tracking_id: integer ID (0-31) for tracking objects across frames (0=ego, 1-31=other agents)

for each video save as .h5 file
[T, 32, 18]

'''

#nuscenes info:
#/Users/ptgh/Desktop/code/cosmos-predict2.5/data/v1.0-mini/v1.0-mini
#calibrated_sensor.json
#attribute.json
#scene.json
#sample.json
#sample_data.json
#sample_annotation.json
#sensor.json
#ego_pose.json
#log.json
#instance.json
#map.json
#category.json
#visibility.json


# Load NuScenes JSON files
nuscenes_root = os.path.join(path, "v1.0-mini", "v1.0-mini")

with open(os.path.join(nuscenes_root, "scene.json"), "r") as f:
    scenes = json.load(f)

with open(os.path.join(nuscenes_root, "sample.json"), "r") as f:
    samples = json.load(f)

with open(os.path.join(nuscenes_root, "sample_annotation.json"), "r") as f:
    annotations = json.load(f)

with open(os.path.join(nuscenes_root, "ego_pose.json"), "r") as f:
    ego_poses = json.load(f)

with open(os.path.join(nuscenes_root, "calibrated_sensor.json"), "r") as f:
    calibrated_sensors = json.load(f)

with open(os.path.join(nuscenes_root, "sample_data.json"), "r") as f:
    sample_data = json.load(f)

with open(os.path.join(nuscenes_root, "category.json"), "r") as f:
    categories = json.load(f)

with open(os.path.join(nuscenes_root, "instance.json"), "r") as f:
    instances = json.load(f)

# Create lookup dictionaries for efficient access
sample_dict = {s["token"]: s for s in samples}
annotation_dict = {a["token"]: a for a in annotations}
ego_pose_dict = {ep["token"]: ep for ep in ego_poses}
calibrated_sensor_dict = {cs["token"]: cs for cs in calibrated_sensors}
sample_data_dict = {sd["token"]: sd for sd in sample_data}
category_dict = {cat["token"]: cat["name"] for cat in categories}
instance_dict = {inst["token"]: inst for inst in instances}

# Build sample -> annotations mapping
sample_to_annotations = {}
for ann in annotations:
    sample_token = ann["sample_token"]
    if sample_token not in sample_to_annotations:
        sample_to_annotations[sample_token] = []
    sample_to_annotations[sample_token].append(ann)

# Build sample -> ego_pose mapping (via sample_data, using CAM_FRONT key frames)
cam_front_samples = [
    sd for sd in sample_data
    if "CAM_FRONT" in sd.get("filename", "") and sd.get("is_key_frame", False)
]

sample_to_ego_pose = {}
sample_to_cam_front_data = {}
for sd in cam_front_samples:
    sample_token = sd["sample_token"]
    ego_pose_token = sd["ego_pose_token"]
    sample_to_ego_pose[sample_token] = ego_pose_token
    sample_to_cam_front_data[sample_token] = sd

# Category to class mapping
def get_class_onehot(category_name):
    """
    Convert NuScenes category name to one-hot vector [ego, vehicle, pedestrian, bicycle]
    
    NuScenes categories:
    - vehicle.* (except vehicle.bicycle): vehicle class
    - human.pedestrian.*: pedestrian class  
    - vehicle.bicycle: bicycle class
    - Others: unknown/none
    """
    # Vehicle categories (excluding bicycle)
    vehicle_categories = [
        'vehicle.car',
        'vehicle.truck',
        'vehicle.bus.bendy',
        'vehicle.bus.rigid',
        'vehicle.construction',
        'vehicle.emergency.ambulance',
        'vehicle.emergency.police',
        'vehicle.trailer',
    ]
    
    # Pedestrian categories
    pedestrian_categories = [
        'human.pedestrian.adult',
        'human.pedestrian.child',
        'human.pedestrian.wheelchair',
        'human.pedestrian.stroller',
        'human.pedestrian.personal_mobility',
        'human.pedestrian.police_officer',
        'human.pedestrian.construction_worker',
    ]
    
    # Bicycle category
    bicycle_categories = [
        'vehicle.bicycle',
        'vehicle.motorcycle',
    ]
    
    if category_name in vehicle_categories:
        return [0, 1, 0, 0]  # vehicle
    elif category_name in pedestrian_categories:
        return [0, 0, 1, 0]  # pedestrian
    elif category_name in bicycle_categories:
        return [0, 0, 0, 1]  # bicycle
    else:
        return [0, 0, 0, 0]  # unknown/other/none (e.g., animal, movable_object.*, static_object.*)

def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to rotation matrix"""
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    return R

def quaternion_to_yaw(q):
    """
    Extract yaw angle (rotation around z-axis) from quaternion [w, x, y, z].
    Yaw represents the heading/direction of the object in the ground plane.
    
    Args:
        q: quaternion [w, x, y, z]
    
    Returns:
        yaw: angle in radians (rotation around z-axis)
    """
    w, x, y, z = q
    # Yaw from quaternion: yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return yaw

def transform_to_ego_frame(world_pos, ego_translation, ego_rotation):
    """
    Transform world coordinates to ego-relative coordinates.
    NuScenes uses: x forward, y left, z up (same as vehicle frame)
    
    Args:
        world_pos: [x, y, z] in world coordinates
        ego_translation: [x, y, z] ego position in world
        ego_rotation: quaternion [w, x, y, z] ego rotation
    
    Returns:
        [x, y, z] in ego frame (x forward, y left, z up)
    """
    # Convert to numpy
    world_pos = np.array(world_pos)
    ego_translation = np.array(ego_translation)
    
    # Get rotation matrix from quaternion
    R = quaternion_to_rotation_matrix(ego_rotation)
    
    # Transform: ego_frame = R^T @ (world_pos - ego_translation)
    relative_pos = world_pos - ego_translation
    ego_frame_pos = R.T @ relative_pos
    
    return ego_frame_pos

def is_in_camera_fov(obj_pos_ego, cam_intrinsics, image_width=1600, image_height=900, max_distance=100.0):
    """
    Check if object is in camera FOV by projecting 3D position to image plane.
    
    Args:
        obj_pos_ego: [x, y, z] in ego frame (x forward, y left, z up)
        cam_intrinsics: camera intrinsics matrix (list or array) [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        image_width: image width in pixels
        image_height: image height in pixels
        max_distance: maximum distance to consider (meters)
    
    Returns:
        bool: True if object projects to image plane and is within bounds
    """
    x, y, z = obj_pos_ego
    
    # Object must be in front of camera
    if x <= 0:
        return False
    
    # Check distance
    distance = np.sqrt(x**2 + y**2 + z**2)
    if distance > max_distance:
        return False
    
    # Convert camera intrinsics to numpy array
    try:
        K = np.array(cam_intrinsics)
        if K.shape != (3, 3):
            # Fallback to simple check if intrinsics are malformed
            return x > 0 and distance < max_distance and abs(y) < 50 and abs(z) < 10
    except:
        # Fallback to simple check if intrinsics can't be parsed
        return x > 0 and distance < max_distance and abs(y) < 50 and abs(z) < 10
    
    # Transform from ego frame to camera frame
    # NuScenes ego frame: x forward, y left, z up
    # Standard camera frame: x right, y down, z forward
    # Transformation: camera_x = -ego_y, camera_y = -ego_z, camera_z = ego_x
    cam_x = -y  # Right in image (positive = right)
    cam_y = -z  # Down in image (positive = down)
    cam_z = x   # Forward (depth, positive = in front)
    
    # Project to image plane using pinhole camera model
    # u = fx * (X/Z) + cx
    # v = fy * (Y/Z) + cy
    if cam_z <= 0:  # Behind camera
        return False
    
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    u = fx * (cam_x / cam_z) + cx
    v = fy * (cam_y / cam_z) + cy
    
    # Check if projected point is within image bounds (with some margin for objects partially visible)
    margin = 100  # pixels margin (objects can be partially out of frame)
    in_bounds = (u >= -margin and u <= image_width + margin and 
                 v >= -margin and v <= image_height + margin)
    
    return in_bounds

# Process each scene
for scene in scenes:
    scene_name = scene["name"]
    scene_token = scene["token"]
    first_sample_token = scene["first_sample_token"]
    
    print(f"\n{'='*70}")
    print(f"Processing Scene: {scene_name}")
    print(f"{'='*70}")
    
    # Collect all frames for this scene
    frames_data = []
    current_sample_token = first_sample_token
    frame_idx = 0
    
    # Store previous ego pose for velocity/acceleration computation
    prev_ego_translation = None
    prev_ego_rotation = None
    prev_ego_velocity_world = None  # Velocity in world frame (for derivatives and relative computation)
    prev_ego_timestamp = None  # Previous frame timestamp (in microseconds)
    
    # Track agent trajectories: instance_token -> list of (timestamp, world_pos, ego_translation)
    # Store ego_translation at agent's timestamp for correct relative velocity computation
    agent_trajectories = {}  # instance_token -> [(timestamp, world_pos, ego_translation), ...]
    # Track previous agent relative velocities in world frame for acceleration computation
    prev_agent_relative_velocities_world = {}  # instance_token -> relative_velocity_world
    prev_agent_timestamps = {}  # instance_token -> timestamp
    
    # Create mapping from instance_token to tracking_id (integer 0-31)
    # 0 is reserved for ego, 1-31 for other agents
    instance_token_to_tracking_id = {}  # instance_token -> tracking_id
    next_tracking_id = 1  # Start from 1 (0 is ego)
    
    while current_sample_token:
        sample = sample_dict[current_sample_token]
        
        # Get timestamp from sample (in microseconds)
        current_timestamp = sample.get("timestamp")
        if current_timestamp is None or current_timestamp == "":
            current_sample_token = sample.get("next")
            frame_idx += 1
            continue
        
        current_timestamp = int(current_timestamp)
        
        # Get ego pose
        if current_sample_token not in sample_to_ego_pose:
            current_sample_token = sample.get("next")
            frame_idx += 1
            continue
            
        ego_pose_token = sample_to_ego_pose[current_sample_token]
        ego_pose = ego_pose_dict[ego_pose_token]
        ego_translation = np.array(ego_pose["translation"])  # [x, y, z]
        ego_rotation = ego_pose["rotation"]  # [w, x, y, z] quaternion
        
        # Compute dt from timestamps (convert microseconds to seconds)
        dt_ego = None
        if prev_ego_timestamp is not None:
            dt_ego = (current_timestamp - prev_ego_timestamp) / 1e6  # Convert to seconds
            # Sanity check: dt should be reasonable (between 0.1 and 10 seconds)
            if dt_ego < 0.1 or dt_ego > 10.0:
                print(f"  Warning: Unusual dt_ego={dt_ego:.3f}s at frame {frame_idx}, using default 0.5s")
                dt_ego = 0.5
        
        # Compute ego velocity and acceleration
        # FIX: Compute all derivatives in world frame, then rotate to ego frame
        ego_velocity_ego = np.zeros(3, dtype=np.float32)
        ego_acceleration_ego = np.zeros(3, dtype=np.float32)
        ego_velocity_world = np.zeros(3, dtype=np.float32)
        ego_acceleration_world = np.zeros(3, dtype=np.float32)
        
        if prev_ego_translation is not None and dt_ego is not None:
            # Step 1: Compute velocity in world frame
            ego_velocity_world = (ego_translation - prev_ego_translation) / dt_ego
            
            # Step 2: Compute acceleration in world frame (using previous world velocity)
            if prev_ego_velocity_world is not None:
                ego_acceleration_world = (ego_velocity_world - prev_ego_velocity_world) / dt_ego
            
            # Step 3: Rotate both velocity and acceleration to current ego frame
            R_current = quaternion_to_rotation_matrix(ego_rotation)
            ego_velocity_ego = R_current.T @ ego_velocity_world
            ego_acceleration_ego = R_current.T @ ego_acceleration_world
        
        # Store ego velocity for relative computation (before updating prev values)
        current_ego_velocity_world = ego_velocity_world.copy()
        
        # Update previous values for next iteration
        prev_ego_translation = ego_translation.copy()
        prev_ego_rotation = ego_rotation
        prev_ego_velocity_world = ego_velocity_world.copy()
        prev_ego_timestamp = current_timestamp
        
        # Get camera calibration for CAM_FRONT
        cam_front_sd = sample_to_cam_front_data.get(current_sample_token)
        if not cam_front_sd:
            current_sample_token = sample.get("next")
            frame_idx += 1
            continue
            
        calibrated_sensor_token = cam_front_sd["calibrated_sensor_token"]
        cam_calib = calibrated_sensor_dict[calibrated_sensor_token]
        cam_intrinsics = cam_calib["camera_intrinsic"]
        
        # Get annotations for this sample
        sample_annotations = sample_to_annotations.get(current_sample_token, [])
        
        # Filter annotations by visibility > 80% (visibility_token == "4")
        VISIBILITY_THRESHOLD_TOKEN = "4"  # v80-100%
        visible_annotations = [
            ann for ann in sample_annotations
            if ann["visibility_token"] == VISIBILITY_THRESHOLD_TOKEN
        ]
        
        # Process each visible annotation
        frame_agents = []
        
        # Add ego vehicle first (always present)
        # Format: [0, 0, 0, vx, vy, vz, ax, ay, az, l, w, h, yaw, tracking_id, ego_class]
        # Ego dimensions: approximate vehicle size (can be set to 0 or typical car size)
        ego_l, ego_w, ego_h = 4.5, 1.8, 1.5  # Typical car dimensions (length, width, height in meters)
        # Ego yaw: in ego frame, ego is always at yaw=0 (ego is the reference)
        ego_yaw = 0.0
        ego_class = [1, 0, 0, 0]  # ego one-hot
        ego_tracking_id = 0.0  # Ego always has tracking_id = 0
        ego_data = np.array([
            0.0, 0.0, 0.0,  # x, y, z (always 0 for ego)
            float(ego_velocity_ego[0]),   # vx (forward)
            float(ego_velocity_ego[1]),   # vy (left)
            float(ego_velocity_ego[2]),   # vz (up)
            float(ego_acceleration_ego[0]), # ax (forward)
            float(ego_acceleration_ego[1]), # ay (left)
            float(ego_acceleration_ego[2]), # az (up)
            float(ego_l),  # l (length)
            float(ego_w),  # w (width)
            float(ego_h),  # h (height)
            float(ego_yaw),  # yaw (rotation around z-axis in radians)
            ego_tracking_id,  # tracking_id (0 for ego)
        ] + ego_class, dtype=np.float32)
        frame_agents.append(ego_data)
        
        # Process other agents
        for ann in visible_annotations:
            world_pos = np.array(ann["translation"])  # [x, y, z] in world coordinates
            instance_token = ann["instance_token"]
            
            # Track agent trajectory with timestamp and ego position at that time
            if instance_token not in agent_trajectories:
                agent_trajectories[instance_token] = []
            # Store: (timestamp, world_pos, ego_translation)
            agent_trajectories[instance_token].append((current_timestamp, world_pos.copy(), ego_translation.copy()))
            
            # Transform to ego frame for position
            ego_pos = transform_to_ego_frame(world_pos, ego_translation, ego_rotation)
            
            # Check if in camera FOV (project to image plane)
            if not is_in_camera_fov(ego_pos, cam_intrinsics, image_width=1600, image_height=900):
                continue
            
            # Get category and convert to class
            instance = instance_dict.get(instance_token, {})
            category_token = instance.get("category_token", "")
            category_name = category_dict.get(category_token, "unknown")
            class_onehot = get_class_onehot(category_name)
            
            # Skip if unknown class
            if class_onehot == [0, 0, 0, 0]:
                continue
            
            # Get or assign tracking_id for this instance
            if instance_token not in instance_token_to_tracking_id:
                if next_tracking_id >= 32:
                    # Skip if we've reached max agents (0-31, where 0 is ego)
                    continue
                instance_token_to_tracking_id[instance_token] = next_tracking_id
                next_tracking_id += 1
            tracking_id = float(instance_token_to_tracking_id[instance_token])
            
            # Get object dimensions (l, w, h) from annotation
            # NuScenes size format: [width, length, height] in meters
            # We want: [length, width, height] where length is along x (forward), width along y (left), height along z (up)
            obj_size = ann.get("size", [0.0, 0.0, 0.0])
            if isinstance(obj_size, list) and len(obj_size) >= 3:
                # Convert from NuScenes [width, length, height] to our [length, width, height]
                obj_w_nuscenes = float(obj_size[0])  # NuScenes width
                obj_l_nuscenes = float(obj_size[1])   # NuScenes length
                obj_h = float(obj_size[2])            # Height (same)
                # Our convention: l=length (forward/x), w=width (left/y), h=height (up/z)
                obj_l = obj_l_nuscenes  # Length along x (forward)
                obj_w = obj_w_nuscenes  # Width along y (left)
                obj_h = obj_h           # Height along z (up)
            else:
                obj_l, obj_w, obj_h = 0.0, 0.0, 0.0
            
            # Get object rotation (yaw angle) from annotation
            # Rotation is relative to world frame, but we want it relative to ego frame
            obj_rotation_world = ann.get("rotation", [1.0, 0.0, 0.0, 0.0])  # Default: no rotation
            if isinstance(obj_rotation_world, list) and len(obj_rotation_world) == 4:
                # Extract yaw from world rotation
                yaw_world = quaternion_to_yaw(obj_rotation_world)
                # Get ego yaw in world frame
                ego_yaw_world = quaternion_to_yaw(ego_rotation)
                # Transform yaw to ego frame: subtract ego yaw
                obj_yaw = yaw_world - ego_yaw_world
                # Normalize to [-pi, pi]
                obj_yaw = np.arctan2(np.sin(obj_yaw), np.cos(obj_yaw))
            else:
                obj_yaw = 0.0
            
            # Compute agent velocity and acceleration relative to ego
            agent_velocity_ego = np.zeros(3, dtype=np.float32)
            agent_acceleration_ego = np.zeros(3, dtype=np.float32)
            
            # Get trajectory for this agent
            traj = agent_trajectories[instance_token]
            if len(traj) >= 2:  # Need at least 2 frames for velocity
                # Get previous timestamp, position, and ego position from when THIS agent was last seen
                # Format: (timestamp, world_pos, ego_translation)
                prev_timestamp, prev_world_pos, prev_ego_trans = traj[-2]
                
                # Ensure prev_world_pos and prev_ego_trans are numpy arrays
                prev_world_pos = np.array(prev_world_pos)
                prev_ego_trans = np.array(prev_ego_trans)
                
                # Compute dt from timestamps (convert microseconds to seconds)
                # dt_agent = time difference between current appearance and last appearance of THIS agent
                dt_agent = (current_timestamp - prev_timestamp) / 1e6  # Convert to seconds
                
                # Sanity check: dt should be reasonable
                if dt_agent < 0.1 or dt_agent > 10.0:
                    # Skip velocity computation if dt is unreasonable
                    dt_agent = None
                
                if dt_agent is not None:
                    # Step 1: Compute displacements over the SAME dt_agent interval
                    agent_disp = world_pos - prev_world_pos
                    ego_disp = ego_translation - prev_ego_trans
                    
                    # Step 2: Compute relative displacement and velocity
                    relative_displacement_world = agent_disp - ego_disp
                    relative_velocity_world = relative_displacement_world / dt_agent
                    
                    # Step 3: Compute relative acceleration in world frame (using previous relative velocity)
                    relative_acceleration_world = np.zeros(3, dtype=np.float32)
                    if instance_token in prev_agent_relative_velocities_world and instance_token in prev_agent_timestamps:
                        prev_relative_velocity_world = prev_agent_relative_velocities_world[instance_token]
                        prev_agent_timestamp = prev_agent_timestamps[instance_token]
                        
                        # Compute dt for acceleration (time between velocity measurements)
                        dt_accel = (current_timestamp - prev_agent_timestamp) / 1e6
                        
                        if dt_accel >= 0.1 and dt_accel <= 10.0:
                            # Compute acceleration in world frame (differencing in consistent frame)
                            relative_acceleration_world = (relative_velocity_world - prev_relative_velocity_world) / dt_accel
                    
                    # Step 4: Rotate both relative velocity and acceleration to current ego frame
                    R_current = quaternion_to_rotation_matrix(ego_rotation)
                    agent_velocity_ego = R_current.T @ relative_velocity_world
                    agent_acceleration_ego = R_current.T @ relative_acceleration_world
                    
                    # Store current relative velocity in world frame and timestamp for next frame
                    prev_agent_relative_velocities_world[instance_token] = relative_velocity_world.copy()
                    prev_agent_timestamps[instance_token] = current_timestamp
            
            # Format: [x, y, z, vx, vy, vz, ax, ay, az, l, w, h, yaw, tracking_id, class_onehot]
            agent_data = np.array([
                float(ego_pos[0]),  # x (forward)
                float(ego_pos[1]),  # y (left)
                float(ego_pos[2]),  # z (up)
                float(agent_velocity_ego[0]),  # vx (relative velocity forward)
                float(agent_velocity_ego[1]),  # vy (relative velocity left)
                float(agent_velocity_ego[2]),  # vz (relative velocity up)
                float(agent_acceleration_ego[0]),  # ax (relative acceleration forward)
                float(agent_acceleration_ego[1]),  # ay (relative acceleration left)
                float(agent_acceleration_ego[2]),  # az (relative acceleration up)
                obj_l,  # l (length)
                obj_w,  # w (width)
                obj_h,  # h (height)
                float(obj_yaw),  # yaw (rotation around z-axis in radians, relative to ego)
                tracking_id,  # tracking_id (1-31)
            ] + class_onehot, dtype=np.float32)
            
            frame_agents.append(agent_data)
        
        # Pad or truncate to 32 agents
        MAX_AGENTS = 32
        FEATURES_PER_AGENT = 18  # [x, y, z, vx, vy, vz, ax, ay, az, l, w, h, yaw, tracking_id, 4 classes]
        if len(frame_agents) < MAX_AGENTS:
            # Pad with zeros
            padding = np.zeros((MAX_AGENTS - len(frame_agents), FEATURES_PER_AGENT), dtype=np.float32)
            frame_agents.extend(padding)
        else:
            # Truncate to MAX_AGENTS
            frame_agents = frame_agents[:MAX_AGENTS]
        
        frame_array = np.array(frame_agents)
        frames_data.append(frame_array)
        
        # Print agent matrix for this frame
        print(f"\nFrame {frame_idx} - Agent Matrix [{frame_array.shape[0]}x{frame_array.shape[1]}]:")
        print("="*130)
        print(f"{'Agent':<6} {'x':<8} {'y':<8} {'z':<8} {'vx':<8} {'vy':<8} {'vz':<8} {'ax':<8} {'ay':<8} {'az':<8} {'l':<6} {'w':<6} {'h':<6} {'yaw':<7} {'ID':<4} {'Class':<15}")
        print("-"*130)
        
        class_names = ['ego', 'vehicle', 'pedestrian', 'bicycle']
        for agent_idx in range(MAX_AGENTS):
            agent = frame_array[agent_idx]
            # Check if agent is valid (has non-zero position or is ego)
            if agent_idx == 0 or np.any(agent[:3] != 0):
                class_vec = agent[14:18]  # Classes are now at indices 14-17
                class_name = class_names[np.argmax(class_vec)] if np.any(class_vec > 0) else 'empty'
                tracking_id = int(agent[13])  # tracking_id at index 13
                yaw = agent[12]  # yaw at index 12
                yaw_deg = np.degrees(yaw)
                
                print(f"{agent_idx:<6} "
                      f"{agent[0]:<8.2f} {agent[1]:<8.2f} {agent[2]:<8.2f} "
                      f"{agent[3]:<8.2f} {agent[4]:<8.2f} {agent[5]:<8.2f} "
                      f"{agent[6]:<8.2f} {agent[7]:<8.2f} {agent[8]:<8.2f} "
                      f"{agent[9]:<6.2f} {agent[10]:<6.2f} {agent[11]:<6.2f} "
                      f"{yaw_deg:<7.1f} {tracking_id:<4} {class_name:<15}")
        
        if (frame_idx + 1) % 10 == 0:
            print(f"\n  Processed {frame_idx + 1} frames...")
        
        # Move to next sample
        current_sample_token = sample.get("next")
        frame_idx += 1
        
        # Safety limit
        if frame_idx >= 200:
            break
    
    if len(frames_data) == 0:
        print(f"  ⚠️  No frames collected for {scene_name}, skipping...")
        continue
    
    # Convert to numpy array: [T, 32, 18]
    kinematic_data = np.array(frames_data, dtype=np.float32)
    
    # Save to .h5 file
    scene_number = scene_name.split('-')[-1]  # Extract number from "scene-0061"
    output_filename = f"{scene_number}.h5"
    output_path = os.path.join(path_kinematic, output_filename)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('frames', data=kinematic_data, compression='gzip')
    
    print(f"  ✅ Saved {kinematic_data.shape[0]} frames to {output_filename}")
    print(f"     Shape: {kinematic_data.shape} [T, 32, 18]")
    print(f"     Agents per frame (avg): {np.sum(np.any(kinematic_data[:, :, :3] != 0, axis=2), axis=1).mean():.1f}")
    
    # Debug break after first scene
    if scene_token == scenes[0]["token"]:
        print(f"\n{'='*70}")
        print("🔍 DEBUG: First scene processed. Breaking for inspection...")
        print(f"DEBUG: name of the scene: {scene_name}")
        print(f"{'='*70}")
        break

print(f"\n{'='*70}")
print("✅ Kinematic dataset creation complete!")
print(f"{'='*70}")
