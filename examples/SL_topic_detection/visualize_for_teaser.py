import argparse
import cv2
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

from pose_format import Pose

import numpy as np
import numpy.ma as ma
import copy

#In theory, we want to do the pose_visualizer for that. So this should not be needed.
POSE_LIMBS = [(15, 21), (16, 20), (18, 20), (3, 7), (14, 16), (11, 23), (6, 8), (15, 17), (16, 22), (4, 5), (5, 6), (12, 24), (23, 24), (0, 1), (9, 10), (1, 2), (0, 4), (11, 13), (15, 19), (16, 18), (12, 14), (17, 19), (2, 3), (11, 12), (13, 15)]
LEFT_HAND_LIMBS = [(p1 + 25, p2 + 25) for p1, p2 in [(3, 4), (0, 5), (17, 18), (0, 17), (13, 14), (13, 17), (18, 19), (5, 6), (5, 9), (14, 15), (0, 1), (9, 10), (1, 2), (9, 13), (10, 11), (19, 20), (6, 7), (15, 16), (2, 3), (11, 12), (7, 8)]]
RIGHT_HAND_LIMBS = [(p1 + 46, p2 + 46) for p1, p2 in [(3, 4), (0, 5), (17, 18), (0, 17), (13, 14), (13, 17), (18, 19), (5, 6), (5, 9), (14, 15), (0, 1), (9, 10), (1, 2), (9, 13), (10, 11), (19, 20), (6, 7), (15, 16), (2, 3), (11, 12), (7, 8)]]
CONNECTIONS = POSE_LIMBS + LEFT_HAND_LIMBS + RIGHT_HAND_LIMBS

LEFT_HAND_LIMBS_ALONE = [(3, 4), (0, 5), (17, 18), (0, 17), (13, 14), (13, 17), (18, 19), (5, 6), (5, 9), (14, 15), (0, 1), (9, 10), (1, 2), (9, 13), (10, 11), (19, 20), (6, 7), (15, 16), (2, 3), (11, 12), (7, 8)]
RIGHT_HAND_LIMBS_ALONE = [(p1 + 21, p2 + 21) for p1, p2 in [(3, 4), (0, 5), (17, 18), (0, 17), (13, 14), (13, 17), (18, 19), (5, 6), (5, 9), (14, 15), (0, 1), (9, 10), (1, 2), (9, 13), (10, 11), (19, 20), (6, 7), (15, 16), (2, 3), (11, 12), (7, 8)]]
CONNECTIONS_OC = LEFT_HAND_LIMBS_ALONE + RIGHT_HAND_LIMBS_ALONE

def load_video_frames(video_path):
    """
    Loads a video file into a list of RGB frames.

    Args:
        video_path (str or Path): Path to the video file.

    Returns:
        tuple: A tuple containing:
            - list: A list of video frames as NumPy arrays (RGB).
            - float: The frames per second (fps) of the video.
            - tuple: The (width, height) of the video.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert from BGR (OpenCV) to RGB (matplotlib)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"Loaded {len(frames)} frames from video.")
    return frames, fps, (width, height)

def filter_pose_components(pose_obj: Pose) -> Pose:
    """
    Filters a Pose object to keep only specified components if they exist.
    This handles cases where a pose might only have hands.
    """
    available_components = [c.name for c in pose_obj.header.components]
    components_to_keep = []
    points_to_keep = {}

    # Check for and add upper body
    if "POSE_LANDMARKS" in available_components:
        components_to_keep.append("POSE_LANDMARKS")
        # Define the points for the upper body (first 25 points)
        points_to_keep['POSE_LANDMARKS'] = pose_obj.header.components[0].points[:25]
    
    # Check for and add hands
    if "LEFT_HAND_LANDMARKS" in available_components:
        components_to_keep.append("LEFT_HAND_LANDMARKS")
    if "RIGHT_HAND_LANDMARKS" in available_components:
        components_to_keep.append("RIGHT_HAND_LANDMARKS")
        
    return pose_obj.get_components(components=components_to_keep, points=points_to_keep)

def draw_pose(ax, pose_obj, frame_idx, title, is_normalized=False, video_dims=None, conf_threshold=0.3):
    """Draws a single pose skeleton on a given matplotlib Axes object."""
    ax.clear()
    ax.set_title(title, fontsize=10)

    pose_data = pose_obj.body.data[frame_idx][0]
    confidence = pose_obj.body.confidence[frame_idx][0]
    
    if is_normalized: #It should always enter here
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1.5, 1.5)
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        # ADDED: Hide axis labels for normalized plots
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        ax.axis('off')
    elif video_dims:
        width, height = video_dims
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

    valid_points = confidence > conf_threshold
    # TODO: here check what happens when we only have hands.
    all_valid_indices = np.where(valid_points)[0]
    
    if pose_data.shape[0]==67: 
        for p1_idx, p2_idx in CONNECTIONS:
            if p1_idx < len(valid_points) and p2_idx < len(valid_points) and valid_points[p1_idx] and valid_points[p2_idx]:
                p1 = pose_data[p1_idx]
                p2 = pose_data[p2_idx]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', linewidth=1)
    else:
        assert pose_data.shape[0] == 42
        for p1_idx, p2_idx in CONNECTIONS_OC:
            if p1_idx < len(valid_points) and p2_idx < len(valid_points) and valid_points[p1_idx] and valid_points[p2_idx]:
                p1 = pose_data[p1_idx]
                p2 = pose_data[p2_idx]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', linewidth=1)

    if pose_data.shape[0]==67:
        body_indices = [i for i in all_valid_indices if i < 25]
        left_hand_indices = [i for i in all_valid_indices if 25 <= i < 46]
        right_hand_indices = [i for i in all_valid_indices if i >= 46]
    else:
        assert pose_data.shape[0] == 42
        body_indices = None
        left_hand_indices = [i for i in all_valid_indices if 0 <= i < 21]
        right_hand_indices = [i for i in all_valid_indices if  i >= 21]

    if body_indices:
        ax.scatter(pose_data[body_indices, 0], pose_data[body_indices, 1], s=7, color='black', zorder=3)
    if left_hand_indices:
        ax.scatter(pose_data[left_hand_indices, 0], pose_data[left_hand_indices, 1], s=7, color='blue', zorder=3)
    if right_hand_indices:
        ax.scatter(pose_data[right_hand_indices, 0], pose_data[right_hand_indices, 1], s=7, color='green', zorder=3)

LEFT_WRIST_BODY_IDX = 15
RIGHT_WRIST_BODY_IDX = 16

def align_hands_to_body(body_pose_obj: Pose, hand_pose_obj: Pose, confidence_threshold=0.3) -> Pose:
    """
    Aligns the hands of one Pose object to the wrist positions of another.

    Args:
        body_pose_obj (Pose): The reference pose with the anchor wrist positions.
        hand_pose_obj (Pose): The pose object whose hands need to be aligned.
        confidence_threshold (float): Confidence threshold for wrist detections.

    Returns:
        Pose: A new Pose object with the hands correctly aligned.
    """
    # Create a deep copy to avoid modifying the original data
    aligned_pose = copy.deepcopy(hand_pose_obj)

    # Extract the necessary data arrays
    body_data = body_pose_obj.body.data
    body_confidence = body_pose_obj.body.confidence
    
    aligned_data = aligned_pose.body.data
    aligned_confidence = aligned_pose.body.confidence

    # Dynamically determine the structure of the pose object we are aligning
    component_names = [c.name for c in aligned_pose.header.components]
    has_body = "POSE_LANDMARKS" in component_names
    has_left_hand = "LEFT_HAND_LANDMARKS" in component_names
    has_right_hand = "RIGHT_HAND_LANDMARKS" in component_names

    # Define keypoint indices based on the detected structure
    if has_body:
        # Structure: Body (25), Left Hand (21), Right Hand (21)
        left_hand_start_idx = 25
        right_hand_start_idx = 46
    else:
        # Structure: Left Hand (21), Right Hand (21)
        left_hand_start_idx = 0
        right_hand_start_idx = 21 if has_left_hand else 0 # Handle case with only one hand

    num_frames = min(len(body_data), len(aligned_data))
    for i in range(num_frames):
        # --- Align Left Hand ---
        if has_left_hand:
            # Anchor wrist is always from the body pose object
            anchor_wrist_pos = body_data[i, 0, LEFT_WRIST_BODY_IDX]
            anchor_wrist_conf = body_confidence[i, 0, LEFT_WRIST_BODY_IDX]

            # The hand's own wrist position (index depends on structure)
            hand_wrist_idx = left_hand_start_idx # First point of the hand component
            hand_wrist_pos = aligned_data[i, 0, hand_wrist_idx]
            hand_wrist_conf = aligned_confidence[i, 0, hand_wrist_idx]

            if anchor_wrist_conf > confidence_threshold and hand_wrist_conf > confidence_threshold:
                offset = anchor_wrist_pos - hand_wrist_pos
                # Apply offset to all 21 points of the left hand
                aligned_data[i, 0, left_hand_start_idx : left_hand_start_idx + 21] += offset
        
        # --- Align Right Hand ---
        if has_right_hand:
            anchor_wrist_pos = body_data[i, 0, RIGHT_WRIST_BODY_IDX]
            anchor_wrist_conf = body_confidence[i, 0, RIGHT_WRIST_BODY_IDX]
            
            hand_wrist_idx = right_hand_start_idx
            hand_wrist_pos = aligned_data[i, 0, hand_wrist_idx]
            hand_wrist_conf = aligned_confidence[i, 0, hand_wrist_idx]

            if anchor_wrist_conf > confidence_threshold and hand_wrist_conf > confidence_threshold:
                offset = anchor_wrist_pos - hand_wrist_pos
                # Apply offset to all 21 points of the right hand
                aligned_data[i, 0, right_hand_start_idx : right_hand_start_idx + 21] += offset

    return aligned_pose

def main(video_id, partition):
    #Set up the paths for all the files:
    #partition = "test"
    #video_id = "_G0RrDVpOZ4-13"

    #Aquests videos ja estan tallats,
    path_to_video_rgb_front = f"/projects/imva/Egosign/egosign/video_level/rgb/{partition}/{video_id}/{video_id}-rgb_front.mp4"
    path_to_video_rgb_side = f"/projects/imva/Egosign/egosign/video_level/rgb/{partition}/{video_id}/{video_id}-rgb_side.mp4"
    path_to_video_rgb_head = f"/projects/imva/Egosign/egosign/video_level/rgb/{partition}/{video_id}/{video_id}-rgb_head.mp4"
    path_to_video_rgb_ego_inside = f"/projects/imva/Egosign/egosign/video_level/rgb/{partition}/{video_id}/{video_id}-rgb_oc.mp4"

    path_to_videos = {
        "front": path_to_video_rgb_front,
        "side": path_to_video_rgb_side,
        "head": path_to_video_rgb_head,
        "inside": path_to_video_rgb_ego_inside
    }

    #Aquests keypoints estan processats pero no tallats, pero tenim els frames dón s´han tallat. 
    path_to_video_mp_front_postprocessed = f"/projects/imva/Egosign/egosign/video_level/features/mediapipe_new/front_smooth_normalized/{partition}/{video_id}.pose"
    path_to_video_oc_projected_postprocessed = f"/projects/imva/Egosign/egosign/video_level/features/mediapipe_new/oc_3d_to_2d_resectioning/{partition}_smooth_normalized/{video_id}.pose"
    path_to_video_combined_postprocessed = f"/projects/imva/Egosign/egosign/video_level/features/mediapipe_new/combined_mp_smoothed_oc_smoothed_resectioning/{partition}/{video_id}.pose"

    # Load video frames
    print("Loading video and pose files...")
    video_frames, fps, video_dims = {}, {}, {}
    for key in path_to_videos:
        print(f"Loading video frames for {key} from {path_to_videos[key]}")
        video_frames[key], fps[key], video_dims[key] = load_video_frames(path_to_videos[key])
    
    #Load the poses
    print("Loading pose files...")
    with open(path_to_video_mp_front_postprocessed, "rb") as f:
        print(f"Loading pose from {path_to_video_mp_front_postprocessed}...")
        mp_front_postprocessed = Pose.read(f.read())
    with open(path_to_video_oc_projected_postprocessed, "rb") as f:
        print(f"Loading pose from {path_to_video_oc_projected_postprocessed}...")
        oc_projected_postprocessed = Pose.read(f.read())
    with open(path_to_video_combined_postprocessed, "rb") as f:
        print(f"Loading pose from {path_to_video_combined_postprocessed}...")
        combined_postprocessed = Pose.read(f.read())
        
    #Filter to only have upper body + hands
    mp_front_postprocessed_filtered = filter_pose_components(mp_front_postprocessed)
    oc_projected_postprocessed_filtered = filter_pose_components(oc_projected_postprocessed)
    combined_postprocessed_filtered = filter_pose_components(combined_postprocessed)
    
    #print("Aligning hands to front-view body pose...")
    ## Align oc_projected's hands to the wrists of mp_front
    #oc_aligned = align_hands_to_body(
    #    body_pose_obj=mp_front_postprocessed_filtered, 
    #    hand_pose_obj=oc_projected_postprocessed_filtered
    #)

    ## Align combined's hands to the wrists of mp_front
    #combined_aligned = align_hands_to_body(
    #    body_pose_obj=mp_front_postprocessed_filtered, 
    #    hand_pose_obj=combined_postprocessed_filtered
    #)
    #print("Done.")

    print("Setting up visualization...")
    num_frames = min(len(video_frames["front"]), 
                     len(video_frames["side"]),
                     len(video_frames["head"]),
                     len(video_frames["inside"]),
                     mp_front_postprocessed_filtered.body.data.shape[0],
                     oc_projected_postprocessed_filtered.body.data.shape[0],
                     combined_postprocessed_filtered.body.data.shape[0],
                    )
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=300)
    fig.tight_layout(pad=0.5)
    fig.suptitle(f"video visualizations {video_id}", fontsize=16)
    
    def update(frame_idx):
        if (frame_idx + 1) % 20 == 0:
            print(f"Processing frame {frame_idx + 1}/{num_frames}...")
        
        axes[0, 0].clear()
        axes[0, 0].imshow(video_frames['front'][frame_idx])
        axes[0, 0].set_title("rgb_front")
        axes[0, 0].axis('off')
        
        axes[0, 1].clear()
        axes[0, 1].imshow(video_frames['side'][frame_idx])
        axes[0, 1].set_title("rgb_side")
        axes[0, 1].axis('off')
        
        axes[1, 0].clear()
        axes[1, 0].imshow(video_frames['head'][frame_idx])
        axes[1, 0].set_title("rgb_head")
        axes[1, 0].axis('off')
        
        axes[1, 1].clear()
        axes[1, 1].imshow(video_frames['inside'][frame_idx])
        axes[1, 1].set_title("rgb_inside")
        axes[1, 1].axis('off')
        
        draw_pose(axes[0, 2], mp_front_postprocessed_filtered, frame_idx, "mp_front", is_normalized=True)
        draw_pose(axes[0, 3], oc_projected_postprocessed_filtered, frame_idx, "oc_projected", is_normalized=True)
        draw_pose(axes[1, 2], combined_postprocessed_filtered, frame_idx, "combined", is_normalized=True)
        
        axes[1, 3].axis('off')
        
        return axes
    
    #We should save it as an mp4, where each frame has this 2x4 grid of visualizations.
    output_filename = f"/home/ltarres/egosign_final_code/examples/SL_topic_detection/visualizations_teaser/{partition}/{video_id}_2.mp4"
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    
    print("--- DEBUG: Saving first frame as PNG to check plotting ---")
    update(0) # Manually call the update function for the first frame
    debug_frame_path = Path(output_filename).with_suffix('.png')
    fig.savefig(debug_frame_path, dpi=150)
    print(f"--- DEBUG: First frame saved to {debug_frame_path} ---")
    # --- End of debugging block ---
    
    print("Creating animation... This may take a while.")
    #num_frames = 75
    ani = FuncAnimation(fig, update, frames=num_frames,)
    # Assuming all videos have the same fps, using 'front' view's fps.
    ani.save(output_filename, writer='ffmpeg', fps=fps['front'], dpi=250)
    plt.close(fig) # Free up memory
    print(f"\nVisualization saved successfully to: {output_filename}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize video and keypoints comparison for How2Sign and Egosign datasets.")
    parser.add_argument("--id", type=str, required=True, help="11-character base video ID (e.g., fzXgYPSnaDs)")
    parser.add_argument("--partition", type=str, choices=["val", "test"], required=True, help="Dataset partition (val/test)")
    args = parser.parse_args()

    main(args.id, args.partition)

'''
To run this:
salloc --nodes=1 --time=06:00:00 --mem=100G
module load Anaconda3/2023.09-0
module load FFmpeg/4.3.2-GCCcore-10.2.0
eval "$(conda shell.bash hook)"
conda activate env_proves

cd egosign_final_code/examples/SL_topic_detection/
python visualize_for_teaser.py --id="_G0RrDVpOZ4-13" --partition test


The idea with this script is to be able to visualize the videos and poses for egosign. In order to kind of "animate" the teaser figure.

So what we need are:
front, side, head and inside (rgb)
and then MP front, Oc projected and combined (2d keypoints) (using the .pose visualizer, although they are normalized, so we need to "denormalize" them)


#We want to run this for a few files:
_G0RrDVpOZ4-13 test --> this for sure


# To check the memory usage:
srun --jobid=1728869 --pty htop


# Okay, I want a bash script that runs this for a few files:

python visualize_for_teaser.py --id="_G0RrDVpOZ4-13" --partition test
python visualize_for_teaser.py --id="FZCF7kPIyOk-8" --partition test
python visualize_for_teaser.py --id="fZgWKh3ENoE-12" --partition test
python visualize_for_teaser.py --id="fzOH00UZg84-15" --partition test
python visualize_for_teaser.py --id="g0fgci8L_rc-8" --partition test

python visualize_for_teaser.py --id="00dWJ4YRRSI-12" --partition val
python visualize_for_teaser.py --id="2SnVWW3MOB4-12" --partition val
python visualize_for_teaser.py --id="46Cwjrd4ua4-14" --partition val
python visualize_for_teaser.py --id="a5yNwUSiYpA-16" --partition val
python visualize_for_teaser.py --id="cDJuwtDYaSg-15" --partition val

'''