'''
In here we want to visualize the data from egosign, and the different projections. That way we can know when the combination has worked.

It should be similar to visualize_mp_processing_front

But this time, we doing with this columns:

front frame, find in path: /projects/imva/Egosign/egosign/video_level/rgb/test/_fZbAxSSbX4-13/_fZbAxSSbX4-13-rgb_front.mp4
original mediapipe, find in path: /projects/imva/Egosign/egosign/video_level/features/mediapipe_new/front_smooth_normalized/test/_fZbAxSSbX4-13.pose
OC to front homography, find in path: /projects/imva/Egosign/egosign/video_level/features/mediapipe_new/oc_3d_to_2d_homography/test_smooth_normalized/_fZbAxSSbX4-13.pose
OC to front resectioning, find in path: /projects/imva/Egosign/egosign/video_level/features/mediapipe_new/oc_3d_to_2d_resectioning/test_smooth_normalized/_fZbAxSSbX4-13.pose
combined homography, find in path: /projects/imva/Egosign/egosign/video_level/features/mediapipe_new/combined_mp_smoothed_oc_smoothed_homography/test/_fZbAxSSbX4-13.pose
combined resectioning, find in path: /projects/imva/Egosign/egosign/video_level/features/mediapipe_new/combined_mp_smoothed_oc_smoothed_resectioning/test/_fZbAxSSbX4-13.pose
'''


'''
In this file, we want to visualize the different ways that we are processing
first column should have the frame
/projects/imva/Egosign/egosign/video_level/rgb/test/_fZbAxSSbX4-13/_fZbAxSSbX4-13-rgb_front.mp4

That is, visualize inside the folders of:
/projects/imva/Egosign/egosign/video_level/features/mediapipe_new/front/test/_fZbAxSSbX4-13.pose
/projects/imva/Egosign/egosign/video_level/features/mediapipe_new/front_smooth/test/_fZbAxSSbX4-13.pose
/projects/imva/Egosign/egosign/video_level/features/mediapipe_new/front_smooth_normalized/test/_fZbAxSSbX4-13.pose

We want to fix the x and y axis and show them in the plot so that we can also observe the normalization

Only visualize upper body + hands.

'''
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

def main(vid_id, partition):
    
    # Construct the paths using pathlib for robustness
    base_path = Path("/projects/imva/Egosign/egosign")
    path_to_video = base_path / f"video_level/rgb/{partition}/{vid_id}/{vid_id}-rgb_front.mp4"
    path_to_mp_original = base_path / f"video_level/features/mediapipe_new/front_smooth_normalized/{partition}/{vid_id}.pose"
    path_to_oc_homography = base_path / f"video_level/features/mediapipe_new/oc_3d_to_2d_homography/{partition}_smooth_normalized/{vid_id}.pose"
    path_to_oc_resectioning = base_path / f"video_level/features/mediapipe_new/oc_3d_to_2d_resectioning/{partition}_smooth_normalized/{vid_id}.pose"
    path_to_combined_homography = base_path / f"video_level/features/mediapipe_new/combined_mp_smoothed_oc_smoothed_homography/{partition}/{vid_id}.pose"
    path_to_combined_resectioning = base_path / f"video_level/features/mediapipe_new/combined_mp_smoothed_oc_smoothed_resectioning/{partition}/{vid_id}.pose"
    
    #Load the video
    print("Loading video and pose files...")
    video_frames, fps, video_dims = load_video_frames(path_to_video)
    
    #Load the .poses
    with open(path_to_mp_original, "rb") as f:
        mp_original_pose = Pose.read(f.read())
    with open(path_to_oc_homography, "rb") as f:
        oc_homography_pose = Pose.read(f.read())
    with open(path_to_oc_resectioning, "rb") as f:
        oc_resectioning_pose = Pose.read(f.read())
    with open(path_to_combined_homography, "rb") as f:
        combined_homography_pose = Pose.read(f.read())
    with open(path_to_combined_resectioning, "rb") as f:
        combined_resectioning_pose = Pose.read(f.read())
    print("All poses loaded.")
    
    # Filter to have only upper body + hands
    print("Filtering pose components for upper body and hands... Only hands if there is nothing more...")
    mp_original_filtered = filter_pose_components(mp_original_pose)
    oc_homography_filtered = filter_pose_components(oc_homography_pose)
    oc_resectioning_filtered = filter_pose_components(oc_resectioning_pose)
    combined_homography_filtered = filter_pose_components(combined_homography_pose)
    combined_resectioning_filtered = filter_pose_components(combined_resectioning_pose)
    
    #For this, let's force the left hand to be closer to the wrist.
    # index of the wrist on the body
    #body_wrist_idx = combined_resectioning_filtered.header.components[0].points.index('LEFT_WRIST')
    ## index of the wrist on the left hand
    #hand_wrist_local_idx = combined_resectioning_filtered.header.components[1].points.index('WRIST')
    #hand_wrist_abs_idx = hand_wrist_local_idx+25
    #
    #hand_points_slice = slice(25, 25+21)
    #
    #for frame_idx in range(combined_resectioning_filtered.body.data.shape[0]):
    #    frame_data = combined_resectioning_filtered.body.data[frame_idx, 0,:,:]
    #    # check that both exist
    #    body_wrist_is_missing = ma.is_masked(frame_data[body_wrist_idx, 0])
    #    hand_wrist_is_missing = ma.is_masked(frame_data[hand_wrist_abs_idx, 0])
    #    
    #    if not (body_wrist_is_missing or hand_wrist_is_missing):
    #        #we need to have both
    #        body_wrist_coords = frame_data.data[body_wrist_idx]
    #        hand_wrist_coords = frame_data.data[hand_wrist_abs_idx]
    #        translation_vector = body_wrist_coords - hand_wrist_coords
    #        original_hand_points = frame_data.data[hand_points_slice]
    #        frame_data.data[hand_points_slice] = original_hand_points + translation_vector
    #        combined_resectioning_filtered.body.data[frame_idx, 0,:,:] = frame_data
    #
    #print("✅ Poses filtered.")
    
    #We want to use matplotlib and func animation in order to visualize all the data. 
    #Each matplotlib frame should have 4 columns:
    #front rgb, original MP, Postprocessed MP, Postprocessed + normalized MP
    print("Setting up visualization...")
    num_frames = min(len(video_frames), 
                     mp_original_filtered.body.data.shape[0],
                     oc_homography_filtered.body.data.shape[0],
                     oc_resectioning_filtered.body.data.shape[0],
                     combined_homography_filtered.body.data.shape[0],
                     combined_resectioning_filtered.body.data.shape[0])
    
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout(pad=0.5)
    
    def update(frame_idx):
        if (frame_idx + 1) % 20 == 0:
            print(f"Processing frame {frame_idx + 1}/{num_frames}...")

        # Column 1: Original RGB video
        axes[0].clear()
        axes[0].imshow(video_frames[frame_idx])
        axes[0].set_title("Original Video")
        axes[0].axis('off')
        
        draw_pose(axes[1], mp_original_filtered, frame_idx, "Original MediaPipe", is_normalized=True)
        draw_pose(axes[2], oc_homography_filtered, frame_idx, "OC to Front (Homography)", is_normalized=True)
        draw_pose(axes[3], oc_resectioning_filtered, frame_idx, "OC to Front (Resectioning)", is_normalized=True)
        draw_pose(axes[4], combined_homography_filtered, frame_idx, "Combined (Homography)", is_normalized=True)
        draw_pose(axes[5], combined_resectioning_filtered, frame_idx, "Combined (Resectioning)", is_normalized=True)

        return axes

    #We should save it as an mp4, where each frame has a combination of this four columns.
    output_filename = f"/home/ltarres/egosign_final_code/examples/SL_topic_detection/visualizations_teaser/visualizations_ego_kps_projections/{partition}/{vid_id}_comparison_displaced.mp4"
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    
    print("--- DEBUG: Saving first frame as PNG to check plotting ---")
    update(0) # Manually call the update function for the first frame
    debug_frame_path = Path(output_filename).with_suffix('.png')
    fig.savefig(debug_frame_path, dpi=150)
    print(f"--- DEBUG: First frame saved to {debug_frame_path} ---")
    # --- End of debugging block ---
    
    print("Creating animation... This may take a while.")
    num_frames = 75
    ani = FuncAnimation(fig, update, frames=num_frames,)
    
    ani.save(output_filename, writer='ffmpeg', fps=fps, dpi=250)
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
salloc --nodes=1 --time=06:00:00 --mem=16G
module load Anaconda3/2023.09-0
module load FFmpeg/4.3.2-GCCcore-10.2.0
eval "$(conda shell.bash hook)"
conda activate env_proves

cd egosign_final_code/examples/SL_topic_detection/
python visualize_kp_projections_ego.py --id="_G0RrDVpOZ4-13" --partition test
python visualize_kp_projections_ego.py --id="_2FBDaOPYig-8" --partition val
'''