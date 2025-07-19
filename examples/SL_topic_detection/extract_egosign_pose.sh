#!/bin/bash
#SBATCH --job-name=mediapipe_extraction
#SBATCH --gres=gpu:1
#SBATCH --output=logs_mp_ego_2/slurm-%A_%a.out
#SBATCH --error=logs_mp_ego_2/slurm-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem 60G
#SBATCH --nodes=1
#SBATCH --array=1-273

module load Anaconda3/2023.09-0

eval "$(conda shell.bash hook)"

conda activate SLTopicDetection_2

# Set environment variables
EGOSIGN_DIR="/projects/imva/Egosign/egosign/"
FPS=30

# TSV file containing alignment data (this is so we have all the id's)
TSV_FILE="${EGOSIGN_DIR}alignment_files/alignment_front_rest.tsv"

# Calculate the number of lines in the TSV file excluding the header
NUM_LINES=$(($(wc -l < "$TSV_FILE") - 1))

# Check if the current task ID exceeds the number of available lines
if [ "$SLURM_ARRAY_TASK_ID" -gt "$NUM_LINES" ]; then
    echo "Error: Task ID exceeds available lines in TSV file."
    exit 1
fi

# Extract the line corresponding to the current task ID
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TSV_FILE")
VIDEO_ID=$(echo "$LINE" | cut -f1)
PARTITION=$(echo "$LINE" | cut -f2)

# Define paths for video and pose files
video_file="${EGOSIGN_DIR}video_level/rgb/${PARTITION}/${VIDEO_ID}/${VIDEO_ID}-rgb_front.mp4"
pose_file="${EGOSIGN_DIR}video_level/features/mediapipe_new/front/${PARTITION}/${VIDEO_ID}.pose"


# Run the Python script for extracting mediapipe features
echo "Running: python ./scripts/extract_mediapipe.py --video-file=\"$video_file\" --poses-file=\"$pose_file\" --fps=\"$FPS\""
python ./scripts/extract_mediapipe.py --video-file="${video_file}" --poses-file="${pose_file}" --fps="${FPS}"
