#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --mem 120G
#SBATCH --nodes=1
#SBATCH --ntasks=8

module load Anaconda3/2023.09-0
eval "$(conda shell.bash hook)"
conda activate SLTopicDetection_2


# Declare environment variables
HOW2SIGN_DIR="path_to/How2Sign"
NUM_JOBS=8
FPS=25

for DATASET_SPLIT in val test train
  do
  echo '*************************************************'
  echo Extracting .pose for partition: $DATASET_SPLIT 
  echo '*************************************************'

  #parallel -j ${NUM_JOBS} --bar \
  #    'id=$(basename {} .mp4);
  #    echo;
  #    echo id $id;
  #    pose_file="${HOW2SIGN_DIR}/video_level/${DATASET_SPLIT}/rgb_front/features/mediapipe/${id}.pose";
  #    echo pose_file $pose_file;
  #    if [ ! -e "$pose_file" ]; then python ./scripts/extract_mediapipe.py --video-file {} --poses-file ${pose_file}; fi'

  # Find and process .mp4 files in parallel
  export HOW2SIGN_DIR
  export DATASET_SPLIT
  export FPS
  find ${HOW2SIGN_DIR}/video_level/${DATASET_SPLIT}/rgb_front/raw_videos -name "*rgb_front.mp4" -type f |
      parallel -j ${NUM_JOBS} --bar \
      'id=$(basename {} .mp4);
      echo;
      echo namefile {};
      pose_file="${HOW2SIGN_DIR}/video_level/${DATASET_SPLIT}/rgb_front/features/mediapipe/${id}.pose";
      echo pose_file $pose_file;
      if [ ! -e "$pose_file" ]; then python ./scripts/extract_mediapipe.py --video-file {} --poses-file ${pose_file} --fps ${FPS}; fi'
  echo '*************************************************'
  echo Done extacting .pose for partition: $DATASET_SPLIT
  echo '*************************************************'
  done