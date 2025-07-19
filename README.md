# EgoSign: A Multiview Video Dataset for Sign Language Understanding
This repository contains the implementation for the EgoSign: A Multiview Video Dataset for Sign Language Understanding paper. 

All the scripts are located inside examples/SL_topic_detection/scripts.

## First steps
Clone this repository, create the conda environment and install Fairseq:
```bash
git clone ...
cd egosign_final_code

conda env create -f examples/sign_language/environment.yml
conda activate SLTopicDetection

pip install --editable .
```

## Downloading the data
The videos and keypoints are provided seperately, some examples of which features to expect are provided in sample_egosign_dataset folder.

Once the dataset has been downloaded, it should follow this structure:
```
├── data/
│   └── egosign/
│       └── mediapipe_keypoints/
│           ├── egosign_test_proves_filtered_smooth_normalized.tsv
│           ├── egosign_val_proves_filtered_smooth_normalized.tsv
│           ├── egosign_oc_test_resectioning_smooth_normalized.tsv
│           ├── egosign_oc_val_resectioning_smooth_normalized.tsv
│           ├── egosign_test_proves_filtered_smooth_normalized_combined_resectioning.tsv
│           ├── egosign_val_proves_filtered_smooth_normalized_combined_resectioning.tsv
│           ├── how2sign_train_calcula_smooth.tsv
│           ├── how2sign_test_calcula_smooth.tsv
│           └── how2sign_val_calcula_smooth.tsv
│           ├── how2sign/
│           │   ├── train/
│           │   │   ├── --7E2sU6zP4-5.pose
│           │   │   └── ...
│           │   ├── val/
│           │   │   ├── -d5dN54tH2E-1.pose
│           │   │   └── ...
│           │   └── test/
│           │       ├── -fZc293MpJk-1-rgb_front.pose
│           │       └── ...
│           │
│           ├── egosign_front/
│           │   ├── val/
│           │   └── test/
│           │
│           ├── egosign_oc_resectioning/
│           │   ├── val/
│           │   └── test/
│           │
│           └── egosign_combined/
│               ├── val/
│               └── test/
```

## Training 
Launch launch_train_baseline_pose_hands.sh for the baseline trainings. 

## Evaluation
Launch launch_inference_pose_hands.sh -> to get the inferences
Launch analysis_outputs.py -> to combine all the outputs
