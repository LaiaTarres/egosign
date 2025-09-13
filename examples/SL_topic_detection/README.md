# EgoSign: A Multiview Video Dataset for Sign Language Understanding
This repository contains the implementation for the EgoSign: A Multiview Video Dataset for Sign Language Understanding paper. 

<p align="center">
    <a href="https://github.com/LaiaTarres/egosign/tree/main/examples/SL_topic_detection/visualizations_teaser/Teaser.jpg">
        <img src="https://github.com/LaiaTarres/egosign/tree/main/examples/SL_topic_detection/visualizations_teaser/Teaser.jpg" alt="Teaser Preview" width="600"/>
    </a>
</p>

If it doesn't charge properly, you can find the videos inside the examples/SL_topic_detection/visualizations_teaser folder.

<table align="center" style="border: none;">
<tr>
<td align="center" style="border: none;"><b>Video ID: 00dWJ4YRRSI-12_2</b></td>
<td align="center" style="border: none;"><b>Video ID: 46Cwjrd4ua4-14_2</b></td>
</tr>
<tr>
<td width="50%" style="border: none;">
<video controls autoplay loop muted width="100%" src="https://github.com/LaiaTarres/egosign/tree/main/examples/SL_topic_detection/visualizations_teaser/00dWJ4YRRSI-12_2.mp4"></video>
</td>
<td width="50%" style="border: none;">
<video controls autoplay loop muted width="100%" src="https://github.com/LaiaTarres/egosign/tree/main/examples/SL_topic_detection/visualizations_teaser/46Cwjrd4ua4-14_2.mp4"></video>
</td>
</tr>
<tr>
<td align="center" colspan="2" style="border: none;"><b>Video ID: FZCF7kPIyOk-8_2</b></td>
</tr>
<tr>
<td align="center" colspan="2" style="border: none;">
<video controls autoplay loop muted width="50%" src="https://github.com/LaiaTarres/egosign/tree/main/examples/SL_topic_detection/visualizations_teaser/FZCF7kPIyOk-8_2.mp4"></video>
</td>
</tr>
</table>


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


