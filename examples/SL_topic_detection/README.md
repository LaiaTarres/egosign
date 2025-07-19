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











Install fairseq:
```bash
pip install --editable ${FAIRSEQ_ROOT}
```

Define the root folder of [How2Sign](https://how2sign.github.io):
```bash
export H2S_ROOT=...
```

The data in `H2S_ROOT` should be organized as follows:
```bash
${H2S_ROOT}
├── test.tsv
├── test.h5
├── train.tsv
├── train.h5
├── val.tsv
└── val.h5
```

Where `h5` files contain the keypoints at video level and `tsv` files contain the text translations and information about the start and end of sentences.


## Prepare the data

Execute the following script to perform the following actions to the data:
- Split the video level keypoints into sentence level keypoints
- Filter out to short or too long examples
- Generate the vocabulary

```bash
python ${FAIRSEQ_ROOT}/examples/sign2vec/prep_how2sign.py \
    --data-root ${H2S_ROOT} \
    --min-n-frames 5 \
    --max-n-frames 4000 \
    --vocab-type unigram \
    --vocab-size 4000 \
```

After the script finishes, the data in `H2S_ROOT` should be organized as follows:

```bash
${H2S_ROOT}
├── test.tsv
├── test_filt.tsv
├── test_sent.h5
├── test.h5
├── train.tsv
├── train_filt.tsv
├── train_sent.h5
├── train.h5
├── val.tsv
├── val_filt.tsv
├── val_sent.h5
└── val.h5
```

Where files ending with `_sent.h5` are the keypoints at sentence level, arrays of shape (seq_len, num_keypoints*4), and `_filt.tsv` are the filtered tsv files.

## [Work in Progress]
