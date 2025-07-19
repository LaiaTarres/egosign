#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --mem 20G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

module load Anaconda3/2023.09-0
eval "$(conda shell.bash hook)"
conda activate SLTopicDetection_3

FEATS_TYPE=mediapipe_keypoints

# For bodyparts
BODYPARTS=only_hands #this is already embedded in CONFIG_NAME
#only_hands
#hands_and_body

#For the datasets, it needs to be conditional
if [ "$BODYPARTS" == "hands_and_body" ]; then
    echo "BODYPARTS is set to 'hands_and_body'. Using corresponding datasets."
    DATASETS_TO_RUN=(
        "How2Sign"
        "EgoSign-rgb"
        "EgoSign-combined-homo"
        "EgoSign-combined-resec"
    )
elif [ "$BODYPARTS" == "only_hands" ]; then
    echo "BODYPARTS is set to 'only_hands'. Using corresponding datasets."
    DATASETS_TO_RUN=(
        "How2Sign"
        "EgoSign-rgb"
        "EgoSign-oc-homo"
        "EgoSign-oc-resec"
        "EgoSign-combined-homo"
        "EgoSign-combined-resec"
    )
else
    echo "Error: BODYPARTS is set to an unknown value: '$BODYPARTS'"
    exit 1
fi


#For the data cleaning:
DATA_CLEANING=smooth_normalized
#raw
#smooth
#smooth_normalized

# TODO: For this we need to change the training data in the task.py, to include the data_cleaning...

echo $DATASET

H2S_ROOT=path_to_dataset/${FEATS_TYPE}

FAIRSEQ_ROOT=../../..

#MODEL_TYPE=lstm
#MODEL_TYPE=transformerCLS
MODEL_TYPE=perceiverIO


SP_MODEL=path_to_dataset/text/spm_unigram8000_en.model

#EXTRA_ARGS=2d_pose_24_april #In theory this is not 2d_pose
#EXTRA_ARGS=2d_pose_handsandbody
#EXTRA_ARGS=2d_pose_handsandbody_2
EXTRA_ARGS=2d_pose_2
#EXTRA_ARGS=2d_pose

CONFIG_NAME=inference_${MODEL_TYPE}_${EXTRA_ARGS}

OUTPUTS_DIR=path_to_code/egosign_final_code/examples/SL_topic_detection/outputs_final
mkdir -p $OUTPUTS_DIR

for DATASET in "${DATASETS_TO_RUN[@]}"
do
    echo "================================================="
    echo "Processing DATASET: ${DATASET}"
    echo "================================================="
    for NUM_EXP in 1 2 3 4 5
    do
        echo NUM_EXP = ${NUM_EXP}

        MODEL_PATH=path_to_models/final_models/${MODEL_TYPE}_${FEATS_TYPE}_${NUM_EXP}_${EXTRA_ARGS}/checkpoint_best.pt
        for DATASET_SPLIT in val test
        do
            echo '*************************************************'
            echo Starting experiment $NUM_EXP, $DATASET_SPLIT split, $FEATS_TYPE features
            echo '*************************************************'
            HYDRA_FULL_ERROR=1 \
            DATA=$H2S_ROOT \
            DICT_PATH=${H2S_ROOT}/categoryName_categoryID.csv \
            MODEL_PATH=$MODEL_PATH \
            CONFIG_NAME=${CONFIG_NAME} \
            SP_MODEL=${SP_MODEL} \
            DATASET_SPLIT=$DATASET_SPLIT \
            DATASET=$DATASET \
            OUTPUTS_FILE=${OUTPUTS_DIR}/${CONFIG_NAME}_${NUM_EXP}_${DATASET_SPLIT}_${DATASET}_${DATA_CLEANING}.pt \
            FEATS_TYPE=$FEATS_TYPE \
            python infer.py
            echo '*************************************************'
            echo Finishing experiment $NUM_EXP, $DATASET_SPLIT split
            echo '*************************************************'
            echo
        done
        echo "-------------------------------------------------"
        echo "Finished all splits for NUM_EXP = ${NUM_EXP}"
        echo "-------------------------------------------------"
        echo
    done
    echo "================================================="
    echo "Finished all experiments for DATASET: ${DATASET}"
    echo "================================================="
    echo
done