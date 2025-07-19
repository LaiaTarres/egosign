#!/bin/bash
#SBATCH --time=14:00:00
#SBATCH --mem 40G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

module load Anaconda3/2023.09-0
module load FFmpeg/4.3.2-GCCcore-10.2.0
eval "$(conda shell.bash hook)"
conda activate SLTopicDetection_3

H2S_ROOT=path_to_dataset/TopicDetection

FAIRSEQ_ROOT=path_to_egosign_final_code

FEATS_TYPE=mediapipe_keypoints

SP_MODEL=${H2S_ROOT}/text/spm_unigram8000_en.model

CURRENT_PATH=$(pwd)

# MODEL_TYPE=lstm
# MODEL_TYPE=transformer
# MODEL_TYPE=transformerCLS
 MODEL_TYPE=perceiverIO

# CONFIG_NAME=baseline
# CONFIG_NAME=baseline_transformer
# CONFIG_NAME=baseline_perceiverIO
# CONFIG_NAME=baseline_${MODEL_TYPE}_${FEATS_TYPE}

EXTRA_PARAMS=2d_pose_2
#EXTRA_PARAMS=2d_pose_handsandbody_2
#EXTRA_PARAMS=2d_pose_handsandbody
#EXTRA_PARAMS=2d_pose
CONFIG_NAME=baseline_${MODEL_TYPE}_${FEATS_TYPE}_${EXTRA_PARAMS}

#for NUM_EXP in 1 2 3 4 5
#for NUM_EXP in 1 2 3
for NUM_EXP in 4 5
do
    echo NUM_EXP = ${NUM_EXP}

    SAVE_DIR=path_to_models/final_models/${MODEL_TYPE}_${FEATS_TYPE}_${NUM_EXP}_${EXTRA_PARAMS}

    echo "Saving to: ${SAVE_DIR}"

    WANDB_MODE=online SEED=$NUM_EXP fairseq-hydra-train \
        +task.data=${H2S_ROOT}/${FEATS_TYPE} \
        +task.dict_path=${H2S_ROOT}/${FEATS_TYPE}/categoryName_categoryID.csv \
        +task.feats_type=$FEATS_TYPE \
        checkpoint.save_dir=$SAVE_DIR \
        bpe.sentencepiece_model=${SP_MODEL} \
        task.dataset='How2Sign' \
        --config-dir ${CURRENT_PATH}/config_pose \
        --config-name ${CONFIG_NAME}
done