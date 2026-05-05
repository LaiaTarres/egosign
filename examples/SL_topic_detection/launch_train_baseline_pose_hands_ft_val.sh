#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem 40G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

set -euo pipefail

module load Anaconda3/2023.09-0
module load FFmpeg/4.3.2-GCCcore-10.2.0
eval "$(conda shell.bash hook)"
conda activate SLTopicDetection_3

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

MODEL_TYPE=${MODEL_TYPE:-transformerCLS}
FEATS_TYPE=${FEATS_TYPE:-dynhamr}
EXTRA_PARAMS=${EXTRA_PARAMS:-3d_pose}
CONFIG_NAME=${CONFIG_NAME:-baseline_${MODEL_TYPE}_${FEATS_TYPE}_${EXTRA_PARAMS}}
H2S_ROOT=${H2S_ROOT:-${REPO_ROOT}/data/egosign/dynhamr}
SP_MODEL=${SP_MODEL:-${REPO_ROOT}/data/text/spm_unigram8000_en.model}
SEEDS=${SEEDS:-5}
EXP_TAG=${EXP_TAG:-finetune}
PRETRAIN_TAG=${PRETRAIN_TAG:-release}
DYNHAMR_INTERP_30FPS=${DYNHAMR_INTERP_30FPS:-true}
DYNHAMR_SRC_FPS=${DYNHAMR_SRC_FPS:-25.0}
DYNHAMR_TGT_FPS=${DYNHAMR_TGT_FPS:-30.0}
DYNHAMR_FILL_MODE=${DYNHAMR_FILL_MODE:-neighbor_average}
DYNHAMR_LONG_GAP_LINEAR_THRESHOLD=${DYNHAMR_LONG_GAP_LINEAR_THRESHOLD:-8}
NORMALIZATION=${NORMALIZATION:-layer_norm}
DATA_AUGMENTATION=${DATA_AUGMENTATION:-true}
TO_CAM_COORDS=${TO_CAM_COORDS:-true}
MAX_UPDATE=${MAX_UPDATE:-10000}
WANDB_MODE=${WANDB_MODE:-online}

for NUM_EXP in ${SEEDS}; do
    SAVE_DIR=${SAVE_DIR:-${REPO_ROOT}/outputs_final/${MODEL_TYPE}_${FEATS_TYPE}_${NUM_EXP}_${EXTRA_PARAMS}_${EXP_TAG}}
    RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-${REPO_ROOT}/outputs_final/${MODEL_TYPE}_${FEATS_TYPE}_${NUM_EXP}_${EXTRA_PARAMS}_${PRETRAIN_TAG}/checkpoint_best.pt}

    if [[ ! -f "${RESUME_FROM_CHECKPOINT}" ]]; then
      echo "[ERROR] RESUME_FROM_CHECKPOINT does not exist: ${RESUME_FROM_CHECKPOINT}"
      exit 1
    fi

    echo "NUM_EXP = ${NUM_EXP}"
    echo "Saving to: ${SAVE_DIR}"
    echo "Resuming from: ${RESUME_FROM_CHECKPOINT}"

    WANDB_MODE=${WANDB_MODE} WANDB_NAME="${MODEL_TYPE}_${FEATS_TYPE}_${NUM_EXP}_${EXTRA_PARAMS}_${EXP_TAG}" SEED=${NUM_EXP} fairseq-hydra-train \
        +task.data=${H2S_ROOT} \
        +task.dict_path=${H2S_ROOT}/categoryName_categoryID.csv \
        task.feats_type=${FEATS_TYPE} \
        task.to_camera_coordinates=${TO_CAM_COORDS} \
        task.dynhamr_temporal_interpolation=${DYNHAMR_INTERP_30FPS} \
        task.dynhamr_source_fps=${DYNHAMR_SRC_FPS} \
        task.dynhamr_target_fps=${DYNHAMR_TGT_FPS} \
        task.dynhamr_fill_mode=${DYNHAMR_FILL_MODE} \
        task.dynhamr_long_gap_linear_threshold=${DYNHAMR_LONG_GAP_LINEAR_THRESHOLD} \
        task.normalization=${NORMALIZATION} \
        task.data_augmentation=${DATA_AUGMENTATION} \
        checkpoint.restore_file=${RESUME_FROM_CHECKPOINT} \
        checkpoint.save_dir=${SAVE_DIR} \
        optimization.max_update=${MAX_UPDATE} \
        bpe.sentencepiece_model=${SP_MODEL} \
        task.dataset='How2Sign' \
        --config-dir ${SCRIPT_DIR}/config_pose \
        --config-name ${CONFIG_NAME}

done
