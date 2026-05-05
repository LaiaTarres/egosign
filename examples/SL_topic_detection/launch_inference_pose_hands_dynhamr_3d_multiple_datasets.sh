#!/bin/bash
#SBATCH --time=00:40:00
#SBATCH --mem 30G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

set -euo pipefail

module load Anaconda3/2023.09-0
eval "$(conda shell.bash hook)"
conda activate SLTopicDetection_3

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SEEDS=${SEEDS:-"1 2 3 4 5"}
DATASETS=${DATASETS:-"How2Sign EgoSign-rgb-dynhamr-processed EgoSign-oc-dynhamr-processed"}
DATASET_SPLITS=${DATASET_SPLITS:-"val test"}

for SEED in ${SEEDS}; do
  for DATASET in ${DATASETS}; do
    for DATASET_SPLIT in ${DATASET_SPLITS}; do
      MODEL_TYPE=${MODEL_TYPE:-transformerCLS}
      FEATS_TYPE=${FEATS_TYPE:-dynhamr}
      EXTRA_PARAMS=${EXTRA_PARAMS:-3d_pose}
      EXP_TAG=${EXP_TAG:-release}
      MODEL_PATH=${MODEL_PATH:-${SCRIPT_DIR}/../../outputs_final/${MODEL_TYPE}_${FEATS_TYPE}_${SEED}_${EXTRA_PARAMS}_${EXP_TAG}/checkpoint_best.pt}

      if [[ ! -f "${MODEL_PATH}" ]]; then
        echo "[ERROR] Missing checkpoint for seed ${SEED}: ${MODEL_PATH}"
        continue
      fi

      echo "Running DynHaMR 3D inference"
      echo "SEED: ${SEED}"
      echo "MODEL_PATH: ${MODEL_PATH}"
      echo "DATASET: ${DATASET}"
      echo "DATASET_SPLIT: ${DATASET_SPLIT}"

      SEED=${SEED} DATASET=${DATASET} DATASET_SPLIT=${DATASET_SPLIT} MODEL_PATH=${MODEL_PATH} \
        bash "${SCRIPT_DIR}/launch_inference_pose_hands_dynhamr_3d.sh"
    done
  done
done
