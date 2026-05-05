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
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

FEATS_TYPE=${FEATS_TYPE:-dynhamr}
MODEL_TYPE=${MODEL_TYPE:-transformerCLS}
CONFIG_NAME=${CONFIG_NAME:-inference_${MODEL_TYPE}_${FEATS_TYPE}_3d_pose}
H2S_ROOT=${H2S_ROOT:-${REPO_ROOT}/data/egosign/dynhamr}
SP_MODEL=${SP_MODEL:-${REPO_ROOT}/data/text/spm_unigram8000_en.model}
OUTPUTS_DIR=${OUTPUTS_DIR:-${REPO_ROOT}/outputs_final}
mkdir -p "${OUTPUTS_DIR}"
RUN_TAG=${RUN_TAG:-release}
DATASET=${DATASET:-EgoSign-rgb-dynhamr-processed}
DATASET_SPLIT=${DATASET_SPLIT:-test}
DATA_CLEANING=${DATA_CLEANING:-dynhamr_preprocessed}
MODEL_PATH=${MODEL_PATH:-${REPO_ROOT}/models/${MODEL_TYPE}_${FEATS_TYPE}/${RUN_TAG}/checkpoint_best.pt}

case "${DATASET}" in
  How2Sign|EgoSign-rgb|EgoSign-rgb-dynhamr|dynhamr-egosign-rgb|EgoSign-rgb-dynhamr-processed|dynhamr-egosign-rgb-processed|dynhamr-rgb-front|EgoSign-oc|dynhamr-oc|EgoSign-oc-dynhamr|dynhamr-egosign-oc|EgoSign-oc-dynhamr-processed)
    ;;
  *)
    echo "[ERROR] Unsupported DATASET: ${DATASET}"
    exit 1
    ;;
esac

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH does not exist: ${MODEL_PATH}"
  exit 1
fi

echo "Running DynHaMR 3D inference"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "DATASET: ${DATASET}"
echo "DATASET_SPLIT: ${DATASET_SPLIT}"
echo "CONFIG_NAME: ${CONFIG_NAME}"

HYDRA_FULL_ERROR=1 \
DATA="${H2S_ROOT}" \
DICT_PATH="${H2S_ROOT}/categoryName_categoryID.csv" \
MODEL_PATH="${MODEL_PATH}" \
CONFIG_NAME="${CONFIG_NAME}" \
SP_MODEL="${SP_MODEL}" \
DATASET_SPLIT="${DATASET_SPLIT}" \
DATASET="${DATASET}" \
OUTPUTS_FILE="${OUTPUTS_DIR}/${CONFIG_NAME}_${RUN_TAG}_${DATASET_SPLIT}_${DATASET}_${DATA_CLEANING}.pt" \
FEATS_TYPE="${FEATS_TYPE}" \
python "${SCRIPT_DIR}/infer.py"
