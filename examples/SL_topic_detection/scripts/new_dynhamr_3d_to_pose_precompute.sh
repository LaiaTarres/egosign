#!/bin/bash
# DynHaMR 3D to .pose Precomputation Launcher (SLURM Array Job)
#
# This script converts DynHaMR 3D hand pose outputs (world coordinates) into .pose format.
# It's designed to run as a SLURM array job for batch processing.
#
# Usage:
#   sbatch --array=0-N new_dynhamr_3d_to_pose_precompute.sh
#   where N = (number of unique videos - 1)

#SBATCH --job-name=dynhamr_3d_precompute
#SBATCH --output=logs_dynhamr_precompute/slurm-%A_%a.out
#SBATCH --error=logs_dynhamr_precompute/slurm-%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

set -euo pipefail

TEMP_UNIQUE_VIDS=""
TASK_TSV=""
TASK_MANIFEST=""

cleanup() {
    [[ -n "${TEMP_UNIQUE_VIDS:-}" && -f "$TEMP_UNIQUE_VIDS" ]] && rm -f "$TEMP_UNIQUE_VIDS"
    [[ -n "${TASK_TSV:-}" && -f "$TASK_TSV" ]] && rm -f "$TASK_TSV"
    if [[ "${KEEP_TASK_MANIFESTS:-0}" != "1" ]]; then
        [[ -n "${TASK_MANIFEST:-}" && -f "$TASK_MANIFEST" ]] && rm -f "$TASK_MANIFEST"
    fi
}
trap cleanup EXIT

# Load your Python environment
# module load Anaconda3/2023.09-0
# eval "$(conda shell.bash hook)"
# conda activate <your_env>

# Path to the conversion script (relative to this directory)
CONVERSION_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/new_dynhamr_3d_to_pose_precompute.py"

# ========== CONFIGURATION ==========
# Edit these paths to point to your data directories

# Input TSV with DynHaMR metadata (columns: id_vid, signs_file, video_length, ...)
INPUT_TSV="<path/to/your/manifest.tsv>"

# Input directory containing per-video DynHaMR folders (with joints_3d_left.npy, joints_3d_right.npy, etc.)
INPUT_DIR="<path/to/your/dynhamr/data>"

# Output directory where .pose files will be written
OUTPUT_POSE_DIR="<path/to/your/output/poses>"

# Directory where per-video manifest shards are optionally saved (set KEEP_TASK_MANIFESTS=1 to enable)
OUTPUT_MANIFEST_DIR="<path/to/your/output/manifests>"

# Temporal resampling: DynHaMR is typically 25 fps, convert to target fps (usually 30 for alignment with other modalities)
SOURCE_FPS=25.0
TARGET_FPS=30.0

# NaN gap filling strategy: "neighbor_average" for short gaps, "hybrid" for mixed (linear for large gaps)
FILL_MODE="neighbor_average"
LONG_GAP_LINEAR_THRESHOLD=8

# Confidence scoring: mark which joints are real detections vs. filled/padded
ENABLE_CONFIDENCE_SCORING=${ENABLE_CONFIDENCE_SCORING:-0}
CONFIDENCE_REAL=${CONFIDENCE_REAL:-1.0}
CONFIDENCE_FILLED=${CONFIDENCE_FILLED:-0.6}
CONFIDENCE_PADDED=${CONFIDENCE_PADDED:-0.1}

# Optional offset for resuming a partial job (rarely needed)
START_OFFSET=${START_OFFSET:-0}

# Set to 1 to keep per-video manifest shards in OUTPUT_MANIFEST_DIR (default: 0 = delete temp files)
KEEP_TASK_MANIFESTS=${KEEP_TASK_MANIFESTS:-0}

# ====================================

mkdir -p "$OUTPUT_POSE_DIR" "$OUTPUT_MANIFEST_DIR"

if [[ ! -f "$INPUT_TSV" ]]; then
    echo "ERROR: input TSV not found: $INPUT_TSV"
    exit 1
fi

if [[ ! -f "$CONVERSION_SCRIPT" ]]; then
    echo "ERROR: conversion script not found: $CONVERSION_SCRIPT"
    exit 1
fi

TEMP_UNIQUE_VIDS=$(mktemp)
tail -n +2 "$INPUT_TSV" | cut -f2 | sort -u > "$TEMP_UNIQUE_VIDS"
TOTAL_UNIQUE_VIDS=$(wc -l < "$TEMP_UNIQUE_VIDS")

if [[ "$SLURM_ARRAY_TASK_ID" -lt 0 ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID must be >= 0, got ${SLURM_ARRAY_TASK_ID}."
    exit 1
fi

if [[ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_UNIQUE_VIDS" ]]; then
    echo "Error: Task ID ${SLURM_ARRAY_TASK_ID} exceeds available unique video ids (${TOTAL_UNIQUE_VIDS})."
    exit 1
fi

if [[ "$START_OFFSET" -lt 0 ]]; then
    echo "Error: START_OFFSET must be >= 0, got ${START_OFFSET}."
    exit 1
fi

LINE_NO=$((SLURM_ARRAY_TASK_ID + 1 + START_OFFSET))
if [[ "$LINE_NO" -gt "$TOTAL_UNIQUE_VIDS" ]]; then
    echo "Error: LINE_NO ${LINE_NO} exceeds total unique video ids (${TOTAL_UNIQUE_VIDS})."
    exit 1
fi

# sed is 1-based line addressing; SLURM arrays are often 0-based.
TARGET_VID=$(sed -n "${LINE_NO}p" "$TEMP_UNIQUE_VIDS")
rm -f "$TEMP_UNIQUE_VIDS"
TEMP_UNIQUE_VIDS=""

if [[ -z "$TARGET_VID" ]]; then
    echo "Error: Could not resolve video id for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

TASK_TSV=$(mktemp --suffix=.tsv)
if [[ "$KEEP_TASK_MANIFESTS" == "1" ]]; then
    TASK_MANIFEST="${OUTPUT_MANIFEST_DIR}/${TARGET_VID}.tsv"
else
    TASK_MANIFEST=$(mktemp --suffix=.tsv)
fi

python - "$INPUT_TSV" "$TARGET_VID" "$TASK_TSV" <<'PY'
import csv
import sys

input_tsv, target_vid, output_tsv = sys.argv[1:4]

with open(input_tsv, "r", encoding="utf-8") as fin, open(output_tsv, "w", encoding="utf-8", newline="") as fout:
    reader = csv.DictReader(fin, delimiter="\t")
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter="\t")
    writer.writeheader()
    for row in reader:
        if row.get("id_vid", "") == target_vid:
            writer.writerow(row)
PY

echo "[INFO] Task ${SLURM_ARRAY_TASK_ID}: ${TARGET_VID}"
EXTRA_CONF_ARGS=()
if [[ "$ENABLE_CONFIDENCE_SCORING" == "1" ]]; then
    EXTRA_CONF_ARGS+=(
        --enable_confidence_scoring
        --confidence_real "$CONFIDENCE_REAL"
        --confidence_filled "$CONFIDENCE_FILLED"
        --confidence_padded "$CONFIDENCE_PADDED"
    )
fi

python "$CONVERSION_SCRIPT" \
    --input_tsv "$TASK_TSV" \
    --input_dir "$INPUT_DIR" \
    --output_pose_dir "$OUTPUT_POSE_DIR" \
    --output_tsv "$TASK_MANIFEST" \
    --source_fps "$SOURCE_FPS" \
    --target_fps "$TARGET_FPS" \
    --fill_mode "$FILL_MODE" \
    --long_gap_linear_threshold "$LONG_GAP_LINEAR_THRESHOLD" \
    "${EXTRA_CONF_ARGS[@]}"

rm -f "$TASK_TSV"
TASK_TSV=""
if [[ "$KEEP_TASK_MANIFESTS" != "1" ]]; then
    rm -f "$TASK_MANIFEST"
    TASK_MANIFEST=""
fi
