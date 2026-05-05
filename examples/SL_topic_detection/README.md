# EgoSign: A Multiview Video Dataset for Sign Language Understanding
This repository contains the implementation for the EgoSign: A Multiview Video Dataset for Sign Language Understanding paper. 

<p align="center">
    <a href="https://github.com/LaiaTarres/egosign/blob/main/examples/SL_topic_detection/visualizations_teaser/Teaser_egosign.jpg">
        <img src="https://raw.githubusercontent.com/LaiaTarres/egosign/main/examples/SL_topic_detection/visualizations_teaser/Teaser_egosign.jpg" alt="Teaser Preview" width="600"/>
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


All the main launchers are in examples/SL_topic_detection/. Helper utilities live in examples/SL_topic_detection/scripts/.

## First steps
Clone this repository, create the conda environment and install Fairseq:
```bash
git clone ...
cd egosign_final_code

conda env create -f examples/sign_language/environment.yml
conda activate SLTopicDetection

pip install --editable .
```

## Dataverse release
The released EgoSign data, code snapshots, examples and trained models are archived in Dataverse:
https://dataverse.csuc.cat/dataset.xhtml?persistentId=doi%3A10.34810%2Fdata3259&version=DRAFT

The uploaded materials follow the structure documented in /gpfs/projects/imva/Egosign/dataverse/readme.txt:
```
├── data/
│   └── egosign/
│       ├── video_level/
│       │   ├── rgb.zip
│       │   └── features/
│       │       ├── dynhamr.zip
│       │       ├── mediapipe.zip
│       │       └── oc.zip
│       ├── how2sign/
│       │   └── features/
│       │       ├── dynhamr.zip
│       │       └── mediapipe.zip
│       ├── models/
│       │   └── models.zip
│       └── visualizations/
└── readme.txt
```

## Training and evaluation
Set `H2S_ROOT`, `SP_MODEL`, `MODEL_PATH`, and `SAVE_DIR` to match your local extraction of the Dataverse release. The launchers below are the public entrypoints for the supported training and evaluation flows.

### 1. Training with MediaPipe only hands
Use `examples/SL_topic_detection/launch_train_baseline.sh` with:
- `FEATS_TYPE=mediapipe_keypoints`
- `BODYPARTS=only_hands`
- `EXTRA_PARAMS=2d_pose_2`

For evaluation, run `examples/SL_topic_detection/launch_inference.sh` and then `examples/SL_topic_detection/launch_analysis_outputs.sh`.

### 2. Training with MediaPipe hands + upper body
Use `examples/SL_topic_detection/launch_train_baseline.sh` with:
- `FEATS_TYPE=mediapipe_keypoints`
- `BODYPARTS=hands_and_body`
- `EXTRA_PARAMS=2d_pose_handsandbody_2`

For evaluation, run `examples/SL_topic_detection/launch_inference.sh` and then `examples/SL_topic_detection/launch_analysis_outputs.sh`.

### MediaPipe visualization (hands-only and hands+upper-body)
Use `examples/SL_topic_detection/visualize_for_teaser.py` to generate a moving grid video with RGB views plus MediaPipe pose views.

- Hands-only MediaPipe is supported (42 keypoints: left+right hands).
- Hands+upper-body MediaPipe is supported (67 keypoints: upper body + hands).

Single video example:
```bash
cd examples/SL_topic_detection
python visualize_for_teaser.py --id "_G0RrDVpOZ4-13" --partition test
```

Batch (SLURM array) example:
```bash
sbatch examples/SL_topic_detection/visualize_for_teaser.sh
```

Output videos are written under `examples/SL_topic_detection/visualizations_teaser/{partition}/` and contain a 2x4 animated grid:
- 4 RGB camera views (front, side, head, inside)
- 3 MediaPipe-based pose panels (front, projected, combined)

### 3. Training with DynHaMR
Use `examples/SL_topic_detection/launch_train_baseline_pose_hands.sh` together with `examples/SL_topic_detection/config_pose/baseline_transformerCLS_dynhamr_3d_pose.yaml`.

The public release also includes the DynHaMR 3D inference entrypoints:
- `examples/SL_topic_detection/launch_inference_pose_hands_dynhamr_3d.sh`
- `examples/SL_topic_detection/launch_inference_pose_hands_dynhamr_3d_multiple_datasets.sh`

### 4. Finetuning best DynHaMR models
Use `examples/SL_topic_detection/launch_train_baseline_pose_hands_ft_val.sh` to resume from the best DynHaMR checkpoint and finetune on the same configuration.

## Optional: DynHaMR Preprocessing and Visualization

The `examples/SL_topic_detection/scripts/` directory contains optional utilities for preprocessing and visualizing DynHaMR hand pose sequences.

### Preprocessing: Converting DynHaMR 3D to .pose format

If you have raw DynHaMR 3D hand pose outputs (as numpy arrays: `joints_3d_left.npy`, `joints_3d_right.npy`, etc.), you can convert them to .pose format for use with the training pipeline:

1. **Configure** `examples/SL_topic_detection/scripts/new_dynhamr_3d_to_pose_precompute.sh`:
   - Set `INPUT_TSV` to your dataset manifest with columns: `id_vid`, `signs_file` (DynHaMR directory path), `video_length`
   - Set `INPUT_DIR` to the parent directory containing per-video DynHaMR folders
   - Set `OUTPUT_POSE_DIR` and `OUTPUT_MANIFEST_DIR` to your output locations
   - Adjust `SOURCE_FPS` (typically 25 for DynHaMR) and `TARGET_FPS` (typically 30)

2. **Submit as SLURM array job**:
   ```bash
   sbatch --array=0-N examples/SL_topic_detection/scripts/new_dynhamr_3d_to_pose_precompute.sh
   ```
   where N = (number of unique videos - 1). The script will automatically extract unique video IDs from your TSV.

3. **Output**:
   - Binary `.pose` files in `OUTPUT_POSE_DIR`
   - Updated manifest TSV mapping video IDs to .pose file paths
   - Per-hand confidence scores indicating real detections vs. filled/padded frames

**Key options** (set as environment variables before running):
- `ENABLE_CONFIDENCE_SCORING`: Mark confidence per-hand (default: 0)
- `CONFIDENCE_REAL`, `CONFIDENCE_FILLED`, `CONFIDENCE_PADDED`: Confidence values for different frame types

**Dependencies**: `pip install pose_format`

### Visualization: Rendering .pose files on RGB video

To visualize precomputed hand poses overlaid on RGB video with synchronized multi-view rendering:

```bash
python examples/SL_topic_detection/scripts/visualize_pose_on_video.py \
    --video_id VIDEO_ID \
    --partition val \
    --dyn_pose path/to/VIDEO_ID.pose \
    --rgb_video path/to/VIDEO_ID.mp4 \
    --dyn_result_dir path/to/calibration/ \
    --output path/to/output.mp4
```

The script will render:
- RGB with hand skeleton overlay
- Front 3D view
- Top 3D view  
- Side 3D view

in a 2×2 grid layout.

**Optional arguments**:
- `--sentence_alignment_tsv`: TSV with sentence/segment frame boundaries
- `--front_alignment_tsv`: TSV with frame offset alignments between RGB and pose sequences
- `--oc_pose`: Path to alternative .pose file for comparison (e.g., Oculus world poses)

**Note**: The visualization template provides the expected interface. The full implementation requires Dyn-HaMR visualization utilities (hand drawing, camera rendering). Refer to the script documentation for implementation details.

**Dependencies**: `pip install opencv-python pose_format scipy`

## Notes
The Dataverse archive is the canonical source for the released data and model artifacts. Any optional preprocessing or visualization helpers should be kept under `examples/SL_topic_detection/scripts/` and invoked from the public release tree, not from internal staging paths.

