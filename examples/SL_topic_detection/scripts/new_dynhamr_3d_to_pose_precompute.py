#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Precompute DynHaMR 3D into .pose files for SL topic detection.

This mirrors the current online DynHaMR 3D loading path up to the point where
the dataset hands the tensor to postprocess.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody

try:
    from pose_format.utils.header import Header, Components, Component, Dimension
    POSE_HEADER_API = "legacy"
except ImportError:
    from pose_format.pose_header import PoseHeader, PoseHeaderDimensions, PoseHeaderComponent
    POSE_HEADER_API = "modern"


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_dynhamr_3d_pose_header():
    points = [f"P{i}" for i in range(21)]
    if POSE_HEADER_API == "legacy":
        components = Components([
            Component("LEFT_HAND_LANDMARKS", [Dimension("X"), Dimension("Y"), Dimension("Z"), Dimension("Confidence")]),
            Component("RIGHT_HAND_LANDMARKS", [Dimension("X"), Dimension("Y"), Dimension("Z"), Dimension("Confidence")]),
        ])
        return Header(version="0.1.0", dimensions=None, components=components)

    components = [
        PoseHeaderComponent("LEFT_HAND_LANDMARKS", points, [], [], "XYZC"),
        PoseHeaderComponent("RIGHT_HAND_LANDMARKS", points, [], [], "XYZC"),
    ]
    return PoseHeader(
        version=0.1,
        dimensions=PoseHeaderDimensions(width=1, height=1, depth=1),
        components=components,
    )


def world_to_cam(points_3d, rotation, translation):
    return points_3d @ rotation.T + translation


def load_track_seq_interval(dyn_dir: Path):
    info_path = dyn_dir / "track_info.json"
    if not info_path.exists():
        return 0, None
    try:
        payload = json.loads(info_path.read_text())
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        seq_interval = meta.get("seq_interval", None)
        if isinstance(seq_interval, list) and len(seq_interval) == 2:
            return int(seq_interval[0]), int(seq_interval[1])
    except Exception:
        pass
    return 0, None


def sanitize_3d_joints_np(arr: np.ndarray, min_depth_m: float = 0.15):
    out = np.array(arr, copy=True, dtype=np.float32)
    bad = out[..., 2] < float(min_depth_m)
    out[bad] = np.nan
    return out


def temporal_fill_nans_neighbor_average_np(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True, dtype=np.float32)
    t = out.shape[0]
    if t <= 1:
        return out

    flat = out.reshape(t, -1)
    for i in range(t):
        for j in range(flat.shape[1]):
            if not np.isnan(flat[i, j]):
                continue

            i1 = i - 1
            while i1 >= 0 and np.isnan(flat[i1, j]):
                i1 -= 1
            before = flat[i1, j] if i1 >= 0 else np.nan

            i2 = i + 1
            while i2 < t and np.isnan(flat[i2, j]):
                i2 += 1
            after = flat[i2, j] if i2 < t else np.nan

            values = []
            if np.isfinite(before):
                values.append(before)
            if np.isfinite(after):
                values.append(after)

            if values:
                flat[i, j] = float(sum(values) / len(values))

    return flat.reshape(out.shape)


def temporal_fill_nans_hybrid_np(arr: np.ndarray, long_gap_linear_threshold: int = 8) -> np.ndarray:
    from scipy.interpolate import interp1d

    out = np.array(arr, copy=True, dtype=np.float32)
    t = out.shape[0]
    if t <= 1:
        return out

    def _get_nan_runs(valid_mask: np.ndarray):
        is_nan = ~valid_mask
        padded = np.concatenate(([False], is_nan, [False]))
        edges = np.diff(padded.astype(int))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0] - 1
        return zip(starts, ends)

    flat = out.reshape(t, -1)
    src_t = np.arange(t, dtype=np.float32)
    for c in range(flat.shape[1]):
        values = flat[:, c]
        valid = np.isfinite(values)
        if valid.all():
            continue

        valid_count = int(valid.sum())
        if valid_count == 0:
            continue
        if valid_count == 1:
            flat[:, c] = np.full((t,), float(values[valid][0]), dtype=np.float32)
            continue

        for s, e in _get_nan_runs(valid):
            gap_len = e - s + 1
            if gap_len > int(long_gap_linear_threshold):
                prev_idx = s - 1
                next_idx = e + 1
                if prev_idx < 0 and next_idx >= t:
                    continue
                elif prev_idx < 0:
                    values[s:e + 1] = values[next_idx]
                elif next_idx >= t:
                    values[s:e + 1] = values[prev_idx]
                else:
                    v_prev = values[prev_idx]
                    v_next = values[next_idx]
                    steps = np.arange(s, e + 1, dtype=np.float32)
                    values[s:e + 1] = v_prev + (v_next - v_prev) * (steps - prev_idx) / (next_idx - prev_idx)
                valid[s:e + 1] = True

        valid_count = int(valid.sum())
        partial_t = src_t[valid]
        partial_v = values[valid]
        this_kind = "cubic" if valid_count > 3 else "quadratic" if valid_count > 2 else "linear"
        f = interp1d(partial_t, partial_v, kind=this_kind, bounds_error=False, fill_value=np.nan)
        interp_values = np.asarray(f(src_t), dtype=np.float32)
        interp_values[src_t < partial_t[0]] = partial_v[0]
        interp_values[src_t > partial_t[-1]] = partial_v[-1]
        flat[:, c] = interp_values

    return flat.reshape(out.shape)


def pad_dyn_sequence_to_video_length_np(xyz: np.ndarray, video_frames_25: int, seq_start_25: int, seq_end_25):
    if int(video_frames_25) <= 0 or len(xyz) == 0:
        return xyz, max(0, int(seq_start_25))

    t = len(xyz)
    out_xyz = np.repeat(xyz[-1:, ...], repeats=int(video_frames_25), axis=0)
    start = int(np.clip(int(seq_start_25), 0, int(video_frames_25) - 1))
    out_xyz[:start] = xyz[0]

    if seq_end_25 is not None:
        place_end = int(np.clip(int(seq_end_25), start, int(video_frames_25)))
        max_place = max(0, place_end - start)
        place_len = min(t, max_place)
    else:
        place_len = min(t, int(video_frames_25) - start)

    if place_len > 0:
        out_xyz[start:start + place_len] = xyz[:place_len]

    return out_xyz, 0


def temporal_linear_resample_ignore_nans(hands_data: np.ndarray, new_len: int) -> np.ndarray:
    from scipy.interpolate import interp1d

    original_len = hands_data.shape[0]
    if original_len <= 1 or new_len <= 1 or new_len == original_len:
        return np.array(hands_data, copy=True)

    joints = hands_data.shape[2]
    dims = hands_data.shape[3]
    flat = hands_data.reshape(original_len, -1)
    output = np.full((new_len, flat.shape[1]), np.nan, dtype=np.float32)

    src_t = np.arange(original_len, dtype=np.float32)
    dst_t = np.linspace(0.0, float(original_len - 1), num=new_len, dtype=np.float32)

    for channel_idx in range(flat.shape[1]):
        values = flat[:, channel_idx]
        valid = np.isfinite(values)
        valid_count = int(valid.sum())
        if valid_count == 0:
            continue
        if valid_count == 1:
            output[:, channel_idx] = values[valid][0]
            continue

        partial_t = src_t[valid]
        partial_v = values[valid]
        this_kind = "cubic" if valid_count > 3 else "quadratic" if valid_count > 2 else "linear"
        f = interp1d(partial_t, partial_v, kind=this_kind, bounds_error=False, fill_value=np.nan)
        interp_values = np.asarray(f(dst_t), dtype=np.float32)
        interp_values[dst_t < partial_t[0]] = partial_v[0]
        interp_values[dst_t > partial_t[-1]] = partial_v[-1]
        output[:, channel_idx] = interp_values

    out_np = output.reshape(new_len, 1, joints, dims)
    return out_np


def temporal_nearest_resample_np(values: np.ndarray, new_len: int) -> np.ndarray:
    original_len = int(values.shape[0])
    if original_len <= 1 or new_len <= 1 or new_len == original_len:
        return np.array(values, copy=True)

    idx = np.rint(np.linspace(0.0, float(original_len - 1), num=int(new_len), dtype=np.float32)).astype(np.int64)
    return values[idx]


def compute_place_window(video_frames_25: int, seq_len_25: int, seq_start_25: int, seq_end_25):
    if int(video_frames_25) <= 0 or int(seq_len_25) <= 0:
        return 0, 0

    start = int(np.clip(int(seq_start_25), 0, int(video_frames_25) - 1))
    if seq_end_25 is not None:
        place_end = int(np.clip(int(seq_end_25), start, int(video_frames_25)))
        max_place = max(0, place_end - start)
        place_len = min(int(seq_len_25), max_place)
    else:
        place_len = min(int(seq_len_25), int(video_frames_25) - start)
    return start, place_len


def expand_hand_confidence_to_points(hand_confidence: np.ndarray) -> np.ndarray:
    left = np.repeat(hand_confidence[:, 0:1], repeats=21, axis=1)
    right = np.repeat(hand_confidence[:, 1:2], repeats=21, axis=1)
    return np.concatenate([left, right], axis=1)


def build_dynhamr_3d_sample(
    dyn_dir: Path,
    video_length: int,
    source_fps: float,
    target_fps: float,
    fill_mode: str,
    long_gap_linear_threshold: int,
    to_camera_coordinates: bool,
    enable_confidence_scoring: bool,
    confidence_real: float,
    confidence_filled: float,
    confidence_padded: float,
):
    left = np.load(dyn_dir / "joints_3d_left.npy").astype(np.float32)
    right = np.load(dyn_dir / "joints_3d_right.npy").astype(np.float32)
    cam_r = np.load(dyn_dir / "cam_R.npy").astype(np.float32)
    cam_t = np.load(dyn_dir / "cam_t.npy").astype(np.float32)

    t = min(len(left), len(right), len(cam_r), len(cam_t))
    left = left[:t]
    right = right[:t]
    cam_r = cam_r[:t]
    cam_t = cam_t[:t]

    if to_camera_coordinates:
        right_cam = np.zeros_like(right)
        left_cam = np.zeros_like(left)
        for frame_idx in range(t):
            if not np.isnan(right[frame_idx]).any():
                right_cam[frame_idx] = world_to_cam(right[frame_idx], cam_r[frame_idx], cam_t[frame_idx])
            else:
                right_cam[frame_idx] = np.nan
            if not np.isnan(left[frame_idx]).any():
                left_cam[frame_idx] = world_to_cam(left[frame_idx], cam_r[frame_idx], cam_t[frame_idx])
            else:
                left_cam[frame_idx] = np.nan
        xyz = np.concatenate([left_cam, right_cam], axis=1)
        xyz = sanitize_3d_joints_np(xyz, min_depth_m=0.15)
    else:
        xyz = np.concatenate([left, right], axis=1)

    left_valid_pre_fill = np.isfinite(xyz[:, :21, :]).all(axis=(1, 2))
    right_valid_pre_fill = np.isfinite(xyz[:, 21:, :]).all(axis=(1, 2))
    hand_valid_pre_fill = np.stack([left_valid_pre_fill, right_valid_pre_fill], axis=1)

    if fill_mode == "neighbor_average":
        xyz = temporal_fill_nans_neighbor_average_np(xyz)
    elif fill_mode == "hybrid":
        xyz = temporal_fill_nans_hybrid_np(xyz, long_gap_linear_threshold=long_gap_linear_threshold)
    else:
        raise ValueError(f"Unknown fill_mode={fill_mode}")

    seq_start_25, seq_end_25 = load_track_seq_interval(dyn_dir)
    orig_len_25 = int(hand_valid_pre_fill.shape[0])
    xyz, _ = pad_dyn_sequence_to_video_length_np(xyz, video_length, seq_start_25, seq_end_25)

    hand_valid_25 = np.zeros((xyz.shape[0], 2), dtype=bool)
    hand_padded_25 = np.zeros((xyz.shape[0], 2), dtype=bool)
    if int(video_length) > 0 and orig_len_25 > 0:
        start_25, place_len_25 = compute_place_window(video_length, orig_len_25, seq_start_25, seq_end_25)
        if start_25 > 0:
            hand_padded_25[:start_25, :] = True
        tail_start = start_25 + place_len_25
        if tail_start < xyz.shape[0]:
            hand_padded_25[tail_start:, :] = True
        if place_len_25 > 0:
            hand_valid_25[start_25:tail_start, :] = hand_valid_pre_fill[:place_len_25, :]
    else:
        keep = min(xyz.shape[0], orig_len_25)
        if keep > 0:
            hand_valid_25[:keep, :] = hand_valid_pre_fill[:keep, :]

    if not np.isfinite(xyz).all():
        xyz = np.nan_to_num(xyz, nan=0.0)

    ratio = target_fps / source_fps if source_fps > 0 else 1.0
    target_len = max(1, int(round(xyz.shape[0] * ratio)))
    xyz_30 = temporal_linear_resample_ignore_nans(xyz[:, None, :, :], target_len).squeeze(1)

    if not np.isfinite(xyz_30).all():
        xyz_30 = np.nan_to_num(xyz_30, nan=0.0)

    # PoseBody confidence uses per-point values. We keep per-hand scores and broadcast to 21 joints per hand.
    if enable_confidence_scoring:
        hand_filled_25 = (~hand_valid_25) & (~hand_padded_25)
        hand_conf_25 = np.full((xyz.shape[0], 2), fill_value=float(confidence_real), dtype=np.float32)
        hand_conf_25[hand_filled_25] = float(confidence_filled)
        hand_conf_25[hand_padded_25] = float(confidence_padded)

        hand_conf_30 = temporal_nearest_resample_np(hand_conf_25, target_len).astype(np.float32, copy=False)
        confidence_points = expand_hand_confidence_to_points(hand_conf_30)
        confidence = confidence_points[:, None, :]

        left_counts = (
            int((~hand_padded_25[:, 0] & hand_valid_25[:, 0]).sum()),
            int(hand_filled_25[:, 0].sum()),
            int(hand_padded_25[:, 0].sum()),
        )
        right_counts = (
            int((~hand_padded_25[:, 1] & hand_valid_25[:, 1]).sum()),
            int(hand_filled_25[:, 1].sum()),
            int(hand_padded_25[:, 1].sum()),
        )
        logger.info(
            "Confidence provenance for %s | left(real/filled/padded)=%s | right(real/filled/padded)=%s",
            dyn_dir.name,
            left_counts,
            right_counts,
        )
    else:
        confidence = np.ones((xyz_30.shape[0], 1, 42), dtype=bool)

    return xyz_30, confidence


def write_pose(output_pose_file: Path, xyz_30: np.ndarray, confidence: np.ndarray, target_fps: float):
    header = create_dynhamr_3d_pose_header()
    if xyz_30.ndim != 3 or xyz_30.shape[1] != 42 or xyz_30.shape[2] != 3:
        raise RuntimeError(f"Unexpected xyz_30 shape before pose write: {tuple(xyz_30.shape)}")

    # PoseBody uses flattened points across all components: (T, 1, 42, 3)
    data = xyz_30[:, None, :, :].astype(np.float32, copy=False)
    print(f"Wrote data.shape: {data.shape}")
    body = NumPyPoseBody(fps=int(round(target_fps)), data=data, confidence=confidence)
    pose = Pose(header=header, body=body)
    output_pose_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pose_file, "wb") as file_obj:
        pose.write(file_obj)
    print(f"Saved file to: {output_pose_file}")


def main():
    parser = argparse.ArgumentParser(description="Precompute DynHaMR 3D into .pose files")
    parser.add_argument("--input_tsv", required=True, help="Original How2Sign DynHaMR TSV")
    parser.add_argument("--input_dir", required=True, help="Directory containing DynHaMR sample folders")
    parser.add_argument("--output_pose_dir", required=True, help="Directory where .pose files will be written")
    parser.add_argument("--output_tsv", required=True, help="Output TSV pointing to the .pose files")
    parser.add_argument("--source_fps", type=float, default=25.0)
    parser.add_argument("--target_fps", type=float, default=30.0)
    parser.add_argument("--fill_mode", choices=["neighbor_average", "hybrid"], default="neighbor_average")
    parser.add_argument("--long_gap_linear_threshold", type=int, default=8)
    parser.add_argument("--to_camera_coordinates", action="store_true", default=True)
    parser.add_argument("--no_camera_coordinates", dest="to_camera_coordinates", action="store_false")
    parser.add_argument("--enable_confidence_scoring", action="store_true", default=False)
    parser.add_argument("--confidence_real", type=float, default=1.0)
    parser.add_argument("--confidence_filled", type=float, default=0.6)
    parser.add_argument("--confidence_padded", type=float, default=0.1)

    args = parser.parse_args()
    logger.info("Coordinate mode: %s", "camera" if args.to_camera_coordinates else "world")
    logger.info("Confidence scoring: %s", "enabled" if args.enable_confidence_scoring else "disabled")

    input_tsv = Path(args.input_tsv).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_pose_dir = Path(args.output_pose_dir).expanduser().resolve()
    output_tsv = Path(args.output_tsv).expanduser().resolve()

    df = pd.read_csv(input_tsv, sep="\t")
    if "id_vid" not in df.columns or "signs_file" not in df.columns:
        raise RuntimeError(f"Input TSV missing required columns: {input_tsv}")

    output_rows = []
    output_pose_dir.mkdir(parents=True, exist_ok=True)

    for id_vid, group in df.groupby("id_vid", sort=False):
        first_row = group.iloc[0]
        signs_file = str(first_row["signs_file"])
        dyn_dir = Path(signs_file)
        if not dyn_dir.is_absolute():
            dyn_dir = input_dir / dyn_dir
        dyn_dir = dyn_dir.expanduser().resolve()

        if not dyn_dir.is_dir():
            raise FileNotFoundError(f"Missing DynHaMR directory for {id_vid}: {dyn_dir}")

        video_length = int(first_row.get("video_length", 0))
        pose_file = output_pose_dir / f"{id_vid}.pose"

        if pose_file.exists():
            logger.info("Skipping existing pose file: %s", pose_file)
        else:
            xyz_30, confidence = build_dynhamr_3d_sample(
                dyn_dir=dyn_dir,
                video_length=video_length,
                source_fps=float(args.source_fps),
                target_fps=float(args.target_fps),
                fill_mode=str(args.fill_mode),
                long_gap_linear_threshold=int(args.long_gap_linear_threshold),
                to_camera_coordinates=bool(args.to_camera_coordinates),
                enable_confidence_scoring=bool(args.enable_confidence_scoring),
                confidence_real=float(args.confidence_real),
                confidence_filled=float(args.confidence_filled),
                confidence_padded=float(args.confidence_padded),
            )
            write_pose(pose_file, xyz_30, confidence, target_fps=float(args.target_fps))

        for _, row in group.iterrows():
            new_row = row.to_dict()
            new_row["signs_file"] = str(pose_file)
            output_rows.append(new_row)

    out_df = pd.DataFrame(output_rows)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_tsv, sep="\t", index=False)
    logger.info("Saved %d rows to %s", len(out_df), output_tsv)


if __name__ == "__main__":
    main()
