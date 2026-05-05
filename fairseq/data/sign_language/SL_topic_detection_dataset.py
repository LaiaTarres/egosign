import os
import sys
import logging
from enum import Enum
from pathlib import Path
from typing import List, Union, Optional
import csv
import json
import copy

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

import torch
import torch.nn.functional as F

from pose_format import Pose # TODO: We will load from the .pose files
from pose_format.numpy.pose_body import NumPyPoseBody

from fairseq.data import FairseqDataset

from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

logger = logging.getLogger(__name__)


class SignFeatsType_TD(Enum):
    text = "text"
    text_albert = "text_albert"
    spot_align = "spot_align"
    mouthings = "mouthings"
    spot_align_albert = "spot_align_albert"
    mouthings_albert = "mouthings_albert"
    keypoints = "keypoints"
    mediapipe_keypoints = "mediapipe_keypoints"
    dynhamr = "dynhamr"
    dynhamr_2d = "dynhamr_2d"
    rotational = "rotational"
    mediapipe_rotational = "mediapipe_rotational"
    i3d = "i3d"
    CNN2d = "CNN2d"
    video = "video"

class NormType_TD(Enum):
    none = "none"
    body="body"
    kp_wise = "kp_wise"
    global_xyz = "global_xyz"
    layer_norm = "layer_norm" #to add the same normalizaiton as original TD
    center_and_scale = "center_and_scale"
    wrist_fixed_scale = "wrist_fixed_scale"  # Fixed-scale wrist normalization for DynHaMR
    global_z_norm = "global_z_norm" # Global Z-Norm (standardization) using precomputed mean/std from the How2Sign training set (post-interpolation), for DynHaMR 3D poses.
    
class SLTopicDetectionDataset(FairseqDataset):
    _debug_dump_instance_counter = 0

    def __init__(
        self,
        ids: List[str],
        feats_files: List[Union[Path, str]],
        offsets: List[int],
        sizes: List[int],
        feats_type: SignFeatsType_TD,
        ids_sent: List[List[str]],
        sentence_sizes: Optional[List[List[int]]] = None,
        normalization: NormType_TD = NormType_TD.body,
        data_augmentation: bool = False,
        min_sample_size: int = 0,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        to_camera_coordinates: bool = True,
        dynhamr_temporal_interpolation: bool = False,
        dynhamr_source_fps: float = 25.0,
        dynhamr_target_fps: float = 30.0,
        dynhamr_fill_mode: str = "neighbor_average",
        dynhamr_long_gap_linear_threshold: int = 10,
        dynhamr_savgol_window: int = 9,
        dynhamr_savgol_polyorder: int = 2,
        dynhamr_temp_map_to_mediapipe_2d: bool = False,
        dynhamr_temp_interpolate_missing: bool = False,
        dynhamr_temp_similarity_scale: float = 1.0,
        dynhamr_temp_translate_x: float = 0.0,
        dynhamr_temp_translate_y: float = 0.0,
        dynhamr_temp_per_video_map_to_mediapipe_2d: bool = False,
        dynhamr_temp_per_video_target_center_x: float = -0.13,
        dynhamr_temp_per_video_target_center_y: float = 0.43,
        dynhamr_temp_per_video_target_wrist_distance: float = 0.30,
        dynhamr_temp_mp_shoulder_norm: bool = False,
        dynhamr_temp_mp_pose_manifest_file: Optional[Union[Path, str]] = None,
        dynhamr_temp_fail_on_zeros: bool = False,
        dynhamr_2d_mp_pose_manifest_file: Optional[Union[Path, str]] = None,
        use_preprocessed_dynhamr_2d: bool = False,
        preprocessed_dynhamr_2d_tsv_path: Optional[Union[Path, str]] = None,
        aug3d_random_resample: bool = False,
        aug3d_resample_p: float = 0.5,
        aug3d_resample_limit: float = 0.2,
        aug3d_frame_noise: bool = False,
        aug3d_frame_noise_ratio: float = 0.1,
        aug3d_frame_noise_std: float = 0.01,
        aug3d_feature_mask: bool = False,
        aug3d_feature_mask_ratio: float = 0.1,
        aug3d_frame_mask: bool = False,
        aug3d_frame_mask_ratio: float = 0.05,
        aug3d_scale: bool = False,
        aug3d_scale_limit: float = 0.1,
        aug3d_shift: bool = False,
        aug3d_shift_std: float = 0.02,
        aug3d_horizontal_flip: bool = False,
        aug3d_horizontal_flip_p: float = 0.5,
        #manifest: pd.DataFrame,
        #ids: List[str],
        #feats_path: Union[Path, str],
        bodyparts: Optional[List[str]] = None,
        feat_dims: List[int] = [0, 1, 2, 3],
        #normalize: bool = False,
        #text_compression_level: TextCompressionLevel = TextCompressionLevel.none,
    ):
        ###
        # What we want is that is as close as possible to the SignFeatsDataset, but with all the funcionalities
        ###
        super().__init__()
        #self.text_compressor = TextCompressor(level=text_compression_level) # TODO: figure out whether we need this.
        
        self.ids = ids
        self.feats_files = feats_files
        self.offsets = offsets
        self.sizes = sizes
        self.sentence_sizes = sentence_sizes if sentence_sizes is not None else [None] * len(self.ids)
        self.feats_type = feats_type
        self.ids_sent = ids_sent
        self.normalization = normalization # I think this we were calling it normalize
        self.data_augmentation = data_augmentation
        self.min_sample_size = min_sample_size
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.shuffle = shuffle
        self.to_camera_coordinates = to_camera_coordinates
        self.dynhamr_temporal_interpolation = dynhamr_temporal_interpolation
        self.dynhamr_source_fps = dynhamr_source_fps
        self.dynhamr_target_fps = dynhamr_target_fps
        self.dynhamr_fill_mode = str(dynhamr_fill_mode)
        self.dynhamr_long_gap_linear_threshold = int(dynhamr_long_gap_linear_threshold)
        self.dynhamr_savgol_window = int(dynhamr_savgol_window)
        self.dynhamr_savgol_polyorder = int(dynhamr_savgol_polyorder)
        self.dynhamr_temp_map_to_mediapipe_2d = dynhamr_temp_map_to_mediapipe_2d
        self.dynhamr_temp_interpolate_missing = dynhamr_temp_interpolate_missing
        self.dynhamr_temp_similarity_scale = dynhamr_temp_similarity_scale
        self.dynhamr_temp_translate_x = dynhamr_temp_translate_x
        self.dynhamr_temp_translate_y = dynhamr_temp_translate_y
        self.dynhamr_temp_per_video_map_to_mediapipe_2d = dynhamr_temp_per_video_map_to_mediapipe_2d
        self.dynhamr_temp_per_video_target_center_x = dynhamr_temp_per_video_target_center_x
        self.dynhamr_temp_per_video_target_center_y = dynhamr_temp_per_video_target_center_y
        self.dynhamr_temp_per_video_target_wrist_distance = dynhamr_temp_per_video_target_wrist_distance
        self.dynhamr_temp_mp_shoulder_norm = dynhamr_temp_mp_shoulder_norm
        self.dynhamr_temp_mp_pose_manifest_file = dynhamr_temp_mp_pose_manifest_file
        self.dynhamr_temp_fail_on_zeros = dynhamr_temp_fail_on_zeros
        self.dynhamr_2d_mp_pose_manifest_file = dynhamr_2d_mp_pose_manifest_file
        self.use_preprocessed_dynhamr_2d = use_preprocessed_dynhamr_2d
        self.preprocessed_dynhamr_2d_tsv_path = preprocessed_dynhamr_2d_tsv_path
        self.aug3d_random_resample = aug3d_random_resample
        self.aug3d_resample_p = aug3d_resample_p
        self.aug3d_resample_limit = aug3d_resample_limit
        self.aug3d_frame_noise = aug3d_frame_noise
        self.aug3d_frame_noise_ratio = aug3d_frame_noise_ratio
        self.aug3d_frame_noise_std = aug3d_frame_noise_std
        self.aug3d_feature_mask = aug3d_feature_mask
        self.aug3d_feature_mask_ratio = aug3d_feature_mask_ratio
        self.aug3d_frame_mask = aug3d_frame_mask
        self.aug3d_frame_mask_ratio = aug3d_frame_mask_ratio
        self.aug3d_scale = aug3d_scale
        self.aug3d_scale_limit = aug3d_scale_limit
        self.aug3d_shift = aug3d_shift
        self.aug3d_shift_std = aug3d_shift_std
        self.aug3d_horizontal_flip = aug3d_horizontal_flip
        self.aug3d_horizontal_flip_p = aug3d_horizontal_flip_p
        self.skipped_ids = []
        self._warned_temp_drop_z_patch = False
        self._warned_temp_dynhamr_interp_patch = False
        self._warned_temp_dynhamr_interp_nans = False
        self._printed_temp_dynhamr_interp_example = False
        self._warned_temp_dynhamr_missing_fill = False
        self._warned_temp_dynhamr_remaining_nans = False
        self._warned_temp_similarity_map = False
        self._warned_temp_per_video_similarity_map = False
        self._warned_temp_mp_shoulder_norm_map = False
        self._warned_temp_zero_values = False
        self._warned_dynhamr_2d_missing_mp_manifest = False
        self._warned_dynhamr_2d_missing_mp_pose = False
        self._warned_dynhamr_2d_remaining_nans = False
        self._warned_dynhamr_2d_pipeline = False
        self._warned_dynhamr_2d_missing_video_len_for_padding = False
        self._warned_dynhamr_2d_padding_length_mismatch = False
        self._warned_dynhamr_2d_nonzero_seq_start_after_padding = False
        self._warned_dynhamr_2d_empty_sentence_span = False
        self._warned_dynhamr_3d_missing_video_len_for_padding = False
        self._warned_dynhamr_3d_padding_length_mismatch = False
        self._warned_dynhamr_3d_nonzero_seq_start_after_padding = False

        self._temp_mp_pose_path_by_vid = {}
        self._dynhamr_2d_mp_pose_path_by_vid = {}
        self._dynhamr_2d_missing_mp_pose_vids = []
        self._dynhamr_2d_mp_shoulder_cache = {}
        self._preprocessed_dynhamr_2d_pose_path_by_vid = {}

        self.dynhamr_2d_select_sentence_span = os.environ.get(
            "SLTD_DYNHAMR_2D_SELECT_SENTENCE_SPAN", "false"
        ).strip().lower() in ("1", "true", "yes", "y")
        self.dynhamr_2d_disable_savgol = os.environ.get(
            "SLTD_DYNHAMR_2D_DISABLE_SAVGOL", "false"
        ).strip().lower() in ("1", "true", "yes", "y")
        self.dynhamr_2d_disable_padding = os.environ.get(
            "SLTD_DYNHAMR_2D_DISABLE_PADDING", "false"
        ).strip().lower() in ("1", "true", "yes", "y")
        self.dynhamr_2d_disable_mp_shoulder_norm = os.environ.get(
            "SLTD_DYNHAMR_2D_DISABLE_MP_SHOULDER_NORM", "false"
        ).strip().lower() in ("1", "true", "yes", "y")
        if self.dynhamr_2d_select_sentence_span:
            logger.warning(
                "[EXPERIMENT] DynHaMR 2D sentence-span selection enabled: selecting frames from first sentence start to last sentence end before postprocess."
            )
        if self.dynhamr_2d_disable_savgol:
            logger.warning("[EXPERIMENT] DynHaMR 2D Savitzky-Golay smoothing disabled.")
        if self.dynhamr_2d_disable_padding:
            logger.warning("[EXPERIMENT] DynHaMR 2D pad-to-video-length disabled.")
        if self.dynhamr_2d_disable_mp_shoulder_norm:
            logger.warning("[EXPERIMENT] DynHaMR 2D MP shoulder center/scale normalization disabled (wrist fallback will be used).")
        
        # Initialize preprocessed DynHaMR 2D mode
        if self.use_preprocessed_dynhamr_2d and self.preprocessed_dynhamr_2d_tsv_path:
            self._preprocessed_dynhamr_2d_pose_path_by_vid = self._read_preprocessed_dynhamr_2d_mappings(
                self.preprocessed_dynhamr_2d_tsv_path
            )
            missing_preprocessed = [
                vid_id for vid_id in self.ids if vid_id not in self._preprocessed_dynhamr_2d_pose_path_by_vid
            ]
            if missing_preprocessed:
                logger.warning(
                    "[WARNING] Preprocessed DynHaMR 2D pose coverage: missing %d/%d id_vid entries",
                    len(missing_preprocessed),
                    len(self.ids),
                )
            else:
                logger.info(
                    "Preprocessed DynHaMR 2D pose coverage: all %d id_vid entries available",
                    len(self.ids),
                )
            logger.info(
                "[INFO] Using preprocessed DynHaMR 2D mode: loading directly from .pose files, "
                "no runtime 3D->2D conversion needed."
            )
        elif self.use_preprocessed_dynhamr_2d and not self.preprocessed_dynhamr_2d_tsv_path:
            logger.warning(
                "[WARNING] use_preprocessed_dynhamr_2d=True but preprocessed_dynhamr_2d_tsv_path not set. "
                "Falling back to standard DynHaMR 2D pipeline."
            )
            self.use_preprocessed_dynhamr_2d = False
        
        if SignFeatsType_TD(self.feats_type) == SignFeatsType_TD.dynhamr_2d:
            if self.dynhamr_2d_mp_pose_manifest_file:
                if "normalized" in str(self.dynhamr_2d_mp_pose_manifest_file):
                    raise RuntimeError(
                        "DynHaMR 2D requires MediaPipe *_smooth TSV for shoulder center/scale, "
                        "not *_smooth_normalized TSV."
                    )
                self._dynhamr_2d_mp_pose_path_by_vid = self._read_dynhamr_2d_mp_pose_mappings(
                    self.dynhamr_2d_mp_pose_manifest_file
                )

                # Report exact MP-pose coverage for DynHaMR 2D ids at startup.
                self._dynhamr_2d_missing_mp_pose_vids = [
                    vid_id for vid_id in self.ids if vid_id not in self._dynhamr_2d_mp_pose_path_by_vid
                ]
                if self._dynhamr_2d_missing_mp_pose_vids:
                    logger.warning(
                        "[WARNING] DynHaMR 2D MP-pose coverage: missing %d/%d id_vid entries. Missing id_vid list: %s",
                        len(self._dynhamr_2d_missing_mp_pose_vids),
                        len(self.ids),
                        ", ".join(self._dynhamr_2d_missing_mp_pose_vids),
                    )
                else:
                    logger.info(
                        "DynHaMR 2D MP-pose coverage: missing 0/%d id_vid entries.",
                        len(self.ids),
                    )
            else:
                self._warned_dynhamr_2d_missing_mp_manifest = True
                logger.warning(
                    "[WARNING] DynHaMR 2D mode enabled without dynhamr_2d_mp_pose_manifest_file. "
                    "Falling back to temporal fps ratio for sequence length and wrist-based center/scale normalization."
                )

            if not self._warned_dynhamr_2d_pipeline:
                print(
                    "[INFO] DynHaMR 2D pipeline enabled: world->camera->pixel, NaN-aware temporal interpolation, "
                    "resample to target fps, normalize by MediaPipe shoulder center/scale when available."
                )
                self._warned_dynhamr_2d_pipeline = True

        if self.dynhamr_temp_mp_shoulder_norm:
            if self.dynhamr_temp_mp_pose_manifest_file:
                self._temp_mp_pose_path_by_vid = self._read_manifest_id_to_signs_file(
                    self.dynhamr_temp_mp_pose_manifest_file
                )
            print(
                "[TEMP PATCH][REMOVE ME] DynHaMR temporary MP-shoulder normalization mapping enabled: "
                "center by wrist midpoint, scale by MediaPipe shoulder distance."
            )

        if self.dynhamr_temporal_interpolation:
            print(
                "[TEMP PATCH][REMOVE ME] DynHaMR temporal interpolation enabled in dataset: "
                f"{self.dynhamr_source_fps}fps -> {self.dynhamr_target_fps}fps"
            )
            print(
                "[TEMP PATCH][REMOVE ME] This is an experiment-only simulation to match sequence length/fps. "
                "Disable/remove once fps strategy is decided."
            )
        if self.dynhamr_temp_map_to_mediapipe_2d:
            print(
                "[TEMP PATCH][REMOVE ME] DynHaMR temporary 2D mapping enabled: "
                "XY projection + optional global similarity transform."
            )

        if self.dynhamr_temp_map_to_mediapipe_2d and self.dynhamr_temp_mp_shoulder_norm:
            print(
                "[TEMP PATCH][REMOVE ME][WARNING] Both DynHaMR mapping modes are enabled. "
                "MP-shoulder normalization mapping will take precedence."
            )

        # The ones that we have removed: do we actually need this? Probably yes!
        self.bodyparts = bodyparts
        self.feat_dims = feat_dims
        self.debug_dump_dir = os.environ.get("SLTDUMP_DIR", "").strip()
        self.debug_dump_enabled = len(self.debug_dump_dir) > 0
        self.debug_dump_every_n = max(1, int(os.environ.get("SLTDUMP_EVERY_N", "1")))
        self.debug_dump_max_batches = max(0, int(os.environ.get("SLTDUMP_MAX_BATCHES", "0")))
        self._debug_dump_batch_idx = 0
        self._debug_dump_saved_batches = 0
        SLTopicDetectionDataset._debug_dump_instance_counter += 1
        self._debug_dump_instance_id = SLTopicDetectionDataset._debug_dump_instance_counter
        if self.debug_dump_enabled:
            os.makedirs(self.debug_dump_dir, exist_ok=True)
            logger.warning(
                "[DEBUG DUMP] Batch dump enabled: "
                f"dir={self.debug_dump_dir}, every_n={self.debug_dump_every_n}, "
                f"max_batches={self.debug_dump_max_batches if self.debug_dump_max_batches > 0 else 'unlimited'}, "
                f"instance_id={self._debug_dump_instance_id}, dataset_size={len(self.ids)}."
            )
        #self.manifest = manifest
        # if feats_type == SignFeatsType.video, feats_path is the directory where .mp4 files of the corresponding split are stored
        #self.ids = [_id for _id in ids]

    def filter_by_length(self, min_sample_size, max_sample_size):
        for _id, sizes, offsets in zip(self.ids[:], self.sizes[:], self.offsets):
            #Now here we have a combination of them, so what is the minimum? The combined ones
            #sum_size = sum(size) #We don't have a list of sizes
            
            # SELECT SENTENCE: when we are working with the filtering of sequences, we need to filter to the maximum size which is: 
            #print(f"LAIA: changed this in SL_topic_detection_dataset.py, in filter_by_length to include filtering when we are cutting to beginning of sentence to end of sentence. ")
            #sum_size = max([offset + size for offset, size in zip(offsets, sizes)]) - min(offsets)
            
            # SELECT WHOLE_VIDEO:
            sum_size = sizes

            if self.dynhamr_2d_select_sentence_span:
                idx = self.ids.index(_id)
                sentence_sizes = self.sentence_sizes[idx] if idx < len(self.sentence_sizes) else None
                if sentence_sizes and len(sentence_sizes) == len(offsets):
                    offsets_30 = [int(round(o * 30.0 / 25.0)) for o in offsets]
                    sizes_30 = [int(round(l * 30.0 / 25.0)) for l in sentence_sizes]
                    sum_size = max([off + size for off, size in zip(offsets_30, sizes_30)]) - min(offsets_30)

            if sum_size < self.min_sample_size or sum_size > self.max_sample_size:
                self.feats_files.pop(self.ids.index(_id))
                self.offsets.pop(self.ids.index(_id))
                self.sizes.pop(self.ids.index(_id))
                self.ids.remove(_id)
                self.skipped_ids.append(_id)
        logger.info(
            f"Filtered {len(self.skipped_ids)} videos, that were too short or too long."
        )
        
    @classmethod
    def from_manifest_file(cls, manifest_file: Union[str, Path], **kwargs):
        '''
        This way, we have the self.manifest loaded directly from the file
        
        TODO: for topic detection, we should combine all the files that have the same video_id, 
        and the different offsets. Because a sample is 1 video, with the combination of all the sentences.
        
        '''
        ids = []
        feats_files = []
        offsets = []
        sizes = []
        sentence_sizes = []
        ids_sent = [] 
        
        #manifest = pd.read_csv(manifest_file, sep="\t") #HELP! Here pandas is not loading some of the lines. Why??
        raw_lines = []
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                raw_lines.append((line_num, line.strip()))
        header = raw_lines[0][1].split('\t')
        manual_data = []
        for line_num, line in raw_lines[1:]:  # Skip header
            fields = line.split('\t')
            if len(fields) != len(header):
                #problematic_lines.append((line_num, line, len(fields)))
                continue

            row_dict = {header[i]: fields[i] for i in range(len(header))}
            manual_data.append(row_dict)
        manifest = pd.DataFrame(manual_data) # Until here is to solve whatever issue we are having with pandas
        
        manifest = manifest.drop_duplicates()
        
        from collections import defaultdict
        grouped_data = defaultdict(lambda: {
            "ids_sent": [],
            "feats_file": None,
            "offsets": [],
            "sizes": [],
        })
        # Iterate thorugh rows and group data by id_vid
        for _, row in manifest.iterrows():
            id_vid = row['id_vid']
            grouped_data[id_vid]["ids_sent"].append(row['id'])  # Original sentence id
            grouped_data[id_vid]["feats_file"] = row['signs_file']  # All rows in id_vid share the same feats_file
            grouped_data[id_vid]["offsets"].append(int(row['signs_offset']))
            grouped_data[id_vid]["sizes"].append(int(row['signs_length']))
            grouped_data[id_vid]["vid_length"]=int(row['video_length'])
            
            #Here we should only keep the ones that are not duplicate for all of them, because right now we have a lot of duplicated ones. 
            
        
        # We need an extra column that is the size of the whole video... because if not, we cannot know the length of the video
        # Process grouped data
        for id_vid, data in grouped_data.items():
            #import pdb; pdb.set_trace() # Here we need to keep the sentence start and end. 
            ids.append(id_vid)  # Use id_vid as the new id
            feats_files.append(data["feats_file"])
            offsets.append(data["offsets"])  # Concatenated offsets
            
            # SELECT SENTENCE: this we need it when we want to then cut the sentences. 
            #sizes.append(data["sizes"])  # Concatenated sizes
            
            # SELECT WHOLE_VIDEO: this is what we do when we are taking all of the lengths
            sizes.append(data["vid_length"])  # Only 1 size
            sentence_sizes.append(data["sizes"])  # Keep sentence lengths for optional sentence-span selection
            
            #total length should be the rest between: sum of the max offset with the corresponding size, and the minimum of the offsets
            #total_length = max([offset + size for offset, size in zip(data["offsets"], data["sizes"])]) - min(data["offsets"])
            #video_lengths.append(total_length)  # Concatenated sizes
            ids_sent.append(data["ids_sent"])  # Original sentence ids    
        
        logger.info(f"loaded {len(ids)} samples")
        
        feats_type = kwargs.pop("feats_type", row['signs_type'])
        return cls(ids, feats_files=feats_files, offsets=offsets, sizes=sizes, sentence_sizes=sentence_sizes,
            feats_type=feats_type, ids_sent=ids_sent, **kwargs)

        # En principi tot lo d'aquí sota no caldria
        '''

        if feats_type not in ['video']:
            if feats_type in ['text', 'spot_align', 'mouthings']:
                self.feats_file = self.manifest.set_index('VIDEO_ID').to_dict()['TEXT']
            else:
                self.feats_file = h5py.File(self.feats_path, 'r')
                if sizes is None:
                    sizes = []
                    for _id in self.ids:
                        _id = _id
                        sizes.append(np.array(self.feats_file[_id]).shape[0])
        self.sizes = sizes

        try:
            import pyarrow as pa
            self.ids = pa.array(self.ids)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass
        '''

    def world_to_cam(self, joints_3d, R, t):
        """
        Transform 3D joints from world coordinates to camera coordinates.
        joints_3d: (21, 3) or (N, 3)
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        Returns: (21, 3) or (N, 3) in camera coordinates
        """
        return joints_3d @ R.T + t

    def _read_manifest_id_to_signs_file(self, manifest_file: Union[Path, str]):
        mapping = {}
        try:
            with open(manifest_file, "r", encoding="utf-8") as file_obj:
                max_int = sys.maxsize
                while True:
                    try:
                        csv.field_size_limit(max_int)
                        break
                    except OverflowError:
                        max_int //= 10
                reader = csv.DictReader(file_obj, delimiter="\t")
                for row in reader:
                    vid = row.get("id_vid", "")
                    signs_file = row.get("signs_file", "")
                    if vid and signs_file and vid not in mapping:
                        mapping[vid] = signs_file
        except Exception as exc:
            logger.warning(
                f"[WARNING] Could not read MP pose manifest for shoulder normalization: {manifest_file} ({exc})"
            )
        return mapping

    def _read_dynhamr_2d_mp_pose_mappings(self, manifest_file: Union[Path, str]):
        manifest_path = Path(manifest_file)
        mappings = self._read_manifest_id_to_signs_file(manifest_path)

        stem = manifest_path.stem
        sibling_candidates = []
        if "_train_" in stem:
            sibling_candidates = [
                manifest_path.with_name(manifest_path.name.replace("_train_", "_val_")),
                manifest_path.with_name(manifest_path.name.replace("_train_", "_test_")),
            ]

        for candidate in sibling_candidates:
            if candidate.exists():
                sibling_map = self._read_manifest_id_to_signs_file(candidate)
                for vid, signs_file in sibling_map.items():
                    if vid not in mappings:
                        mappings[vid] = signs_file

        return mappings

    def _read_preprocessed_dynhamr_2d_mappings(self, tsv_file: Union[Path, str]):
        """
        Read preprocessed DynHaMR 2D TSV and create id_vid -> pose_file mapping.
        Also stores video_length for potential future use.
        """
        mappings = {}
        try:
            with open(tsv_file, "r", encoding="utf-8") as file_obj:
                max_int = sys.maxsize
                while True:
                    try:
                        csv.field_size_limit(max_int)
                        break
                    except OverflowError:
                        max_int //= 10
                reader = csv.DictReader(file_obj, delimiter="\t")
                for row in reader:
                    vid = row.get("id_vid", "")
                    signs_file = row.get("signs_file", "")
                    if vid and signs_file and vid not in mappings:
                        mappings[vid] = signs_file
        except Exception as exc:
            logger.warning(
                f"[WARNING] Could not read preprocessed DynHaMR 2D TSV: {tsv_file} ({exc})"
            )
        return mappings

    def _read_preprocessed_dynhamr_2d_pose(self, pose_file: Union[Path, str], vid_id: str):
        """
        Read preprocessed DynHaMR 2D .pose file and extract (T, 1, 42, 2) tensor.
        
        The .pose file contains:
        - data: (T, 2_components, 21_joints, 3_dims) where dims are [X, Y, Z]
        - confidence: (T, 2_components, 21_joints)
        
        We extract X and Y coordinates and reshape to (T, 1, 42, 2).
        """
        try:
            with open(pose_file, "rb") as f:
                pose = Pose.read(f.read())
            
            # Extract data: (T, 2_components, 21_joints, 3_dims)
            data = pose.body.data  # (T, 2, 21, 3)
            
            # Extract confidence: (T, 2_components, 21_joints)
            confidence = pose.body.confidence  # (T, 2, 21)
            
            # Flatten components and joints: (T, 2*21) = (T, 42)
            t_frames = data.shape[0]
            xy_flat = np.zeros((t_frames, 42, 2), dtype=np.float32)
            
            # Left hand (component 0)
            xy_flat[:, :21, 0] = data[:, 0, :, 0]  # X
            xy_flat[:, :21, 1] = data[:, 0, :, 1]  # Y
            
            # Right hand (component 1)
            xy_flat[:, 21:, 0] = data[:, 1, :, 0]  # X
            xy_flat[:, 21:, 1] = data[:, 1, :, 1]  # Y
            
            # Convert to torch and reshape: (T, 1, 42, 2)
            xy_tensor = torch.from_numpy(xy_flat).float().unsqueeze(1)
            return xy_tensor
            
        except Exception as e:
            logger.error(f"ERROR reading preprocessed pose file {pose_file} for {vid_id}: {e}", exc_info=True)
            raise

    def _read_preprocessed_dynhamr_3d_pose(self, pose_file: Union[Path, str], vid_id: str):
        """
        Read preprocessed DynHaMR 3D .pose file and extract (T, 1, 42, 3) tensor.

        The precomputed .pose file stores two components:
        - LEFT_HAND_LANDMARKS
        - RIGHT_HAND_LANDMARKS

        This method flattens them into the same (T, 1, 42, 3) layout expected by the
        existing DynHaMR 3D postprocess path.
        """
        try:
            with open(pose_file, "rb") as f:
                pose = Pose.read(f.read())

            data = pose.body.data
            if isinstance(data, np.ma.MaskedArray):
                data = np.asarray(np.ma.filled(data, np.nan), dtype=np.float32)
            else:
                data = np.asarray(data, dtype=np.float32)

            # Canonical precomputed format: PoseBody(T, People=1, Points=42, Dims=3)
            if data.ndim == 4 and data.shape[1] == 1 and data.shape[2] == 42 and data.shape[3] >= 3:
                xyz_tensor = torch.from_numpy(data[:, :, :, :3]).float()
                return xyz_tensor

            # Legacy experimental format: (T, 2_components, 21_joints, 3)
            if data.ndim == 4 and data.shape[1] == 2 and data.shape[2] == 21 and data.shape[3] >= 3:
                t_frames = data.shape[0]
                xyz_flat = np.zeros((t_frames, 42, 3), dtype=np.float32)
                xyz_flat[:, :21, :] = data[:, 0, :, :3]
                xyz_flat[:, 21:, :] = data[:, 1, :, :3]
                xyz_tensor = torch.from_numpy(xyz_flat).float().unsqueeze(1)
                return xyz_tensor

            # Known malformed files from older writer bug where axis-1 was treated as components.
            if data.ndim == 4 and data.shape[1] == 2 and data.shape[2] == 42 and data.shape[3] >= 3:
                raise RuntimeError(
                    f"Unexpected malformed DynHaMR 3D .pose shape for {vid_id}: {tuple(data.shape)}. "
                    f"This file was likely written with People=2 instead of People=1. "
                    f"Please regenerate with the fixed precompute script."
                )

            raise RuntimeError(
                f"Unexpected DynHaMR 3D .pose shape for {vid_id}: {tuple(data.shape)}"
            )

        except Exception as e:
            logger.error(f"ERROR reading preprocessed DynHaMR 3D pose file {pose_file} for {vid_id}: {e}", exc_info=True)
            raise

    def _extract_mp_shoulder_distance(self, vid_id):
        fallback = 1.0
        mp_pose_path = self._temp_mp_pose_path_by_vid.get(vid_id, None)
        if not mp_pose_path:
            return fallback

        try:
            with open(mp_pose_path, "rb") as f:
                raw_mp_pose = Pose.read(f.read())

            comp_idx = next(
                (idx for idx, c in enumerate(raw_mp_pose.header.components) if c.name == "POSE_LANDMARKS"),
                None,
            )
            if comp_idx is None:
                return fallback

            left_sh = raw_mp_pose.body.data[:, comp_idx, 11, :2]
            right_sh = raw_mp_pose.body.data[:, comp_idx, 12, :2]

            valid_sh = ~(left_sh.mask.any(axis=1) | right_sh.mask.any(axis=1))
            if valid_sh.any():
                dists = np.linalg.norm(left_sh[valid_sh] - right_sh[valid_sh], axis=1)
                if len(dists) > 0:
                    mp_shoulder_dist = float(np.median(dists))
                    if np.isfinite(mp_shoulder_dist) and mp_shoulder_dist > 1e-8:
                        return mp_shoulder_dist
        except Exception as exc:
            logger.warning(f"[WARNING] Could not extract MP shoulders for {vid_id}: {exc}")

        return fallback

    def _load_track_seq_start(self, dyn_dir: Union[Path, str]) -> int:
        info_path = Path(dyn_dir) / "track_info.json"
        if not info_path.exists():
            return 0
        try:
            payload = json.loads(info_path.read_text())
            meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
            seq_interval = meta.get("seq_interval", None)
            if isinstance(seq_interval, list) and len(seq_interval) == 2:
                return int(seq_interval[0])
        except Exception:
            return 0
        return 0

    def _load_track_seq_interval(self, dyn_dir: Union[Path, str]):
        info_path = Path(dyn_dir) / "track_info.json"
        if not info_path.exists():
            return 0, None
        try:
            payload = json.loads(info_path.read_text())
            meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
            seq_interval = meta.get("seq_interval", None)
            if isinstance(seq_interval, list) and len(seq_interval) == 2:
                return int(seq_interval[0]), int(seq_interval[1])
        except Exception:
            return 0, None
        return 0, None

    def _pad_dyn_sequence_to_video_length_np(
        self,
        xy: np.ndarray,
        valid: np.ndarray,
        video_frames_25: int,
        seq_start_25: int,
        seq_end_25: Optional[int],
    ):
        if int(video_frames_25) <= 0 or len(xy) == 0:
            if not self._warned_dynhamr_2d_missing_video_len_for_padding:
                logger.warning(
                    "[WARNING] DynHaMR 2D padding skipped because target video length is unavailable or invalid. "
                    "Using raw track length instead."
                )
                self._warned_dynhamr_2d_missing_video_len_for_padding = True
            return xy, valid, max(0, int(seq_start_25))

        t = len(xy)
        out_xy = np.repeat(xy[-1:, ...], repeats=int(video_frames_25), axis=0)
        out_valid = np.repeat(valid[-1:, ...], repeats=int(video_frames_25), axis=0)

        start = int(np.clip(int(seq_start_25), 0, int(video_frames_25) - 1))
        out_xy[:start] = xy[0]
        out_valid[:start] = valid[0]

        if seq_end_25 is not None:
            place_end = int(np.clip(int(seq_end_25), start, int(video_frames_25)))
            max_place = max(0, place_end - start)
            place_len = min(t, max_place)
        else:
            place_len = min(t, int(video_frames_25) - start)

        if place_len > 0:
            out_xy[start:start + place_len] = xy[:place_len]
            out_valid[start:start + place_len] = valid[:place_len]

        return out_xy, out_valid, 0

    def _pad_dyn_sequence_3d_to_video_length_np(
        self,
        xyz: np.ndarray,
        video_frames_25: int,
        seq_start_25: int,
        seq_end_25: Optional[int],
    ):
        if int(video_frames_25) <= 0 or len(xyz) == 0:
            if not self._warned_dynhamr_3d_missing_video_len_for_padding:
                logger.warning(
                    "[WARNING] DynHaMR 3D padding skipped because target video length is unavailable or invalid. "
                    "Using raw track length instead."
                )
                self._warned_dynhamr_3d_missing_video_len_for_padding = True
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

    def _smooth_2d_sequence_savgol_np(
        self,
        xy: np.ndarray,
        valid: np.ndarray,
        window_length: int,
        polyorder: int,
    ):
        out = np.array(xy, copy=True, dtype=np.float32)
        t = out.shape[0]
        if t < 3:
            return out, valid & np.isfinite(out).all(axis=-1)

        req_w = max(3, int(window_length))
        if req_w % 2 == 0:
            req_w += 1
        max_w = t if (t % 2 == 1) else (t - 1)
        w = min(req_w, max_w)
        if w <= int(polyorder):
            return out, valid & np.isfinite(out).all(axis=-1)

        idx = np.arange(t, dtype=np.float32)
        for hs, he in [(0, 21), (21, 42)]:
            hand = out[:, hs:he, :]
            hand_valid = valid[:, hs:he] & np.isfinite(hand).all(axis=-1)

            hand_filled = np.array(hand, copy=True, dtype=np.float32)
            for j in range(hand_filled.shape[1]):
                for d in range(2):
                    series = hand_filled[:, j, d]
                    m = hand_valid[:, j] & np.isfinite(series)
                    count = int(m.sum())
                    if count == 0:
                        continue
                    if count == 1:
                        series[:] = float(series[m][0])
                    elif count < t:
                        series[:] = np.interp(idx, idx[m], series[m]).astype(np.float32)
                    hand_filled[:, j, d] = series

            hand_smoothed = savgol_filter(
                hand_filled,
                window_length=w,
                polyorder=min(int(polyorder), w - 1),
                axis=0,
                mode="interp",
            ).astype(np.float32)

            valid3 = hand_valid[..., None]
            out[:, hs:he, :] = np.where(valid3, hand_smoothed, out[:, hs:he, :])

        out_valid = valid & np.isfinite(out).all(axis=-1)
        return out, out_valid

    def _temporal_fill_nans_np(self, arr: np.ndarray, long_gap_linear_threshold: int = 10) -> np.ndarray:
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
                if gap_len > long_gap_linear_threshold:
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

    def _temporal_fill_nans_neighbor_average_np(self, arr: np.ndarray) -> np.ndarray:
        """Fill NaN gaps with prev/next neighbor average per temporal channel.

        This mirrors the MediaPipe-style interpolation used in
        visualize_mediapipe_vs_dynhamr_pixel.py when fill_mode=neighbor_average.
        """
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

                vals = []
                if np.isfinite(before):
                    vals.append(before)
                if np.isfinite(after):
                    vals.append(after)

                if vals:
                    flat[i, j] = float(sum(vals) / len(vals))

        return flat.reshape(out.shape)

    def _temporal_resample_linear_np(self, arr: np.ndarray, new_len: int) -> np.ndarray:
        old_len = arr.shape[0]
        if old_len <= 1 or new_len <= 1 or old_len == new_len:
            return np.array(arr, copy=True)

        def _interp_1d_with_fallback(src_t: np.ndarray, values: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
            valid = np.isfinite(values)
            valid_count = int(valid.sum())
            if valid_count == 0:
                return np.full((len(dst_t),), np.nan, dtype=np.float32)
            if valid_count == 1:
                return np.full((len(dst_t),), float(values[valid][0]), dtype=np.float32)

            partial_t = src_t[valid]
            partial_v = values[valid]
            this_kind = "cubic" if valid_count > 3 else "quadratic" if valid_count > 2 else "linear"
            f = interp1d(partial_t, partial_v, kind=this_kind, bounds_error=False, fill_value=np.nan)
            interp_values = np.asarray(f(dst_t), dtype=np.float32)

            interp_values[dst_t < partial_t[0]] = partial_v[0]
            interp_values[dst_t > partial_t[-1]] = partial_v[-1]
            return interp_values

        src_t = np.arange(old_len, dtype=np.float32)
        dst_t = np.linspace(0.0, float(old_len - 1), num=new_len, dtype=np.float32)
        flat = arr.reshape(old_len, -1)
        out = np.full((new_len, flat.shape[1]), np.nan, dtype=np.float32)

        for c in range(flat.shape[1]):
            values = flat[:, c]
            out[:, c] = _interp_1d_with_fallback(src_t, values, dst_t)

        return out.reshape((new_len,) + arr.shape[1:])

    def _interpolate_pose_like_pose_format(self, pose: Pose, new_fps: int, kind: str = "cubic") -> Pose:
        data = pose.body.data
        confidence = pose.body.confidence

        frames = int(data.shape[0])
        if frames == 1:
            raise ValueError("Can't interpolate single frame")

        new_frames_count = int(round(frames * float(new_fps) / float(pose.body.fps)))
        steps = np.linspace(0.0, 1.0, frames)
        new_steps = np.linspace(0.0, 1.0, new_frames_count)

        transposed = np.ma.transpose(data, axes=(2, 1, 0, 3))
        masked_confidence = np.ma.array(confidence, mask=confidence == 0)
        confidence_exp = np.ma.expand_dims(masked_confidence.transpose((2, 1, 0)), axis=3)
        points = np.ma.concatenate([transposed, confidence_exp], axis=3)

        new_people = []
        for people in points:
            new_point_frames = []
            for frames_data in people:
                mask = frames_data.transpose()[0].mask
                partial_steps = np.ma.array(steps, mask=mask).compressed()

                if partial_steps.shape[0] == 0:
                    new_point_frames.append(np.zeros((new_frames_count, frames_data.shape[1]), dtype=np.float32))
                    continue

                partial_frames = frames_data.compressed().reshape(partial_steps.shape[0], frames_data.shape[1])

                if len(partial_steps) == 1:
                    interp_values = np.repeat(partial_frames, repeats=new_frames_count, axis=0)
                    new_point_frames.append(interp_values.astype(np.float32))
                    continue

                this_kind = (
                    kind
                    if len(partial_steps) > 3
                    else "quadratic"
                    if len(partial_steps) > 2 and kind == "cubic"
                    else "linear"
                )
                f = interp1d(partial_steps, partial_frames, axis=0, kind=this_kind)

                first_step = float(partial_steps[0])
                last_step = float(partial_steps[-1])
                if first_step == 0.0 and last_step == 1.0:
                    new_point_frames.append(np.asarray(f(new_steps), dtype=np.float32))
                    continue

                first_step_where = np.argwhere(new_steps >= first_step)
                first_step_index = int(first_step_where[0][0]) if len(first_step_where) > 0 else 0

                last_step_where = np.argwhere(new_steps > last_step)
                last_step_index = int(last_step_where[0][0]) if len(last_step_where) > 0 else len(new_steps)

                if first_step_index == last_step_index:
                    new_point_frames.append(np.zeros((len(new_steps), frames_data.shape[1]), dtype=np.float32))
                else:
                    frame_data = np.asarray(f(new_steps[first_step_index:last_step_index]), dtype=np.float32)
                    new_point_frames.append(
                        np.concatenate(
                            [
                                np.zeros((first_step_index, frames_data.shape[1]), dtype=np.float32),
                                frame_data,
                                np.zeros((len(new_steps) - last_step_index, frames_data.shape[1]), dtype=np.float32),
                            ]
                        )
                    )

            new_people.append(np.stack(new_point_frames, axis=0))

        new_data = np.stack(new_people, axis=0).transpose((2, 1, 0, 3))
        dimensions, new_confidence = np.split(new_data, [-1], axis=3)
        new_confidence = np.squeeze(new_confidence, axis=3)

        new_body = NumPyPoseBody(fps=new_fps, data=dimensions, confidence=new_confidence)
        return Pose(header=copy.deepcopy(pose.header), body=new_body)

    def _distance_batch_np(self, p1s: np.ndarray, p2s: np.ndarray) -> np.ndarray:
        squared = (p1s - p2s) ** 2
        summed = squared.sum(axis=-1)
        return summed ** 0.5

    def _compute_global_normalization_stats_from_pose(
        self,
        pose: Pose,
        p1_index: int,
        p2_index: int,
        scale_factor: float = 1.0,
    ):
        transposed = np.ma.transpose(pose.body.data, axes=(2, 1, 0, 3))
        p1s = transposed[p1_index]
        p2s = transposed[p2_index]

        center = ((p2s + p1s) / 2).mean(axis=(0, 1))
        mean_distance = self._distance_batch_np(p1s, p2s).mean()

        center_np = np.asarray(np.ma.filled(center, 0.0), dtype=np.float32)
        mean_distance_value = float(np.ma.filled(mean_distance, np.nan))
        if not np.isfinite(mean_distance_value) or abs(mean_distance_value) < 1e-8:
            mean_distance_value = 1.0

        scale = float(scale_factor) / mean_distance_value
        return center_np, scale

    def _extract_mp_global_center_scale_30(self, vid_id: str):
        if vid_id in self._dynhamr_2d_mp_shoulder_cache:
            return self._dynhamr_2d_mp_shoulder_cache[vid_id]

        mp_pose_path = self._dynhamr_2d_mp_pose_path_by_vid.get(vid_id, None)
        if not mp_pose_path:
            if not self._warned_dynhamr_2d_missing_mp_pose:
                logger.warning(
                    "[WARNING] DynHaMR 2D mode could not find MediaPipe pose for some videos. "
                    "Using wrist-based fallback center/scale for those samples."
                )
                self._warned_dynhamr_2d_missing_mp_pose = True
            self._dynhamr_2d_mp_shoulder_cache[vid_id] = (None, None)
            return None, None

        try:
            with open(mp_pose_path, "rb") as f:
                raw_mp_pose = Pose.read(f.read())

            pose_30 = self._interpolate_pose_like_pose_format(
                raw_mp_pose,
                int(round(self.dynhamr_target_fps)),
                kind="cubic",
            )
            if len(pose_30.header.components) >= 4:
                pose_30.header.components[2], pose_30.header.components[3] = (
                    pose_30.header.components[3],
                    pose_30.header.components[2],
                )

            pose_keep = pose_30.get_components(["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"])
            norm_info = pose_keep.header.normalization_info(
                p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
                p2=("POSE_LANDMARKS", "LEFT_SHOULDER"),
            )
            center, scale = self._compute_global_normalization_stats_from_pose(
                pose_keep,
                p1_index=norm_info.p1,
                p2_index=norm_info.p2,
            )
            self._dynhamr_2d_mp_shoulder_cache[vid_id] = (center, scale)
            return center, scale
        except Exception as exc:
            logger.warning(f"[WARNING] Could not extract MP global center/scale for DynHaMR 2D sample {vid_id}: {exc}")
            self._dynhamr_2d_mp_shoulder_cache[vid_id] = (None, None)
            return None, None

    def _project_world_to_pixel_np(self, points_world, cam_r, cam_t, cam_intrins):
        x_c = points_world @ cam_r.T + cam_t
        x = x_c[:, 0]
        y = x_c[:, 1]
        z = x_c[:, 2]

        fx, fy, cx, cy = float(cam_intrins[0]), float(cam_intrins[1]), float(cam_intrins[2]), float(cam_intrins[3])
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (np.abs(z) > 1e-8)

        u = np.full_like(x, np.nan, dtype=np.float32)
        v = np.full_like(y, np.nan, dtype=np.float32)
        u[valid] = (x[valid] / z[valid]) * fx + cx
        v[valid] = (y[valid] / z[valid]) * fy + cy
        return np.stack([u, v], axis=-1), valid

    def _sanitize_3d_joints_np(self, arr: np.ndarray, min_depth_m: float = 0.15):
        out = np.array(arr, copy=True, dtype=np.float32)
        bad = out[..., 2] < float(min_depth_m)
        out[bad] = np.nan
        bad_frames = np.where(np.any(bad, axis=1))[0].astype(np.int32)
        return out, bad_frames

    def _load_dynhamr_pixel_xy_25(self, dyn_dir: Union[Path, str], video_frames_25: Optional[int] = None):
        dyn_dir = Path(dyn_dir)
        left = np.load(dyn_dir / "joints_3d_left.npy").astype(np.float32)
        right = np.load(dyn_dir / "joints_3d_right.npy").astype(np.float32)
        cam_r = np.load(dyn_dir / "cam_R.npy").astype(np.float32)
        cam_t = np.load(dyn_dir / "cam_t.npy").astype(np.float32)
        cam_k = np.load(dyn_dir / "cam_intrins.npy").astype(np.float32)

        t = min(len(left), len(right), len(cam_r), len(cam_t), len(cam_k))

        left, bad_frames_l = self._sanitize_3d_joints_np(left[:t], min_depth_m=0.15)
        right, bad_frames_r = self._sanitize_3d_joints_np(right[:t], min_depth_m=0.15)
        n_bad_l = int(np.isnan(left[..., 0]).sum())
        n_bad_r = int(np.isnan(right[..., 0]).sum())

        if self.dynhamr_fill_mode == "neighbor_average":
            left = self._temporal_fill_nans_neighbor_average_np(left)
            right = self._temporal_fill_nans_neighbor_average_np(right)
        elif self.dynhamr_fill_mode == "hybrid":
            left = self._temporal_fill_nans_np(
                left,
                long_gap_linear_threshold=max(0, int(self.dynhamr_long_gap_linear_threshold)),
            )
            right = self._temporal_fill_nans_np(
                right,
                long_gap_linear_threshold=max(0, int(self.dynhamr_long_gap_linear_threshold)),
            )
        else:
            raise ValueError(
                f"Unknown dynhamr_fill_mode={self.dynhamr_fill_mode}. "
                "Expected one of: neighbor_average, hybrid"
            )
        cam_r = cam_r[:t]
        cam_t = cam_t[:t]
        cam_k = cam_k[:t]

        out_xy = np.full((t, 42, 2), np.nan, dtype=np.float32)
        out_valid = np.zeros((t, 42), dtype=bool)

        for i in range(t):
            l_xy, l_valid = self._project_world_to_pixel_np(left[i], cam_r[i], cam_t[i], cam_k[i])
            r_xy, r_valid = self._project_world_to_pixel_np(right[i], cam_r[i], cam_t[i], cam_k[i])
            out_xy[i, :21] = l_xy
            out_xy[i, 21:] = r_xy
            out_valid[i, :21] = l_valid
            out_valid[i, 21:] = r_valid

        seq_start_25, seq_end_25 = self._load_track_seq_interval(dyn_dir)
        if not self.dynhamr_2d_disable_padding:
            out_xy, out_valid, seq_start_25 = self._pad_dyn_sequence_to_video_length_np(
                out_xy,
                out_valid,
                video_frames_25=int(video_frames_25) if video_frames_25 is not None else 0,
                seq_start_25=seq_start_25,
                seq_end_25=seq_end_25,
            )
        if not self.dynhamr_2d_disable_savgol:
            out_xy, out_valid = self._smooth_2d_sequence_savgol_np(
                out_xy,
                out_valid,
                window_length=max(3, int(self.dynhamr_savgol_window)),
                polyorder=max(1, int(self.dynhamr_savgol_polyorder)),
            )

        if video_frames_25 is not None:
            if int(len(out_xy)) != int(video_frames_25) and not self._warned_dynhamr_2d_padding_length_mismatch:
                logger.warning(
                    "[WARNING] DynHaMR 2D padded length mismatch: expected video_frames_25=%d, got %d.",
                    int(video_frames_25), int(len(out_xy))
                )
                self._warned_dynhamr_2d_padding_length_mismatch = True
            if int(seq_start_25) != 0 and not self._warned_dynhamr_2d_nonzero_seq_start_after_padding:
                logger.warning(
                    "[WARNING] DynHaMR 2D seq_start after padding is not zero (%d). Check track_info/padding assumptions.",
                    int(seq_start_25)
                )
                self._warned_dynhamr_2d_nonzero_seq_start_after_padding = True

        return out_xy, out_valid

    def _build_dynhamr_2d_sample(self, vid_dir: Union[Path, str], vid_id: str, video_frames_25: Optional[int] = None):
        xy_25, _ = self._load_dynhamr_pixel_xy_25(vid_dir, video_frames_25=video_frames_25)
        center_30, scale_30 = (None, None)
        if not self.dynhamr_2d_disable_mp_shoulder_norm:
            center_30, scale_30 = self._extract_mp_global_center_scale_30(vid_id)

        src_fps = float(self.dynhamr_source_fps)
        tgt_fps = float(self.dynhamr_target_fps)
        ratio = tgt_fps / src_fps if src_fps > 0 else 1.0
        target_len = max(1, int(round(xy_25.shape[0] * ratio)))

        xy_30 = self._temporal_resample_linear_np(xy_25, target_len)

        if not np.isfinite(xy_30).all():
            xy_30 = np.nan_to_num(xy_30, nan=0.0)

        return torch.from_numpy(xy_30).float().unsqueeze(1)

    def _apply_temp_mp_shoulder_norm_xy(self, feats_xy, vid_id):
        mp_shoulder_dist = self._extract_mp_shoulder_distance(vid_id)

        left_wrist = feats_xy[:, :, 0, :]
        right_wrist = feats_xy[:, :, 21, :]
        wrist_center = (left_wrist + right_wrist) / 2.0

        mapped = (feats_xy - wrist_center[:, :, None, :]) / float(mp_shoulder_dist)

        if not self._warned_temp_mp_shoulder_norm_map:
            print(
                "[TEMP PATCH][REMOVE ME] DynHaMR MP-shoulder normalization mapping ACTIVE: "
                f"median_mp_shoulder_dist={mp_shoulder_dist:.6f}."
            )
            self._warned_temp_mp_shoulder_norm_map = True
        return mapped

    def _check_zero_values(self, feats_data, context=""):
        finite = torch.isfinite(feats_data)
        finite_count = int(finite.sum().item())
        if finite_count <= 0:
            return

        zeros_mask = (feats_data.abs() <= 1e-12) & finite
        zero_count = int(zeros_mask.sum().item())
        if zero_count <= 0:
            return

        zero_ratio = float(zero_count) / float(finite_count)
        msg = (
            "[WARNING] Zero values detected in DynHaMR tensor after interpolation/NaN handling"
            f" ({context}): zero_count={zero_count}, finite_count={finite_count}, zero_ratio={zero_ratio:.6f}."
        )
        if self.dynhamr_temp_fail_on_zeros:
            raise RuntimeError(msg)

        if not self._warned_temp_zero_values:
            logger.warning(msg)
            self._warned_temp_zero_values = True

    def _temporal_linear_resample_ignore_nans(self, hands_data, new_len):
        """
        Resample along time with cubic/quadratic/linear fallback while ignoring NaNs per channel.
        Channels that are all-NaN remain NaN.
        """
        original_len = hands_data.shape[0]
        if original_len <= 1 or new_len <= 1 or new_len == original_len:
            return hands_data

        device = hands_data.device
        data_np = hands_data.detach().cpu().numpy()
        joints = data_np.shape[2]
        dims = data_np.shape[3]

        flat = data_np.reshape(original_len, -1)
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
        return torch.from_numpy(out_np).to(device=device)

    def _fill_missing_temporally(self, hands_data, context=""):
        """
        Fill NaNs by temporal interpolation per joint/dimension channel.
        Remaining all-NaN channels are fallback-filled with 0.0.
        """
        if hands_data.shape[0] <= 1:
            return torch.nan_to_num(hands_data, nan=0.0)

        if torch.isnan(hands_data).any() and not self._warned_temp_dynhamr_missing_fill:
            print(
                "[TEMP PATCH][REMOVE ME] Filling DynHaMR missing values via temporal interpolation "
                f"({context}) before normalization."
            )
            self._warned_temp_dynhamr_missing_fill = True

        filled = self._temporal_linear_resample_ignore_nans(hands_data, hands_data.shape[0])
        if torch.isnan(filled).any():
            if not self._warned_temp_dynhamr_remaining_nans:
                print(
                    "[TEMP PATCH][REMOVE ME][WARNING] Some DynHaMR channels are all-NaN across time; "
                    "fallback-filling remaining NaNs with 0.0."
                )
                self._warned_temp_dynhamr_remaining_nans = True
            filled = torch.nan_to_num(filled, nan=0.0)
        return filled

    def _apply_temp_global_similarity_xy(self, feats_xy):
        """
        Temporary global similarity mapping to align DynHaMR XY with MediaPipe-like space.
        Uses one global translation and one global isotropic scale.
        """
        scale = float(self.dynhamr_temp_similarity_scale)
        tx = float(self.dynhamr_temp_translate_x)
        ty = float(self.dynhamr_temp_translate_y)

        mapped = feats_xy + feats_xy.new_tensor([tx, ty]).view(1, 1, 1, 2)
        mapped = mapped * scale

        if not self._warned_temp_similarity_map:
            print(
                "[TEMP PATCH][REMOVE ME] DynHaMR global similarity mapping ACTIVE: "
                f"translate=({tx:.6f}, {ty:.6f}), scale={scale:.6f}."
            )
            self._warned_temp_similarity_map = True
        return mapped

    def _apply_temp_per_video_similarity_xy(self, feats_xy):
        """
        Temporary per-video similarity mapping for DynHaMR XY.
        One scale and one translation are estimated from the current video:
        - scale: aligns median wrist distance to a target distance
        - translation: aligns median wrist midpoint to a target center
        """
        left_wrist = feats_xy[:, :, 0, :]
        right_wrist = feats_xy[:, :, 21, :]

        wrist_mid = (left_wrist + right_wrist) / 2.0
        wrist_valid = torch.isfinite(wrist_mid).all(dim=-1).squeeze(1)
        if wrist_valid.any():
            center_current = wrist_mid[wrist_valid].reshape(-1, 2).median(dim=0).values
        else:
            flat_xy = feats_xy.reshape(-1, 2)
            finite_rows = torch.isfinite(flat_xy).all(dim=-1)
            if finite_rows.any():
                center_current = flat_xy[finite_rows].mean(dim=0)
            else:
                center_current = feats_xy.new_tensor([0.0, 0.0])

        wrist_dist = torch.norm(right_wrist - left_wrist, dim=-1).squeeze(1)
        wrist_dist_valid = torch.isfinite(wrist_dist) & (wrist_dist > 1e-8)
        if wrist_dist_valid.any():
            median_wrist_dist = wrist_dist[wrist_dist_valid].median()
            target_wrist_dist = float(self.dynhamr_temp_per_video_target_wrist_distance)
            scale = float(target_wrist_dist / float(median_wrist_dist))
        else:
            scale = 1.0

        target_center = feats_xy.new_tensor([
            float(self.dynhamr_temp_per_video_target_center_x),
            float(self.dynhamr_temp_per_video_target_center_y),
        ])

        translation = target_center - center_current * scale
        mapped = feats_xy * scale + translation.view(1, 1, 1, 2)

        if not self._warned_temp_per_video_similarity_map:
            center_x = float(center_current[0].item())
            center_y = float(center_current[1].item())
            target_x = float(target_center[0].item())
            target_y = float(target_center[1].item())
            print(
                "[TEMP PATCH][REMOVE ME] DynHaMR per-video mapping ACTIVE: "
                f"scale={scale:.6f}, center=({center_x:.6f}, {center_y:.6f}), "
                f"target_center=({target_x:.6f}, {target_y:.6f}), "
                f"target_wrist_dist={self.dynhamr_temp_per_video_target_wrist_distance:.6f}."
            )
            self._warned_temp_per_video_similarity_map = True
        return mapped

    def _maybe_interpolate_dynhamr_fps(self, hands_3d, vid_id):
        """
        [TEMP PATCH][REMOVE ME]
        Optional deterministic temporal resampling for DynHaMR sequences,
        used to simulate a target fps (e.g., 25 -> 30) without re-extraction.
        """
        if not self.dynhamr_temporal_interpolation:
            return hands_3d

        src_fps = float(self.dynhamr_source_fps)
        tgt_fps = float(self.dynhamr_target_fps)
        if src_fps <= 0 or tgt_fps <= 0:
            if not self._warned_temp_dynhamr_interp_patch:
                print(
                    "[TEMP PATCH][REMOVE ME][WARNING] Invalid fps values for DynHaMR interpolation "
                    f"(source={src_fps}, target={tgt_fps}). Skipping interpolation."
                )
                self._warned_temp_dynhamr_interp_patch = True
            return hands_3d

        if abs(src_fps - tgt_fps) < 1e-8:
            return hands_3d

        original_len = hands_3d.shape[0]
        if original_len <= 1:
            return hands_3d

        ratio = tgt_fps / src_fps
        new_len = max(2, int(round(original_len * ratio)))
        if new_len == original_len:
            return hands_3d

        if torch.isnan(hands_3d).any() and not self._warned_temp_dynhamr_interp_nans:
            print(
                "[TEMP PATCH][REMOVE ME] NaN-aware DynHaMR temporal interpolation active: "
                "resampling with channel-wise interpolation over valid timestamps."
            )
            self._warned_temp_dynhamr_interp_nans = True

        hands_3d = self._temporal_linear_resample_ignore_nans(hands_3d, new_len)

        if not self._printed_temp_dynhamr_interp_example:
            print(
                "[TEMP PATCH][REMOVE ME] DynHaMR interpolation example: "
                f"vid_id={vid_id}, len {original_len} -> {new_len}, fps {src_fps} -> {tgt_fps}."
            )
            self._printed_temp_dynhamr_interp_example = True

        return hands_3d

    def __getitem__(self, index):
        _id = self.ids[index]
        feats_file = self.feats_files[index]
        offset = self.offsets[index]
        length = self.sizes[index]
        
        if SignFeatsType_TD(self.feats_type) == SignFeatsType_TD.dynhamr:
            feats_file_path = Path(feats_file)
            if not (feats_file_path.is_file() and feats_file_path.suffix == ".pose"):
                pose_candidate = Path(str(feats_file_path) + ".pose")
                if pose_candidate.is_file():
                    feats_file_path = pose_candidate
            if feats_file_path.is_file() and feats_file_path.suffix == ".pose":
                hands_3d = self._read_preprocessed_dynhamr_3d_pose(feats_file_path, _id)
            else:
                # DynHaMR: Load from numpy files and optionally project to camera coordinates
                import os

                # feats_file points to the video directory
                vid_dir = feats_file

                # Load 3D joints in world coordinates
                right_3d = np.load(os.path.join(vid_dir, "joints_3d_right.npy"))  # (T, 21, 3)
                left_3d = np.load(os.path.join(vid_dir, "joints_3d_left.npy"))   # (T, 21, 3)

                # Conditionally project to camera coordinates
                if self.to_camera_coordinates:
                    # Load camera parameters for projection
                    cam_R = np.load(os.path.join(vid_dir, "cam_R.npy"))  # (T, 3, 3)
                    cam_t = np.load(os.path.join(vid_dir, "cam_t.npy"))  # (T, 3)

                    # Match lengths before projection (same behavior as comparison script)
                    T = min(len(cam_R), len(left_3d), len(right_3d))
                    left_3d = left_3d[:T]
                    right_3d = right_3d[:T]
                    cam_R = cam_R[:T]
                    cam_t = cam_t[:T]
                    right_cam = np.zeros_like(right_3d)
                    left_cam = np.zeros_like(left_3d)

                    # Project to camera coordinates
                    for t in range(T):
                        if not np.isnan(right_3d[t]).any():
                            right_cam[t] = self.world_to_cam(right_3d[t], cam_R[t], cam_t[t])
                        else:
                            right_cam[t] = np.nan

                        if not np.isnan(left_3d[t]).any():
                            left_cam[t] = self.world_to_cam(left_3d[t], cam_R[t], cam_t[t])
                        else:
                            left_cam[t] = np.nan

                    # Concatenate left and right hands: (T, 42, 3)
                    hands_3d = np.concatenate([left_cam, right_cam], axis=1)  # (T, 42, 3)
                else:
                    # Use raw world coordinates without camera projection
                    T = min(len(left_3d), len(right_3d))
                    hands_3d = np.concatenate([left_3d[:T], right_3d[:T]], axis=1)  # (T, 42, 3)

                if self.to_camera_coordinates:
                    hands_3d, _ = self._sanitize_3d_joints_np(hands_3d, min_depth_m=0.15)

                if self.dynhamr_fill_mode == "neighbor_average":
                    hands_3d = self._temporal_fill_nans_neighbor_average_np(hands_3d)
                elif self.dynhamr_fill_mode == "hybrid":
                    hands_3d = self._temporal_fill_nans_np(
                        hands_3d,
                        long_gap_linear_threshold=max(0, int(self.dynhamr_long_gap_linear_threshold)),
                    )
                else:
                    raise ValueError(
                        f"Unknown dynhamr_fill_mode={self.dynhamr_fill_mode}. "
                        "Expected one of: neighbor_average, hybrid"
                    )

                seq_start_25, seq_end_25 = self._load_track_seq_interval(vid_dir)
                try:
                    target_video_frames_25 = int(length)
                except Exception:
                    target_video_frames_25 = 0

                hands_3d, seq_start_25 = self._pad_dyn_sequence_3d_to_video_length_np(
                    hands_3d,
                    video_frames_25=target_video_frames_25,
                    seq_start_25=seq_start_25,
                    seq_end_25=seq_end_25,
                )

                if target_video_frames_25 > 0:
                    if int(len(hands_3d)) != int(target_video_frames_25) and not self._warned_dynhamr_3d_padding_length_mismatch:
                        logger.warning(
                            "[WARNING] DynHaMR 3D padded length mismatch: expected video_frames_25=%d, got %d.",
                            int(target_video_frames_25), int(len(hands_3d))
                        )
                        self._warned_dynhamr_3d_padding_length_mismatch = True
                    if int(seq_start_25) != 0 and not self._warned_dynhamr_3d_nonzero_seq_start_after_padding:
                        logger.warning(
                            "[WARNING] DynHaMR 3D seq_start after padding is not zero (%d). Check track_info/padding assumptions.",
                            int(seq_start_25)
                        )
                        self._warned_dynhamr_3d_nonzero_seq_start_after_padding = True

                if not np.isfinite(hands_3d).all():
                    hands_3d = np.nan_to_num(hands_3d, nan=0.0)

                # Convert to torch and add channel dimension to match expected format: (T, 1, 42, 3)
                hands_3d = torch.from_numpy(hands_3d).float().unsqueeze(1)
                hands_3d = self._maybe_interpolate_dynhamr_fps(hands_3d, _id)

            # LAIA: commented out this normalization. 
            # Mirror DynHaMR 2D fallback behavior: per-frame wrist-center and wrist-distance normalization.
            # hands_3d = self.normalize_by_wrists(hands_3d, use_fixed_scale=False)
            
            # Create a pseudo-pose object to match the expected interface
            class PseudoPose:
                def __init__(self, data, vid_id):
                    self.body = type('obj', (object,), {'data': data})()
                    self.vid_id = vid_id
            
            pose = PseudoPose(hands_3d, _id)
            pose = self.postprocess(pose)

        elif SignFeatsType_TD(self.feats_type) == SignFeatsType_TD.dynhamr_2d:
            # Check if using preprocessed mode
            if self.use_preprocessed_dynhamr_2d and _id in self._preprocessed_dynhamr_2d_pose_path_by_vid:
                # Load directly from preprocessed .pose file
                pose_file = self._preprocessed_dynhamr_2d_pose_path_by_vid[_id]
                hands_2d = self._read_preprocessed_dynhamr_2d_pose(pose_file, _id)
            else:
                # Fall back to standard online preprocessing
                vid_dir = feats_file
                try:
                    target_video_frames_25 = int(length)
                except Exception:
                    target_video_frames_25 = None
                hands_2d = self._build_dynhamr_2d_sample(vid_dir, _id, video_frames_25=target_video_frames_25)

            class PseudoBody:
                def __init__(self, data):
                    self.data = data

                def select_frames(self, frames_list):
                    if not frames_list:
                        return self
                    frame_idx = torch.as_tensor(frames_list, dtype=torch.long, device=self.data.device)
                    self.data = torch.index_select(self.data, 0, frame_idx)
                    return self

            class PseudoPose:
                def __init__(self, data, vid_id):
                    self.body = PseudoBody(data)
                    self.vid_id = vid_id

            pose = PseudoPose(hands_2d, _id)

            if self.dynhamr_2d_select_sentence_span:
                offsets_30 = [int(round(o * 30.0 / 25.0)) for o in self.offsets[index]]
                current_sentence_sizes = self.sentence_sizes[index] if index < len(self.sentence_sizes) else None
                if current_sentence_sizes and len(current_sentence_sizes) == len(offsets_30):
                    sizes_30 = [int(round(l * 30.0 / 25.0)) for l in current_sentence_sizes]
                    frames_list = list(
                        range(
                            min(offsets_30),
                            max([off + size for off, size in zip(offsets_30, sizes_30)]),
                        )
                    )
                    frames_list = [fr for fr in frames_list if 0 <= fr < pose.body.data.shape[0]]
                    if frames_list:
                        pose.body = pose.body.select_frames(frames_list)
                    elif not self._warned_dynhamr_2d_empty_sentence_span:
                        logger.warning(
                            "[WARNING] DynHaMR 2D sentence-span selection produced empty frame list for %s. Using full sequence.",
                            _id,
                        )
                        self._warned_dynhamr_2d_empty_sentence_span = True
                elif not self._warned_dynhamr_2d_empty_sentence_span:
                    logger.warning(
                        "[WARNING] DynHaMR 2D sentence-span selection missing/invalid sentence sizes for %s. Using full sequence.",
                        _id,
                    )
                    self._warned_dynhamr_2d_empty_sentence_span = True

            pose = self.postprocess(pose)
            
        elif SignFeatsType_TD(self.feats_type) == SignFeatsType_TD.mediapipe_keypoints:
            with open(feats_file, "rb") as f:
                pose = Pose.read(f.read())
            
            if not pose.body.data.flags.writeable:
                pose.body.data = pose.body.data.copy() # We could also force this: pose.body.data.setflags(write=True)

            # SELECT SENTENCE: this here lets you select from start of first sentence to end of last sentence.
            #frames_list = []
            ## Select the frames corresponding to the sentences, given the offset and length
            #print("We are doing this to select the beginning of the first sentence until the end of the last one.")
            ##for offset_i, length_i in zip(offset, length):
            ##    frames_list.extend(range(offset_i, offset_i+length_i))
            ## TODO: with this we are taking the first timestamp until the last of each sentence.
            ##This offsets are at 25fps, we should convert them at 30fps beforehand:
            #offsets_25 = [int(round(o * 30 / 25)) for o in self.offsets[index]]
            #sizes_25 = [int(round(l * 30 / 25)) for l in self.sizes[index]]
            #frames_list = list(range(min(offsets_25), max([offset + size for offset, size in zip(offsets_25, sizes_25)]))) 
            ##frames_list = list(range(offset, offset+length)) #this is the case where there is only one sentence
            ## Fix to bypass some examples that are wrong, out of range.
            #frames_list = [fr for fr in frames_list if fr < pose.body.data.shape[0]]
            #pose.body = pose.body.select_frames(frames_list) #This is when we want to take from the first of the first sentence until the last of the last sentence
            
            # SELECT WHOLE_VIDEO: we don't have to select_frames, because all trames are already there
            pose = self.postprocess(pose)
        elif SignFeatsType_TD(self.feats_type) == SignFeatsType_TD.i3d or SignFeatsType_TD(self.feats_type) == SignFeatsType_TD.openpose:
            with open(feats_file, "rb") as f:
                pose = np.load(f)
            pose = self.postprocess(pose)

        # Pretty much the same but here the dataloader expects: 
        # return {"id": index, "h2s_id": fn, "source": feats}
        return {"id": index, "vid_id": _id, "source": pose}
    
    @staticmethod
    def list_avail_ids(self):
        return self.ids

    def __len__(self):
        return len(self.sizes)

    def center_and_scale(self, feats):
        """
        If the spatial relationships of keypoints relative to the image center are more important (e.g., in action recognition or pose estimation), 
        you can center the coordinates at the image center and scale by a fixed factor (e.g., the half-diagonal of the image)
        
        """
        # Compute the image center
        
        center_x = feats.header.dimensions.width / 2
        center_y = feats.header.dimensions.height / 2

        # Compute the scale (diagonal of the image)
        scale = (feats.header.dimensions.width**2 + feats.header.dimensions.height**2)**0.5
        # TODO: check if this scale is correct
        
        # Center the coordinates at the image center
        feats.body.data[..., 0] = (feats.body.data[..., 0] - center_x) / scale  # Normalize x-coordinates
        feats.body.data[..., 1] = (feats.body.data[..., 1] - center_y) / scale  # Normalize y-coordinates
        return feats
    
    def normalize_by_wrists(self, feats_data, use_fixed_scale=False, reference_scale=0.3):
        """
        Normalize 3D hand poses using wrist keypoints as reference (analogous to shoulder normalization).
        For MediaPipe hand landmarks, wrist is index 0 for each hand.
        feats_data: (T, 1, 42, 3) where first 21 joints = left hand, next 21 = right hand
        
        Args:
            use_fixed_scale: If True, use fixed reference scale instead of variable wrist distance
            reference_scale: Fixed scale in meters (default 0.3m = typical wrist separation distance)
        
        Returns: normalized features with wrist center as origin
        
        CRITICAL: use_fixed_scale=True preserves hand-to-hand spatial relationships!
        Variable scaling (False) makes hands appear larger when close together, destroying interaction info.
        """
        # Extract wrist positions (first joint of each hand)
        left_wrist = feats_data[:, :, 0:1, :]   # (T, 1, 1, 3)
        right_wrist = feats_data[:, :, 21:22, :]  # (T, 1, 1, 3)
        
        # Compute center between both wrists as origin
        wrist_center = (left_wrist + right_wrist) / 2.0  # (T, 1, 1, 3)
        
        # Center all joints around wrist center
        feats_centered = feats_data - wrist_center  # (T, 1, 42, 3)
        
        if use_fixed_scale:
            # Use fixed reference scale - preserves hand-to-hand distance relationships
            # This is CRITICAL for sign language: hands close together (e.g., "MEET") vs far apart ("SEPARATE")
            feats_normalized = feats_centered / reference_scale
        else:
            # Compute scale as distance between wrists (analogous to shoulder width)
            # WARNING: This makes hands appear magnified when close together!
            wrist_distance = torch.norm(right_wrist - left_wrist, dim=-1, keepdim=True)  # (T, 1, 1, 1)
            wrist_distance = torch.clamp(wrist_distance, min=1e-6)  # Avoid division by zero
            feats_normalized = feats_centered / wrist_distance  # (T, 1, 42, 3)
        
        return feats_normalized
    
    def augment_3d_poses(self, feats_data):
        """
        Apply optional 3D data augmentations for DynHaMR hand poses.
        Augmentations are intentionally separated and independently configurable,
        and only run when self.data_augmentation is True.
        """
        if not self.data_augmentation:
            return feats_data

        # Current policy: apply these 3D augmentations only in world coordinates.
        if self.to_camera_coordinates:
            return feats_data

        if self.aug3d_random_resample and torch.rand(1).item() < self.aug3d_resample_p:
            feats_data = self._aug3d_random_resample(feats_data, self.aug3d_resample_limit)

        if self.aug3d_horizontal_flip and torch.rand(1).item() < self.aug3d_horizontal_flip_p:
            feats_data = self._aug3d_horizontal_flip(feats_data)

        if self.aug3d_scale:
            scale = 1.0 + torch.empty(1, device=feats_data.device).uniform_(
                -self.aug3d_scale_limit, self.aug3d_scale_limit
            )
            feats_data = feats_data * scale

        if self.aug3d_shift:
            shift = torch.empty(3, device=feats_data.device).normal_(0.0, self.aug3d_shift_std)
            feats_data = feats_data + shift.view(1, 1, 1, 3)

        if self.aug3d_frame_noise:
            frame_mask = torch.rand(feats_data.shape[0], device=feats_data.device) < self.aug3d_frame_noise_ratio
            if frame_mask.any():
                noise = torch.randn_like(feats_data) * self.aug3d_frame_noise_std
                feats_data = torch.where(frame_mask.view(-1, 1, 1, 1), feats_data + noise, feats_data)

        if self.aug3d_feature_mask:
            joint_mask = torch.rand(feats_data.shape[2], device=feats_data.device) < self.aug3d_feature_mask_ratio
            if joint_mask.any():
                feats_data = feats_data.masked_fill(joint_mask.view(1, 1, -1, 1), 0.0)

        if self.aug3d_frame_mask:
            frame_mask = torch.rand(feats_data.shape[0], device=feats_data.device) < self.aug3d_frame_mask_ratio
            if frame_mask.all():
                frame_mask[torch.randint(0, feats_data.shape[0], (1,), device=feats_data.device)] = False
            if frame_mask.any():
                feats_data = feats_data.masked_fill(frame_mask.view(-1, 1, 1, 1), 0.0)
        
        return feats_data

    def _aug3d_horizontal_flip(self, feats_data):
        """
        Horizontal flip for world coordinates:
        - mirror x axis
        - swap left/right hand keypoint blocks (first 21 and second 21 joints)
        """
        feats_data = feats_data.clone()
        feats_data[..., 0] = -feats_data[..., 0]

        left = feats_data[:, :, :21, :].clone()
        right = feats_data[:, :, 21:, :].clone()
        feats_data[:, :, :21, :] = right
        feats_data[:, :, 21:, :] = left
        return feats_data

    def _aug3d_random_resample(self, feats_data, limit):
        """
        Random temporal resampling with a variable speed factor,
        keeping output sequence length unchanged.
        """
        T = feats_data.shape[0]
        if T <= 2:
            return feats_data

        scale = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * limit
        new_T = max(2, int(round(T * scale)))

        flat = feats_data.squeeze(1).reshape(T, -1).transpose(0, 1).unsqueeze(0)  # (1, C, T)
        resampled = F.interpolate(flat, size=new_T, mode="linear", align_corners=False)
        resampled = resampled.squeeze(0).transpose(0, 1).reshape(new_T, 1, feats_data.shape[2], feats_data.shape[3])

        if new_T > T:
            start = torch.randint(0, new_T - T + 1, (1,), device=feats_data.device).item()
            resampled = resampled[start:start + T]
        elif new_T < T:
            pad_len = T - new_T
            pad_frames = resampled[-1:].repeat(pad_len, 1, 1, 1)
            resampled = torch.cat([resampled, pad_frames], dim=0)

        return resampled
    
    def postprocess(self, feats):
        from fairseq.data.sign_language.utils import (
            select_keypoints_by_bodypart,
            select_keypoints_by_dimension,
        )
        
        if SignFeatsType_TD[self.feats_type] is SignFeatsType_TD.dynhamr:
            # DynHaMR: feats.body.data is already a torch tensor with shape (T, 1, 42, 3)
            feats_data = feats.body.data  # (T, 1, 42, 3)
            defer_layer_norm_after_map = (
                (self.dynhamr_temp_map_to_mediapipe_2d or self.dynhamr_temp_mp_shoulder_norm)
                and self.normalization == NormType_TD.layer_norm.name
            )
            
            # Handle NaN values
            if self.dynhamr_temp_interpolate_missing:
                feats_data = self._fill_missing_temporally(feats_data, context="postprocess")
            else:
                feats_data = torch.nan_to_num(feats_data, nan=0.0)

            self._check_zero_values(feats_data, context="after_nan_handling")

            # --- ADD THIS WARNING BLOCK ---
            # Check if an entire hand is strictly zero across all frames and dimensions
            left_hand_zeros = (feats_data[:, :, :21, :].abs().sum() == 0)
            right_hand_zeros = (feats_data[:, :, 21:, :].abs().sum() == 0)
            if left_hand_zeros:
                logger.warning("[WARNING] Entire LEFT hand is zero for this sequence!")
            if right_hand_zeros:
                logger.warning("[WARNING] Entire RIGHT hand is zero for this sequence!")
            # ------------------------------
            
            # Apply data augmentation BEFORE normalization (only during training)
            feats_data = self.augment_3d_poses(feats_data)
            # TODO: laia check if this was affecting during training, but I don't think it should...
            
            # Apply normalization based on config (DynHaMR-specific)
            if self.normalization == NormType_TD.none.name:
                # No normalization - use raw camera coordinates
                pass
            elif self.normalization == NormType_TD.wrist_fixed_scale.name:
                # Use wrist-based normalization with FIXED scale (preserves hand interactions)
                feats_data = self.normalize_by_wrists(feats_data, use_fixed_scale=True, reference_scale=0.3)
            elif self.normalization == NormType_TD.body.name:
                # Use wrist-based normalization with VARIABLE scale (legacy, not recommended for sign language)
                feats_data = self.normalize_by_wrists(feats_data, use_fixed_scale=False)
            elif self.normalization == NormType_TD.layer_norm.name:
                if not defer_layer_norm_after_map:
                    # Layer normalization: standardize each dimension (x, y, z) independently
                    seq_len, _, n_feats, n_dims = feats_data.shape  # (T, 1, 42, 3)
                    feats_split = feats_data.permute(3, 0, 1, 2)  # (3, T, 1, 42)
                    with torch.no_grad():
                        feats_norm_split = F.layer_norm(feats_split, feats_split.shape[1:])
                    feats_data = feats_norm_split.permute(1, 2, 3, 0)  # Back to (T, 1, 42, 3)
            elif self.normalization == NormType_TD.kp_wise.name:
                # Standardize each keypoint independently across time
                mean = feats_data.mean(dim=0, keepdim=True)  # (1, 1, 42, 3)
                std = feats_data.std(dim=0, keepdim=True) + 1e-8
                feats_data = (feats_data - mean) / std
            elif self.normalization == NormType_TD.global_z_norm.name:
                # 3D Metric Normalization: Global Z-Norm (Standardization)
                # Computed offline from the How2Sign training set (post-interpolation) -> slt_how2sign_wicv2023/examples/SL_topic_detection/scripts/extract_global_stats.py
                global_mean_3d = torch.tensor([0.027149, 0.119636, 1.212672], device=feats_data.device)
                global_std_3d  = torch.tensor([0.101370, 0.157092, 0.180285], device=feats_data.device)
                
                # Subtract the mean to center at 0, divide by std to achieve unit variance
                feats_data = (feats_data - global_mean_3d) / global_std_3d
            
            if self.dynhamr_temp_mp_shoulder_norm:
                feats_data = feats_data[..., :2]
                if not self._warned_temp_drop_z_patch:
                    print(
                        "[TEMP PATCH][REMOVE ME] DynHaMR temporary XY mode ACTIVE: discarding Z."
                    )
                    self._warned_temp_drop_z_patch = True

                vid_id = getattr(feats, "vid_id", "")
                feats_data = self._apply_temp_mp_shoulder_norm_xy(feats_data, vid_id)

                self._check_zero_values(feats_data, context="after_mp_shoulder_norm")

                if defer_layer_norm_after_map:
                    feats_split = feats_data.permute(3, 0, 1, 2)  # (2, T, 1, 42)
                    with torch.no_grad():
                        feats_norm_split = F.layer_norm(feats_split, feats_split.shape[1:])
                    feats_data = feats_norm_split.permute(1, 2, 3, 0)  # Back to (T, 1, 42, 2)

            elif self.dynhamr_temp_map_to_mediapipe_2d:
                feats_data = feats_data[..., :2]
                if not self._warned_temp_drop_z_patch:
                    print(
                        "[TEMP PATCH][REMOVE ME] DynHaMR temporary XY mode ACTIVE: discarding Z."
                    )
                    self._warned_temp_drop_z_patch = True
                if self.dynhamr_temp_per_video_map_to_mediapipe_2d:
                    feats_data = self._apply_temp_per_video_similarity_xy(feats_data)
                else:
                    feats_data = self._apply_temp_global_similarity_xy(feats_data)

                self._check_zero_values(feats_data, context="after_similarity_mapping")

                if defer_layer_norm_after_map:
                    feats_split = feats_data.permute(3, 0, 1, 2)  # (2, T, 1, 42)
                    with torch.no_grad():
                        feats_norm_split = F.layer_norm(feats_split, feats_split.shape[1:])
                    feats_data = feats_norm_split.permute(1, 2, 3, 0)  # Back to (T, 1, 42, 2)
            
            return feats_data

        elif SignFeatsType_TD[self.feats_type] is SignFeatsType_TD.dynhamr_2d:
            feats_data = feats.body.data  # (T, 1, 42, 2)
            if feats_data.shape[-1] != 2:
                raise RuntimeError(
                    f"DynHaMR 2D pipeline expected XY (last dim=2), got shape={tuple(feats_data.shape)}"
                )

            # Apply shoulder/wrist normalization first (before layer_norm/kp_wise)
            vid_id = getattr(feats, "vid_id", "")
            center_30, scale_30 = (None, None)
            if not self.dynhamr_2d_disable_mp_shoulder_norm:
                center_30, scale_30 = self._extract_mp_global_center_scale_30(vid_id)
            
            if center_30 is not None and scale_30 is not None:
                center_arr = np.asarray(center_30, dtype=np.float32).reshape(-1)
                if center_arr.shape[0] < 2:
                    raise RuntimeError(
                        f"Invalid global center for DynHaMR 2D sample {vid_id}: shape={center_arr.shape}"
                    )
                center_xy = center_arr[:2]
                feats_data = (feats_data - torch.from_numpy(center_xy).float().view(1, 1, -1)) * float(scale_30)
            else:
                # Fallback: wrist-based normalization (variable scale per frame)
                #print(f"feats_np.shape: {feats_np.shape}")
                feats_np = feats_data.cpu().numpy() if isinstance(feats_data, torch.Tensor) else feats_data
                left_wrist = feats_np[:, :, 0:1, :]
                right_wrist = feats_np[:, :, 21:22, :]
                center = (left_wrist + right_wrist) * 0.5
                dist = np.linalg.norm(right_wrist - left_wrist, axis=-1, keepdims=True).astype(np.float32)
                dist = np.clip(dist, 1e-6, None)
                feats_data = torch.from_numpy((feats_np - center) / dist).float().to(feats_data.device) if isinstance(feats_data, torch.Tensor) else (feats_np - center) / dist

            # Handle NaNs after normalization (same as old code)
            if not torch.isfinite(feats_data).all():
                if not self._warned_dynhamr_2d_remaining_nans:
                    logger.warning(
                        "[WARNING] DynHaMR 2D sample still has NaNs after shoulder/wrist normalization. "
                        "Replacing remaining NaNs with 0.0 as a safety fallback."
                    )
                    self._warned_dynhamr_2d_remaining_nans = True
                feats_data = torch.nan_to_num(feats_data, nan=0.0)

            if self.normalization == NormType_TD.none.name:
                pass
            elif self.normalization == NormType_TD.layer_norm.name:
                print(f"[DH] Shape: {feats_data.shape}, Mean: {feats_data.mean():.6f}, Std: {feats_data.std():.6f}, "
                      f"Min: {feats_data.min():.6f}, Max: {feats_data.max():.6f}, Contains NaN: {torch.isnan(feats_data).any()}")
                feats_split = feats_data.permute(3, 0, 1, 2)  # (2, T, 1, 42)
                with torch.no_grad():
                    feats_norm_split = F.layer_norm(feats_split, feats_split.shape[1:])
                feats_data = feats_norm_split.permute(1, 2, 3, 0)
            elif self.normalization == NormType_TD.kp_wise.name:
                mean = feats_data.mean(dim=0, keepdim=True)
                std = feats_data.std(dim=0, keepdim=True) + 1e-8
                feats_data = (feats_data - mean) / std

            return feats_data
            
        elif SignFeatsType_TD[self.feats_type] is SignFeatsType_TD.mediapipe_keypoints:
            # Filter out with the self.bodyparts that we need and feats, depending on this, 
            #feats, n_feats = select_keypoints_by_bodypart(feats, feats_type=self.feats_type, bodyparts=self.bodyparts) #self.bodyparts = ['upperbody', 'right_hand', 'left_hand'], then it is unchanged and n_feats=50
            #feats = select_keypoints_by_dimension(feats, self.feat_dims, feats_type=self.feats_type) #This doesn't do anything because we are selecting the 3 dimentions
            
            if self.data_augmentation:
                feats = feats.augment2d(rotation_std=0.0, shear_std=0.1, scale_std=0.2) #la rotation std fa algo raro, així que anem a posar-ho a zero.
                #feats = feats.flip()
                #feats = feats.interpolate(30, kind='cubic') #this is to match the famerate of 30
                # add a flip
            if len(feats.header.components) >=3 : #This means that we have all the keypoints
                if self.bodyparts == ["LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"] or self.bodyparts == ["lefthand", "righthand"]: #This usually means to select the hands.
                    feats = feats.get_components(
                        ["LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]
                    )# here we should pass the self.bodyparts
                    if self.feat_dims == [0,1]:
                        for i in range(len(feats.header.components)):
                            feats.header.components[i].format = "XYC"
                        feats.body.data = feats.body.data[:, :, :, :2] #We are removing the "depth" dimension
                elif self.bodyparts == ["UPPER_BODY","LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]: #Here we need to select only the upper body.
                    upper_body_points = feats.header.components[0].points[:25] #This will discard the points 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX' and keeps waist up
                    feats = feats.get_components(
                        components=["POSE_LANDMARKS","LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
                        points={"POSE_LANDMARKS": upper_body_points}
                    )
                    if self.feat_dims == [0,1]:
                        for i in range(len(feats.header.components)):
                            feats.header.components[i].format = "XYC"
                        feats.body.data = feats.body.data[:, :, :, :2] #We are removing the "depth" dimension
                else:
                    import mediapipe as mp
                    mp_holistic = mp.solutions.holistic
                    FACEMESH_CONTOURS_POINTS = [
                        str(p) for p in sorted(set([p for p_tup in list(mp_holistic.FACEMESH_CONTOURS) for p in p_tup]))
                    ]
                    #This is the same as we are doing above. But I like the implementation on top better. Since it is simpler.
                    POSE_RM = ['LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
                            'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
                    POSE_POINTS = [kp.name for kp in mp_holistic.PoseLandmark if kp.name not in POSE_RM]
                    
                    feats = feats.get_components(
                        ["FACE_LANDMARKS", "POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
                        {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS, "POSE_LANDMARKS": POSE_POINTS}
                    ) #This is to get the contours of the faces
            elif len(feats.header.components) == 2: #This means that we only have the hands.
                # TODO: implement this for the egosign dataset
                # but we already have the hands, so no need to select
                if self.feat_dims == [0,1]:
                    feats.header.components[0].format = "XYC" #This is for when we only have the hands in the .pose file
                    feats.header.components[1].format = "XYC"
                    feats.body.data = feats.body.data[:, :, :, :2] #We are removing the "depth" dimension
            
            #First we need to divide the "pixel" value by the width and height of the image: if we have pixel values, we need to devide by the height and width
            #feats.body.data[..., 0] = feats.body.data[..., 0] #/ feats.header.dimensions.width #Since we are normalizing by shoulder, we do not diviide here.
            #feats.body.data[..., 1] = feats.body.data[..., 1] #/ feats.header.dimensions.height
            
            if self.normalization == NormType_TD.body:
                if "POSE_LANDMARKS" in feats.header.components[:].name:
                    # This we can only do it if feats.header.components has an element named "POSE_LANDMARKS"
                    normalize_info = feats.header.normalization_info(
                        p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
                        p2=("POSE_LANDMARKS", "LEFT_SHOULDER")
                    )
                    feats.normalize(normalize_info)
                else:
                    raise NotImplementedError(f"POSE_LANDMARKS not found in the components of the header, so we cannot normalize for the body")
            elif self.normalization == NormType_TD.kp_wise:
                #This is actually standardization, it transforms the data to have zero mean and unit variance, for each of the features.
                mean, std = feats.normalize_distribution(axis=(0, 1))
            elif self.normalization == NormType_TD.global_xyz:
                #This we can only do it if we have z
                mean, std = feats.normalize_distribution(axis=(0, 1, 2))
            elif self.normalization == NormType_TD.layer_norm.name or self.normalization == NormType_TD.body.name: # TODO: Check why here we only have this string ? 
                num_components = len(feats.header.components)
                seq_len, _, n_feats, n_dims = feats.body.data.shape
                n_feats = n_feats // num_components
                print(f"[MP] Shape: {feats.body.data.shape}, Mean: {feats.body.data.mean():.6f}, Std: {feats.body.data.std():.6f}, "
                      f"Min: {feats.body.data.min():.6f}, Max: {feats.body.data.max():.6f}")
                    
                # for layer normalization, we want to normalize each feature within the same sample independently. 
                feats_split = feats.body.data.transpose(3, 0, 1, 2) # to have the x and y dimensions in the first axis (from (3628, 1, 42, 2) to (2, 3628, 1, 42))
                with torch.no_grad():
                    feats_norm_split = F.layer_norm(torch.from_numpy(feats_split), feats_split.shape[1:]) #to do layer norm independently for each dimension , it does it into dimension (3628, 1, 42)
                feats.body.data = np.ma.MaskedArray(feats_norm_split.numpy().transpose(1, 2, 3, 0), feats.body.data.mask)
                
            elif self.normalization == NormType_TD.center_and_scale.name:
                feats = self.center_and_scale(feats)
            else:
                pass
            
            feats = feats.torch()
            
        elif SignFeatsType_TD[self.feats_type] in [SignFeatsType_TD.rotational, SignFeatsType_TD.mediapipe_rotational]:
            feats_split = feats.reshape(-1, 48, 6).permute(2, 0, 1)
            with torch.no_grad():
                feats_norm_split = F.layer_norm(feats_split, feats_split.shape[1:])
            feats = feats_norm_split.permute(1, 2, 0).reshape(-1, 48 * 6).contiguous()
        elif (SignFeatsType_TD[self.feats_type] is SignFeatsType_TD.i3d or
              SignFeatsType_TD[self.feats_type] is SignFeatsType_TD.CNN2d or
              SignFeatsType_TD[self.feats_type] is SignFeatsType_TD.video or
              SignFeatsType_TD[self.feats_type] is SignFeatsType_TD.spot_align_albert or
              SignFeatsType_TD[self.feats_type] is SignFeatsType_TD.mouthings_albert or
              SignFeatsType_TD[self.feats_type] is SignFeatsType_TD.text_albert):
            with torch.no_grad():
                feats = F.layer_norm(feats.float(), feats.shape)
        elif SignFeatsType_TD[self.feats_type] in [SignFeatsType_TD.text, SignFeatsType_TD.spot_align, SignFeatsType_TD.mouthings]:
            pass
        else:
            raise NotImplementedError(f'Using {self.feats_type} which is not SignFeatsType_TD.i3d'
                                      ' nor SignFeatsType_TD.spot_align_albert'
                                      ' nor SignFeatsType_TD.mouthings_albert'
                                      ' nor SignFeatsType_TD.keypoints nor SignFeatsType_TD.mediapipe_keypoints'
                                      ' nor SignFeatsType_TD.rotational nor SignFeatsType_TD.mediapipe_rotational'
                                      ' nor SignFeatsType_TD.2dCNN nor SignFeatsType_TD.video'
                                      ' nor SignFeatsType_TD.text nor SignFeatsType_TD.spot_align'
                                      ' nor SignFeatsType_TD.text nor SignFeatsType_TD.mouthings'
                                      )
        return feats

    def _maybe_dump_collated_batch(self, src_tokens, src_lengths, vid_ids, ids, skipped_all_padding=0):
        if not self.debug_dump_enabled:
            return

        feats_type_value = self.feats_type
        if hasattr(feats_type_value, "value"):
            feats_type_str = str(feats_type_value.value)
        elif hasattr(feats_type_value, "name"):
            feats_type_str = str(feats_type_value.name)
        else:
            feats_type_str = str(feats_type_value)

        batch_idx = self._debug_dump_batch_idx
        self._debug_dump_batch_idx += 1

        if batch_idx % self.debug_dump_every_n != 0:
            return

        if self.debug_dump_max_batches > 0 and self._debug_dump_saved_batches >= self.debug_dump_max_batches:
            return

        filename = (
            f"{feats_type_str}_inst{self._debug_dump_instance_id:03d}_n{len(self.ids)}"
            f"_pid{os.getpid()}_batch{batch_idx:06d}.pt"
        )
        out_path = os.path.join(self.debug_dump_dir, filename)

        payload = {
            "feats_type": feats_type_str,
            "batch_idx": batch_idx,
            "pid": os.getpid(),
            "dump_instance_id": int(self._debug_dump_instance_id),
            "dataset_num_samples": int(len(self.ids)),
            "skipped_all_padding": int(skipped_all_padding),
            "vid_ids": list(vid_ids),
            "ids": list(ids),
            "src_tokens": src_tokens.detach().cpu(),
            "src_lengths": src_lengths.detach().cpu(),
            "shape": tuple(src_tokens.shape),
        }
        torch.save(payload, out_path)
        self._debug_dump_saved_batches += 1

        if self._debug_dump_saved_batches <= 5:
            logger.warning(
                f"[DEBUG DUMP] Saved batch #{batch_idx} ({feats_type_str}) to {out_path}"
            )
    
    '''
    The other postprocess that might be useful:
    def postprocess(self, pose):
        
        if SignFeatsType[self.feats_type] in [SignFeatsType.mediapipe, SignFeatsType.openpose]:
            import mediapipe as mp
            mp_holistic = mp.solutions.holistic
            FACEMESH_CONTOURS_POINTS = [
                str(p) for p in sorted(set([p for p_tup in list(mp_holistic.FACEMESH_CONTOURS) for p in p_tup]))
            ]
            POSE_RM = ['LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
                    'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
            POSE_POINTS = [kp.name for kp in mp_holistic.PoseLandmark if kp.name not in POSE_RM]
            pose = pose.get_components(
                ["FACE_LANDMARKS", "POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
                {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS, "POSE_LANDMARKS": POSE_POINTS}
            )

            if self.normalization == NormType.body:
                normalize_info = pose.header.normalization_info(
                    p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
                    p2=("POSE_LANDMARKS", "LEFT_SHOULDER")
                )
                pose.normalize(normalize_info)
            elif self.normalization == NormType.kp_wise:
                mean, std = pose.normalize_distribution(axis=(0, 1))
            elif self.normalization == NormType.global_xyz:
                mean, std = pose.normalize_distribution(axis=(0, 1, 2))
            else:
                pass
            if self.data_augmentation:
                pose = pose.augment2d()
            pose = pose.torch()
                
        elif (SignFeatsType[self.feats_type] in [SignFeatsType.i3d, SignFeatsType.CNN2d]):
            pose = torch.from_numpy(pose)
        else:
            raise NotImplementedError(f'Using {self.feats_type} which is not SignFeatsType.i3d'
                                      ' nor SignFeatsType.mediapipe nor SignFeatsType.openpose'
                                      ' nor SignFeatsType.2dCNN '
                                      )
        return pose
    '''

    def collater(self, samples):
        if self.feats_type == SignFeatsType_TD.mediapipe_keypoints.name:
            #import pdb; pdb.set_trace() # Here we are limiting the sizes to the length of the .poses.
            sizes = [s["source"].body.data.shape[0] for s in samples]
            collated_sources = []
        elif self.feats_type == SignFeatsType_TD.dynhamr.name or self.feats_type == SignFeatsType_TD.dynhamr_2d.name:
            # DynHaMR: source is already a torch tensor
            sizes = [s["source"].shape[0] for s in samples]
            collated_sources = []
        elif self.feats_type in ['video']:
            collated_sources = samples[0].new_zeros(len(samples), max_length, samples[0].shape[-1])
            sizes = [len(s["source"]) for s in samples]
        else:
            collated_sources = samples[0].new_zeros(len(samples), max_length, *samples[0].shape[-3:])
            sizes = [len(s["source"]) for s in samples]
        
        #Check if this is ordered, and if not, we need to order it. Because the LSTM is giving us errors...
        sorted_indices = sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True)
        #Then apply this to the samples and sizes
        samples = [samples[i] for i in sorted_indices]
        sizes = [sizes[i] for i in sorted_indices]

        max_length = sizes[0] #We can do this since it is sorted
        ids = []
        vid_ids = []
        padding_masks = []
        sizes=[]
        i=0
        skipped_all_padding = 0
        skipped_vid_ids = []
        for sample in samples:
            feat = sample["source"]
            if self.feats_type == SignFeatsType_TD.mediapipe_keypoints.name:
                if feat.body.data.shape[1] > 1:
                    logger.warning(f"More than one person in frame, keeping just the first one")
                
                feat.body.data = feat.body.data[:, 0]
                
                padding_mask = (~feat.body.data.mask).sum((1,2)) > 0 #we should expect to see all falses here, if not, something happened there and the hands were not detected
                if padding_mask.any().item():  # This is a slightly cleaner way to write: not (~padding_mask).all().item()
                    num_padded = padding_mask.sum().item()
                    padded_indices = torch.where(padding_mask)[0].tolist()
                    logger.warning(
                        f"Video {sample['vid_id']} (ID: {sample['id']}) has {num_padded} empty frame(s) padded internally "
                        f"at indices: {padded_indices}"
                    )
                if padding_mask.all():
                    skipped_all_padding += 1
                    skipped_vid_ids.append(sample["vid_id"])
                    continue
                diff_length = max_length - len(padding_mask)
                sizes.append(sample["source"].body.data.shape[0])
                ids.append(sample["id"])
                vid_ids.append(sample["vid_id"])
                padding_masks.append(
                    F.pad(padding_mask, (0, diff_length), value=True)
                )
                collated_sources.append(
                    F.pad(feat.body.data.data, (0, 0, 0, 0, 0, diff_length), value=0.0)
                )
            elif self.feats_type == SignFeatsType_TD.dynhamr.name or self.feats_type == SignFeatsType_TD.dynhamr_2d.name:
                # DynHaMR: feat is already a torch tensor (T, 1, 42, 3)
                if feat.shape[1] > 1:
                    logger.warning(f"More than one person in frame, keeping just the first one")
                    feat = feat[:, 0:1, :, :]
                
                # Remove the person dimension: (T, 1, 42, 3) -> (T, 42, 3)
                feat = feat[:, 0, :, :]
                
                # Create padding mask (all frames are valid unless they're all zeros)
                padding_mask = ~(feat.abs().sum(dim=(1, 2)) > 0)  # (T,)
                if padding_mask.any().item():  # This is a slightly cleaner way to write: not (~padding_mask).all().item()
                    num_padded = padding_mask.sum().item()
                    padded_indices = torch.where(padding_mask)[0].tolist()
                    logger.warning(
                        f"Video {sample['vid_id']} (ID: {sample['id']}) has {num_padded} empty frame(s) padded internally "
                        f"at indices: {padded_indices}"
                    )
                    
                if padding_mask.all():
                    skipped_all_padding += 1
                    skipped_vid_ids.append(sample["vid_id"])
                    continue
                
                diff_length = max_length - len(padding_mask)
                sizes.append(sample["source"].shape[0])
                ids.append(sample["id"])
                vid_ids.append(sample["vid_id"])
                padding_masks.append(
                    F.pad(padding_mask, (0, diff_length), value=True)
                )
                # Pad the features: (T, 42, D) -> (max_length, 42, D)
                collated_sources.append(
                    F.pad(feat, (0, 0, 0, 0, 0, diff_length), value=0.0)
                )
            else:
                sizes.append(len(sample["source"]))
                ids.append(sample["id"])
                vid_ids.append(sample["vid_id"])
                diff = sample["source"].shape[0] - max_length
                if self.feats_type not in ['video']:
                    collated_sources[i] = torch.cat(
                        [feat, feat.new_full((-diff, feat.shape[-1]), 0.0)]
                    )
                    i+=1
                else:
                    collated_sources[i] = torch.cat(
                        [feat, feat.new_full((-diff, *feat.shape[-3:]), 0.0)]
                    )
                    i+=1
        
        src_tokens= torch.stack(collated_sources).float() 
        src_lengths=torch.Tensor(sizes)
        if skipped_all_padding > 0:
            logger.warning(
                f"[COLLATER] Skipped {skipped_all_padding} sample(s) due to all-padding frames. "
                f"Examples: {skipped_vid_ids[:10]}"
            )
        self._maybe_dump_collated_batch(
            src_tokens,
            src_lengths,
            vid_ids,
            ids,
            skipped_all_padding=skipped_all_padding,
        )
        #if isinstance(src_tokens, torch.Tensor):
            #print(f"\n[Collater {self.feats_type}]")
            #print(f"  Shape: {src_tokens.shape} (should be [B, T, D] or similar)")
            #print(f"  Mean: {src_tokens.mean():.6f}, Std: {src_tokens.std():.6f}")
            #print(f"  Min: {src_tokens.min():.6f}, Max: {src_tokens.max():.6f}")
            #print(f"  Contains NaN: {torch.isnan(src_tokens).any()}")
            #print(f"  src_lengths: {src_lengths}")
        if self.feats_type == SignFeatsType_TD.mediapipe_keypoints.name:
            #padding_masks = torch.stack(padding_masks)
            return {
                'vid_id': vid_ids,
                'id': torch.LongTensor(ids),
                'net_input': {
                    'src_tokens': torch.stack(collated_sources).float(), 
                    'src_lengths': torch.Tensor(sizes)  # FIXME: If you use buckets
                }
            }
        elif self.feats_type == SignFeatsType_TD.dynhamr.name or self.feats_type == SignFeatsType_TD.dynhamr_2d.name:
            # DynHaMR: sources are already torch tensors
            return {
                'vid_id': vid_ids,
                'id': torch.LongTensor(ids),
                'net_input': {
                    'src_tokens': torch.stack(collated_sources).float(),  # (batch, max_length, 42, 3)
                    'src_lengths': torch.Tensor(sizes)
                }
            }
        else:
            return {
                'vid_id': vid_ids,
                'id': torch.LongTensor(ids),
                'net_input': {
                    'src_tokens': collated_sources, 
                    'src_lengths': torch.Tensor(sizes)  # FIXME: If you use buckets
                }
            }

    def num_tokens(self, index):
        return self.size(index) 

    def size(self, index): #Probably this is done to filter, I don't know where else
        # TODO: I think the length here
        #this is to have the size that is the beginning and end of sentence
        # SELECT SENTENCE: this here lets you select from start of first sentence to end of last sentence. 
        #total_length = max([offset + size for offset, size in zip(self.offsets[index], self.sizes[index])]) - min(self.offsets[index]) 
        #return total_length
        
        if self.dynhamr_2d_select_sentence_span:
            sentence_sizes = self.sentence_sizes[index] if index < len(self.sentence_sizes) else None
            if sentence_sizes and len(sentence_sizes) == len(self.offsets[index]):
                offsets_30 = [int(round(o * 30.0 / 25.0)) for o in self.offsets[index]]
                sizes_30 = [int(round(l * 30.0 / 25.0)) for l in sentence_sizes]
                return max([off + size for off, size in zip(offsets_30, sizes_30)]) - min(offsets_30)

        #return sum(self.sizes[index])
        return self.sizes[index] # This is to load the whole video, since in self.sizes we have the total length of the .pose

    def ordered_indices(self):
        if self.shuffle:
            #total_sizes = [sum(s) for s in self.sizes]
            # This order here is in case we are not filtering with sentence, and we are taking the full video. 
            total_sizes = self.sizes

            if self.dynhamr_2d_select_sentence_span:
                total_sizes = [self.size(index) for index in range(len(self.sizes))]
            
            # SELECT SENTENCE: this here lets you select from start of first sentence to end of last sentence. 
            #total_sizes = [max([offset + size for offset, size in zip(self.offsets[index], self.sizes[index])]) - min(self.offsets[index])for index in range(len(self.sizes))]

            order = np.lexsort(
                [np.random.permutation(len(self)), np.array(total_sizes)]
            )
            return order[::-1] #reverse order so that it is descending order
        else:
            return np.arange(len(self))