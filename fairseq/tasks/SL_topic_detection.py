# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
from pathlib import Path
from argparse import Namespace
import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from fairseq.data import Dictionary
from fairseq.data import AddTargetDataset
from fairseq.data import LanguagePairDataset

from dataclasses import dataclass
from dataclasses import field
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.constants import ChoiceEnum
from typing import Optional
from omegaconf import MISSING, II

from fairseq.data.sign_language import (
    SignFeatsType_TD,
    SLTopicDetectionDataset,
    SLTopicDetectionDatasetOld,
    NormType_TD,
)
from fairseq.data.text_compressor import TextCompressor
from fairseq.data.text_compressor import TextCompressionLevel
from fairseq.tasks import FairseqTask
from fairseq.tasks import register_task
from fairseq import metrics


logger = logging.getLogger(__name__)


@dataclass
class SLTopicDetectionConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    dict_path: str = field(
        default = MISSING,
        metadata={'help': 'Path to dictionary mapping category number to category name'},
    )
    modeling_task: str = field(
        default = 'classification',
        metadata={'help': 'Modeling task.'},
    )
    num_labels: str = field(
        default=10, metadata={'help': 'Number of labelswhen modeling_task is classification'}
    )
    max_source_positions: Optional[int] = field(
        default=5500, metadata={"help": "max number of frames in the source sequence"}
    )
    min_source_positions: Optional[int] = field(
        default=150, metadata={"help": "min number of frames in the source sequence"}
    )
    max_target_positions: Optional[int] = field(
        default=512, metadata={"help": "max number of tokens in the target sequence"}
    )
    normalization: ChoiceEnum([x.name for x in NormType_TD]) = field(
        default=NormType_TD.body.name,
        metadata={"help": "select the type of normalization to apply: none (raw coords), wrist_fixed_scale (recommended for DynHaMR), body (variable scale - legacy), layer_norm, kp_wise"},
    )
    normalization_type: Optional[str] = field(
        default=None,
        metadata={"help": "alias for normalization parameter - if provided, overrides normalization field"},
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    #body_parts: str = field(
    #    default = "face,upperbody,lefthand,righthand",
    #    metadata={"help": "Select the keypoints that you want to use. Options: 'face','upperbody','lowerbody','lefthand', 'righthand'"},
    #)
    body_parts: str = field(
        default = "LEFT_HAND_LANDMARKS,RIGHT_HAND_LANDMARKS",
        metadata={"help": "Select the keypoints that you want to use. Options: 'face','upperbody','lowerbody','LEFT_HAND_LANDMARKS', 'RIGHT_HAND_LANDMARKS'"},
    )
    feat_dims: str = field(
        default = "0,1,2",
        metadata={"help": "Select the keypoints dimensions that you want to use. Options: 0, 1, 2, 3"},
    )
    shuffle_dataset: bool = field(
        default=True,
        metadata={"help": "set True to shuffle the dataset between epochs"},
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={"help": "number of buckets"},
    )
    text_compression_level: ChoiceEnum([x.name for x in TextCompressionLevel]) = field(
        default="none",
        metadata={
            "help": "compression level for texts (e.g. audio filenames, "
            "target texts): none/low/high (default: none). "
        },
    )
    feats_type: ChoiceEnum([x.name for x in SignFeatsType_TD]) = field(
        default="keypoints",
        metadata={
            "help": (
                "type of features for the sign input data:"
                "keypoints/mediapipe_keypoints/dynhamr/dynhamr_2d/rotational/mediapipe_rotational/i3d/spot_align/spot_align_albert/mouthings/mouthings_albert/text/text_albert (default: keypoints)."
            )
        },
    )
    eval_accuracy: bool = field(
        default=True,
        metadata={'help': 'set to True to evaluate validation accuracy'},
    )
    dataset: str = field(
        default="",
        metadata={'help': 'Dataset to use'},
    )
    data_augmentation: bool = field(
        default=False,
        metadata={'help': 'set to True to add data augmentation'},
    )
    to_camera_coordinates: bool = field(
        default=True,
        metadata={'help': 'if True, project 3D joints from world to camera coordinates (for DynHaMR); if False, use raw world coordinates'},
    )
    dynhamr_temporal_interpolation: bool = field(
        default=False,
        metadata={'help': '[TEMP PATCH][REMOVE ME] if True, temporally resample DynHaMR sequences from source fps to target fps at load time'},
    )
    dynhamr_source_fps: float = field(
        default=25.0,
        metadata={'help': '[TEMP PATCH][REMOVE ME] assumed original DynHaMR fps before optional temporal interpolation'},
    )
    dynhamr_target_fps: float = field(
        default=30.0,
        metadata={'help': '[TEMP PATCH][REMOVE ME] target fps for optional DynHaMR temporal interpolation'},
    )
    dynhamr_fill_mode: str = field(
        default="neighbor_average",
        metadata={'help': 'DynHaMR NaN fill mode before 25->30 interpolation (neighbor_average|hybrid)'},
    )
    dynhamr_long_gap_linear_threshold: int = field(
        default=10,
        metadata={'help': 'When dynhamr_fill_mode=hybrid, use linear interpolation for NaN gaps longer than this threshold'},
    )
    dynhamr_temp_map_to_mediapipe_2d: bool = field(
        default=False,
        metadata={'help': '[TEMP PATCH][REMOVE ME] if True, use temporary DynHaMR 2D mapping path (XY projection + missing-data interpolation + optional global similarity transform)'},
    )
    dynhamr_temp_interpolate_missing: bool = field(
        default=False,
        metadata={'help': '[TEMP PATCH][REMOVE ME] if True, fill DynHaMR NaNs via temporal interpolation per channel before normalization'},
    )
    dynhamr_temp_similarity_scale: float = field(
        default=1.0,
        metadata={'help': '[TEMP PATCH][REMOVE ME] global isotropic scale for temporary DynHaMR 2D mapping (1.0 keeps original distances)'},
    )
    dynhamr_temp_translate_x: float = field(
        default=0.0,
        metadata={'help': '[TEMP PATCH][REMOVE ME] global X translation for temporary DynHaMR 2D mapping'},
    )
    dynhamr_temp_translate_y: float = field(
        default=0.0,
        metadata={'help': '[TEMP PATCH][REMOVE ME] global Y translation for temporary DynHaMR 2D mapping'},
    )
    dynhamr_temp_per_video_map_to_mediapipe_2d: bool = field(
        default=False,
        metadata={'help': '[TEMP PATCH][REMOVE ME] if True, estimate one per-video scale+translation in XY using wrist statistics (overrides global similarity params when both are set)'},
    )
    dynhamr_temp_per_video_target_center_x: float = field(
        default=-0.13,
        metadata={'help': '[TEMP PATCH][REMOVE ME] target X center for per-video DynHaMR XY mapping'},
    )
    dynhamr_temp_per_video_target_center_y: float = field(
        default=0.43,
        metadata={'help': '[TEMP PATCH][REMOVE ME] target Y center for per-video DynHaMR XY mapping'},
    )
    dynhamr_temp_per_video_target_wrist_distance: float = field(
        default=0.30,
        metadata={'help': '[TEMP PATCH][REMOVE ME] target median wrist distance for per-video DynHaMR XY mapping'},
    )
    dynhamr_2d_mp_pose_manifest_file: Optional[str] = field(
        default=None,
        metadata={'help': 'Manifest TSV (id_vid, signs_file) for MediaPipe pose files used to compute shoulder center/scale in DynHaMR 2D pipeline'},
    )
    aug3d_random_resample: bool = field(
        default=False,
        metadata={'help': 'enable random temporal resampling augmentation for DynHaMR 3D data'},
    )
    aug3d_resample_p: float = field(
        default=0.5,
        metadata={'help': 'probability of applying random temporal resampling when enabled'},
    )
    aug3d_resample_limit: float = field(
        default=0.2,
        metadata={'help': 'temporal scale perturbation limit for resampling (e.g., 0.2 -> scale in [0.8, 1.2])'},
    )
    aug3d_frame_noise: bool = field(
        default=False,
        metadata={'help': 'enable random frame-level Gaussian noise for DynHaMR 3D data'},
    )
    aug3d_frame_noise_ratio: float = field(
        default=0.1,
        metadata={'help': 'ratio of frames to perturb when frame noise is enabled'},
    )
    aug3d_frame_noise_std: float = field(
        default=0.01,
        metadata={'help': 'standard deviation of Gaussian noise for noisy frames'},
    )
    aug3d_feature_mask: bool = field(
        default=False,
        metadata={'help': 'enable random keypoint masking for DynHaMR 3D data'},
    )
    aug3d_feature_mask_ratio: float = field(
        default=0.1,
        metadata={'help': 'ratio of keypoints to mask when feature masking is enabled'},
    )
    aug3d_frame_mask: bool = field(
        default=False,
        metadata={'help': 'enable random all-keypoints frame masking for DynHaMR 3D data'},
    )
    aug3d_frame_mask_ratio: float = field(
        default=0.05,
        metadata={'help': 'ratio of full frames to mask when frame masking is enabled'},
    )
    aug3d_scale: bool = field(
        default=False,
        metadata={'help': 'enable slight random global scaling for DynHaMR 3D data'},
    )
    aug3d_scale_limit: float = field(
        default=0.1,
        metadata={'help': 'global scale perturbation limit (e.g., 0.1 -> scale in [0.9, 1.1])'},
    )
    aug3d_shift: bool = field(
        default=False,
        metadata={'help': 'enable slight random global translation for DynHaMR 3D data'},
    )
    aug3d_shift_std: float = field(
        default=0.02,
        metadata={'help': 'standard deviation for global translation noise in 3D augmentation'},
    )
    aug3d_horizontal_flip: bool = field(
        default=False,
        metadata={'help': 'enable horizontal flip with left/right hand swap for DynHaMR 3D data'},
    )
    aug3d_horizontal_flip_p: float = field(
        default=0.5,
        metadata={'help': 'probability of applying horizontal flip when enabled'},
    )
    tpu: bool = II("common.tpu")
    bpe_sentencepiece_model: str = II("bpe.sentencepiece_model")

@register_task("SL_topic_detection", dataclass=SLTopicDetectionConfig)
class SLTopicDetectionTask(FairseqTask):
    def __init__(self, cfg, label_dict=None, src_dict=None):  # TODO: check that src_dict is passed when text data is used
        super().__init__(cfg)
        self.label_dict = label_dict
        self.src_dict = src_dict
        if SignFeatsType_TD[cfg.feats_type] in [SignFeatsType_TD.text, SignFeatsType_TD.spot_align, SignFeatsType_TD.mouthings]:
            self.bpe_tokenizer = self.build_bpe(
                Namespace(
                    bpe='sentencepiece',
                    sentencepiece_model=cfg.bpe_sentencepiece_model
                )
            )
        self.softmax = nn.Softmax(dim=1)

    @classmethod
    def setup_task(cls, cfg):
        # Handle normalization_type override (for backward compatibility with launch scripts)
        if cfg.normalization_type is not None:
            cfg.normalization = cfg.normalization_type

        # ---------------------------------------------------------------------
        # TEMP PATCH (REMOVE AFTER EXPERIMENT): DynHaMR 3D -> 2D(XY) ablation
        # We force feat_dims to "0,1" so the model input size matches 2D setup.
        # NOTE: remove this block once the ablation experiment is completed.
        # ---------------------------------------------------------------------
        if SignFeatsType_TD[cfg.feats_type] == SignFeatsType_TD.dynhamr:
            if cfg.dynhamr_temp_map_to_mediapipe_2d:
                original_feat_dims = cfg.feat_dims
                cfg.feat_dims = "0,1"
                print(
                    "[TEMP PATCH][REMOVE ME] DynHaMR temporary 2D mapping ACTIVE: "
                    f"forcing task.feat_dims from '{original_feat_dims}' to '{cfg.feat_dims}'."
                )
                print(
                    "[TEMP PATCH][REMOVE ME] Mapping uses XY + optional global similarity transform. "
                    "Disable task.dynhamr_temp_map_to_mediapipe_2d to recover regular behavior."
                )
                if cfg.dynhamr_temp_per_video_map_to_mediapipe_2d:
                    print(
                        "[TEMP PATCH][REMOVE ME] DynHaMR per-video mapping ACTIVE: "
                        "one scale+translation per video (wrist-based statistics)."
                    )
            if cfg.dynhamr_temporal_interpolation:
                print(
                    "[TEMP PATCH][REMOVE ME] DynHaMR temporal interpolation ACTIVE: "
                    f"resampling sequences from {cfg.dynhamr_source_fps}fps to {cfg.dynhamr_target_fps}fps at dataset load time."
                )
                print(
                    "[TEMP PATCH][REMOVE ME] This is an experiment-only simulation; "
                    "remove/disable once fps strategy is finalized."
                )

        if SignFeatsType_TD[cfg.feats_type] == SignFeatsType_TD.dynhamr_2d:
            original_feat_dims = cfg.feat_dims
            cfg.feat_dims = "0,1"
            if original_feat_dims != cfg.feat_dims:
                print(
                    "[INFO] DynHaMR 2D mode ACTIVE: "
                    f"forcing task.feat_dims from '{original_feat_dims}' to '{cfg.feat_dims}'."
                )
        
        if 'SEED' in os.environ:
            seed = int(os.environ.get('SEED'))
            torch.manual_seed(seed)
            np.random.seed(seed)
        if SignFeatsType_TD[cfg.feats_type] in [SignFeatsType_TD.text, SignFeatsType_TD.spot_align, SignFeatsType_TD.mouthings]:
            # cfg.bpe_sentencepiece_model = os.environ.get('SP_MODEL', cfg.bpe_sentencepiece_model) ## TODO: this is a temporary fix for ALTI on transformerCLS
            dict_path = Path(cfg.bpe_sentencepiece_model).with_suffix('.txt')
            # print(f'dict_path = {dict_path}')
            if not dict_path.is_file():
                raise FileNotFoundError(f"Dict not found: {dict_path.as_posix()}")
            src_dict = Dictionary.load(dict_path.as_posix())
            logger.info(
                f"dictionary size ({dict_path.name}): " f"{len(src_dict):,}"
            )
            return cls(cfg, src_dict=src_dict)
        return cls(cfg)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs): # TODO: why are we getting val even when we changed the subset for test?
        is_train_split = "train" in split
        root_dir = Path(self.cfg.data)
        assert root_dir.is_dir(), f'{root_dir} does not exist'
        if not is_train_split:
            if SignFeatsType_TD(self.cfg.feats_type) in [SignFeatsType_TD.mediapipe_keypoints, SignFeatsType_TD.dynhamr, SignFeatsType_TD.dynhamr_2d]:
                if self.cfg.dataset=="How2Sign":
                    if SignFeatsType_TD(self.cfg.feats_type) == SignFeatsType_TD.dynhamr:
                        # For DynHaMR 3D, use the precomputed .pose manifest by default.
                        # manifest_file = root_dir / f"how2sign_{split}_dynhamr.tsv"
                        manifest_file = root_dir / f"how2sign_{split}_dynhamr_preprocessed.tsv"
                    else:
                        #manifest_file = root_dir / f"how2sign_{split}_proves_filtered_for_egosign.tsv"
                        #manifest_file = root_dir / f"how2sign_{split}_calcula_smooth.tsv"
                        #manifest_file = root_dir / f"how2sign_{split}_proves_filtered_for_egosign_smooth.tsv"
                        manifest_file = root_dir / f"how2sign_{split}_proves_filtered_for_egosign_smooth_normalized.tsv"
                elif self.cfg.dataset in ["EgoSign-rgb", "EgoSign-rgb-dynhamr", "dynhamr-egosign-rgb"]:
                    if SignFeatsType_TD(self.cfg.feats_type) in [SignFeatsType_TD.dynhamr, SignFeatsType_TD.dynhamr_2d]:
                        manifest_file = root_dir / f"egosign_{split}_dynhamr.tsv"
                    else:
                        #manifest_file = root_dir / f"egosign_{split}_proves_filtered.tsv"
                        manifest_file = root_dir / f"egosign_{split}_proves_filtered_smooth_normalized.tsv"
                elif self.cfg.dataset in ["EgoSign-rgb-dynhamr-processed", "dynhamr-egosign-rgb-processed"]:
                    if SignFeatsType_TD(self.cfg.feats_type) in [SignFeatsType_TD.dynhamr, SignFeatsType_TD.dynhamr_2d]:
                        manifest_file = root_dir / f"egosign_{split}_dynhamr_preprocessed.tsv"
                    else:
                        raise ValueError(
                            "dataset='EgoSign-rgb-dynhamr-processed' requires feats_type in {dynhamr, dynhamr_2d}."
                        )
                elif self.cfg.dataset=="dynhamr-rgb-front":
                    if SignFeatsType_TD(self.cfg.feats_type) in [SignFeatsType_TD.dynhamr, SignFeatsType_TD.dynhamr_2d]:
                        manifest_file = root_dir / f"egosign_{split}_dynhamr.tsv"
                    else:
                        raise ValueError(
                            "dataset='dynhamr-rgb-front' requires feats_type in {dynhamr, dynhamr_2d}."
                        )
                elif self.cfg.dataset in ["EgoSign-oc", "dynhamr-oc", "EgoSign-oc-dynhamr", "dynhamr-egosign-oc", "EgoSign-oc-dynhamr-processed"]:
                    # For 3D data, "oc" means oculus (same as EgoSign-oc-homo for 2D)
                    if SignFeatsType_TD(self.cfg.feats_type) in [SignFeatsType_TD.dynhamr, SignFeatsType_TD.dynhamr_2d]:
                        manifest_file = root_dir / f"egosign_oc_{split}_dynhamr.tsv"
                    else:
                        manifest_file = root_dir / f"egosign_oc_{split}_homography_smooth_normalized.tsv"
                elif self.cfg.dataset=="EgoSign-oc-homo":
                    #manifest_file = root_dir / f"egosign_oc_{split}_proves_filtered.tsv"
                    #manifest_file = root_dir / f"egosign_oc_{split}_proves_filtered_smooth_normalized.tsv"
                    manifest_file = root_dir / f"egosign_oc_{split}_homography_smooth_normalized.tsv"
                elif self.cfg.dataset=="EgoSign-oc-resec":
                    manifest_file = root_dir / f"egosign_oc_{split}_resectioning_smooth_normalized.tsv"
                elif self.cfg.dataset=="EgoSign-combined-homo":
                    manifest_file = root_dir / f"egosign_{split}_proves_filtered_smooth_normalized_combined.tsv"
                elif self.cfg.dataset=="EgoSign-combined-resec":
                    manifest_file = root_dir / f"egosign_{split}_proves_filtered_smooth_normalized_combined_resectioning.tsv"
                else:
                    #This means we are running the old version of the script, check utils for the names of the landmarks...
                    manifest_file = root_dir / f"{split}_filt.csv"
            else:
                manifest_file = root_dir / f"{split}.csv"
        else:
            if self.cfg.dataset=="":
                manifest_file = root_dir / f"{split}_filt.csv" #This means we are running the old version of the script, check utils for the names of the landmarks...
            elif self.cfg.dataset in ["EgoSign-rgb", "EgoSign-rgb-dynhamr", "dynhamr-egosign-rgb", "dynhamr-rgb-front"]:
                if SignFeatsType_TD(self.cfg.feats_type) in [SignFeatsType_TD.dynhamr, SignFeatsType_TD.dynhamr_2d]:
                    manifest_file = root_dir / f"egosign_{split}_dynhamr.tsv"
                else:
                    manifest_file = root_dir / f"egosign_{split}_proves_filtered_smooth_normalized.tsv"
            elif self.cfg.dataset in ["EgoSign-rgb-dynhamr-processed", "dynhamr-egosign-rgb-processed"]:
                if SignFeatsType_TD(self.cfg.feats_type) in [SignFeatsType_TD.dynhamr, SignFeatsType_TD.dynhamr_2d]:
                    manifest_file = root_dir / f"egosign_{split}_dynhamr_preprocessed.tsv"
                else:
                    manifest_file = root_dir / f"egosign_{split}_proves_filtered_smooth_normalized.tsv"
            elif self.cfg.dataset in ["EgoSign-oc", "dynhamr-oc", "EgoSign-oc-dynhamr", "dynhamr-egosign-oc", "EgoSign-oc-dynhamr-processed"]:
                if SignFeatsType_TD(self.cfg.feats_type) in [SignFeatsType_TD.dynhamr, SignFeatsType_TD.dynhamr_2d]:
                    manifest_file = root_dir / f"egosign_oc_{split}_dynhamr.tsv"
                else:
                    manifest_file = root_dir / f"egosign_oc_{split}_homography_smooth_normalized.tsv"
            else: 
                if SignFeatsType_TD(self.cfg.feats_type) == SignFeatsType_TD.dynhamr:
                    # For DynHaMR 3D, use the precomputed .pose manifest by default.
                    # manifest_file = root_dir / f"how2sign_{split}_dynhamr.tsv"
                    manifest_file = root_dir / f"how2sign_{split}_dynhamr_preprocessed.tsv"
                    #If we want to keep fine_tuning, with the added oc data:
                    #logger.warning(
                    #    "Using DynHaMR 3D manifest for How2Sign w/ val from OC. Make sure this is intentional and that the file exists."
                    #)
                    #manifest_file = root_dir / f"how2sign_{split}_dynhamr_preprocessed_w_val_oc.tsv"
                    #manifest_file = root_dir / f"how2sign_{split}_dynhamr_preprocessed_only_val_oc.tsv"
                    
                else:
                    #manifest_file = root_dir / f"how2sign_{split}_calcula_smooth.tsv" #We only have training data for How2Sign.
                    #manifest_file = root_dir / f"how2sign_{split}_proves_filtered.tsv"
                    manifest_file = root_dir / f"how2sign_{split}_calcula_smooth_normalized_-10files.tsv"
                logger.info(f"remember to change the manifest file in task: manifest_file = {manifest_file}")
        
        if SignFeatsType_TD(self.cfg.feats_type) in [
            SignFeatsType_TD.keypoints, SignFeatsType_TD.mediapipe_keypoints, SignFeatsType_TD.dynhamr, SignFeatsType_TD.dynhamr_2d,
            SignFeatsType_TD.rotational, SignFeatsType_TD.mediapipe_rotational,
            SignFeatsType_TD.i3d, SignFeatsType_TD.spot_align_albert, SignFeatsType_TD.text_albert, SignFeatsType_TD.mouthings_albert
        ]:
            feats_path = root_dir / f"{split}_filt.h5"        
        elif SignFeatsType_TD(self.cfg.feats_type) == SignFeatsType_TD.dynhamr:
            # DynHaMR data is organized differently - each video has its own directory
            # The actual paths will be in the manifest file (signs_file column)
            feats_path = None        
        elif SignFeatsType_TD(self.cfg.feats_type) == SignFeatsType_TD.video:
            DATA_PATH = {
                'train': '/home/alvaro/Documents/ML and DL/How2Sign/TFG/Sign-Language-Topic-Detection/data/How2Sign/video',
                'val': '/home/alvaro/Documents/ML and DL/How2Sign/TFG/Sign-Language-Topic-Detection/data/How2Sign/video',
                'test': '/home/alvaro/Documents/ML and DL/How2Sign/TFG/Sign-Language-Topic-Detection/data/How2Sign/video',
            }
            feats_path = DATA_PATH[split]
        elif SignFeatsType_TD(self.cfg.feats_type) in [SignFeatsType_TD.text, SignFeatsType_TD.spot_align, SignFeatsType_TD.mouthings]:
            feats_path = None
        else:
            raise NotImplementedError(
                (
                    'Features other than i3d, keypoints, rotational, spot_align, spot_align_albert, mouthings, mouthings_albert text or text_albert'
                    'are not available for How2Sign yet'
                )
            )

        if self.cfg.num_batch_buckets > 0 or self.cfg.tpu:
            raise NotImplementedError("Pending to implement bucket_pad_length_dataset wrapper")

        print(f'manifest_file {manifest_file}', flush=True)

        # TODO: from this old implementation, what we have here is:
        # manifest_file = /home/ltarres/temp_data/How2Sign/TopicDetection/mediapipe_keypoints/val_filt.csv
        # This manifest_file has VIDEO_ID	CATEGORY_ID	START_FRAME	END_FRAME and that's it.
        # feats_path = /home/ltarres/temp_data/How2Sign/TopicDetection/mediapipe_keypoints/val_filt.h5
        # feats_type = mediapipe_keypoints
        # bodyparts = ['lefthand', 'righthand']
        # feat_dims = [0, 1]
        # min_sample_size = 150
        # max_sample_size = 250000
        # shuffle = True
        # normalize = True
        if SignFeatsType_TD(self.cfg.feats_type) in [SignFeatsType_TD.mediapipe_keypoints, SignFeatsType_TD.dynhamr, SignFeatsType_TD.dynhamr_2d]:
            if manifest_file.stem == 'val_filt' or manifest_file.stem == 'test_filt' or manifest_file.stem == 'train_filt':
                self.datasets[split] = SLTopicDetectionDatasetOld.from_manifest_file(
                manifest_file=manifest_file,
                feats_path=feats_path,
                feats_type=self.cfg.feats_type,
                bodyparts=self.cfg.body_parts.split(','),
                feat_dims=[int(d) for d in self.cfg.feat_dims.split(',')],
                min_sample_size=self.cfg.min_source_positions,
                max_sample_size=self.cfg.max_source_positions,
                shuffle=self.cfg.shuffle_dataset,
                normalize=True, #Crec que està a true
                )
                '''
                target_id_old = "EQWFrWeRVjQ"
                target_id_old_index = self.datasets[split].ids.index(target_id_old).as_py()
                old_feats = self.datasets[split].__getitem__(int(target_id_old_index))["source"]
                # id : 110
                # h2s_id : EQWFrWeRVjQ

                old_feats.shape -> torch.Size([2417, 84])

                # To check the normalization, it is normalizing all the x and all the y.
                mean_per_dim = old_feats.reshape(-1, 42, 2).permute(2, 0, 1).mean(dim=(1, 2))  # Mean per dimension
                std_per_dim = old_feats.reshape(-1, 42, 2).permute(2, 0, 1).std(dim=(1, 2))  # Std per dimension

                print(f"Mean per dimension (should be ~0): {mean_per_dim}")
                print(f"Std per dimension (should be ~1): {std_per_dim}")
                '''
            else:
                #Print all the parameters that we enter here
                print(f"manifest_file = {manifest_file}")
                print(f"feats_type = {self.cfg.feats_type}")
                print(f"bodyparts = {self.cfg.body_parts.split(',')}")
                print(f"feat_dims = {[int(d) for d in self.cfg.feat_dims.split(',')]}")
                print(f"min_sample_size = {self.cfg.min_source_positions}")
                print(f"max_sample_size = {self.cfg.max_source_positions}") 
                print(f"shuffle = {self.cfg.shuffle_dataset}")
                print(f"normalization = {self.cfg.normalization}")
                print(f"to_camera_coordinates = {self.cfg.to_camera_coordinates}")
                print(f"dynhamr_temporal_interpolation = {self.cfg.dynhamr_temporal_interpolation}")
                print(f"dynhamr_source_fps = {self.cfg.dynhamr_source_fps}")
                print(f"dynhamr_target_fps = {self.cfg.dynhamr_target_fps}")
                print(f"dynhamr_fill_mode = {self.cfg.dynhamr_fill_mode}")
                print(f"dynhamr_long_gap_linear_threshold = {self.cfg.dynhamr_long_gap_linear_threshold}")
                print(f"dynhamr_temp_map_to_mediapipe_2d = {self.cfg.dynhamr_temp_map_to_mediapipe_2d}")
                print(f"dynhamr_temp_interpolate_missing = {self.cfg.dynhamr_temp_interpolate_missing}")
                print(f"dynhamr_temp_similarity_scale = {self.cfg.dynhamr_temp_similarity_scale}")
                print(f"dynhamr_temp_translate_x = {self.cfg.dynhamr_temp_translate_x}")
                print(f"dynhamr_temp_translate_y = {self.cfg.dynhamr_temp_translate_y}")
                print(f"dynhamr_temp_per_video_map_to_mediapipe_2d = {self.cfg.dynhamr_temp_per_video_map_to_mediapipe_2d}")
                print(f"dynhamr_temp_per_video_target_center_x = {self.cfg.dynhamr_temp_per_video_target_center_x}")
                print(f"dynhamr_temp_per_video_target_center_y = {self.cfg.dynhamr_temp_per_video_target_center_y}")
                print(f"dynhamr_temp_per_video_target_wrist_distance = {self.cfg.dynhamr_temp_per_video_target_wrist_distance}")
                print(f"dynhamr_2d_mp_pose_manifest_file = {self.cfg.dynhamr_2d_mp_pose_manifest_file}")
                print(f"aug3d_random_resample = {self.cfg.aug3d_random_resample}")
                print(f"aug3d_resample_p = {self.cfg.aug3d_resample_p}")
                print(f"aug3d_resample_limit = {self.cfg.aug3d_resample_limit}")
                print(f"aug3d_frame_noise = {self.cfg.aug3d_frame_noise}")
                print(f"aug3d_frame_noise_ratio = {self.cfg.aug3d_frame_noise_ratio}")
                print(f"aug3d_frame_noise_std = {self.cfg.aug3d_frame_noise_std}")
                print(f"aug3d_feature_mask = {self.cfg.aug3d_feature_mask}")
                print(f"aug3d_feature_mask_ratio = {self.cfg.aug3d_feature_mask_ratio}")
                print(f"aug3d_frame_mask = {self.cfg.aug3d_frame_mask}")
                print(f"aug3d_frame_mask_ratio = {self.cfg.aug3d_frame_mask_ratio}")
                print(f"aug3d_scale = {self.cfg.aug3d_scale}")
                print(f"aug3d_scale_limit = {self.cfg.aug3d_scale_limit}")
                print(f"aug3d_shift = {self.cfg.aug3d_shift}")
                print(f"aug3d_shift_std = {self.cfg.aug3d_shift_std}")
                print(f"aug3d_horizontal_flip = {self.cfg.aug3d_horizontal_flip}")
                print(f"aug3d_horizontal_flip_p = {self.cfg.aug3d_horizontal_flip_p}")  
                
                self.datasets[split] = SLTopicDetectionDataset.from_manifest_file(
                    manifest_file=manifest_file,
                    feats_type=self.cfg.feats_type,
                    #normalization=self.cfg.normalization,
                    data_augmentation=(self.cfg.data_augmentation and is_train_split),
                    min_sample_size=self.cfg.min_source_positions,
                    max_sample_size=self.cfg.max_source_positions,
                    shuffle=self.cfg.shuffle_dataset,
                    bodyparts=self.cfg.body_parts.split(','),
                    feat_dims=[int(d) for d in self.cfg.feat_dims.split(',')],
                    normalization=self.cfg.normalization,
                    to_camera_coordinates=self.cfg.to_camera_coordinates,
                    dynhamr_temporal_interpolation=self.cfg.dynhamr_temporal_interpolation,
                    dynhamr_source_fps=self.cfg.dynhamr_source_fps,
                    dynhamr_target_fps=self.cfg.dynhamr_target_fps,
                    dynhamr_fill_mode=self.cfg.dynhamr_fill_mode,
                    dynhamr_long_gap_linear_threshold=self.cfg.dynhamr_long_gap_linear_threshold,
                    dynhamr_temp_map_to_mediapipe_2d=self.cfg.dynhamr_temp_map_to_mediapipe_2d,
                    dynhamr_temp_interpolate_missing=self.cfg.dynhamr_temp_interpolate_missing,
                    dynhamr_temp_similarity_scale=self.cfg.dynhamr_temp_similarity_scale,
                    dynhamr_temp_translate_x=self.cfg.dynhamr_temp_translate_x,
                    dynhamr_temp_translate_y=self.cfg.dynhamr_temp_translate_y,
                    dynhamr_temp_per_video_map_to_mediapipe_2d=self.cfg.dynhamr_temp_per_video_map_to_mediapipe_2d,
                    dynhamr_temp_per_video_target_center_x=self.cfg.dynhamr_temp_per_video_target_center_x,
                    dynhamr_temp_per_video_target_center_y=self.cfg.dynhamr_temp_per_video_target_center_y,
                    dynhamr_temp_per_video_target_wrist_distance=self.cfg.dynhamr_temp_per_video_target_wrist_distance,
                    dynhamr_2d_mp_pose_manifest_file=self.cfg.dynhamr_2d_mp_pose_manifest_file,
                    aug3d_random_resample=self.cfg.aug3d_random_resample,
                    aug3d_resample_p=self.cfg.aug3d_resample_p,
                    aug3d_resample_limit=self.cfg.aug3d_resample_limit,
                    aug3d_frame_noise=self.cfg.aug3d_frame_noise,
                    aug3d_frame_noise_ratio=self.cfg.aug3d_frame_noise_ratio,
                    aug3d_frame_noise_std=self.cfg.aug3d_frame_noise_std,
                    aug3d_feature_mask=self.cfg.aug3d_feature_mask,
                    aug3d_feature_mask_ratio=self.cfg.aug3d_feature_mask_ratio,
                    aug3d_frame_mask=self.cfg.aug3d_frame_mask,
                    aug3d_frame_mask_ratio=self.cfg.aug3d_frame_mask_ratio,
                    aug3d_scale=self.cfg.aug3d_scale,
                    aug3d_scale_limit=self.cfg.aug3d_scale_limit,
                    aug3d_shift=self.cfg.aug3d_shift,
                    aug3d_shift_std=self.cfg.aug3d_shift_std,
                    aug3d_horizontal_flip=self.cfg.aug3d_horizontal_flip,
                    aug3d_horizontal_flip_p=self.cfg.aug3d_horizontal_flip_p,
                )
            if is_train_split:
                self.datasets[split].filter_by_length(
                    self.cfg.min_source_positions, self.cfg.max_source_positions
                )
        else:
            #I think this was for the SLTopicDetectionDatasetOld, so it never enters here...
            self.datasets[split] = SLTopicDetectionDataset.from_manifest_file(
                manifest_file=manifest_file,
                feats_path=feats_path,
                feats_type=self.cfg.feats_type,
                bodyparts=self.cfg.body_parts.split(','),
                feat_dims=[int(d) for d in self.cfg.feat_dims.split(',')],
                min_sample_size=self.cfg.min_source_positions,
                max_sample_size=self.cfg.max_source_positions,
                shuffle=self.cfg.shuffle_dataset,
                normalization=self.cfg.normalization,
            )

        #data = pd.read_csv(manifest_file, sep="\t") #This was not reading all the labels, 
        raw_lines = []
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                raw_lines.append((line_num, line.strip()))
        header = raw_lines[0][1].split('\t')
        manual_data = []
        for line_num, line in raw_lines[1:]: 
            fields = line.split('\t')
            row_dict = {header[i]: fields[i] for i in range(len(header))}
            manual_data.append(row_dict)
        data = pd.DataFrame(manual_data)
        
        text_compressor = TextCompressor(level=self.cfg.text_compression_level)
        
        
        if manifest_file.stem == f'{split}_filt':#This is for the old version...
            labels = [
                text_compressor.compress(str(row['CATEGORY_ID']))
                for _, row in data.iterrows()
                if row['VIDEO_ID'] not in self.datasets[split].skipped_ids
            ]
        else:
            labels = [
            text_compressor.compress(str(data[data['id_vid'] == id_vid]['topic'].iloc[0]))
            for id_vid in self.datasets[split].ids
            if id_vid in data['id_vid'].values and id_vid not in self.datasets[split].skipped_ids
            ]

        assert len(labels) == len(self.datasets[split]), (
            f"The length of the labels list ({len(labels)}) and the dataset length"
            f" after skipping some ids ({len(self.datasets[split].skipped_ids)})"
            f" do not match. Original dataset length is ({len(self.datasets[split])})"
        )

        def process_label_fn(label):
            return torch.tensor([int(label)]) - 1

        def label_len_fn(label):
            return len(torch.tensor([int(label)]))

        if SignFeatsType_TD(self.cfg.feats_type) in [SignFeatsType_TD.text, SignFeatsType_TD.spot_align, SignFeatsType_TD.mouthings]:
            # TODO: decide if input text data should be compressed also
            def process_sentence_fn(sentence):
                tokens = self.source_dictionary.encode_line(
                            self.bpe_tokenizer.encode(sentence),
                            append_eos=False,
                            add_if_not_exist=False,
                        )
                return tokens

            def sentence_len_fn(tokens):
                return tokens.numel()

            sentences = [
                process_sentence_fn(row['TEXT'])
                for i, row in data.iterrows()
                if row['VIDEO_ID'] not in self.datasets[split].skipped_ids
            ]
            lengths = [sentence_len_fn(tokens) for tokens in sentences]

            assert len(sentences) == len(self.datasets[split]), (
                f"The length of the sentences list ({len(sentences)}) and the dataset's length"
                f" after skipping some ids ({len(self.datasets[split].skipped_ids)})"
                f" do not match. Original dataset length is ({len(self.datasets[split])})"
            )

            labels = [
                torch.tensor([int(row['CATEGORY_ID'])]) - 1
                for _, row in data.iterrows()
                if row['VIDEO_ID'] not in self.datasets[split].skipped_ids
            ]

            self.datasets[split] = LanguagePairDataset(
                src=sentences,
                src_sizes=lengths,
                src_dict=self.source_dictionary,
                tgt=labels,
                tgt_sizes=torch.ones(len(labels)),  # targets have length 1
                left_pad_source=False,
                # Since our target is a single class label, there's no need for
                # teacher forcing. If we set this to ``True`` then our Model's
                # ``forward()`` method would receive an additional argument called
                # *prev_output_tokens* that would contain a shifted version of the
                # target sequence.
                input_feeding=False,
                append_eos_to_target=False,
                eos=self.source_dictionary.eos(),
            )
        else:
            self.datasets[split] = AddTargetDataset(
                self.datasets[split],
                labels,
                pad=0,
                eos=None,
                batch_targets=True,
                process_label=process_label_fn,
                label_len_fn=label_len_fn,
                add_to_input=False, # TODO: figure out why this is False and not True
            )

    @property
    def target_dictionary(self):
        return self.label_dict

    @property
    def source_dictionary(self):
        return self.src_dict

    def max_positions(self):
        return self.cfg.max_source_positions, self.cfg.max_target_positions

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = []
        for l in lines:
            h5_file, _id = l.split(':')
            feats_path = h5py.File(h5_file, "r")
            n_frames.append(np.array(feats_path[_id]).shape[0])
        return lines, n_frames

    # TODO: Implement this method
    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        raise NotImplementedError
        return SpeechToTextDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )

    #Add this for validation
    def build_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg)
        if from_checkpoint:
            pass  # TODO: Implement this
        return model

    #Add this for validation
    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_accuracy:
            model.eval()
            with torch.no_grad():
                out = model(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'])
                preds = torch.argmax(self.softmax(out), dim=1)

            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            targets = sample['target']
            logging_output['_acc_counts_'] = sum(
                torch.eq(
                    preds.flatten(),
                    targets.flatten()
                    )
                ).item()
            logging_output['_acc_totals_'] = targets.flatten().shape[0]
            
            logging_output['preds'] = preds.flatten().tolist()
            logging_output['targets'] = targets.flatten().tolist()
            
        return loss, sample_size, logging_output

    def inference_step(
        self, sample, model, output_attentions=None, targets_container=None, preds_container=None,
    ):
        #The above is during training, this is for the inference post-training.
        model.eval()
        with torch.no_grad():
            if output_attentions:
                out = model(
                    sample['net_input']['src_tokens'],
                    sample['net_input']['src_lengths'],
                    output_attentions=output_attentions
                )
            else:
                out = model(
                    sample['net_input']['src_tokens'],
                    sample['net_input']['src_lengths']
                )

            preds = torch.argmax(self.softmax(out), dim=1)

        # we split counts into separate entries so that they can be
        # summed efficiently across workers using fast-stat-sync
        targets = sample['target']
        if targets_container is not None:
            targets_container.append(targets)
        if preds_container is not None:
            preds_container.append(preds)

        counts = sum(
            torch.eq(
                preds.flatten(),
                targets.flatten()
                )
            ).item()
        total = targets.flatten().shape[0]
        return counts, total

    #Add this for validation
    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_accuracy:

            def sum_logs(key):
                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            counts.append(sum_logs('_acc_counts_'))
            totals.append(sum_logs('_acc_totals_'))
            
            if max(totals) > 0: #this is in the case where we are doing inference
                all_preds = []
                all_targets = []
                for log in logging_outputs:
                    all_preds.extend(log['preds']) #log.get('_preds', [])
                    all_targets.extend(log['targets'])
                
                # We need to do this in order to then compute the confusion_matrix (?)
                metrics.log_array('preds', np.array(all_preds)) #Actually, this should accumulate them all, right?
                metrics.log_array('targets', np.array(all_targets))
                
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar('_acc_counts_', np.array(counts))
                metrics.log_scalar('_acc_totals_', np.array(totals))

                def compute_accuracy(meters):
                    acc = meters['_acc_counts_'].sum[0] / meters['_acc_totals_'].sum[0]
                    return round(acc, 2)

                metrics.log_derived('acc', compute_accuracy)

                def _compute_macro_prf(meters):
                    classes = np.arange(10)
                    num_classes = len(classes)
                    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

                    for t, p in zip(meters['targets'].all_values, meters['preds'].all_values):
                        confusion_matrix[t, p] += 1

                    tp = np.diag(confusion_matrix).astype(np.float64)
                    predicted_positives = np.sum(confusion_matrix, axis=0).astype(np.float64)
                    actual_positives = np.sum(confusion_matrix, axis=1).astype(np.float64)

                    precision_per_class = np.divide(
                        tp,
                        predicted_positives,
                        out=np.zeros_like(tp),
                        where=predicted_positives != 0,
                    )
                    recall_per_class = np.divide(
                        tp,
                        actual_positives,
                        out=np.zeros_like(tp),
                        where=actual_positives != 0,
                    )
                    f1_per_class = np.divide(
                        2 * precision_per_class * recall_per_class,
                        precision_per_class + recall_per_class,
                        out=np.zeros_like(tp),
                        where=(precision_per_class + recall_per_class) != 0,
                    )

                    precision_macro = float(np.mean(precision_per_class))
                    recall_macro = float(np.mean(recall_per_class))
                    f1_macro = float(np.mean(f1_per_class))

                    return precision_macro, recall_macro, f1_macro

                metrics.log_derived('precision_macro', lambda meters: _compute_macro_prf(meters)[0])
                metrics.log_derived('recall_macro', lambda meters: _compute_macro_prf(meters)[1])
                metrics.log_derived('f1_macro', lambda meters: _compute_macro_prf(meters)[2])
                
                #Here we need to figure out how to plot the confusion matrix
                # Aggregate the confusion matrix meter
                
                def compute_confusion_matrix(meters):
                    #from sklearn.metrics import confusion_matrix
                    #confusion_matrix = confusion_matrix(meters['targets'], meters['preds'])
                    # Get unique class labels
                    classes = np.arange(10) #np.unique(np.concatenate((meters['preds'].all_values, meters['targets'].all_values)))
                    num_classes = len(classes)

                    # Initialize confusion matrix
                    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

                    # Fill confusion matrix
                    for t, p in zip(meters['targets'].all_values, meters['preds'].all_values):
                        confusion_matrix[t, p] += 1 #Rows are targets and columns are predictions
    
                    return confusion_matrix
                
                metrics.log_derived_matrix('confusion_matrix', compute_confusion_matrix)
                
                