

import os
import sys
import logging
from enum import Enum
from pathlib import Path
from typing import List, Union, Optional

import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from pose_format import Pose # TODO: We will load from the .pose files

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
    rotational = "rotational"
    mediapipe_rotational = "mediapipe_rotational"
    i3d = "i3d"
    CNN2d = "CNN2d"
    video = "video"

class NormType_TD(Enum):
    body="body"
    kp_wise = "kp_wise"
    global_xyz = "global_xyz"
    layer_norm = "layer_norm" #to add the same normalizaiton as original TD
    center_and_scale = "center_and_scale"
    
class SLTopicDetectionDataset(FairseqDataset):
    def __init__(
        self,
        ids: List[str],
        feats_files: List[Union[Path, str]],
        offsets: List[int],
        sizes: List[int],
        feats_type: SignFeatsType_TD,
        ids_sent: List[List[str]],
        normalization: NormType_TD = NormType_TD.body,
        data_augmentation: bool = False,
        min_sample_size: int = 0,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
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
        self.feats_type = feats_type
        self.ids_sent = ids_sent
        self.normalization = normalization # I think this we were calling it normalize
        self.data_augmentation = data_augmentation
        self.min_sample_size = min_sample_size
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.shuffle = shuffle
        self.skipped_ids = []

        # The ones that we have removed: do we actually need this? Probably yes!
        self.bodyparts = bodyparts
        self.feat_dims = feat_dims
        #self.manifest = manifest
        # if feats_type == SignFeatsType.video, feats_path is the directory where .mp4 files of the corresponding split are stored
        #self.ids = [_id for _id in ids]

    def filter_by_length(self, min_sample_size, max_sample_size):
        for _id, size in zip(self.ids[:], self.sizes[:]):
            #Now here we have a combination of them, so what is the minimum? The combined ones
            #sum_size = sum(size) #We don't have a list of sizes
            sum_size=size
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
        
        # We need an extra column that is the size of the whole video... because if not, we cannot know the length of the video
        # Process grouped data
        for id_vid, data in grouped_data.items():
            ids.append(id_vid)  # Use id_vid as the new id
            feats_files.append(data["feats_file"])
            offsets.append(data["offsets"])  # Concatenated offsets
            #sizes.append(data["sizes"])  # Concatenated sizes
            sizes.append(data["vid_length"])  # Only 1 size
            #total length should be the rest between: sum of the max offset with the corresponding size, and the minimum of the offsets
            #total_length = max([offset + size for offset, size in zip(data["offsets"], data["sizes"])]) - min(data["offsets"])
            #video_lengths.append(total_length)  # Concatenated sizes
            ids_sent.append(data["ids_sent"])  # Original sentence ids    
        
        logger.info(f"loaded {len(ids)} samples")
        
        feats_type = row['signs_type']
        return cls(ids, feats_files=feats_files, offsets=offsets, sizes=sizes,
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

    def __getitem__(self, index):
        _id = self.ids[index]
        feats_file = self.feats_files[index]
        offset = self.offsets[index]
        length = self.sizes[index]
        
        if SignFeatsType_TD(self.feats_type) == SignFeatsType_TD.mediapipe_keypoints:
            with open(feats_file, "rb") as f:
                pose = Pose.read(f.read())
            
            if not pose.body.data.flags.writeable:
                pose.body.data = pose.body.data.copy() # We could also force this: pose.body.data.setflags(write=True)

            # TODO: make here the option to load the sentences independently, or the full video
            #frames_list = []
            # Select the frames corresponding to the sentences, given the offset and length
            #for offset_i, length_i in zip(offset, length):
            #    frames_list.extend(range(offset_i, offset_i+length_i))
            
            # TODO: with this we are taking the first timestamp until the last of each sentence.
            #frames_list = list(range(min(offset), max([offset + size for offset, size in zip(self.offsets[index], self.sizes[index])]))) 

            #frames_list = list(range(offset, offset+length)) #this is the case where there is only one sentence

            # Fix to bypass some examples that are wrong, out of range.
            #frames_list = [fr for fr in frames_list if fr < pose.body.data.shape[0]]
            
            #pose.body = pose.body.select_frames(frames_list) #This is when we want to take from the first of the first sentence until the last of the last sentence
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
    
    def postprocess(self, feats):
        # TODO: this, we still need to check which post-processing we want to do.
        from fairseq.data.sign_language.utils import (
            select_keypoints_by_bodypart,
            select_keypoints_by_dimension,
        )
        if SignFeatsType_TD[self.feats_type] is SignFeatsType_TD.mediapipe_keypoints:
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
            
            #First we need to divide the "pixel" value by the width and height of the image # I don't remember when I added this...
            feats.body.data[..., 0] = feats.body.data[..., 0] #/ feats.header.dimensions.width #Since we are normalizing by shoulder, we do not diviide here.
            feats.body.data[..., 1] = feats.body.data[..., 1] #/ feats.header.dimensions.height
            
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
                # for layer normalization, we want to normalize each feature within the same sample independently. 
                feats_split = feats.body.data.transpose(3, 0, 1, 2) # to have the x and y dimensions in the first axis (from (3628, 1, 42, 2) to (2, 3628, 1, 42))
                with torch.no_grad():
                    feats_norm_split = F.layer_norm(torch.from_numpy(feats_split), feats_split.shape[1:]) #to do layer norm independently for each dimension , it does it into dimension (3628, 1, 42)
                feats.body.data = np.ma.MaskedArray(feats_norm_split.numpy().transpose(1, 2, 3, 0), feats.body.data.mask)
                #In How2Sign, from the resulting array, what did we normalize?
                
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
        #Here this should break because we are working with poses now.
        if self.feats_type == SignFeatsType_TD.mediapipe_keypoints.name:
            sizes = [s["source"].body.data.shape[0] for s in samples]
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
        for sample in samples:
            feat = sample["source"]
            if self.feats_type == SignFeatsType_TD.mediapipe_keypoints.name:
                if feat.body.data.shape[1] > 1:
                    logger.warning(f"More than one person in frame, keeping just the first one")
                
                feat.body.data = feat.body.data[:, 0]
                
                padding_mask = (~feat.body.data.mask).sum((1,2)) > 0
                if padding_mask.all():
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

        if self.feats_type == SignFeatsType_TD.mediapipe_keypoints.name:
            #padding_masks = torch.stack(padding_masks)
            if len(sizes) != torch.stack(collated_sources).float().shape[0]:
                import pdb; pdb.set_trace()
            return {
                'vid_id': vid_ids,
                'id': torch.LongTensor(ids),
                'net_input': {
                    'src_tokens': torch.stack(collated_sources).float(), 
                    'src_lengths': torch.Tensor(sizes)  # FIXME: If you use buckets
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
        return self.size(index) # TODO: What is this??

    def size(self, index): #Probably this is done to filter, I don't know where else
        # TODO: I think the length here
        #this is to have the size that is the beginning and end of sentence
        #total_length = max([offset + size for offset, size in zip(self.offsets[index], self.sizes[index])]) - min(self.offsets[index])
        #return total_length
        #return sum(self.sizes[index])
        return self.sizes[index]

    def ordered_indices(self):
        # TODO: I think the length here should be different, not the sum of sizes but something different.
        if self.shuffle:
            #total_sizes = [sum(s) for s in self.sizes]
            #total_sizes = self.sizes
            #total_sizes = [max([offset + size for offset, size in zip(self.offsets[index], self.sizes[index])]) - min(self.offsets[index])for index in range(len(self.sizes))]
            #order = np.lexsort( #orders by the sizes
            #    [np.random.permutation(len(self)), np.array(total_sizes)] #random permutation of indeces
            #)
            order = np.lexsort(
                [np.random.permutation(len(self)), np.array(self.sizes)]
            )
            return order[::-1] #reverse order so that it is descending order
        else:
            return np.arange(len(self))
