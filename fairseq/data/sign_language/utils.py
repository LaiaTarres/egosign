import torch
import torch.nn.functional as F
from typing import Union

from typing import List, Tuple, Optional

from fairseq.data.sign_language import SignFeatsType, SignFeatsType_TD

def get_num_feats(
    feats_type: Union[SignFeatsType,SignFeatsType_TD],
    bodyparts: Optional[List[str]] = None,
    feat_dims: Optional[List[int]] = None
) -> int:
    num_feats = {
        SignFeatsType.mediapipe: {
            'face': 70,
            'upperbody': 8,
            'lowerbody': 16,
            'lefthand': 21,
            'righthand': 21
            #'LEFT_HAND_LANDMARKS': 21,
            #'RIGHT_HAND_LANDMARKS': 21
        }, # TODO: change this for the right one
        #SignFeatsType.mediapipe: {
        #    'face': 70,
        #    'upperbody': 8,
        #    'lowerbody': 16,
        #    'lefthand': 21,
        #    'righthand': 21
        #},
        SignFeatsType.openpose: {
            'face': 70,
            'upperbody': 8,
            'lowerbody': 16,
            'lefthand': 21,
            'righthand': 21
        },
        SignFeatsType.i3d: 1024,
        SignFeatsType.CNN2d: 1024,
        SignFeatsType_TD.video: (720, 1280),
        SignFeatsType_TD.i3d: 1024,
        SignFeatsType_TD.CNN2d: 1024,
        SignFeatsType_TD.keypoints: {
            'face': 70,
            'upperbody': 8,
            'lowerbody': 16,
            'lefthand': 21,
            'righthand': 21
        },
        SignFeatsType_TD.mediapipe_keypoints: {
            'face': 70,
            'upperbody': 8,
            'lowerbody': 16,
            'lefthand': 21,
            'righthand': 21,
            'UPPER_BODY': 25, # TODO: This is the body minus the 
            'LEFT_HAND_LANDMARKS': 21, 
            'RIGHT_HAND_LANDMARKS': 21
        }, # TODO: The old one has lefthand and righthand but the new one has LEFT_HAND_LANDMARKS and RIGHT_HAND_LANDMARKS
        SignFeatsType_TD.rotational: 288,
        SignFeatsType_TD.mediapipe_rotational: 288,
        SignFeatsType_TD.text: 256,  # TODO: decide which dim to return, or if this function should be called at all when using text as input
        SignFeatsType_TD.text_albert: 768,
        SignFeatsType_TD.spot_align: 256,
        SignFeatsType_TD.spot_align_albert: 768,
        SignFeatsType_TD.mouthings: 256,
        SignFeatsType_TD.mouthings_albert: 768,
    }
    if (feats_type is SignFeatsType.i3d or
        feats_type is SignFeatsType.CNN2d or
        feats_type is SignFeatsType_TD.i3d or
        feats_type is SignFeatsType_TD.CNN2d or
        feats_type is SignFeatsType_TD.video or
        feats_type is SignFeatsType_TD.rotational or
        feats_type is SignFeatsType_TD.mediapipe_rotational or
        feats_type is SignFeatsType_TD.text or
        feats_type is SignFeatsType_TD.text_albert or
        feats_type is SignFeatsType_TD.spot_align or
        feats_type is SignFeatsType_TD.spot_align_albert or
        feats_type is SignFeatsType_TD.mouthings or
        feats_type is SignFeatsType_TD.mouthings_albert
        ):
        return num_feats[feats_type]
    elif feats_type in [SignFeatsType.openpose, SignFeatsType.mediapipe, SignFeatsType_TD.keypoints, SignFeatsType_TD.mediapipe_keypoints]:
        return sum([num_feats[feats_type][b] for b in bodyparts]) * len(feat_dims)
    else:
        raise AttributeError(f"Feat type selected not supported: {feats_type}")


def select_keypoints_by_bodypart(
        keypoints: torch.Tensor,
        feats_type: Union[SignFeatsType,SignFeatsType_TD],
        bodyparts: Optional[List[str]] = None,
        datasetType: str = 'How2Sign',
) -> Tuple[torch.Tensor, int]:
    if datasetType == 'Phoenix' or \
        (feats_type in SignFeatsType._value2member_map_ and SignFeatsType[feats_type] == SignFeatsType.mediapipe) or \
        (feats_type in SignFeatsType_TD._value2member_map_ and SignFeatsType_TD[feats_type] == SignFeatsType_TD.mediapipe_keypoints):
        if (feats_type in SignFeatsType_TD._value2member_map_ and SignFeatsType_TD[feats_type] == SignFeatsType_TD.mediapipe_keypoints) and bodyparts == ['lefthand', 'righthand']:
            #We need to select both hands, which were the indeces? We have 50 keypoints, since we have upper body, left hand and right hand (it should be 8 + 21 + 21)
            #the keypoints is a list of x,y,z, for each keypoint concatenated
            keypoints = keypoints[:, 8*3:]
            return keypoints.reshape(-1, 42*3).contiguous(), 42
        else:
            return keypoints.reshape(-1, 50*3).contiguous(), 50

    # TODO: double check that this is the order of the bodyparts, but the .pose library already handles this.
    BODY_IDX = {
        'face': torch.arange(70),           # 0-69
        'upperbody': torch.arange(70,78),   # 70-78
        'lowerbody': torch.arange(78,95),   # 79-94
        'lefthand': torch.arange(95,116),   # 95-115
        'righthand': torch.arange(116,137)  # 116-136
    }

    import pdb; pdb.set_trace()
    if bodyparts is None:
        bodyparts = list(BODY_IDX.keys())

    assert len(bodyparts) > 0, "You haven't selected any bodypart!"
    assert all([b in BODY_IDX.keys() for b in bodyparts]), f"You have selected a bodypart that doesn't exist! The options are: {list(BODY_IDX.keys())}"

    selected_idx = torch.cat([BODY_IDX[b] for b in bodyparts])

    keypoints = keypoints.reshape(-1, 137, 4)
    keypoints_selected = keypoints[:, selected_idx]
    keypoints = keypoints_selected.reshape(-1, len(selected_idx) * 4).contiguous()

    return keypoints, len(selected_idx)


def select_keypoints_by_dimension(
        keypoints: torch.Tensor,
        dimensions: List[int],
        feats_type: Union[SignFeatsType,SignFeatsType_TD],
        datasetType: str = 'How2Sign',
) -> torch.Tensor:
    assert len(dimensions) > 0, "You haven't selected any dimensions!"
    assert all([idx<4 for idx in dimensions]), "You have selected a dimension that doesn't exist! The options are: 0 for x, 1 for y, 2 for z and 3 for confidence score "
    if datasetType == 'Phoenix' or \
        (feats_type in SignFeatsType._value2member_map_ and SignFeatsType[feats_type] == SignFeatsType.mediapipe) or \
        (feats_type in SignFeatsType_TD._value2member_map_ and SignFeatsType_TD[feats_type] == SignFeatsType_TD.mediapipe_keypoints):
        
        if (feats_type in SignFeatsType_TD._value2member_map_ and SignFeatsType_TD[feats_type] == SignFeatsType_TD.mediapipe_keypoints) and dimensions == [0,1]:
            #We need to select the 0 and 1
            n_keypoints = int(keypoints.size(-1) / 3)
            keypoints = keypoints.reshape(-1, n_keypoints, 3)
            selected_idx = torch.LongTensor(dimensions)
            keypoints_selected = keypoints[:, :, selected_idx]
            keypoints = keypoints_selected.reshape(-1, n_keypoints * len(selected_idx)).contiguous() 
            return keypoints
        else:
            return keypoints.reshape(-1, 50*3).contiguous()

    selected_idx = torch.LongTensor(dimensions)

    n_keypoints = int(keypoints.size(-1) / 4)
    keypoints = keypoints.reshape(-1, n_keypoints, 4)
    keypoints_selected = keypoints[:, :, selected_idx]
    keypoints = keypoints_selected.reshape(-1, n_keypoints * len(selected_idx)).contiguous()  

    return keypoints