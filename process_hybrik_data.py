"""
From: https://github.com/ZhengyiLuo/EmbodiedPose/blob/master/scripts/process_hybrik_data.py
All credit to author.
"""

import argparse
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import numpy as np
import pickle as pk
from scipy.spatial.transform import Rotation as sRot
import matplotlib.pyplot as plt
import joblib
from uhc.smpllib.smpl_parser import SMPL_Parser, SMPLH_BONE_ORDER_NAMES, SMPLH_Parser
import torch


def smpl_op_to_op(pred_joints2d):
    new_2d = np.concatenate(
        [
            pred_joints2d[..., [1, 4], :].mean(axis=-2, keepdims=True),
            pred_joints2d[..., 1:8, :],
            pred_joints2d[..., 9:11, :],
            pred_joints2d[..., 12:, :],
        ],
        axis=-2,
    )
    return new_2d


MUJOCO_2_SMPL = np.array(
    [
        0,
        1,
        5,
        9,
        2,
        6,
        10,
        3,
        7,
        11,
        4,
        8,
        12,
        14,
        19,
        13,
        15,
        20,
        16,
        21,
        17,
        22,
        18,
        23,
    ]
)
SMPL_2_OP = np.array(
    [
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
)  # 25 joints -> 14 joints
OP_14_to_OP_12 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        False,
        True,
        True,
    ]
)


SMPL_JOINTS = {
    "hips": 0,
    "leftUpLeg": 1,
    "rightUpLeg": 2,
    "spine": 3,
    "leftLeg": 4,
    "rightLeg": 5,
    "spine1": 6,
    "leftFoot": 7,
    "rightFoot": 8,
    "spine2": 9,
    "leftToeBase": 10,
    "rightToeBase": 11,
    "neck": 12,
    "leftShoulder": 13,
    "rightShoulder": 14,
    "head": 15,
    "leftArm": 16,
    "rightArm": 17,
    "leftForeArm": 18,
    "rightForeArm": 19,
    "leftHand": 20,
    "rightHand": 21,
}
SMPL_PARENTS = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    12,
    12,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
]

# SMPLH_PATH = './copycat/humor/body_model/smplh'
# SMPLX_PATH = './copycat/body_models/smplx'
# SMPL_PATH = './copycat/body_models/smpl'
# VPOSER_PATH = './copycat/body_models/vposer_v1_0'

SMPLH_PATH = "UniversalHumanoidControl/data/smpl/smplh"
SMPLX_PATH = "UniversalHumanoidControl/data/smpl/smplx"
SMPL_PATH = "UniversalHumanoidControl/data/smpl/"

# chosen virtual mocap markers that are "keypoints" to work with
KEYPT_VERTS = [
    4404,
    920,
    3076,
    3169,
    823,
    4310,
    1010,
    1085,
    4495,
    4569,
    6615,
    3217,
    3313,
    6713,
    6785,
    3383,
    6607,
    3207,
    1241,
    1508,
    4797,
    4122,
    1618,
    1569,
    5135,
    5040,
    5691,
    5636,
    5404,
    2230,
    2173,
    2108,
    134,
    3645,
    6543,
    3123,
    3024,
    4194,
    1306,
    182,
    3694,
    4294,
    744,
]


#
# From https://github.com/vchoutas/smplify-x/blob/master/smplifyx/utils.py
# Please see license for usage restrictions.
#
def smpl_to_openpose(
    model_type="smplx",
    use_hands=True,
    use_face=True,
    use_face_contour=False,
    openpose_format="coco25",
):
    """Returns the indices of the permutation that maps SMPL to OpenPose

    Parameters
    ----------
    model_type: str, optional
        The type of SMPL-like model that is used. The default mapping
        returned is for the SMPLX model
    use_hands: bool, optional
        Flag for adding to the returned permutation the mapping for the
        hand keypoints. Defaults to True
    use_face: bool, optional
        Flag for adding to the returned permutation the mapping for the
        face keypoints. Defaults to True
    use_face_contour: bool, optional
        Flag for appending the facial contour keypoints. Defaults to False
    openpose_format: bool, optional
        The output format of OpenPose. For now only COCO-25 and COCO-19 is
        supported. Defaults to 'coco25'

    """
    if openpose_format.lower() == "coco25":
        if model_type == "smpl":
            return np.array(
                [
                    24,
                    12,
                    17,
                    19,
                    21,
                    16,
                    18,
                    20,
                    0,
                    2,
                    5,
                    8,
                    1,
                    4,
                    7,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                ],
                dtype=np.int32,
            )
        elif model_type == "smplh":
            body_mapping = np.array(
                [
                    52,
                    12,
                    17,
                    19,
                    21,
                    16,
                    18,
                    20,
                    0,
                    2,
                    5,
                    8,
                    1,
                    4,
                    7,
                    53,
                    54,
                    55,
                    56,
                    57,
                    58,
                    59,
                    60,
                    61,
                    62,
                ],
                dtype=np.int32,
            )
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array(
                    [
                        20,
                        34,
                        35,
                        36,
                        63,
                        22,
                        23,
                        24,
                        64,
                        25,
                        26,
                        27,
                        65,
                        31,
                        32,
                        33,
                        66,
                        28,
                        29,
                        30,
                        67,
                    ],
                    dtype=np.int32,
                )
                rhand_mapping = np.array(
                    [
                        21,
                        49,
                        50,
                        51,
                        68,
                        37,
                        38,
                        39,
                        69,
                        40,
                        41,
                        42,
                        70,
                        46,
                        47,
                        48,
                        71,
                        43,
                        44,
                        45,
                        72,
                    ],
                    dtype=np.int32,
                )
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == "smplx":
            body_mapping = np.array(
                [
                    55,
                    12,
                    17,
                    19,
                    21,
                    16,
                    18,
                    20,
                    0,
                    2,
                    5,
                    8,
                    1,
                    4,
                    7,
                    56,
                    57,
                    58,
                    59,
                    60,
                    61,
                    62,
                    63,
                    64,
                    65,
                ],
                dtype=np.int32,
            )
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array(
                    [
                        20,
                        37,
                        38,
                        39,
                        66,
                        25,
                        26,
                        27,
                        67,
                        28,
                        29,
                        30,
                        68,
                        34,
                        35,
                        36,
                        69,
                        31,
                        32,
                        33,
                        70,
                    ],
                    dtype=np.int32,
                )
                rhand_mapping = np.array(
                    [
                        21,
                        52,
                        53,
                        54,
                        71,
                        40,
                        41,
                        42,
                        72,
                        43,
                        44,
                        45,
                        73,
                        49,
                        50,
                        51,
                        74,
                        46,
                        47,
                        48,
                        75,
                    ],
                    dtype=np.int32,
                )

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(
                    76, 127 + 17 * use_face_contour, dtype=np.int32
                )
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError("Unknown model type: {}".format(model_type))
    elif openpose_format == "coco19":
        if model_type == "smpl":
            return np.array(
                [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28],
                dtype=np.int32,
            )
        elif model_type == "smplh":
            body_mapping = np.array(
                [52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 53, 54, 55, 56],
                dtype=np.int32,
            )
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array(
                    [
                        20,
                        34,
                        35,
                        36,
                        57,
                        22,
                        23,
                        24,
                        58,
                        25,
                        26,
                        27,
                        59,
                        31,
                        32,
                        33,
                        60,
                        28,
                        29,
                        30,
                        61,
                    ],
                    dtype=np.int32,
                )
                rhand_mapping = np.array(
                    [
                        21,
                        49,
                        50,
                        51,
                        62,
                        37,
                        38,
                        39,
                        63,
                        40,
                        41,
                        42,
                        64,
                        46,
                        47,
                        48,
                        65,
                        43,
                        44,
                        45,
                        66,
                    ],
                    dtype=np.int32,
                )
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == "smplx":
            body_mapping = np.array(
                [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59],
                dtype=np.int32,
            )
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array(
                    [
                        20,
                        37,
                        38,
                        39,
                        60,
                        25,
                        26,
                        27,
                        61,
                        28,
                        29,
                        30,
                        62,
                        34,
                        35,
                        36,
                        63,
                        31,
                        32,
                        33,
                        64,
                    ],
                    dtype=np.int32,
                )
                rhand_mapping = np.array(
                    [
                        21,
                        52,
                        53,
                        54,
                        65,
                        40,
                        41,
                        42,
                        66,
                        43,
                        44,
                        45,
                        67,
                        49,
                        50,
                        51,
                        68,
                        46,
                        47,
                        48,
                        69,
                    ],
                    dtype=np.int32,
                )

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(
                    70, 70 + 51 + 17 * use_face_contour, dtype=np.int32
                )
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError("Unknown model type: {}".format(model_type))
    else:
        raise ValueError("Unknown joint format: {}".format(openpose_format))


def xyxy2xywh(bbox):  # from HybrIK
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def process_hybrik(input_file, output_file):
    data_dir = "data/smpl"
    smpl_parser_n = SMPL_Parser(model_path=data_dir, gender="neutral")
    smpl_parser_m = SMPL_Parser(model_path=data_dir, gender="male")
    smpl_parser_f = SMPL_Parser(model_path=data_dir, gender="female")

    smpl2op_map = smpl_to_openpose(
        "smpl",
        use_hands=False,
        use_face=False,
        use_face_contour=False,
        openpose_format="coco25",
    )
    smpl_2op_submap = smpl2op_map[smpl2op_map < 22]

    res_data = pk.load(open(input_file, "rb"))

    pose_mat = np.array(res_data["pred_thetas"])
    trans_orig = np.array(res_data["transl"]).squeeze()
    bbox = np.array(res_data["bbox"]).squeeze()

    B = pose_mat.shape[0]
    pose_aa = sRot.from_matrix(pose_mat.reshape(-1, 3, 3)).as_rotvec().reshape(B, -1)
    pose_aa_orig = pose_aa.copy()

    ## Apply the rotation to make z the up direction
    transform = sRot.from_euler("xyz", np.array([-np.pi / 2, 0, 0]), degrees=False)
    new_root = (transform * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec()
    pose_aa[:, :3] = new_root
    transform.as_matrix(), sRot.from_rotvec(pose_aa[0, :3]).as_matrix()
    trans = trans_orig.dot(transform.as_matrix().T)
    diff_trans = trans[0, 2] - 0.92
    trans[:, 2] = trans[:, 2] - diff_trans

    scale = (bbox[:, 2] - bbox[:, 0]) / 256
    trans[:, 1] = trans[:, 1] / scale
    beta = res_data["pred_betas"][0]

    kp_25 = np.zeros([B, 25, 3])
    kp_25_idxes = np.arange(25)[SMPL_2_OP][
        OP_14_to_OP_12
    ]  # wow this needs to get better...

    uv_29 = res_data["pred_uvd"][:, :24]
    pts_12 = smpl_op_to_op(uv_29[:, smpl_2op_submap, :])
    kp_25[:, kp_25_idxes] = pts_12

    for i in range(B):
        bbox_xywh = xyxy2xywh(bbox[i])
        kp_25[i] = kp_25[i] * bbox_xywh[2]
        kp_25[i, :, 0] = kp_25[i, :, 0] + bbox_xywh[0]
        kp_25[i, :, 1] = kp_25[i, :, 1] + bbox_xywh[1]
        kp_25[i, :, 2] = 1  # probability

    ### Assemblle camera
    idx = 0  # Only need to use the first frame, since after that Embodied pose will take over tracking.
    height, width = res_data["height"][idx], res_data["width"][idx]
    focal = 1000.0
    bbox_xywh = xyxy2xywh(bbox[idx])
    focal = focal / 256 * bbox_xywh[2]  # A little hacky
    focal = 2 * focal / min(height, width)

    full_R = sRot.from_euler("xyz", np.array([np.pi / 2, 0, 0])).as_matrix()
    full_t = np.array([0, -diff_trans, 0])
    K = np.array(
        [
            [
                res_data["height"][0] / 2 * (focal / scale[idx]),
                0,
                res_data["width"][0] / 2,
            ],
            [
                0.0,
                res_data["height"][0] / 2 * (focal / scale[idx]),
                res_data["height"][0] / 2,
            ],
            [0.0, 0, 1.0],
        ]
    )

    cam = {
        "full_R": full_R,
        "full_t": full_t,
        "K": K,
        "img_w": res_data["width"][0],
        "img_h": res_data["height"][0],
        "scene_name": None,
    }

    pose_mat = (
        sRot.from_rotvec(pose_aa.reshape(B * 24, 3)).as_matrix().reshape(B, 24, 3, 3)
    )
    pose_body = pose_mat[:, 1:22]
    root_orient = pose_mat[:, 0:1]
    new_dict = {}
    start = 0
    end = B
    key = "00"
    new_dict[key] = {
        "joints2d": kp_25[start:end].copy(),
        "pose_body": pose_body[start:end],
        "root_orient": root_orient[start:end],
        "trans": trans.squeeze()[start:end],
        "pose_aa": pose_aa.reshape(-1, 72)[start:end],
        "joints": np.zeros([B, 22, 3]),
        "seq_name": "01",
        "pose_6d": np.zeros([B, 24, 6]),
        "beta": beta,
        "gender": "neutral",
        "seq_name": key,
        "trans_vel": np.zeros([B, 1, 3]),
        "joints_vel": np.zeros([B, 22, 3]),
        "root_orient_vel": np.zeros([B, 1, 3]),
        "points3d": None,
        "cam": cam,
    }
    joblib.dump(new_dict, output_file)
