"""
CHARLIE CODE - Visualise from processed SMPL format data
"""


import sys

import os
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import json

from smpl_visualizer.vis_sport import SportVisualizer
from smpl_visualizer.smpl import SMPL, SMPL_MODEL_DIR
from utils.racket import infer_racket_from_smpl


def visualize_motion(
    num_test, nframes, joint_rot_all, joint_pos_all, betas, trans_all, events, filename="motionvis.mp4", interactive=False,
):
    """
    Visualize motion using the provided body rotations, positions, and betas.
    """
    result_dir = os.path.join(".", "kyrgios")
    print("Save video results to {}".format(result_dir))

    visualizer = SportVisualizer(
        verbose=False,
        show_smpl=True,
        show_skeleton=False,
        show_racket=True,
        correct_root_height=True,
        gender="male",
    )

    if True:
        smpl = SMPL(SMPL_MODEL_DIR, create_transl=False, gender="male")

    for tid in range(num_test):

        result_sub_dir = os.path.join(result_dir, "{:03}".format(tid + 1))
        os.makedirs(result_sub_dir, exist_ok=True)
        print("Running test", tid + 1)

        joint_pos_all = joint_pos_all.reshape(-1, nframes, 24, 3)
        joint_rot_all = joint_rot_all.reshape(-1, nframes, 24, 3)
        trans_all = trans_all.reshape(-1, nframes, 3)

        # joint_rot_all[..., -1, :] = torch.FloatTensor([0, 0, np.pi / 2])

        betas=torch.zeros((nframes, 10)).float()

        if True:
            smpl_motion = smpl(
                global_orient=joint_rot_all[:, :, 0].reshape(-1, 3),
                body_pose=joint_rot_all[:, :, 1:].reshape(-1, 69),
                betas=torch.zeros((nframes, 10)).float(),
                root_trans=trans_all.reshape(-1, 3),
                return_full_pose=True,
                orig_joints=True,
            )
            print("Shapes")
            print(smpl_motion.joints.reshape(1, nframes, 24, 3).shape)
            # print(trans_all.reshape(1, nframes, 1, 3).shape)
            joint_pos_all = smpl_motion.joints.reshape(
                1, nframes, 24, 3
            ) - trans_all.reshape(1, nframes, 1, 3)
            racket_all = []
            racket_all.append([])
            for i in range(nframes):
                racket_all[0] += [
                    infer_racket_from_smpl(
                        joint_pos_all[0][i].numpy(),
                        joint_rot_all[0][i].numpy(),
                        sport="tennis",
                        righthand=True,
                    )
                ]

        smpl_seq = {
            "trans": trans_all,
            "orient": None,
            "betas": betas,
        }
        if False:
            smpl_seq["joint_pos"] = joint_pos_all
        else:
            smpl_seq["joint_rot"] = joint_rot_all.view(1, nframes, 24 * 3)

        init_args = {
            "smpl_seq": smpl_seq,
            "num_actors": 1,
            "sport": "tennis",
            "camera": "front",
            "racket_seq": racket_all,
        }
        vid_path = os.path.join(result_sub_dir, filename)
        if interactive:
            visualizer.show_animation(
                init_args=init_args,
                fps=30,
                window_size=(1000, 1000),
                enable_shadow=True,
            )
        else:
            visualizer.save_animation_as_video(
                vid_path,
                init_args=init_args,
                fps=30,
                window_size=(1920, 1080),
                enable_shadow=True,
                cleanup=True,
            )
        # get the video, add a flash at each event
        print("Adding events")
        event_frames = [e["frame"] for e in events["events"] if e["label"] == "near_court_swing"]
        # Open the original video
        cap = cv2.VideoCapture(vid_path)
        WIDTH, HEIGHT = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('tmp.mp4', fourcc, 30.0, (WIDTH, HEIGHT))

        frame_idx = 0
        event_set = set(event_frames)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in event_set:
                print("event")
                # Flash the screen green
                frame[:] = [0, 255, 0]

            assert frame.shape[1] == WIDTH and frame.shape[0] == HEIGHT, "Frame dimensions do not match the specified video dimensions."
            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        # Overwrite the original video with the new video
        os.replace('tmp.mp4', vid_path.replace(".mp4", "_events.mp4"))



# Example usage
# Make sure to pass your arrays `joint_rot_all`, `joint_pos_all`, `betas`, `trans_all`
# # visualize_motion(1, joint_rot_all.shape[0], joint_rot_all, joint_pos_all, betas, trans_all)
# files = os.listdir("/home/charlie/Documents/ATPIL/Training/vid2player3d/tennis_data")
# clips = set([tuple(f.split("_")[2:4]) for f in files])
# clips = sorted(list(clips))
# clips = clips[1:]
# for point, hit in clips:
#     print(f"Processing {point}_{hit}", end="... ")
#     joint_rot_all = np.load(
#         f"/home/charlie/Documents/ATPIL/Training/vid2player3d/tennis_data/file_res_{point}_{hit}_joint_rot.npy"
#     ).reshape(1, -1, 24, 3)
#     joint_pos_all = np.load(
#         f"/home/charlie/Documents/ATPIL/Training/vid2player3d/tennis_data/file_res_{point}_{hit}_joint_pos.npy"
#     ).reshape(1, -1, 24, 3)
#     betas = np.zeros((10), dtype=np.float32)
#     # trans is just the root position
#     trans_all = joint_pos_all[:, :, 0, :]


#     visualize_motion(
#         1,
#         joint_rot_all.shape[1],
#         torch.from_numpy(joint_rot_all).float(),
#         torch.from_numpy(joint_pos_all),
#         torch.from_numpy(betas).float(),
#         torch.from_numpy(trans_all).float(),
#         filename=f"{point}_{hit}.mp4"
#     )
#     print("Done.")
import joblib
import numpy as np
from motion_vae import dataset
from motion_vae.config import *

opt = MotionVAEOption()
opt.load("kyrgios")
data = dataset.Video3DPoseDataset(opt)
item = data[0]
joint_rot = item["joint_rot"]
trans = item["trans"]
visualize_motion(
    1, len(joint_rot), joint_rot, None, None, trans, None, "mvae"
)
print(item)
exit()



# pick 20 random motions


for key in random_keys:
    vid_name = key.replace("file_res_", "")
    # get the events
    event_file = f"/home/charlie/Documents/aux/sequence_hits/{vid_name}.json"
    with open(event_file, "r") as f:
        events = json.load(f)
    print(motion_dict[key].keys())
    trans_all = motion_dict[key]["trans"]
    joint_rot_all = motion_dict[key]["pose_aa"]
    print(joint_rot_all.shape)
    joint_pos_all = motion_dict[key]["pose_aa"]
    betas = motion_dict[key]["beta"]
    visualize_motion(
        1,
        joint_rot_all.shape[0],
        torch.from_numpy(joint_rot_all).float(),
        torch.from_numpy(joint_pos_all),
        torch.from_numpy(betas).float(),
        torch.from_numpy(trans_all).float(),
        events,
        filename=f"video/{key}.mp4",

    )