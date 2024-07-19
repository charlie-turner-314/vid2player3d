import sys

import os
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from smpl_visualizer.vis_sport import SportVisualizer
from smpl_visualizer.smpl import SMPL, SMPL_MODEL_DIR
from utils.racket import infer_racket_from_smpl


def visualize_motion(
    num_test, nframes, joint_rot_all, joint_pos_all, betas, trans_all, interactive=False
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
        print(joint_rot_all.shape)

        # joint_pos_all = joint_pos_all.reshape(-1, nframes, 24, 3)
        joint_rot_all = joint_rot_all.reshape(-1, nframes, 24, 3)
        trans_all = trans_all.reshape(-1, nframes, 3)

        joint_rot_all[..., -1, :] = torch.FloatTensor([0, 0, np.pi / 2])

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
        vid_path = os.path.join(result_sub_dir, "random_front.mp4")
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
                window_size=(1000, 1000),
                enable_shadow=True,
                cleanup=True,
            )


# Example usage
# Make sure to pass your arrays `joint_rot_all`, `joint_pos_all`, `betas`, `trans_all`
# # visualize_motion(1, joint_rot_all.shape[0], joint_rot_all, joint_pos_all, betas, trans_all)
joint_rot_all = np.load(
    "/home/charlie/Documents/ATPIL/Training/vid2player3d/tennis_data/file_res_p0_h1_joint_rot.npy"
).reshape(1, -1, 24, 3)
joint_pos_all = np.load(
    "/home/charlie/Documents/ATPIL/Training/vid2player3d/tennis_data/file_res_p0_h1_joint_pos.npy"
).reshape(1, -1, 24, 3)
betas = np.zeros((10), dtype=np.float32)
# trans is just the root position
trans_all = joint_pos_all[:, :, 0, :]
# print(trans_all[:, :70, ...].round(2))
# print(joint_rot_all.shape)
# print(joint_rot_all[:, :70, 1, :].round(2))


# Convert joint_quat_all to joint_rot_all (quaternion to axis-angle)
# print(joint_quat_all.shape)
# joint_rot_all = np.zeros((1, joint_quat_all.shape[0], 24, 3))
# for i in range(joint_quat_all.shape[0]):
#     for j in range(joint_quat_all.shape[1]):
#         r = R.from_quat(joint_quat_all[i, j])
#         joint_rot_all[0, i, j] = r.as_rotvec()
# # print(joint_rot_all.shape)

# print(joint_rot_all[:, :70, 1, :].round(2))


# # # # Option 2: Open original data
# import joblib

# with open(
#     "/home/charlie/Documents/Kyrgios_Medvedev_2022/processed/processed_data.pkl", "rb"
# ) as f:
#     data = joblib.load(f)

# data = data["file_res_p0_h1"]

# # # we have trans, pose_aa, betas
# trans_all = data['trans'].reshape(1, -1, 3)
# joint_rot_all = data["pose_aa"].reshape(1, -1, 24, 3)
# betas = data['beta']
# joint_pos_all = np.ndarray([])
# print("*" * 50)
# print(trans_all.round(2))
# print(joint_rot_all[:, :70, 1, :].round(2))

# exit()
# print("trans_all:", trans_all.shape)

# print("Shapes:")
# print("joint_rot:", joint_rot_all.shape)
# print("betas:", betas.shape)
# print("joint_pos_all:", joint_pos_all.shape)

# # visualise the rotations as an animation
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# for i in range(joint_rot_all.shape[1]):
#     ax.cla()
#     # as angle-axis
#     ax.quiver(
#         np.zeros(24),
#         np.zeros(24),
#         np.zeros(24),
#         joint_rot_all[0, i, :, 0],
#         joint_rot_all[0, i, :, 1],
#         joint_rot_all[0, i, :, 2],
#     )
#     # as euler angles
#     ax.set_xlim([-1, 1])
#     ax.set_ylim([-1, 1])
#     ax.set_zlim([-1, 1])
#     plt.pause(0.1)
#     plt.show()


# exit()


visualize_motion(
    1,
    joint_rot_all.shape[1],
    torch.from_numpy(joint_rot_all).float(),
    torch.from_numpy(joint_pos_all),
    torch.from_numpy(betas).float(),
    torch.from_numpy(trans_all).float(),
)



# Visualise the joint_pos in 3d for the first frame

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# frame = 170

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(joint_pos_all[0, frame, :, 0], joint_pos_all[0, frame, :, 1], joint_pos_all[0, frame, :, 2])

# # annotate 
# for i in range(joint_pos_all.shape[2]):
#     ax.text(joint_pos_all[0, frame, i, 0], joint_pos_all[0, frame, i, 1], joint_pos_all[0, frame, i, 2], str(i))

# lines = [
#     (0, 1, 2, 3, 4), # left leg
#     (0, 5, 6, 7, 8), # right leg
#     (0, 9, 10, 11, 12), # spine
#     (11, 14, 15, 16), # left arm
#     (11, 19, 20, 21), # right arm
#     (12, 13, 22, 23, 18, 17, 13), # head
# ]

# for line in lines:
#     ax.plot(
#         joint_pos_all[0, frame, line, 0],
#         joint_pos_all[0, frame, line, 1],
#         joint_pos_all[0, frame, line, 2],
#         color="black",
#     )


# plt.show()

