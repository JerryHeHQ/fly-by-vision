# conda create -n smpl-env python=3.10
# conda activate smpl-env
# conda install numpy scipy chumpy
# conda install pytorch torchvision torchaudio
# pip install smplx trimesh "pyglet<2"

import torch
import smplx
import trimesh
import math

JOINT_NAMES = [
    # "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
]

model_path = r"../smplx_models"

smplx_model = smplx.create(
    model_path=model_path,
    model_type="smplx",
    gender="neutral",
    use_pca=False,
    batch_size=1,
    create_global_orient=True,
    create_body_pose=True,
    create_left_hand_pose=True,
    create_right_hand_pose=True,
    create_betas=True,
    create_expression=True,
)

body_pose = torch.zeros((1, smplx_model.NUM_BODY_JOINTS * 3))
left_hand_pose = torch.zeros((1, smplx_model.NUM_HAND_JOINTS * 3))
right_hand_pose = torch.zeros((1, smplx_model.NUM_HAND_JOINTS * 3))
betas = torch.zeros((1, smplx_model.betas.shape[-1]))
global_orient = torch.zeros((1, 3))
transl = torch.zeros((1, 3))
expression = torch.zeros((1, smplx_model.num_expression_coeffs))

# Position ray direction
left_shoulder_idx = JOINT_NAMES.index("left_shoulder")
right_shoulder_idx = JOINT_NAMES.index("right_shoulder")
body_pose[0, left_shoulder_idx * 3 + 2] = math.radians(-75)
body_pose[0, right_shoulder_idx * 3 + 1] = math.radians(60)
body_pose[0, right_shoulder_idx * 3 + 2] = math.radians(-30)

# Pointing gesture
right_hand_pose[0, 0 * 3 + 2] = math.radians(-30)
right_hand_pose[0, 1 * 3 + 2] = math.radians(-30)
right_hand_pose[0, 2 * 3 + 2] = math.radians(-15)

curl_angle_1 = math.radians(75)
curl_angle_2 = math.radians(30)
curl_angle_3 = math.radians(90)

right_hand_pose[0, 3 * 3 + 0] = math.radians(-15)
right_hand_pose[0, 3 * 3 + 2] = curl_angle_1
right_hand_pose[0, 4 * 3 + 2] = curl_angle_2
right_hand_pose[0, 5 * 3 + 2] = curl_angle_3

right_hand_pose[0, 6 * 3 + 0] = math.radians(-45)
right_hand_pose[0, 6 * 3 + 2] = curl_angle_1
right_hand_pose[0, 7 * 3 + 2] = curl_angle_2
right_hand_pose[0, 8 * 3 + 0] = math.radians(-45)
right_hand_pose[0, 8 * 3 + 2] = curl_angle_3

right_hand_pose[0, 9 * 3 + 0] = math.radians(-30)
right_hand_pose[0, 9 * 3 + 2] = curl_angle_1
right_hand_pose[0, 10 * 3 + 2] = curl_angle_2
right_hand_pose[0, 11 * 3 + 2] = curl_angle_3

right_hand_pose[0, 12 * 3 + 0] = math.radians(60)
right_hand_pose[0, 12 * 3 + 1] = math.radians(-15)
right_hand_pose[0, 13 * 3 + 0] = math.radians(30)
right_hand_pose[0, 14 * 3 + 1] = math.radians(-30)

output = smplx_model(
    body_pose=body_pose,
    left_hand_pose=left_hand_pose,
    right_hand_pose=right_hand_pose,
    betas=betas,
    global_orient=global_orient,
    transl=transl,
    expression=expression
)

vertices = output.vertices.detach().cpu().numpy()[0]
faces = smplx_model.faces

mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
scene = trimesh.Scene(mesh)
scene.show()

mesh.export("../smplx_models/smplx_point_pose.obj")