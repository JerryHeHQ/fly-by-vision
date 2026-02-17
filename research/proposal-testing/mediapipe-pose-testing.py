# conda create -n mediapipe-pose-env python=3.10 -y
# conda activate mediapipe-pose-env
# pip uninstall mediapipe -y
# pip uninstall mediapipe -y
# pip install --no-cache-dir mediapipe==0.10.9
# pip install opencv-python matplotlib numpy

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mediapipe.framework.formats import landmark_pb2
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

image_path = "pngtree.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = pose.process(image_rgb)
if not results.pose_landmarks:
    print("No pose detected!")
else:
    print("Pose detected!")

# Full Keypoint Image
output_image = image.copy()
if results.pose_landmarks:
    mp_drawing.draw_landmarks(
        output_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )

plt.figure(figsize=(8,8))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Ray drawing helper
def draw_ray(blank, elbow_lm, wrist_lm, color):
    elbow_pt = np.array([elbow_lm.x * w, elbow_lm.y * h])
    wrist_pt = np.array([wrist_lm.x * w, wrist_lm.y * h])

    direction = wrist_pt - elbow_pt
    norm = np.linalg.norm(direction)

    if norm > 1e-6:
        direction = direction / norm
        ray_length = 150

        ray_end = elbow_pt + direction * ray_length

        elbow_pt_int = tuple(elbow_pt.astype(int))
        ray_end_int = tuple(ray_end.astype(int))

        cv2.line(blank, elbow_pt_int, ray_end_int, color, 4)

# Sparse Keypoint Image
if results.pose_landmarks:
    selected_indices = [11, 12, 23, 24, 13, 14, 15, 16]

    filtered_landmarks = landmark_pb2.NormalizedLandmarkList()
    for idx in selected_indices:
        filtered_landmarks.landmark.append(
            results.pose_landmarks.landmark[idx]
        )

    connections = [
        (0,1),
        (0,2),
        (1,3),
        (2,3),
        (0,4), (4,6),
        (1,5), (5,7)
    ]

    h, w, _ = image.shape
    blank = image.copy()

    mp_drawing.draw_landmarks(
        blank,
        filtered_landmarks,
        connections
    )

    # 4 = left elbow, 6 = left wrist
    # 5 = right elbow, 7 = right wrist

    draw_ray(blank,
             filtered_landmarks.landmark[4],
             filtered_landmarks.landmark[6],
             (255, 0, 0))

    draw_ray(blank,
             filtered_landmarks.landmark[5],
             filtered_landmarks.landmark[7],
             (0, 255, 0))

    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

if results.pose_landmarks:
    for i, lm in enumerate(results.pose_landmarks.landmark):
        print(f"Landmark {i}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")
