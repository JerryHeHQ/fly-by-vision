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

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

# Load image
image_path = "pngtree.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Pose detection
results = pose.process(image_rgb)
if not results.pose_landmarks:
    print("No pose detected!")
else:
    print("Pose detected!")

# Full keypoint visualization
output_image = image.copy()
if results.pose_landmarks:
    mp_drawing.draw_landmarks(output_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Helper: draw a ray between two landmarks
def draw_ray(blank, elbow_lm, wrist_lm, color):
    elbow_pt = np.array([elbow_lm.x * w, elbow_lm.y * h])
    wrist_pt = np.array([wrist_lm.x * w, wrist_lm.y * h])

    direction = wrist_pt - elbow_pt
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        direction /= norm
        ray_length = 150
        ray_end = elbow_pt + direction * ray_length
        cv2.line(blank, tuple(elbow_pt.astype(int)), tuple(ray_end.astype(int)), color, 4)

# Sparse keypoint visualization
if results.pose_landmarks:
    selected_indices = [11, 12, 23, 24, 13, 14, 15, 16]  # shoulders, hips, elbows, wrists
    filtered_landmarks = landmark_pb2.NormalizedLandmarkList()
    for idx in selected_indices:
        filtered_landmarks.landmark.append(results.pose_landmarks.landmark[idx])

    connections = [(0,1), (0,2), (1,3), (2,3), (0,4), (4,6), (1,5), (5,7)]
    h, w, _ = image.shape
    blank = image.copy()
    mp_drawing.draw_landmarks(blank, filtered_landmarks, connections)

    # Draw rays from elbows to wrists
    draw_ray(blank, filtered_landmarks.landmark[4], filtered_landmarks.landmark[6], (255, 0, 0))
    draw_ray(blank, filtered_landmarks.landmark[5], filtered_landmarks.landmark[7], (0, 255, 0))

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# Print 2D coordinates
if results.pose_landmarks:
    input_2d = []
    for i, lm in enumerate(results.pose_landmarks.landmark):
        print(f"Landmark {i}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")
        input_2d.extend([lm.x, lm.y])


# 3D plot helpers
def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    x_limits, y_limits, z_limits = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    x_range, y_range, z_range = x_limits[1]-x_limits[0], y_limits[1]-y_limits[0], z_limits[1]-z_limits[0]
    max_range = max(x_range, y_range, z_range) / 2
    x_mid, y_mid, z_mid = np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)
    ax.set_xlim3d(x_mid-max_range, x_mid+max_range)
    ax.set_ylim3d(y_mid-max_range, y_mid+max_range)
    ax.set_zlim3d(z_mid-max_range, z_mid+max_range)

def apply_zoom(ax, xs, ys, zs, zoom=0.6):
    """Zoom into the 3D plot around the skeleton center."""
    x_center, y_center, z_center = np.mean(xs), np.mean(ys), np.mean(zs)
    x_range = (max(xs)-min(xs)) * zoom / 2
    y_range = (max(ys)-min(ys)) * zoom / 2
    z_range = (max(zs)-min(zs)) * zoom / 2
    ax.set_xlim(x_center-x_range, x_center+x_range)
    ax.set_ylim(y_center-y_range, y_center+y_range)
    ax.set_zlim(z_center-z_range, z_center+z_range)

# Full 3D pose visualization
if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark
    xs, ys, zs = [lm.x for lm in landmarks], [lm.y for lm in landmarks], [lm.z for lm in landmarks]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c='r', s=50)
    for start_idx, end_idx in mp_pose.POSE_CONNECTIONS:
        ax.plot([landmarks[start_idx].x, landmarks[end_idx].x],
                [landmarks[start_idx].y, landmarks[end_idx].y],
                [landmarks[start_idx].z, landmarks[end_idx].z], c='b')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('3D Pose Landmarks')
    ax.view_init(elev=-90, azim=-90)
    apply_zoom(ax, xs, ys, zs, zoom=0.4)
    set_axes_equal(ax)
    plt.show()

# Sparse 3D pose visualization with rays
if results.pose_landmarks:
    sparse_landmarks = [landmarks[i] for i in selected_indices]
    xs, ys, zs = [lm.x for lm in sparse_landmarks], [lm.y for lm in sparse_landmarks], [lm.z for lm in sparse_landmarks]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c='r', s=50)

    sparse_connections = [(0,1), (0,2), (1,3), (2,3), (0,4), (4,6), (1,5), (5,7)]
    for start_idx, end_idx in sparse_connections:
        ax.plot([sparse_landmarks[start_idx].x, sparse_landmarks[end_idx].x],
                [sparse_landmarks[start_idx].y, sparse_landmarks[end_idx].y],
                [sparse_landmarks[start_idx].z, sparse_landmarks[end_idx].z], c='gray')

    # Rays from elbows to wrists
    ax.plot([sparse_landmarks[4].x, sparse_landmarks[6].x],
            [sparse_landmarks[4].y, sparse_landmarks[6].y],
            [sparse_landmarks[4].z, sparse_landmarks[6].z], c='green', linewidth=2)
    ax.plot([sparse_landmarks[5].x, sparse_landmarks[7].x],
            [sparse_landmarks[5].y, sparse_landmarks[7].y],
            [sparse_landmarks[5].z, sparse_landmarks[7].z], c='blue', linewidth=2)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Sparse 3D Pose with Rays')
    ax.view_init(elev=-90, azim=-90)
    apply_zoom(ax, xs, ys, zs, zoom=0.4)
    set_axes_equal(ax)
    plt.show()
