import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np

from ..typing import VideoArray, ArticulatorArray
from typing import Tuple

def track_hands(video: VideoArray, fps=25) -> Tuple[ArticulatorArray, int]:
    _, v_len, __, v_height, v_width = video.shape

    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path='data/pose_landmarker.task'),
        running_mode=vision.RunningMode.VIDEO)
    detector = vision.PoseLandmarker.create_from_options(options)

    video = video[0].transpose(0, 2, 3, 1).astype(np.uint8)
    hand_track = np.zeros((2, v_len, 4), dtype=float)
    for n in range(v_len):
        frame = video[n]
        image = mp.Image(mp.ImageFormat.SRGB, frame)
        detection = detector.detect_for_video(image, n*fps)
        lwrist = detection.pose_landmarks[0][15]
        rwrist = detection.pose_landmarks[0][16]
        hand_track[0, n, :2] = [rwrist.x * v_width, rwrist.y * v_height]
        hand_track[1, n, :2] = [lwrist.x * v_width, lwrist.y * v_height]

    return hand_track, 0

