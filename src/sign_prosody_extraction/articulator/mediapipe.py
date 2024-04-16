import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np

from ..typing import VideoArray, ArticulatorArray
from typing import Tuple
from . import compute_speed
from .. import cache


detector = None


@cache
def track_hands(video: VideoArray, fps=25) -> Tuple[ArticulatorArray, int]:
    global detector
    if not detector:
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path="data/pose_landmarker.task"
            ),
            running_mode=vision.RunningMode.VIDEO,
        )
        detector = vision.PoseLandmarker.create_from_options(options)

    _, v_len, __, v_height, v_width = video.shape

    video = video[0].transpose(0, 2, 3, 1).astype(np.uint8)
    hand_tracks = np.zeros((2, v_len, 2), dtype=float)
    for n in range(v_len):
        frame = video[n]
        image = mp.Image(mp.ImageFormat.SRGB, frame)
        detection = detector.detect_for_video(image, n * fps)
        lwrist = detection.pose_landmarks[0][15]
        rwrist = detection.pose_landmarks[0][16]
        hand_tracks[0, n, :2] = [rwrist.x * v_width, rwrist.y * v_height]
        hand_tracks[1, n, :2] = [lwrist.x * v_width, lwrist.y * v_height]

    return compute_speed(hand_tracks, window_length=14), 0
