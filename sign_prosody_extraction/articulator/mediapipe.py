import os
from typing import Tuple

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

from .. import cache
from ..typing import ArticulatorArray, VideoArray
from . import compute_speed

detector = None

POSE_LANDMARKER = os.getenv("POSE_LANDMARKER", "data/pose_landmarker.task")
if not os.path.exists(POSE_LANDMARKER):
    POSE_LANDMARKER = "pose_landmarker.task"
if not os.path.exists(POSE_LANDMARKER):
    raise FileNotFoundError(
        "PoseLandmarker model not found, please download it first "
        "(https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models)"
    )


@cache
def track_hands(video: VideoArray, fps=25) -> Tuple[ArticulatorArray, int]:
    global detector
    if not detector:
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=POSE_LANDMARKER,
            ),
            running_mode=vision.RunningMode.VIDEO,
        )
        detector = vision.PoseLandmarker.create_from_options(options)

    _, v_len, __, v_height, v_width = video.shape

    video = video[0].transpose(0, 2, 3, 1).astype(np.uint8)
    hand_tracks = np.zeros((2, v_len, 2), dtype=float)
    for n in range(v_len):
        frame = video[n].copy()
        image = mp.Image(mp.ImageFormat.SRGB, frame)
        detection = detector.detect_for_video(image, n * fps)
        landmarks = detection.pose_landmarks[0]
        hand_tracks[0, n, :2] = average_hand(landmarks, (15, 17, 19), v_width, v_height)
        hand_tracks[1, n, :2] = average_hand(landmarks, (16, 18, 20), v_width, v_height)
    return compute_speed(hand_tracks, window_length=14), 0


def average_hand(all_landmarks, indices, v_width, v_height):
    xs = [all_landmarks[i].x * v_width for i in indices]
    ys = [all_landmarks[i].y * v_height for i in indices]
    return [sum(xs) / len(xs), sum(ys) / len(ys)]
