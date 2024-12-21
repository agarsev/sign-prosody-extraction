import os

import imageio.v3 as iio
import numpy as np
from joblib import Memory

from . import plot
from .typing import VideoArray

CACHE_DIR = os.getenv("CACHE_DIR")

if CACHE_DIR:
    memory = Memory(CACHE_DIR, verbose=0, bytes_limit=os.getenv("CACHE_LIMIT", "500M"))

    def cache(func):
        return memory.cache(func)

else:

    def cache(func):
        return func


@cache
def load_video(video_file: str) -> VideoArray:
    frames = iio.imread(str(video_file), plugin="FFMPEG")
    video = np.transpose(frames.astype(np.float32), (0, 3, 1, 2))
    return video[None, :]


def get_tracker(algorithm: str):
    if algorithm == "mediapipe":
        from .articulator.mediapipe import track_hands
    else:
        from .articulator.cotracker import track_hands
    return track_hands
