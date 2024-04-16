from joblib import Memory
import imageio.v3 as iio
import os
import numpy as np

from .typing import VideoArray

if not os.getenv("NO_CACHE", False):
    memory = Memory(".joblib", verbose=0, bytes_limit="500M")

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
