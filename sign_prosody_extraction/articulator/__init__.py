import numpy as np
from scipy.signal import savgol_filter

from ..typing import TrackXYArray, TrackXYVTArray


def compute_speed(tracks: TrackXYArray, window_length) -> TrackXYVTArray:
    dx = savgol_filter(
        tracks[:, :, 0], window_length=window_length, polyorder=5, deriv=1, axis=1
    )
    dy = savgol_filter(
        tracks[:, :, 1], window_length=window_length, polyorder=5, deriv=1, axis=1
    )
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    return np.concatenate([tracks, r[:, :, None], theta[:, :, None]], axis=2)
