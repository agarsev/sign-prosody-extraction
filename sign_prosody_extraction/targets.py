import numpy as np
from scipy.signal import savgol_filter


def get_target_points(track):
    dt = savgol_filter(track[:, 2], window_length=5, polyorder=2, deriv=1)
    signs = np.sign(dt)
    minima = ((signs[:-1] < 0) & (signs[1:] > 0)).nonzero()[0]
    maxima = ((signs[:-1] > 0) & (signs[1:] < 0)).nonzero()[0]
    #  minima between first and last maxima (preparation and relaxation)
    return np.array([m for m in minima if m > maxima[0] and m < maxima[-1]])
