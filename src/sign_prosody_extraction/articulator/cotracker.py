from typing import Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans

from .. import cache
from ..typing import ArticulatorArray, TrackXYArray, TrackXYVTArray, VideoArray
from . import compute_speed


def track_hands(video: VideoArray) -> Tuple[ArticulatorArray, int]:
    _, v_len, __, v_height, v_width = video.shape

    # Find first frame where hands are inside of frame
    # track backward
    flipped = np.flip(video, axis=1).copy()
    backtrack = track_movement(flipped, start_point=v_len // 2)
    # Get last frame where the y coordinate is lower than v_height
    out_of_bounds = np.where(backtrack[:, 1] > v_height * 0.9)[0]
    first_frame = v_len - out_of_bounds[0] if len(out_of_bounds) else 0

    return (
        track_movement(video, start_point=first_frame)[None, first_frame:],
        first_frame,
    )


def track_movement(video: VideoArray, start_point) -> ArticulatorArray:
    tracks_xy = track_video_online(video, start=start_point)
    tracks_xyra = compute_speed(tracks_xy, window_length=7)
    fg = separate_fg(tracks_xyra)
    return np.mean(fg, axis=0)


cotracker = torch.hub.load(
    "facebookresearch/co-tracker", "cotracker3_online", verbose=False
).to("cuda")


def track_video(video: VideoArray, grid_size=50, start=0) -> TrackXYArray:
    # The number of points is grid_size*grid_size
    # pred_tracks: NDArray[Shape["1, * frames, * points, [x, y]"], Float]
    pred_tracks, _ = cotracker(
        torch.from_numpy(video).cuda(), grid_size=grid_size, grid_query_frame=start
    )
    return pred_tracks.permute(0, 2, 1, 3).squeeze().cpu().numpy()


@cache
def track_video_online(video: VideoArray, grid_size=50, start=0) -> TrackXYArray:
    # The number of points is grid_size*grid_size
    # pred_tracks: NDArray[Shape["1, * frames, * points, [x, y]"], Float]
    torch_video = torch.from_numpy(video[:, start:]).cuda()
    cotracker(video_chunk=torch_video, is_first_step=True, grid_size=grid_size)
    for ind in range(0, torch_video.shape[1] - cotracker.step, cotracker.step):
        pred_tracks, _ = cotracker(
            video_chunk=torch_video[:, ind : ind + cotracker.step * 2]
        )
    arranged = pred_tracks.permute(0, 2, 1, 3).squeeze().cpu().numpy()
    zeros = np.zeros((arranged.shape[0], start, 2))
    return np.concatenate((zeros, arranged), axis=1)


def separate_fg(tracks: TrackXYVTArray) -> TrackXYVTArray:
    """Separate tracks by speed into foreground and background, return foreground"""
    velos = tracks[:, :, 2]  # track, frame, speed
    kmeans = KMeans(n_clusters=2, n_init=4).fit(velos)
    c1 = np.array([tracks[i] for i in range(len(tracks)) if kmeans.labels_[i] == 0])
    c2 = np.array([tracks[i] for i in range(len(tracks)) if kmeans.labels_[i] == 1])
    # Get mean speed accross cluster of mean speed in the track
    avg1 = np.mean(c1[:, :, 2].mean(axis=1))
    avg2 = np.mean(c2[:, :, 2].mean(axis=1))
    if avg1 > avg2:
        return c1
    else:
        return c2
