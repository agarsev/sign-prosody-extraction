from pathlib import Path
import torch

from .typing import VideoArray, ArticulatorArray

def overlay_tracks (video: VideoArray, track: ArticulatorArray, output):
    from .articulator.cotracker import cotracker
    from cotracker.utils.visualizer import Visualizer
    output = Path(output)
    vis = Visualizer(save_dir=output.parent, pad_value=0, mode="optical_flow",
                     tracks_leave_trace=10, linewidth=2)
    # remove r and theta, transpose and add additional dimensions
    cotracks = torch.from_numpy(track[:, :, 0:2]).permute(1, 0, 2)[None]
    vis.visualize(torch.from_numpy(video), cotracks, filename=output.stem)
