from pathlib import Path
import torch
import numpy as np

from .typing import VideoArray, ArticulatorArray


def overlay_tracks(video: VideoArray, track: ArticulatorArray, output="track.mp4"):
    from .articulator.cotracker import cotracker  # noqa: F401
    from cotracker.utils.visualizer import Visualizer

    output = Path(output)
    vis = Visualizer(
        save_dir=output.parent,
        pad_value=0,
        mode="optical_flow",
        tracks_leave_trace=10,
        linewidth=2,
    )
    # remove r and theta, transpose and add additional dimensions
    cotracks = torch.from_numpy(track[:, :, 0:2]).permute(1, 0, 2)[None]
    vis.visualize(torch.from_numpy(video), cotracks, filename=output.stem)


def get_thumbnails(video: VideoArray, targets, first_frame, frames, output="thumb"):
    from torchvision.transforms import ToPILImage

    if frames == "FIRST":
        frames = [targets[0]]
    elif frames == "LAST":
        frames = [targets[-1]]
    elif frames == "ALL":
        frames = targets
    else:
        frames = [targets[i] for i in frames]
    to_pil = ToPILImage()
    for i, f in enumerate(frames):
        image = video[0, f + first_frame].transpose(1, 2, 0).astype(np.uint8)
        to_pil(image).save(f"{output}_{i}.png")


def clip_video(video_file, start, end, output="clip.mp4", fps=25):
    from subprocess import run

    run(
        [
            "ffmpeg",
            "-v",
            "warning",
            "-ss",
            str(start / fps),
            "-to",
            str(end / fps),
            "-i",
            video_file,
            output,
        ]
    )
