# The command line interface for the application. There is one entry point,
# which receives a number of videos to process. There are different options
# to control the process.
from pathlib import Path

import click

from . import get_tracker, load_video, visualize


@click.command()
@click.argument("videos", nargs=-1, type=click.Path(exists=True), required=True)
@click.option(
    "--cotracker",
    "algorithm",
    flag_value="cotracker",
    default=True,
    help="Use the original CoTracker algorithm for tracking.",
)
@click.option(
    "--mediapipe",
    "algorithm",
    flag_value="mediapipe",
    help="Use the alternative MediaPipe algorithm for tracking (pose_landmarker model needs to be available).",
)
@click.option(
    "--track-video/--no-track-video",
    default=False,
    help="Output a video with the extracted tracks overlaid. The filename will be the original filename with '_track' appended.",
)
@click.option(
    "--targets/--no-targets",
    "find_targets",
    default=False,
    help="Find target points in the video. Enabled by default if thumbnails or clip are specified.",
)
@click.option(
    "--plot/--no-plot",
    default=False,
    help="Output a plot with the extracted prosody. The filename will be the original filename with '_plot' appended.",
)
@click.option(
    "--thumbnails",
    type=str,
    help="""Generate thumbnails at the specified target points. Can be FIRST, LAST,
    ALL, or a list of target point numbers. Output filenames will be the original
    filename with '_thumb_<n>' appended.""",
)
@click.option(
    "--clip/--no-clip",
    default=False,
    help="Clip the video from the first target to the last (requires FFMPEG). The filename will be the original filename with '_clip' appended.",
)
@click.option("--everything", is_flag=True, help="Enable all output options.")
@click.option(
    "--filename",
    type=str,
    help="Use this string instead of the original filename for outputs.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="output",
    help="Output directory for results.",
)
@click.option(
    "--csv/--no-csv",
    default=False,
    help="Export the analysis results as a csv time series importable by ELAN, for example.",
)
@click.version_option()
def main(
    videos,
    algorithm,
    track_video,
    find_targets,
    plot,
    thumbnails,
    clip,
    everything,
    filename,
    output_dir,
    csv,
):
    """Command line tool implementing the methodology outlined in "Automated
    Extraction of Prosodic Structure from Unannotated Sign Language Video"
    (Sevilla et al., 2024)."""

    track_hands = get_tracker(algorithm)
    if everything:
        track_video = plot = clip = True
        thumbnails = "ALL"
    if thumbnails or clip:
        find_targets = True
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for video_file in videos:
        video = load_video(video_file)
        hands, first_frame = track_hands(video)
        ofile = Path(video_file).stem if filename is None else filename

        if track_video:
            visualize.overlay_tracks(
                video[:, first_frame:], hands, output_dir / f"{ofile}_track.mp4"
            )

        targets = []
        if find_targets:
            from .targets import get_target_points

            targets = get_target_points(hands[0])

        if plot:
            from .plot import plot_prosody

            plot_prosody(
                hands,
                output_dir / f"{ofile}_plot.png",
                long=len(video[0]) > 100,
                points=targets,
            )

        if thumbnails:
            visualize.get_thumbnails(
                video, targets, first_frame, thumbnails, output_dir / f"{ofile}_thumb"
            )

        if clip:
            visualize.clip_video(
                video_file,
                targets[0] + first_frame,
                targets[-1] + first_frame,
                output_dir / f"{ofile}_clip.mp4",
            )

        if csv:
            from .csvexport import export_csv

            export_csv(
                video_file,
                hands,
                output_dir / f"{ofile}_ts.csv",
            )
