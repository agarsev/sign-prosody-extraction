# The command line interface for the application. There is one entry point,
# which receives a number of videos to process. There are different options
# to control the process.
import click

from . import load_video, visualize


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
    help="Use the alternative MediaPipe algorithm for tracking.",
)
@click.option(
    "--track-video/--no-track-video",
    default=False,
    help="Output a video with the extracted tracks overlaid.",
)
@click.option(
    "--targets/--no-targets", "find_targets", default=True,
    help="Find target points in the video."
)
@click.option(
    "--plot/--no-plot", default=False, help="Output a plot with the extracted prosody."
)
@click.option(
    "--thumbnails",
    type=str,
    help="""Generate thumbnails at the specified frames. Can be FIRST, LAST, ALL, or a list of frame numbers""",
)
@click.option(
    "--clip/--no-clip",
    default=False,
    help="Clip the video from the first target to the last (requires FFMPEG).",
)
@click.version_option()
def main(videos, algorithm, track_video, find_targets, plot, thumbnails, clip):
    """Command line tool implementing the methodology outlined in "Automated
    Extraction of Prosodic Structure from Unannotated Sign Language Video"
    (Sevilla et al., 2024)."""
    if algorithm == "cotracker":
        from .articulator.cotracker import track_hands
    elif algorithm == "mediapipe":
        from .articulator.mediapipe import track_hands
    if thumbnails:
        find_targets = True

    for video_file in videos:
        video = load_video(video_file)
        hands, first_frame = track_hands(video)

        if track_video:
            visualize.overlay_tracks(video[:, first_frame:], hands, "track.mp4")

        targets = []
        if find_targets:
            from .targets import get_target_points

            targets = get_target_points(hands[0])

        if plot:
            from .plot import plot_prosody

            plot_prosody(hands, "plot.png", points=targets)

        if thumbnails:
            visualize.get_thumbnails(video, targets, first_frame, thumbnails)

        if clip:
            visualize.clip_video(
                video_file, targets[0] + first_frame, targets[-1] + first_frame
            )
