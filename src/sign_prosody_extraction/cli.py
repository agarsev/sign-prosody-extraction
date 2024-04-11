# The command line interface for the application. There is one entry point,
# which receives a number of videos to process. There are different options
# to control the process.
import click
import imageio.v3 as iio
import numpy as np

# Returned shape is 1, n_frames, 3 (channels?), height, width
def load_video (video_file):
    frames = iio.imread(str(video_file), plugin="FFMPEG")
    video = np.transpose(frames.astype(np.float32), (0,3,1,2))
    return video[None, :]


@click.command()
@click.argument('videos', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('--visualize', is_flag=True, help='Visualize the extracted tracks')
@click.option('--cotracker', 'algorithm', flag_value='cotracker', default=True, help='Use the CoTracker algorithm')
@click.option('--mediapipe', 'algorithm', flag_value='mediapipe', help='Use the MediaPipe algorithm')
def main(videos, visualize, algorithm):
    if algorithm == 'cotracker':
        from .articulator.cotracker import track_hands
    elif algorithm == 'mediapipe':
        from .articulator.mediapipe import track_hands
    from .plot import plot_prosody
    from .visualize import overlay_tracks
    for video_file in videos:
        video = load_video(video_file)
        hands, first_frame = track_hands(video)
        plot_prosody(hands, "plot.png")
        overlay_tracks(video[:, first_frame:], hands, "track.mp4")
