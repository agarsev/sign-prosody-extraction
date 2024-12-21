from pathlib import Path

import imageio.v3 as iio

from .typing import ArticulatorArray


def csv(video_file, track: ArticulatorArray, output_file):
    of = Path(output_file).open("w")
    two_hands = len(track) > 1
    video_meta = iio.immeta(str(video_file))
    duration = 1 / float(video_meta.get("fps", 20))

    of.write("time;h1_x;h1_y;h1_speed;h1_direction")
    if two_hands:
        of.write(";h2_x;h2_y;h2_speed;h2_direction")
    of.write("\n")

    for i, frame in enumerate(track[0]):
        data = [i * duration, *frame]
        if two_hands:
            data = [*data, *track[1][i]]
        of.write(";".join(str(round(x, 3)) for x in data))
        of.write("\n")
