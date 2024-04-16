from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from pathlib import Path
import pandas as pd

from .typing import ArticulatorArray

# Colors for plotting direction
colors = ["red", "green", "gold", "blue", "red"]  # H, X (right in the image), L, Y
nodes = [0, 0.25, 0.5, 0.75, 1]
cmap = LinearSegmentedColormap.from_list("custom", list(zip(nodes, colors)), N=256)

scale_img = plt.imread(Path(__file__).parent.parent / "img/dir_scale.png")


# Receive numpy array
# long: make a wider plot
# fps to write seconds value in axis
# marks is a list of dicts with start, end and gloss to highlight parts
# points is a list of points to demarkate
def plot_prosody(
    track: ArticulatorArray, output, long=False, fps=25, areas=[], points=[]
):
    plt.figure(figsize=(8 if long else 4, 3), dpi=300)
    ax = plt.gca()

    for m in areas:
        start = m["start"]
        end = m["end"]
        ax.axvspan(start, end, color="olive", alpha=0.1)
        ax.text(
            (start + end) / 2,
            ax.get_ylim()[1] * 0.9,
            m["gloss"],
            fontfamily="Tex Gyre Heros",
            ha="center",
            va="top",
        )

    for p in points:
        ax.axvline(x=p / fps, linewidth=1, color="olive", alpha=0.2)

    offset = 0
    if len(track) > 1:
        plot_hand(ax, track[1], 0, fps)
        offset = max(track[1][:, 2])
        ax.axhline(offset * 1.2, color="black", linewidth=0.5, linestyle="--")
        ax.text(
            0, offset * 1.3, "H1 ", fontfamily="Tex Gyre Heros", va="bottom", ha="right"
        )
        ax.text(0, offset, "H2 ", fontfamily="Tex Gyre Heros", va="top", ha="right")
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0] - xlim[1] * 0.05, xlim[1])
    plot_hand(ax, track[0], offset * 1.4, fps)

    plt.tight_layout()
    plt.subplots_adjust(left=0.06, bottom=0.2)
    plt.xlabel("Time (s)")
    plt.locator_params(nbins=20 if long else 10)
    plt.ylabel("Velocity (pixels)")
    ax.set_yticks([])

    axins = inset_axes(
        ax, width="8%" if long else "15%", height="20%", loc="upper right"
    )
    axins.imshow(scale_img)
    axins.axis("off")

    plt.savefig(output)


def plot_hand(ax, hand, offset, fps):
    df = pd.DataFrame(hand, columns=["x", "y", "vel", "angle"])
    df["normangle"] = (0.25 + df["angle"] / (2 * np.pi)) % 1.0
    df["color"] = df["normangle"].apply(cmap)
    df["vel"] = df["vel"] + offset

    for i in range(1, len(df)):
        ax.plot(
            [j / fps for j in df.index[i - 1 : i + 1]],
            df["vel"].iloc[i - 1 : i + 1],
            color=df["color"].iloc[i],
        )
