from nptyping import NDArray, Shape, Float

# Numpy NDArray for a video
VideoArray = NDArray[Shape["1, * frames, 3 channels, * height, * width"], Float]

# Numpy NDArray for a number of positional tracks on a video, with (x, y) coords
TrackXYArray = NDArray[Shape["* tracks, * frames, [x, y]"], Float]

# Numpy NDArray for a number of tracks on a video, including velocity and
# direction components
TrackXYVTArray = NDArray[Shape["* tracks, * frames, [x, y, v, theta]"], Float]

# Numpy NDArray for a single articulator track, with position, velocity and
# direction components
ArticulatorArray = NDArray[Shape["2 hands, * frames, [x, y, v, theta]"], Float]
