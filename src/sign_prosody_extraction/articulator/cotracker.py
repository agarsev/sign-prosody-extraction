import torch

cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2",
                           verbose=False).to('cuda')

def track_video (video, grid_size=50, start=0):
    # Returned shape is 1, n_frames, n_points (grid_size**2), 2 (x,y)
    pred_tracks, _ = cotracker(torch.from_numpy(video).cuda(), grid_size=grid_size, grid_query_frame=start)
    # I return n_tracks (n_points), n_frames, 2 (x,y)
    return pred_tracks.permute(0, 2, 1, 3).squeeze().cpu().numpy()

from cotracker.utils.visualizer import Visualizer
vis = Visualizer(save_dir="./saved_videos", pad_value=0, mode="optical_flow",
                 tracks_leave_trace=10, linewidth=2)

def visualize_tracks (video, tracks, filename):
    # remove r and theta, transpose and add batch dimension
    cotracks = torch.from_numpy(tracks[:, :, 0:2]).permute(1, 0, 2)[None]
    vis.visualize(torch.from_numpy(video), cotracks, filename=filename)
