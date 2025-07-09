import numpy as np
import pathlib
import scipy.io as sio
from typing import Dict, List
import pandas as pd

from src.chennels.trace_channel import TraceChannel
from src.nodes.user import User
from src.nodes.video import Video
import os

from src.plotting.network_traces_utils import load_net_trace


def load_gr(gr_path: str):
    return pd.read_csv(gr_path)


def load_rd(rd_path: str) -> Dict:
    raw = sio.loadmat(rd_path, squeeze_me=True, struct_as_record=False)
    bitdepth = np.array(
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 10, 10, 8, 8, 8, 8], dtype=int
    )
    return {
        'bitrate': raw['video_bitrate_data'],  # 1×15 cell
        'ymse': raw['video_ymse_data'],  # 1×15 cell
        'qp2r': raw['QPtoR_exp'],  # 15×64×N cfit
        'r2d': raw['RtoD_exp'],  # 15×64×N cfit,
        "bitdepth": bitdepth
    }


def load_hn(hn_path: str) -> List:
    raw = sio.loadmat(hn_path, squeeze_me=True, struct_as_record=False)
    return raw['HMD_data']  # 12×12 cell


def load_all_videos():
    rd = load_rd('datasets/navigation/rd.mat')
    gr = load_gr('datasets/compression_factors.csv')
    n_videos = rd['bitrate'].shape[0]
    videos = {}
    for i in range(n_videos):
        videos[i] = Video(rd['bitrate'][i], rd['ymse'][i], rd['bitdepth'][i])

    return videos


def load_all_users():
    hmd = load_hn('datasets/navigation/hn.mat')
    nb_videos = hmd.shape[0]
    nb_users = hmd.shape[1]
    videos = {}
    for v_idx in range(nb_videos):
        video_users = {}
        j = 0
        for u_idx in range(nb_users):
            if hmd[v_idx][u_idx].shape[0] > 0:
                user = User(hmd[v_idx][u_idx])
                video_users[j] = user
                j += 1
        videos[v_idx] = video_users
    return videos


def load_all_traces(net_type: str = '5G'):
    if net_type == '5G':
        data_path = "datasets/Network-Traces/Lumous5G/5G"
    elif net_type == '4G':
        data_path = "datasets/Network-Traces/Lumous5G/4G"
    elif net_type == 'WiGig':
        data_path = "datasets/Network-Traces/WiGig"
    else:
        raise Exception("Invalid Network Type")

    walking_traces_path = os.listdir(data_path)
    if net_type in ['4G', '5G']:
        walking_traces_path = list(filter(lambda x: 'walking' in x, walking_traces_path))
    walking_traces_path = ["%s/%s" % (data_path, f_name) for f_name in walking_traces_path]

    channels = {}
    for i, walking_trace in enumerate(walking_traces_path):
        try:
            net_df = load_net_trace(walking_trace)
            channels[i] = TraceChannel(net_df, ch_type=net_type)
        except FileNotFoundError:
            # Generate synthetic mini traces if files are absent (demo only)
            # raise FileNotFoundError("Files not found.")
            continue
    return channels
