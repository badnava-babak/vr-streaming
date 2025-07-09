from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Callable

import os

from src.chennels.trace_channel import ZeroDelayChannel, TraceChannel
from src.commons.io import load_hn, load_rd, load_gr
from src.envs.live_video_streaming import StreamingEnv
from src.nodes.camera import TraceCamera
from src.nodes.edge_server import EdgeNode
from src.nodes.vr_device import VRDevice
from src.plotting.network_traces_utils import load_net_trace

if __name__ == "__main__":

    video_id = 0
    user_id = 0
    trace_id = 0
    qp_profiles = {str(k): k for k in [5, 10, 15, 20, 25, 30, 35]}  # example QP values

    try:
        _5g_data_path = "datasets/Network-Traces/Lumous5G/5G"
        _4g_data_path = "datasets/Network-Traces/Lumous5G/4G"

        files = os.listdir(_5g_data_path)
        walking_traces_path = list(filter(lambda x: 'walking' in x, files))
        walking_traces_path = ["%s/%s" % (_5g_data_path, f_name) for f_name in walking_traces_path]

        net_df = load_net_trace(walking_traces_path[trace_id])

        hmd = load_hn('datasets/navigation/hn.mat')
        rd = load_rd('datasets/navigation/rd.mat')
        gr = load_gr('datasets/compression_factors.csv')

    except FileNotFoundError:
        # Generate synthetic mini traces if files are absent (demo only)
        raise FileNotFoundError("Files not found.")

    vid_bitrate = rd['bitrate'][video_id]  # length-15 list, each (7, 64, N)
    vid_ymse = rd['ymse'][video_id]
    vid_bitdepth = rd['bitdepth'][video_id]

    horizon = vid_bitrate.shape[2]
    # uplink_channel = TraceChannel(net_df)
    uplink_channel = ZeroDelayChannel()
    cam = TraceCamera(fps=30,
                      encoding_profiles=qp_profiles,
                      uplink=uplink_channel, bitrate_df=vid_bitrate)

    edge = EdgeNode(processing_rate=550e6)
    downlink_channel = TraceChannel(net_df)
    vr = VRDevice(channel=downlink_channel, processing_rate=550e6)

    env = StreamingEnv(cam, edge, [vr], horizon=horizon / 30)
    env.run(lambda fid: "5")

    # profiles = {"low": 0.2, "medium": 0.4, "high": 0.6}
    # cam = Camera360(fps=30,
    #                 base_frame_size=8_000_000,
    #                 encoding_profiles=profiles,
    #                 uplink=Channel(40e6))
    # edge = EdgeNode(processing_rate=120e6)
    # user_chan = Channel(60e6)
    # vr = VRUser(downlink=user_chan)
    #
    # env = StreamingEnv(cam, edge, [vr], horizon=2.0)
    # env.run(lambda fid: "high")

    print(f"Mean end‑to‑end latency: {np.mean(env.stats.latency) * 1000:.1f} ms")
