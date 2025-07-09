from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Callable

from src.commons.stats import EpisodeStats
from src.nodes.camera import Camera360
from src.nodes.edge_server import EdgeNode
from src.nodes.vr_device import VRDevice


# 7 ────────────────────────────────────────────────────────────────────────
# Simulator orchestration
# -------------------------------------------------------------------------

class StreamingEnv:
    def __init__(self,
                 camera: Camera360,
                 edge: EdgeNode,
                 users: List[VRDevice],
                 horizon: float = 10.0):
        self.camera = camera
        self.edge = edge
        self.users = users
        self.horizon = horizon
        self.stats = EpisodeStats()

    def run(self,
            policy: Callable[[int], str] | None = None):

        t = 0.0
        dt = 1.0 / self.camera.fps
        while t < self.horizon:
            # idx, act = policy.choose_action()
            profile = 4
            channel = 0

            # prof = profile_policy(self.camera._ctr)
            frame = self.camera.capture(t, profile)
            # camera → edge
            tx = self.camera.transmit(frame)
            arrival_edge = t + tx

            if channel != 0:  # edge computing
                finish_processing = self.edge.process(arrival_edge, frame, profile)
            else:  # local computing
                finish_processing = arrival_edge

            # edge → every user (multicast assumption)
            for user in self.users:
                if channel == 0:  # local computing
                    reception_time = user.receive(finish_processing, frame, profile)
                    display_time = user.process(reception_time, frame, profile)
                else:
                    display_time = user.receive(finish_processing, frame, profile)
                self.stats.record_frame(frame.timestamp, display_time)
            t += dt
