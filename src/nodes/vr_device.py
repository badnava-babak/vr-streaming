from __future__ import annotations

from typing import List, Tuple
import pandas as pd
import numpy as np

from src.chennels.two_way_channel import TwoWayChannel
from src.commons.frame import Frame
from src.nodes.user import User
from src.nodes.video import Video


class VRDevice:
    def __init__(self,
                 channels: List[TwoWayChannel],
                 processing_rate: float,
                 cpu_freq: float,
                 video: Video,
                 user: User,
                 buffer_max: float = 10.0):  # seconds

        self.cpu_freq = cpu_freq
        self.video = video
        self.user = user
        self.channels = channels
        self.buffer = 0.0
        self.stall_time = 0.0
        self.buffer_max = buffer_max
        self.play_time = 0.0  # time already displayed
        self.proc_rate = processing_rate
        self.time_available = 0.0  # next free time
        self._ctr = 0
        self.q_levels = 7
        self.nb_channels = len(channels)
        self.kappa = 1.e-27
        self.normalization_factors = self.calc_normalization_factors()

    def reset(self):
        self.stall_time = 0.0
        self.play_time = 0.0
        self.buffer = 0.0
        self.time_available = 0.0  # next free time
        self._ctr = 0
        for ch in self.channels:
            ch.reset()

    def get_state(self, ctr: int, t: float):

        task = self.get_task(ctr, t)
        task_sizes = task.get_sizes()
        task_comp_intensities = task.get_computational_intensities()
        task_response_sizes = task.get_response_sizes()
        self.time_available

        n_task_sizes = (np.array(task_sizes) - self.normalization_factors["size"]["mean"]) / \
                       self.normalization_factors["size"]["std"]
        n_task_comp_intensities = (np.array(task_comp_intensities) - self.normalization_factors["comp_intensity"][
            "mean"]) / self.normalization_factors["comp_intensity"]["std"]
        n_task_response_sizes = (np.array(task_response_sizes) - self.normalization_factors["response_size"]["mean"]) / \
                                self.normalization_factors["response_size"]["std"]

        uplink_rates = np.array([ch.get_uplink_rate() for ch in self.channels])
        downlink_rates = np.array([ch.get_downlink_rate() for ch in self.channels])
        uplink_mean = np.array(
            [self.normalization_factors['channels'][i]['uplink']['mean'] for i in range(len(self.channels))])
        uplink_std = np.array(
            [self.normalization_factors['channels'][i]['uplink']['std'] for i in range(len(self.channels))])
        downlink_mean = np.array([self.normalization_factors['channels'][i]['downlink']['mean'] for i in
                                  range(len(self.channels))])
        downlink_std = np.array(
            [self.normalization_factors['channels'][i]['downlink']['std'] for i in range(len(self.channels))])

        n_uplink_rates = (uplink_rates - uplink_mean) / uplink_std
        n_downlink_rates = (downlink_rates - downlink_mean) / downlink_std

        return dict(
            task_sizes=n_task_sizes,
            task_comp_intensities=n_task_comp_intensities,
            task_response_sizes=n_task_response_sizes,
            ch_uplink_rates=n_uplink_rates,
            ch_downlink_rates=n_downlink_rates,
            uplink_rates=uplink_rates,
            downlink_rates=downlink_rates

        )

    def get_task(self, ctr: int, t: float) -> Frame:
        viewport = self.user.get_viewport(ctr * self.target_fps)
        frame = self.video.get_frame(ctr, t, viewport, self.target_fps)
        return frame

    def calc_normalization_factors(self):
        N = int(self.video.length / self.target_fps)
        frames = []
        for ctr in range(N):
            viewport = self.user.get_viewport(ctr)
            frame = self.video.get_frame(ctr, 0., viewport, self.target_fps)
            frames.append(frame)
        frames_sizes = np.array([f.get_sizes() for f in frames])
        response_sizes = np.array([f.get_response_sizes() for f in frames])
        comp_intensities = np.array([f.get_computational_intensities() for f in frames])
        psnr_values = np.array([f.get_psnrs() for f in frames])

        return dict(
            size=self._normalization_factors(frames_sizes),
            response_size=self._normalization_factors(response_sizes),
            comp_intensity=self._normalization_factors(comp_intensities),
            psnr=self._normalization_factors(psnr_values),
            channels={i: ch.get_stats() for i, ch in enumerate(self.channels)}
        )

    def _normalization_factors(self, vec):
        max_ = np.max(vec, axis=0)
        min_ = np.min(vec, axis=0)
        mean_ = np.mean(vec, axis=0)
        std_ = np.std(vec, axis=0)
        return {
            "max": max_,
            "min": min_,
            "mean": mean_,
            "std": std_
        }

    def process(self, arrive_time: float, comp_intensity: float, proceed: bool) -> Tuple[float, float]:
        start = max(arrive_time, self.time_available)
        dur = comp_intensity / self.proc_rate
        if proceed:
            self.time_available = start + dur
        return dur, self._computation_energy_consumption(comp_intensity)

    def _computation_energy_consumption(self, comp_intensity: float) -> float:
        return self.kappa * comp_intensity * np.power(self.cpu_freq, 2)

    def receive(self, bits: int, channel: int) -> Tuple[float, float]:
        tx_time, energy_consumption = self.channels[channel].time_to_rx(bits, True)
        for ch in self.channels:
            if ch == channel:
                continue
            ch.time_to_rx(bits, True)
        return tx_time, energy_consumption

    def update_buffer(self, processing_time: float):
        # update buffer after arrival
        self.stall_time += max(0., processing_time - self.buffer)

        self.buffer = max(0.0, self.buffer - processing_time)
        self.buffer = max(self.buffer + 1.0, 0)
        self.buffer = min(self.buffer, self.buffer_max)
        self.play_time = processing_time

        return processing_time

    def send(self, bits: int, channel: int) -> Tuple[float, float]:
        tx_time, tx_energy = self.channels[channel].time_to_tx(bits, True)
        for ch in self.channels:
            if ch == channel:
                continue
            ch.time_to_tx(bits, True)

        return tx_time, tx_energy

    @property
    def target_fps(self):
        return 30  # placeholder
