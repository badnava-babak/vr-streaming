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

        # tx_times = np.zeros((self.q_levels, self.nb_channels))
        # tx_energies = np.zeros((self.q_levels, self.nb_channels))
        # rx_times = np.zeros((self.q_levels, self.nb_channels))
        # rx_energies = np.zeros((self.q_levels, self.nb_channels))
        # for q in range(0, self.q_levels):
        #     for ch in range(0, self.nb_channels):
        #         tx_times[q][ch], tx_energies[q][ch] = self.channels[ch].time_to_tx(task_sizes[q], False)
        #         rx_times[q][ch], rx_energies[q][ch] = self.channels[ch].time_to_rx(task_response_sizes[q], False)
        rates = [ch.get_rates() for ch in self.channels]

        return dict()

    def get_task(self, ctr: int, t: float) -> Frame:
        viewport = self.user.get_viewport(ctr)
        frame = self.video.get_frame(ctr, t, viewport)
        return frame

    def process(self, arrive_time: float, comp_intensity: float, proceed: bool) -> Tuple[float, float]:
        start = max(arrive_time, self.time_available)
        dur = comp_intensity / self.proc_rate
        if proceed:
            self.time_available = start + dur
        return dur, self._computation_energy_consumption(comp_intensity)

    def _computation_energy_consumption(self, comp_intensity: float) -> float:
        return self.kappa * comp_intensity * np.power(self.cpu_freq, 2)

    def receive(self, bits: int, channel: int) -> Tuple[float, float]:
        tx_time, energy_consumption = self.channels[channel].time_to_tx(bits, True)
        return tx_time, energy_consumption

    def update_buffer(self, processing_time: float):
        # update buffer after arrival
        self.stall_time += max(0., processing_time - self.buffer)

        self.buffer = max(0.0, self.buffer - processing_time)
        self.buffer = max(self.buffer + 1.0 / self.target_fps, 0)
        self.buffer = min(self.buffer, self.buffer_max)
        self.play_time = processing_time

        return processing_time

    def send(self, bits: int, channel: int) -> Tuple[float, float]:
        return self.channels[channel].time_to_rx(bits, True)

    @property
    def target_fps(self):
        return 30  # placeholder
