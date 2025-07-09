from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd

from src.chennels.trace_channel import Channel


class TwoWayChannel:
    """Throughput determined by a time‑stamped trace."""

    def __init__(self, uplink: Channel, downlink: Channel):
        # Ensure time_s is sorted
        super().__init__()
        self.uplink = uplink
        self.downlink = downlink

    def reset(self) -> None:
        self.uplink.reset()
        self.downlink.reset()

    def time_to_tx(self, bits: int, move_forward: bool) -> Tuple[float, float]:
        return self.uplink.time_to_tx(bits, move_forward)

    def time_to_rx(self, bits: int, move_forward: bool) -> Tuple[float, float]:
        return self.downlink.time_to_tx(bits, move_forward)

    def get_uplink_rate(self):
        return self.uplink.sample_rate()

    def get_downlink_rate(self):
        return self.downlink.sample_rate()

    def get_rates(self):
        return [self.get_uplink_rate(), self.get_downlink_rate()]
