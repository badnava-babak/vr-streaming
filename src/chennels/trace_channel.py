from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd


class Channel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sample_rate(self) -> float:
        pass

    @abstractmethod
    def _rate_at_index(self, idx: int) -> float:
        pass

    @abstractmethod
    def time_to_tx(self, bits: int, move_forward: bool) -> Tuple[float, float]:
        pass


class ZeroDelayChannel(Channel):
    """Throughput determined by a time‑stamped trace."""

    def _rate_at_index(self, idx: int) -> float:
        raise RuntimeError("Use time_to_tx(time) in the simulator loop")

    def sample_rate(self):
        raise RuntimeError("Use time_to_tx(time) in the simulator loop")

    def time_to_tx(self, bits: int, move_forward: bool) -> Tuple[float, float]:
        return 0., 0.


class StochasticChannel(Channel):
    """Simple stochastic throughput channel (in bit/s)."""

    def __init__(self, mean_rate: float, sigma: float = 0.25, rng: np.random.Generator | None = None):
        super().__init__()
        self.mean_log = np.log(mean_rate)
        self.sigma = sigma
        self.rng = np.random.default_rng() if rng is None else rng

    def sample_rate(self) -> float:
        """Log‑normal random throughput."""
        return float(self.rng.lognormal(self.mean_log, self.sigma))

    def _rate_at_index(self, idx: int) -> float:
        raise RuntimeError("Use sample_rate(time) in the simulator loop")

    def time_to_tx(self, bits: int, move_forward: bool) -> Tuple[float, float]:
        remaining = bits
        tx_time = 0
        while remaining > 0:
            rate = self.sample_rate()
            if rate >= remaining:
                tx_time += remaining / rate
                return tx_time, 0
            else:
                tx_time += 1.
            remaining -= rate
        return tx_time, 0


# ------------------------------------------------------------------
# Trace‑driven Channel
# ------------------------------------------------------------------
class TraceChannel(Channel):
    """Throughput determined by a time‑stamped trace."""

    def __init__(self, trace_df: pd.DataFrame, ch_type: str):
        # Ensure time_s is sorted
        super().__init__()
        self.ch_type = ch_type
        self.t = trace_df["time_ms"].to_numpy() / 1000.
        self.bps = trace_df["throughput_Mbps"].to_numpy() * 1e6
        self.Tmax = self.t[-1]
        self.idx = 0

        if ch_type == '5G':
            self.tx_power = 5.27 * 1.e-3
        elif ch_type == '4G':
            self.tx_power = 57.99 * 1.e-3
        elif ch_type == 'WiGig':
            self.tx_power = 4500 * 1.e-3

    def reset(self):
        self.idx = 0

    def _rate_at_index(self, idx: int) -> float:
        return float(self.bps[int(idx % self.Tmax)])

    def sample_rate(self):
        rate_t = self._rate_at_index(self.idx)
        rate_t1 = self._rate_at_index(self.idx + 1)
        return (rate_t + rate_t1) / 2

    def time_to_tx(self, bits: int, move_forward: bool) -> Tuple[float, float]:
        """
        Return the duration (seconds) needed to send `bits`
        """
        idx = self.idx
        remaining = bits
        tx_time = 0.
        energy_consumption = 0.
        while remaining > 0:
            rate = self._rate_at_index(idx)
            if remaining <= rate:
                dur = remaining / rate
                tx_time += dur
                if self.ch_type == 'WiGig':
                    energy_consumption += dur * self.tx_power
                else:
                    energy_consumption += dur * self.tx_power * rate * 1.e-6
                if move_forward:
                    self.idx += (idx - self.idx)
                return tx_time, energy_consumption
            else:
                tx_time += 1.
                if self.ch_type == 'WiGig':
                    energy_consumption += self.tx_power
                else:
                    energy_consumption += self.tx_power * rate * 1.e-6
            remaining -= rate
            idx += 1

        raise RuntimeError("Somthing went wrong in time_to_tx!")
