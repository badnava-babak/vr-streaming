from __future__ import annotations

from src.commons.frame import Frame


class EdgeNode:
    def __init__(self,
                 processing_rate: float):  # bit/s equivalent
        self.proc_rate = processing_rate
        self.time_available = 0.0  # next free time

    def reset(self):
        self.time_available = 0

    def process(self, arrive_time: float, bits: float, proceed: bool) -> float:
        """Return finish time of processing."""
        start = max(arrive_time, self.time_available)
        dur = bits / self.proc_rate
        if proceed:
            self.time_available = start + dur
        return dur
