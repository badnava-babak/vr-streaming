from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.chennels.trace_channel import Channel
from src.commons.frame import Frame


class Camera360:
    def __init__(self,
                 fps: int,
                 encoding_profiles: Dict[int, float],
                 uplink: Channel,
                 rng: np.random.Generator | None = None):
        """
        encoding_profiles maps profile‑name → compression‑factor
        """
        self.fps = fps
        self.profiles = encoding_profiles
        self.uplink = uplink
        self.rng = np.random.default_rng() if rng is None else rng
        self._ctr = 0

    def capture(self, t: float, profile: int) -> Frame:
        """Capture & encode one frame at time t using profile."""
        comp = self.profiles[profile]
        enc_time = 1.0 / self.fps  # simplification
        frm = Frame(id=self._ctr,
                    uncompressed_size=self.base_frame_size,
                    timestamp=t)
        self._ctr += 1
        return frm

    def transmit(self, frame: Frame) -> float:
        """Return transmission time (s)."""
        return self.uplink.time_to_tx(bits=frame.uncompressed_size)
        # return frame.uncompressed_size / rate


# ------------------------------------------------------------------
# Trace‑driven Camera bit‑rate
# ------------------------------------------------------------------

class TraceCamera(Camera360):
    def __init__(self, *, bitrate_df: pd.DataFrame, **kwargs):
        super().__init__(**kwargs)
        self.bitrates = bitrate_df
        self.max_id = len(self.bitrates[0, 0])

    def capture(self, t: float, profile: str):
        if self._ctr >= self.max_id:
            raise StopIteration("Video trace exhausted.")
        content = self.bitrates[:, :, self._ctr]
        raw = int(content[0].sum())
        frame = Frame(id=self._ctr,
                      uncompressed_size=raw,
                      timestamp=t,
                      content=content,
                      encoded_size=content.sum(axis=1))
        self._ctr += 1
        return frame
