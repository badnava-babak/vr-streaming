from __future__ import annotations

import numpy as np
import pandas as pd

from src.commons.frame import Frame


class Video:
    def __init__(self,
                 bitrate_df: pd.DataFrame,
                 ymse_df: pd.DataFrame,
                 bit_depth: int
                 ):
        self.bitrate_df = bitrate_df  # (QP, tile index, N=frame index)
        self.ymse_df = ymse_df  # (QP, tile index, N=frame index)
        self.bit_depth = bit_depth

    def get_frame(self, ctr: int, t: float, viewport: np.ndarray) -> Frame:
        content = self.bitrate_df[:, :, ctr]
        ymse_info = self.ymse_df[:, :, ctr]
        raw = int(content[0].sum())
        frame = Frame(id=ctr,
                      uncompressed_size=raw,
                      timestamp=t,
                      content=content,
                      encoded_size=content.sum(axis=1),
                      ymse=ymse_info,
                      viewport=viewport,
                      bit_depth=self.bit_depth
                      )
        return frame
