from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Callable


@dataclass
class Frame:
    """Represents one 360‑video frame/segment."""
    id: int
    uncompressed_size: int  # bits
    encoded_size: np.array  # bits
    timestamp: float  # capture time (s)
    content: np.array
    ymse: np.array
    viewport: np.array
    bit_depth: int

    def get_size(self, q_idx: int) -> int:
        base_layer = int(self.content[-1].sum())
        enhanced_layer = int(self.content[q_idx][self.viewport.reshape(-1)].sum())
        return base_layer + enhanced_layer

    def get_ymse(self, q_idx: int) -> float:
        return self.ymse[q_idx][self.viewport.reshape(-1)].mean()

    def get_psnr(self, q_idx: int) -> float:
        ymse = self.ymse[q_idx][self.viewport.reshape(-1)]
        nominator = np.power(np.power(2, self.bit_depth) - 1, 2)
        return 10 * np.log10(nominator / ymse).mean()

    def get_psnrs(self) -> List[float]:
        return [self.get_psnr(q) for q in range(self.content.shape[0])]

    def get_sizes(self) -> List[int]:
        return [self.get_size(q) for q in range(self.content.shape[0])]

    def get_response_size(self, q_idx: int) -> int:
        return int(self.get_size(q_idx) * .1)

    def get_response_sizes(self) -> List[int]:
        return [self.get_response_size(q) for q in range(self.content.shape[0])]

    def get_computational_intensity(self, q_idx: int) -> int:
        return int(0.6 * self.get_size(q_idx))

    def get_computational_intensities(self) -> List[int]:
        return [self.get_computational_intensity(q) for q in range(self.content.shape[0])]
