from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class DeviceStats:
    latency: List[float] = field(default_factory=list)
    energy_consumption: List[float] = field(default_factory=list)
    ymse: List[float] = field(default_factory=list)
    psnr: List[float] = field(default_factory=list)
    quality_decision: List[int] = field(default_factory=list)
    offloading_decision: List[int] = field(default_factory=list)
    stall_times: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(
            latency=np.array(self.latency),
            energy_consumption=np.array(self.energy_consumption),
            ymse=np.array(self.ymse),
            psnr=np.array(self.psnr),
            quality_decision=np.array(self.quality_decision),
            offloading_decision=np.array(self.offloading_decision),
            stall_times=np.array(self.stall_times)
        )

    def summary_stats(self):
        m = {
            'latency_mean': np.mean(self.latency),
            'latency_p95': np.percentile(self.latency, 95),
            'latency_p05': np.percentile(self.latency, 5),
            'energy_mean': np.mean(self.energy_consumption),
            'energy_p5': np.percentile(self.energy_consumption, 5),
            'energy_p95': np.percentile(self.energy_consumption, 95),
            'psnr_mean': np.mean(self.psnr),
            'psnr_p05': np.percentile(self.psnr, 5),
            'psnr_p95': np.percentile(self.psnr, 95),
            'ymse_mean': np.mean(self.ymse),
            'ymse_p05': np.percentile(self.ymse, 5),
            'ymse_p95': np.percentile(self.ymse, 95),
            'stall_total': self.stall_times[-1],
            'offload_ratio': np.mean(np.array(self.offloading_decision) > 0),
            '5G_ratio': np.mean(np.array(self.offloading_decision) == 1),
            '4G_ratio': np.mean(np.array(self.offloading_decision) == 2),
            'WiGig_ratio': np.mean(np.array(self.offloading_decision) == 3),
        }
        return m


@dataclass
class EpisodeStats:
    device_stats: List[DeviceStats] = field(default_factory=list)
    stall_events: int = 0

    def __init__(self, nb_devices: int):
        self.device_stats = [DeviceStats() for _ in range(nb_devices)]

    def record_stats(self, device_id: int,
                     latency: float,
                     energy_consumption: float,
                     psnr: float,
                     ymse: float,
                     stall_time: float,
                     quality_decision: int,
                     offloading_decision: int
                     ):
        self.device_stats[device_id].latency.append(latency)
        self.device_stats[device_id].energy_consumption.append(energy_consumption)
        self.device_stats[device_id].psnr.append(psnr)
        self.device_stats[device_id].ymse.append(ymse)
        self.device_stats[device_id].quality_decision.append(quality_decision)
        self.device_stats[device_id].offloading_decision.append(offloading_decision)
        self.device_stats[device_id].stall_times.append(stall_time)

    def to_dict(self) -> dict:
        return dict(
            {
                i: device.to_dict() for i, device in enumerate(self.device_stats)
            }
        )

    def summary_stats(self):
        device_metrics = []
        for d in self.device_stats:
            device_metrics.append(d.summary_stats())
        overall = {k: np.mean([d[k] for d in device_metrics]) for k in device_metrics[0]}
        return {'per_device': device_metrics, 'overall': overall}

