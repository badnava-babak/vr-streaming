from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import numpy as np
import pandas as pd


@dataclass
class DeviceStats:
    latency: List[float] = field(default_factory=list)
    energy_consumption: List[float] = field(default_factory=list)
    ymse: List[float] = field(default_factory=list)
    psnr: List[float] = field(default_factory=list)
    quality_decision: List[int] = field(default_factory=list)
    offloading_decision: List[int] = field(default_factory=list)
    stall_times: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    task_size: List[int] = field(default_factory=list)
    task_res_size: List[int] = field(default_factory=list)
    comp_intensity: List[int] = field(default_factory=list)
    rx_times: List[float] = field(default_factory=list)
    tx_times: List[float] = field(default_factory=list)
    video_id: List[int] = field(default_factory=list)
    ch_uplink_rates: List[List[float]] = field(default_factory=list)
    ch_downlink_rates: List[List[float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(
            latency=np.array(self.latency),
            energy_consumption=np.array(self.energy_consumption),
            ymse=np.array(self.ymse),
            psnr=np.array(self.psnr),
            quality_decision=np.array(self.quality_decision),
            offloading_decision=np.array(self.offloading_decision),
            stall_times=np.array(self.stall_times),
            rewards=np.array(self.rewards),
            task_size=np.array(self.task_size),
            task_res_size=np.array(self.task_res_size),
            comp_intensity=np.array(self.comp_intensity),
            rx_time=np.array(self.rx_times),
            tx_time=np.array(self.tx_times),
            uplink_5g_rates=np.array(self.ch_uplink_rates)[:, 0],
            uplink_4g_rates=np.array(self.ch_uplink_rates)[:, 1],
            uplink_wigig_rates=np.array(self.ch_uplink_rates)[:, 2],
            downlink_5g_rates=np.array(self.ch_downlink_rates)[:, 0],
            downlink_4g_rates=np.array(self.ch_downlink_rates)[:, 1],
            downlink_wigig_rates=np.array(self.ch_downlink_rates)[:, 2],
            video_id=np.array(self.video_id),
        )

    def summary_stats(self):
        m = {
            'reward_mean': np.mean(self.rewards),
            'latency_mean': np.mean(self.latency),
            'latency_p95': np.percentile(self.latency, 95),
            'latency_p05': np.percentile(self.latency, 5),
            'energy_mean': np.mean(self.energy_consumption),
            'energy_p05': np.percentile(self.energy_consumption, 5),
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
        for q in range(7):
            m.update({
                f'quality_{q}_ratio': np.mean(np.array(self.quality_decision) == q),
            })
        return m


@dataclass
class EpisodeStats:
    device_stats: List[DeviceStats] = field(default_factory=list)
    stall_events: int = 0
    w: List[float] = field(default_factory=list)

    def __init__(self, nb_devices: int, w: List[float]):
        self.device_stats = [DeviceStats() for _ in range(nb_devices)]
        self.w = w

    def record_stats(self, device_id: int,
                     latency: float,
                     energy_consumption: float,
                     psnr: float,
                     ymse: float,
                     stall_time: float,
                     quality_decision: int,
                     offloading_decision: int,
                     task_size: int,
                     comp_intensity: int,
                     task_res_size: int,
                     tx_time: float,
                     rx_time: float,
                     ch_uplink_rates: List[float],
                     ch_downlink_rates: List[float],
                     device_reward: float,
                     video_id: int
                     ):
        self.device_stats[device_id].latency.append(latency)
        self.device_stats[device_id].energy_consumption.append(energy_consumption)
        self.device_stats[device_id].psnr.append(psnr)
        self.device_stats[device_id].ymse.append(ymse)
        self.device_stats[device_id].quality_decision.append(quality_decision)
        self.device_stats[device_id].offloading_decision.append(offloading_decision)
        self.device_stats[device_id].stall_times.append(stall_time)
        self.device_stats[device_id].task_size.append(task_size)
        self.device_stats[device_id].comp_intensity.append(comp_intensity)
        self.device_stats[device_id].task_res_size.append(task_res_size)
        self.device_stats[device_id].rx_times.append(rx_time)
        self.device_stats[device_id].tx_times.append(tx_time)
        self.device_stats[device_id].ch_uplink_rates.append(ch_uplink_rates)
        self.device_stats[device_id].ch_downlink_rates.append(ch_downlink_rates)
        self.device_stats[device_id].rewards.append(device_reward)
        self.device_stats[device_id].video_id.append(video_id)

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
        overall['stall_time_mean'] = np.mean([d.stall_times[-1] for d in self.device_stats])
        overall['stall_time_p05'] = np.percentile([d.stall_times[-1] for d in self.device_stats], 5)
        overall['stall_time_p95'] = np.percentile([d.stall_times[-1] for d in self.device_stats], 95)
        overall['deadline_violation'] = (np.array([d.latency for d in self.device_stats]) > 1.).astype(int).mean()

        offloading_decision = np.array([d.offloading_decision for d in self.device_stats]).flatten()
        local_processing = (offloading_decision == 0).astype(int)
        quality_decision = np.array([d.quality_decision for d in self.device_stats]).flatten()
        task_size = np.array([d.task_size for d in self.device_stats]).flatten()

        df = pd.DataFrame(np.stack([task_size, offloading_decision,
                                    quality_decision, local_processing], axis=1),
                          columns=['task_size', 'offloading_decision',
                                   'quality_decision', 'local_processing']
                          )
        df['task_size_q'], bins = pd.cut(df['task_size'], 10, labels=False, retbins=True)

        df.groupby(['task_size_q', 'quality_decision']).agg('count')

        return {'per_device': device_metrics, 'overall': overall}
