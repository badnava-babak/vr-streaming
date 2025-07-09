from __future__ import annotations

from typing import List, Callable

import numpy as np

from src.commons.stats import EpisodeStats
from src.nodes.edge_server import EdgeNode
from src.nodes.vr_device import VRDevice
from src.policies.optimal_dm import DecisionMaker


class ElasticTaskOffloadingEnv:
    def __init__(self,
                 edge: EdgeNode,
                 devices: List[VRDevice],
                 horizon: float = 36.0):
        self.edge = edge
        self.devices = devices
        self.horizon = horizon
        self.nb_devices = len(self.devices)
        self.stats = EpisodeStats(self.nb_devices)

    def reset(self):
        self.stats = EpisodeStats(self.nb_devices)
        self.edge.reset()
        for d in self.devices:
            d.reset()

    def run(self,
            policy: DecisionMaker):
        self.reset()

        t = 0.0
        dt = 1.0 / 30.
        step = 0
        while t < self.horizon:
            # idx, act = policy.choose_action()
            quality_idx = np.random.randint(0, high=7, size=self.nb_devices)
            channel_idx = np.random.randint(0, high=1, size=self.nb_devices)

            # tasks arrive
            tasks = []
            for device in self.devices:
                # device.get_state(step, t)
                tasks.append(device.get_task(step, t))
            # tasks arrived
            # TODO: decision making section

            for device_id in range(self.nb_devices):
                task, device = tasks[device_id], self.devices[device_id]
                q_idx, offloading_decision = policy(step, t, device, self.edge)

                # q_idx = quality_idx[device_id]
                # ch_idx = channel_idx[device_id]
                local_computation = (offloading_decision == 0)
                ch_idx = offloading_decision - 1

                comp_intensity = task.get_computational_intensity(q_idx)
                if local_computation:
                    # local computing
                    processing_time, energy_consumption = device.process(t, comp_intensity, True)
                else:
                    # offload to edge server
                    task_size = task.get_size(q_idx)
                    task_response_size = task.get_response_size(q_idx)
                    # device -> edge
                    tx_time, tx_energy = device.send(task_size, ch_idx)
                    arrival_edge = t + tx_time
                    # process on edge server
                    exec_time = self.edge.process(arrival_edge, comp_intensity, True)
                    # edge -> device
                    rx_time, rx_energy = device.receive(task_response_size, ch_idx)

                    processing_time = exec_time + rx_time
                    energy_consumption = rx_energy + tx_energy

                # update playback buffer
                device.update_buffer(processing_time)

                # record keeping
                self.stats.record_stats(device_id,
                                        processing_time,
                                        energy_consumption,
                                        task.get_psnr(q_idx),
                                        task.get_ymse(q_idx),
                                        device.stall_time,
                                        q_idx,
                                        offloading_decision
                                        )

            t += dt
            step += 1

        return self.stats
