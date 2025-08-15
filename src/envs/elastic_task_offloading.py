from __future__ import annotations

from typing import List, Callable

import numpy as np

from src.commons.stats import EpisodeStats
from src.nodes.edge_server import EdgeNode
from src.nodes.vr_device import VRDevice
from src.policies.optimal_dm import DecisionMaker, OptimalDecisionMaker
from src.policies.ppg_policy import PPGPolicy, MultiTaskPPGPolicy, CentralizedMultiTaskPPGPolicy


class ElasticTaskOffloadingEnv:
    def __init__(self,
                 edge: EdgeNode,
                 devices: List[VRDevice],
                 weights: List[float],
                 horizon: float = 36.0,
                 ):
        self.edge = edge
        self.devices = devices
        self.horizon = horizon
        self.nb_devices = len(self.devices)
        self.w = weights
        self.stats = EpisodeStats(self.nb_devices, w=self.w)

    def reset(self):
        self.stats = EpisodeStats(self.nb_devices, w=self.w)
        self.edge.reset()
        for d in self.devices:
            d.reset()

    def run(self,
            policy: DecisionMaker):
        self.reset()

        t = 0.0
        dt = 1.0
        step = 0
        while t < self.horizon:
            # idx, act = policy.choose_action()
            q_idxs = np.zeros(self.nb_devices, dtype=np.int32)
            offloading_decisions = np.zeros(self.nb_devices, dtype=np.int32)

            # tasks arrive
            tasks = []
            state = {}
            for device_id in range(self.nb_devices):
                device = self.devices[device_id]
                tasks.append(device.get_task(step, t))
                state[device_id] = device.get_state(step, t)

            # Decision-Making Stage
            if isinstance(policy, OptimalDecisionMaker):
                for device_id in range(self.nb_devices):
                    q_idxs[device_id], offloading_decisions[device_id] = policy(step, t, device, self.edge)
            elif isinstance(policy, MultiTaskPPGPolicy):
                for device_id in range(self.nb_devices):
                    q_idxs[device_id], offloading_decisions[device_id] = policy(state[device_id])
            elif isinstance(policy, CentralizedMultiTaskPPGPolicy):
                q_idxs, offloading_decisions = policy(state)
            # tasks arrived
            # TODO: decision making section
            # resource_allocation_profile = [tasks[i].get_computational_intensity(q_idxs[i]) for i in
            #                                range(self.nb_devices)]
            # resource_allocation_profile *= (offloading_decisions != 0)
            # resource_allocation_profile = resource_allocation_profile / np.sum(resource_allocation_profile)

            rewards = []
            for device_id in range(self.nb_devices):
                task, device = tasks[device_id], self.devices[device_id]
                q_idx, offloading_decision = q_idxs[device_id], offloading_decisions[device_id]

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
                    comp_intensities = np.array(
                        [tasks[i].get_computational_intensity(q_idx) for i in range(self.nb_devices)])
                    offloaded_computations = ((offloading_decisions != 0).astype(int) * comp_intensities)
                    allocated_portion = offloaded_computations[device_id] / offloaded_computations.sum()

                    exec_time = self.edge.process(arrival_edge, comp_intensity, allocated_portion, True)
                    # edge -> device
                    rx_time, rx_energy = device.receive(task_response_size, ch_idx)

                    processing_time = tx_time + exec_time + rx_time
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
                rewards.append((task.get_psnr(q_idx), processing_time, energy_consumption))

                # Reward recording
                if isinstance(policy, MultiTaskPPGPolicy):
                    policy.record_reward(task.get_psnr(q_idx), processing_time, energy_consumption, t >= 35)
            if isinstance(policy, CentralizedMultiTaskPPGPolicy):
                policy.record_reward(rewards, t >= 35)
            t += dt
            step += 1

        return self.stats
