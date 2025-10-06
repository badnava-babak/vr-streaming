from typing import Tuple, Callable

import numpy as np

from src.nodes.edge_server import EdgeNode
from src.nodes.vr_device import VRDevice


class DecisionMaker(Callable):
    def __init__(self):
        pass

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass


class OptimalDecisionMaker(DecisionMaker):
    def __init__(self, action_dim: Tuple[int, int], weights: Tuple[float, float, float]):
        super().__init__()
        self.action_dim = action_dim
        self.q_levels = action_dim[0]
        self.nb_channels = action_dim[1] - 1
        self.w = weights

    def __call__(self, ctr: int, t: float,
                 vr_device: VRDevice, edge_server: EdgeNode) -> Tuple[int, int]:
        task = vr_device.get_task(ctr, t)
        task_sizes = task.get_sizes()
        task_comp_intensities = task.get_computational_intensities()
        task_response_sizes = task.get_response_sizes()

        # tx_times = np.zeros((self.q_levels, self.nb_channels))
        # tx_energies = np.zeros((self.q_levels, self.nb_channels))
        # rx_times = np.zeros((self.q_levels, self.nb_channels))
        # rx_energies = np.zeros((self.q_levels, self.nb_channels))
        # for q in range(0, self.q_levels):
        #     for ch in range(0, self.nb_channels):
        #         tx_times[q][ch], tx_energies[q][ch] = vr_device.channels[ch].time_to_tx(task_sizes[q], False)
        #         rx_times[q][ch], rx_energies[q][ch] = vr_device.channels[ch].time_to_rx(task_response_sizes[q], False)

        obj_values = np.zeros(self.action_dim)
        stall_times = np.zeros(self.action_dim)
        total_times = np.zeros(self.action_dim)
        psnr_vals = np.zeros(self.action_dim)
        energy_vals = np.zeros(self.action_dim)
        for q in range(0, self.q_levels):
            for act in range(0, self.action_dim[1]):
                if act == 0:  # local computation
                    total_time, total_energy = vr_device.process(t, task_comp_intensities[q], False)
                else:
                    ch = act - 1
                    # device -> edge
                    tx_time, tx_energy = vr_device.channels[ch].time_to_tx(task_sizes[q], False)
                    arrival_edge = t + tx_time
                    # process on edge server
                    # comp_intensities = np.array(
                    #     [tasks[i].get_computational_intensity(q) for i in range(edge_server.nb_devices)])
                    # offloaded_computations = ((offloading_decisions != 0).astype(int) * comp_intensities)
                    # allocated_portion = offloaded_computations[device_id] / offloaded_computations.sum()

                    exec_time = edge_server.process(arrival_edge, task_comp_intensities[q], 1., False)
                    # edge -> device
                    rx_time, rx_energy = vr_device.channels[ch].time_to_rx(task_response_sizes[q], False)

                    total_time = tx_time + exec_time + rx_time
                    total_energy = tx_energy + rx_energy

                psnr = task.get_psnr(q)
                stall_times[q][act] = max(0., total_time - vr_device.buffer)
                total_times[q][act] = total_time
                psnr_vals[q][act] = psnr
                energy_vals[q][act] = total_energy

                obj_values[q][act] = self.w[0] * psnr
                # obj_values[q][act] -= self.w[1] * stall_times[q][act]
                obj_values[q][act] -= self.w[1] * total_time
                obj_values[q][act] -= self.w[2] * total_energy
                # if (1-total_time) < 0.:
                #     obj_values[q][act] += 10 * (1 - total_time)


        psnr_z = (psnr_vals - psnr_vals.min()) / (psnr_vals.max() - psnr_vals.min() + 1.e-10)
        # stall_z = (stall_times - stall_times.min()) / (stall_times.max() - stall_times.min() + 1.e-10)
        stall_z = (stall_times) / (stall_times.max() + 1.e-10)
        energy_z = (energy_vals - energy_vals.min()) / (energy_vals.max() - energy_vals.min() + 1.e-10)
        # obj_values = self.w[0] * psnr_z - self.w[1] * stall_z - self.w[2] * energy_z

        obj_values += 500 * (1 - total_times).clip(-np.inf, 0)

        optimal_q = np.argmax(obj_values) // 4
        optimal_ch = np.argmax(obj_values) % 4
        # if 1 - total_times[optimal_q, optimal_ch] < 0:
        #     print('Deadline Violation', 1- total_times)
            # raise Exception('Deadline Violation')
        return optimal_q, optimal_ch
