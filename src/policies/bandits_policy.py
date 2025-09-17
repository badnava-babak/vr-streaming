from typing import Tuple

from src.policies.bandits.epsilon_greedy import NeuralEpsilonGreedy
from src.policies.optimal_dm import DecisionMaker
import numpy as np

from src.policies.ppg_policy import PPGPolicy


class BanditPolicy(PPGPolicy):
    def __init__(self, num_channels: int,
                 num_users: int, weights: Tuple[float, float, float]):
        super().__init__()
        self.w = weights

        L = 7
        state_dim = 27
        action_dim = L * (num_channels + 1)
        independent = False
        self.ppg_agent = NeuralEpsilonGreedy(state_dim, action_dim,  epsilon=.2)

    # def record_reward(self, psnr, processing_time, energy_consumption, done):
    #     reward = self.w[0] * psnr - self.w[1] * processing_time - self.w[2] * energy_consumption
    #     if (1 - processing_time) < 0.:
    #         reward += 10 * (1 - processing_time)
    #     self.ppg_agent.buffer.rewards.append(reward)
    #     self.ppg_agent.buffer.is_terminals.append(done)

    def record_reward(self, rewards: Tuple[float, float, float], done):
        # r = np.array(rewards)
        reward = 0
        for r in rewards:
            reward += self.w[0] * r[0] - self.w[1] * r[1] - self.w[2] * r[2]
            if (1 - r[1]) < 0.:
                reward += 500 * (1 - r[1])
        for r in rewards:
            self.ppg_agent.buffer.rewards.append(reward/len(rewards))
            self.ppg_agent.buffer.is_terminals.append(done)

    def __call__(self, states) -> Tuple[np.array, np.array]:
        qs = []
        chs = []
        for key, state in states.items():
            taks_info = []
            ch_info = []
            for k, v in state.items():
                if 'task_' in k:
                    taks_info.append(v)
                elif 'ch_' in k:
                    ch_info.append(v)

            ch_info = np.array(ch_info).flatten()
            s = np.concatenate([np.array(taks_info).flatten(), ch_info])

            action = self.ppg_agent.take_action(s)

            # 7 x 4
            _q = action // 4
            _ch = action % 4
            qs.append(_q)
            chs.append(_ch)
        return np.array(qs), np.array(chs)

