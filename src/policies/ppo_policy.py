from typing import Tuple

import numpy as np

from src.policies.ppg_policy import PPGPolicy
from src.policies.ppo.PPO import PPO


class PPOPolicy(PPGPolicy):
    def __init__(self, num_channels: int,
                 num_users: int, weights: Tuple[float, float, float]):
        super().__init__()
        self.w = weights

        K_epochs = 80  # update policy for K epochs in one PPO update\
        eps_clip = 0.2  # clip parameter for PPO
        gamma = 0.0  # discount factor
        lr_actor = 0.0003  # learning rate for actor network
        lr_critic = 0.001  # learning rate for critic network

        L = 7
        state_dim = (27,)
        action_dim = num_channels + 1
        independent = False
        self.ppg_agents = {}

        for n in range(num_users):
            self.ppg_agents[n] = PPO(state_dim, action_dim, lr_actor, lr_critic,
                                     gamma, K_epochs, eps_clip, independent,
                                     mid_layer_size=208,)

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass
    def update(self):
        for i, ppg_agent in self.ppg_agents.items():
            ppg_agent.update()
    def record_reward(self, rewards: Tuple[float, float, float], done):
        # r = np.array(rewards)
        # reward = 0
        # for r in rewards:
        #     reward += self.w[0] * r[0] - self.w[1] * r[1] - self.w[2] * r[2]
        #     if (1 - r[1]) < 0.:
        #         reward += 500 * (1 - r[1])

        for i, r in enumerate(rewards):
            self.ppg_agents[i].buffer.rewards.append(np.mean(rewards))
            self.ppg_agents[i].buffer.is_terminals.append(done)

    def __call__(self, states) -> Tuple[np.array, np.array]:
        qs = []
        chs = []

        for i, ppg_agent in self.ppg_agents.items():
            state = states[i]
            taks_info = []
            ch_info = []
            for k, v in state.items():
                if 'task_' in k:
                    taks_info.append(v)
                elif 'ch_' in k:
                    ch_info.append(v)

            ch_info = np.array(ch_info).flatten()
            s = np.concatenate([np.array(taks_info).flatten(), ch_info])
            action = ppg_agent.select_action(s)[0]
            # 7 x 4
            _q = 3
            _ch = action % 4
            qs.append(_q)
            chs.append(_ch)
        return np.array(qs), np.array(chs)
