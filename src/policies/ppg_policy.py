from typing import Tuple

from src.policies.optimal_dm import DecisionMaker
import numpy as np

from src.policies.ppg.PPG import PPG


class PPGPolicy(DecisionMaker):
    def update(self):
        self.ppg_agent.update()

    def save_model(self, path):
        if isinstance(self, PPGPolicy):
            self.ppg_agent.save(path)

    def load_model(self, path):
        if isinstance(self, MultiTaskPPGPolicy) or isinstance(self, CentralizedMultiTaskPPGPolicy):
            self.ppg_agent.load(path)


class MultiTaskPPGPolicy(PPGPolicy):
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
        action_dim = (L, num_channels + 1)
        independent = False
        self.ppg_agent = PPG(state_dim, action_dim, lr_actor, lr_critic,
                             gamma, K_epochs, eps_clip, independent,
                             mid_layer_size=208, mode='DeCentralized')

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
            action = self.ppg_agent.select_action(s)[0]
            # 7 x 4
            _q = action // 4
            _ch = action % 4
            qs.append(_q)
            chs.append(_ch)
        return np.array(qs), np.array(chs)


class CentralizedMultiTaskPPGPolicy(PPGPolicy):
    def __init__(self, num_channels: int,
                 num_users: int, weights: Tuple[float, float, float]):
        super().__init__()
        self.w = weights

        K_epochs = 80  # update policy for K epochs in one PPO update\
        eps_clip = 0.2  # clip parameter for PPO
        gamma = 0.00  # discount factor
        lr_actor = 0.0003  # learning rate for actor network
        lr_critic = 0.001  # learning rate for critic network

        L = 7
        state_dim = (num_users, 27,)
        action_dim = (num_users, L, num_channels + 1)
        independent = True
        self.ppg_agent = PPG(state_dim, action_dim, lr_actor, lr_critic,
                             gamma, K_epochs, eps_clip, independent,
                             mid_layer_size=208, mode='Centralized')

    def record_reward(self, rewards: Tuple[float, float, float], done):
        # r = np.array(rewards)
        reward = 0
        for r in rewards:
            reward += self.w[0] * r[0] - self.w[1] * r[1] - self.w[2] * r[2]
            if (1 - r[1]) < 0.:
                reward += 500 * (1 - r[1])
        self.ppg_agent.buffer.rewards.append(reward/len(rewards))
        self.ppg_agent.buffer.is_terminals.append(done)

    def __call__(self, state) -> Tuple[np.array, np.array]:
        ms = []
        for i in range(len(state)):
            taks_info = []
            ch_info = []
            for k, v in state[i].items():
                if 'task_' in k:
                    taks_info.append(v)
                elif 'ch_' in k:
                    ch_info.append(v)

            ch_info = np.array(ch_info).flatten()
            s = np.concatenate([np.array(taks_info).flatten(), ch_info])
            ms.append(s)
        ms = np.array(ms).flatten()
        actions = self.ppg_agent.select_action(ms)

        q_s = []
        ch_s = []
        for action in actions:
            # 7 x 4
            _q = action // 4
            _ch = action % 4
            q_s.append(_q)
            ch_s.append(_ch)

        return np.array(q_s), np.array(ch_s)

