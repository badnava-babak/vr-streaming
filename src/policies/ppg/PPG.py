import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Dirichlet
from torch.distributions import Categorical
import numpy as np

from src.policies.ppg.buffer import RolloutBuffer
from src.policies.ppg.model import ActorCritic

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


class PPG:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 centralized, action_std_init=0.6, action_max=1, testing_stage=False, mid_layer_size=656):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.aux_training_epochs = 6
        self.ppg_repeat = 2

        self.training_epoch = 0
        self.centralized = centralized

        self.buffer = RolloutBuffer()
        self.ppg_buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, centralized, action_std_init, action_max,
                                  mid_layer_size, testing_stage).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9, verbose=True)
        # self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.5, total_iters=4)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.0002, max_lr=0.005,
                                                           step_size_up=1000,
                                                           step_size_down=2000, mode="exp_range",
                                                           cycle_momentum=False)

        self.policy_old = ActorCritic(state_dim, action_dim, centralized, action_std_init,
                                      action_max, mid_layer_size, testing_stage).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss(reduction='none')
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def set_action_std_testing_stage(self, min_action_std):
        if min_action_std != self.action_std:
            print("--------------------------------------------------------------------------------------------")
            if self.has_continuous_action_space:
                self.action_std = min_action_std
                self.set_action_std(self.action_std)
                print("setting actor output action_std for testing stage to : ", self.action_std)
            else:
                print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, comp_dist_action=None):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val, aux_value = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.detach().cpu().numpy().flatten()

    def update(self):
        self.training_epoch += 1

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        if self.centralized:
            advantages = advantages.repeat(old_actions.shape[1], 1).T

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy, aux_value = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            aux_value = torch.squeeze(aux_value)

            # Finding the ratio (pi_theta / pi_theta__old)
            diff = logprobs - old_logprobs.detach()
            # ratios = torch.exp((diff - diff.mean(dim=0)) / (diff.std() + 1e-7))
            # ratios = torch.exp((diff - diff.min(dim=0)[0]) / (diff.max(dim=0)[0] - diff.min(dim=0)[0] + 1e-7))

            ratios = torch.exp(diff)
            # ratios = torch.exp(logprobs - old_logprobs.detach())
            # if not self.has_continuous_action_space:
            #     ratios = torch.prod(ratios, dim=1)
            #     dist_entropy = torch.sum(dist_entropy, dim=1)

            # Finding Surrogate Loss

            # surr1 = torch.clamp(ratios, 0.2, 1.8) * advantages
            # surr1 = ratios * advantages
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            ppo_loss = -torch.min(surr1, surr2)
            ppg_loss = torch.where(torch.less(advantages, 0), torch.maximum(ppo_loss, 3. * advantages), ppo_loss)

            loss = ppg_loss - .01 * dist_entropy
            # loss = ppg_loss
            value_loss = 0.5 * self.MseLoss(state_values, rewards)
            if self.centralized:
                value_loss = value_loss.repeat(loss.shape[1], 1).T

            loss += value_loss
            # loss += self.MseLoss(aux_value, rewards)
            # loss = torch.clamp(loss, -0.5, 0.5)

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.)
            self.optimizer.step()
        # if self.training_epoch % 100 == 0:
        #     self.scheduler.step()
        # if torch.isnan(self.policy.actor.action_logits.weight).any():
        #     print('Nan')

        self.ppg_buffer.states += old_states
        self.ppg_buffer.actions += old_actions
        self.ppg_buffer.state_values += old_state_values
        self.ppg_buffer.logprobs += old_logprobs

        # ppg here
        if self.training_epoch % self.ppg_repeat == 0:
            # convert list to tensor
            old_states = torch.squeeze(torch.stack(self.ppg_buffer.states, dim=0)).detach().to(device)
            old_actions = torch.squeeze(torch.stack(self.ppg_buffer.actions, dim=0)).detach().to(device)
            old_logprobs = torch.squeeze(torch.stack(self.ppg_buffer.logprobs, dim=0)).detach().to(device)
            old_state_values = torch.squeeze(torch.stack(self.ppg_buffer.state_values, dim=0)).detach().to(device)

            for i in range(self.aux_training_epochs):
                # Evaluating old actions and values
                logprobs, state_values, dist_entropy, aux_value = self.policy.evaluate(old_states, old_actions)

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                aux_value = torch.squeeze(aux_value)

                loss = self.MseLoss(old_state_values, state_values)
                loss += self.MseLoss(old_state_values, aux_value)
                loss += 0.5 * self.kl_loss(logprobs, old_logprobs)

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.)
                self.optimizer.step()

            self.ppg_buffer.clear()
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
