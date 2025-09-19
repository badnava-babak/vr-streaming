import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, centralized, action_max, mid_layer_size,
                 testing_stage=False) -> None:
        super().__init__()
        # self.l1 = nn.Linear(state_dim, 64)
        # self.a1 = nn.Tanh()
        # self.l2 = nn.Linear(64, 64)
        # self.a2 = nn.Tanh()
        # self.l3 = nn.Linear(64, action_dim)

        self.action_dim = action_dim
        self.centralized = centralized
        self.testing_stage = testing_stage

        self.action_max = action_max
        if centralized:
            self.shared_layers = nn.Sequential(
                nn.Linear(state_dim[0] * state_dim[1], mid_layer_size),
                nn.ReLU(),
                nn.Linear(mid_layer_size, 128),
                nn.ReLU(),
            )
            self.aux_value = nn.Linear(128, 1)
        else:
            self.shared_layers = nn.Sequential(
                # nn.Conv1d(state_dim[0], 16, 3, stride=1),
                nn.Linear(state_dim[0], mid_layer_size),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(mid_layer_size, 128),
                nn.ReLU(),
            )

            self.aux_value = nn.Linear(128, 1)

        if centralized:
            self.action_logits = nn.Linear(128, action_dim[0] * action_dim[1] * action_dim[2])
            self.reshaped_action = nn.Unflatten(-1, (action_dim[0], action_dim[1] * action_dim[2]))
            self.action = nn.Softmax(dim=-1)
        else:
            self.action_logits = nn.Linear(128, action_dim)
            # self.reshaped_action = nn.Unflatten(-1, self.action_dim)
            # self.reshaped_action = torch.reshape(self.action_logits, self.action_dim)
            self.action = nn.Softmax(dim=-1)

    def forward(self, x):
        emb = self.shared_layers(x)

        action_logits = self.action_logits(emb)

        if self.centralized:
            action_reshaped = self.reshaped_action(action_logits)
            action_out = self.action(action_reshaped)
        else:
            # if self.testing_stage:
            # reshaped_action = reshaped_action / 10
            action_out = self.action(action_logits)
            # action_out = self.reshaped_action(action_p)
            # action_out = torch.reshape(action_out, self.action_dim)

        aux_out = self.aux_value(emb)
        return action_out, aux_out


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, centralized, action_std_init, action_max,
                 mid_layer_size, testing_stage=False):
        super(ActorCritic, self).__init__()

        self.centralized = centralized
        self.testing_stage = testing_stage

        # actor
        self.actor = Actor(state_dim, action_dim, centralized, action_max, mid_layer_size, testing_stage)
        if centralized:
            self.critic = nn.Sequential(
                nn.Linear(state_dim[0] * state_dim[1], mid_layer_size),
                nn.ReLU(),
                nn.Linear(mid_layer_size, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        else:
            self.critic = nn.Sequential(
                # nn.Conv1d(state_dim[0], 16, 3, stride=1),
                nn.Linear(state_dim[0], mid_layer_size),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(mid_layer_size, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state, comp_dist_action=None):
        action_probs, aux_value = self.actor(torch.unsqueeze(state, 0))
        dist = Categorical(action_probs)
        if self.testing_stage:
            action = action_probs.argmax(dim=1)
        else:
            action = dist.sample()

        # action = torch.softmax(action_sampled, dim=1)
        # if self.has_continuous_action_space:

        action_logprob = dist.log_prob(action)
        state_val = self.critic(torch.unsqueeze(state, 0))

        return action.detach(), action_logprob.detach(), state_val.detach()[0], aux_value.detach()[0]

    def evaluate(self, state, action):
        action_probs, aux_value = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy, aux_value
