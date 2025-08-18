import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import *

device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()


def masked_mse_loss(prediction, target, mask):
    """
    Calculates the Mean Squared Error (MSE) loss only for masked elements.

    Args:
        prediction (torch.Tensor): The predicted tensor.
        target (torch.Tensor): The ground truth target tensor.
        mask (torch.Tensor): A binary mask tensor (1 for elements to include, 0 for elements to ignore).
                             It should have the same shape as prediction and target.

    Returns:
        torch.Tensor: The masked MSE loss.
    """
    # Calculate the element-wise squared difference
    squared_diff = (prediction - target) ** 2

    # Apply the mask to the squared differences
    # This sets the squared difference to 0 for masked-out elements
    masked_squared_diff = squared_diff * mask.float()

    # Sum the masked squared differences
    sum_masked_squared_diff = torch.sum(masked_squared_diff)

    # Calculate the number of non-zero elements in the mask
    num_masked_elements = torch.sum(mask.float())

    # Avoid division by zero if no elements are masked
    if num_masked_elements == 0:
        return torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
    else:
        # Calculate the masked MSE
        masked_mse = sum_masked_squared_diff / num_masked_elements
        return masked_mse


class ReplayBuffer:

    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def sample(self, n):
        size = len(self.actions)
        idx = np.random.randint(0, size, size=n)
        return [self.states[i] for i in idx], [self.actions[i] for i in idx], [self.rewards[i] for i in idx]


class Model(nn.Module):

    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()
        self.affine1 = nn.Linear(input_size, hidden_size)
        self.affine2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        return self.affine2(x)


class NeuralEpsilonGreedy:

    def __init__(self, context_dim, action_dim, epsilon=.2, hidden_size=128, lr=3e-4, reg=0.000625):
        self.action_dim = action_dim
        self.T = 0
        self.reg = reg
        self.epsilon = epsilon
        self.net = Model(context_dim, hidden_size, action_dim)
        self.hidden_size = hidden_size
        self.net.to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.device = device

        self.theta0 = torch.cat(
            [w.flatten() for w in self.net.parameters() if w.requires_grad]
        )
        self.buffer = ReplayBuffer()

    def take_action(self, context):
        context = torch.tensor(context, dtype=torch.float32)
        context = context.to(self.device)
        with torch.no_grad():
            p = self.net(context).cpu().numpy()

        if np.random.rand() > self.epsilon:
            action = np.argmax(p)
        else:
            action = np.random.choice(np.arange(self.action_dim))

        self.buffer.states.append(context)
        self.buffer.actions.append(action)

        return action

    def update2(self, context, action, reward):
        context = torch.tensor(context, dtype=torch.float32)
        context = context.to(self.device)
        self.buffer.add(context[action].cpu().numpy(), reward)
        self.T += 1
        self.train()
        if self.T % 3600 == 0:
            self.epsilon = max(self.epsilon - .02, 0.01)
            # print('Epsilon Updated to: {}'.format(self.epsilon))

    def update(self):
        # if self.T > self.action_dim and self.T % 10 == 0:
        self.T += 1
        if self.T % 2 == 0:
            self.epsilon = max(self.epsilon - .02, 0.01)
        for _ in range(2):
            x, y, z = self.buffer.sample(64)

            states = torch.squeeze(torch.stack(x, dim=0)).detach().to(device)
            actions = torch.tensor(np.array(y), dtype=torch.long).to(device)
            actions = F.one_hot(actions, num_classes=self.action_dim)
            rewards = torch.tensor(np.array(z), dtype=torch.float32).to(device)

            # x = torch.tensor(x, dtype=torch.float32).to(self.device)
            # y = torch.tensor(y, dtype=torch.float32).to(self.device).view(-1, 1)
            y_hat = self.net(states)
            y = rewards.reshape(-1, 1) * actions

            loss = masked_mse_loss(y_hat, y, actions)
            # loss = F.mse_loss(y_hat, y) * actions
            loss += self.reg * torch.norm(torch.cat(
                [w.flatten() for w in self.net.parameters() if w.requires_grad]
            ) - self.theta0) ** 2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    import torch
    import torch.nn as nn

