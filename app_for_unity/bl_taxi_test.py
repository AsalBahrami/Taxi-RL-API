import gymnasium as gym
import torch
import torch.nn as nn
from torch.nn import functional as F


# In[23]:


# neuronale Netzwerk (beste Entscheidungen)
# DQN definieren(PyTorch)
# Taxi-Umgebung erstellen/simulation(gym)

class BLDuelingDQN(nn.Module):
    """Dueling DQN that computes Q-values through value and advantage."""

    def __init__(self, state_shape, action_shape, number_of_nodes):
        super(BLDuelingDQN, self).__init__()
        self.emb = nn.Embedding(state_shape, 4)
        self.fc1 = nn.Linear(4, number_of_nodes[0])
        self.fc_h_v = nn.Linear(number_of_nodes[0], number_of_nodes[1])
        self.fc_h_a = nn.Linear(number_of_nodes[0], number_of_nodes[1])
        self.fc_z_v = nn.Linear(number_of_nodes[1], 1)
        self.fc_z_a = nn.Linear(number_of_nodes[1], action_shape)
        self.explain = False
        self.visits = torch.zeros(500)

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.int64)
        self.visits[obs] += 0.00001
        x = self.emb(obs)
        x = F.leaky_relu(self.fc1(x.squeeze(dim=1)))
        value = self.fc_z_v(F.leaky_relu(self.fc_h_v(x)))  # Value stream
        advantage = self.fc_z_a(F.leaky_relu(
            self.fc_h_a(x)))  # Advantage stream

        # Combine two streams of DQN
        Q_values = value + advantage - advantage.mean(1, keepdim=True)
        if self.explain:
            return Q_values, state, value, advantage
        else:
            return Q_values, state
        # TODO: Don't use embedding


# In[29]:

# gesamte Umgebung decoden in einer Liste

# i is the current state of environment
def decode(i):
    out = []
    out.append(i % 4)
    i //= 4
    out.append(i % 5)
    i //= 5
    out.append(i % 5)
    i //= 5
    out.append(i)

    if not (0 <= i < 5):
        print("decoding error")
        return []

    return list(reversed(out))


def encode(state_list):
    i = state_list[0]
    i = i * 5 + state_list[1]
    i = i * 5 + state_list[2]
    i = i * 4 + state_list[3]
    return i


# added to view for debugging gym error
env = gym.make('Taxi-v3', render_mode='ansi')
