# rl_networks.py
"""
Contains the PyTorch neural network architectures for all RL agents.
- DDQN_Network: For the outer-layer agent.
- Actor: For the inner-layer MADDPG agents.
- Critic: For the centralized MADDPG critic.
"""
import torch.nn as nn
import torch.nn.functional as F

from config import *


class DDQN_Network(nn.Module):
    """Neural Network for the DDQN (outer-layer) agent."""

    def __init__(self, state_dim, action_space):
        super(DDQN_Network, self).__init__()
        # As per paper: two hidden layers of 64x64
        self.layer1 = nn.Linear(state_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, action_space)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.output_layer(x)


class Actor(nn.Module):
    """Actor Network for the MADDPG (inner-layer) agents."""

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # As per paper: three hidden layers 128x128x64
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        # Use tanh to bound the action output between -1 and 1
        return torch.tanh(self.output_layer(x))


class Critic(nn.Module):
    """Critic Network for the MADDPG (centralized training)."""

    def __init__(self, num_agents, state_dim, action_dim):
        super(Critic, self).__init__()
        # The critic's input is the joint state and joint action of all agents
        joint_state_dim = num_agents * state_dim
        joint_action_dim = num_agents * action_dim
        input_dim = joint_state_dim + joint_action_dim

        # As per paper: two hidden layers of 64x64
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, joint_state, joint_action):
        # Concatenate joint state and action along the last dimension
        x = torch.cat([joint_state, joint_action], dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.output_layer(x)
