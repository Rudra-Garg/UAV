# ddqn_agent.py
"""
Implements the DDQN agent for the outer loop (UAV number selection).
"""
import random

import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from config import *
from replay_buffer import ReplayBuffer
from rl_networks import DDQN_Network


class DDQNAgent:
    def __init__(self, state_dim, action_space):
        self.state_dim = state_dim
        self.action_space = action_space
        self.epsilon = DDQN_EPSILON_START

        self.policy_net = DDQN_Network(state_dim, action_space).to(DEVICE)
        self.target_net = DDQN_Network(state_dim, action_space).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=DDQN_LEARNING_RATE)
        self.memory = ReplayBuffer(DDQN_BUFFER_SIZE, DDQN_BATCH_SIZE)

    def select_action(self, state):
        """Selects an action using an epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_space)  # Random action

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.policy_net(state)
            return np.argmax(q_values.cpu().data.numpy())

    def learn(self):
        """Trains the agent by replaying experiences."""
        if len(self.memory) < DDQN_BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()

        # We need to reshape actions to be used as indices
        actions = actions.long()

        # Get Q-values for current states from the policy network
        current_q = self.policy_net(states).gather(1, actions)

        # Get next Q-values from the target network
        # Detach ensures that gradients don't flow back to the target network
        next_q_values = self.target_net(next_states).detach()

        # Select the best action from the policy network for the next state
        best_next_actions = self.policy_net(next_states).detach().argmax(1).unsqueeze(1)

        # Get the Q-value for the best next action from the target network (Double Q-learning)
        max_next_q = next_q_values.gather(1, best_next_actions)

        # Compute the expected Q-values
        expected_q = rewards + (DDQN_GAMMA * max_next_q * (1 - dones))

        # Compute loss
        loss = F.mse_loss(current_q, expected_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > DDQN_EPSILON_END:
            self.epsilon *= DDQN_EPSILON_DECAY

    def update_target_network(self):
        """Soft update of the target network's weights."""
        target_net_weights = self.target_net.state_dict()
        policy_net_weights = self.policy_net.state_dict()
        for key in policy_net_weights:
            target_net_weights[key] = policy_net_weights[key] * DDQN_TAU + target_net_weights[key] * (1 - DDQN_TAU)
        self.target_net.load_state_dict(target_net_weights)
