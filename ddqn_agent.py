# ddqn_agent.py
"""
Implements the DDQN agent for the outer loop (UAV number selection).

This agent learns a policy to decide how many UAVs should be deployed
based on a high-level, global state of the environment. It includes
methods for action selection, learning from a replay buffer, and
saving/loading its learned network weights.
"""
import os
import random

import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from config import *
from replay_buffer import ReplayBuffer
from rl_networks import DDQN_Network


class DDQNAgent:
    def __init__(self, state_dim, action_space):
        """Initializes the DDQN agent's components."""
        self.state_dim = state_dim
        self.action_space = action_space
        self.epsilon = DDQN_EPSILON_START

        # Create the policy and target networks
        self.policy_net = DDQN_Network(state_dim, action_space).to(DEVICE)
        self.target_net = DDQN_Network(state_dim, action_space).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode

        # Setup the optimizer and replay memory
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=DDQN_LEARNING_RATE)
        self.memory = ReplayBuffer(DDQN_BUFFER_SIZE, DDQN_BATCH_SIZE)

    def select_action(self, state, evaluation=False):
        """
        Selects an action using an epsilon-greedy policy for training, or a purely greedy
        policy for evaluation.

        Args:
            state (np.array): The current global state.
            evaluation (bool): If True, disables exploration (epsilon=0).

        Returns:
            int: The chosen action (number of UAVs to deploy, 0-indexed).
        """
        # During evaluation, we always choose the best known action.
        if evaluation:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q_values = self.policy_net(state)
                return np.argmax(q_values.cpu().data.numpy())

        # During training, use the epsilon-greedy strategy.
        if random.random() < self.epsilon:
            return random.randrange(self.action_space)  # Explore: select a random action

        # Exploit: select the best action from the policy network.
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.policy_net(state)
            return np.argmax(q_values.cpu().data.numpy())

    def learn(self):
        """Trains the agent by sampling a batch of experiences from the replay buffer."""
        # Do not learn until the memory has enough experiences.
        if len(self.memory) < DDQN_BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()
        actions = actions.long()  # Actions need to be long type for gather()

        # Get Q-values for the actions that were actually taken.
        current_q = self.policy_net(states).gather(1, actions)

        # In Double DQN, we use the policy_net to select the best action for the next state,
        # but we get the Q-value for that action from the target_net.
        with torch.no_grad():
            best_next_actions = self.policy_net(next_states).detach().argmax(1).unsqueeze(1)
            next_q_values = self.target_net(next_states).detach()
            max_next_q = next_q_values.gather(1, best_next_actions)

        # Compute the target Q-value using the Bellman equation.
        expected_q = rewards + (DDQN_GAMMA * max_next_q * (1 - dones))

        # Calculate the Mean Squared Error loss between current and expected Q-values.
        loss = F.mse_loss(current_q, expected_q)

        # Perform backpropagation.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon to reduce exploration over time.
        if self.epsilon > DDQN_EPSILON_END:
            self.epsilon *= DDQN_EPSILON_DECAY

    def update_target_network(self):
        """
        Performs a soft update of the target network's weights, blending them
        with the policy network's weights. This improves training stability.
        """
        target_net_weights = self.target_net.state_dict()
        policy_net_weights = self.policy_net.state_dict()
        for key in policy_net_weights:
            target_net_weights[key] = policy_net_weights[key] * (1 - DDQN_TAU) + policy_net_weights[key] * DDQN_TAU
        self.target_net.load_state_dict(target_net_weights)

    def save(self, directory):
        """Saves the policy network's learned weights to a file."""
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.policy_net.state_dict(), os.path.join(directory, 'ddqn_policy_net.pth'))

    def load(self, directory):
        """Loads learned weights from a file into both policy and target networks."""
        self.policy_net.load_state_dict(torch.load(os.path.join(directory, 'ddqn_policy_net.pth')))
        # Copy the loaded weights to the target network as well.
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Set networks to evaluation mode.
        self.policy_net.eval()
        self.target_net.eval()
