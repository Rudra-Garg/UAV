# maddpg_agent.py
"""
Implements the MADDPG agents for the inner loop (UAV positioning).

This file contains two main classes:
- DDPGAgent: A single agent with an Actor network for choosing actions.
- MADDPGController: Manages a collection of DDPGAgents and a centralized
  Critic network for training them. It coordinates action selection,
  learning, and model saving/loading for the entire team of UAVs.
"""
import os

import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from config import *
from replay_buffer import ReplayBuffer
from rl_networks import Actor, Critic


class DDPGAgent:
    """A single agent in the MADDPG setup."""
    def __init__(self, state_dim, action_dim, agent_id):
        self.id = agent_id
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.target_actor = Actor(state_dim, action_dim).to(DEVICE)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=MADDPG_LEARNING_RATE_ACTOR)
        # Initialize target actor with the same weights as the main actor
        self.target_actor.load_state_dict(self.actor.state_dict())

    def select_action(self, state):
        """Selects a deterministic action based on the current policy."""
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy().flatten()

    def update_target_actor(self):
        """Performs a soft update of the target actor's weights."""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(MADDPG_TAU * param.data + (1.0 - MADDPG_TAU) * target_param.data)

class MADDPGController:
    """Manages all DDPG agents and the centralized critic."""
    def __init__(self, num_agents, state_dim, action_dim):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.agents = [DDPGAgent(state_dim, action_dim, i) for i in range(num_agents)]

        # The critic is centralized and takes joint state and action information.
        self.critic = Critic(num_agents, state_dim, action_dim).to(DEVICE)
        self.target_critic = Critic(num_agents, state_dim, action_dim).to(DEVICE)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=MADDPG_LEARNING_RATE_CRITIC)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # A single replay buffer is shared by all agents.
        self.memory = ReplayBuffer(MADDPG_BUFFER_SIZE, MADDPG_BATCH_SIZE)

    def select_actions(self, states, evaluation=False):
        """
        Gets actions from all agents. Adds noise for exploration during training.

        Args:
            states (list of np.array): The list of local states for each agent.
            evaluation (bool): If True, disables exploration noise.

        Returns:
            list of np.array: The list of chosen actions for each agent.
        """
        actions = []
        for i, state in enumerate(states):
            action = self.agents[i].select_action(state)
            # Add Gaussian noise for exploration during training.
            if not evaluation:
                action += np.random.normal(0, 0.1, size=self.action_dim)
            # Clip the action to be within a valid range (e.g., -1 to 1).
            actions.append(np.clip(action, -1, 1))
        return actions

    def learn(self):
        """
        Trains all agent actors and the centralized critic using a batch of
        experiences from the shared replay buffer.
        """
        if len(self.memory) < MADDPG_BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()

        # Reshape tensors to reflect the multi-agent structure
        states = states.view(MADDPG_BATCH_SIZE, self.num_agents, self.state_dim)
        actions = actions.view(MADDPG_BATCH_SIZE, self.num_agents, self.action_dim)
        rewards = rewards.view(MADDPG_BATCH_SIZE, self.num_agents, 1)
        next_states = next_states.view(MADDPG_BATCH_SIZE, self.num_agents, self.state_dim)
        dones = dones.view(MADDPG_BATCH_SIZE, self.num_agents, 1)

        # --- Update Critic ---
        with torch.no_grad():
            # Get the next actions from the target actors for the next states
            next_actions = torch.stack(
                [self.agents[i].target_actor(next_states[:, i, :]) for i in range(self.num_agents)], dim=1)
            # Flatten states and actions for critic input
            flat_next_states = next_states.reshape(MADDPG_BATCH_SIZE, -1)
            flat_next_actions = next_actions.reshape(MADDPG_BATCH_SIZE, -1)
            # Calculate the target Q-value
            target_q = self.target_critic(flat_next_states, flat_next_actions)
            expected_q = rewards[:, 0, :] + (MADDPG_GAMMA * target_q * (1 - dones[:, 0, :]))

        flat_states = states.reshape(MADDPG_BATCH_SIZE, -1)
        flat_actions = actions.reshape(MADDPG_BATCH_SIZE, -1)
        # Get the current Q-value from the critic
        current_q = self.critic(flat_states, flat_actions)

        # Calculate and backpropagate the critic's loss
        critic_loss = F.mse_loss(current_q, expected_q)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # --- Update Actors ---
        # Get actions from the current policies (not target)
        policy_actions = torch.stack([self.agents[i].actor(states[:, i, :]) for i in range(self.num_agents)], dim=1)
        flat_policy_actions = policy_actions.reshape(MADDPG_BATCH_SIZE, -1)

        # The actor loss is the negative of the critic's Q-value output.
        # This encourages the actor to produce actions that the critic values highly.
        actor_loss = -self.critic(flat_states, flat_policy_actions).mean()

        # Backpropagate the actor loss to update each agent's actor network.
        for agent in self.agents: agent.optimizer_actor.zero_grad()
        actor_loss.backward()
        for agent in self.agents: agent.optimizer_actor.step()

    def update_targets(self):
        """Soft update all target networks (actors and critic)."""
        for agent in self.agents:
            agent.update_target_actor()

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(MADDPG_TAU * param.data + (1.0 - MADDPG_TAU) * target_param.data)

    def save(self, directory):
        """Saves the actor and critic networks to files."""
        if not os.path.exists(directory):
            os.makedirs(directory)
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), os.path.join(directory, f'maddpg_actor_{i}.pth'))
        torch.save(self.critic.state_dict(), os.path.join(directory, 'maddpg_critic.pth'))

    def load(self, directory):
        """Loads the actor and critic networks from files."""
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(torch.load(os.path.join(directory, f'maddpg_actor_{i}.pth')))
            agent.target_actor.load_state_dict(agent.actor.state_dict())  # Copy to target net
            agent.actor.eval()
            agent.target_actor.eval()
        self.critic.load_state_dict(torch.load(os.path.join(directory, 'maddpg_critic.pth')))
        self.target_critic.load_state_dict(self.critic.state_dict())  # Copy to target net
        self.critic.eval()
        self.target_critic.eval()
