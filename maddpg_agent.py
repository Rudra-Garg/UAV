# maddpg_agent.py
"""
Implements the MADDPG agents for the inner loop (UAV positioning).
"""
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from config import *
from replay_buffer import ReplayBuffer
from rl_networks import Actor, Critic


class DDPGAgent:
    """A single DDPG agent for MADDPG."""

    def __init__(self, state_dim, action_dim, agent_id):
        self.id = agent_id
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.target_actor = Actor(state_dim, action_dim).to(DEVICE)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=MADDPG_LEARNING_RATE_ACTOR)
        self.target_actor.load_state_dict(self.actor.state_dict())

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy().flatten()

    def update_target_actor(self):
        """Soft update for the target actor network."""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(MADDPG_TAU * param.data + (1.0 - MADDPG_TAU) * target_param.data)


class MADDPGController:
    def __init__(self, num_agents, state_dim, action_dim):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.agents = [DDPGAgent(state_dim, action_dim, i) for i in range(num_agents)]

        self.critic = Critic(num_agents, state_dim, action_dim).to(DEVICE)
        self.target_critic = Critic(num_agents, state_dim, action_dim).to(DEVICE)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=MADDPG_LEARNING_RATE_CRITIC)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.memory = ReplayBuffer(MADDPG_BUFFER_SIZE, MADDPG_BATCH_SIZE)

    def select_actions(self, states):
        """Get actions from all agents."""
        actions = []
        for i, state in enumerate(states):
            action = self.agents[i].select_action(state)
            # Add some noise for exploration
            action += np.random.normal(0, 0.1, size=self.action_dim)
            actions.append(np.clip(action, -1, 1))
        return actions

    def learn(self):
        """Train all agents and the centralized critic."""
        if len(self.memory) < MADDPG_BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()

        # Reshape to (batch_size, num_agents, dim)
        states = states.view(MADDPG_BATCH_SIZE, self.num_agents, self.state_dim)
        actions = actions.view(MADDPG_BATCH_SIZE, self.num_agents, self.action_dim)
        rewards = rewards.view(MADDPG_BATCH_SIZE, self.num_agents, 1)
        next_states = next_states.view(MADDPG_BATCH_SIZE, self.num_agents, self.state_dim)
        dones = dones.view(MADDPG_BATCH_SIZE, self.num_agents, 1)

        # --- Update Critic ---
        with torch.no_grad():
            next_actions = torch.stack(
                [self.agents[i].target_actor(next_states[:, i, :]) for i in range(self.num_agents)], dim=1)

            # Flatten for critic input
            flat_next_states = next_states.reshape(MADDPG_BATCH_SIZE, -1)
            flat_next_actions = next_actions.reshape(MADDPG_BATCH_SIZE, -1)

            target_q = self.target_critic(flat_next_states, flat_next_actions)

            # Use per-agent reward and done signal
            expected_q = rewards[:, 0, :] + (MADDPG_GAMMA * target_q * (1 - dones[:, 0, :]))

        # Flatten for critic input
        flat_states = states.reshape(MADDPG_BATCH_SIZE, -1)
        flat_actions = actions.reshape(MADDPG_BATCH_SIZE, -1)

        current_q = self.critic(flat_states, flat_actions)

        critic_loss = F.mse_loss(current_q, expected_q)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # --- Update Actors ---
        # We need to re-calculate actions using the current policy for the gradient
        policy_actions = torch.stack([self.agents[i].actor(states[:, i, :]) for i in range(self.num_agents)], dim=1)
        flat_policy_actions = policy_actions.reshape(MADDPG_BATCH_SIZE, -1)

        # The actor loss is the negative of the Q-value, we want to maximize Q
        actor_loss = -self.critic(flat_states, flat_policy_actions).mean()

        # Zero out all actor optimizers
        for agent in self.agents:
            agent.optimizer_actor.zero_grad()

        actor_loss.backward()

        # Step each actor optimizer
        for agent in self.agents:
            agent.optimizer_actor.step()

    def update_targets(self):
        """Soft update all target networks."""
        for agent in self.agents:
            agent.update_target_actor()

        # Update target critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(MADDPG_TAU * param.data + (1.0 - MADDPG_TAU) * target_param.data)
