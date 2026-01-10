# replay_buffer.py
"""
A generic Replay Buffer for Deep Reinforcement Learning.
"""
import random
from collections import deque

import numpy as np
import torch

from config import DEVICE


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        # Convert to torch tensors
        # For multi-agent scenarios, states/actions/rewards/next_states can be lists of arrays
        # We need to stack them properly to preserve the multi-agent dimension
        
        # Check if we have multi-agent data (lists of arrays) or single-agent (arrays)
        first_state = experiences[0][0]
        if isinstance(first_state, (list, tuple)):
            # Multi-agent case: stack along a new axis to preserve agent dimension
            states = torch.from_numpy(np.array([e[0] for e in experiences if e is not None])).float().to(DEVICE)
            actions = torch.from_numpy(np.array([e[1] for e in experiences if e is not None])).float().to(DEVICE)
            rewards = torch.from_numpy(np.array([e[2] for e in experiences if e is not None])).float().to(DEVICE)
            next_states = torch.from_numpy(np.array([e[3] for e in experiences if e is not None])).float().to(DEVICE)
            
            # For dones, handle both scalar and list cases
            dones_list = [e[4] for e in experiences if e is not None]
            if isinstance(dones_list[0], bool):
                # Scalar done for all agents - broadcast it
                dones = torch.from_numpy(np.array([[d] * len(first_state) for d in dones_list]).astype(np.uint8)).float().to(DEVICE)
            else:
                dones = torch.from_numpy(np.array(dones_list).astype(np.uint8)).float().to(DEVICE)
        else:
            # Single-agent case: use vstack as before
            states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(DEVICE)
            actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float().to(DEVICE)
            rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(DEVICE)
            next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(DEVICE)
            dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
