# benchmark_agents.py
"""
Implements the simple, non-learning benchmark agents for comparison during evaluation.
"""
import numpy as np

from config import *


class OUPRS_Agent:
    """One-UAV Position Random Scheme."""

    def __init__(self, env):
        self.num_uavs = 1
        env.reset(num_uavs=self.num_uavs)

    def select_actions(self, states):
        # Return one random action vector
        return [(np.random.rand(2) * 2 - 1) * UAV_MAX_SPEED]


class OUPOS_Agent:
    """One-UAV Position Optimize Scheme."""

    def __init__(self, env):
        self.num_uavs = 1
        self.env = env
        self.env.reset(num_uavs=self.num_uavs)

    def select_actions(self, states):
        # Simple optimization: move towards the center of mass of all vehicles.
        if not self.env.vehicles:
            return [[0, 0]]

        vehicle_positions = np.array([v.position[:2] for v in self.env.vehicles])
        center_of_mass = np.mean(vehicle_positions, axis=0)

        uav_position = self.env.uavs[0].position[:2]
        direction_vector = center_of_mass - uav_position

        # Normalize the direction vector
        norm = np.linalg.norm(direction_vector)
        if norm > 0:
            direction_vector /= norm

        return [direction_vector * UAV_MAX_SPEED]


class MRUPRS_Agent:
    """Multi-UAV Random-Position Scheme."""

    def __init__(self, env, num_uavs=5):  # Use a fixed number for consistent comparison
        self.num_uavs = num_uavs
        env.reset(num_uavs=self.num_uavs)

    def select_actions(self, states):
        # Return a list of random action vectors
        return [(np.random.rand(2) * 2 - 1) * UAV_MAX_SPEED for _ in range(self.num_uavs)]
