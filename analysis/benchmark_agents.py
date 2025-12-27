import numpy as np
from scipy.cluster.vq import kmeans

from config import *


class OUPRS_Agent:
    """One UAV, Pure Random Scheme"""

    def __init__(self, env):
        self.num_uavs = 1

    def select_actions(self, env, states):
        return [(np.random.rand(2) * 2 - 1) * UAV_MAX_SPEED]


class OUPOS_Agent:
    """One UAV, Position Optimized Scheme (Center of Mass)"""

    def __init__(self, env):
        self.num_uavs = 1

    def select_actions(self, env, states):
        if not env.vehicles:
            return [[0, 0]]

        # FIXED: Use .values() to iterate over vehicle objects
        vehicle_list = list(env.vehicles.values()) if isinstance(env.vehicles, dict) else env.vehicles

        if not vehicle_list:
            return [[0, 0]]

        vehicle_positions = np.array([v.position[:2] for v in vehicle_list])
        center_of_mass = np.mean(vehicle_positions, axis=0)

        # Direction from current UAV pos to center of mass
        uav_position = env.uavs[0].position[:2]
        direction_vector = center_of_mass - uav_position

        norm = np.linalg.norm(direction_vector)
        if norm > 0:
            direction_vector /= norm

        return [direction_vector * UAV_MAX_SPEED]


class MRUPRS_Agent:
    """Multi-UAV, Random Position Random Scheme"""

    def __init__(self, env):
        self.num_uavs = np.random.randint(3, 8)

    def select_actions(self, env, states):
        return [(np.random.rand(2) * 2 - 1) * UAV_MAX_SPEED for _ in range(self.num_uavs)]


class MRUPOS_Agent:
    """Multi-UAV, Random-Position Optimized Scheme (K-Means Clustering)"""

    def __init__(self, env):
        # Random between 3-7 for variability in benchmarks
        self.num_uavs = np.random.randint(3, 8)

    def select_actions(self, env, states):
        # FIXED: Use .values() to iterate over vehicle objects
        vehicle_list = list(env.vehicles.values()) if isinstance(env.vehicles, dict) else env.vehicles

        if not vehicle_list:
            return [np.zeros(2) for _ in range(self.num_uavs)]

        vehicle_positions = np.array([v.position[:2] for v in vehicle_list])

        # Handle edge case: fewer vehicles than UAVs
        k = min(len(vehicle_positions), self.num_uavs)
        if k < 1:
            return [np.zeros(2) for _ in range(self.num_uavs)]

        # Compute centroids
        centroids, _ = kmeans(vehicle_positions, k)

        # If fewer centroids than UAVs, recycle centroids
        actions = []
        for i, uav in enumerate(env.uavs):
            target = centroids[i % len(centroids)]
            direction = target - uav.position[:2]

            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm

            actions.append(direction * UAV_MAX_SPEED)

        return actions


class MOUPRS_Agent:
    """Multi-Objective UAV Position Random Scheme (Uses DDQN for number, Random for position)"""

    def __init__(self, env, ddqn_agent):
        self.ddqn_agent = ddqn_agent
        self.num_uavs = None

    def select_actions(self, env, states):
        if self.num_uavs is None:
            # Decide num_uavs once based on initial state (or update periodically)
            outer_state = env.get_ddqn_state()
            self.num_uavs = self.ddqn_agent.select_action(outer_state, evaluation=True) + 1

        # Return random velocities
        return [(np.random.rand(2) * 2 - 1) * UAV_MAX_SPEED for _ in range(self.num_uavs)]
