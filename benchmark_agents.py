# benchmark_agents.py (updated with new agents)
import numpy as np
from scipy.cluster.vq import kmeans
from scipy.optimize import minimize

from config import *


class OUPRS_Agent:
    def __init__(self, env):
        self.num_uavs = 1

    def select_actions(self, env, states):
        return [(np.random.rand(2) * 2 - 1) * UAV_MAX_SPEED]

class OUPOS_Agent:
    def __init__(self, env):
        self.num_uavs = 1

    def select_actions(self, env, states):
        if not env.vehicles:
            return [[0, 0]]
        vehicle_positions = np.array([v.position[:2] for v in env.vehicles])
        center_of_mass = np.mean(vehicle_positions, axis=0)
        uav_position = env.uavs[0].position[:2]
        direction_vector = center_of_mass - uav_position
        norm = np.linalg.norm(direction_vector)
        if norm > 0:
            direction_vector /= norm
        return [direction_vector * UAV_MAX_SPEED]

class MRUPRS_Agent:
    def __init__(self, env, num_uavs=5):
        self.num_uavs = num_uavs

    def select_actions(self, env, states):
        return [(np.random.rand(2) * 2 - 1) * UAV_MAX_SPEED for _ in range(self.num_uavs)]


# New: Multi-UAV Random-Position Optimize Scheme (random num_uavs, optimized positions via K-means)
class MRUPOS_Agent:
    def __init__(self, env):
        self.num_uavs = np.random.randint(3, 8)  # Random between 3-7 for variability

    def select_actions(self, env, states):
        if not env.vehicles:
            return [np.zeros(2) for _ in range(self.num_uavs)]
        vehicle_positions = np.array([v.position[:2] for v in env.vehicles])
        centroids, _ = kmeans(vehicle_positions, self.num_uavs)
        actions = []
        for i, uav in enumerate(env.uavs):
            direction = centroids[i] - uav.position[:2]
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm
            actions.append(direction * UAV_MAX_SPEED)
        return actions


class MOUPRS_Agent:
    def __init__(self, env, ddqn_agent):
        self.ddqn_agent = ddqn_agent
        self.num_uavs = None

    def select_actions(self, env, states):
        if self.num_uavs is None:
            outer_state = env.get_ddqn_state()
            self.num_uavs = self.ddqn_agent.select_action(outer_state, evaluation=True) + 1
            # Removed: env.reset(num_uavs=self.num_uavs)
        return [(np.random.rand(2) * 2 - 1) * UAV_MAX_SPEED for _ in range(self.num_uavs)]


# New: Multi-UAV Particle Swarm Optimize Scheme (fixed num_uavs, PSO for positions)
class MUPSOS_Agent:
    def __init__(self, env, num_uavs=5):
        self.num_uavs = num_uavs

    def select_actions(self, env, states):
        def cost_func(pos_flat):
            positions = pos_flat.reshape((self.num_uavs, 2))
            total_cost = 0
            vehicle_pos = np.array([v.position[:2] for v in env.vehicles])
            for pos in positions:
                dists = np.linalg.norm(vehicle_pos - pos, axis=1)
                total_cost += np.sum(dists)  # Minimize sum distances
            return total_cost

        initial_pos = np.random.uniform(0, AREA_WIDTH, (self.num_uavs * 2,))
        res = minimize(cost_func, initial_pos, method='Nelder-Mead', options={'maxiter': 50})
        optimized_pos = res.x.reshape((self.num_uavs, 2))
        actions = []
        for i, uav in enumerate(env.uavs):
            direction = optimized_pos[i] - uav.position[:2]
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm
            actions.append(direction * UAV_MAX_SPEED)
        return actions


# New: MKUPRS (assuming Multi-Kmeans-UAV Position Random Scheme, random num, K-means positions)
class MKUPRS_Agent:
    def __init__(self, env):
        self.num_uavs = np.random.randint(3, 8)

    def select_actions(self, env, states):
        return MRUPOS_Agent(env).select_actions(env, states)  # Reuse K-means logic
