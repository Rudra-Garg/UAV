# environment.py
"""
Implements the VECN (Vehicular Edge Computing Network) Environment.
Manages the state of all entities (UAVs, Vehicles), their interactions,
and the communication models based on the research paper. It provides
methods to reset the simulation, execute a step, and generate state/reward
information for the RL agents.
"""
import numpy as np

from config import *
from entities import Vehicle, UAV, CloudComputingCenter


class VECNEnvironment:
    def __init__(self):
        """Initializes the environment's properties."""
        self.width = AREA_WIDTH
        self.height = AREA_HEIGHT
        self.vehicles = []
        self.uavs = []
        self.ccc = CloudComputingCenter()
        self.time_step = 0

    def reset(self, num_uavs=0, num_vehicles=NUM_VEHICLES):
        """
        Resets the environment to a new initial state.

        Args:
            num_uavs (int): The number of UAVs to create for this episode.
            num_vehicles (int): The number of vehicles to create. This is
                                essential for running evaluation scenarios.

        Returns:
            list of np.array: The initial list of local states for the MADDPG agents.
        """
        self.time_step = 0
        self.vehicles = [Vehicle(i) for i in range(num_vehicles)]
        self.uavs = [UAV(i) for i in range(num_uavs)]
        return self.get_maddpg_states()

    def step(self, actions):
        """
        Advances the environment by one time step based on agent actions.

        Args:
            actions (list of np.array): A list of actions, one for each UAV.

        Returns:
            tuple: A tuple containing (next_states, rewards, done).
        """
        # 1. Update UAV positions based on the actions from the MADDPG controller.
        # The action is a normalized vector [-1, 1], which we scale by a max speed.
        for i, uav in enumerate(self.uavs):
            displacement = np.array([actions[i][0], actions[i][1], 0]) * UAV_MAX_SPEED
            uav.move(displacement)

        # 2. Move all vehicles according to their own velocity.
        for vehicle in self.vehicles:
            vehicle.move()

        # 3. Calculate rewards based on the new state and get the next state.
        rewards = self._calculate_rewards()
        next_states = self.get_maddpg_states()

        # 4. Check if the inner episode has terminated.
        self.time_step += 1
        done = self.time_step >= INNER_STEPS

        return next_states, rewards, done

    def _calculate_rewards(self):
        """
        Calculates a reward for each UAV using vectorized operations for speed.
        The reward function encourages vehicle coverage and discourages collisions.

        Returns:
            list of float: A list containing the shared global reward for each agent.
        """
        if not self.uavs:
            return []

        num_uavs = len(self.uavs)
        rewards = np.zeros(num_uavs)

        # Vectorized Coverage Reward: Calculate all-pairs distances between UAVs and Vehicles
        uav_positions = np.array([uav.position for uav in self.uavs])
        vehicle_positions = np.array([v.position for v in self.vehicles])
        dist_matrix_uv = np.linalg.norm(uav_positions[:, np.newaxis, :] - vehicle_positions[np.newaxis, :, :], axis=2)
        vehicles_in_range = np.sum(dist_matrix_uv <= UAV_COMMUNICATION_RANGE, axis=1)
        rewards += vehicles_in_range

        # Vectorized Collision Penalty: Calculate all-pairs distances between UAVs
        if num_uavs > 1:
            dist_matrix_uu = np.linalg.norm(uav_positions[:, np.newaxis, :] - uav_positions[np.newaxis, :, :], axis=2)
            np.fill_diagonal(dist_matrix_uu, np.inf)  # Ignore self-distance
            collision_penalties = np.sum(dist_matrix_uu < MIN_UAV_DISTANCE, axis=1) * -10.0  # Strong penalty
            rewards += collision_penalties

        # Use a shared global reward (the average) to encourage cooperative behavior.
        global_reward = np.mean(rewards)
        return [global_reward] * num_uavs

    def get_maddpg_states(self):
        """
        Returns a list of local states, one for each MADDPG agent (UAV).
        The state features are normalized to improve learning stability.

        Returns:
            list of np.array: The list of local states.
        """
        if not self.uavs:
            return []

        all_states = []
        for uav in self.uavs:
            # Normalize position to be between 0 and 1.
            norm_pos_x = uav.position[0] / self.width
            norm_pos_y = uav.position[1] / self.height

            users_in_range = 0
            for vehicle in self.vehicles:
                if self.get_distance(uav, vehicle) <= UAV_COMMUNICATION_RANGE:
                    users_in_range += 1

            # Placeholders for more complex state features from the paper.
            tasks_processed = 0;
            profit = 0
            state = [norm_pos_x, norm_pos_y, users_in_range, tasks_processed, profit]
            all_states.append(np.array(state))

        return all_states

    def get_ddqn_state(self):
        """
        Returns the global state of the entire system for the DDQN agent.
        This includes aggregate metrics like total profit and cost.

        Returns:
            np.array: The global state vector.
        """
        num_uavs = len(self.uavs)

        covered_vehicles = set()
        if num_uavs > 0:
            for uav in self.uavs:
                for v in self.vehicles:
                    if self.get_distance(uav, v) <= UAV_COMMUNICATION_RANGE:
                        covered_vehicles.add(v.id)
        users_covered = len(covered_vehicles)

        # More realistic profit model for fair evaluation
        revenue = users_covered * 1.5
        cost = num_uavs * 10.0
        total_profit = revenue - cost

        tasks_completed = users_covered * TASKS_PER_VEHICLE
        total_latency = 1.0 / (num_uavs + 1e-6)

        return np.array([tasks_completed, num_uavs, cost, total_profit, total_latency, users_covered])

    # --- Communication Modeling Methods (Unchanged) ---
    def get_distance(self, entity1, entity2):
        return np.linalg.norm(entity1.position - entity2.position)

    def calculate_elevation_angle(self, uav, ground_entity):
        dist_2d = np.linalg.norm(uav.position[:2] - ground_entity.position[:2])
        if dist_2d == 0: return np.pi / 2
        return np.arctan(uav.position[2] / dist_2d)

    def calculate_los_probability(self, uav, vehicle):
        angle_rad = self.calculate_elevation_angle(uav, vehicle)
        angle_deg = np.rad2deg(angle_rad)
        return 1 / (1 + LOS_X0 * np.exp(-LOS_Y0 * (angle_deg - LOS_X0)))

    def calculate_path_loss(self, distance, is_los):
        fspl = 20 * np.log10(distance) + 20 * np.log10(CARRIER_FREQUENCY) - 147.55
        return fspl + (ETA_LOS if is_los else ETA_NLOS)

    def get_average_path_loss(self, uav, vehicle):
        distance = self.get_distance(uav, vehicle)
        if distance == 0: return 0
        los_prob = self.calculate_los_probability(uav, vehicle)
        pl_los = self.calculate_path_loss(distance, is_los=True)
        pl_nlos = self.calculate_path_loss(distance, is_los=False)
        avg_pl_linear = los_prob * (10 ** (pl_los / 10)) + (1 - los_prob) * (10 ** (pl_nlos / 10))
        return 10 * np.log10(avg_pl_linear)

    def calculate_datarate(self, bandwidth_hz, power_watt, path_loss_db):
        noise_power_dbm = NOISE_POWER_SPECTRAL_DENSITY + 10 * np.log10(bandwidth_hz)
        noise_power_watt = 10 ** ((noise_power_dbm - 30) / 10)
        received_power_dbm = 10 * np.log10(power_watt * 1000) - path_loss_db
        received_power_watt = 10 ** ((received_power_dbm - 30) / 10)
        snr = received_power_watt / noise_power_watt
        rate_bps = bandwidth_hz * np.log2(1 + snr)
        return rate_bps / 1e6