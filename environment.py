# environment.py
"""
Implements the VECN (Vehicular Edge Computing Network) Environment.
Manages the state of all entities and the communication models.
Now includes logic for state/reward generation and action handling.
"""
import numpy as np

from config import *
from entities import Vehicle, UAV, CloudComputingCenter


class VECNEnvironment:
    def __init__(self):
        self.width = AREA_WIDTH
        self.height = AREA_HEIGHT
        self.vehicles = []
        self.uavs = []
        self.ccc = CloudComputingCenter()
        self.time_step = 0

    def reset(self, num_uavs=0):
        """Resets the environment. Can now specify number of UAVs."""
        self.time_step = 0
        self.vehicles = [Vehicle(i) for i in range(NUM_VEHICLES)]
        self.uavs = [UAV(i) for i in range(num_uavs)]
        return self.get_maddpg_states()

    def add_uavs(self, num_uavs):
        """Helper to add UAVs, now done primarily in reset."""
        self.uavs = [UAV(i) for i in range(num_uavs)]
        print(f"Added {len(self.uavs)} UAVs to the environment.")

    def step(self, actions):
        """
        Advance the environment by one time step based on agent actions.
        'actions' is a list of actions, one for each UAV.
        """
        # 1. Update UAV positions based on actions
        for i, uav in enumerate(self.uavs):
            # Action is [-1, 1], scale it to a reasonable movement speed
            displacement = np.array([actions[i][0], actions[i][1], 0]) * UAV_MAX_SPEED
            uav.move(displacement)

        # 2. Move all vehicles
        for vehicle in self.vehicles:
            vehicle.move()

        # 3. Calculate rewards and get next state
        # For simplicity, we'll use a placeholder reward logic.
        # A more complex implementation would simulate task offloading.
        rewards = self._calculate_rewards()
        next_states = self.get_maddpg_states()

        # Done is true if the inner episode ends
        self.time_step += 1
        done = self.time_step >= INNER_STEPS

        return next_states, rewards, done

    # --- State and Reward Calculation ---

    def _calculate_rewards(self):
        """
        Calculates a reward for each UAV.
        A simple reward: +1 for each vehicle in range, -0.1 for being too close to another UAV.
        """
        rewards = np.zeros(len(self.uavs))
        for i, uav in enumerate(self.uavs):
            # Reward for coverage
            vehicles_in_range = 0
            for vehicle in self.vehicles:
                if self.get_distance(uav, vehicle) <= UAV_COMMUNICATION_RANGE:
                    vehicles_in_range += 1
            rewards[i] += vehicles_in_range

            # Penalty for collision risk
            for j, other_uav in enumerate(self.uavs):
                if i != j and self.get_distance(uav, other_uav) < MIN_UAV_DISTANCE:
                    rewards[i] -= 10  # Strong penalty

        # Return a shared global reward (average of individual rewards) to encourage cooperation
        global_reward = np.mean(rewards)
        return [global_reward] * len(self.uavs)

    def get_maddpg_states(self):
        """
        Returns a list of local states, one for each MADDPG agent.
        State: [norm_pos_x, norm_pos_y, users_in_range, tasks_processed (placeholder), profit (placeholder)]
        """
        if not self.uavs:
            return []

        all_states = []
        for uav in self.uavs:
            # Normalize position
            norm_pos_x = uav.position[0] / self.width
            norm_pos_y = uav.position[1] / self.height

            # Count users in range
            users_in_range = 0
            for vehicle in self.vehicles:
                if self.get_distance(uav, vehicle) <= UAV_COMMUNICATION_RANGE:
                    users_in_range += 1

            # Placeholders for more complex state features
            tasks_processed = 0
            profit = 0

            state = [norm_pos_x, norm_pos_y, users_in_range, tasks_processed, profit]
            all_states.append(np.array(state))

        return all_states

    def get_ddqn_state(self):
        """
        Returns the global state for the DDQN agent.
        State: [tasks_completed, num_uavs, total_cost, total_profit, total_latency, users_covered]
        This is a placeholder as we aren't simulating the full economy yet.
        """
        num_uavs = len(self.uavs)

        # Placeholder calculations
        tasks_completed = np.random.randint(50, 100) * num_uavs
        total_cost = 10 * num_uavs  # Simplified cost model
        total_profit = (np.random.rand() * 50 - 10) * num_uavs
        total_latency = 1.0 / (num_uavs + 1e-6)

        covered_vehicles = set()
        for uav in self.uavs:
            for v in self.vehicles:
                if self.get_distance(uav, v) <= UAV_COMMUNICATION_RANGE:
                    covered_vehicles.add(v.id)
        users_covered = len(covered_vehicles)

        return np.array([tasks_completed, num_uavs, total_cost, total_profit, total_latency, users_covered])

    # --- Communication Modeling (Unchanged from Phase 1) ---
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
