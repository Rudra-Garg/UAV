# environment.py
"""
Implements the VECN Environment.
UPDATED: Fixes numerical instability (exploding rewards) by normalizing
and clipping values in the state and reward calculations.
"""
import numpy as np

from config import *
from entities import Vehicle, UAV, CloudComputingCenter


class VECNEnvironment:
    # ... (init, reset, step are the same) ...
    def __init__(self):
        self.width = AREA_WIDTH
        self.height = AREA_HEIGHT
        self.vehicles = []
        self.uavs = []
        self.ccc = CloudComputingCenter()
        self.time_step = 0
        self.total_tasks_in_step = 0
        self.completed_tasks_in_step = 0

    def reset(self, num_uavs=0, num_vehicles=NUM_VEHICLES):
        self.time_step = 0
        self.uavs = [UAV(i) for i in range(num_uavs)]
        self.vehicles = [Vehicle(i) for i in range(num_vehicles)]
        for v in self.vehicles:
            v.generate_tasks()
        return self.get_maddpg_states()

    def step(self, actions):
        for i, uav in enumerate(self.uavs):
            displacement = np.array([actions[i][0], actions[i][1], 0]) * UAV_MAX_SPEED
            uav.move(displacement)
        for vehicle in self.vehicles:
            vehicle.move()
        self._simulate_task_offloading()
        rewards = self._calculate_rewards()
        next_states = self.get_maddpg_states()
        self.time_step += 1
        done = self.time_step >= INNER_STEPS
        return next_states, rewards, done

    def _simulate_task_offloading(self):
        for uav in self.uavs: uav.reset_for_episode()
        all_tasks = [task for vehicle in self.vehicles for task in vehicle.tasks if not task.is_completed]
        self.total_tasks_in_step = len(all_tasks)
        self.completed_tasks_in_step = 0

        for task in all_tasks:
            vehicle = self.vehicles[task.owner_id]
            nearby_uavs = [u for u in self.uavs if self.get_distance(u, vehicle) <= UAV_COMMUNICATION_RANGE]
            if not nearby_uavs: continue

            best_latency = float('inf')
            best_option = None

            for uav in nearby_uavs:
                if uav.has_service(task.service_type) and uav.F_remain > task.cpu_cycles_req:
                    t_trans = task.data_size_bits / (self.calculate_datarate_user_to_uav(vehicle, uav) * 1e6 + 1e-9)
                    t_compute = task.cpu_cycles_req / uav.F_remain
                    latency = t_trans + t_compute
                    if latency < best_latency and latency <= task.latency_constraint:
                        best_latency = latency
                        profit = self._calculate_earn_task(task, latency, uav)
                        best_option = ('local_uav', uav, latency, profit)

            if best_option is None:
                for uav in nearby_uavs:
                    t_trans_user_uav = task.data_size_bits / (
                                self.calculate_datarate_user_to_uav(vehicle, uav) * 1e6 + 1e-9)
                    t_trans_uav_ccc = task.data_size_bits / (self.calculate_datarate_uav_to_ccc(uav) * 1e6 + 1e-9)
                    latency = t_trans_user_uav + t_trans_uav_ccc
                    if latency < best_latency and latency <= task.latency_constraint:
                        best_latency = latency
                        profit = self._calculate_earn_task(task, latency)
                        best_option = ('cloud', uav, latency, profit)

            if best_option:
                task.is_completed = True
                task.completed_latency = best_option[2]
                task.profit_generated = best_option[3]
                self.completed_tasks_in_step += 1
                uav_involved = best_option[1]
                uav_involved.tasks_processed_count += 1
                uav_involved.profit_generated += task.profit_generated
                if best_option[0] == 'local_uav':
                    uav_involved.F_remain -= task.cpu_cycles_req

    def _calculate_earn_task(self, task, latency, uav=None):
        """
        Implements the profit calculation from Eq. 19 & 20.
        FIXED: Added clipping and normalization to prevent numerical explosion.
        """
        # Add a small epsilon to latency to prevent division by zero if latency is somehow 0
        latency = max(latency, 1e-9)
        term_latency = DELTA_LATENCY * (task.latency_constraint / latency)

        if uav:  # If processed on a UAV
            ec_ij = ENERGY_PER_SECOND_PROCESSING * latency
            term_size = DELTA_SIZE * (task.data_size_bits / 1e6)

            # --- FIX: Prevent division by small number ---
            # We normalize the remaining computation by the total, so it's a ratio (0 to 1).
            # We add a small epsilon to the denominator to prevent division by zero.
            normalized_f_remain = uav.F_remain / uav.F_total
            term_comp = DELTA_COMPUTATION * (ec_ij / (normalized_f_remain + 1e-9))

            # --- FIX: Clip the term to a reasonable maximum value ---
            term_comp = np.clip(term_comp, 0, 1000)

            return term_latency + term_size + term_comp
        else:  # If processed on cloud
            return term_latency

    def _calculate_rewards(self):
        """
        The reward is the total profit generated by each UAV in the step.
        FIXED: The final reward is now scaled to be in a stable range.
        """
        if not self.uavs: return []
        rewards = [uav.profit_generated for uav in self.uavs]
        global_reward = np.mean(rewards) if rewards else 0

        # --- FIX: Scale the final reward ---
        scaled_reward = global_reward * REWARD_SCALING_FACTOR
        return [scaled_reward] * len(self.uavs)

    def get_maddpg_states(self):
        """
        Generates local states for each UAV agent.
        FIXED: Normalize the profit feature in the state vector.
        """
        if not self.uavs: return []
        all_states = []
        for uav in self.uavs:
            # --- FIX: Normalize the profit by an arbitrary large number to keep it small ---
            # This prevents the network input from exploding.
            normalized_profit = uav.profit_generated / 1000.0

            state = [
                uav.position[0] / self.width,
                uav.position[1] / self.height,
                len([v for v in self.vehicles if self.get_distance(uav, v) <= UAV_COMMUNICATION_RANGE]),
                uav.tasks_processed_count,
                normalized_profit  # Use the normalized value
            ]
            all_states.append(np.array(state))
        return all_states

    def get_ddqn_state(self):
        """
        Generates the global state.
        FIXED: Net profit is also scaled before being returned.
        """
        num_uavs = len(self.uavs)
        total_profit = sum(uav.profit_generated for uav in self.uavs)
        total_cost = sum(BETA_MAINTENANCE + BETA_COMPUTATION * u.F_total for u in self.uavs)
        net_profit = total_profit - total_cost

        # --- FIX: Scale the net profit for the DDQN agent's state/reward ---
        scaled_net_profit = net_profit * REWARD_SCALING_FACTOR

        tasks_completed = sum(uav.tasks_processed_count for uav in self.uavs)
        avg_latency = np.mean([t.completed_latency for v in self.vehicles for t in v.tasks if
                               t.is_completed]) if tasks_completed > 0 else 0

        covered_vehicles = set()
        if num_uavs > 0:
            for uav in self.uavs:
                for v in self.vehicles:
                    if self.get_distance(uav, v) <= UAV_COMMUNICATION_RANGE:
                        covered_vehicles.add(v.id)

        return np.array([tasks_completed, num_uavs, total_cost, scaled_net_profit, avg_latency, len(covered_vehicles)])

    # ... (Communication Modeling functions remain the same, just adding safety epsilons) ...
    def get_distance(self, e1, e2):
        return np.linalg.norm(e1.position - e2.position)

    def calculate_datarate_user_to_uav(self, user, uav):
        avg_pl_db = self.get_average_path_loss(uav, user)
        return self.calculate_datarate(BANDWIDTH_UAV_USER, POWER_UAV_USER, avg_pl_db)

    def calculate_datarate_uav_to_ccc(self, uav):
        return self.calculate_datarate(BANDWIDTH_UAV_CCC, POWER_CCC, 60)

    def calculate_elevation_angle(self, uav, ground):
        dist_2d = np.linalg.norm(uav.position[:2] - ground.position[:2])
        return np.pi / 2 if dist_2d == 0 else np.arctan(uav.position[2] / dist_2d)

    def calculate_los_probability(self, uav, v):
        angle_deg = np.rad2deg(self.calculate_elevation_angle(uav, v))
        return 1 / (1 + LOS_X0 * np.exp(-LOS_Y0 * (angle_deg - LOS_X0)))

    def calculate_path_loss(self, d, is_los):
        fspl = 20 * np.log10(d) + 20 * np.log10(CARRIER_FREQUENCY) - 147.55
        return fspl + (ETA_LOS if is_los else ETA_NLOS)

    def get_average_path_loss(self, uav, v):
        d = self.get_distance(uav, v)
        if d == 0: return 0
        los_prob = self.calculate_los_probability(uav, v)
        pl_los = self.calculate_path_loss(d, True)
        pl_nlos = self.calculate_path_loss(d, False)
        avg_pl = los_prob * (10 ** (pl_los / 10)) + (1 - los_prob) * (10 ** (pl_nlos / 10))
        return 10 * np.log10(avg_pl)

    def calculate_datarate(self, bw_hz, p_watt, pl_db):
        noise_watt = 10 ** ((NOISE_POWER_SPECTRAL_DENSITY - 30) / 10) * bw_hz
        rx_watt = 10 ** ((10 * np.log10(p_watt * 1000) - pl_db - 30) / 10)
        snr = rx_watt / noise_watt
        # Add safety epsilon to bitrate calculation
        return (bw_hz * np.log2(1 + snr)) if snr > 0 else 0
