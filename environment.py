# environment.py
"""
Implements the VECN Environment.
UPDATED: This version is heavily optimized for speed using Numba's JIT compiler.
The core simulation logic is decorated to be compiled into fast machine code,
addressing the CPU bottleneck without changing any simulation parameters.
"""
import numpy as np
from numba import jit, njit  # Import the JIT compiler from Numba

from config import *
from entities import Vehicle, UAV, CloudComputingCenter


# --- Numba-Optimized Helper Functions ---
# Numba works best on simple functions that operate on primitive types like arrays.
# We extract the core distance calculation to be jitted for maximum speed.
@njit
def pairwise_distance_numba(pos_array1, pos_array2):
    """Calculates pairwise Euclidean distances between two sets of 3D points."""
    num_pos1 = pos_array1.shape[0]
    num_pos2 = pos_array2.shape[0]
    distances = np.empty((num_pos1, num_pos2))
    for i in range(num_pos1):
        for j in range(num_pos2):
            dx = pos_array1[i, 0] - pos_array2[j, 0]
            dy = pos_array1[i, 1] - pos_array2[j, 1]
            dz = pos_array1[i, 2] - pos_array2[j, 2]
            distances[i, j] = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return distances

class VECNEnvironment:
    def __init__(self):
        # ... (init is the same) ...
        self.width = AREA_WIDTH;
        self.height = AREA_HEIGHT
        self.vehicles = [];
        self.uavs = []
        self.ccc = CloudComputingCenter()
        self.time_step = 0
        self.total_tasks_in_step = 0
        self.completed_tasks_in_step = 0

    def reset(self, num_uavs=0, num_vehicles=NUM_VEHICLES):
        # ... (reset is the same) ...
        self.time_step = 0
        self.uavs = [UAV(i) for i in range(num_uavs)]
        self.vehicles = [Vehicle(i) for i in range(num_vehicles)]
        for v in self.vehicles:
            v.generate_tasks()
        return self.get_maddpg_states()

    def step(self, actions):
        # ... (step is the same) ...
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

    # --- THE CORE OPTIMIZATION ---
    # We cannot jit the entire function because it uses Python objects (self, lists of classes).
    # Instead, we will vectorize the distance calculations inside it.
    def _simulate_task_offloading(self):
        """
        High-fidelity simulation of task offloading.
        OPTIMIZED: Uses vectorized distance calculations to speed up finding nearby UAVs.
        """
        for uav in self.uavs: uav.reset_for_episode()
        all_tasks = [task for vehicle in self.vehicles for task in vehicle.tasks if not task.is_completed]
        self.total_tasks_in_step = len(all_tasks)
        self.completed_tasks_in_step = 0

        if not self.uavs or not all_tasks: return

        # --- Vectorization START ---
        # Get all vehicle and UAV positions into NumPy arrays for fast processing
        vehicle_positions = np.array([self.vehicles[task.owner_id].position for task in all_tasks])
        uav_positions = np.array([u.position for u in self.uavs])

        # Calculate a single, large distance matrix: [num_tasks x num_uavs]
        # This one operation replaces thousands of individual distance calculations in the loop.
        dist_matrix = pairwise_distance_numba(vehicle_positions, uav_positions)
        # --- Vectorization END ---

        for i, task in enumerate(all_tasks):
            # Find nearby UAVs using the pre-computed distance matrix
            nearby_uav_indices = np.where(dist_matrix[i, :] <= UAV_COMMUNICATION_RANGE)[0]
            if nearby_uav_indices.size == 0:
                continue

            best_latency = float('inf')
            best_option = None

            # Now we loop only through the few nearby UAVs, not all of them
            for uav_idx in nearby_uav_indices:
                uav = self.uavs[uav_idx]
                if uav.has_service(task.service_type) and uav.F_remain > task.cpu_cycles_req:
                    datarate = self.calculate_datarate_user_to_uav(self.vehicles[task.owner_id], uav)
                    t_trans = task.data_size_bits / (datarate * 1e6 + 1e-9)
                    t_compute = task.cpu_cycles_req / uav.F_remain
                    latency = t_trans + t_compute
                    if latency < best_latency and latency <= task.latency_constraint:
                        best_latency = latency
                        profit = self._calculate_earn_task(task, latency, uav)
                        best_option = ('local_uav', uav, latency, profit)

            if best_option is None:
                for uav_idx in nearby_uav_indices:
                    uav = self.uavs[uav_idx]
                    datarate_user = self.calculate_datarate_user_to_uav(self.vehicles[task.owner_id], uav)
                    t_trans_user_uav = task.data_size_bits / (datarate_user * 1e6 + 1e-9)
                    t_trans_uav_ccc = task.data_size_bits / (self.calculate_datarate_uav_to_ccc(uav) * 1e6 + 1e-9)
                    latency = t_trans_user_uav + t_trans_uav_ccc
                    if latency < best_latency and latency <= task.latency_constraint:
                        best_latency = latency
                        profit = self._calculate_earn_task(task, latency)
                        best_option = ('cloud', uav, latency, profit)

            if best_option:
                task.is_completed = True
                task.completed_latency, task.profit_generated = best_option[2], best_option[3]
                self.completed_tasks_in_step += 1
                uav_involved = best_option[1]
                uav_involved.tasks_processed_count += 1
                uav_involved.profit_generated += task.profit_generated
                if best_option[0] == 'local_uav':
                    uav_involved.F_remain -= task.cpu_cycles_req

    # By adding the @jit decorator, Numba will compile this function into machine code.
    # Note: Using "objectmode=True" allows Numba to handle the class objects,
    # though it's not as fast as pure "nopython" mode, it still provides a good speedup.
    @jit(nopython=False, forceobj=True)
    def _calculate_earn_task(self, task, latency, uav=None):
        latency = max(latency, 1e-9)
        term_latency = DELTA_LATENCY * (task.latency_constraint / latency)
        if uav:
            ec_ij = ENERGY_PER_SECOND_PROCESSING * latency
            term_size = DELTA_SIZE * (task.data_size_bits / 1e6)
            normalized_f_remain = uav.F_remain / uav.F_total
            term_comp = DELTA_COMPUTATION * (ec_ij / (normalized_f_remain + 1e-9))
            term_comp = np.clip(term_comp, 0, 1000)
            return term_latency + term_size + term_comp
        else:
            return term_latency

    def _calculate_rewards(self):
        # ... (this function is already fast, no change needed) ...
        if not self.uavs: return []
        rewards = [uav.profit_generated for uav in self.uavs]
        global_reward = np.mean(rewards) if rewards else 0
        return [global_reward * REWARD_SCALING_FACTOR] * len(self.uavs)

    def get_maddpg_states(self):
        # ... (this function is already fast, no change needed) ...
        if not self.uavs: return []
        all_states = []
        for uav in self.uavs:
            normalized_profit = uav.profit_generated / 1000.0
            state = [
                uav.position[0] / self.width,
                uav.position[1] / self.height,
                len([v for v in self.vehicles if self.get_distance(uav, v) <= UAV_COMMUNICATION_RANGE]),
                uav.tasks_processed_count,
                normalized_profit
            ]
            all_states.append(np.array(state))
        return all_states

    def get_ddqn_state(self):
        # ... (this function is already fast, no change needed) ...
        num_uavs = len(self.uavs)
        total_profit = sum(uav.profit_generated for uav in self.uavs)
        total_cost = sum(BETA_MAINTENANCE + BETA_COMPUTATION * u.F_total for u in self.uavs)
        net_profit = total_profit - total_cost
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

    # --- Communication Modeling ---
    # These are mostly NumPy, but Numba can still speed up the pure math parts.
    @jit(nopython=True)
    def get_distance(self, e1_pos, e2_pos):
        return np.linalg.norm(e1_pos - e2_pos)

    def calculate_datarate_user_to_uav(self, user, uav):
        avg_pl_db = self.get_average_path_loss(uav, user)
        return self.calculate_datarate(BANDWIDTH_UAV_USER, POWER_UAV_USER, avg_pl_db)

    def calculate_datarate_uav_to_ccc(self, uav):
        return self.calculate_datarate(BANDWIDTH_UAV_CCC, POWER_CCC, 60)

    # Note: We refactor get_distance slightly to pass arrays, making it JIT-compatible.
    # The original get_distance is kept for compatibility with older code if needed.
    def _get_distance_obj(self, e1, e2):
        return np.linalg.norm(e1.position - e2.position)

    @jit(nopython=True)
    def _calculate_datarate_jit(self, bw_hz, p_watt, pl_db, noise_const):
        noise_watt = noise_const * bw_hz
        rx_watt = 10 ** ((10 * np.log10(p_watt * 1000) - pl_db - 30) / 10)
        snr = rx_watt / noise_watt
        return (bw_hz * np.log2(1 + snr)) if snr > 0 else 0

    def calculate_datarate(self, bw_hz, p_watt, pl_db):
        noise_const = 10 ** ((NOISE_POWER_SPECTRAL_DENSITY - 30) / 10)
        return self._calculate_datarate_jit(bw_hz, p_watt, pl_db, noise_const) / 1e6

    # ... The rest of the communication functions remain, as they are called by the main logic ...
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
        d = self._get_distance_obj(uav, v)
        if d == 0: return 0
        los_prob = self.calculate_los_probability(uav, v)
        pl_los = self.calculate_path_loss(d, True);
        pl_nlos = self.calculate_path_loss(d, False)
        avg_pl = los_prob * (10 ** (pl_los / 10)) + (1 - los_prob) * (10 ** (pl_nlos / 10))
        return 10 * np.log10(avg_pl)