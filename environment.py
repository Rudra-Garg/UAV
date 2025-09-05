# environment.py
"""
Implements the VECN Environment.
CORRECTED for Numba TypingError: The jitted function for calculating the
datarate has been moved outside the class to be a standalone helper function.
This is necessary because Numba's `nopython=True` mode cannot handle
class instances ('self') as arguments.
"""
import numpy as np
from numba import jit, njit
import math

from config import *
from entities import Vehicle, UAV, CloudComputingCenter


# --- Numba-Optimized Helper Functions ---

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


# --- Phase 4 CORRECTION: Moved from inside the class to be a standalone function ---
@jit(nopython=True)
def _calculate_datarate_numba(bw_hz, p_watt, pl_db, noise_const):
    """
    Numba-optimized core data rate calculation based on the Shannon-Hartley theorem.
    This is a pure function that does not depend on class state ('self').
    """
    noise_watt = noise_const * bw_hz
    # Convert power from W to dBm, apply path loss, then convert back to W for SNR calculation
    rx_watt_dbm = 10 * np.log10(p_watt * 1000) - pl_db
    rx_watt = 10 ** ((rx_watt_dbm - 30) / 10)
    snr = rx_watt / noise_watt
    return (bw_hz * np.log2(1 + snr)) if snr > 0 else 0.0


# --- End of CORRECTION ---


class VECNEnvironment:
    # ... (__init__, reset are unchanged) ...
    def __init__(self):
        self.width, self.height, self.vehicles, self.uavs, self.ccc = AREA_WIDTH, AREA_HEIGHT, [], [], CloudComputingCenter()
        self.time_step, self.total_tasks_in_step, self.completed_tasks_in_step = 0, 0, 0

    def reset(self, num_uavs=0, num_vehicles=NUM_VEHICLES):
        self.time_step = 0
        self.uavs = [UAV(i) for i in range(num_uavs)]
        self.vehicles = [Vehicle(i) for i in range(num_vehicles)]
        for v in self.vehicles: v.generate_tasks()
        return self.get_maddpg_states()

    def step(self, actions):
        # ... (step function logic is correct and unchanged) ...
        for uav in self.uavs:
            uav.energy_consumed_this_step = 0.0
            uav.profit_this_step = 0.0
        self._update_active_tasks()
        for i, uav in enumerate(self.uavs): uav.move(np.array([actions[i][0], actions[i][1], 0]) * UAV_MAX_SPEED)
        for vehicle in self.vehicles: vehicle.move()
        self._assign_new_tasks()
        self._consume_hover_energy()
        rewards = self._calculate_rewards()
        next_states = self.get_maddpg_states()
        self.time_step += 1
        done = self.time_step >= INNER_STEPS
        return next_states, rewards, done

    # --- All asynchronous logic (_update_active_tasks, _assign_new_tasks, etc.) is correct and unchanged ---
    def _update_active_tasks(self):
        all_tasks = [task for vehicle in self.vehicles for task in vehicle.tasks]
        for task in all_tasks:
            if task.status == 'UPLOADING' and self.time_step >= task.upload_complete_time:
                task.status = 'RELAYING' if task.entry_uav.id != task.target_uav.id else 'COMPUTING'
            if task.status == 'RELAYING' and self.time_step >= task.relay_complete_time:
                task.status = 'COMPUTING'
                task.entry_uav.status = 'IDLE'
            if task.status == 'COMPUTING' and self.time_step >= task.compute_complete_time:
                task.status = 'COMPLETED'
                task.is_completed = True
                task.completed_latency = self.time_step - task.time_initiated
                if task.target_uav:
                    task.target_uav.status = 'IDLE'
                    task.target_uav.F_remain += task.cpu_cycles_req
                if task.entry_uav:
                    task.entry_uav.profit_this_step += task.profit_generated
                    task.entry_uav.profit_generated += task.profit_generated
                    task.entry_uav.tasks_processed_count += 1
                    if task.target_uav and task.entry_uav.id == task.target_uav.id:
                        task.entry_uav.status = 'IDLE'
                    elif not task.target_uav and self.time_step >= task.upload_complete_time:
                        task.entry_uav.status = 'IDLE'

    def _calculate_task_energy_cost(self, task, offload_type, entry_uav, target_uav=None):
        cost_entry_uav, cost_target_uav = 0, 0
        task_size_mbit = task.data_size_bits / 1e6
        if offload_type == 'local_uav':
            cost_entry_uav += ENERGY_COMM_JOULE_PER_MBIT * task_size_mbit + ENERGY_COMPUTATION_JOULE_PER_GCYCLE * (
                        task.cpu_cycles_req / 1e9)
        elif offload_type == 'relay_uav':
            cost_entry_uav += ENERGY_COMM_JOULE_PER_MBIT * task_size_mbit * 2
            cost_target_uav += ENERGY_COMM_JOULE_PER_MBIT * task_size_mbit + ENERGY_COMPUTATION_JOULE_PER_GCYCLE * (
                        task.cpu_cycles_req / 1e9)
        elif offload_type == 'cloud':
            cost_entry_uav += ENERGY_COMM_JOULE_PER_MBIT * task_size_mbit * 2
        return cost_entry_uav, cost_target_uav

    def _assign_new_tasks(self):
        pending_tasks = [task for v in self.vehicles for task in v.tasks if task.status == 'PENDING']
        if not self.uavs or not pending_tasks: return
        service_to_uav_map = {}
        for uav in self.uavs:
            for st in uav.cache:
                if st not in service_to_uav_map: service_to_uav_map[st] = []
                service_to_uav_map[st].append(uav)
        for task in pending_tasks:
            vehicle = self.vehicles[task.owner_id]
            best_latency, best_option = float('inf'), None
            idle_uavs_in_range = [u for u in self.uavs if
                                  u.status == 'IDLE' and self._get_distance_obj(u, vehicle) <= UAV_COMMUNICATION_RANGE]
            if not idle_uavs_in_range: continue
            for uav in idle_uavs_in_range:
                cost_uav, _ = self._calculate_task_energy_cost(task, 'local_uav', uav)
                if uav.has_service(
                        task.service_type) and uav.F_remain > task.cpu_cycles_req and uav.current_energy > cost_uav:
                    datarate = self.calculate_datarate_user_to_uav(vehicle, uav)
                    latency = (task.data_size_bits / (datarate * 1e6 + 1e-9)) + (task.cpu_cycles_req / uav.F_remain)
                    if latency < best_latency and (self.time_step + latency) <= task.latency_constraint:
                        best_latency, best_option = latency, ('local_uav', uav, uav, latency,
                                                              self._calculate_earn_task(task, latency, uav))
            target_uavs = service_to_uav_map.get(task.service_type, [])
            if target_uavs:
                for entry_uav in idle_uavs_in_range:
                    for target_uav in target_uavs:
                        if entry_uav.id == target_uav.id or target_uav.status != 'IDLE': continue
                        cost_entry, cost_target = self._calculate_task_energy_cost(task, 'relay_uav', entry_uav,
                                                                                   target_uav)
                        if target_uav.F_remain > task.cpu_cycles_req and entry_uav.current_energy > cost_entry and target_uav.current_energy > cost_target:
                            rate1, rate2 = self.calculate_datarate_user_to_uav(vehicle,
                                                                               entry_uav), self.calculate_datarate_uav_to_uav(
                                entry_uav, target_uav)
                            latency = (task.data_size_bits / (rate1 * 1e6 + 1e-9)) + (
                                        task.data_size_bits / (rate2 * 1e6 + 1e-9)) + (
                                                  task.cpu_cycles_req / target_uav.F_remain)
                            if latency < best_latency and (self.time_step + latency) <= task.latency_constraint:
                                best_latency, best_option = latency, ('relay_uav', entry_uav, target_uav, latency,
                                                                      self._calculate_earn_task(task, latency,
                                                                                                entry_uav))
            for uav in idle_uavs_in_range:
                cost_uav, _ = self._calculate_task_energy_cost(task, 'cloud', uav)
                if uav.current_energy > cost_uav:
                    rate_user, rate_ccc = self.calculate_datarate_user_to_uav(vehicle,
                                                                              uav), self.calculate_datarate_uav_to_ccc(
                        uav)
                    latency = (task.data_size_bits / (rate_user * 1e6 + 1e-9)) + (
                                task.data_size_bits / (rate_ccc * 1e6 + 1e-9))
                    if latency < best_latency and (self.time_step + latency) <= task.latency_constraint:
                        best_latency, best_option = latency, ('cloud', uav, None, latency,
                                                              self._calculate_earn_task(task, latency))
            if best_option:
                offload_type, entry_uav, target_uav, latency, profit = best_option
                task.status = 'UPLOADING'
                task.time_initiated = self.time_step
                task.profit_generated = profit
                task.entry_uav, task.target_uav = entry_uav, target_uav
                rate1 = self.calculate_datarate_user_to_uav(self.vehicles[task.owner_id], entry_uav)
                upload_duration = math.ceil(task.data_size_bits / (rate1 * 1e6 + 1e-9))
                task.upload_complete_time = self.time_step + upload_duration
                if offload_type == 'local_uav':
                    compute_duration = math.ceil(task.cpu_cycles_req / entry_uav.F_remain)
                    task.compute_complete_time = task.upload_complete_time + compute_duration
                    entry_uav.record_cache_hit(task.service_type)
                elif offload_type == 'relay_uav':
                    rate2 = self.calculate_datarate_uav_to_uav(entry_uav, target_uav)
                    relay_duration = math.ceil(task.data_size_bits / (rate2 * 1e6 + 1e-9))
                    compute_duration = math.ceil(task.cpu_cycles_req / target_uav.F_remain)
                    task.relay_complete_time = task.upload_complete_time + relay_duration
                    task.compute_complete_time = task.relay_complete_time + compute_duration
                    target_uav.record_cache_hit(task.service_type)
                elif offload_type == 'cloud':
                    task.compute_complete_time = task.upload_complete_time
                    if np.random.rand() < CACHE_UPDATE_PROBABILITY: entry_uav.update_cache(task.service_type)
                cost_entry, cost_target = self._calculate_task_energy_cost(task, offload_type, entry_uav, target_uav)
                entry_uav.consume_energy(cost_entry)
                entry_uav.status = 'BUSY'
                if target_uav:
                    target_uav.consume_energy(cost_target)
                    target_uav.F_remain -= task.cpu_cycles_req
                    if entry_uav.id != target_uav.id: target_uav.status = 'BUSY'

    def _consume_hover_energy(self):
        for uav in self.uavs:
            if uav.current_energy > 0: uav.consume_energy(ENERGY_HOVER_WATT * 1.0)

    def _calculate_rewards(self):
        if not self.uavs: return []
        rewards = [(uav.profit_this_step - uav.energy_consumed_this_step * ENERGY_REWARD_PENALTY) for uav in self.uavs]
        global_reward = np.mean(rewards) if rewards else 0
        return [global_reward * REWARD_SCALING_FACTOR] * len(self.uavs)

    def get_maddpg_states(self):
        if not self.uavs: return []
        all_states = []
        for uav in self.uavs:
            status_numeric = 1.0 if uav.status == 'BUSY' else 0.0
            state = [
                uav.position[0] / self.width, uav.position[1] / self.height,
                len([v for v in self.vehicles if self._get_distance_obj(uav, v) <= UAV_COMMUNICATION_RANGE]),
                uav.tasks_processed_count, uav.profit_generated / 1000.0,
                uav.current_energy / uav.max_energy if uav.max_energy > 0 else 0,
                status_numeric
            ]
            all_states.append(np.array(state))
        return all_states

    @jit(nopython=False, forceobj=True)
    def _calculate_earn_task(self, task, latency, uav=None):
        latency = max(latency, 1e-9)
        term_latency = DELTA_LATENCY * (task.latency_constraint / latency)
        if uav:
            required_computation = 0.5 * latency
            term_size = DELTA_SIZE * (task.data_size_bits / 1e6)
            normalized_f_remain = uav.F_remain / uav.F_total
            term_comp = DELTA_COMPUTATION * (required_computation / (normalized_f_remain + 1e-9))
            term_comp = np.clip(term_comp, 0, 1000)
            return term_latency + term_size + term_comp
        else:
            return term_latency

    def get_ddqn_state(self):
        num_uavs = len(self.uavs)
        total_profit = sum(uav.profit_generated for uav in self.uavs)
        total_cost = sum(BETA_MAINTENANCE + BETA_COMPUTATION * u.F_total for u in self.uavs)
        net_profit = total_profit - total_cost
        tasks_completed = sum(uav.tasks_processed_count for uav in self.uavs)
        avg_latency = np.mean([t.completed_latency for v in self.vehicles for t in v.tasks if
                               t.is_completed]) if tasks_completed > 0 else 0
        covered_vehicles = set()
        if num_uavs > 0:
            for uav in self.uavs:
                for v in self.vehicles:
                    if self._get_distance_obj(uav, v) <= UAV_COMMUNICATION_RANGE: covered_vehicles.add(v.id)
        return np.array([tasks_completed, num_uavs, total_cost, net_profit * REWARD_SCALING_FACTOR, avg_latency,
                         len(covered_vehicles)])

    # --- Communication Modeling ---
    def calculate_datarate_uav_to_uav(self, uav1, uav2):
        return self.calculate_datarate(BANDWIDTH_UAV_UAV, POWER_UAV_UAV, self.get_average_path_loss(uav1, uav2))

    def calculate_datarate_user_to_uav(self, user, uav):
        return self.calculate_datarate(BANDWIDTH_UAV_USER, POWER_UAV_USER, self.get_average_path_loss(uav, user))

    def calculate_datarate_uav_to_ccc(self, uav):
        return self.calculate_datarate(BANDWIDTH_UAV_CCC, POWER_CCC, 60)

    def _get_distance_obj(self, e1, e2):
        return np.linalg.norm(e1.position - e2.position)

    def calculate_datarate(self, bw_hz, p_watt, pl_db):
        """Wrapper function that calls the external, Numba-optimized version."""
        noise_const = 10 ** ((NOISE_POWER_SPECTRAL_DENSITY - 30) / 10)
        # --- Phase 4 CORRECTION: Call the external function ---
        return _calculate_datarate_numba(bw_hz, p_watt, pl_db, noise_const) / 1e6

    def calculate_elevation_angle(self, e1, e2):
        dist_2d = np.linalg.norm(e1.position[:2] - e2.position[:2])
        delta_h = abs(e1.position[2] - e2.position[2])
        return np.pi / 2 if dist_2d == 0 else np.arctan(delta_h / dist_2d)

    def calculate_los_probability(self, uav, entity):
        angle_deg = np.rad2deg(self.calculate_elevation_angle(uav, entity))
        return 1 / (1 + LOS_X0 * np.exp(-LOS_Y0 * (angle_deg - LOS_X0)))

    def calculate_path_loss(self, d, is_los):
        fspl = 20 * np.log10(d) + 20 * np.log10(CARRIER_FREQUENCY) - 147.55
        return fspl + (ETA_LOS if is_los else ETA_NLOS)

    def get_average_path_loss(self, e1, e2):
        d = self._get_distance_obj(e1, e2)
        if d == 0: return 0
        los_prob = self.calculate_los_probability(e1, e2)
        pl_los, pl_nlos = self.calculate_path_loss(d, True), self.calculate_path_loss(d, False)
        avg_pl = los_prob * (10 ** (pl_los / 10)) + (1 - los_prob) * (10 ** (pl_nlos / 10))
        return 10 * np.log10(avg_pl)
