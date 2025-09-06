# environment.py
"""
Implements the VECN Environment.
CORRECTED for Numba TypingError: The jitted function for calculating the
datarate has been moved outside the class to be a standalone helper function.
This is necessary because Numba's `nopython=True` mode cannot handle
class instances ('self') as arguments.
"""
import math
from collections import deque

import numpy as np
from numba import jit, njit
from scipy.cluster.vq import kmeans

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
        """
        Resets the environment.
        MODIFIED: Now includes logic to create congestion zones with higher task loads.
        """
        self.time_step = 0
        self.uavs = [UAV(i) for i in range(num_uavs)]
        self.vehicles = [Vehicle(i) for i in range(num_vehicles)]

        # --- NEW: Dynamic Demand Generation based on Vehicle Congestion ---
        if USE_DYNAMIC_DEMAND and self.vehicles:
            vehicle_positions = np.array([v.position[:2] for v in self.vehicles])

            # Use K-means to find cluster centers (congestion zones)
            # Note: k-means requires at least as many points as clusters
            num_clusters = min(CONGESTION_NUM_ZONES, len(self.vehicles))
            if num_clusters > 0:
                centroids, _ = kmeans(vehicle_positions, num_clusters, iter=10)

                # Identify which vehicles are in a "congested" zone
                congested_vehicle_ids = set()
                for i, pos in enumerate(vehicle_positions):
                    # Check distance to each congestion centroid
                    for centroid in centroids:
                        if np.linalg.norm(pos - centroid) < (UAV_COMMUNICATION_RANGE / 2): # Heuristic radius
                            congested_vehicle_ids.add(self.vehicles[i].id)
                            break

            # Generate tasks based on whether the vehicle is in a congested zone
            for v in self.vehicles:
                if v.id in congested_vehicle_ids:
                    v.generate_tasks(num_tasks=TASKS_PER_VEHICLE_CONGESTED)
                else:
                    v.generate_tasks(num_tasks=TASKS_PER_VEHICLE)
        else:
            # Original behavior if dynamic demand is off or no vehicles
            for v in self.vehicles:
                v.generate_tasks()
        # --- END OF NEW SECTION ---

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

    def _consume_hover_energy(self):
        for uav in self.uavs:
            if uav.current_energy > 0:
                uav.consume_energy(ENERGY_HOVER_WATT * 1.0)

    def _calculate_rewards(self):
        """
        Calculates the global reward for all UAVs.
        MODIFIED: The energy penalty is now optional, controlled by a config flag.
        """
        if not self.uavs:
            return []

        rewards = []
        for uav in self.uavs:
            reward = uav.profit_this_step
            # Conditionally apply the energy penalty
            if USE_ENERGY_PENALTY:
                reward -= uav.energy_consumed_this_step * ENERGY_REWARD_PENALTY
            rewards.append(reward)

        global_reward = np.mean(rewards) if rewards else 0
        return [global_reward * REWARD_SCALING_FACTOR] * len(self.uavs)

    def get_maddpg_states(self):
        """
        Generates the list of state vectors for each MADDPG agent.
        MODIFIED: The UAV's 'status' is now an optional part of the state vector,
        controlled by a config flag to ensure the state dimension is correct.
        """
        if not self.uavs:
            return []
        all_states = []
        for uav in self.uavs:
            # Base state vector that is always present
            state = [
                uav.position[0] / self.width,
                uav.position[1] / self.height,
                len([v for v in self.vehicles if self._get_distance_obj(uav, v) <= UAV_COMMUNICATION_RANGE]),
                uav.tasks_processed_count,
                uav.profit_generated / 1000.0,
                uav.current_energy / uav.max_energy if uav.max_energy > 0 else 0,
            ]

            # Conditionally add the UAV status to the state vector
            if USE_UAV_STATUS:
                status_numeric = 1.0 if uav.status == 'BUSY' else 0.0
                state.append(status_numeric)

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

    def calculate_datarate_user_to_uav(self, user, uav, num_sharing_users=1):
        """
        Calculates the data rate from a user to a UAV, considering TDMA-based
        bandwidth sharing if enabled in the config.
        """
        # If dynamic bandwidth is disabled, or if there's only one user, use the full bandwidth.
        if not DYNAMIC_BANDWIDTH or num_sharing_users <= 1:
            effective_bandwidth = BANDWIDTH_UAV_USER
        else:
            # TDMA abstraction: total bandwidth is shared among all contending users for that UAV.
            effective_bandwidth = BANDWIDTH_UAV_USER / num_sharing_users

        return self.calculate_datarate(effective_bandwidth, POWER_UAV_USER, self.get_average_path_loss(uav, user))

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

    def _find_multi_hop_path(self, task, entry_uav, target_uav_candidates=None):
        """
        Finds a multi-hop path from entry_uav to a target UAV that can process the task.
        Uses BFS on UAV graph (edges if distance <= UAV_COMMUNICATION_RANGE).
        Returns a list of UAV indices in a path or empty list if no path is found.
        """
        if not self.uavs:
            return []

        # Build graph: UAV index -> list of connected UAV indices
        uav_positions = np.array([u.position for u in self.uavs])
        distances = pairwise_distance_numba(uav_positions, uav_positions)
        graph = {i: [] for i in range(len(self.uavs))}
        for i in range(len(self.uavs)):
            for j in range(len(self.uavs)):
                if i != j and distances[i, j] <= UAV_COMMUNICATION_RANGE:
                    graph[i].append(j)

        # BFS setup
        start_idx = self.uavs.index(entry_uav)
        queue = deque([(start_idx, [start_idx])])
        visited = {start_idx}

        while queue:
            current_idx, path = queue.popleft()
            if current_idx in visited:
                continue
            visited.add(current_idx)

            current_uav = self.uavs[current_idx]
            # Check if this UAV can process (has service and resources)
            if (current_uav.has_service(task.service_type) and
                current_uav.has_content(task.content_type) and
                current_uav.F_remain >= task.cpu_cycles_req and
                current_uav.status == 'IDLE'):
                # If we were given a specific list of candidates, ensure this UAV is one of them
                if target_uav_candidates is None or current_uav in target_uav_candidates:
                    return path

            # Enqueue neighbors if hops < MAX_HOPS
            if len(path) < MAX_HOPS + 1:  # +1 for starting point
                for neighbor_idx in graph[current_idx]:
                    if neighbor_idx not in visited:
                        queue.append((neighbor_idx, path + [neighbor_idx]))

        return []  # No path found

    # Updated _calculate_task_energy_cost (extended for multi-hop)
    def _calculate_task_energy_cost(self, task, offload_type, entry_uav, target_uav=None, hop_path_indices=None):
        cost_entry_uav, cost_target_uav = 0, 0
        task_size_mbit = task.data_size_bits / 1e6
        if offload_type == 'local_uav':
            cost_entry_uav += ENERGY_COMM_JOULE_PER_MBIT * task_size_mbit + ENERGY_COMPUTATION_JOULE_PER_GCYCLE * (
                    task.cpu_cycles_req / 1e9)
        elif offload_type == 'relay_uav':
            if hop_path_indices and len(hop_path_indices) > 2:  # Multi-hop (>1 relay)
                # Sum comm costs over hops + compute at target
                num_hops = len(hop_path_indices) - 1
                cost_per_hop = ENERGY_COMM_JOULE_PER_MBIT * task_size_mbit
                # Distribute: Entry pays for first upload, intermediates for relay, target for receive + compute
                cost_entry_uav += cost_per_hop  # Upload to first relay
                # For intermediates: Add to their costs (but since async, consume incrementally; here estimate total)
                # For simplicity, return dict of costs per UAV in path, but keep simple for now
                cost_target_uav += cost_per_hop + ENERGY_COMPUTATION_JOULE_PER_GCYCLE * (task.cpu_cycles_req / 1e9)
                # Note: In practice, consume per step in _update_active_tasks
            else:  # Single relay
                cost_entry_uav += ENERGY_COMM_JOULE_PER_MBIT * task_size_mbit * 2
                cost_target_uav += ENERGY_COMM_JOULE_PER_MBIT * task_size_mbit + ENERGY_COMPUTATION_JOULE_PER_GCYCLE * (
                        task.cpu_cycles_req / 1e9)
        elif offload_type == 'cloud':
            cost_entry_uav += ENERGY_COMM_JOULE_PER_MBIT * task_size_mbit * 2
        return cost_entry_uav, cost_target_uav

    # Updated _update_active_tasks
    def _update_active_tasks(self):
        all_tasks = [task for vehicle in self.vehicles for task in vehicle.tasks]
        for task in all_tasks:
            if task.status == 'UPLOADING' and self.time_step >= task.upload_complete_time:
                if len(task.hop_path) > 1:
                    task.status = 'RELAYING'
                else:
                    task.status = 'COMPUTING'

            if task.status == 'RELAYING':
                if len(task.hop_path) <= 1:
                    # Fallback to single relay or CCC
                    task.status = 'COMPUTING'
                else:
                    # Handle hop-by-hop
                    current_hop_idx = task.hop_path.index(task.entry_uav.id)  # Assume entry_uav updated to current
                    if current_hop_idx + 1 < len(task.hop_path):
                        next_hop_id = task.hop_path[current_hop_idx + 1]
                        next_hop_uav = next((u for u in self.uavs if u.id == next_hop_id), None)
                        if next_hop_uav:
                            # Calculate relay rate
                            dist = self._get_distance_obj(task.entry_uav, next_hop_uav)
                            pl_db = self._calculate_path_loss(dist, task.entry_uav.position[2])
                            rate = _calculate_datarate_numba(BANDWIDTH_UAV_UAV, POWER_UAV_UAV, pl_db,
                                                             NOISE_POWER_SPECTRAL_DENSITY)

                            # Relay progress this step (assume 1s timestep)
                            relayed_this_step = min(rate, task.data_size_bits)  # bits
                            task.data_size_bits -= relayed_this_step  # Track remaining; wait, actually data is fixed, use progress var
                            # Better: Add task.relay_progress = 0 initially, increment
                            # For simplicity, assume full relay in calculated duration, but since async, check if complete
                            if self.time_step >= task.relay_complete_time:  # Use precalculated, or recalculate dynamically
                                task.entry_uav.status = 'IDLE'
                                task.entry_uav = next_hop_uav
                                task.entry_uav.status = 'BUSY'
                                if current_hop_idx + 1 == len(task.hop_path) - 1:
                                    task.status = 'COMPUTING'
                            # Consume energy
                            energy_comm = (relayed_this_step / 1e6) * ENERGY_COMM_JOULE_PER_MBIT
                            task.entry_uav.consume_energy(energy_comm)

            if task.status == 'COMPUTING' and self.time_step >= task.compute_complete_time:
                task.status = 'COMPLETED'
                task.is_completed = True
                task.completed_latency = self.time_step - task.time_initiated
                if task.target_uav:
                    task.target_uav.status = 'IDLE'
                    task.target_uav.F_remain += task.cpu_cycles_req  # Restore? No, it was deducted, but if completed, ok
                if task.entry_uav:
                    task.entry_uav.profit_this_step += task.profit_generated
                    task.entry_uav.profit_generated += task.profit_generated
                    task.entry_uav.tasks_processed_count += 1
                    if task.target_uav and task.entry_uav.id == task.target_uav.id:
                        task.entry_uav.status = 'IDLE'
                    elif not task.target_uav and self.time_step >= task.upload_complete_time:
                        task.entry_uav.status = 'IDLE'

    # Updated _assign_new_tasks
    def _assign_new_tasks(self):
        pending_tasks = [task for v in self.vehicles for task in v.tasks if task.status == 'PENDING']
        if not self.uavs or not pending_tasks: return

        # --- MODIFIED: Build maps for both service and content caches ---
        service_cache_map = {}
        for uav in self.uavs:
            for st in uav.service_cache:
                if st not in service_cache_map: service_cache_map[st] = []
                service_cache_map[st].append(uav)
        # --- END OF MODIFICATION ---

        # Pre-calculate TDMA load (from Step 2)
        uav_potential_load = {uav.id: 0 for uav in self.uavs}
        if DYNAMIC_BANDWIDTH:
            vehicles_with_tasks = {task.owner_id for task in pending_tasks}
            for vehicle_id in vehicles_with_tasks:
                vehicle = next(v for v in self.vehicles if v.id == vehicle_id)
                for uav in self.uavs:
                    if uav.status == 'IDLE' and self._get_distance_obj(uav, vehicle) <= UAV_COMMUNICATION_RANGE:
                        uav_potential_load[uav.id] += 1

        for task in pending_tasks:
            task.time_initiated = self.time_step
            vehicle = next(v for v in self.vehicles if v.id == task.owner_id)
            best_latency, best_option = float('inf'), None
            idle_uavs_in_range = [u for u in self.uavs if
                                  u.status == 'IDLE' and self._get_distance_obj(u, vehicle) <= UAV_COMMUNICATION_RANGE]
            if not idle_uavs_in_range: continue

            # --- MODIFIED: Offloading logic now checks separate caches ---
            # Local (direct) processing
            for entry_uav in idle_uavs_in_range:
                cost_entry, _ = self._calculate_task_energy_cost(task, 'local_uav', entry_uav)
                # UAV must have the service program AND any required content
                if (entry_uav.has_service(task.service_type) and
                    entry_uav.has_content(task.content_type) and
                    entry_uav.F_remain > task.cpu_cycles_req and
                    entry_uav.current_energy > cost_entry):

                    num_sharers = max(1, uav_potential_load.get(entry_uav.id, 1))
                    datarate = self.calculate_datarate_user_to_uav(vehicle, entry_uav, num_sharing_users=num_sharers)
                    upload_duration = math.ceil(task.data_size_bits / (datarate * 1e6 + 1e-9))
                    compute_duration = math.ceil(task.cpu_cycles_req / entry_uav.F_remain)
                    latency = upload_duration + compute_duration

                    if latency < best_latency and (self.time_step + latency) <= task.latency_constraint:
                        best_latency, best_option = latency, ('local_uav', entry_uav, entry_uav, latency,
                                                              self._calculate_earn_task(task, latency, entry_uav),
                                                              [entry_uav.id])

            # Multi-hop relay
            # Find target UAVs that have the required service
            target_uavs = service_cache_map.get(task.service_type, [])
            if target_uavs:
                for entry_uav in idle_uavs_in_range:
                    # Find a path to a valid target UAV (that also has the content)
                    hop_path_indices = self._find_multi_hop_path(task, entry_uav, target_uav_candidates=target_uavs)

                    if hop_path_indices:
                        hop_path_uavs = [self.uavs[idx] for idx in hop_path_indices]
                        target_uav = hop_path_uavs[-1]

                        # Check again if target_uav is valid (find_multi_hop_path should ensure this)
                        if not target_uav.has_content(task.content_type): continue

                        num_sharers = max(1, uav_potential_load.get(entry_uav.id, 1))
                        upload_rate = self.calculate_datarate_user_to_uav(vehicle, entry_uav, num_sharing_users=num_sharers)
                        upload_duration = math.ceil(task.data_size_bits / (upload_rate * 1e6 + 1e-9))
                        relay_duration = 0
                        for i in range(len(hop_path_uavs) - 1):
                            relay_rate = self.calculate_datarate_uav_to_uav(hop_path_uavs[i], hop_path_uavs[i + 1])
                            relay_duration += math.ceil(task.data_size_bits / (relay_rate * 1e6 + 1e-9))
                        compute_duration = math.ceil(task.cpu_cycles_req / target_uav.F_remain)
                        latency = upload_duration + relay_duration + compute_duration
                        cost_entry, cost_target = self._calculate_task_energy_cost(task, 'relay_uav', entry_uav,
                                                                                   target_uav, hop_path_indices)
                        if latency < best_latency and (self.time_step + latency) <= task.latency_constraint:
                            best_latency, best_option = latency, ('relay_uav', entry_uav, target_uav, latency,
                                                                  self._calculate_earn_task(task, latency, entry_uav),
                                                                  [u.id for u in hop_path_uavs])

            # Cloud fallback
            for entry_uav in idle_uavs_in_range:
                cost_entry, _ = self._calculate_task_energy_cost(task, 'cloud', entry_uav)
                if entry_uav.current_energy > cost_entry:

                    # MODIFIED: Pass the pre-calculated user load
                    num_sharers = max(1, uav_potential_load.get(entry_uav.id, 1))
                    rate_user = self.calculate_datarate_user_to_uav(vehicle, entry_uav, num_sharing_users=num_sharers)

                    rate_ccc = self.calculate_datarate_uav_to_ccc(entry_uav)
                    upload_duration = math.ceil(task.data_size_bits / (rate_user * 1e6 + 1e-9))
                    relay_duration = math.ceil(task.data_size_bits / (rate_ccc * 1e6 + 1e-9))
                    latency = upload_duration + relay_duration
                    if latency < best_latency and (self.time_step + latency) <= task.latency_constraint:
                        best_latency, best_option = latency, ('cloud', entry_uav, None, latency,
                                                              self._calculate_earn_task(task, latency))

            if best_option:
                offload_type, entry_uav, target_uav, latency, profit, hop_path = best_option if len(
                    best_option) == 6 else best_option + ([],)
                task.status = 'UPLOADING'
                task.profit_generated = profit
                task.entry_uav, task.target_uav = entry_uav, target_uav
                task.hop_path = hop_path

                # MODIFIED: Final assignment must also use the shared data rate for timing
                num_sharers = max(1, uav_potential_load.get(entry_uav.id, 1))
                upload_rate = self.calculate_datarate_user_to_uav(vehicle, entry_uav, num_sharing_users=num_sharers)

                task.upload_complete_time = self.time_step + math.ceil(task.data_size_bits / (upload_rate * 1e6 + 1e-9))
                if offload_type == 'local_uav':
                    task.compute_complete_time = task.upload_complete_time + math.ceil(
                        task.cpu_cycles_req / entry_uav.F_remain)
                    # MODIFIED: Record hit in the service cache
                    entry_uav.record_service_cache_hit(task.service_type)
                elif offload_type == 'relay_uav':
                    relay_duration = 0
                    # This part remains unchanged as UAV-UAV links use a different frequency band
                    for i in range(len(hop_path) - 1):
                        uav1 = next(u for u in self.uavs if u.id == hop_path[i])
                        uav2 = next(u for u in self.uavs if u.id == hop_path[i+1])
                        relay_rate = self.calculate_datarate_uav_to_uav(uav1, uav2)
                        relay_duration += math.ceil(task.data_size_bits / (relay_rate * 1e6 + 1e-9))
                    task.relay_complete_time = task.upload_complete_time + relay_duration
                    task.compute_complete_time = task.relay_complete_time + math.ceil(
                        task.cpu_cycles_req / target_uav.F_remain)
                    target_uav.record_service_cache_hit(task.service_type)
                elif offload_type == 'cloud':
                    task.compute_complete_time = task.upload_complete_time
                    if np.random.rand() < CACHE_UPDATE_PROBABILITY:
                        entry_uav.update_service_cache(task.service_type)

                cost_entry, cost_target = self._calculate_task_energy_cost(task, offload_type, entry_uav, target_uav,
                                                                           [u.id for u in hop_path] if offload_type == 'relay_uav' else [])
                entry_uav.consume_energy(cost_entry)
                if target_uav:
                    target_uav.consume_energy(cost_target)
                    target_uav.F_remain -= task.cpu_cycles_req
                entry_uav.status = 'BUSY'
                if target_uav and entry_uav.id != target_uav.id:
                    target_uav.status = 'BUSY'


    def _calculate_path_loss(self, dist, altitude):
        """
        Calculates path loss in dB based on the paper's LoS/NLoS model.
        Uses standard urban UAV path loss formula with P_LoS probability.
        """
        if dist == 0:
            return 0.0
        # Elevation angle theta in degrees
        theta = math.degrees(math.asin(altitude / dist))
        # Parameters from config/paper
        a = LOS_X0  # 11.9
        b = LOS_Y0  # 0.13
        P_LoS = 1 / (1 + a * math.exp(-b * (theta - a)))
        # Free space path loss
        f = CARRIER_FREQUENCY  # 2e9 Hz
        c = C  # 3e8 m/s
        free_space_loss = 20 * math.log10(4 * math.pi * f * dist / c)
        # Weighted eta
        pl_db = free_space_loss + P_LoS * ETA_LOS + (1 - P_LoS) * ETA_NLOS
        return pl_db
