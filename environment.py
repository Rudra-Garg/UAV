# environment.py
import logging
import math
from collections import deque, Counter

import numpy as np
from numba import jit, njit
from scipy.cluster.vq import kmeans

from config import *
from demand_generator import DemandGenerator
from entities import Vehicle, UAV, CloudComputingCenter, Task

# Get a logger for this module
logger = logging.getLogger(__name__)


# --- Numba-Optimized Helper Functions ---
@njit
def pairwise_distance_numba(pos_array1, pos_array2):
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


@jit(nopython=True)
def _calculate_datarate_numba(bw_hz, p_watt, pl_db, noise_const):
    noise_watt = noise_const * bw_hz
    rx_watt_dbm = 10 * np.log10(p_watt * 1000) - pl_db
    rx_watt = 10 ** ((rx_watt_dbm - 30) / 10)
    snr = rx_watt / noise_watt
    return (bw_hz * np.log2(1 + snr)) if snr > 0 else 0.0


class VECNEnvironment:
    def __init__(self):
        self.width, self.height = AREA_WIDTH, AREA_HEIGHT
        self.vehicles, self.uavs, self.ccc = [], [], CloudComputingCenter()
        self.time_step = 0
        # History now stores tuples: (service_id, zone_id)
        self.request_history = deque(maxlen=PREDICTION_SEQUENCE_LENGTH * 2)
        self.demand_generator = DemandGenerator()
        self.local_offload_count = 0
        self.relay_offload_count = 0
        self.cloud_offload_count = 0

    def _get_zone_id(self, position):
        """Determines the zone ID (0-3) based on x,y position."""
        x, y = position[0], position[1]
        mid_x, mid_y = self.width / 2, self.height / 2

        if x < mid_x and y < mid_y:
            return 0  # Bottom-Left
        elif x >= mid_x and y < mid_y:
            return 1  # Bottom-Right
        elif x < mid_x and y >= mid_y:
            return 2  # Top-Left
        else:
            return 3  # Top-Right

    def _initialize_vehicle_positions_with_hotspots(self, num_vehicles, num_hotspots, hotspot_radius, hotspot_ratio):
        self.vehicles = []
        if num_hotspots == 0:
            self.vehicles = [Vehicle(i) for i in range(num_vehicles)]
            return

        hotspots = [np.random.rand(2) * np.array([self.width, self.height]) for _ in range(num_hotspots)]
        num_hotspot_vehicles = int(num_vehicles * hotspot_ratio)
        num_random_vehicles = num_vehicles - num_hotspot_vehicles
        vehicle_id_counter = 0

        for i in range(num_hotspot_vehicles):
            vehicle = Vehicle(vehicle_id_counter)
            chosen_hotspot = hotspots[i % num_hotspots]
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, hotspot_radius)
            offset = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            pos_2d = chosen_hotspot + offset
            vehicle.position = np.append(pos_2d, 0)
            vehicle.position[0] = np.clip(vehicle.position[0], 0, self.width)
            vehicle.position[1] = np.clip(vehicle.position[1], 0, self.height)
            self.vehicles.append(vehicle)
            vehicle_id_counter += 1

        for _ in range(num_random_vehicles):
            vehicle = Vehicle(vehicle_id_counter)
            pos_2d = np.random.rand(2) * np.array([self.width, self.height])
            vehicle.position = np.append(pos_2d, 0)
            self.vehicles.append(vehicle)
            vehicle_id_counter += 1

    def reset(self, num_uavs=0, num_vehicles=NUM_VEHICLES):
        logger.info("Resetting environment with %d UAVs and %d vehicles.", num_uavs, num_vehicles)
        self.time_step = 0
        self.local_offload_count = 0
        self.relay_offload_count = 0
        self.cloud_offload_count = 0
        self.request_history.clear()
        self.demand_generator.reset()

        chosen_scenario_name = np.random.choice(
            TRAFFIC_SCENARIOS['SCENARIO_NAMES'],
            p=TRAFFIC_SCENARIOS['SCENARIO_WEIGHTS']
        )
        scenario_params = TRAFFIC_SCENARIOS['SCENARIOS'][chosen_scenario_name]

        self._initialize_vehicle_positions_with_hotspots(
            num_vehicles=num_vehicles,
            num_hotspots=scenario_params['num_hotspots'],
            hotspot_radius=scenario_params['hotspot_radius'],
            hotspot_ratio=scenario_params['hotspot_ratio']
        )

        # --- GENERATE TASKS USING NEW LOGIC ---
        for v in self.vehicles:
            v.tasks = []
            zone_id = self._get_zone_id(v.position)

            for i in range(TASKS_PER_VEHICLE):
                # Pass time, vehicle_id, and ZONE to generator
                service, content = self.demand_generator.generate_next_request(
                    self.time_step, v.id, zone_id
                )

                new_task = Task(f"{v.id}-{i}", v.id, service, content)
                v.tasks.append(new_task)

                # Store Tuple (Service, Zone) for LSTM
                self.request_history.append((service, zone_id))

        # UAV Deployment (Clustering logic)
        centroids = []
        if USE_DYNAMIC_DEMAND and self.vehicles:
            vehicle_positions = np.array([v.position[:2] for v in self.vehicles])
            num_clusters = min(scenario_params['num_hotspots'], len(self.vehicles))
            if num_clusters > 0:
                centroids, _ = kmeans(vehicle_positions, num_clusters, iter=10)

        self.uavs = [UAV(i) for i in range(num_uavs)]
        if self.uavs and len(centroids) > 0:
            for i, uav in enumerate(self.uavs):
                centroid = centroids[i % len(centroids)]
                offset = (np.random.rand(2) * 2 - 1) * 100
                uav.position[0] = centroid[0] + offset[0]
                uav.position[1] = centroid[1] + offset[1]

        return self.get_maddpg_states()

    def step(self, actions):
        for uav in self.uavs:
            uav.profit_this_step = 0.0
            # Energy cost logic handled inside offloading functions

        self._update_active_tasks()
        for i, uav in enumerate(self.uavs):
            uav.move(np.array([actions[i][0], actions[i][1], 0]) * UAV_MAX_SPEED)
        for vehicle in self.vehicles:
            vehicle.move()

        if USE_SIMPLIFIED_OFFLOADING:
            self._assign_new_tasks_simplified()
        else:
            self._assign_new_tasks()

        self._consume_hover_energy()
        rewards = self._calculate_rewards()
        next_states = self.get_maddpg_states()
        self.time_step += 1
        done = self.time_step >= INNER_STEPS
        return next_states, rewards, done

    def get_recent_requests(self, num_requests):
        """Returns the last N (service, zone) tuples."""
        return list(self.request_history)[-num_requests:]

    # --- (Remaining methods kept mostly identical but ensuring imports and types are correct) ---

    def _consume_hover_energy(self):
        for uav in self.uavs:
            if uav.current_energy > 0:
                uav.consume_energy(ENERGY_HOVER_WATT * 1.0)

    def _calculate_rewards(self):
        if not self.uavs: return []
        rewards = []
        for uav in self.uavs:
            reward = uav.profit_this_step
            if USE_ENERGY_PENALTY:
                reward -= (uav.energy_consumed_this_step * ENERGY_REWARD_PENALTY)
            rewards.append(reward)
            uav.energy_consumed_this_step = 0.0
        global_reward = np.mean(rewards) if rewards else 0
        return [global_reward * REWARD_SCALING_FACTOR] * len(self.uavs)

    def get_maddpg_states(self):
        if not self.uavs: return []
        all_states = []
        for uav in self.uavs:
            state = [
                uav.position[0] / self.width,
                uav.position[1] / self.height,
                len([v for v in self.vehicles if self._get_distance_obj(uav, v) <= UAV_COMMUNICATION_RANGE]),
                uav.tasks_processed_count,
                uav.profit_generated / 1000.0,
                uav.current_energy / uav.max_energy if uav.max_energy > 0 else 0,
            ]
            if USE_UAV_STATUS:
                state.append(1.0 if uav.status == 'BUSY' else 0.0)
            all_states.append(np.array(state))
        return all_states

    def get_ddqn_state(self):
        num_uavs = len(self.uavs)
        total_profit = sum(uav.profit_generated for uav in self.uavs)
        total_cost = sum(BETA_MAINTENANCE + BETA_COMPUTATION * u.F_total for u in self.uavs)
        net_profit = total_profit - total_cost
        tasks_completed = sum(uav.tasks_processed_count for uav in self.uavs)

        completed_tasks = [t for v in self.vehicles for t in v.tasks if t.is_completed]
        avg_latency = np.mean([t.completed_latency for t in completed_tasks]) if completed_tasks else 0

        covered_vehicles = set()
        if num_uavs > 0:
            for uav in self.uavs:
                for v in self.vehicles:
                    if self._get_distance_obj(uav, v) <= UAV_COMMUNICATION_RANGE:
                        covered_vehicles.add(v.id)

        return np.array([tasks_completed, num_uavs, total_cost, net_profit * REWARD_SCALING_FACTOR, avg_latency,
                         len(covered_vehicles)])

    # --- Helper Calculations ---
    def _get_distance_obj(self, e1, e2):
        return np.linalg.norm(e1.position - e2.position)

    def calculate_datarate(self, bw_hz, p_watt, pl_db):
        noise_const = 10 ** ((NOISE_POWER_SPECTRAL_DENSITY - 30) / 10)
        return _calculate_datarate_numba(bw_hz, p_watt, pl_db, noise_const) / 1e6

    def calculate_path_loss(self, d, is_los):
        fspl = 20 * np.log10(d) + 20 * np.log10(CARRIER_FREQUENCY) - 147.55
        return fspl + (ETA_LOS if is_los else ETA_NLOS)

    def calculate_los_probability(self, uav, entity):
        dist_2d = np.linalg.norm(uav.position[:2] - entity.position[:2])
        delta_h = abs(uav.position[2] - entity.position[2])
        angle_deg = np.rad2deg(np.arctan(delta_h / dist_2d) if dist_2d > 0 else np.pi / 2)
        return 1 / (1 + LOS_X0 * np.exp(-LOS_Y0 * (angle_deg - LOS_X0)))

    def get_average_path_loss(self, e1, e2):
        d = self._get_distance_obj(e1, e2)
        if d == 0: return 0
        los_prob = self.calculate_los_probability(e1, e2)
        pl_los = self.calculate_path_loss(d, True)
        pl_nlos = self.calculate_path_loss(d, False)
        avg_pl = los_prob * (10 ** (pl_los / 10)) + (1 - los_prob) * (10 ** (pl_nlos / 10))
        return 10 * np.log10(avg_pl)

    def calculate_datarate_user_to_uav(self, user, uav, num_sharing_users=1):
        bw = BANDWIDTH_UAV_USER / num_sharing_users if DYNAMIC_BANDWIDTH else BANDWIDTH_UAV_USER
        return self.calculate_datarate(bw, POWER_UAV_USER, self.get_average_path_loss(uav, user))

    def calculate_datarate_uav_to_uav(self, uav1, uav2):
        return self.calculate_datarate(BANDWIDTH_UAV_UAV, POWER_UAV_UAV, self.get_average_path_loss(uav1, uav2))

    def calculate_datarate_uav_to_ccc(self, uav):
        return self.calculate_datarate(BANDWIDTH_UAV_CCC, POWER_CCC, 60)  # Fixed PL to CCC

    # --- Task Assignment Logic (Simplified) ---
    def _calculate_task_energy_cost(self, task, offload_type, entry_uav, target_uav=None, hop_path=None):
        # Simple heuristic for now
        cost = 0.0
        size_mb = task.data_size_bits / 1e6
        # Receiving
        cost += ENERGY_COMM_JOULE_PER_MBIT * size_mb
        if offload_type == 'local_uav':
            cost += ENERGY_COMPUTATION_JOULE_PER_GCYCLE * (task.cpu_cycles_req / 1e9)
        elif offload_type == 'relay_uav':
            # Sending to next
            cost += ENERGY_COMM_JOULE_PER_MBIT * size_mb
        elif offload_type == 'cloud':
            # Sending to cloud
            cost += ENERGY_COMM_JOULE_PER_MBIT * size_mb
        return cost, 0

    def _assign_new_tasks(self):
        # Placeholder if you switch back to non-simplified
        self._assign_new_tasks_simplified()

    def _assign_new_tasks_simplified(self):
        pending = [t for v in self.vehicles for t in v.tasks if t.status == 'PENDING']
        if not pending or not self.uavs: return

        for task in pending:
            vehicle = next(v for v in self.vehicles if v.id == task.owner_id)
            idle_uavs = [u for u in self.uavs if
                         u.status == 'IDLE' and self._get_distance_obj(u, vehicle) <= UAV_COMMUNICATION_RANGE]

            if not idle_uavs: continue

            # 1. LOCAL (Best)
            for uav in idle_uavs:
                if uav.has_service(task.service_type) and uav.has_content(
                        task.content_type) and uav.F_remain >= task.cpu_cycles_req:
                    self._finalize_assignment(task, 'LOCAL', uav, uav)
                    break
            else:
                # 2. RELAY (Okay)
                # Find closest entry
                entry = min(idle_uavs, key=lambda u: self._get_distance_obj(u, vehicle))
                # Find valid target
                targets = [u for u in self.uavs if u.status == 'IDLE' and u.id != entry.id and
                           u.has_service(task.service_type) and u.has_content(task.content_type) and
                           self._get_distance_obj(entry, u) <= UAV_COMMUNICATION_RANGE]

                if targets:
                    target = min(targets, key=lambda u: self._get_distance_obj(entry, u))
                    self._finalize_assignment(task, 'RELAY', entry, target)
                else:
                    # 3. CLOUD (Worst)
                    self._finalize_assignment(task, 'CLOUD', entry, None)

    def _finalize_assignment(self, task, mode, entry, target):
        task.status = 'UPLOADING'
        task.entry_uav = entry
        task.target_uav = target
        entry.status = 'BUSY'

        vehicle = next(v for v in self.vehicles if v.id == task.owner_id)
        rate = self.calculate_datarate_user_to_uav(vehicle, entry)
        up_time = math.ceil((task.data_size_bits / 1e6) / (rate + 1e-9) * LATENCY_SCALING_FACTOR)
        task.upload_complete_time = self.time_step + up_time

        if mode == 'LOCAL':
            self.local_offload_count += 1
            comp_time = math.ceil(task.cpu_cycles_req / (target.F_remain + 1e-9) * LATENCY_SCALING_FACTOR)
            task.compute_complete_time = task.upload_complete_time + comp_time
            target.F_remain -= task.cpu_cycles_req
            entry.consume_energy(self._calculate_task_energy_cost(task, 'local_uav', entry)[0])

        elif mode == 'RELAY':
            self.relay_offload_count += 1
            target.status = 'BUSY'
            relay_rate = self.calculate_datarate_uav_to_uav(entry, target)
            relay_time = math.ceil((task.data_size_bits / 1e6) / (relay_rate + 1e-9) * LATENCY_SCALING_FACTOR)
            comp_time = math.ceil(task.cpu_cycles_req / (target.F_remain + 1e-9) * LATENCY_SCALING_FACTOR)
            task.relay_complete_time = task.upload_complete_time + relay_time
            task.compute_complete_time = task.relay_complete_time + comp_time
            target.F_remain -= task.cpu_cycles_req
            entry.consume_energy(self._calculate_task_energy_cost(task, 'relay_uav', entry)[0])

        elif mode == 'CLOUD':
            self.cloud_offload_count += 1
            cloud_rate = self.calculate_datarate_uav_to_ccc(entry)
            relay_time = math.ceil((task.data_size_bits / 1e6) / (cloud_rate + 1e-9) * LATENCY_SCALING_FACTOR)
            task.compute_complete_time = task.upload_complete_time + relay_time + CLOUD_COMPUTE_LATENCY
            entry.consume_energy(self._calculate_task_energy_cost(task, 'cloud', entry)[0])

    def _update_active_tasks(self):
        for v in self.vehicles:
            for task in v.tasks:
                if task.status == 'UPLOADING' and self.time_step >= task.upload_complete_time:
                    task.status = 'RELAYING' if task.target_uav and task.target_uav != task.entry_uav else (
                        'COMPUTING' if task.target_uav else 'RELAYING')  # Cloud also relays
                    if task.target_uav is None: task.status = 'RELAYING'  # Fix for cloud

                if task.status == 'RELAYING':
                    # Check if cloud or uav relay
                    limit = task.relay_complete_time if task.target_uav else (
                                task.upload_complete_time + 5)  # Cloud assumption
                    # Cloud hack: relay time handled in finalize
                    if self.time_step >= (task.relay_complete_time if hasattr(task,
                                                                              'relay_complete_time') else task.compute_complete_time - CLOUD_COMPUTE_LATENCY):
                        task.status = 'COMPUTING'

                if task.status == 'COMPUTING' and self.time_step >= task.compute_complete_time:
                    task.status = 'COMPLETED'
                    task.is_completed = True
                    task.completed_latency = self.time_step  # Simplified

                    if task.entry_uav: task.entry_uav.status = 'IDLE'
                    if task.target_uav:
                        task.target_uav.status = 'IDLE'
                        task.target_uav.F_remain += task.cpu_cycles_req

    def get_episode_statistics(self):
        if not self.vehicles: return {}
        all_tasks = [t for v in self.vehicles for t in v.tasks]
        counts = Counter(t.status for t in all_tasks)
        return {
            'Sim/active_vehicles': len(self.vehicles),
            'Tasks/completed': counts.get('COMPLETED', 0),
            'Offloading/local': self.local_offload_count,
            'Offloading/cloud': self.cloud_offload_count
        }
