# environment.py
"""
Implements the VECN Environment.
CORRECTED for Numba TypingError: The jitted function for calculating the
datarate has been moved outside the class to be a standalone helper function.
This is necessary because Numba's `nopython=True` mode cannot handle
class instances ('self') as arguments.
"""
import logging
import math
import os
from collections import deque, Counter

import numpy as np
import traci
import traci.exceptions
from numba import jit, njit
from scipy.cluster.vq import kmeans

from config import *
from entities import Vehicle, UAV, CloudComputingCenter

# Get a logger for this module
logger = logging.getLogger(__name__)


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
    def __init__(self, visualize=False):
        self.width, self.height, self.uavs, self.ccc = AREA_WIDTH, AREA_HEIGHT, [], CloudComputingCenter()
        self.time_step, self.total_tasks_in_step, self.completed_tasks_in_step = 0, 0, 0

        # self.vehicles is now a dictionary, keyed by SUMO vehicle ID
        self.vehicles = {}

        # Vehicle spawning management
        self.available_edges = []
        self.target_vehicle_count = 0
        self.vehicle_spawn_counter = 0

        self.local_offload_count = 0
        self.relay_offload_count = 0
        self.cloud_offload_count = 0

        self.visualize = visualize

        if visualize:
            self.sumo_binary = "sumo-gui"
        else:
            self.sumo_binary = "sumo"

        self.sumo_cmd = [
            self.sumo_binary,
            "-c", "sumo_scenario/grid.sumocfg",
            "--step-length", "1",
            "--quit-on-end", "--start",
            "--no-warnings", "--no-step-log",
            "--error-log", "sumo_errors.log",
        ]

    def reset(self, num_uavs=0, num_vehicles=NUM_VEHICLES):
        logger.info("Resetting environment with %d UAVs and target of %d vehicles.", num_uavs, num_vehicles)
        self.time_step = 0
        self.target_vehicle_count = num_vehicles
        self.vehicle_spawn_counter = 0
        self.local_offload_count = 0
        self.relay_offload_count = 0
        self.cloud_offload_count = 0
        try:
            if traci.isLoaded():
                # FIXED: Use self.visualize instead of checking GUI
                if self.visualize:
                    for uav in self.uavs:
                        try:
                            traci.polygon.remove(f"range_{uav.id}")
                        except traci.exceptions.TraCIException:
                            pass
                traci.close()
        except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException):
            pass

        # --- MODIFICATION: Select a random real-world scenario for this episode ---
        selected_city = random.choice(SUMO_SCENARIO_POOL)
        sumo_config_path = os.path.join("sumo_scenario", f"{selected_city}.sumocfg")
        self.sumo_cmd[2] = sumo_config_path
        logger.info("Starting SUMO with real-world scenario: %s", sumo_config_path)
        # --- END OF MODIFICATION ---

        traci.start(self.sumo_cmd)

        # Load the network to get available edges for spawning
        traci.simulationStep()
        self._load_network_edges()

        # Spawn initial vehicle fleet
        self._spawn_initial_vehicles(num_vehicles)

        # Run a few steps to let vehicles distribute
        for _ in range(5):
            traci.simulationStep()

        # Sync our Python vehicle objects with SUMO
        self._update_vehicles_from_sumo()
        logger.info("Initial vehicle count: %d", len(self.vehicles))

        self.uavs = [UAV(i) for i in range(num_uavs)]
        for uav in self.uavs:
            uav.reset_for_episode()  # Ensure UAVs are fully reset
            uav_id_sumo = f"uav_{uav.id}"
            traci.vehicle.add(
                vehID=uav_id_sumo,
                routeID='dummy_route',
                typeID='UAV_TYPE'
            )
            traci.vehicle.setColor(uav_id_sumo, (255, 0, 0, 255))
            traci.vehicle.setShapeClass(uav_id_sumo, "aircraft")
            traci.vehicle.moveToXY(
                vehID=uav_id_sumo,
                edgeID="",
                laneIndex=-1,
                x=uav.position[0],
                y=uav.position[1],
                keepRoute=2
            )

            # FIXED: Use self.visualize
            if self.visualize:
                poly_id = f"range_{uav.id}"
                color = (0, 255, 0, 60)
                shape = self._get_circle_shape(uav.position, UAV_COMMUNICATION_RANGE)
                traci.polygon.add(poly_id, shape, color, layer=0, fill=True)

        if self.uavs and self.vehicles:
            vehicle_positions = np.array([v.position[:2] for v in self.vehicles.values()])
            if len(vehicle_positions) >= num_uavs > 0:
                centroids, _ = kmeans(vehicle_positions, num_uavs, iter=10)
                for i, uav in enumerate(self.uavs):
                    centroid_to_assign = centroids[i % len(centroids)]
                    offset = (np.random.rand(2) * 2 - 1) * 100
                    uav.position[0] = centroid_to_assign[0] + offset[0]
                    uav.position[1] = centroid_to_assign[1] + offset[1]
                    traci.vehicle.moveToXY(f"uav_{uav.id}", "", -1, uav.position[0], uav.position[1], keepRoute=2)

                    # FIXED: Use self.visualize
                    if self.visualize:
                        poly_id = f"range_{uav.id}"
                        shape = self._get_circle_shape(uav.position, UAV_COMMUNICATION_RANGE)
                        traci.polygon.setShape(poly_id, shape)

        return self.get_maddpg_states()

    def _load_network_edges(self):
        """Load available edges from the SUMO network for vehicle spawning."""
        all_edges = traci.edge.getIDList()
        # Filter out internal edges (those with ':' in the name)
        self.available_edges = [e for e in all_edges if ':' not in e]
        logger.info("Loaded %d available edges for vehicle spawning", len(self.available_edges))

    def _spawn_initial_vehicles(self, target_count):
        """Spawn initial vehicle fleet at random positions."""
        if not self.available_edges:
            logger.warning("No available edges for vehicle spawning!")
            return

        for i in range(target_count):
            self._spawn_single_vehicle()

        logger.info("Spawned %d initial vehicles", target_count)

    def _spawn_single_vehicle(self):
        """Spawn a single vehicle on a random edge."""
        if not self.available_edges:
            return False

        try:
            vehicle_id = f"vehicle_{self.vehicle_spawn_counter}"
            self.vehicle_spawn_counter += 1

            # Pick random start and end edges
            from_edge = np.random.choice(self.available_edges)
            to_edge = np.random.choice(self.available_edges)

            # Try to find a route
            try:
                route = traci.simulation.findRoute(from_edge, to_edge)
                if route and len(route.edges) > 0:
                    route_id = f"route_{vehicle_id}"
                    traci.route.add(route_id, route.edges)

                    # Add the vehicle
                    traci.vehicle.add(
                        vehID=vehicle_id,
                        routeID=route_id,
                        typeID='DEFAULT_VEHTYPE'
                    )
                    return True
            except traci.exceptions.TraCIException:
                # No valid route found, try different edges
                pass

            return False

        except Exception as e:
            logger.debug(f"Failed to spawn vehicle: {e}")
            return False

    def _maintain_vehicle_count(self):
        """Respawn vehicles to maintain target count."""
        current_count = len([v_id for v_id in traci.vehicle.getIDList() if not v_id.startswith('uav_')])

        if current_count < self.target_vehicle_count:
            vehicles_to_spawn = self.target_vehicle_count - current_count
            spawned = 0
            # Try to spawn up to 10 vehicles per step to avoid slowdown
            for _ in range(min(vehicles_to_spawn, 10)):
                if self._spawn_single_vehicle():
                    spawned += 1

            if spawned > 0:
                logger.debug(f"Respawned {spawned} vehicles (current: {current_count + spawned}/{self.target_vehicle_count})")

    def step(self, actions, visualize=False, episode_num=0, step_num=0):
        for uav in self.uavs:
            uav.profit_this_step = 0.0

        self._update_active_tasks()

        for i, uav in enumerate(self.uavs):
            uav.move(np.array([actions[i][0], actions[i][1], 0]) * UAV_MAX_SPEED)

        for uav in self.uavs:
            traci.vehicle.moveToXY(
                vehID=f"uav_{uav.id}",
                edgeID="",
                laneIndex=-1,
                x=uav.position[0],
                y=uav.position[1],
                keepRoute=2
            )

            # FIXED: Use self.visualize
            if self.visualize:
                try:
                    poly_id = f"range_{uav.id}"
                    shape = self._get_circle_shape(uav.position, UAV_COMMUNICATION_RANGE)
                    traci.polygon.setShape(poly_id, shape)
                except traci.exceptions.TraCIException:
                    pass

        traci.simulationStep()

        # Maintain vehicle count by respawning
        self._maintain_vehicle_count()

        self._update_vehicles_from_sumo()

        if USE_SIMPLIFIED_OFFLOADING:
            self._assign_new_tasks_simplified()
        else:
            self._assign_new_tasks()

        self._consume_hover_energy()
        rewards = self._calculate_rewards()
        next_states = self.get_maddpg_states()
        self.time_step += 1
        done = self.time_step >= INNER_STEPS

        if done:
            traci.close()

        return next_states, rewards, done

    def close(self):
        """
        Properly close the SUMO connection and clean up resources.
        This method can be called multiple times safely.
        """
        try:
            # Check if TraCI is loaded before attempting to close
            if traci.isLoaded():
                logger.debug("Closing SUMO connection...")

                # Clean up visualization elements if they exist
                if self.visualize and self.uavs:
                    for uav in self.uavs:
                        try:
                            traci.polygon.remove(f"range_{uav.id}")
                        except (traci.exceptions.TraCIException, traci.exceptions.FatalTraCIError):
                            pass  # Polygon might not exist or connection already closed

                # Close the TraCI connection
                traci.close()
                logger.debug("SUMO connection closed successfully.")

        except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException) as e:
            # Connection might already be closed or never opened
            logger.debug(f"TraCI close attempt: {e}")
            pass

        except Exception as e:
            # Catch any other unexpected errors during cleanup
            logger.warning(f"Unexpected error during environment close: {e}")
            pass

    def __del__(self):
        """
        Destructor to ensure SUMO is closed when the environment object is deleted.
        """
        self.close()

    def _get_circle_shape(self, center_pos, radius, num_points=20):
        """Generates a list of (x,y) points for a circle polygon."""
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        points = []
        for angle in angles:
            x = center_pos[0] + radius * np.cos(angle)
            y = center_pos[1] + radius * np.sin(angle)
            points.append((x, y))
        return points

    def _update_vehicles_from_sumo(self):
        """
        NEW helper method to synchronize Python vehicle objects with SUMO.
        - Removes departed vehicles.
        - Adds newly arrived vehicles.
        - Updates positions of all current vehicles.
        """
        all_sumo_ids = traci.vehicle.getIDList()
        current_sumo_ids = set(v_id for v_id in all_sumo_ids if not v_id.startswith('uav_'))

        # Remove vehicles that have left the simulation
        for v_id in list(self.vehicles.keys()):
            if v_id not in current_sumo_ids:
                del self.vehicles[v_id]

        # Add new vehicles and update existing ones
        for v_id in current_sumo_ids:
            if v_id not in self.vehicles:
                self.vehicles[v_id] = Vehicle(v_id)
                # Generate tasks for the new vehicle
                self.vehicles[v_id].generate_tasks()

            # Update position and speed from SUMO
            pos = traci.vehicle.getPosition(v_id)
            self.vehicles[v_id].position = np.array([pos[0], pos[1], 0])

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
        total_profit = 0
        total_energy_penalty = 0

        for uav in self.uavs:
            reward = uav.profit_this_step
            total_profit += reward
            # Conditionally apply the energy penalty
            if USE_ENERGY_PENALTY:
                penalty = uav.energy_consumed_this_step * ENERGY_REWARD_PENALTY
                reward -= penalty
                total_energy_penalty += penalty
            rewards.append(reward)
            uav.energy_consumed_this_step = 0.0

        logger.debug("Reward calc: Profit=%.2f, EnergyPenalty=%.4f", total_profit, total_energy_penalty)

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
                len([v for v in self.vehicles.values() if self._get_distance_obj(uav, v) <= UAV_COMMUNICATION_RANGE]),
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
        avg_latency = np.mean([t.completed_latency for v in self.vehicles.values() for t in v.tasks if
                               t.is_completed]) if tasks_completed > 0 else 0
        covered_vehicles = set()
        if num_uavs > 0:
            for uav in self.uavs:
                for v in self.vehicles.values():
                    if self._get_distance_obj(uav, v) <= UAV_COMMUNICATION_RANGE: covered_vehicles.add(v.id)
        return np.array([tasks_completed, num_uavs, total_cost, net_profit * REWARD_SCALING_FACTOR, avg_latency,
                         len(covered_vehicles)])

    # --- Communication Modeling ---
    def calculate_datarate_uav_to_uav(self, uav1, uav2):
        return self.calculate_datarate(BANDWIDTH_UAV_UAV, POWER_UAV_UAV, self.get_average_path_loss(uav1, uav2))

    def calculate_datarate_user_to_uav(self, user, uav, num_sharing_users=1):
        """
        Calculates the data rate from a user to a UAV, considering TDMA-based
        bandwidth sharing if enabled in the
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
        visited = set()

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
                    if neighbor_idx not in path:  # simple cycle check
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
        all_tasks = [task for vehicle in self.vehicles.values() for task in vehicle.tasks]
        for task in all_tasks:
            original_status = task.status

            if task.status == 'UPLOADING' and self.time_step >= task.upload_complete_time:
                # After uploading, if there's a different target, it needs relaying.
                if task.target_uav and task.target_uav.id != task.entry_uav.id:
                    task.status = 'RELAYING'
                # If target is cloud (None) it also needs relaying.
                elif task.target_uav is None:
                    task.status = 'RELAYING'
                # Otherwise, it was a local computation.
                else:
                    task.status = 'COMPUTING'

            if task.status == 'RELAYING' and self.time_step >= task.relay_complete_time:
                task.status = 'COMPUTING'

            if task.status == 'COMPUTING' and self.time_step >= task.compute_complete_time:
                task.status = 'COMPLETED'
                task.is_completed = True
                task.completed_latency = self.time_step - task.time_initiated

                # Assign profit to the entry UAV that initiated the offload
                if task.entry_uav:
                    task.profit_generated = self._calculate_earn_task(task, task.completed_latency, task.entry_uav)
                    task.entry_uav.profit_this_step += task.profit_generated
                    task.entry_uav.profit_generated += task.profit_generated
                    task.entry_uav.tasks_processed_count += 1

                # --- CORRECTION: Free up ALL involved UAVs ---
                if task.entry_uav:
                    task.entry_uav.status = 'IDLE'
                if task.target_uav:
                    task.target_uav.status = 'IDLE'
                    # Restore computational resources to the processing UAV
                    task.target_uav.F_remain += task.cpu_cycles_req

            if task.status != original_status:
                logger.debug("Task %s status changed from %s to %s at step %d.", task.id, original_status, task.status,
                             self.time_step)

    def _assign_new_tasks(self):
        pending_tasks = [task for v in self.vehicles.values() for task in v.tasks if task.status == 'PENDING']

        # High-level log for the start of the assignment phase
        logger.debug("Attempting to assign %d pending tasks at step %d.", len(pending_tasks), self.time_step)

        if not self.uavs:
            logger.debug("No UAVs available, skipping assignment.")
            return

        # Detailed log for UAV status
        for uav in self.uavs:
            logger.debug(
                f"UAV {uav.id}: Status={uav.status}, Energy={uav.current_energy:.1f}, F_remain={uav.F_remain:.1f}, "
                f"Service_cache(size={len(uav.service_cache)}): {uav.service_cache[:5]}, "
                f"Content_cache(size={len(uav.content_cache)}): {uav.content_cache[:5]}")

        # Pre-calculate TDMA load
        uav_potential_load = {uav.id: 0 for uav in self.uavs}
        if DYNAMIC_BANDWIDTH:
            vehicles_with_tasks = {task.owner_id for task in pending_tasks}
            for vehicle_id in vehicles_with_tasks:
                vehicle = next(v for v in self.vehicles.values() if v.id == vehicle_id)
                for uav in self.uavs:
                    if uav.status == 'IDLE' and self._get_distance_obj(uav, vehicle) <= UAV_COMMUNICATION_RANGE:
                        uav_potential_load[uav.id] += 1

        for task in pending_tasks:
            vehicle = next(v for v in self.vehicles.values() if v.id == task.owner_id)
            idle_uavs_in_range = [u for u in self.uavs if
                                  u.status == 'IDLE' and self._get_distance_obj(u, vehicle) <= UAV_COMMUNICATION_RANGE]

            # Detailed log for the task being evaluated
            logger.debug(
                f"Evaluating Task {task.id}: {len(idle_uavs_in_range)} idle UAVs in range. Requires service={task.service_type}, content={task.content_type}")

            if not idle_uavs_in_range:
                continue

            task.time_initiated = self.time_step
            task_size_mbit = task.data_size_bits / 1e6

            best_local_option = None
            best_local_latency = float('inf')

            # --- Step 1: Prioritize Local Processing ---
            for entry_uav in idle_uavs_in_range:
                cost_entry, _ = self._calculate_task_energy_cost(task, 'local_uav', entry_uav)
                if (entry_uav.has_service(task.service_type) and
                        entry_uav.has_content(task.content_type) and
                        entry_uav.F_remain > task.cpu_cycles_req and
                        entry_uav.current_energy > cost_entry):
                    num_sharers = max(1, uav_potential_load.get(entry_uav.id, 1))
                    datarate = self.calculate_datarate_user_to_uav(vehicle, entry_uav, num_sharing_users=num_sharers)
                    upload_duration = math.ceil(task_size_mbit / (datarate + 1e-9))
                    compute_duration = math.ceil(task.cpu_cycles_req / (entry_uav.F_remain + 1e-9))
                    latency = upload_duration + compute_duration
                    if latency < best_local_latency and (self.time_step + latency) <= task.latency_constraint:
                        best_local_latency = latency
                        profit = self._calculate_earn_task(task, latency, entry_uav)
                        best_local_option = ('local_uav', entry_uav, entry_uav, latency, profit, [entry_uav.id])

            if best_local_option:
                logger.debug(f"Task {task.id}: Found viable local option. Assigning.")
                self._finalize_task_assignment(task, best_local_option, vehicle, uav_potential_load)
                continue

            # --- Step 2: Try Relaying if Local Failed ---
            logger.debug(f"Task {task.id}: No suitable local UAV found. Evaluating relay options.")
            best_relay_option = None
            best_relay_latency = float('inf')
            target_uavs_with_service = [u for u in self.uavs if u.has_service(task.service_type)]

            if target_uavs_with_service:
                for entry_uav in idle_uavs_in_range:
                    hop_path_indices = self._find_multi_hop_path(task, entry_uav,
                                                                 target_uav_candidates=target_uavs_with_service)
                    if hop_path_indices:
                        hop_path_uavs = [self.uavs[idx] for idx in hop_path_indices]
                        target_uav = hop_path_uavs[-1]
                        num_sharers = max(1, uav_potential_load.get(entry_uav.id, 1))
                        upload_rate = self.calculate_datarate_user_to_uav(vehicle, entry_uav,
                                                                          num_sharing_users=num_sharers)
                        upload_duration = math.ceil(task_size_mbit / (upload_rate + 1e-9))
                        relay_duration = sum(math.ceil(task_size_mbit / (
                                self.calculate_datarate_uav_to_uav(hop_path_uavs[i], hop_path_uavs[i + 1]) + 1e-9))
                                             for i in range(len(hop_path_uavs) - 1))
                        compute_duration = math.ceil(task.cpu_cycles_req / (target_uav.F_remain + 1e-9))
                        latency = upload_duration + relay_duration + compute_duration
                        if latency < best_relay_latency and (self.time_step + latency) <= task.latency_constraint:
                            best_relay_latency = latency
                            profit = self._calculate_earn_task(task, latency, entry_uav)
                            best_relay_option = ('relay_uav', entry_uav, target_uav, latency, profit,
                                                 [u.id for u in hop_path_uavs])

            if best_relay_option:
                logger.debug(f"Task {task.id}: Found viable relay option. Assigning.")
                self._finalize_task_assignment(task, best_relay_option, vehicle, uav_potential_load)
                continue

            # --- Step 3: Use Cloud as a Last Resort ---
            logger.debug(f"Task {task.id}: No suitable relay path found. Evaluating cloud as last resort.")
            best_cloud_option = None
            best_cloud_latency = float('inf')

            for entry_uav in idle_uavs_in_range:
                cost_entry, _ = self._calculate_task_energy_cost(task, 'cloud', entry_uav)
                if entry_uav.current_energy > cost_entry:
                    num_sharers = max(1, uav_potential_load.get(entry_uav.id, 1))
                    rate_user = self.calculate_datarate_user_to_uav(vehicle, entry_uav, num_sharing_users=num_sharers)
                    rate_ccc = self.calculate_datarate_uav_to_ccc(entry_uav)
                    upload_duration = math.ceil(task_size_mbit / (rate_user + 1e-9))
                    relay_duration = math.ceil(task_size_mbit / (rate_ccc + 1e-9))
                    latency = upload_duration + relay_duration
                    if latency < best_cloud_latency and (self.time_step + latency) <= task.latency_constraint:
                        best_cloud_latency = latency
                        profit = self._calculate_earn_task(task, latency)
                        best_cloud_option = ('cloud', entry_uav, None, latency, profit, [entry_uav.id])

            if best_cloud_option:
                logger.debug(f"Task {task.id}: Assigning to cloud via best entry UAV.")
                self._finalize_task_assignment(task, best_cloud_option, vehicle, uav_potential_load)
            else:
                logger.debug(f"Task {task.id}: No viable offloading option found. Task remains PENDING.")

    def _finalize_task_assignment(self, task, best_option, vehicle, uav_potential_load):
        """
        Helper method to finalize the assignment and update states.
        """
        offload_type, entry_uav, target_uav, latency, profit, hop_path_ids = best_option

        # RESTORED THIS IMPORTANT LOG
        logger.info("Task %s assigned: Type=%s, EntryUAV=%d, TargetUAV=%s, Latency=%.2f, Profit=%.2f",
                    task.id, offload_type, entry_uav.id, target_uav.id if target_uav else "CCC", latency, profit)

        task.status = 'UPLOADING'
        task.profit_generated = profit
        task.entry_uav, task.target_uav = entry_uav, target_uav
        task.hop_path = hop_path_ids

        num_sharers = max(1, uav_potential_load.get(entry_uav.id, 1))
        upload_rate = self.calculate_datarate_user_to_uav(vehicle, entry_uav, num_sharing_users=num_sharers)
        task.upload_complete_time = self.time_step + math.ceil(task.data_size_bits / (upload_rate * 1e6 + 1e-9))

        if offload_type == 'local_uav':
            task.compute_complete_time = task.upload_complete_time + math.ceil(
                task.cpu_cycles_req / (entry_uav.F_remain + 1e-9))
            entry_uav.record_service_cache_hit(task.service_type)
            entry_uav.record_content_cache_hit(task.content_type)
        elif offload_type == 'relay_uav':
            relay_duration = 0
            for i in range(len(hop_path_ids) - 1):
                uav1 = next(u for u in self.uavs if u.id == hop_path_ids[i])
                uav2 = next(u for u in self.uavs if u.id == hop_path_ids[i + 1])
                relay_rate = self.calculate_datarate_uav_to_uav(uav1, uav2)
                relay_duration += math.ceil(task.data_size_bits / (relay_rate * 1e6 + 1e-9))
            task.relay_complete_time = task.upload_complete_time + relay_duration
            task.compute_complete_time = task.relay_complete_time + math.ceil(
                task.cpu_cycles_req / (target_uav.F_remain + 1e-9))
            target_uav.record_service_cache_hit(task.service_type)
            target_uav.record_content_cache_hit(task.content_type)
        elif offload_type == 'cloud':
            CLOUD_COMPUTE_TIME = 5  # Added a nominal compute time for realism
            relay_duration = math.ceil(
                task.data_size_bits / (self.calculate_datarate_uav_to_ccc(entry_uav) * 1e6 + 1e-9))
            task.compute_complete_time = task.upload_complete_time + relay_duration + CLOUD_COMPUTE_TIME

        if np.random.rand() < CACHE_UPDATE_PROBABILITY:
            entry_uav.update_service_cache(task.service_type)
            entry_uav.update_content_cache(task.content_type)

        cost_entry, cost_target = self._calculate_task_energy_cost(task, offload_type, entry_uav, target_uav,
                                                                   hop_path_ids if offload_type == 'relay_uav' else [])
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

    def _assign_new_tasks_simplified(self):
        """
        NEW cache-aware task assignment logic.
        1. Tries to find a local UAV with the right cache.
        2. If not, tries to find a relay path to a UAV with the right cache.
        3. If not, offloads to the cloud as a last resort.
        """
        pending_tasks = [task for v in self.vehicles.values() for task in v.tasks if task.status == 'PENDING']
        if not self.uavs or not pending_tasks:
            return

        for task in pending_tasks:
            vehicle = next(v for v in self.vehicles.values() if v.id == task.owner_id)

            # Find all IDLE UAVs within the vehicle's communication range
            idle_uavs_in_range = [
                u for u in self.uavs if u.status == 'IDLE' and
                                        self._get_distance_obj(u, vehicle) <= UAV_COMMUNICATION_RANGE
            ]
            if not idle_uavs_in_range:
                continue  # No UAVs available for this task right now

            task.time_initiated = self.time_step

            # --- Priority 1: Find a LOCAL processor ---
            # Search for a UAV in range that has the right service/content and resources
            local_processor = None
            found_local_candidate = False
            for uav in idle_uavs_in_range:
                # Add detailed checks and print statements
                has_service = uav.has_service(task.service_type)
                has_content = uav.has_content(task.content_type)
                has_resources = uav.F_remain >= task.cpu_cycles_req

                if has_service and has_content and has_resources:
                    local_processor = uav
                    found_local_candidate = True  # Mark that we found one
                    logger.info(f"Task {task.id}: Found suitable LOCAL processor UAV {uav.id}.")
                    break  # Found one, stop searching

                else:
                    # This is the crucial part: log WHY it failed
                    reasons = []
                    if not has_service: reasons.append("SERVICE_CACHE_MISS")
                    if not has_content: reasons.append("CONTENT_CACHE_MISS")
                    if not has_resources: reasons.append("INSUFFICIENT_RESOURCES")
                    logger.debug(f"Task {task.id}: Skipping local UAV {uav.id}. Reasons: {', '.join(reasons)}")

            if local_processor:
                self._finalize_task_assignment_simplified(task, 'LOCAL_UAV', local_processor, local_processor)
                continue  # Assignment successful, move to the next task

            # --- Priority 2: Find a RELAY path ---
            # If no local processor was found, we need an entry point and a separate target
            entry_uav = min(idle_uavs_in_range, key=lambda u: self._get_distance_obj(u, vehicle))

            # Search ALL UAVs (not just those in range of the vehicle) for a suitable target
            potential_targets = [
                uav for uav in self.uavs if
                uav.status == 'IDLE' and uav.id != entry_uav.id and
                self._get_distance_obj(uav,
                                       entry_uav) <= UAV_COMMUNICATION_RANGE and  # Target must be in range of entry UAV
                uav.has_service(task.service_type) and
                uav.has_content(task.content_type) and
                uav.F_remain >= task.cpu_cycles_req
            ]

            if potential_targets:
                # Find the closest valid target to the entry UAV
                target_uav = min(potential_targets, key=lambda u: self._get_distance_obj(u, entry_uav))
                self._finalize_task_assignment_simplified(task, 'RELAY_UAV', entry_uav, target_uav)
                continue  # Assignment successful, move to the next task

            # --- Priority 3: Offload to CLOUD as a last resort ---
            # Use the closest UAV in range as the entry point to the cloud
            cloud_entry_uav = min(idle_uavs_in_range, key=lambda u: self._get_distance_obj(u, vehicle))
            self._finalize_task_assignment_simplified(task, 'CLOUD', cloud_entry_uav, None)

    def get_episode_statistics(self):
        """
        Calculates and returns a dictionary of detailed statistics for the completed episode.
        This should be called at the end of an episode, before `reset`.
        """
        if not self.vehicles:
            return {}  # Return empty dict if no vehicles

        # --- Task Statistics ---
        all_tasks = [task for v in self.vehicles.values() for task in v.tasks]
        task_status_counts = Counter(t.status for t in all_tasks)

        # --- Vehicle Coverage ---
        covered_vehicles = set()
        if self.uavs:
            for uav in self.uavs:
                for v in self.vehicles.values():
                    if self._get_distance_obj(uav, v) <= UAV_COMMUNICATION_RANGE:
                        covered_vehicles.add(v.id)
        coverage_ratio = len(covered_vehicles) / len(self.vehicles) if self.vehicles else 0

        # --- UAV Fleet Statistics ---
        if self.uavs:
            avg_energy_pct = np.mean([u.current_energy / u.max_energy for u in self.uavs]) * 100
            avg_compute_load_pct = (1 - np.mean([u.F_remain / u.F_total for u in self.uavs])) * 100
            busy_uavs = sum(1 for u in self.uavs if u.status == 'BUSY')
            uav_busy_pct = (busy_uavs / len(self.uavs)) * 100
        else:
            avg_energy_pct, avg_compute_load_pct, uav_busy_pct = 0, 0, 0

        stats = {
            'SUMO/active_vehicles': len(self.vehicles),
            'SUMO/coverage_ratio': coverage_ratio,
            'Tasks/pending': task_status_counts.get('PENDING', 0),
            'Tasks/uploading': task_status_counts.get('UPLOADING', 0),
            'Tasks/computing': task_status_counts.get('COMPUTING', 0),
            'Tasks/completed_in_episode': task_status_counts.get('COMPLETED', 0),
            'Offloading/local_uav': self.local_offload_count,
            'Offloading/relay_uav': self.relay_offload_count,
            'Offloading/cloud': self.cloud_offload_count,
            'UAV/avg_energy_remaining_pct': avg_energy_pct,
            'UAV/avg_compute_load_pct': avg_compute_load_pct,
            'UAV/busy_pct': uav_busy_pct,
        }
        return stats

    def _finalize_task_assignment_simplified(self, task, destination, entry_uav, target_uav):
        """
        NEW helper to finalize assignment for the 3 simplified UAV-based scenarios.
        """

        if destination == 'LOCAL_UAV':
            self.local_offload_count += 1
        elif destination == 'RELAY_UAV':
            self.relay_offload_count += 1
        elif destination == 'CLOUD':
            self.cloud_offload_count += 1
        task_size_mbit = task.data_size_bits / 1e6
        vehicle = next(v for v in self.vehicles.values() if v.id == task.owner_id)

        # Common first step: Upload from vehicle to entry UAV
        datarate_to_entry = self.calculate_datarate_user_to_uav(vehicle, entry_uav)
        raw_upload_seconds = task_size_mbit / (datarate_to_entry + 1e-9)
        upload_duration = math.ceil(raw_upload_seconds * LATENCY_SCALING_FACTOR)
        task.upload_complete_time = self.time_step + upload_duration

        task.status = 'UPLOADING'
        task.entry_uav = entry_uav
        task.target_uav = target_uav
        entry_uav.status = 'BUSY'

        if destination == 'LOCAL_UAV':
            raw_compute_seconds = task.cpu_cycles_req / (target_uav.F_remain + 1e-9)
            compute_duration = math.ceil(raw_compute_seconds * LATENCY_SCALING_FACTOR)
            task.compute_complete_time = task.upload_complete_time + compute_duration
            target_uav.F_remain -= task.cpu_cycles_req
            logger.info(
                f"Task {task.id} assigned to LOCAL UAV {entry_uav.id}. Latency: {upload_duration + compute_duration} steps.")

        elif destination == 'RELAY_UAV':
            target_uav.status = 'BUSY'
            datarate_relay = self.calculate_datarate_uav_to_uav(entry_uav, target_uav)
            raw_relay_seconds = task_size_mbit / (datarate_relay + 1e-9)
            relay_duration = math.ceil(raw_relay_seconds * LATENCY_SCALING_FACTOR)
            raw_compute_seconds = task.cpu_cycles_req / (target_uav.F_remain + 1e-9)
            compute_duration = math.ceil(raw_compute_seconds * LATENCY_SCALING_FACTOR)
            task.relay_complete_time = task.upload_complete_time + relay_duration
            task.compute_complete_time = task.relay_complete_time + compute_duration
            target_uav.F_remain -= task.cpu_cycles_req
            logger.info(
                f"Task {task.id} assigned to RELAY UAV {target_uav.id} via {entry_uav.id}. Latency: {upload_duration + relay_duration + compute_duration} steps.")

        elif destination == 'CLOUD':
            datarate_to_cloud = self.calculate_datarate_uav_to_ccc(entry_uav)
            raw_relay_seconds = task_size_mbit / (datarate_to_cloud + 1e-9)
            relay_duration = math.ceil(raw_relay_seconds * LATENCY_SCALING_FACTOR)

            task.relay_complete_time = task.upload_complete_time + relay_duration
            task.compute_complete_time = task.relay_complete_time + CLOUD_COMPUTE_LATENCY
            logger.info(
                f"Task {task.id} assigned to CLOUD via UAV {entry_uav.id}. Latency: {upload_duration + relay_duration + CLOUD_COMPUTE_LATENCY} steps.")

        # Simplified energy cost - for now, just apply to the entry UAV
        cost_entry, _ = self._calculate_task_energy_cost(task, 'local_uav', entry_uav)  # Approximation is fine
        entry_uav.consume_energy(cost_entry)
