# task_manager.py
import math

from config import *
from .comms import CommunicationModel


class TaskManager:
    def __init__(self):
        self.comm = CommunicationModel()
        self.local_count = 0
        self.relay_count = 0
        self.cloud_count = 0

    def reset_stats(self):
        self.local_count = 0
        self.relay_count = 0
        self.cloud_count = 0

    def assign_tasks(self, vehicles, uavs, current_time_step):
        """
        Iterates through pending tasks and assigns them to UAVs or Cloud.
        Uses the 'Simplified' Waterfall logic: Local -> Relay -> Cloud.
        """
        if not uavs: return

        pending_tasks = [t for v in vehicles for t in v.tasks if t.status == 'PENDING']

        for task in pending_tasks:
            vehicle = next(v for v in vehicles if v.id == task.owner_id)

            # Find UAVs within range
            idle_uavs = [u for u in uavs if u.status == 'IDLE' and
                         self.comm.get_distance(u, vehicle) <= UAV_COMMUNICATION_RANGE]

            if not idle_uavs: continue

            # --- STRATEGY 1: LOCAL OFFLOAD (Preferred) ---
            # Criteria: Has Service, Has Content, Has Compute
            best_local = None
            for uav in idle_uavs:
                if (uav.has_service(task.service_type) and
                        uav.has_content(task.content_type) and
                        uav.F_remain >= task.cpu_cycles_req):
                    best_local = uav
                    break

            if best_local:
                self._finalize_assignment(task, 'LOCAL', best_local, best_local, vehicle, current_time_step)
                continue

            # --- STRATEGY 2: RELAY OFFLOAD ---
            # Entry UAV in range -> Target UAV (anywhere) with resources
            # Find closest entry UAV
            entry_uav = min(idle_uavs, key=lambda u: self.comm.get_distance(u, vehicle))

            # Find potential targets (IDLE, Has Resources, In Range of Entry)
            # Note: Relaxing range check to allow multi-hop in future, but keeping simple 1-hop for now
            potential_targets = [u for u in uavs if u.status == 'IDLE' and u.id != entry_uav.id and
                                 u.has_service(task.service_type) and
                                 u.has_content(task.content_type) and
                                 self.comm.get_distance(entry_uav, u) <= UAV_COMMUNICATION_RANGE]

            if potential_targets:
                target_uav = min(potential_targets, key=lambda u: self.comm.get_distance(entry_uav, u))
                self._finalize_assignment(task, 'RELAY', entry_uav, target_uav, vehicle, current_time_step)
                continue

            # --- STRATEGY 3: CLOUD OFFLOAD (Fallback) ---
            # Uses entry UAV to bridge to cloud
            self._finalize_assignment(task, 'CLOUD', entry_uav, None, vehicle, current_time_step)

    def _finalize_assignment(self, task, mode, entry, target, vehicle, start_time):
        task.status = 'UPLOADING'
        task.entry_uav = entry
        task.target_uav = target
        entry.status = 'BUSY'

        # 1. Upload Time (Vehicle -> Entry)
        rate_up = self.comm.compute_rate_uav_user(vehicle, entry)
        t_up = math.ceil(task.data_size_bits / 1e6 / (rate_up + 1e-9) * LATENCY_SCALING_FACTOR)
        task.upload_complete_time = start_time + t_up

        energy_cost = 0.0

        # 2. Compute/Relay Time & Energy
        if mode == 'LOCAL':
            self.local_count += 1
            t_comp = math.ceil(task.cpu_cycles_req / (entry.F_remain + 1e-9) * LATENCY_SCALING_FACTOR)
            task.compute_complete_time = task.upload_complete_time + t_comp
            entry.F_remain -= task.cpu_cycles_req

            # Energy: Rx + Compute
            energy_cost = self._calc_energy(task, rx=True, comp=True)
            entry.consume_energy(energy_cost)

        elif mode == 'RELAY':
            self.relay_count += 1
            target.status = 'BUSY'

            rate_relay = self.comm.compute_rate_uav_uav(entry, target)
            t_relay = math.ceil(task.data_size_bits / 1e6 / (rate_relay + 1e-9) * LATENCY_SCALING_FACTOR)

            t_comp = math.ceil(task.cpu_cycles_req / (target.F_remain + 1e-9) * LATENCY_SCALING_FACTOR)

            task.relay_complete_time = task.upload_complete_time + t_relay
            task.compute_complete_time = task.relay_complete_time + t_comp

            target.F_remain -= task.cpu_cycles_req

            # Entry Energy: Rx + Tx
            entry.consume_energy(self._calc_energy(task, rx=True, tx=True))
            # Target Energy: Rx + Comp
            target.consume_energy(self._calc_energy(task, rx=True, comp=True))

        elif mode == 'CLOUD':
            self.cloud_count += 1
            rate_cloud = self.comm.compute_rate_uav_cloud(entry)
            t_relay = math.ceil(task.data_size_bits / 1e6 / (rate_cloud + 1e-9) * LATENCY_SCALING_FACTOR)

            task.compute_complete_time = task.upload_complete_time + t_relay + CLOUD_COMPUTE_LATENCY

            # Entry Energy: Rx + Tx (to cloud)
            entry.consume_energy(self._calc_energy(task, rx=True, tx=True))

    def _calc_energy(self, task, rx=False, tx=False, comp=False):
        cost = 0.0
        mb = task.data_size_bits / 1e6
        if rx: cost += ENERGY_COMM_JOULE_PER_MBIT * mb
        if tx: cost += ENERGY_COMM_JOULE_PER_MBIT * mb
        if comp: cost += ENERGY_COMPUTATION_JOULE_PER_GCYCLE * (task.cpu_cycles_req / 1e9)
        return cost

    def update_task_statuses(self, vehicles, current_time):
        """Updates the lifecycle state of all tasks."""
        for v in vehicles:
            for task in v.tasks:
                if task.is_completed: continue

                # UPLOADING -> RELAYING/COMPUTING
                if task.status == 'UPLOADING' and current_time >= task.upload_complete_time:
                    if task.target_uav and task.target_uav != task.entry_uav:
                        task.status = 'RELAYING'
                    elif task.target_uav is None:  # Cloud
                        task.status = 'RELAYING'
                    else:
                        task.status = 'COMPUTING'

                # RELAYING -> COMPUTING
                if task.status == 'RELAYING':
                    # If Cloud, we just wait for compute time (simulated latency)
                    completion_target = task.relay_complete_time if task.target_uav else task.compute_complete_time - CLOUD_COMPUTE_LATENCY
                    if current_time >= completion_target:
                        task.status = 'COMPUTING'

                # COMPUTING -> COMPLETED
                if task.status == 'COMPUTING' and current_time >= task.compute_complete_time:
                    task.status = 'COMPLETED'
                    task.is_completed = True
                    task.completed_latency = current_time - task.time_initiated

                    # Free up resources
                    if task.entry_uav: task.entry_uav.status = 'IDLE'
                    if task.target_uav:
                        task.target_uav.status = 'IDLE'
                        task.target_uav.F_remain += task.cpu_cycles_req

                    # Calculate Profit
                    self._calculate_profit(task)

    def _calculate_profit(self, task):
        # Profit = Reward - Penalty
        # Simplified reward logic based on config
        latency = max(task.completed_latency, 1e-9)

        term_latency = DELTA_LATENCY * (task.latency_constraint / latency)
        term_size = DELTA_SIZE * (task.data_size_bits / 1e6)

        profit = term_latency + term_size

        if task.entry_uav:
            task.entry_uav.profit_generated += profit
            task.entry_uav.profit_this_step += profit
            task.entry_uav.tasks_processed_count += 1
