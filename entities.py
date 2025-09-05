# entities.py
"""
Defines the classes for all entities in the simulation environment.
UPDATED for Phase 4: Task and UAV classes have been significantly expanded
to support asynchronous processing over multiple time steps.
"""

import numpy as np

from config import *


class Task:
    """Represents a computational task with a multi-stage lifecycle."""

    def __init__(self, task_id, owner_vehicle_id):
        self.id = task_id
        self.owner_id = owner_vehicle_id
        self.data_size_bits = np.random.uniform(*TASK_DATA_SIZE_RANGE) * 1e6
        self.service_type = np.random.randint(0, NUM_SERVICE_TYPES)
        self.cpu_cycles_req = self.data_size_bits * TASK_CPU_CYCLES_PER_BIT
        self.latency_constraint = np.random.uniform(*LATENCY_CONSTRAINT_RANGE)

        # --- Phase 4 NEW: State machine and lifecycle tracking ---
        self.status = 'PENDING'  # PENDING, UPLOADING, RELAYING, COMPUTING, COMPLETED
        self.entry_uav = None  # The first UAV that receives the task from the vehicle
        self.target_uav = None  # The UAV that will ultimately process the task

        self.time_initiated = -1
        self.upload_complete_time = -1
        self.relay_complete_time = -1
        self.compute_complete_time = -1
        # --- End of NEW Attributes ---

        # Final metrics are now set only upon completion
        self.is_completed = False
        self.completed_latency = float('inf')
        self.profit_generated = 0.0

    def __repr__(self):
        return f"Task(id={self.id}, status={self.status}, size={self.data_size_bits / 1e6:.2f} Mbit)"


class Vehicle:
    # ... (Class unchanged from Phase 3) ...
    def __init__(self, vehicle_id):
        self.id = vehicle_id
        pos_2d = np.random.rand(2) * np.array([AREA_WIDTH, AREA_HEIGHT])
        self.position = np.append(pos_2d, 0)
        speed = np.random.uniform(VEHICLE_MIN_SPEED, VEHICLE_MAX_SPEED)
        angle = np.random.uniform(0, 2 * np.pi)
        self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle), 0])
        self.tasks = []

    def move(self, timestep=1):
        self.position += self.velocity * timestep
        self.position[0] %= AREA_WIDTH
        self.position[1] %= AREA_HEIGHT

    def generate_tasks(self): self.tasks = [Task(f"{self.id}-{i}", self.id) for i in range(TASKS_PER_VEHICLE)]

    def __repr__(self): return f"Vehicle(id={self.id}, pos={self.position})"


class UAV:
    """Represents a UAV that can now be IDLE or BUSY with asynchronous tasks."""

    def __init__(self, uav_id):
        self.id = uav_id
        pos_2d = np.random.rand(2) * np.array([AREA_WIDTH, AREA_HEIGHT])
        self.position = np.append(pos_2d, UAV_ALTITUDE)

        self.F_total = UAV_COMPUTATIONAL_RESOURCES
        initial_cache_content = np.random.choice(NUM_SERVICE_TYPES, size=UAV_CACHE_SIZE, replace=False)
        self.cache = list(initial_cache_content)
        self.cache_usage_counts = {service: 0 for service in self.cache}

        self.max_energy = np.random.uniform(*UAV_ENERGY_CAPACITY_JOULES)

        # --- Phase 4 NEW: UAV status for resource management ---
        self.status = 'IDLE'  # IDLE or BUSY
        # --- End of NEW ---

        # State variables reset each episode
        self.F_remain = self.F_total
        self.current_energy = self.max_energy
        self.tasks_processed_count = 0
        self.profit_generated = 0.0
        self.profit_this_step = 0.0
        self.energy_consumed_this_step = 0.0

    def has_service(self, service_type):
        return service_type in self.cache

    def move(self, action):
        self.position += action
        self.position[0] = np.clip(self.position[0], 0, AREA_WIDTH)
        self.position[
            1] = np.clip(self.position[1], 0, AREA_HEIGHT)
        self.position[2] = UAV_ALTITUDE

    def consume_energy(self, amount):
        consumed = min(self.current_energy, amount)
        self.current_energy -= consumed
        self.energy_consumed_this_step += consumed

    def record_cache_hit(self, service_type):
        if service_type in self.cache_usage_counts: self.cache_usage_counts[service_type] += 1

    def update_cache(self, new_service_type):
        if new_service_type in self.cache: return
        if len(self.cache) < UAV_CACHE_SIZE:
            self.cache.append(new_service_type)
            self.cache_usage_counts[new_service_type] = 1
        else:
            if not self.cache_usage_counts: return
            lfu_service = min(self.cache_usage_counts, key=self.cache_usage_counts.get)
            self.cache.remove(lfu_service)
            del self.cache_usage_counts[lfu_service]
            self.cache.append(new_service_type)
            self.cache_usage_counts[new_service_type] = 1

    def reset_for_episode(self):
        """Resets the state that changes within an episode, including status."""
        self.F_remain = self.F_total
        self.tasks_processed_count = 0
        self.profit_generated = 0.0
        self.current_energy = self.max_energy
        self.energy_consumed_this_step = 0.0
        initial_cache_content = np.random.choice(NUM_SERVICE_TYPES, size=UAV_CACHE_SIZE, replace=False)
        self.cache = list(initial_cache_content)
        self.cache_usage_counts = {service: 0 for service in self.cache}
        # --- Phase 4 CHANGE: Reset status ---
        self.status = 'IDLE'
        # --- End of CHANGE ---

    def __repr__(self):
        return f"UAV(id={self.id}, status={self.status}, energy={self.current_energy / self.max_energy:.2%})"


class CloudComputingCenter:
    def __init__(self): self.id = "CCC"
