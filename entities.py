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

        # This ensures generated tasks align with what UAVs are likely to have cached.
        self.service_type = (np.random.zipf(POPULARITY_ZIPF_ALPHA, 1)[0] - 1) % NUM_SERVICE_TYPES

        # Tasks now have a chance to require a specific piece of content as well
        if np.random.rand() < 0.5:
            self.content_type = (np.random.zipf(POPULARITY_ZIPF_ALPHA, 1)[0] - 1) % NUM_CONTENT_TYPES
        else:
            self.content_type = None

        self.cpu_cycles_req = self.data_size_bits * TASK_CPU_CYCLES_PER_BIT
        self.latency_constraint = np.random.uniform(*LATENCY_CONSTRAINT_RANGE)

        # --- Phase 4 NEW: State machine and lifecycle tracking ---
        self.status = 'PENDING'  # PENDING, UPLOADING, RELAYING, COMPUTING, COMPLETED
        self.entry_uav = None
        self.target_uav = None
        self.hop_path = []
        self.time_initiated = -1
        self.upload_complete_time = -1
        self.relay_complete_time = -1
        self.compute_complete_time = -1

        self.is_completed = False
        self.completed_latency = float('inf')
        self.profit_generated = 0.0

    def __repr__(self):
        content_str = f", content={self.content_type}" if self.content_type is not None else ""
        return f"Task(id={self.id}, status={self.status}, service={self.service_type}{content_str})"


class Vehicle:
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

    def generate_tasks(self, num_tasks=TASKS_PER_VEHICLE):
        """MODIFIED: Generates a specific number of tasks for the vehicle."""
        self.tasks = [Task(f"{self.id}-{i}", self.id) for i in range(num_tasks)]

    def __repr__(self):
        return f"Vehicle(id={self.id}, pos={self.position})"


class UAV:
    """Represents a UAV with separate service and content caches."""

    def __init__(self, uav_id):
        self.id = uav_id
        pos_2d = np.random.rand(2) * np.array([AREA_WIDTH, AREA_HEIGHT])
        self.position = np.append(pos_2d, UAV_ALTITUDE)
        self.F_total = UAV_COMPUTATIONAL_RESOURCES
        self.max_energy = np.random.uniform(*UAV_ENERGY_CAPACITY_JOULES)
        self.status = 'IDLE'

        num_services_to_cache = min(SERVICE_CACHE_SIZE, NUM_SERVICE_TYPES)
        # This line uses random choice, which is what we will change:
        self.service_cache = set(np.random.choice(range(NUM_SERVICE_TYPES), size=num_services_to_cache, replace=False))
        self.content_cache = set()
        self._precache_items()
        # --- END OF MODIFICATION ---

        # State variables reset each episode
        self.F_remain = self.F_total
        self.current_energy = self.max_energy
        self.tasks_processed_count = 0
        self.profit_generated = 0.0
        self.profit_this_step = 0.0
        self.energy_consumed_this_step = 0.0

    def _precache_items(self):
        """Fills the cache with a random set of services and content."""
        # Ensure we don't try to cache more items than exist
        num_services_to_cache = min(SERVICE_CACHE_SIZE, NUM_SERVICE_TYPES)
        self.service_cache = set(np.random.choice(range(NUM_SERVICE_TYPES), size=num_services_to_cache, replace=False))

        num_content_to_cache = min(CONTENT_CACHE_SIZE, NUM_CONTENT_TYPES)
        self.content_cache = set(np.random.choice(range(NUM_CONTENT_TYPES), size=num_content_to_cache, replace=False))

    def has_service(self, service_type):
        return service_type in self.service_cache

    def has_content(self, content_type):
        """Checks for content. Returns True if task requires no content."""
        if content_type is None:
            return True
        return content_type in self.content_cache

    def move(self, action):
        self.position += action
        self.position[0] = np.clip(self.position[0], 0, AREA_WIDTH)
        self.position[1] = np.clip(self.position[1], 0, AREA_HEIGHT)
        self.position[2] = UAV_ALTITUDE

    def consume_energy(self, amount):
        consumed = min(self.current_energy, amount)
        self.current_energy -= consumed
        self.energy_consumed_this_step += consumed

    def reset_for_episode(self):
        self.F_remain = self.F_total
        self.tasks_processed_count = 0
        self.profit_generated = 0.0
        self.current_energy = self.max_energy
        self.energy_consumed_this_step = 0.0
        self.status = 'IDLE'
        self._precache_items()

    def __repr__(self):
        return f"UAV(id={self.id}, status={self.status}, energy={self.current_energy / self.max_energy:.2%})"


class CloudComputingCenter:
    def __init__(self): self.id = "CCC"
