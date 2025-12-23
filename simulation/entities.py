# entities.py
"""
Defines the classes for all entities in the simulation environment.
UNIFIED VERSION: Supports both SUMO and Kinematic Python modes.
"""

import numpy as np

from config import *


class Task:
    """Represents a computational task with a multi-stage lifecycle."""

    def __init__(self, task_id, owner_vehicle_id, service_type=None, content_type=None):
        self.id = task_id
        self.owner_id = owner_vehicle_id
        self.data_size_bits = np.random.uniform(*TASK_DATA_SIZE_RANGE) * 1e6

        # Azure/LSTM specific attributes
        self.service_type = service_type
        self.content_type = content_type

        self.cpu_cycles_req = self.data_size_bits * TASK_CPU_CYCLES_PER_BIT
        self.latency_constraint = np.random.uniform(*LATENCY_CONSTRAINT_RANGE)

        # Lifecycle tracking
        self.status = 'PENDING'  # PENDING, UPLOADING, RELAYING, COMPUTING, COMPLETED
        self.entry_uav = None
        self.target_uav = None
        self.hop_path = []

        # Timing metrics
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
    """
    Represents a ground vehicle.
    In 'SUMO' mode, position is updated externally by the Environment via TraCI.
    In 'PYTHON_KINEMATIC' mode, position is updated internally via velocity vectors.
    """

    def __init__(self, vehicle_id):
        self.id = vehicle_id

        # Initialize position randomly within the area
        pos_2d = np.random.rand(2) * np.array([AREA_WIDTH, AREA_HEIGHT])
        self.position = np.append(pos_2d, 0)  # z=0 for ground vehicles

        # Initialize kinematic properties (used primarily in Python mode)
        speed = np.random.uniform(VEHICLE_MIN_SPEED, VEHICLE_MAX_SPEED)
        angle = np.random.uniform(0, 2 * np.pi)
        self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle), 0])

        self.tasks = []

    def move(self, timestep=1):
        """
        Updates the vehicle's position.
        """
        if SIMULATION_MODE == 'PYTHON_KINEMATIC':
            # Simple 2D movement with wrapping boundaries
            self.position += self.velocity * timestep
            self.position[0] %= AREA_WIDTH
            self.position[1] %= AREA_HEIGHT

        # In 'SUMO' mode, this method does nothing.
        # The Environment class is responsible for reading new positions from TraCI.

    def add_task(self, task):
        """Adds a generated task to the vehicle's queue."""
        self.tasks.append(task)

    def __repr__(self):
        return f"Vehicle(id={self.id}, pos={self.position})"


class UAV:
    """
    Represents a UAV with separate service and content caches.
    Supports both RL-based movement (inner loop) and LSTM-based cache updates.
    """

    def __init__(self, uav_id):
        self.id = uav_id

        # Initialize position
        pos_2d = np.random.rand(2) * np.array([AREA_WIDTH, AREA_HEIGHT])
        self.position = np.append(pos_2d, UAV_ALTITUDE)

        # Resources
        self.F_total = UAV_COMPUTATIONAL_RESOURCES
        self.max_energy = np.random.uniform(*UAV_ENERGY_CAPACITY_JOULES)
        self.status = 'IDLE'

        # Caching
        self.service_cache = set()
        self.content_cache = set()
        self._precache_items_randomly()

        # State variables reset each episode
        self.F_remain = self.F_total
        self.current_energy = self.max_energy
        self.tasks_processed_count = 0
        self.profit_generated = 0.0
        self.profit_this_step = 0.0
        self.energy_consumed_this_step = 0.0

    def _precache_items_randomly(self):
        """Fills the cache with a random set of services and content (Reactive Baseline)."""
        num_services = min(SERVICE_CACHE_SIZE, NUM_SERVICE_TYPES)
        self.service_cache = set(np.random.choice(range(NUM_SERVICE_TYPES), size=num_services, replace=False))

        num_content = min(CONTENT_CACHE_SIZE, NUM_CONTENT_TYPES)
        self.content_cache = set(np.random.choice(range(NUM_CONTENT_TYPES), size=num_content, replace=False))

    def update_cache_from_prediction(self, top_k_services, top_k_content):
        """
        Dynamically updates the cache based on predictions from the LSTM model.
        Used when USE_PREDICTIVE_CACHING is True.
        """
        self.service_cache = set(top_k_services)
        self.content_cache = set(top_k_content)

    def has_service(self, service_type):
        return service_type in self.service_cache

    def has_content(self, content_type):
        """Checks for content. Returns True if task requires no content (None)."""
        if content_type is None:
            return True
        return content_type in self.content_cache

    def move(self, action):
        """
        Updates UAV position based on RL action (velocity vector).
        This updates the internal state. In SUMO mode, the Environment must
        read this state and synchronize the SUMO proxy object.
        """
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
        self._precache_items_randomly()

    def __repr__(self):
        return f"UAV(id={self.id}, status={self.status}, energy={self.current_energy / self.max_energy:.2%})"


class CloudComputingCenter:
    def __init__(self):
        self.id = "CCC"
