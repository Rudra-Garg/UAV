# entities.py
"""
Defines the classes for all entities in the simulation environment.
UPDATED: Task and UAV classes now have detailed economic properties.
"""

import numpy as np

from config import *


class Task:
    """Represents a computational task with detailed requirements."""

    def __init__(self, task_id, owner_vehicle_id):
        self.id = task_id
        self.owner_id = owner_vehicle_id
        # Data size in bits (original was Mbits)
        self.data_size_bits = np.random.uniform(*TASK_DATA_SIZE_RANGE) * 1e6
        # The specific type of service required to process this task
        self.service_type = np.random.randint(0, NUM_SERVICE_TYPES)
        # Total CPU cycles required to process the task (cr_i,j in paper)
        self.cpu_cycles_req = self.data_size_bits * TASK_CPU_CYCLES_PER_BIT
        # Latency constraint in seconds (tr_i,j in paper)
        self.latency_constraint = np.random.uniform(*LATENCY_CONSTRAINT_RANGE)

        # --- State Tracking ---
        self.is_completed = False
        self.completed_latency = float('inf')
        self.profit_generated = 0.0

    def __repr__(self):
        return f"Task(id={self.id}, type={self.service_type}, size={self.data_size_bits / 1e6:.2f} Mbit)"


class Vehicle:
    """Represents a vehicle user that generates tasks."""

    def __init__(self, vehicle_id):
        self.id = vehicle_id
        pos_2d = np.random.rand(2) * np.array([AREA_WIDTH, AREA_HEIGHT])
        self.position = np.append(pos_2d, 0)
        speed = np.random.uniform(VEHICLE_MIN_SPEED, VEHICLE_MAX_SPEED)
        angle = np.random.uniform(0, 2 * np.pi)
        self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle), 0])
        # Tasks are now reset at the beginning of each episode in the environment
        self.tasks = []

    def move(self, timestep=1):
        """Updates the vehicle's position."""
        self.position += self.velocity * timestep
        self.position[0] %= AREA_WIDTH
        self.position[1] %= AREA_HEIGHT

    def generate_tasks(self):
        """Creates a new set of tasks for the episode."""
        self.tasks = [Task(f"{self.id}-{i}", self.id) for i in range(TASKS_PER_VEHICLE)]

    def __repr__(self):
        return f"Vehicle(id={self.id}, pos={self.position})"


class UAV:
    """Represents a UAV with computational and caching capabilities."""

    def __init__(self, uav_id):
        self.id = uav_id
        pos_2d = np.random.rand(2) * np.array([AREA_WIDTH, AREA_HEIGHT])
        self.position = np.append(pos_2d, UAV_ALTITUDE)

        # --- Economic Properties ---
        self.F_total = UAV_COMPUTATIONAL_RESOURCES
        # At the start, the UAV has all its resources available. This will be consumed by tasks.
        self.F_remain = self.F_total
        # Pre-cache a random set of unique services at the start of its life
        self.cache = np.random.choice(NUM_SERVICE_TYPES, size=UAV_CACHE_SIZE, replace=False)

        # --- State Tracking for the Episode ---
        self.tasks_processed_count = 0
        self.profit_generated = 0.0

    def has_service(self, service_type):
        """Checks if the UAV has the required service in its cache."""
        return service_type in self.cache

    def move(self, action):
        """Updates the UAV's position based on an action."""
        self.position += action
        self.position[0] = np.clip(self.position[0], 0, AREA_WIDTH)
        self.position[1] = np.clip(self.position[1], 0, AREA_HEIGHT)
        self.position[2] = UAV_ALTITUDE

    def reset_for_episode(self):
        """Resets the state that changes within an episode."""
        self.F_remain = self.F_total
        self.tasks_processed_count = 0
        self.profit_generated = 0.0

    def __repr__(self):
        return f"UAV(id={self.id}, pos={self.position})"


class CloudComputingCenter:
    """Represents the central cloud server."""

    def __init__(self):
        self.id = "CCC"
