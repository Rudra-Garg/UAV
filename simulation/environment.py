import logging
from collections import deque

import numpy as np
from scipy.cluster.vq import kmeans

from config import *
from prediction import DemandGenerator
from .comms import CommunicationModel
from .entities import UAV, Task
from .physics import PhysicsConnector
from .tasks import TaskManager

logger = logging.getLogger(__name__)


class VECNEnvironment:
    def __init__(self):
        self.width = AREA_WIDTH
        self.height = AREA_HEIGHT

        # Components
        self.physics = PhysicsConnector(self.width, self.height)
        self.task_manager = TaskManager()
        self.demand_generator = DemandGenerator()
        self.comm = CommunicationModel()

        # State
        self.uavs = []
        # vehicles is a dict {id: Vehicle}
        self.vehicles = {}
        self.time_step = 0
        self.request_history = deque(maxlen=PREDICTION_SEQUENCE_LENGTH * 2)

    def reset(self, num_uavs=0, num_vehicles=NUM_VEHICLES):
        logger.debug(f"Resetting Env: {num_uavs} UAVs, {num_vehicles} Vehicles. Mode: {SIMULATION_MODE}")

        self.time_step = 0
        self.task_manager.reset_stats()
        self.demand_generator.reset()
        self.request_history.clear()

        # 1. Reset Physics (Spawns Vehicles)
        raw_vehicles = self.physics.reset(num_vehicles)
        if isinstance(raw_vehicles, list):
            self.vehicles = {v.id: v for v in raw_vehicles}
        else:
            self.vehicles = raw_vehicles

        # 2. Reset UAVs (Cluster Deployment)
        self.uavs = [UAV(i) for i in range(num_uavs)]
        if self.uavs and self.vehicles:
            self._deploy_uavs_clustered()

        # 2.5 ADD UAVs TO SUMO IMMEDIATELY AFTER DEPLOYMENT
        self.physics.add_uavs_to_sumo(self.uavs)

        # 3. Generate Initial Tasks
        self._generate_tasks_for_all()

        return self.get_maddpg_states()

    def step(self, actions):
        # 1. Clear step stats
        for uav in self.uavs:
            uav.profit_this_step = 0.0

        # 2. Physics Step (Move UAVs and Vehicles)
        # Apply actions (velocities) to UAVs
        for i, uav in enumerate(self.uavs):
            if i < len(actions):
                uav.move(np.array([actions[i][0], actions[i][1], 0]) * UAV_MAX_SPEED)

        # Physics connector updates vehicles and returns updated dict
        self.vehicles = self.physics.step(self.uavs, self.vehicles)

        # 3. Task Generation (Continuous)
        self._generate_tasks_for_all()

        # 4. Task Management (Assignment & Lifecycle)
        # Need list of vehicles for iteration
        vehicle_list = list(self.vehicles.values())
        self.task_manager.update_task_statuses(vehicle_list, self.time_step)
        self.task_manager.assign_tasks(vehicle_list, self.uavs, self.time_step)

        # 5. Energy & Rewards
        self._consume_hover_energy()
        rewards = self._calculate_rewards()

        # 6. Next State
        next_states = self.get_maddpg_states()
        self.time_step += 1
        done = self.time_step >= INNER_STEPS

        if done:
            self.physics.close()

        return next_states, rewards, done

    # --- Internal Helpers ---

    def _deploy_uavs_clustered(self):
        """Deploys UAVs to centroids of vehicle clusters."""
        positions = np.array([v.position[:2] for v in self.vehicles.values()])
        if len(positions) >= len(self.uavs):
            centroids, _ = kmeans(positions, len(self.uavs), iter=10)
            for i, uav in enumerate(self.uavs):
                c = centroids[i % len(centroids)]
                offset = (np.random.rand(2) * 2 - 1) * 100
                uav.position[0] = c[0] + offset[0]
                uav.position[1] = c[1] + offset[1]

    def _generate_tasks_for_all(self):
        """Generates tasks for vehicles based on DemandGenerator."""
        for v in self.vehicles.values():
            # Only generate if buffer not full
            if len([t for t in v.tasks if not t.is_completed]) >= TASKS_PER_VEHICLE:
                continue

            zone_id = self._get_zone_id(v.position)
            # Chance to generate task this step?
            # Original code generated bulk at start, but continuous is better.
            # For compatibility with original logic: check if empty or replenish
            if not v.tasks:  # Simple logic: replenish if empty
                for k in range(TASKS_PER_VEHICLE):
                    s_id, c_id = self.demand_generator.generate_next_request(self.time_step, v.id, zone_id)
                    new_task = Task(f"{v.id}-{self.time_step}-{k}", v.id, s_id, c_id)
                    v.add_task(new_task)
                    self.request_history.append((s_id, zone_id))

    def _get_zone_id(self, pos):
        row = int(pos[1] // (self.height / 2))
        col = int(pos[0] // (self.width / 2))
        # Map 2x2 grid to 0..3
        return min(row * 2 + col, NUM_ZONES - 1)

    def _consume_hover_energy(self):
        for uav in self.uavs:
            if uav.current_energy > 0:
                uav.consume_energy(ENERGY_HOVER_WATT)

    def _calculate_rewards(self):
        if not self.uavs: return []
        rewards = []
        for uav in self.uavs:
            r = uav.profit_this_step
            if USE_ENERGY_PENALTY:
                r -= (uav.energy_consumed_this_step * ENERGY_REWARD_PENALTY)
            rewards.append(r)
            uav.energy_consumed_this_step = 0.0

        # Shared reward
        global_r = np.mean(rewards) if rewards else 0
        return [global_r * REWARD_SCALING_FACTOR] * len(self.uavs)

    # --- State Getters ---

    def get_maddpg_states(self):
        if not self.uavs: return []
        states = []
        vehicle_list = list(self.vehicles.values())

        for uav in self.uavs:
            # Local density
            nearby = len([v for v in vehicle_list if self.comm.get_distance(uav, v) <= UAV_COMMUNICATION_RANGE])

            s = [
                uav.position[0] / self.width,
                uav.position[1] / self.height,
                nearby,
                uav.tasks_processed_count,
                uav.profit_generated / 1000.0,
                uav.current_energy / uav.max_energy if uav.max_energy > 0 else 0
            ]
            if USE_UAV_STATUS:
                s.append(1.0 if uav.status == 'BUSY' else 0.0)
            states.append(np.array(s))
        return states

    def get_ddqn_state(self):
        if not self.uavs: return np.zeros(DDQN_STATE_DIM)

        total_profit = sum(u.profit_generated for u in self.uavs)
        total_cost = sum(BETA_MAINTENANCE + BETA_COMPUTATION * u.F_total for u in self.uavs)
        net = total_profit - total_cost

        tasks_done = sum(u.tasks_processed_count for u in self.uavs)

        # Global Avg Latency
        completed = [t for v in self.vehicles.values() for t in v.tasks if t.is_completed]
        avg_lat = np.mean([t.completed_latency for t in completed]) if completed else 0

        # Coverage
        covered = set()
        vehicle_list = list(self.vehicles.values())
        for u in self.uavs:
            for v in vehicle_list:
                if self.comm.get_distance(u, v) <= UAV_COMMUNICATION_RANGE:
                    covered.add(v.id)

        return np.array([
            tasks_done,
            len(self.uavs),
            total_cost,
            net * REWARD_SCALING_FACTOR,
            avg_lat,
            len(covered)
        ])

    def get_recent_requests(self, n):
        return list(self.request_history)[-n:]

    def get_episode_statistics(self):
        completed = 0
        for v in self.vehicles.values():
            completed += sum(1 for t in v.tasks if t.is_completed)

        return {
            'Sim/active_vehicles': len(self.vehicles),
            'Tasks/completed': completed,
            'Offloading/local': self.task_manager.local_count,
            'Offloading/cloud': self.task_manager.cloud_count
        }

    def close(self):
        self.physics.close()
