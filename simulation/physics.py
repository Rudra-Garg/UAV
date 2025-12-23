import logging
import random

import numpy as np

from config import *
from entities import Vehicle

# Conditional import for SUMO
if SIMULATION_MODE == 'SUMO':
    try:
        import traci
        import traci.exceptions
    except ImportError:
        print("⚠️ SUMO (traci) not found. Switch SIMULATION_MODE to 'PYTHON_KINEMATIC' if you don't have SUMO.")

logger = logging.getLogger(__name__)


class PhysicsConnector:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.sumo_cmd = None

        if SIMULATION_MODE == 'SUMO':
            self._setup_sumo_cmd()

    def _setup_sumo_cmd(self):
        sumo_binary = "sumo-gui" if VISUALIZATION else "sumo"
        self.sumo_cmd = [
            sumo_binary,
            "-c", "sumo_scenario/grid.sumocfg",  # Placeholder, updated in reset
            "--step-length", "1",
            "--quit-on-end", "--start",
            "--no-warnings", "--no-step-log"
        ]

    def reset(self, num_vehicles):
        """Resets the physics engine and returns initial list of vehicles."""
        vehicles = []

        if SIMULATION_MODE == 'SUMO':
            self._reset_sumo()
            # In SUMO, vehicles are spawned by the .rou.xml files loaded by TraCI
            # We just need to sync the python objects to what SUMO creates.
            # Initial step to load network
            traci.simulationStep()
            vehicles = self._sync_sumo_vehicles({})

        else:
            # Python Mode: Create vehicles manually
            vehicles = self._initialize_python_vehicles(num_vehicles)

        return vehicles

    def _reset_sumo(self):
        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass

        # Select random city
        city = random.choice(SUMO_SCENARIO_POOL)
        config_path = os.path.join("sumo_scenario", f"{city}.sumocfg")

        # Fallback if specific city file missing
        if not os.path.exists(config_path):
            logger.warning(f"Config {config_path} not found. Using default grid.")
            config_path = "sumo_scenario/grid.sumocfg"

        self.sumo_cmd[2] = config_path
        logger.info(f"Starting SUMO: {city}")
        traci.start(self.sumo_cmd)

    def _initialize_python_vehicles(self, num_vehicles):
        # Choose scenario logic for initial distribution
        scenario = np.random.choice(
            TRAFFIC_SCENARIOS['SCENARIO_NAMES'],
            p=TRAFFIC_SCENARIOS['SCENARIO_WEIGHTS']
        )
        params = TRAFFIC_SCENARIOS['SCENARIOS'][scenario]

        vehicles = []
        # Hotspot generation logic
        num_hotspots = params['num_hotspots']
        hotspots = [np.random.rand(2) * [self.width, self.height] for _ in range(num_hotspots)]

        for i in range(num_vehicles):
            v = Vehicle(i)
            # If we are in a hotspot scenario, bias positions
            if num_hotspots > 0 and i < num_vehicles * params['hotspot_ratio']:
                center = hotspots[i % num_hotspots]
                offset = (np.random.rand(2) * 2 - 1) * params['hotspot_radius']
                pos = np.clip(center + offset, 0, [self.width, self.height])
                v.position = np.append(pos, 0)
            vehicles.append(v)

        return vehicles

    def step(self, uavs, vehicles):
        """Advances physics by one step."""

        if SIMULATION_MODE == 'SUMO':
            # 1. Move UAVs (Visual only in SUMO)
            for uav in uavs:
                uav_id = f"uav_{uav.id}"
                # Ensure UAV exists in SUMO for visualization
                try:
                    traci.vehicle.moveToXY(uav_id, "", -1, uav.position[0], uav.position[1], keepRoute=2)
                except traci.exceptions.TraCIException:
                    # Add UAV if missing
                    try:
                        traci.vehicle.add(uav_id, "dummy_route", typeID="UAV_TYPE")
                        traci.vehicle.setColor(uav_id, (0, 255, 0, 255))
                    except:
                        pass

            # 2. Step SUMO
            traci.simulationStep()

            # 3. Sync Vehicles (Remove departed, add new, update positions)
            return self._sync_sumo_vehicles(vehicles)

        else:
            # Python Kinematic Mode
            # 1. Move UAVs (State update is handled by UAV class, just need to ensure bounds)
            # 2. Move Vehicles
            for v in vehicles.values():
                v.move()
            return vehicles

    def _sync_sumo_vehicles(self, vehicle_dict):
        """
        Synchronizes the Python vehicle dictionary with SUMO's state.
        Returns the updated dictionary {id: VehicleObj}.
        """
        # Get all current vehicle IDs from SUMO
        sumo_ids = set(traci.vehicle.getIDList())

        # Filter out UAVs (which we inject manually)
        vehicle_ids = {vid for vid in sumo_ids if not vid.startswith("uav_")}

        # 1. Remove vehicles that left simulation
        current_ids = set(vehicle_dict.keys())
        left_ids = current_ids - vehicle_ids
        for vid in left_ids:
            del vehicle_dict[vid]

        # 2. Update existing & Add new
        for vid in vehicle_ids:
            pos = traci.vehicle.getPosition(vid)

            if vid not in vehicle_dict:
                # New vehicle found in SUMO
                new_v = Vehicle(vid)
                new_v.position = np.array([pos[0], pos[1], 0])
                # Generate tasks immediately for new vehicles
                new_v.generate_tasks()
                vehicle_dict[vid] = new_v
            else:
                # Update position
                vehicle_dict[vid].position = np.array([pos[0], pos[1], 0])

        return vehicle_dict

    def close(self):
        if SIMULATION_MODE == 'SUMO':
            try:
                traci.close()
            except:
                pass
