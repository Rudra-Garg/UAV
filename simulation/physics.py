import logging
import random

import numpy as np

import config
from config import *
from simulation.entities import Vehicle

# Try to import traci - will be used if SIMULATION_MODE is 'SUMO'
traci = None
try:
    import traci as _traci
    import traci.exceptions
    traci = _traci
except ImportError:
    pass

logger = logging.getLogger(__name__)


def _wait_for_port(port, timeout=5):
    """Wait until the port is available."""
    import socket
    import time
    start_time = time.time()
    while time.time() - start_time < timeout:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        if result != 0:  # Port is free (connection refused)
            return True
        time.sleep(0.2)
    return False


def _get_simulation_mode():
    """Get the current simulation mode from config."""
    return config.SIMULATION_MODE


def _ensure_traci():
    """Ensure traci is available when needed."""
    global traci
    if traci is None:
        try:
            import traci as _traci
            import traci.exceptions
            traci = _traci
        except ImportError:
            raise ImportError("⚠️ SUMO (traci) not found. Install SUMO or switch SIMULATION_MODE to 'PYTHON_KINEMATIC'.")


class PhysicsConnector:
    def __init__(self, width, height, sumo_port=None):
        self.width = width
        self.height = height
        self.sumo_cmd = None
        self.sumo_port = sumo_port if sumo_port is not None else 8813
        self._sumo_ended = False  # Track if SUMO simulation has ended

        if _get_simulation_mode() == 'SUMO':
            _ensure_traci()
            self._setup_sumo_cmd()

    def _setup_sumo_cmd(self):
        sumo_binary = "sumo-gui" if VISUALIZATION else "sumo"
        self.sumo_cmd = [
            sumo_binary,
            "-c", "sumo_scenario/grid.sumocfg",
            "--step-length", "1",
            "--quit-on-end", "--start",
            "--no-warnings", "--no-step-log"
            # Note: --remote-port is set by traci.start(port=...) automatically
        ]

    def reset(self, num_vehicles):
        """Resets the physics engine and returns initial list of vehicles."""
        vehicles = []

        if _get_simulation_mode() == 'SUMO':
            _ensure_traci()
            self._setup_sumo_cmd()  # Ensure sumo_cmd is set up
            self._reset_sumo()
            # In SUMO, vehicles are spawned by the .rou.xml files loaded by TraCI
            # Initial step to load network and spawn initial vehicles
            traci.simulationStep()
            vehicles = self._sync_sumo_vehicles({})

        else:
            # Python Mode: Create vehicles manually
            vehicles = self._initialize_python_vehicles(num_vehicles)

        return vehicles

    def _reset_sumo(self):
        import time
        self._sumo_ended = False  # Reset flag on new simulation
        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass
        
        # Wait for port to be fully released
        if not _wait_for_port(self.sumo_port, timeout=3):
            logger.warning(f"Port {self.sumo_port} may still be in use, proceeding anyway...")

        # Select random city
        city = random.choice(SUMO_SCENARIO_POOL)
        config_path = os.path.join("sumo_scenario", f"{city}.sumocfg")

        # Fallback if specific city file missing
        if not os.path.exists(config_path):
            logger.error(f"CRITICAL: SUMO Config file not found: {config_path}")
            logger.error("Please run: python tools/generate_traffic.py")
            # Fallback to grid if specific city missing
            config_path = "sumo_scenario/grid.sumocfg"
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"No SUMO configuration files found in sumo_scenario/")

        self.sumo_cmd[2] = config_path
        logger.info(f"Starting SUMO: {city}")
        traci.start(self.sumo_cmd)

        # --- DYNAMICALLY INJECT UAV DEFINITIONS (SAFEGUARDED) ---

        # 1. Check/Create UAV_TYPE
        try:
            if "UAV_TYPE" not in traci.vehicletype.getIDList():
                # Try copying from a standard type if it exists, else create fresh
                existing_types = traci.vehicletype.getIDList()
                if "DEFAULT_VEHTYPE" in existing_types:
                    traci.vehicletype.copy("DEFAULT_VEHTYPE", "UAV_TYPE")
                elif len(existing_types) > 0:
                    traci.vehicletype.copy(existing_types[0], "UAV_TYPE")
                else:
                    # Fallback creation (rarely needed if .add.xml loaded)
                    # Note: add() requires complex params, copy is safer.
                    # If this fails, we assume .add.xml handled it.
                    pass

                # Apply UAV attributes
                traci.vehicletype.setColor("UAV_TYPE", (0, 255, 0, 255))
                traci.vehicletype.setLength("UAV_TYPE", 1.0)
                traci.vehicletype.setShapeClass("UAV_TYPE", "aircraft")
                traci.vehicletype.setMinGap("UAV_TYPE", 0)
                traci.vehicletype.setSpeedMode("UAV_TYPE", 0)  # Disable physics checks
        except traci.exceptions.TraCIException:
            # Type likely defined in .add.xml
            pass

        # 2. Check/Create dummy_route
        try:
            if "dummy_route" not in traci.route.getIDList():
                edge_list = traci.edge.getIDList()
                # Filter internal edges (starting with :)
                valid_edges = [e for e in edge_list if not e.startswith(":")]
                if valid_edges:
                    traci.route.add("dummy_route", [valid_edges[0]])
                else:
                    logger.error("No valid edges found in SUMO network to create UAV route!")
        except traci.exceptions.TraCIException:
            # Route likely defined in .add.xml
            pass

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

        if _get_simulation_mode() == 'SUMO':
            # Check if SUMO has already ended
            if self._sumo_ended:
                # Continue with existing vehicle positions (no movement)
                return vehicles
            
            # Check if simulation has ended (no more scheduled events)
            try:
                min_expected = traci.simulation.getMinExpectedNumber()
                if min_expected == 0:
                    logger.warning("SUMO simulation has no more vehicles/events. Freezing vehicle positions.")
                    self._sumo_ended = True
                    return vehicles
            except (traci.exceptions.TraCIException, traci.exceptions.FatalTraCIError):
                self._sumo_ended = True
                return vehicles
            
            # 1. Move UAVs (they should already exist from reset)
            for uav in uavs:
                uav_id = f"uav_{uav.id}"
                try:
                    traci.vehicle.moveToXY(uav_id, "", -1, uav.position[0], uav.position[1], keepRoute=2)
                except traci.exceptions.TraCIException as e:
                    # If it still doesn't exist (shouldn't happen), log and skip
                    logger.warning(f"UAV {uav_id} not found in SUMO, skipping: {e}")

            # 2. Step SUMO
            try:
                traci.simulationStep()
            except traci.exceptions.FatalTraCIError:
                logger.error("SUMO simulation crashed or closed unexpectedly.")
                self._sumo_ended = True
                return vehicles

            # 3. Sync Vehicles
            return self._sync_sumo_vehicles(vehicles)

        else:
            # Python Kinematic Mode
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
            try:
                pos = traci.vehicle.getPosition(vid)
                if vid not in vehicle_dict:
                    # New vehicle found in SUMO
                    new_v = Vehicle(vid)
                    new_v.position = np.array([pos[0], pos[1], 0])
                    # Note: Task generation is handled centrally in Environment, not here.
                    vehicle_dict[vid] = new_v
                else:
                    # Update position
                    vehicle_dict[vid].position = np.array([pos[0], pos[1], 0])
            except traci.exceptions.TraCIException:
                # Vehicle might have left in the exact millisecond between getIDList and getPosition
                continue

        return vehicle_dict

    def add_uavs_to_sumo(self, uavs):
        """Add UAVs to SUMO simulation after they're created."""
        if _get_simulation_mode() != 'SUMO':
            return

        for uav in uavs:
            uav_id = f"uav_{uav.id}"
            try:
                # Check if UAV already exists
                if uav_id not in traci.vehicle.getIDList():
                    traci.vehicle.add(uav_id, "dummy_route", typeID="UAV_TYPE")
                    traci.vehicle.setColor(uav_id, (0, 255, 0, 255))
                    # Move to initial position
                    traci.vehicle.moveToXY(uav_id, "", -1, uav.position[0], uav.position[1], keepRoute=2)
            except traci.exceptions.TraCIException as e:
                logger.error(f"Failed to add UAV {uav_id} during reset: {e}")

    def close(self):
        if _get_simulation_mode() == 'SUMO':
            try:
                if traci is not None and traci.isLoaded():
                    traci.close()
                # Longer delay to ensure port is fully released before next episode
                import time
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error closing SUMO: {e}")
                # Only force kill if we have a known port
                try:
                    import subprocess
                    import psutil
                    # Find and kill only SUMO processes using our specific port
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        try:
                            if 'sumo' in proc.info['name'].lower():
                                cmdline = proc.info['cmdline']
                                if cmdline and str(self.sumo_port) in ' '.join(cmdline):
                                    proc.kill()
                                    logger.info(f"Killed SUMO process on port {self.sumo_port}")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except ImportError:
                    # If psutil not available, use pkill with caution
                    import subprocess
                    subprocess.run(['pkill', '-9', 'sumo'], stderr=subprocess.DEVNULL)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup SUMO processes: {cleanup_error}")
