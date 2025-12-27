# test_sumo_scenarios.py
import os
import sys
import time

import numpy as np

# Ensure we can import from the project
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *

try:
    import traci
    import traci.exceptions
except ImportError:
    print("CRITICAL: 'traci' library not found. Ensure SUMO_HOME is set and tools are in PATH.")
    sys.exit(1)

# List of scenarios to test
SCENARIOS_TO_TEST = [
    'delhi', 'mumbai', 'guwahati', 'bangaluru',
    'paris', 'london', 'nyc', 'tokyo'
]


def test_scenario(city_name, use_gui=True):
    print(f"\n{'=' * 50}")
    print(f"TESTING SCENARIO: {city_name}")
    print(f"{'=' * 50}")

    config_path = os.path.join("sumo_scenario", f"{city_name}.sumocfg")

    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print("   -> Did you run 'python tools/generate_traffic.py'?")
        return False

    sumo_binary = "sumo-gui" if use_gui else "sumo"

    cmd = [
        sumo_binary,
        "-c", config_path,
        "--step-length", "1",
        "--no-step-log", "true",
        "--start", "true",
        "--quit-on-end", "true"
    ]

    try:
        print(f"1. Starting SUMO ({sumo_binary})...")
        traci.start(cmd)

        # --- Check 1: Route Definitions ---
        print("2. Verifying Definitions...")
        routes = traci.route.getIDList()
        if "dummy_route" not in routes:
            print("   ❌ ERROR: 'dummy_route' is missing!")
            print("      This route is required for UAVs. It should be in the .add.xml file.")
            traci.close()
            return False
        else:
            print("   ✅ 'dummy_route' found.")

        # --- Check 2: Vehicle Types ---
        vtypes = traci.vehicletype.getIDList()
        if "UAV_TYPE" not in vtypes:
            print("   ❌ ERROR: 'UAV_TYPE' is missing!")
            print("      The UAV vehicle type definition is missing from .add.xml.")
            traci.close()
            return False
        else:
            print("   ✅ 'UAV_TYPE' found.")

        # --- Check 3: Spawning a UAV ---
        print("3. Attempting to spawn a Test UAV...")
        uav_id = "test_uav_001"
        try:
            traci.vehicle.add(uav_id, "dummy_route", typeID="UAV_TYPE")
            traci.vehicle.setColor(uav_id, (0, 255, 0, 255))  # Green
            traci.vehicle.setShapeClass(uav_id, "aircraft")
            # Move to a generic coordinate (e.g., 500, 500) or center of network
            traci.vehicle.moveToXY(uav_id, "", -1, 500, 500, keepRoute=2)
            print("   ✅ UAV added command accepted.")
        except traci.exceptions.TraCIException as e:
            print(f"   ❌ FAILED to add UAV: {e}")
            traci.close()
            return False

        # --- Check 4: Simulation Step ---
        print("4. Running 50 Simulation Steps...")
        for step in range(50):
            traci.simulationStep()

            # Check if UAV still exists
            id_list = traci.vehicle.getIDList()
            if uav_id in id_list:
                # Move it in a circle to verify physics control
                x = 500 + 100 * np.cos(step * 0.1)
                y = 500 + 100 * np.sin(step * 0.1)
                traci.vehicle.moveToXY(uav_id, "", -1, x, y, keepRoute=2)
            else:
                print(f"   ❌ ERROR: UAV disappeared at step {step}!")
                print("      It might have crashed, finished its route instantly, or been filtered out.")
                traci.close()
                return False

            if use_gui:
                time.sleep(0.05)  # Slow down slightly to see it

        print(f"   ✅ Scenario {city_name} PASSED.")
        traci.close()
        return True

    except Exception as e:
        print(f"   ❌ CRITICAL EXCEPTION: {e}")
        try:
            traci.close()
        except:
            pass
        return False


if __name__ == "__main__":
    print("--- SUMO SCENARIO DIAGNOSTIC TOOL ---")
    print("This script will open SUMO-GUI for each city to verify files and UAV spawning.")

    results = {}

    for city in SCENARIOS_TO_TEST:
        # Pass False to verify quickly without opening windows, or True to debug visually
        success = test_scenario(city, use_gui=True)
        results[city] = "PASS" if success else "FAIL"
        time.sleep(1)

    print("\n" + "=" * 30)
    print("FINAL RESULTS")
    print("=" * 30)
    for city, status in results.items():
        print(f"{city:<15} : {status}")
