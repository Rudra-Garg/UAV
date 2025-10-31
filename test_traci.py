#!/usr/bin/env python
"""
Test script to verify SUMO integration works correctly.
Run this before running the main training script.
"""
import os
import sys

import traci


def test_sumo_connection():
    """Test basic SUMO connection and vehicle spawning."""
    print("Testing SUMO Integration...")

    # 1. Check SUMO_HOME environment variable
    if 'SUMO_HOME' not in os.environ:
        sumo_home = r'C:\Program Files (x86)\Eclipse\Sumo'
        if os.path.exists(sumo_home):
            os.environ['SUMO_HOME'] = sumo_home
            print(f"✓ Set SUMO_HOME to: {sumo_home}")
        else:
            print("✗ SUMO not found! Please install SUMO or update the path.")
            return False
    else:
        print(f"✓ SUMO_HOME found: {os.environ['SUMO_HOME']}")

    # 2. Check if SUMO binary exists
    sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo.exe')
    if not os.path.exists(sumo_binary):
        print(f"✗ SUMO binary not found at: {sumo_binary}")
        return False
    print(f"✓ SUMO binary found")

    # 3. Check if scenario files exist
    config_file = "sumo_scenario/grid.sumocfg"
    if not os.path.exists(config_file):
        print(f"✗ SUMO config file not found: {config_file}")
        return False
    print(f"✓ SUMO config file found")

    # 4. Try to start SUMO
    try:
        sumo_cmd = [
            sumo_binary,
            "-c", config_file,
            "--step-length", "1",
            "--quit-on-end",
            "--no-warnings"
        ]
        print("\n✓ Starting SUMO simulation...")
        traci.start(sumo_cmd)

        # 5. Run a few simulation steps
        print("✓ Running 10 simulation steps...")
        for step in range(10):
            traci.simulationStep()
            vehicle_ids = traci.vehicle.getIDList()
            print(f"  Step {step + 1}: {len(vehicle_ids)} vehicles in simulation")

        # 6. Close cleanly
        traci.close()
        print("\n✓ SUMO integration test PASSED!")
        return True

    except Exception as e:
        print(f"\n✗ SUMO integration test FAILED: {e}")
        try:
            traci.close()
        except:
            pass
        return False


if __name__ == "__main__":
    success = test_sumo_connection()
    sys.exit(0 if success else 1)
