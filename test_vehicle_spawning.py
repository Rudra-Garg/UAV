#!/usr/bin/env python
"""
Test script to verify the new vehicle spawning system works correctly.
"""
import matplotlib.pyplot as plt
import numpy as np

from config import *
from environment import VECNEnvironment


def test_vehicle_spawning():
    """Test that vehicles spawn and maintain count throughout simulation."""
    print("=" * 60)
    print("TESTING VEHICLE SPAWNING SYSTEM")
    print("=" * 60)

    env = VECNEnvironment(visualize=False)
    target_vehicles = 100
    num_uavs = 5

    print(f"\nTarget vehicles: {target_vehicles}")
    print(f"Number of UAVs: {num_uavs}")
    print(f"Simulation steps: {INNER_STEPS}")

    # Reset environment
    print("\n--- Resetting environment ---")
    env.reset(num_uavs=num_uavs, num_vehicles=target_vehicles)

    initial_count = len(env.vehicles)
    print(f"Initial vehicle count: {initial_count}")

    # Track vehicle counts and coverage over time
    vehicle_counts = [initial_count]
    coverage_ratios = []

    # Calculate initial coverage
    covered = set()
    for uav in env.uavs:
        for v in env.vehicles.values():
            dist = np.linalg.norm(uav.position[:2] - v.position[:2])
            if dist <= UAV_COMMUNICATION_RANGE:
                covered.add(v.id)
    coverage_ratios.append(len(covered) / max(len(env.vehicles), 1))

    print(f"Initial coverage: {len(covered)}/{len(env.vehicles)} ({coverage_ratios[0]:.1%})")

    # Run simulation
    print("\n--- Running simulation ---")
    states = env.get_maddpg_states()

    for step in range(INNER_STEPS):
        # Random actions for UAVs
        actions = [(np.random.rand(2) * 2 - 1) for _ in range(num_uavs)]

        # Step the environment
        next_states, rewards, done = env.step(actions)
        states = next_states

        # Track metrics
        vehicle_counts.append(len(env.vehicles))

        # Calculate coverage
        covered = set()
        for uav in env.uavs:
            for v in env.vehicles.values():
                dist = np.linalg.norm(uav.position[:2] - v.position[:2])
                if dist <= UAV_COMMUNICATION_RANGE:
                    covered.add(v.id)
        coverage_ratios.append(len(covered) / max(len(env.vehicles), 1))

        # Print progress every 20 steps
        if (step + 1) % 20 == 0:
            print(f"Step {step + 1}: Vehicles={len(env.vehicles)}, Coverage={coverage_ratios[-1]:.1%}")

        if done:
            break

    env.close()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Target vehicles: {target_vehicles}")
    print(f"Initial vehicles: {initial_count}")
    print(f"Final vehicles: {vehicle_counts[-1]}")
    print(f"Average vehicles: {np.mean(vehicle_counts):.1f}")
    print(f"Min vehicles: {min(vehicle_counts)}")
    print(f"Max vehicles: {max(vehicle_counts)}")
    print(f"\nInitial coverage: {coverage_ratios[0]:.1%}")
    print(f"Final coverage: {coverage_ratios[-1]:.1%}")
    print(f"Average coverage: {np.mean(coverage_ratios):.1%}")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    steps = range(len(vehicle_counts))

    # Vehicle count over time
    ax1.plot(steps, vehicle_counts, 'b-', linewidth=2, label='Actual Vehicles')
    ax1.axhline(y=target_vehicles, color='r', linestyle='--', linewidth=2, label='Target')
    ax1.set_xlabel('Simulation Step', fontsize=12)
    ax1.set_ylabel('Number of Vehicles', fontsize=12)
    ax1.set_title('Vehicle Count Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Coverage over time
    ax2.plot(steps, [c * 100 for c in coverage_ratios], 'g-', linewidth=2)
    ax2.set_xlabel('Simulation Step', fontsize=12)
    ax2.set_ylabel('Coverage (%)', fontsize=12)
    ax2.set_title('UAV Coverage Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('vehicle_spawning_test.png', dpi=150)
    print("\n✅ Plot saved to 'vehicle_spawning_test.png'")
    plt.close()

    # Test passes if we maintain reasonable vehicle count
    avg_vehicles = np.mean(vehicle_counts)
    if avg_vehicles >= target_vehicles * 0.8:
        print("\n✅ TEST PASSED: Vehicle spawning system is working correctly!")
        return True
    else:
        print(f"\n❌ TEST FAILED: Average vehicle count ({avg_vehicles:.1f}) is too low!")
        return False


if __name__ == "__main__":
    success = test_vehicle_spawning()
    exit(0 if success else 1)
