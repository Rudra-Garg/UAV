# diagnose_coverage_v2.py
"""
Updated diagnostic script with the new vehicle spawning system.
"""

import matplotlib.pyplot as plt
import numpy as np

from config import *
from environment import VECNEnvironment


def analyze_scenario(city_name, num_vehicles=100, num_uavs_to_test=None):
    """Analyze coverage for a specific SUMO scenario with proper vehicle spawning."""
    if num_uavs_to_test is None:
        num_uavs_to_test = [1, 5, 10, 15, 20]

    print(f"\n{'=' * 60}")
    print(f"Analyzing: {city_name}")
    print(f"Target vehicles: {num_vehicles}")
    print(f"{'=' * 60}")

    results = {}

    for num_uavs in num_uavs_to_test:
        print(f"\nTesting with {num_uavs} UAVs...")

        env = VECNEnvironment(visualize=False)
        env.reset(num_uavs=num_uavs, num_vehicles=num_vehicles)

        if not env.vehicles:
            print("  ⚠️ No vehicles spawned!")
            env.close()
            continue

        # Get vehicle positions
        vehicle_positions = np.array([v.position[:2] for v in env.vehicles.values()])
        actual_num_vehicles = len(vehicle_positions)

        # Calculate spread
        x_range = vehicle_positions[:, 0].max() - vehicle_positions[:, 0].min()
        y_range = vehicle_positions[:, 1].max() - vehicle_positions[:, 1].min()
        area_km2 = (x_range * y_range) / 1e6 if x_range > 0 and y_range > 0 else 0.001
        density = actual_num_vehicles / area_km2

        # Calculate coverage
        covered = set()
        uav_loads = []
        for uav in env.uavs:
            count = 0
            for v in env.vehicles.values():
                dist = np.linalg.norm(uav.position[:2] - v.position[:2])
                if dist <= UAV_COMMUNICATION_RANGE:
                    covered.add(v.id)
                    count += 1
            uav_loads.append(count)

        coverage_ratio = len(covered) / actual_num_vehicles if actual_num_vehicles > 0 else 0

        # Calculate tasks
        total_tasks = sum(len(v.tasks) for v in env.vehicles.values())
        covered_vehicle_tasks = sum(
            len(v.tasks) for v in env.vehicles.values()
            if v.id in covered
        )
        tasks_coverage = covered_vehicle_tasks / total_tasks if total_tasks > 0 else 0

        print(f"  Vehicles: {actual_num_vehicles}/{num_vehicles}")
        print(f"  Spread: {x_range / 1000:.1f}km x {y_range / 1000:.1f}km = {area_km2:.1f}km²")
        print(f"  Density: {density:.1f} vehicles/km²")
        print(f"  Coverage: {len(covered)}/{actual_num_vehicles} ({coverage_ratio:.1%})")
        print(f"  Tasks in range: {covered_vehicle_tasks}/{total_tasks} ({tasks_coverage:.1%})")
        print(f"  UAV loads: avg={np.mean(uav_loads):.1f}, min={min(uav_loads)}, max={max(uav_loads)}")

        results[num_uavs] = {
            'coverage': coverage_ratio,
            'vehicles': actual_num_vehicles,
            'area': area_km2,
            'density': density,
            'avg_load': np.mean(uav_loads),
            'tasks_in_range': tasks_coverage
        }

        env.close()

    return results


def plot_results(all_results):
    """Plot comparison across scenarios."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    for city, results in all_results.items():
        uav_counts = sorted(results.keys())
        coverages = [results[n]['coverage'] for n in uav_counts]
        avg_loads = [results[n]['avg_load'] for n in uav_counts]
        tasks_in_range = [results[n]['tasks_in_range'] for n in uav_counts]

        axes[0, 0].plot(uav_counts, coverages, 'o-', label=city, linewidth=2)
        axes[0, 1].plot(uav_counts, avg_loads, 'o-', label=city, linewidth=2)
        axes[1, 0].plot(uav_counts, tasks_in_range, 'o-', label=city, linewidth=2)

    axes[0, 0].set_title('Coverage vs UAVs', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Number of UAVs')
    axes[0, 0].set_ylabel('Coverage Ratio')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.1])

    axes[0, 1].set_title('Avg Vehicles per UAV', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Number of UAVs')
    axes[0, 1].set_ylabel('Avg Load')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('Tasks in UAV Range', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Number of UAVs')
    axes[1, 0].set_ylabel('Task Coverage Ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.1])

    # Summary table
    axes[1, 1].axis('off')
    summary_text = "City Characteristics\n" + "=" * 40 + "\n"
    for city, results in all_results.items():
        r = results[1]  # Use 1 UAV case for reference
        summary_text += f"\n{city}:\n"
        summary_text += f"  Vehicles: {r['vehicles']}\n"
        summary_text += f"  Area: {r['area']:.1f} km²\n"
        summary_text += f"  Density: {r['density']:.1f} veh/km²\n"

    axes[1, 1].text(0.1, 0.9, summary_text,
                    transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top',
                    fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('coverage_analysis_v2.png', dpi=150)
    print("\n✅ Analysis plot saved to 'coverage_analysis_v2.png'")
    plt.close()


def main():
    print("=" * 60)
    print("SUMO COVERAGE DIAGNOSTIC (V2 - WITH SPAWNING)")
    print("=" * 60)
    print(f"\nCurrent settings:")
    print(f"  UAV Communication Range: {UAV_COMMUNICATION_RANGE}m")
    print(f"  UAV Altitude: {UAV_ALTITUDE}m")
    print(f"  Tasks per vehicle: {TASKS_PER_VEHICLE}")
    print(f"  Target vehicles per scenario: 100")

    # Test a subset of cities
    cities_to_test = ['delhi', 'mumbai', 'paris']

    all_results = {}
    for city in cities_to_test:
        if city in SUMO_SCENARIO_POOL:
            results = analyze_scenario(
                city,
                num_vehicles=100,
                num_uavs_to_test=[1, 5, 10, 15, 20, 25]
            )
            all_results[city] = results

    if all_results:
        plot_results(all_results)

        print("\n" + "=" * 60)
        print("RECOMMENDATIONS:")
        print("=" * 60)

        # Analyze results and give recommendations
        avg_coverage_1uav = np.mean([r[1]['coverage'] for r in all_results.values()])
        avg_coverage_10uav = np.mean([r[10]['coverage'] for r in all_results.values()])

        if avg_coverage_1uav < 0.3:
            print("⚠️  CRITICAL: Single UAV covers < 30% of vehicles")
            print("   → INCREASE UAV_COMMUNICATION_RANGE to 2000m or more")
            print("   → INCREASE UAV_ALTITUDE to 100m")
        elif avg_coverage_1uav >= 0.3:
            print(f"✅ Good: Single UAV covers {avg_coverage_1uav:.1%} of vehicles")

        if avg_coverage_10uav < 0.7:
            print("⚠️  WARNING: 10 UAVs only cover < 70% of vehicles")
            print("   → Consider INCREASING UAV_COMMUNICATION_RANGE")
        else:
            print(f"✅ Good: 10 UAVs cover {avg_coverage_10uav:.1%} of vehicles")

        avg_load_1uav = np.mean([r[1]['avg_load'] for r in all_results.values()])
        if avg_load_1uav < 20:
            print(f"⚠️  LOW LOAD: Single UAV only serves ~{avg_load_1uav:.0f} vehicles")
            print("   → Multi-UAV deployment is NECESSARY for this scenario")
        else:
            print(f"✅ Good: Single UAV serves {avg_load_1uav:.0f} vehicles on average")


if __name__ == "__main__":
    main()
