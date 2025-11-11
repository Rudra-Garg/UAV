"""
Test script to verify benchmark agents handle edge cases correctly.
"""
import numpy as np
from benchmark_agents import MRUPOS_Agent
from entities import Vehicle, UAV


class MockEnv:
    """Mock environment for testing."""
    def __init__(self, num_vehicles, num_uavs):
        self.vehicles = {f"v_{i}": Vehicle(f"v_{i}") for i in range(num_vehicles)}
        # Set random positions for vehicles
        for v in self.vehicles.values():
            v.position = np.random.rand(3) * 1000

        self.uavs = [UAV(i) for i in range(num_uavs)]
        for uav in self.uavs:
            uav.position = np.random.rand(3) * 1000


def test_mrupos_agent():
    """Test MRUPOS_Agent with various scenarios."""
    print("Testing MRUPOS_Agent...")

    # Test 1: More vehicles than UAVs (normal case)
    print("\nTest 1: More vehicles (20) than UAVs (5)")
    env = MockEnv(num_vehicles=20, num_uavs=5)
    agent = MRUPOS_Agent(env)
    agent.num_uavs = 5
    actions = agent.select_actions(env, None)
    print(f"  Actions returned: {len(actions)} (expected: 5)")
    assert len(actions) == 5, "Should return 5 actions"
    print("  ✓ Test 1 passed")

    # Test 2: Fewer vehicles than UAVs (edge case)
    print("\nTest 2: Fewer vehicles (2) than UAVs (5)")
    env = MockEnv(num_vehicles=2, num_uavs=5)
    agent = MRUPOS_Agent(env)
    agent.num_uavs = 5
    actions = agent.select_actions(env, None)
    print(f"  Actions returned: {len(actions)} (expected: 5)")
    assert len(actions) == 5, "Should return 5 actions"
    print("  ✓ Test 2 passed")

    # Test 3: No vehicles (edge case)
    print("\nTest 3: No vehicles (0) with UAVs (5)")
    env = MockEnv(num_vehicles=0, num_uavs=5)
    agent = MRUPOS_Agent(env)
    agent.num_uavs = 5
    actions = agent.select_actions(env, None)
    print(f"  Actions returned: {len(actions)} (expected: 5)")
    assert len(actions) == 5, "Should return 5 actions"
    print("  ✓ Test 3 passed")

    # Test 4: Equal vehicles and UAVs
    print("\nTest 4: Equal vehicles (5) and UAVs (5)")
    env = MockEnv(num_vehicles=5, num_uavs=5)
    agent = MRUPOS_Agent(env)
    agent.num_uavs = 5
    actions = agent.select_actions(env, None)
    print(f"  Actions returned: {len(actions)} (expected: 5)")
    assert len(actions) == 5, "Should return 5 actions"
    print("  ✓ Test 4 passed")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_mrupos_agent()

