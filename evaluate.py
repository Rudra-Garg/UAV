# evaluate.py
"""
Main script for Phase 3: Evaluation and Benchmarking.

This script loads the trained MUCEDS agent, initializes benchmark agents,
runs them across various scenarios (different numbers of vehicles),
collects performance metrics, and plots the results for comparison.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from benchmark_agents import *
from config import *
from ddqn_agent import DDQNAgent
from environment import VECNEnvironment
from maddpg_agent import MADDPGController


def run_evaluation_episode(env, agent, agent_type='MUCEDS', num_vehicles=NUM_VEHICLES):
    """Runs a single episode for a given agent and returns the final profit."""
    num_uavs = 0

    # --- MODIFIED: Reset and setup logic moved here ---
    if agent_type == 'MUCEDS':
        ddqn_agent, maddpg_controllers = agent
        outer_state = env.get_ddqn_state()
        num_uavs = ddqn_agent.select_action(outer_state, evaluation=True) + 1
        maddpg_controller = maddpg_controllers.get(num_uavs)  # Use .get for safety
    else:  # Benchmark agents
        num_uavs = agent.num_uavs

    # Reset the environment with the correct number of UAVs AND vehicles for this run
    inner_states = env.reset(num_uavs=num_uavs, num_vehicles=num_vehicles)
    # --- End of MODIFIED section ---

    for _ in range(INNER_STEPS):
        if not env.uavs:  # Safety check if no UAVs were deployed
            break

        if agent_type == 'MUCEDS':
            if maddpg_controller:
                actions = maddpg_controller.select_actions(inner_states, evaluation=True)
            else:
                actions = [np.zeros(MADDPG_ACTION_DIM) for _ in range(num_uavs)]
        else:  # Benchmark agents
            actions = agent.select_actions(env, inner_states)

        next_inner_states, _, _ = env.step(actions)
        inner_states = next_inner_states

    final_state = env.get_ddqn_state()
    return final_state[3]


def main():
    print("--- Starting Phase 3: Evaluation and Benchmarking ---")

    env = VECNEnvironment()

    print("Loading trained MUCEDS agent...")
    ddqn_agent = DDQNAgent(state_dim=DDQN_STATE_DIM, action_space=DDQN_ACTION_SPACE)
    ddqn_agent.load(MODEL_SAVE_PATH)

    maddpg_controllers = {}
    for i in range(1, DDQN_ACTION_SPACE + 1):
        path = os.path.join(MODEL_SAVE_PATH, f"maddpg_{i}_agents")
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'maddpg_actor_0.pth')):
            controller = MADDPGController(num_agents=i, state_dim=MADDPG_STATE_DIM, action_dim=MADDPG_ACTION_DIM)
            controller.load(path)
            maddpg_controllers[i] = controller

    agents_to_evaluate = {
        "MUCEDS": (ddqn_agent, maddpg_controllers),
        "OUPRS": OUPRS_Agent(env),
        "OUPOS": OUPOS_Agent(env),
        "MRUPRS": MRUPRS_Agent(env),
        "MRUPOS": MRUPOS_Agent(env),
        "MOUPRS": MOUPRS_Agent(env, ddqn_agent),
        "MUPSOS": MUPSOS_Agent(env),
        "MKUPRS": MKUPRS_Agent(env)  # Added to match graph (red line)
    }

    results = {name: [] for name in agents_to_evaluate.keys()}
    vehicle_scenarios = list(EVAL_SCENARIO_VEHICLES)

    print("Running evaluation scenarios...")
    for num_vehicles in tqdm(vehicle_scenarios, desc="Vehicle Scenarios"):
        for agent_name, agent in agents_to_evaluate.items():
            episode_profits = []
            # --- MODIFIED: Pass num_vehicles to the run function ---
            for _ in range(EVAL_EPISODES):
                profit = run_evaluation_episode(env, agent, agent_type=agent_name, num_vehicles=num_vehicles)
                episode_profits.append(profit)

            avg_profit = np.mean(episode_profits)
            results[agent_name].append(avg_profit)

    print("Plotting results...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

    for idx, (agent_name, profits) in enumerate(results.items()):
        ax.plot(vehicle_scenarios, profits, marker='o', linestyle='--', label=agent_name, color=colors[idx])

    ax.set_title('Comparison of System Profit vs. Number of Users', fontsize=16)
    ax.set_xlabel('Number of Vehicle Users', fontsize=12)
    ax.set_ylabel('Average System Profit', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)

    plt.savefig('evaluation_profit_results.png')
    print("Results saved to 'evaluation_profit_results.png'")
    plt.show()


if __name__ == "__main__":
    main()
