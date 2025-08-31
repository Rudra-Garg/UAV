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

from benchmark_agents import OUPRS_Agent, OUPOS_Agent, MRUPRS_Agent
from config import *
from ddqn_agent import DDQNAgent
from environment import VECNEnvironment
from maddpg_agent import MADDPGController


def run_evaluation_episode(env, agent, agent_type='MUCEDS'):
    """Runs a single episode for a given agent and returns the final profit."""
    if agent_type == 'MUCEDS':
        # MUCEDS (our trained agent) uses the HRL structure
        ddqn_agent, maddpg_controllers = agent

        # 1. DDQN selects the number of UAVs
        outer_state = env.get_ddqn_state()
        num_uavs = ddqn_agent.select_action(outer_state, evaluation=True) + 1
        env.reset(num_uavs=num_uavs, num_vehicles=len(env.vehicles))

        # Load the corresponding MADDPG controller if it exists
        if num_uavs in maddpg_controllers:
            maddpg_controller = maddpg_controllers[num_uavs]
        else:  # Fallback if a specific model wasn't saved
            maddpg_controller = None

    else:
        # Benchmark agents have a fixed number of UAVs
        num_uavs = agent.num_uavs

    # Run the inner loop for one episode
    inner_states = env.get_maddpg_states()
    for _ in range(INNER_STEPS):
        if agent_type == 'MUCEDS':
            if maddpg_controller:
                actions = maddpg_controller.select_actions(inner_states, evaluation=True)
            else:  # If no controller, do nothing
                actions = [np.zeros(MADDPG_ACTION_DIM) for _ in range(num_uavs)]
        else:  # Benchmark agents
            actions = agent.select_actions(inner_states)

        next_inner_states, _, _ = env.step(actions)
        inner_states = next_inner_states

    # Return the final profit from the episode
    final_state = env.get_ddqn_state()
    return final_state[3]  # Index 3 is total_profit


def main():
    print("--- Starting Phase 3: Evaluation and Benchmarking ---")

    env = VECNEnvironment()

    # --- 1. Load the trained MUCEDS agent ---
    print("Loading trained MUCEDS agent...")
    ddqn_agent = DDQNAgent(state_dim=DDQN_STATE_DIM, action_space=DDQN_ACTION_SPACE)
    ddqn_agent.load(MODEL_SAVE_PATH)

    # Load all available MADDPG controllers
    maddpg_controllers = {}
    for i in range(1, DDQN_ACTION_SPACE + 1):
        path = os.path.join(MODEL_SAVE_PATH, f"maddpg_{i}_agents")
        if os.path.exists(path):
            controller = MADDPGController(num_agents=i, state_dim=MADDPG_STATE_DIM, action_dim=MADDPG_ACTION_DIM)
            controller.load(path)
            maddpg_controllers[i] = controller

    agents_to_evaluate = {
        "MUCEDS": (ddqn_agent, maddpg_controllers),
        "OUPRS": OUPRS_Agent(env),
        "OUPOS": OUPOS_Agent(env),
        "MRUPRS": MRUPRS_Agent(env)
    }

    results = {name: [] for name in agents_to_evaluate.keys()}
    vehicle_scenarios = list(EVAL_SCENARIO_VEHICLES)

    # --- 2. Run evaluation across all scenarios ---
    print("Running evaluation scenarios...")
    for num_vehicles in tqdm(vehicle_scenarios, desc="Vehicle Scenarios"):
        # Reset env with the current number of vehicles for all agents to face
        env.reset(num_vehicles=num_vehicles)

        for agent_name, agent in agents_to_evaluate.items():
            episode_profits = []
            for _ in range(EVAL_EPISODES):
                profit = run_evaluation_episode(env, agent, agent_type=agent_name)
                episode_profits.append(profit)

            avg_profit = np.mean(episode_profits)
            results[agent_name].append(avg_profit)

    # --- 3. Plot the results ---
    print("Plotting results...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    for agent_name, profits in results.items():
        ax.plot(vehicle_scenarios, profits, marker='o', linestyle='--', label=agent_name)

    ax.set_title('Comparison of System Profit vs. Number of Users', fontsize=16)
    ax.set_xlabel('Number of Vehicle Users', fontsize=12)
    ax.set_ylabel('Average System Profit', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)

    # Save the figure and show it
    plt.savefig('evaluation_profit_results.png')
    print("Results saved to 'evaluation_profit_results.png'")
    plt.show()


if __name__ == "__main__":
    main()
