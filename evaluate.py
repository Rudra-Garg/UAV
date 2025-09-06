# evaluate.py
"""
Main script for Phase 3: Evaluation and Benchmarking.

This script loads the trained MUCEDS agent, initializes benchmark agents,
runs them across various scenarios (different numbers of vehicles),
collects performance metrics, and plots the results for comparison.
"""
import os

import matplotlib.pyplot as plt
from tqdm import tqdm

from benchmark_agents import *
from config import *
from ddqn_agent import DDQNAgent
from environment import VECNEnvironment
from maddpg_agent import MADDPGController


def run_evaluation_episode(env, agent, agent_type='MUCEDS', num_vehicles=NUM_VEHICLES):
    """
    Runs a single episode for a given agent and returns a dictionary of key metrics.
    MODIFIED: Handles the special case of MOUPRS agent's dynamic UAV selection.
    """
    num_uavs = 0
    maddpg_controller = None # Only used by MUCEDS

    # --- MODIFIED SECTION: Determine number of UAVs ---
    if agent_type == 'MUCEDS':
        ddqn_agent, maddpg_controllers = agent
        outer_state = env.get_ddqn_state()
        num_uavs = ddqn_agent.select_action(outer_state, evaluation=True) + 1
        maddpg_controller = maddpg_controllers.get(num_uavs)
    else:  # Benchmark agents
        if agent_type == 'MOUPRS':
            # MOUPRS is a special benchmark that uses DDQN to select its UAV count.
            outer_state = env.get_ddqn_state()
            num_uavs = agent.ddqn_agent.select_action(outer_state, evaluation=True) + 1
            # Crucially, we also update the agent's internal state for its own logic.
            agent.num_uavs = num_uavs
        else:
            # All other benchmark agents have a fixed number of UAVs.
            num_uavs = agent.num_uavs
    # --- END OF MODIFIED SECTION ---

    # num_uavs is now guaranteed to be an integer.
    inner_states = env.reset(num_uavs=num_uavs, num_vehicles=num_vehicles)

    if not env.uavs:  # Safety check if no UAVs were deployed
        return {'profit': 0, 'tasks_completed': 0, 'avg_latency': INNER_STEPS}

    for _ in range(INNER_STEPS):
        if agent_type == 'MUCEDS':
            actions = maddpg_controller.select_actions(inner_states, evaluation=True) if maddpg_controller else [np.zeros(MADDPG_ACTION_DIM) for _ in range(num_uavs)]
        else:
            # This 'else' correctly covers all benchmark agents now, including MOUPRS.
            actions = agent.select_actions(env, inner_states)

        next_inner_states, _, done = env.step(actions)
        inner_states = next_inner_states
        if done:
            break

    final_state = env.get_ddqn_state()
    tasks_completed = final_state[0]
    avg_latency = final_state[4] if tasks_completed > 0 else INNER_STEPS

    return {
        'profit': final_state[3],
        'tasks_completed': tasks_completed,
        'avg_latency': avg_latency
    }

def plot_comparison(metric_name, results, scenarios, ylabel, title, filename):
    """Helper function to generate and save a comparison plot for a given metric."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

    for idx, agent_name in enumerate(results.keys()):
        # Extract the specific metric's data for the agent
        metric_values = [res[metric_name] for res in results[agent_name]]
        ax.plot(scenarios, metric_values, marker='o', linestyle='--', label=agent_name, color=colors[idx])

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Number of Vehicle Users', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)

    plt.savefig(filename)
    print(f"Results saved to '{filename}'")
    plt.show()


def main():
    print("--- Starting Enhanced Evaluation ---")

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
    }

    # MODIFIED: Results structure to hold multiple metrics
    results = {name: [] for name in agents_to_evaluate.keys()}
    vehicle_scenarios = list(EVAL_SCENARIO_VEHICLES)

    print("Running evaluation scenarios...")
    for num_vehicles in tqdm(vehicle_scenarios, desc="Vehicle Scenarios"):
        for agent_name, agent in agents_to_evaluate.items():

            # Store list of metric dicts for each episode
            episode_results = []
            for _ in range(EVAL_EPISODES):
                metrics = run_evaluation_episode(env, agent, agent_type=agent_name, num_vehicles=num_vehicles)
                episode_results.append(metrics)

            # Average the metrics over all episodes for this data point
            avg_metrics = {
                'profit': np.mean([res['profit'] for res in episode_results]),
                'tasks_completed': np.mean([res['tasks_completed'] for res in episode_results]),
                'avg_latency': np.mean([res['avg_latency'] for res in episode_results]),
            }
            results[agent_name].append(avg_metrics)

    print("Plotting results...")

    # --- NEW: Generate a separate plot for each key metric ---
    plot_comparison(
        metric_name='profit',
        results=results,
        scenarios=vehicle_scenarios,
        ylabel='Average System Profit',
        title='Comparison of System Profit vs. Number of Users (Fig. 5)',
        filename='evaluation_profit_results.png'
    )

    plot_comparison(
        metric_name='tasks_completed',
        results=results,
        scenarios=vehicle_scenarios,
        ylabel='Average Number of Processed Tasks',
        title='Comparison of Processed Tasks vs. Number of Users (Fig. 6)',
        filename='evaluation_tasks_results.png'
    )

    plot_comparison(
        metric_name='avg_latency',
        results=results,
        scenarios=vehicle_scenarios,
        ylabel='Average Task Latency (steps)',
        title='Comparison of Task Latency vs. Number of Users (Fig. 7)',
        filename='evaluation_latency_results.png'
    )


if __name__ == "__main__":
    main()
