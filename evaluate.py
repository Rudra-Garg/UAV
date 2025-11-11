# evaluate_multithreaded.py
"""
Optimized evaluation script with parallel execution for better performance with SUMO.
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
from tqdm import tqdm

from benchmark_agents import *
from config import *
from ddqn_agent import DDQNAgent
from environment import VECNEnvironment
from maddpg_agent import MADDPGController


def run_evaluation_episode(args):
    """
    Runs a single episode for a given agent. This function is designed to be
    called by worker processes, so it receives all necessary data as arguments.
    """
    agent_type, num_vehicles, episode_idx, model_save_path = args

    # Each worker creates its own environment and loads its own agents
    env = VECNEnvironment()

    # Load agents based on type
    if agent_type == 'MUCEDS':
        ddqn_agent = DDQNAgent(state_dim=DDQN_STATE_DIM, action_space=DDQN_ACTION_SPACE)
        ddqn_agent.load(model_save_path)

        maddpg_controllers = {}
        for i in range(1, DDQN_ACTION_SPACE + 1):
            path = os.path.join(model_save_path, f"maddpg_{i}_agents")
            if os.path.exists(path) and os.path.exists(os.path.join(path, 'maddpg_actor_0.pth')):
                controller = MADDPGController(num_agents=i, state_dim=MADDPG_STATE_DIM, action_dim=MADDPG_ACTION_DIM)
                controller.load(path)
                maddpg_controllers[i] = controller

        agent = (ddqn_agent, maddpg_controllers)
    elif agent_type == 'MOUPRS':
        ddqn_agent = DDQNAgent(state_dim=DDQN_STATE_DIM, action_space=DDQN_ACTION_SPACE)
        ddqn_agent.load(model_save_path)
        agent = MOUPRS_Agent(env, ddqn_agent)
    elif agent_type == 'OUPRS':
        agent = OUPRS_Agent(env)
    elif agent_type == 'OUPOS':
        agent = OUPOS_Agent(env)
    elif agent_type == 'MRUPRS':
        agent = MRUPRS_Agent(env)
    elif agent_type == 'MRUPOS':
        agent = MRUPOS_Agent(env)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Set seed for reproducibility
    np.random.seed(num_vehicles * 1000 + episode_idx)

    # Determine number of UAVs
    num_uavs = 0
    maddpg_controller = None

    if agent_type == 'MUCEDS':
        ddqn_agent, maddpg_controllers = agent
        outer_state = env.get_ddqn_state()
        num_uavs = ddqn_agent.select_action(outer_state, evaluation=True) + 1
        maddpg_controller = maddpg_controllers.get(num_uavs)
    elif agent_type == 'MOUPRS':
        outer_state = env.get_ddqn_state()
        num_uavs = agent.ddqn_agent.select_action(outer_state, evaluation=True) + 1
        agent.num_uavs = num_uavs
    else:
        num_uavs = agent.num_uavs

    # Run episode
    inner_states = env.reset(num_uavs=num_uavs, num_vehicles=num_vehicles)

    if not env.uavs:
        env.close()  # Important: close SUMO connection
        return {'profit': 0, 'tasks_completed': 0, 'avg_latency': INNER_STEPS}

    for _ in range(INNER_STEPS):
        if agent_type == 'MUCEDS':
            actions = maddpg_controller.select_actions(inner_states, evaluation=True) if maddpg_controller else [
                np.zeros(MADDPG_ACTION_DIM) for _ in range(num_uavs)]
        else:
            actions = agent.select_actions(env, inner_states)

        next_inner_states, _, done = env.step(actions)
        inner_states = next_inner_states
        if done:
            break

    final_state = env.get_ddqn_state()
    tasks_completed = final_state[0]
    avg_latency = final_state[4] if tasks_completed > 0 else INNER_STEPS

    result = {
        'profit': final_state[3],
        'tasks_completed': tasks_completed,
        'avg_latency': avg_latency
    }

    env.close()  # Important: close SUMO connection
    return result


def plot_comparison(metric_name, results, scenarios, ylabel, title, filename):
    """Helper function to generate and save a comparison plot for a given metric."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

    for idx, agent_name in enumerate(results.keys()):
        metric_values = [res[metric_name] for res in results[agent_name]]
        ax.plot(scenarios, metric_values, marker='o', linestyle='--', label=agent_name, color=colors[idx])

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Number of Vehicle Users', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    print(f"Results saved to '{filename}'")
    plt.close(fig)


def main():
    print("--- Starting Multithreaded Evaluation ---")

    # Create output directory
    os.makedirs('Evaluations_1/m1', exist_ok=True)

    # Define all agents to evaluate
    agent_types = ["MUCEDS", "OUPRS", "OUPOS", "MRUPRS", "MRUPOS", "MOUPRS"]
    vehicle_scenarios = list(EVAL_SCENARIO_VEHICLES)

    # Prepare all tasks
    tasks = []
    for num_vehicles in vehicle_scenarios:
        for agent_type in agent_types:
            for episode_idx in range(EVAL_EPISODES):
                tasks.append((agent_type, num_vehicles, episode_idx, MODEL_SAVE_PATH))

    print(f"Running {len(tasks)} evaluation episodes across multiple processes...")

    # Use ProcessPoolExecutor for parallel execution
    # Adjust max_workers based on your system (typically CPU count - 1)
    max_workers = min(os.cpu_count() - 1, 10)  # Limit to 8 to avoid overwhelming SUMO

    results_raw = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_evaluation_episode, task): task for task in tasks}

        with tqdm(total=len(tasks), desc="Evaluating") as pbar:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    results_raw.append((task[0], task[1], result))  # (agent_type, num_vehicles, metrics)
                except Exception as e:
                    print(f"\nError in task {task}: {e}")
                    results_raw.append(
                        (task[0], task[1], {'profit': 0, 'tasks_completed': 0, 'avg_latency': INNER_STEPS}))
                pbar.update(1)

    # Aggregate results
    print("\nAggregating results...")
    results = {agent_type: {num_v: [] for num_v in vehicle_scenarios} for agent_type in agent_types}

    for agent_type, num_vehicles, metrics in results_raw:
        results[agent_type][num_vehicles].append(metrics)

    # Calculate averages
    final_results = {agent_type: [] for agent_type in agent_types}

    for agent_type in agent_types:
        for num_vehicles in vehicle_scenarios:
            episode_results = results[agent_type][num_vehicles]
            avg_metrics = {
                'profit': np.mean([res['profit'] for res in episode_results]),
                'tasks_completed': np.mean([res['tasks_completed'] for res in episode_results]),
                'avg_latency': np.mean([res['avg_latency'] for res in episode_results]),
            }
            final_results[agent_type].append(avg_metrics)

    print("Plotting results...")

    plot_comparison(
        metric_name='profit',
        results=final_results,
        scenarios=vehicle_scenarios,
        ylabel='Average System Profit',
        title='Comparison of System Profit vs. Number of Users (Fig. 5)',
        filename='Evaluations/evaluation_profit_results.png'
    )

    plot_comparison(
        metric_name='tasks_completed',
        results=final_results,
        scenarios=vehicle_scenarios,
        ylabel='Average Number of Processed Tasks',
        title='Comparison of Processed Tasks vs. Number of Users (Fig. 6)',
        filename='Evaluations/evaluation_tasks_results.png'
    )

    plot_comparison(
        metric_name='avg_latency',
        results=final_results,
        scenarios=vehicle_scenarios,
        ylabel='Average Task Latency (steps)',
        title='Comparison of Task Latency vs. Number of Users (Fig. 7)',
        filename='Evaluations/evaluation_latency_results.png'
    )

    print("\n--- Evaluation Complete ---")


if __name__ == "__main__":
    main()
