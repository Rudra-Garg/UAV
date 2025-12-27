"""
Optimized evaluation script for the Python-only version with parallel execution.
Now accepts command-line arguments for custom model paths.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
from tqdm import tqdm

# Adjust path to import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.benchmark_agents import *
from config import *
from agents.ddqn import DDQNAgent
from simulation.environment import VECNEnvironment
from agents.maddpg import MADDPGController


def run_evaluation_episode_worker(args):
    """
    Worker function to run a single episode. Designed for parallel execution.
    """
    agent_type, num_vehicles, episode_seed, model_save_path = args

    # Each worker process must create its own environment instance.
    env = VECNEnvironment()

    # Seed for reproducibility within the worker.
    np.random.seed(episode_seed)

    # 1. Load the Outer Agent (DDQN)
    ddqn_agent_for_benchmarks = DDQNAgent(state_dim=DDQN_STATE_DIM, action_space=DDQN_ACTION_SPACE)
    # Check if we are loading a specific timestamped run
    try:
        ddqn_agent_for_benchmarks.load(model_save_path)
    except FileNotFoundError:
        # Fallback if the path is wrong, but better to fail explicitly
        print(f"Error: Could not load DDQN model from {model_save_path}")
        return None

    # 2. Setup the Agent logic based on type
    if agent_type == 'MUCEDS':
        maddpg_controllers = {}
        # We try to load controllers for various agent counts
        # In a real scenario, you likely only saved the specific K used in the last episode
        # or saved a bank of them. This loop checks for what exists.
        for i in range(1, DDQN_ACTION_SPACE + 1):
            path = os.path.join(model_save_path, f"maddpg_{i}_agents")
            if os.path.exists(path) and os.path.exists(os.path.join(path, 'maddpg_actor_0.pth')):
                controller = MADDPGController(num_agents=i, state_dim=MADDPG_STATE_DIM, action_dim=MADDPG_ACTION_DIM)
                controller.load(path)
                maddpg_controllers[i] = controller
        agent = (ddqn_agent_for_benchmarks, maddpg_controllers)

    elif agent_type == 'MOUPRS':
        agent = MOUPRS_Agent(env, ddqn_agent_for_benchmarks)
    else:
        # Dynamically create other benchmark agents
        agent_class = globals()[f"{agent_type}_Agent"]
        agent = agent_class(env)

    # 3. Run the Episode
    num_uavs = 0
    maddpg_controller = None

    if agent_type == 'MUCEDS':
        ddqn_agent, maddpg_controllers = agent
        outer_state = env.get_ddqn_state()
        num_uavs = ddqn_agent.select_action(outer_state, evaluation=True) + 1
        maddpg_controller = maddpg_controllers.get(num_uavs)
        # If we selected a K for which we don't have a trained controller (common in partial saves)
        # We might need a fallback or return 0 performance.
        if num_uavs not in maddpg_controllers:
            # Fallback: Just dont assign controller, actions will be zero/random
            maddpg_controller = None

    elif agent_type == 'MOUPRS':
        outer_state = env.get_ddqn_state()
        num_uavs = agent.ddqn_agent.select_action(outer_state, evaluation=True) + 1
        agent.num_uavs = num_uavs
    else:
        num_uavs = agent.num_uavs

    inner_states = env.reset(num_uavs=num_uavs, num_vehicles=num_vehicles)

    if not env.uavs:
        return {'profit': 0, 'tasks_completed': 0, 'avg_latency': INNER_STEPS}

    for _ in range(INNER_STEPS):
        if agent_type == 'MUCEDS':
            if maddpg_controller:
                actions = maddpg_controller.select_actions(inner_states, evaluation=True)
            else:
                # Hover if no controller found
                actions = [np.zeros(MADDPG_ACTION_DIM) for _ in range(num_uavs)]
        else:
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
    """Helper function to generate and save a comparison plot."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

    for idx, agent_name in enumerate(results.keys()):
        if agent_name not in results or not results[agent_name]: continue

        # Aggregate across vehicle scenarios
        # results structure: {agent: {num_vehicles: [list of episode metrics]}}
        y_values = []
        x_values = []

        for num_v in scenarios:
            if num_v in results[agent_name]:
                ep_data = results[agent_name][num_v]
                if ep_data:
                    # Filter out None results from failed runs
                    valid_data = [x for x in ep_data if x is not None]
                    if valid_data:
                        val = np.mean([res[metric_name] for res in valid_data])
                        y_values.append(val)
                        x_values.append(num_v)

        if y_values:
            ax.plot(x_values, y_values, marker='o', linestyle='--', label=agent_name, color=colors[idx % len(colors)])

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Number of Vehicle Users', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.savefig(filename)
    print(f"Plot saved to '{filename}'")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained UAV Agents")

    # ARGUMENT: Path to the specific timestamped folder
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the timestamped model directory (e.g., models/experiment_2025-...)')

    # ARGUMENT: Where to save plots
    parser.add_argument('--output_dir', type=str, default='Evaluations',
                        help='Directory to save evaluation plots')

    args = parser.parse_args()

    print(f"--- Starting Evaluation ---")
    print(f"Loading models from: {args.model_path}")
    print(f"Saving results to:   {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    agent_types = ["MUCEDS", "OUPRS", "OUPOS", "MRUPRS", "MRUPOS", "MOUPRS"]
    vehicle_scenarios = list(EVAL_SCENARIO_VEHICLES)

    # 1. Create a flat list of all tasks to run
    tasks = []
    for num_vehicles in vehicle_scenarios:
        for agent_type in agent_types:
            for i in range(EVAL_EPISODES):
                seed = num_vehicles * 1000 + i
                # Pass the command line model path
                tasks.append((agent_type, num_vehicles, seed, args.model_path))

    # 2. Run tasks in parallel
    print(f"Distributing {len(tasks)} total episodes across CPU cores...")

    # Structure: {agent_name: {num_vehicles: [metrics_dict, ...]}}
    aggregated_results = {name: {num_v: [] for num_v in vehicle_scenarios} for name in agent_types}

    # Use max_workers=os.cpu_count() or slightly less
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
        future_to_task = {executor.submit(run_evaluation_episode_worker, task): task for task in tasks}

        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Evaluating Episodes"):
            agent_type, num_vehicles, _, _ = future_to_task[future]
            try:
                result_metrics = future.result()
                if result_metrics:
                    aggregated_results[agent_type][num_vehicles].append(result_metrics)
            except Exception as e:
                print(f"ERROR: Task {agent_type} with {num_vehicles} vehicles failed: {e}")

    # 3. Calculate final averages & Plotting
    # Note: We pass aggregated_results directly to plot now to handle averaging cleanly there
    print("Plotting final results...")

    plot_comparison('profit', aggregated_results, vehicle_scenarios, 'Average System Profit',
                    'Comparison of System Profit', os.path.join(args.output_dir, 'evaluation_profit_results.png'))

    plot_comparison('tasks_completed', aggregated_results, vehicle_scenarios, 'Average Processed Tasks',
                    'Comparison of Processed Tasks', os.path.join(args.output_dir, 'evaluation_tasks_results.png'))

    plot_comparison('avg_latency', aggregated_results, vehicle_scenarios, 'Average Task Latency (steps)',
                    'Comparison of Task Latency', os.path.join(args.output_dir, 'evaluation_latency_results.png'))


if __name__ == "__main__":
    # Windows multiprocessing support
    import multiprocessing

    multiprocessing.freeze_support()
    main()
