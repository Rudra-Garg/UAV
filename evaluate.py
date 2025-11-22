# evaluate_parallel.py
"""
Optimized evaluation script for the Python-only version with parallel execution.
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
from tqdm import tqdm

from benchmark_agents import *
from config import *
from ddqn_agent import DDQNAgent
from environment import VECNEnvironment  # Python-only environment
from maddpg_agent import MADDPGController


def run_evaluation_episode_worker(args):
    """
    Worker function to run a single episode. Designed for parallel execution.
    It initializes its own environment and agents to ensure process safety.
    """
    agent_type, num_vehicles, episode_seed, model_save_path = args

    # Each worker process must create its own environment instance.
    env = VECNEnvironment()

    # Seed for reproducibility within the worker.
    np.random.seed(episode_seed)

    # Load agents based on the specified type.
    ddqn_agent_for_benchmarks = DDQNAgent(state_dim=DDQN_STATE_DIM, action_space=DDQN_ACTION_SPACE)
    ddqn_agent_for_benchmarks.load(model_save_path)

    if agent_type == 'MUCEDS':
        maddpg_controllers = {}
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

    # --- Run the actual episode logic (copied from the original script) ---
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

    inner_states = env.reset(num_uavs=num_uavs, num_vehicles=num_vehicles)

    if not env.uavs:
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

    return {
        'profit': final_state[3],
        'tasks_completed': tasks_completed,
        'avg_latency': avg_latency
    }


def plot_comparison(metric_name, results, scenarios, ylabel, title, filename):
    """Helper function to generate and save a comparison plot."""
    # This function remains unchanged from the original evaluate.py
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
    plt.savefig(filename)
    print(f"Plot saved to '{filename}'")
    plt.close(fig)


def main():
    print("--- Starting Parallel Evaluation for Python-Only Version ---")
    os.makedirs('Evaluations', exist_ok=True)

    agent_types = ["MUCEDS", "OUPRS", "OUPOS", "MRUPRS", "MRUPOS", "MOUPRS"]
    vehicle_scenarios = list(EVAL_SCENARIO_VEHICLES)

    # 1. Create a flat list of all tasks to run
    tasks = []
    for num_vehicles in vehicle_scenarios:
        for agent_type in agent_types:
            for i in range(EVAL_EPISODES):
                # Each task needs a unique seed for reproducibility
                seed = num_vehicles * 1000 + i
                tasks.append((agent_type, num_vehicles, seed, MODEL_SAVE_PATH))

    # 2. Run tasks in parallel
    print(f"Distributing {len(tasks)} total episodes across CPU cores...")
    raw_results = []
    # Adjust max_workers based on your system's CPU count
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        # Use a dictionary to map futures to their original task arguments
        future_to_task = {executor.submit(run_evaluation_episode_worker, task): task for task in tasks}

        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Evaluating Episodes"):
            agent_type, num_vehicles, _, _ = future_to_task[future]
            try:
                result_metrics = future.result()
                raw_results.append((agent_type, num_vehicles, result_metrics))
            except Exception as e:
                print(f"ERROR: Task {agent_type} with {num_vehicles} vehicles failed: {e}")

    # 3. Aggregate results
    # A nested dictionary to hold the lists of results: {agent_name: {num_vehicles: [res1, res2, ...]}}
    aggregated_results = {name: {num_v: [] for num_v in vehicle_scenarios} for name in agent_types}
    for agent_type, num_vehicles, metrics in raw_results:
        aggregated_results[agent_type][num_vehicles].append(metrics)

    # 4. Calculate final averages
    final_results = {name: [] for name in agent_types}
    for agent_name in agent_types:
        for num_vehicles in vehicle_scenarios:
            episode_metrics = aggregated_results[agent_name][num_vehicles]
            avg_metrics = {
                'profit': np.mean([res['profit'] for res in episode_metrics]),
                'tasks_completed': np.mean([res['tasks_completed'] for res in episode_metrics]),
                'avg_latency': np.mean([res['avg_latency'] for res in episode_metrics]),
            }
            final_results[agent_name].append(avg_metrics)

    # 5. Plotting
    print("Plotting final results...")
    plot_comparison('profit', final_results, vehicle_scenarios, 'Average System Profit', 'Comparison of System Profit',
                    'Evaluations_1/lstm/evaluation_profit_results.png')
    plot_comparison('tasks_completed', final_results, vehicle_scenarios, 'Average Processed Tasks',
                    'Comparison of Processed Tasks', 'Evaluations_1/lstm/evaluation_tasks_results.png')
    plot_comparison('avg_latency', final_results, vehicle_scenarios, 'Average Task Latency (steps)',
                    'Comparison of Task Latency', 'Evaluations_1/lstm/evaluation_latency_results.png')


if __name__ == "__main__":
    # This is crucial for multiprocessing on some platforms (like Windows)
    import multiprocessing

    multiprocessing.freeze_support()
    main()
