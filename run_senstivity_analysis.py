import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Import base config and other necessary classes
import config
from ddqn_agent import DDQNAgent
from environment import VECNEnvironment
from maddpg_agent import MADDPGController

# --- Parameters to Analyze ---
PARAMETERS_TO_ANALYZE = {
    'BETA_MAINTENANCE': np.linspace(10, 25000, 50),
    'DELTA_LATENCY': np.linspace(1.0, 100.0, 50),
    'ENERGY_REWARD_PENALTY': np.linspace(0.0, 0.001, 50)
}
NUM_EVAL_EPISODES = 10
ANALYSIS_RESULTS_DIR = "sensitivity_analysis_results"


def safe_close_env(env):
    """Safely close environment if it has a close method."""
    if hasattr(env, 'close') and callable(getattr(env, 'close')):
        try:
            env.close()
        except Exception as e:
            pass  # Silently ignore close errors


def run_single_analysis(task_info):
    """
    Worker function for the sensitivity analysis.
    Takes a parameter, its value, and a seed, then runs one episode.
    Each worker creates its own environment and agents.
    """
    param_name, value, seed, model_save_path = task_info
    np.random.seed(seed)

    env = None
    original_value = None

    try:
        # Create a temporary config override for this process
        original_value = getattr(config, param_name)
        setattr(config, param_name, value)

        # Load agents (each worker needs its own)
        ddqn_agent = DDQNAgent(state_dim=config.DDQN_STATE_DIM, action_space=config.DDQN_ACTION_SPACE)
        ddqn_agent.load(model_save_path)

        maddpg_controllers = {}
        for i in range(1, config.DDQN_ACTION_SPACE + 1):
            path = os.path.join(model_save_path, f"maddpg_{i}_agents")
            if os.path.exists(os.path.join(path, 'maddpg_actor_0.pth')):
                controller = MADDPGController(num_agents=i, state_dim=config.MADDPG_STATE_DIM,
                                              action_dim=config.MADDPG_ACTION_DIM)
                controller.load(path)
                maddpg_controllers[i] = controller

        # Create environment with modified config
        env = VECNEnvironment()

        # Get initial state and select UAVs
        outer_state = env.get_ddqn_state()
        num_uavs = ddqn_agent.select_action(outer_state, evaluation=True) + 1
        maddpg_controller = maddpg_controllers.get(num_uavs)

        # Reset environment with selected number of UAVs
        inner_states = env.reset(num_uavs=num_uavs)

        if not env.uavs or not maddpg_controller:
            return param_name, value, {'net_profit': 0, 'tasks_completed': 0, 'avg_latency': 0, 'uavs_deployed': 0}

        # Run the episode
        for _ in range(config.INNER_STEPS):
            actions = maddpg_controller.select_actions(inner_states, evaluation=True)
            next_inner_states, _, done = env.step(actions)
            inner_states = next_inner_states
            if done:
                break

        # Collect final metrics
        final_state = env.get_ddqn_state()
        metrics = {
            'tasks_completed': final_state[0],
            'uavs_deployed': final_state[1],
            'net_profit': final_state[3],
            'avg_latency': final_state[4] if final_state[0] > 0 else 0
        }

        return param_name, value, metrics

    except Exception as e:
        print(f"\nError in analysis for {param_name}={value}, seed={seed}: {str(e)}")
        import traceback
        traceback.print_exc()
        return param_name, value, {'net_profit': 0, 'tasks_completed': 0, 'avg_latency': 0, 'uavs_deployed': 0}

    finally:
        # Clean up environment
        if env is not None:
            safe_close_env(env)
            del env

        # Restore original config value
        if original_value is not None:
            setattr(config, param_name, original_value)


def plot_results(param_name, values, results):
    """Generate and save sensitivity analysis plots."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Sensitivity Analysis for: {param_name}', fontsize=16, y=0.95)

    # Net Profit
    axs[0, 0].plot(values, results['net_profit'], 'o-', color='b')
    axs[0, 0].set_title('Average Net Profit')
    axs[0, 0].set_xlabel(param_name)
    axs[0, 0].set_ylabel('Net Profit')
    axs[0, 0].grid(True)

    # Tasks Completed
    axs[0, 1].plot(values, results['tasks_completed'], 'o-', color='g')
    axs[0, 1].set_title('Average Tasks Completed')
    axs[0, 1].set_xlabel(param_name)
    axs[0, 1].set_ylabel('Tasks Completed')
    axs[0, 1].grid(True)

    # Average Latency
    axs[1, 0].plot(values, results['avg_latency'], 'o-', color='r')
    axs[1, 0].set_title('Average Task Latency')
    axs[1, 0].set_xlabel(param_name)
    axs[1, 0].set_ylabel('Latency (steps)')
    axs[1, 0].grid(True)

    # UAVs Deployed
    axs[1, 1].plot(values, results['uavs_deployed'], 'o-', color='purple')
    axs[1, 1].set_title('Average UAVs Deployed')
    axs[1, 1].set_xlabel(param_name)
    axs[1, 1].set_ylabel('Number of UAVs')
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    save_path = os.path.join(ANALYSIS_RESULTS_DIR, f'sensitivity_{param_name}.png')
    plt.savefig(save_path, dpi=100)
    print(f"Plot saved to {save_path}")
    plt.close(fig)


def main():
    print("--- Starting Multithreaded Sensitivity Analysis ---")

    if not os.path.exists(ANALYSIS_RESULTS_DIR):
        os.makedirs(ANALYSIS_RESULTS_DIR)

    # 1. Create a flat list of all analysis tasks
    tasks = []
    for param_name, values_range in PARAMETERS_TO_ANALYZE.items():
        for i, value in enumerate(values_range):
            for episode in range(NUM_EVAL_EPISODES):
                seed = int(value * 1000) + episode
                tasks.append((param_name, value, seed, config.MODEL_SAVE_PATH))

    total_tasks = len(tasks)
    print(f"Running {total_tasks} analysis simulations...")

    # 2. Adjust max_workers based on system
    # SUMO can be resource-intensive, so limit parallelism
    max_workers = min(os.cpu_count() - 1 if os.cpu_count() else 1, 10)
    print(f"Using {max_workers} parallel workers")

    # 3. Run parallel analysis
    raw_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_analysis, task): task for task in tasks}

        with tqdm(total=total_tasks, desc="Analyzing") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=120)  # 2 minute timeout per episode
                    raw_results.append(result)
                except Exception as e:
                    task = futures[future]
                    print(f"\nFailed task {task}: {e}")
                    # Add default result for failed task
                    raw_results.append((task[0], task[1],
                                        {'net_profit': 0, 'tasks_completed': 0, 'avg_latency': 0, 'uavs_deployed': 0}))
                pbar.update(1)

    # 4. Aggregate results
    print("\nAggregating results...")
    aggregated_results = defaultdict(lambda: defaultdict(list))
    for param_name, value, metrics in raw_results:
        aggregated_results[param_name][value].append(metrics)

    # 5. Process and plot results for each parameter
    for param_name, values_range in PARAMETERS_TO_ANALYZE.items():
        print(f"\n--- Plotting Results for: {param_name} ---")

        final_results = {
            'net_profit': [],
            'tasks_completed': [],
            'avg_latency': [],
            'uavs_deployed': []
        }

        for value in values_range:
            episode_metrics_list = aggregated_results[param_name][value]
            if episode_metrics_list:
                for key in final_results.keys():
                    final_results[key].append(np.mean([m[key] for m in episode_metrics_list]))
            else:
                # Handle missing data
                for key in final_results.keys():
                    final_results[key].append(0)

        plot_results(param_name, values_range, final_results)

    print("\n--- Sensitivity Analysis Complete ---")


if __name__ == "__main__":
    main()
