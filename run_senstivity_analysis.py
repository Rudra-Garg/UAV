# run_sensitivity_analysis.py
"""
Main script for Phase 5: Sensitivity Analysis.

This script loads the final trained HRL agent and systematically evaluates its
performance under a range of different economic conditions. It modifies key
parameters from the config file one by one, runs a series of evaluation
episodes, and plots the results to show how sensitive the system's performance
is to our initial assumptions.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Import the base configuration and agent/environment classes
import config
from ddqn_agent import DDQNAgent
from environment import VECNEnvironment
from maddpg_agent import MADDPGController

# --- Parameters to Analyze ---
# Define the parameters we want to test and the range of values for each.
# Format: { 'parameter_name_in_config': np.linspace(start, end, num_points) }
PARAMETERS_TO_ANALYZE = {
    'BETA_MAINTENANCE': np.linspace(500, 2500, 9),  # How does UAV fixed cost affect deployment?
    'DELTA_LATENCY': np.linspace(1.0, 10.0, 9),  # How does the reward for speed affect profit?
    'ENERGY_REWARD_PENALTY': np.linspace(0.0, 0.0005, 9)  # How does energy cost affect behavior?
}

# --- Evaluation Settings ---
NUM_EVAL_EPISODES = 10  # Number of episodes to average for each data point
ANALYSIS_RESULTS_DIR = "sensitivity_analysis_results"


def run_analysis_episode(env, ddqn_agent, maddpg_controllers):
    """
    Runs a single evaluation episode for the trained agent.
    Returns a dictionary of key performance metrics for this episode.
    """
    # Let the DDQN agent choose the optimal number of UAVs
    outer_state = env.get_ddqn_state()
    num_uavs = ddqn_agent.select_action(outer_state, evaluation=True) + 1

    # Get the corresponding MADDPG controller
    maddpg_controller = maddpg_controllers.get(num_uavs)

    # Reset the environment with the chosen number of UAVs
    inner_states = env.reset(num_uavs=num_uavs)

    if not env.uavs or not maddpg_controller:
        # Handle the case where 0 UAVs are deployed or controller is missing
        return {'net_profit': 0, 'tasks_completed': 0, 'avg_latency': 0, 'uavs_deployed': 0}

    # Run the inner loop simulation
    for _ in range(config.INNER_STEPS):
        actions = maddpg_controller.select_actions(inner_states, evaluation=True)
        next_inner_states, _, done = env.step(actions)
        inner_states = next_inner_states
        if done:
            break

    # Get final state metrics
    final_state = env.get_ddqn_state()
    metrics = {
        'tasks_completed': final_state[0],
        'uavs_deployed': final_state[1],
        'net_profit': final_state[3],
        'avg_latency': final_state[4] if final_state[0] > 0 else 0
    }
    return metrics


def plot_results(param_name, values, results):
    """Generates and saves a plot summarizing the analysis for one parameter."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Sensitivity Analysis for: {param_name}', fontsize=16, y=0.95)

    # Plot Net Profit
    axs[0, 0].plot(values, results['net_profit'], 'o-', color='b')
    axs[0, 0].set_title('Average Net Profit')
    axs[0, 0].set_xlabel(param_name)
    axs[0, 0].set_ylabel('Profit')
    axs[0, 0].grid(True)

    # Plot Tasks Completed
    axs[0, 1].plot(values, results['tasks_completed'], 'o-', color='g')
    axs[0, 1].set_title('Average Tasks Completed')
    axs[0, 1].set_xlabel(param_name)
    axs[0, 1].set_ylabel('Tasks')
    axs[0, 1].grid(True)

    # Plot UAVs Deployed
    axs[1, 0].plot(values, results['uavs_deployed'], 'o-', color='r')
    axs[1, 0].set_title('Average Number of UAVs Deployed')
    axs[1, 0].set_xlabel(param_name)
    axs[1, 0].set_ylabel('Number of UAVs')
    axs[1, 0].grid(True)

    # Plot Average Latency
    axs[1, 1].plot(values, results['avg_latency'], 'o-', color='purple')
    axs[1, 1].set_title('Average Task Latency')
    axs[1, 1].set_xlabel(param_name)
    axs[1, 1].set_ylabel('Latency (s)')
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    save_path = os.path.join(ANALYSIS_RESULTS_DIR, f'sensitivity_{param_name}.png')
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close(fig)


def main():
    """Main function to load agents and run the sensitivity analysis."""
    print("--- Starting Phase 5: Sensitivity Analysis ---")

    # Create the results directory if it doesn't exist
    if not os.path.exists(ANALYSIS_RESULTS_DIR):
        os.makedirs(ANALYSIS_RESULTS_DIR)

    # --- Load Trained Agents ---
    print("Loading pre-trained models...")
    try:
        ddqn_agent = DDQNAgent(state_dim=config.DDQN_STATE_DIM, action_space=config.DDQN_ACTION_SPACE)
        ddqn_agent.load(config.MODEL_SAVE_PATH)

        maddpg_controllers = {}
        for i in range(1, config.DDQN_ACTION_SPACE + 1):
            path = os.path.join(config.MODEL_SAVE_PATH, f"maddpg_{i}_agents")
            if os.path.exists(os.path.join(path, 'maddpg_actor_0.pth')):
                controller = MADDPGController(num_agents=i, state_dim=config.MADDPG_STATE_DIM,
                                              action_dim=config.MADDPG_ACTION_DIM)
                controller.load(path)
                maddpg_controllers[i] = controller
        print(f"Successfully loaded DDQN agent and {len(maddpg_controllers)} MADDPG controllers.")
    except FileNotFoundError:
        print("\nERROR: Trained models not found. Please run the training script (main.py) first.")
        return

    env = VECNEnvironment()

    # --- Run Analysis for Each Parameter ---
    for param_name, values_range in PARAMETERS_TO_ANALYZE.items():
        print(f"\n--- Analyzing Parameter: {param_name} ---")

        # Store original value to restore it later
        original_value = getattr(config, param_name)

        # This dictionary will store the averaged results for plotting
        analysis_results = {metric: [] for metric in ['net_profit', 'tasks_completed', 'avg_latency', 'uavs_deployed']}

        # Iterate over the range of values for the current parameter
        for value in tqdm(values_range, desc=f"Testing {param_name}"):
            # Dynamically set the new value in the config module
            setattr(config, param_name, value)

            # Since the environment's methods may depend on config values, we should
            # reload it or re-initialize it to be safe.
            # For this codebase, direct modification is okay, but re-init is safer.
            env = VECNEnvironment()

            # Store metrics from multiple episodes to get a stable average
            episode_metrics = {metric: [] for metric in analysis_results.keys()}

            for _ in range(NUM_EVAL_EPISODES):
                metrics = run_analysis_episode(env, ddqn_agent, maddpg_controllers)
                for key in metrics:
                    episode_metrics[key].append(metrics[key])

            # Average the results and store them
            for key in analysis_results.keys():
                analysis_results[key].append(np.mean(episode_metrics[key]))

        # Restore the original config value
        setattr(config, param_name, original_value)

        # Plot the results for the analyzed parameter
        plot_results(param_name, values_range, analysis_results)

    print("\n--- Sensitivity Analysis Complete ---")


if __name__ == "__main__":
    main()
