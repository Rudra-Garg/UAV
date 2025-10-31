"""
Main entry point for running the HRL training loop.
This script integrates the entire simulation and learning process and now includes
logic to save the final trained models for later evaluation.
"""
import datetime
import os
import time
from collections import deque

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config import *
from ddqn_agent import DDQNAgent
from environment import VECNEnvironment
from maddpg_agent import MADDPGController

if VISUALIZATION:
    pass
import logging


# Define a filter to allow only a specific log level
class SingleLevelFilter(logging.Filter):
    def __init__(self, level):
        super().__init__()
        self.level = level

    def filter(self, record):
        return record.levelno == self.level


# --- Centralized Logger Setup ---
# 1. Get the root logger
root_logger = logging.getLogger()
# Prevent duplicate handlers if this script is run in an interactive session
if root_logger.hasHandlers():
    root_logger.handlers.clear()
root_logger.setLevel(logging.DEBUG)  # Set the lowest-level to capture all messages

# 2. Create a detailed formatter
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 3. Create log directories if they don't exist
log_dir = f'logs/{timestamp}'
os.makedirs(log_dir, exist_ok=True)

# 4. Create a file handler for DEBUG messages ONLY
debug_file_handler = logging.FileHandler(f'{log_dir}/debug.log', mode='w')
debug_file_handler.setLevel(logging.DEBUG)
debug_file_handler.setFormatter(log_formatter)
debug_file_handler.addFilter(SingleLevelFilter(logging.DEBUG))
root_logger.addHandler(debug_file_handler)

# 5. Create a file handler for INFO and higher level messages
info_file_handler = logging.FileHandler(f'{log_dir}/info.log', mode='w')
info_file_handler.setLevel(logging.INFO)
info_file_handler.setFormatter(log_formatter)
root_logger.addHandler(info_file_handler)

# Mute Numba's logs to keep the output clean
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
# --- End of Logger Setup ---


# This module's logger will inherit the root logger's configuration
logger = logging.getLogger(__name__)
logger.info("Training started with config: TOTAL_EPISODES=%d, INNER_STEPS=%d", TOTAL_EPISODES, INNER_STEPS)


def run_training():
    """Initializes all components and executes the main HRL training loop."""
    logger.info("--- Initializing HRL Training on device: %s ---", DEVICE)

    writer = SummaryWriter(f"runs/muceds_experiment_{timestamp}")
    env = VECNEnvironment(visualize=VISUALIZATION)
    ddqn_agent = DDQNAgent(state_dim=DDQN_STATE_DIM, action_space=DDQN_ACTION_SPACE)

    scores_window = deque(maxlen=100)
    start_time = time.time()
    maddpg_controller = None  # Initialize to handle case where first episode has 0 UAVs

    for episode in range(TOTAL_EPISODES):
        current_episode_num = episode + 1

        logger.info("========== Starting Episode %d ==========", episode + 1)
        outer_state = env.get_ddqn_state()
        logger.debug("DDQN Agent received outer_state: %s", outer_state)

        num_uavs = ddqn_agent.select_action(outer_state) + 1
        logger.info("DDQN Agent selected to deploy %d UAVs", num_uavs)

        if num_uavs > 0:
            maddpg_controller = MADDPGController(num_agents=num_uavs, state_dim=MADDPG_STATE_DIM,
                                                 action_dim=MADDPG_ACTION_DIM)
            logger.info("Initialized MADDPGController for %d agents.", num_uavs)
            inner_states = env.reset(num_uavs=num_uavs, num_vehicles=NUM_VEHICLES)

            logger.info("--- Starting Inner Loop (MADDPG) for %d steps ---", INNER_STEPS)
            for t in range(INNER_STEPS):
                current_step_num = t + 1

                actions = maddpg_controller.select_actions(inner_states)
                next_inner_states, rewards, done = env.step(actions)

                flat_states = np.concatenate(inner_states)
                flat_actions = np.concatenate(actions)
                flat_next_states = np.concatenate(next_inner_states)
                maddpg_controller.memory.add(flat_states, flat_actions, rewards[0], flat_next_states, done)

                maddpg_controller.learn()
                maddpg_controller.update_targets()

                inner_states = next_inner_states
                if done:
                    break
            logger.info("--- Inner Loop (MADDPG) Finished ---")
        else:  # Handle case where 0 UAVs are selected
            # We still need to run an empty environment to get the next state
            env.reset(num_uavs=0)
            # The loop will be skipped, and we'll just get the final state.

        next_outer_state = env.get_ddqn_state()
        outer_reward = next_outer_state[3]
        logger.info("Episode %d finished with outer_reward: %.2f", episode + 1, outer_reward)

        ddqn_action = num_uavs - 1
        ddqn_agent.memory.add(outer_state, ddqn_action, outer_reward, next_outer_state, False)
        ddqn_agent.learn()
        ddqn_agent.update_target_network()

        scores_window.append(outer_reward)
        avg_score = np.mean(scores_window)

        writer.add_scalar('Profit/Average_Profit_100_Episodes', avg_score, episode + 1)
        writer.add_scalar('Profit/Episode_Profit', outer_reward, episode + 1)
        writer.add_scalar('DDQN/Epsilon', ddqn_agent.epsilon, episode + 1)
        writer.add_scalar('DDQN/UAVs_Chosen', num_uavs, episode + 1)

        elapsed_time = time.time() - start_time
        avg_time_per_episode = elapsed_time / (episode + 1)
        episodes_remaining = TOTAL_EPISODES - (episode + 1)
        eta_seconds = avg_time_per_episode * episodes_remaining
        eta_formatted = str(datetime.timedelta(seconds=int(eta_seconds)))

        print(
            f'\rEpisode {episode + 1}/{TOTAL_EPISODES}\tAvg Score: {avg_score:.2f}\tUAVs: {num_uavs}\tETA: {eta_formatted}',
            end="")
        if (episode + 1) % 100 == 0:
            print(
                f'\rEpisode {episode + 1}/{TOTAL_EPISODES}\tAvg Score: {avg_score:.2f}\tUAVs: {num_uavs}\tETA: {eta_formatted}')

        logger.info("========== Finished Episode %d ==========\n", episode + 1)

    writer.close()
    total_training_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info("--- Training Finished in %s ---", total_training_time)
    print(f"\n--- Training Finished in {total_training_time} ---")

    # --- Save the final trained models ---
    logger.info("--- Saving trained models ---")
    print("--- Saving trained models ---")
    ddqn_agent.save(MODEL_SAVE_PATH)

    # The MADDPG controller is ephemeral and changes based on `num_uavs`.
    # A robust approach would be to train and save a separate MADDPG model for each possible `num_uavs`.
    # For simplicity, we save the controller from the very last episode as a representative sample.
    if maddpg_controller is not None:
        maddpg_save_path = os.path.join(MODEL_SAVE_PATH, f"maddpg_{maddpg_controller.num_agents}_agents")
        maddpg_controller.save(maddpg_save_path)
        logger.info("Saved MADDPG model for %d agents to %s.", maddpg_controller.num_agents, maddpg_save_path)
        print(f"Saved MADDPG model for {maddpg_controller.num_agents} agents.")

    logger.info("Models saved to '%s' directory.", MODEL_SAVE_PATH)
    print(f"Models saved to '{MODEL_SAVE_PATH}' directory.")


if __name__ == "__main__":
    run_training()
