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

from cache_predictor import LSTMCachePredictor
from config import *
from ddqn_agent import DDQNAgent
from environment import VECNEnvironment
from maddpg_agent import MADDPGController

if VISUALIZATION:
    from visualizer import Visualizer
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
    # MODIFIED: Pass visualize=False since this is the non-SUMO version
    env = VECNEnvironment()
    ddqn_agent = DDQNAgent(state_dim=DDQN_STATE_DIM, action_space=DDQN_ACTION_SPACE)

    scores_window = deque(maxlen=100)
    start_time = time.time()
    maddpg_controller = None  # Initialize to handle case where first episode has 0 UAVs

    predictor = None
    if USE_PREDICTIVE_CACHING:
        try:
            predictor = LSTMCachePredictor().to(DEVICE) # Architecture updated
            predictor.load_state_dict(torch.load("lstm_cache_predictor.pth", map_location=DEVICE))
            predictor.eval()
            print("✅ Loaded Context-Aware Predictor.")
        except Exception as e:
            print(f"⚠️ Load failed: {e}")

    visualizer = None
    # If it's meant to stay open, create it once at the very beginning.
    if VISUALIZATION and VISUALIZER_STAYS_OPEN:
        print("\nInitializing persistent visualization window...")
        visualizer = Visualizer(env.width, env.height)

    for episode in range(TOTAL_EPISODES):

        current_episode_num = episode + 1
        is_snapshot_episode = current_episode_num in EPISODES_TO_SNAPSHOT
        if VISUALIZATION and not VISUALIZER_STAYS_OPEN:
            # If it's a snapshot episode and the window isn't open, create it.
            if is_snapshot_episode and visualizer is None:
                print(f"\nOpening visualization for snapshot episode {current_episode_num}...")
                visualizer = Visualizer(env.width, env.height)
            # If it's NOT a snapshot episode and the window IS open, close it.
            elif not is_snapshot_episode and visualizer is not None:
                print(f"\nClosing visualization after snapshot episode {current_episode_num - 1}.")
                visualizer.close()
                visualizer = None
                logger.info("========== Starting Episode %d ==========", episode + 1)

        outer_state = env.get_ddqn_state()
        logger.debug("DDQN Agent received outer_state: %s", outer_state)

        num_uavs = ddqn_agent.select_action(outer_state) + 1
        logger.info("DDQN Agent selected to deploy %d UAVs", num_uavs)

        # Initialize trackers for inner loop losses
        episode_maddpg_critic_losses = []
        episode_maddpg_actor_losses = []

        if num_uavs > 0:
            maddpg_controller = MADDPGController(num_agents=num_uavs, state_dim=MADDPG_STATE_DIM,
                                                 action_dim=MADDPG_ACTION_DIM)
            logger.info("Initialized MADDPGController for %d agents.", num_uavs)
            inner_states = env.reset(num_uavs=num_uavs, num_vehicles=NUM_VEHICLES)

            logger.info("--- Starting Inner Loop (MADDPG) for %d steps ---", INNER_STEPS)
            for t in range(INNER_STEPS):

                if visualizer:
                    visualizer.draw(env.uavs, env.vehicles, episode + 1, t + 1)

                if predictor and t > 0 and t % CACHE_UPDATE_INTERVAL == 0:
                    # 1. Get History (Now returns tuples)
                    recent_data = env.get_recent_requests(PREDICTION_SEQUENCE_LENGTH)

                    if len(recent_data) >= PREDICTION_SEQUENCE_LENGTH:
                        # 2. Unpack Services and Zones
                        s_seq = [x[0] for x in recent_data]
                        z_seq = [x[1] for x in recent_data]

                        # 3. Create Tensors
                        s_tensor = torch.LongTensor([s_seq]).to(DEVICE)
                        z_tensor = torch.LongTensor([z_seq]).to(DEVICE)

                        with torch.no_grad():
                            # 4. Pass BOTH to model
                            s_preds, c_preds = predictor(s_tensor, z_tensor)

                        # 5. Get Probabilities & Top-K
                        # Using probabilities is better than raw scores for confidence
                        s_probs = torch.softmax(s_preds, dim=1).cpu().numpy().flatten()
                        c_probs = torch.softmax(c_preds, dim=1).cpu().numpy().flatten()

                        # Get indices of highest probability items
                        top_k_s = s_probs.argsort()[-SERVICE_CACHE_SIZE:][::-1]
                        top_k_c = c_probs.argsort()[-CONTENT_CACHE_SIZE:][::-1]

                        for uav in env.uavs:
                            uav.update_cache_from_prediction(top_k_s, top_k_c)

                actions = maddpg_controller.select_actions(inner_states)
                next_inner_states, rewards, done = env.step(actions)

                # Ensure all lists have content before concatenating
                if not all(s is not None and len(s) > 0 for s in inner_states): continue
                if not all(a is not None and len(a) > 0 for a in actions): continue
                if not all(ns is not None and len(ns) > 0 for ns in next_inner_states): continue

                flat_states = np.concatenate(inner_states)
                flat_actions = np.concatenate(actions)
                flat_next_states = np.concatenate(next_inner_states)
                maddpg_controller.memory.add(flat_states, flat_actions, rewards[0], flat_next_states, done)

                # learn() method in MADDPG returns two values
                critic_loss, actor_loss = maddpg_controller.learn()
                if critic_loss is not None and actor_loss is not None:
                    if critic_loss > 0:  # Only append if a learning step was actually performed
                        episode_maddpg_critic_losses.append(critic_loss)
                        episode_maddpg_actor_losses.append(actor_loss)
                maddpg_controller.update_targets()

                inner_states = next_inner_states
                if done:
                    break
            logger.info("--- Inner Loop (MADDPG) Finished ---")
        else:  # Handle case where 0 UAVs are selected
            env.reset(num_uavs=0, num_vehicles=NUM_VEHICLES)

        next_outer_state = env.get_ddqn_state()
        outer_reward = next_outer_state[3]
        logger.info("Episode %d finished with outer_reward: %.2f", episode + 1, outer_reward)

        ddqn_action = num_uavs - 1
        ddqn_agent.memory.add(outer_state, ddqn_action, outer_reward, next_outer_state, False)
        # This now correctly returns the loss value
        ddqn_loss = ddqn_agent.learn()
        ddqn_agent.update_target_network()

        scores_window.append(outer_reward)
        avg_score = np.mean(scores_window)

        # --- FULLY SYNCHRONIZED TENSORBOARD LOGGING ---
        current_episode_num = episode + 1

        # 1. Log core profit and agent metrics
        writer.add_scalar('Profit/Average_Profit_100_Episodes', avg_score, current_episode_num)
        writer.add_scalar('Profit/Episode_Profit', outer_reward, current_episode_num)
        writer.add_scalar('DDQN/Epsilon', ddqn_agent.epsilon, current_episode_num)
        writer.add_scalar('DDQN/UAVs_Chosen', num_uavs, current_episode_num)

        # 2. Log RL loss values
        writer.add_scalar('Loss/DDQN_Critic_Loss', ddqn_loss, current_episode_num)
        if episode_maddpg_critic_losses:
            writer.add_scalar('Loss/MADDPG_Avg_Critic_Loss', np.mean(episode_maddpg_critic_losses), current_episode_num)
            writer.add_scalar('Loss/MADDPG_Avg_Actor_Loss', np.mean(episode_maddpg_actor_losses), current_episode_num)

        # 3. Log detailed environment statistics
        episode_stats = env.get_episode_statistics()
        if episode_stats:  # Check if stats are available
            for key, value in episode_stats.items():
                # Replace "SUMO/" with "Sim/" for clarity in the non-SUMO version
                key = key.replace("SUMO/", "Sim/")
                writer.add_scalar(key, value, current_episode_num)

        # 4. Log latency from the outer state
        avg_latency = next_outer_state[4]
        writer.add_scalar('Tasks/average_latency', avg_latency, current_episode_num)
        # --- END OF LOGGING BLOCK ---

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

    if visualizer is not None:
        visualizer.close()

    writer.close()
    total_training_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info("--- Training Finished in %s ---", total_training_time)
    print(f"\n--- Training Finished in {total_training_time} ---")

    logger.info("--- Saving trained models ---")
    print("--- Saving trained models ---")
    ddqn_agent.save(MODEL_SAVE_PATH)

    if maddpg_controller is not None:
        maddpg_save_path = os.path.join(MODEL_SAVE_PATH, f"maddpg_{maddpg_controller.num_agents}_agents")
        maddpg_controller.save(maddpg_save_path)
        logger.info("Saved MADDPG model for %d agents to %s.", maddpg_controller.num_agents, maddpg_save_path)
        print(f"Saved MADDPG model for {maddpg_controller.num_agents} agents.")

    logger.info("Models saved to '%s' directory.", MODEL_SAVE_PATH)
    print(f"Models saved to '{MODEL_SAVE_PATH}' directory.")


if __name__ == "__main__":
    run_training()
