# main.py
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
    from visualizer import Visualizer


def run_training():
    """Initializes all components and executes the main HRL training loop."""
    print(f"--- Starting HRL Training on device: {DEVICE} ---")

    writer = SummaryWriter("runs/muceds_experiment_1")
    env = VECNEnvironment()
    ddqn_agent = DDQNAgent(state_dim=DDQN_STATE_DIM, action_space=DDQN_ACTION_SPACE)

    if VISUALIZATION:
        visualizer = Visualizer(env.width, env.height)

    scores_window = deque(maxlen=100)
    start_time = time.time()
    maddpg_controller = None  # Initialize to handle case where first episode has 0 UAVs

    for episode in range(TOTAL_EPISODES):
        outer_state = env.get_ddqn_state()
        num_uavs = ddqn_agent.select_action(outer_state) + 1

        if num_uavs > 0:
            maddpg_controller = MADDPGController(num_agents=num_uavs, state_dim=MADDPG_STATE_DIM,
                                                 action_dim=MADDPG_ACTION_DIM)
            inner_states = env.reset(num_uavs=num_uavs)

            for t in range(INNER_STEPS):
                if VISUALIZATION:
                    visualizer.draw(env.uavs, env.vehicles, episode + 1, t + 1)

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

        next_outer_state = env.get_ddqn_state()
        outer_reward = next_outer_state[3]
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

    writer.close()
    total_training_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"\n--- Training Finished in {total_training_time} ---")

    # --- Save the final trained models ---
    print("--- Saving trained models ---")
    ddqn_agent.save(MODEL_SAVE_PATH)

    # The MADDPG controller is ephemeral and changes based on `num_uavs`.
    # A robust approach would be to train and save a separate MADDPG model for each possible `num_uavs`.
    # For simplicity, we save the controller from the very last episode as a representative sample.
    if maddpg_controller is not None:
        maddpg_save_path = os.path.join(MODEL_SAVE_PATH, f"maddpg_{maddpg_controller.num_agents}_agents")
        maddpg_controller.save(maddpg_save_path)
        print(f"Saved MADDPG model for {maddpg_controller.num_agents} agents.")

    print(f"Models saved to '{MODEL_SAVE_PATH}' directory.")


if __name__ == "__main__":
    run_training()
