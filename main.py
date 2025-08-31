# main.py
"""
Main entry point for running the HRL training loop.
"""
import datetime
import time
from collections import deque
import numpy as np
from config import *
from ddqn_agent import DDQNAgent
from environment import VECNEnvironment
from maddpg_agent import MADDPGController


def run_training():
    print(f"--- Starting HRL Training on device: {DEVICE} ---")

    env = VECNEnvironment()
    ddqn_agent = DDQNAgent(state_dim=DDQN_STATE_DIM, action_space=DDQN_ACTION_SPACE)

    scores_window = deque(maxlen=100)
    start_time = time.time()  # <--- ADDED: Record the start time

    for episode in range(TOTAL_EPISODES):
        # --- Outer Loop (DDQN) ---
        outer_state = env.get_ddqn_state()
        # Action is the number of UAVs to deploy. Add 1 because action is 0-indexed.
        num_uavs = ddqn_agent.select_action(outer_state) + 1

        # --- Inner Loop (MADDPG) ---
        if num_uavs > 0:
            maddpg_controller = MADDPGController(num_agents=num_uavs, state_dim=MADDPG_STATE_DIM,
                                                 action_dim=MADDPG_ACTION_DIM)
            inner_states = env.reset(num_uavs=num_uavs)

            episode_rewards = np.zeros(num_uavs)

            for t in range(INNER_STEPS):
                actions = maddpg_controller.select_actions(inner_states)

                next_inner_states, rewards, done = env.step(actions)

                # Store experience in MADDPG replay buffer
                # We need to flatten states and actions for the buffer
                flat_states = np.concatenate(inner_states)
                flat_actions = np.concatenate(actions)
                flat_next_states = np.concatenate(next_inner_states)
                # Use a single reward and done signal for the whole team
                maddpg_controller.memory.add(flat_states, flat_actions, rewards[0], flat_next_states, done)

                maddpg_controller.learn()
                maddpg_controller.update_targets()

                inner_states = next_inner_states
                episode_rewards += rewards

                if done:
                    break

        # --- Post-Inner Loop (Update DDQN) ---
        # The reward for the DDQN is the final profit from the simulation
        next_outer_state = env.get_ddqn_state()
        outer_reward = next_outer_state[3]  # Index 3 is total_profit

        # DDQN action is 0-indexed
        ddqn_action = num_uavs - 1
        ddqn_agent.memory.add(outer_state, ddqn_action, outer_reward, next_outer_state, False)
        ddqn_agent.learn()
        ddqn_agent.update_target_network()

        scores_window.append(outer_reward)

        # --- ADDED: ETA Calculation Logic ---
        elapsed_time = time.time() - start_time
        avg_time_per_episode = elapsed_time / (episode + 1)
        episodes_remaining = TOTAL_EPISODES - (episode + 1)
        eta_seconds = avg_time_per_episode * episodes_remaining
        eta_formatted = str(datetime.timedelta(seconds=int(eta_seconds)))

        print(
            f'\rEpisode {episode}/{TOTAL_EPISODES}\tAvg Score: {np.mean(scores_window):.2f}\tUAVs: {num_uavs}\tETA: {eta_formatted}',
            end="")
        if (episode + 1) % 100 == 0:
            print(
                f'\rEpisode {episode + 1}/{TOTAL_EPISODES}\tAvg Score: {np.mean(scores_window):.2f}\tUAVs: {num_uavs}\tETA: {eta_formatted}')

    # --- ADDED: Final Training Time ---
    total_training_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"\n--- Training Finished in {total_training_time} ---")


if __name__ == "__main__":
    run_training()
