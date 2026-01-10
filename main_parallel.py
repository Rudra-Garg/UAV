"""
====================================================================================
IMPROVED TRAINING SCRIPT FOR UAV-VECN (Sequential with Proper Cleanup)
====================================================================================
This script provides improved training with proper SUMO cleanup between episodes.

Key Features:
- Proper SUMO process cleanup (fixes the ~600 episode crash)
- Reuses MADDPG controllers for efficiency
- Full TensorBoard integration
- Saves models with UAV-count suffixes

Note: Currently runs sequentially due to TraCI global state limitations.
True parallel execution requires TraCI labeled connections (future enhancement).
====================================================================================
"""

import logging
import multiprocessing as mp
import os
import time
from collections import deque
from functools import partial

import numpy as np
import torch
import torch.multiprocessing as torch_mp
from torch.utils.tensorboard import SummaryWriter

from agents import DDQNAgent, MADDPGController
from config import *
from simulation import VECNEnvironment

# Conditional Imports
if USE_PREDICTIVE_CACHING:
    from prediction import LSTMCachePredictor

# --- Logging Setup ---
os.makedirs("logs", exist_ok=True)

# Generate experiment name with simulation mode, caching mode, and timestamp
experiment_name = get_experiment_name()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/training_log_{experiment_name}_parallel.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_single_episode(episode_num, worker_id, sumo_port, shared_ddqn_dict, predictor_path=None):
    """
    Run a single episode in a separate process.
    
    Args:
        episode_num: Episode number for logging
        worker_id: Unique worker ID for this process
        sumo_port: SUMO port number for this worker
        shared_ddqn_dict: Shared state dict for DDQN agent
        predictor_path: Path to predictor model (if using predictive caching)
    
    Returns:
        Tuple of (episode_num, experiences, statistics, maddpg_experiences)
    """
    try:
        # Set process-specific logging
        process_logger = logging.getLogger(f"Worker-{worker_id}")
        
        # Initialize environment with unique SUMO port
        env = VECNEnvironment(sumo_port=sumo_port)
        
        # Create local DDQN agent and load shared weights
        ddqn_agent = DDQNAgent(state_dim=DDQN_STATE_DIM, action_space=DDQN_ACTION_SPACE)
        if shared_ddqn_dict is not None:
            ddqn_agent.policy_net.load_state_dict(shared_ddqn_dict)
        
        # Load predictor if needed
        predictor = None
        if USE_PREDICTIVE_CACHING and predictor_path and os.path.exists(predictor_path):
            predictor = LSTMCachePredictor(
                input_dim=PREDICTION_INPUT_DIM,
                hidden_dim=PREDICTION_HIDDEN_DIM,
                num_layers=PREDICTION_NUM_LAYERS,
                num_services=PREDICTION_NUM_SERVICES
            )
            predictor.load_state_dict(torch.load(predictor_path, map_location=DEVICE))
            predictor.eval()
            env.set_predictor(predictor)
        
        # A. Outer Loop (DDQN): Choose Num UAVs
        outer_state = env.get_ddqn_state()
        num_uavs = ddqn_agent.select_action(outer_state) + 1
        
        # B. Reset Environment
        inner_states = env.reset(num_uavs=num_uavs)
        
        # Tracking for this episode
        episode_reward = 0
        episode_profit = 0
        ddqn_experience = None
        maddpg_experiences = []
        
        # C. Inner Loop (MADDPG)
        if num_uavs > 0:
            maddpg_controller = MADDPGController(
                num_agents=num_uavs,
                state_dim=MADDPG_STATE_DIM,
                action_dim=MADDPG_ACTION_DIM
            )
            
            for step in range(INNER_STEPS):
                # Select actions
                actions = []
                for i in range(num_uavs):
                    action = maddpg_controller.agents[i].select_action(inner_states[i])
                    actions.append(action)
                
                # Step environment
                next_states, rewards, done = env.step(actions)
                
                # Store experience
                maddpg_controller.memory.add(inner_states, actions, rewards, next_states, done)
                maddpg_experiences.append({
                    'states': inner_states.copy(),
                    'actions': actions.copy(),
                    'rewards': rewards.copy(),
                    'next_states': next_states.copy(),
                    'done': done
                })
                
                episode_reward += np.mean(rewards)
                inner_states = next_states
                
                if done:
                    break
        else:
            # No UAVs: Fast-forward through episode
            for _ in range(INNER_STEPS):
                env.step([])
        
        # E. Outer Loop Experience
        next_outer_state = env.get_ddqn_state()
        outer_reward = next_outer_state[3]  # Net Profit
        episode_profit = outer_reward
        
        ddqn_experience = {
            'state': outer_state,
            'action': num_uavs - 1,
            'reward': outer_reward,
            'next_state': next_outer_state,
            'done': False
        }
        
        # F. Collect statistics
        stats = env.get_episode_statistics()
        stats['episode_reward'] = episode_reward
        stats['episode_profit'] = episode_profit
        stats['num_uavs'] = num_uavs
        
        # Cleanup
        env.close()
        
        # Return results (logging happens in main process)
        return episode_num, ddqn_experience, stats, maddpg_experiences
        
    except Exception as e:
        logger.error(f"Error in episode {episode_num} (Worker {worker_id}): {e}")
        import traceback
        traceback.print_exc()
        return episode_num, None, None, None


def run_parallel_training():
    """Main parallel training loop."""
    logger.info(f"--- STARTING PARALLEL TRAINING ---")
    logger.info(f"Mode: {SIMULATION_MODE} | Caching: {'PREDICTIVE' if USE_PREDICTIVE_CACHING else 'REACTIVE'}")
    
    # Determine number of workers
    # Note: Due to TraCI global state limitations, we run episodes sequentially
    # but with proper SUMO cleanup to prevent the ~600 episode crash
    num_workers = 1  # Sequential execution for SUMO stability
    logger.info(f"Running in sequential mode with proper SUMO cleanup (fixes 600-episode issue)")
    
    # Determine logging interval
    if TOTAL_EPISODES >= 100:
        log_interval = 100
    elif TOTAL_EPISODES >= 10:
        log_interval = 10
    else:
        log_interval = 1
    
    logger.info(f"Total Episodes: {TOTAL_EPISODES} | Logging every {log_interval} episodes")
    
    # Setup TensorBoard and save paths
    session_save_path = os.path.join(MODEL_SAVE_PATH, f"experiment_{experiment_name}")
    os.makedirs(session_save_path, exist_ok=True)
    logger.info(f"Models will be saved to: {session_save_path}")
    writer = SummaryWriter(f"runs/experiment_{experiment_name}")
    
    # Initialize master DDQN agent (runs in main process)
    ddqn_agent = DDQNAgent(state_dim=DDQN_STATE_DIM, action_space=DDQN_ACTION_SPACE)
    
    # Predictor path
    predictor_path = "models/lstm_cache_predictor.pth" if USE_PREDICTIVE_CACHING else None
    
    # Tracking
    scores_window = deque(maxlen=100)
    start_time = time.time()
    
    # MADDPG controllers for different UAV counts (reused across episodes)
    maddpg_controllers = {}
    
    # Set multiprocessing start method
    try:
        torch_mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Create process pool
    pool = mp.Pool(processes=num_workers)
    
    try:
        # Process episodes in batches
        episode = 1
        while episode <= TOTAL_EPISODES:
            batch_size = min(num_workers, TOTAL_EPISODES - episode + 1)
            batch_episodes = list(range(episode, episode + batch_size))
            
            # Get current shared DDQN state
            shared_state_dict = ddqn_agent.policy_net.state_dict()
            
            # Assign unique SUMO ports to each worker
            worker_args = []
            for i, ep in enumerate(batch_episodes):
                worker_id = i
                sumo_port = 8813 + worker_id  # Base port + worker offset
                worker_args.append((ep, worker_id, sumo_port, shared_state_dict, predictor_path))
            
            # Run batch of episodes in parallel
            batch_start_time = time.time()
            results = pool.starmap(run_single_episode, worker_args)
            batch_time = time.time() - batch_start_time
            
            # Process results
            for ep_num, ddqn_exp, stats, maddpg_exps in results:
                if ddqn_exp is None:
                    logger.warning(f"Episode {ep_num} failed, skipping...")
                    continue
                
                # Add DDQN experience to replay buffer
                ddqn_agent.memory.add(
                    ddqn_exp['state'],
                    ddqn_exp['action'],
                    ddqn_exp['reward'],
                    ddqn_exp['next_state'],
                    ddqn_exp['done']
                )
                
                # Update MADDPG controller if we have experiences
                if maddpg_exps and len(maddpg_exps) > 0:
                    num_uavs = stats['num_uavs']
                    
                    # Create/reuse MADDPG controller for this UAV count
                    if num_uavs not in maddpg_controllers:
                        maddpg_controllers[num_uavs] = MADDPGController(
                            num_agents=num_uavs,
                            state_dim=MADDPG_STATE_DIM,
                            action_dim=MADDPG_ACTION_DIM
                        )
                    
                    controller = maddpg_controllers[num_uavs]
                    
                    # Add experiences to MADDPG replay buffer
                    for exp in maddpg_exps:
                        controller.memory.add(
                            exp['states'],
                            exp['actions'],
                            exp['rewards'],
                            exp['next_states'],
                            exp['done']
                        )
                
                # Update tracking
                scores_window.append(stats['episode_profit'])
                avg_score = np.mean(scores_window) if scores_window else 0
                
                # TensorBoard logging (matching main.py)
                writer.add_scalar('Profit/Episode', stats['episode_profit'], ep_num)
                writer.add_scalar('Profit/Average_100', avg_score, ep_num)
                writer.add_scalar('System/UAVs_Deployed', stats['num_uavs'], ep_num)
                
                # Time metrics
                elapsed_time = time.time() - start_time
                avg_time_per_episode = elapsed_time / ep_num if ep_num > 0 else 0
                remaining_episodes = TOTAL_EPISODES - ep_num
                estimated_time_left = avg_time_per_episode * remaining_episodes
                
                writer.add_scalar('Time/Elapsed_Seconds', elapsed_time, ep_num)
                writer.add_scalar('Time/Estimated_Remaining_Seconds', estimated_time_left, ep_num)
                writer.add_scalar('Time/Average_Seconds_Per_Episode', avg_time_per_episode, ep_num)
                
                # Detailed stats from environment (matching main.py)
                for k, v in stats.items():
                    if k not in ['episode_profit', 'episode_reward', 'num_uavs']:  # Avoid duplicates
                        writer.add_scalar(k, v, ep_num)
                
                # Periodic logging
                if ep_num % log_interval == 0 or ep_num == 1:
                    elapsed_time = time.time() - start_time
                    avg_time_per_episode = elapsed_time / ep_num if ep_num > 0 else 0
                    remaining_episodes = TOTAL_EPISODES - ep_num
                    estimated_time_left = avg_time_per_episode * remaining_episodes
                    
                    # Format time strings for console output
                    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                    remaining_str = time.strftime("%H:%M:%S", time.gmtime(estimated_time_left))
                    
                    print(
                        f"Episode {ep_num}/{TOTAL_EPISODES} | Avg Profit: {avg_score:.2f} | "
                        f"UAVs: {stats['num_uavs']} | ε: {ddqn_agent.epsilon:.2f} | "
                        f"Elapsed: {elapsed_str} | ETA: {remaining_str}"
                    )
            
            # Learn from experiences (after each batch)
            ddqn_loss = ddqn_agent.learn()
            ddqn_agent.update_target_network()
            
            # Learn MADDPG (for each controller that has enough experiences)
            for num_uavs, controller in maddpg_controllers.items():
                if len(controller.memory) >= MADDPG_BATCH_SIZE:
                    c_loss, a_loss = controller.learn()
                    controller.update_targets()
            
            # Update epsilon decay
            if ddqn_agent.epsilon > DDQN_EPSILON_END:
                ddqn_agent.epsilon *= DDQN_EPSILON_DECAY
            
            # Periodic model saving
            if episode % 100 == 0:
                ddqn_agent.save(session_save_path)
                for num_uavs, controller in maddpg_controllers.items():
                    controller.save(session_save_path, suffix=f"_{num_uavs}uavs")
                logger.info(f"Checkpoint saved at episode {episode}")
            
            episode += batch_size
        
        # Final cleanup
        pool.close()
        pool.join()
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        pool.terminate()
        pool.join()
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        pool.terminate()
        pool.join()
    
    # Save final models
    writer.close()
    logger.info("Training Complete.")
    print("\n--- Saving Final Models ---")
    ddqn_agent.save(session_save_path)
    
    for num_uavs, controller in maddpg_controllers.items():
        controller.save(session_save_path, suffix=f"_{num_uavs}uavs")
    
    total_time = time.time() - start_time
    logger.info(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    logger.info(f"Average episodes per second: {TOTAL_EPISODES/total_time:.2f}")


if __name__ == "__main__":
    # Ensure proper multiprocessing context
    mp.set_start_method('spawn', force=True)
    run_parallel_training()
