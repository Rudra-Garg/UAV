# main.py
import datetime
import logging
import time
from collections import deque

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agents import DDQNAgent, MADDPGController
from config import *
from simulation import VECNEnvironment

# Conditional Imports
if USE_PREDICTIVE_CACHING:
    from prediction import LSTMCachePredictor

if VISUALIZATION and SIMULATION_MODE == 'PYTHON_KINEMATIC':
    from visualization import Visualizer

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_training():
    logger.info(f"--- STARTING TRAINING ---")
    logger.info(f"Mode: {SIMULATION_MODE} | Caching: {'PREDICTIVE' if USE_PREDICTIVE_CACHING else 'REACTIVE'}")

    # 1. Setup TensorBoard
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f"runs/experiment_{timestamp}")

    # 2. Initialize Environment & Agents
    env = VECNEnvironment()
    ddqn_agent = DDQNAgent(state_dim=DDQN_STATE_DIM, action_space=DDQN_ACTION_SPACE)

    # 3. Initialize Predictor (Optional)
    predictor = None
    if USE_PREDICTIVE_CACHING:
        model_path = "lstm_cache_predictor.pth"  # Ensure this exists (trained via train_cache_predictor.py)
        if os.path.exists(model_path):
            predictor = LSTMCachePredictor().to(DEVICE)
            predictor.load_state_dict(torch.load(model_path, map_location=DEVICE))
            predictor.eval()
            logger.info("✅ LSTM Predictor loaded successfully.")
        else:
            logger.warning(f"⚠️ Predictor model not found at {model_path}. Caching will fallback to random/reactive.")

    # 4. Initialize Visualizer (Only for Python Mode)
    visualizer = None
    if VISUALIZATION and SIMULATION_MODE == 'PYTHON_KINEMATIC':
        visualizer = Visualizer(env.width, env.height)

    # Tracking
    scores_window = deque(maxlen=100)
    maddpg_controller = None

    start_time = time.time()

    # --- MAIN LOOP ---
    for episode in range(1, TOTAL_EPISODES + 1):

        # A. Outer Loop (DDQN): Choose Num UAVs
        outer_state = env.get_ddqn_state()
        num_uavs = ddqn_agent.select_action(outer_state) + 1

        # B. Reset Environment
        # env.reset() handles vehicle spawning (via SUMO or Python) and task generation
        inner_states = env.reset(num_uavs=num_uavs)

        # C. Initialize Inner Agent (MADDPG) if needed
        if num_uavs > 0:
            # Ideally, we should persist controllers for specific K, 
            # but for standard implementation we re-init or load from a bank.
            # Here we re-init for simplicity of the snippet.
            maddpg_controller = MADDPGController(num_agents=num_uavs,
                                                 state_dim=MADDPG_STATE_DIM,
                                                 action_dim=MADDPG_ACTION_DIM)

            episode_actor_loss = []
            episode_critic_loss = []

            # D. Inner Loop (Steps)
            for t in range(INNER_STEPS):

                # --- PREDICTIVE CACHING UPDATE ---
                if predictor and (t % CACHE_UPDATE_INTERVAL == 0):
                    recent_data = env.get_recent_requests(PREDICTION_SEQUENCE_LENGTH)
                    if len(recent_data) >= PREDICTION_SEQUENCE_LENGTH:
                        # Prepare batch (size 1)
                        s_seq = torch.LongTensor([[x[0] for x in recent_data]]).to(DEVICE)
                        z_seq = torch.LongTensor([[x[1] for x in recent_data]]).to(DEVICE)

                        with torch.no_grad():
                            s_logits, c_logits = predictor(s_seq, z_seq)

                        # Top-K
                        s_probs = torch.softmax(s_logits, dim=1).cpu().numpy().flatten()
                        c_probs = torch.softmax(c_logits, dim=1).cpu().numpy().flatten()

                        top_s = s_probs.argsort()[-SERVICE_CACHE_SIZE:][::-1]
                        top_c = c_probs.argsort()[-CONTENT_CACHE_SIZE:][::-1]

                        for uav in env.uavs:
                            uav.update_cache_from_prediction(top_s, top_c)

                # --- RL STEP ---
                actions = maddpg_controller.select_actions(inner_states)
                next_inner_states, rewards, done = env.step(actions)

                # Store experience
                if len(inner_states) > 0:
                    flat_s = np.concatenate(inner_states)
                    flat_a = np.concatenate(actions)
                    flat_ns = np.concatenate(next_inner_states)
                    maddpg_controller.memory.add(flat_s, flat_a, rewards[0], flat_ns, done)

                # Learn
                c_loss, a_loss = maddpg_controller.learn()
                if c_loss is not None:
                    episode_critic_loss.append(c_loss)
                    episode_actor_loss.append(a_loss)
                    maddpg_controller.update_targets()

                inner_states = next_inner_states

                # Visualization
                if visualizer:
                    current_profit = rewards[0] if rewards else 0
                    visualizer.draw(env.uavs, env.vehicles, episode, t, current_profit)

                if done:
                    break

        # E. Outer Loop Learning
        next_outer_state = env.get_ddqn_state()
        outer_reward = next_outer_state[3]  # Net Profit

        ddqn_agent.memory.add(outer_state, num_uavs - 1, outer_reward, next_outer_state, False)
        ddqn_loss = ddqn_agent.learn()
        ddqn_agent.update_target_network()

        scores_window.append(outer_reward)
        avg_score = np.mean(scores_window)

        # F. Logging
        writer.add_scalar('Profit/Episode', outer_reward, episode)
        writer.add_scalar('Profit/Average_100', avg_score, episode)
        writer.add_scalar('System/UAVs_Deployed', num_uavs, episode)

        # Detailed stats from environment
        stats = env.get_episode_statistics()
        for k, v in stats.items():
            writer.add_scalar(k, v, episode)

        print(
            f"\rEpisode {episode}/{TOTAL_EPISODES} | Avg Profit: {avg_score:.2f} | UAVs: {num_uavs} | ε: {ddqn_agent.epsilon:.2f}",
            end="")

        if episode % 100 == 0:
            logger.info(f"Ep {episode} Summary: Profit={avg_score:.2f}, UAVs={num_uavs}")
            # Save Checkpoints
            ddqn_agent.save(MODEL_SAVE_PATH)

    # Cleanup
    env.close()
    if visualizer: visualizer.close()
    writer.close()
    logger.info("Training Complete.")


if __name__ == "__main__":
    run_training()
