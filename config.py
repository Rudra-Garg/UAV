# config.py
"""
Centralized configuration file for simulation parameters.
Values are based on Table II and the Parameter Setting section of the paper.
"""

import torch

# --- Device Configuration ---
# Automatically select CUDA if available, otherwise fall back to CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Simulation Area and Time ---
AREA_WIDTH = 10000  # meters (10km)
AREA_HEIGHT = 10000  # meters (10km)

# --- UAV Parameters ---
UAV_ALTITUDE = 50  # meters
UAV_COMMUNICATION_RANGE = 1000  # meters
UAV_CACHE_CAPACITY = 3e4  # MB (3 * 10^4 MB)
MIN_UAV_DISTANCE = 500  # meters for safety constraint
UAV_MAX_SPEED = 50  # meters per step, used to scale the RL agent's action output

# --- Vehicle Parameters ---
NUM_VEHICLES = 100  # Default number of vehicles for training
VEHICLE_MIN_SPEED = 1.5  # m/s
VEHICLE_MAX_SPEED = 3.0  # m/s
TASKS_PER_VEHICLE = 6

# --- Task Parameters ---
TASK_DATA_SIZE_RANGE = (50, 100)  # Mbit
TASK_CPU_CYCLES_RANGE = (1e9, 5e9)  # Example range, not specified in paper
LATENCY_CONSTRAINT_RANGE = (0.5, 2.0)  # seconds
NUM_TASK_TYPES = 100  # Paper: 100 types of tasks, services, content

# --- Communication Channel Parameters (from Table II) ---
BANDWIDTH_UAV_USER = 2e6  # Hz (2 MHz)
BANDWIDTH_UAV_UAV = 3e6  # Hz (3 MHz)
BANDWIDTH_UAV_CCC = 20e6  # Hz (20 MHz)
NOISE_POWER_SPECTRAL_DENSITY = -96  # dBm/Hz -> -126 dBW/Hz
CARRIER_FREQUENCY = 2e9  # Hz (2 GHz) for path loss calculation

# --- Power Parameters (from Table II) ---
POWER_UAV_USER = 0.5  # Watt (using average for simplicity)
POWER_UAV_UAV = 0.7  # Watt
POWER_CCC = 5  # Watt

# --- Path Loss Parameters ---
C = 3e8  # Speed of light
LOS_X0 = 11.95
LOS_Y0 = 0.136
ETA_LOS = 1.8  # dB
ETA_NLOS = 30  # dB

# --- HRL Training Parameters ---
TOTAL_EPISODES = 1000  # Outer loop episodes for the entire training run
INNER_STEPS = 100  # Inner loop (simulation) steps per episode

# --- DDQN (Outer Layer) Parameters ---
DDQN_ACTION_SPACE = 15  # Max number of UAVs to deploy, e.g., 1 to 15
DDQN_STATE_DIM = 6  # [tasks_completed, num_uavs, total_cost, total_profit, total_latency, users_covered]
DDQN_LEARNING_RATE = 0.0005
DDQN_BUFFER_SIZE = 50000
DDQN_BATCH_SIZE = 64
DDQN_GAMMA = 0.99  # Discount factor
DDQN_EPSILON_START = 1.0  # Starting value for exploration rate
DDQN_EPSILON_END = 0.01  # Minimum value for exploration rate
DDQN_EPSILON_DECAY = 0.995  # Multiplicative factor for annealing epsilon
DDQN_TAU = 0.005  # For soft target network updates

# --- MADDPG (Inner Layer) Parameters ---
# Per-agent state: [norm_pos_x, norm_pos_y, users_in_range, tasks_processed (placeholder), profit (placeholder)]
MADDPG_STATE_DIM = 5
MADDPG_ACTION_DIM = 2  # [move_x, move_y]
MADDPG_LEARNING_RATE_ACTOR = 0.0001
MADDPG_LEARNING_RATE_CRITIC = 0.001
MADDPG_BUFFER_SIZE = 100000
MADDPG_BATCH_SIZE = 128
MADDPG_GAMMA = 0.95  # Discount factor
MADDPG_TAU = 0.01  # For soft target network updates

# --- VISUALIZATION PARAMETERS ---
VISUALIZATION = False  # Set to True to show the simulation, False to run headless for speed
SCREEN_WIDTH = 800  # pixels
SCREEN_HEIGHT = 800  # pixels

# --- MODEL SAVING AND EVALUATION ---
MODEL_SAVE_PATH = "models/"  # Directory to save/load trained models
EVAL_EPISODES = 10  # Number of episodes to average over for each data point in evaluation
EVAL_SCENARIO_VEHICLES = range(50, 121, 10)  # Vehicle counts to test: 50, 60, ..., 120
