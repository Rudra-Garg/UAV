# config.py
"""
Centralized configuration file for simulation parameters.
This version adheres as strictly as possible to the values and models
described in the research paper. All assumptions are clearly noted.
"""

import torch

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Simulation Area and Time ---
AREA_WIDTH = 10000
AREA_HEIGHT = 10000

# --- UAV Parameters (From Paper Section V & Table II) ---
UAV_ALTITUDE = 50  # From Section V-A
UAV_COMMUNICATION_RANGE = 1000  # From Section V-A
MIN_UAV_DISTANCE = 500  # From Section V-A, D_min
UAV_MAX_SPEED = 50  # General assumption for simulation dynamics
# --- From Table II, using lower-bound values ---
# NOTE: Paper specifies Fux in MHz, likely a typo for Gcycles/sec or similar.
# We interpret 2.2MHz as a proxy for 22 Gcycles/sec for a powerful edge node.
UAV_COMPUTATIONAL_RESOURCES = 22e9  # Based on F_ux range [2.2, 2.5] (interpreted as Gcycles/sec)
UAV_CACHE_SIZE = 10  # General assumption, as paper's P_ux is in GB.

# --- Vehicle Parameters (From Paper Section V-A) ---
NUM_VEHICLES = 100
VEHICLE_MIN_SPEED = 1.5
VEHICLE_MAX_SPEED = 3.0
TASKS_PER_VEHICLE = 6

# --- Task Parameters (From Paper Section V-A) ---
TASK_DATA_SIZE_RANGE = (50, 100)  # Mbit. FROM PAPER. This is the key high value.
NUM_SERVICE_TYPES = 100  # From Section V-A
# --- ASSUMPTION: The paper does not specify these crucial task parameters ---
# We assume a computational load that is challenging but possible for the UAVs.
TASK_CPU_CYCLES_PER_BIT = 200
# We MUST assume a high latency constraint to make the problem solvable with the large data sizes.
LATENCY_CONSTRAINT_RANGE = (10.0, 20.0)  # seconds.

# --- Economic & Energy Parameters ---
# --- ASSUMPTION: The paper defines the formulas but not the Beta/Delta weight values. ---
# These values are chosen to create a challenging but stable economic simulation.
# Cost function weights (Eq. 18)
BETA_COMPUTATION = 1e-12  # Must be very small as F_ux is very large
BETA_MAINTENANCE = 1000.0  # High fixed cost to deploy a UAV
# Gain function weights (Eq. 19 & 20)
DELTA_LATENCY = 5.0  # Prioritize completing tasks quickly
DELTA_SIZE = 0.5
DELTA_COMPUTATION = 0.5
# Other economic params
ENERGY_PER_SECOND_PROCESSING = 0.5  # gamma_0 in paper, an assumption
REWARD_SCALING_FACTOR = 1.0  # The economy is now defined by Beta/Delta, less scaling needed.

# --- Communication & Power (From Paper Table II) ---
BANDWIDTH_UAV_USER = 2e6
BANDWIDTH_UAV_UAV = 3e6
BANDWIDTH_UAV_CCC = 20e6
NOISE_POWER_SPECTRAL_DENSITY = -96
CARRIER_FREQUENCY = 2e9
POWER_UAV_USER = 0.5  # Average of range [0.1, 1.0] W
POWER_UAV_UAV = 0.7  # Average of range [0.4, 1.0] W
POWER_CCC = 5  # Average of range [2, 8] W
ETA_LOS = 1.8
ETA_NLOS = 30  # From Table II

# --- Path Loss Parameters (From Paper Table II) ---
C = 3e8
LOS_X0 = 11.95
LOS_Y0 = 0.136

# --- HRL Training Parameters ---
TOTAL_EPISODES = 1000
INNER_STEPS = 100

# --- DDQN (Outer Layer) Parameters ---
DDQN_ACTION_SPACE = 15
DDQN_STATE_DIM = 6
DDQN_LEARNING_RATE = 0.0005
DDQN_BUFFER_SIZE = 50000
DDQN_BATCH_SIZE = 64
DDQN_GAMMA = 0.99
DDQN_EPSILON_START = 1.0
DDQN_EPSILON_END = 0.01
DDQN_EPSILON_DECAY = 0.995
DDQN_TAU = 0.005

# --- MADDPG (Inner Layer) Parameters ---
MADDPG_STATE_DIM = 5
MADDPG_ACTION_DIM = 2
MADDPG_LEARNING_RATE_ACTOR = 0.0001
MADDPG_LEARNING_RATE_CRITIC = 0.001
MADDPG_BUFFER_SIZE = 100000
MADDPG_BATCH_SIZE = 128
MADDPG_GAMMA = 0.95
MADDPG_TAU = 0.01

# --- VISUALIZATION & EVALUATION ---
VISUALIZATION = False
MODEL_SAVE_PATH = "models/"
EVAL_EPISODES = 10
EVAL_SCENARIO_VEHICLES = range(50, 121, 10)
