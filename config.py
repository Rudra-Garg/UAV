# config.py
"""
Centralized configuration file for simulation parameters.
CORRECTED for Phase 4: The MADDPG_STATE_DIM is now correctly set to 7 to
account for the addition of the UAV status feature.
"""

import torch

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Simulation Area and Time ---
AREA_WIDTH = 10000
AREA_HEIGHT = 10000

# --- UAV Parameters (From Paper Section V & Table II) ---
UAV_ALTITUDE = 50
UAV_COMMUNICATION_RANGE = 1000
MIN_UAV_DISTANCE = 500
UAV_MAX_SPEED = 50
UAV_COMPUTATIONAL_RESOURCES = 2.25e6
UAV_CACHE_SIZE = 3

# --- Vehicle Parameters (From Paper Section V-A) ---
NUM_VEHICLES = 100
VEHICLE_MIN_SPEED = 1.5
VEHICLE_MAX_SPEED = 3.0
TASKS_PER_VEHICLE = 6
USE_DYNAMIC_DEMAND = True       # Master switch to enable/disable this feature
CONGESTION_NUM_ZONES = 3        # Number of "congestion" clusters to create
TASKS_PER_VEHICLE_CONGESTED = 8 # Number of tasks for vehicles inside a congestion zone

# --- Task Parameters (From Paper Section V-A) ---
TASK_DATA_SIZE_RANGE = (50, 100)  # Mbit
NUM_SERVICE_TYPES = 100
TASK_CPU_CYCLES_PER_BIT = 0.5
LATENCY_CONSTRAINT_RANGE = (40.0, 60.0)

# --- Economic & Reward Parameters ---
BETA_COMPUTATION = 1e-12
BETA_MAINTENANCE = 100.0
DELTA_LATENCY = 10.0
DELTA_SIZE = 2.0
DELTA_COMPUTATION = 0.5
REWARD_SCALING_FACTOR = 1.0

# --- Energy Parameters (from Phase 2) ---
UAV_ENERGY_CAPACITY_JOULES = (800000.0, 1000000.0)
ENERGY_HOVER_WATT = 200.0
ENERGY_COMPUTATION_JOULE_PER_GCYCLE = 10e-9
ENERGY_COMM_JOULE_PER_MBIT = 0.5
ENERGY_REWARD_PENALTY = 0.0001

# --- Dynamic Caching Parameters (from Phase 3) ---
CACHE_UPDATE_PROBABILITY = 0.2  # 20% chance

# --- Communication & Power (From Paper Table II) ---
BANDWIDTH_UAV_USER = 2e6
BANDWIDTH_UAV_UAV = 3e6
BANDWIDTH_UAV_CCC = 20e6
NOISE_POWER_SPECTRAL_DENSITY = -96
CARRIER_FREQUENCY = 2e9
POWER_UAV_USER = 0.5
POWER_UAV_UAV = 0.7
POWER_CCC = 5
ETA_LOS = 1.8
ETA_NLOS = 30
USE_ENERGY_PENALTY = False  # If True, subtracts energy cost from profit in reward calculation.
USE_UAV_STATUS = False      # If True, includes a UAV's IDLE/BUSY status in its MADDPG state.


# --- Path Loss Parameters (Unchanged) ---
C = 3e8
LOS_X0 = 11.9
LOS_Y0 = 0.13

# --- HRL Training Parameters ---
TOTAL_EPISODES = 1000
INNER_STEPS = 100

# --- DDQN (Outer Layer) Parameters ---
DDQN_ACTION_SPACE = 15
DDQN_STATE_DIM = 6
DDQN_LEARNING_RATE = 0.0005
DDQN_BUFFER_SIZE = 50000
DDQN_BATCH_SIZE = 64
DDQN_GAMMA = 0.95
DDQN_EPSILON_START = 0.1
DDQN_EPSILON_END = 0.01
DDQN_EPSILON_DECAY = 0.995
DDQN_TAU = 0.005

# --- MADDPG (Inner Layer) Parameters ---
# Phase 4 CORRECTION: State dimension increased by 1 to include UAV status
MADDPG_STATE_DIM = 7 if USE_UAV_STATUS else 6
MADDPG_ACTION_DIM = 2
MADDPG_LEARNING_RATE_ACTOR = 0.0005
MADDPG_LEARNING_RATE_CRITIC = 0.0005
MADDPG_BUFFER_SIZE = 100000
MADDPG_BATCH_SIZE = 128
MADDPG_GAMMA = 0.95
MADDPG_TAU = 0.01


MAX_HOPS = 2
DYNAMIC_BANDWIDTH = True            # Enable/disable the TDMA bandwidth sharing model
TDMA_SLOTS_PER_STEP = 10

# --- VISUALIZATION & EVALUATION ---
VISUALIZATION = False
MODEL_SAVE_PATH = "models/"
EVAL_EPISODES = 10
EVAL_SCENARIO_VEHICLES = range(50, 121, 10)
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 800
# --- Hybrid Caching Parameters---
SERVICE_CACHE_SIZE = 2             # Max number of services a UAV can cache
CONTENT_CACHE_SIZE = 1             # Max number of content items a UAV can cache
NUM_CONTENT_TYPES = 50             # Number of distinct content types in the simulation
POPULARITY_ZIPF_ALPHA = 1.2        # Alpha parameter for Zipf distribution for popularity
