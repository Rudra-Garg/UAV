# config.py
"""
=========================================================================================
CENTRALIZED CONFIGURATION FILE (UNIFIED)
Matches logic for:
1. Physical Simulation (SUMO vs. Kinematic Python)
2. Task Generation (Azure Traces / Workflows)
3. Caching (LSTM Predictive vs. Reactive)
4. RL Agents (DDQN + MADDPG)
=========================================================================================
"""
import os

import torch

# ========================================================================================
# A. GLOBAL MODES & SWITCHES
# ========================================================================================

# --- Simulation Mode ---
# Options: 'SUMO', 'PYTHON_KINEMATIC'
SIMULATION_MODE = 'SUMO'

# --- Caching Mode ---
# If True, uses the LSTM model to pre-fetch content.
USE_PREDICTIVE_CACHING = False

# --- Hardware Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Visualization ---
VISUALIZATION = False
VISUALIZER_STAYS_OPEN = False
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
SAVE_VISUALIZATION_IMAGES = True

# ========================================================================================
# B. CORE SIMULATION SETTINGS
# ========================================================================================

AREA_WIDTH = 10000
AREA_HEIGHT = 10000
TOTAL_EPISODES = 1000
INNER_STEPS = 100

# ========================================================================================
# C. ENTITY PARAMETERS (UAVs & Vehicles)
# ========================================================================================

# --- UAV Fleet ---
UAV_ALTITUDE = 50
UAV_COMMUNICATION_RANGE = 1000
MIN_UAV_DISTANCE = 500
UAV_MAX_SPEED = 50
UAV_COMPUTATIONAL_RESOURCES = 2.25e9
UAV_ENERGY_CAPACITY_JOULES = (800000.0, 1000000.0)

# --- Vehicle Fleet ---
NUM_VEHICLES = 200  # Default target number of vehicles
VEHICLE_MIN_SPEED = 1.5
VEHICLE_MAX_SPEED = 3.0

# ========================================================================================
# D. DATA & DEMAND GENERATION (AZURE TRACES)
# ========================================================================================

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

AZURE_MATRIX_FILE = os.path.join(DATA_DIR, 'azure_workload_matrix.npy')
AZURE_META_FILE = os.path.join(DATA_DIR, 'azure_service_meta.csv')
TASK_REQUEST_DATA_FILE = os.path.join(DATA_DIR, 'task_request_data.csv')

# Task constraints
TASKS_PER_VEHICLE = 6
USE_DYNAMIC_DEMAND = True
TASKS_PER_VEHICLE_CONGESTED = 8
TASK_DATA_SIZE_RANGE = (50, 100)  # Mbits
TASK_CPU_CYCLES_PER_BIT = 0.5
LATENCY_CONSTRAINT_RANGE = (40.0, 60.0)  # Steps

# ========================================================================================
# E. TRAFFIC & GEOGRAPHY
# ========================================================================================

TRAFFIC_SCENARIOS = {
    'SCENARIO_NAMES': ['UNIFORM', 'SINGLE_CONGESTION', 'MULTI_CONGESTION'],
    'SCENARIO_WEIGHTS': [0.2, 0.4, 0.4],
    'SCENARIOS': {
        'UNIFORM': {
            'num_hotspots': 0, 'hotspot_radius': 0, 'hotspot_ratio': 0.0
        },
        'SINGLE_CONGESTION': {
            'num_hotspots': 1, 'hotspot_radius': 2000, 'hotspot_ratio': 0.7
        },
        'MULTI_CONGESTION': {
            'num_hotspots': 3, 'hotspot_radius': 1500, 'hotspot_ratio': 0.8
        }
    }
}

# Real-world SUMO maps
SUMO_SCENARIO_POOL = [
    'delhi', 'mumbai', 'guwahati', 'bangaluru', 'paris', 'london', 'nyc', 'tokyo'
]

# ========================================================================================
# F. CACHING & PREDICTION PARAMETERS
# ========================================================================================

NUM_SERVICE_TYPES = 20
NUM_CONTENT_TYPES = 50
NUM_ZONES = 4  # Spatial zones for the LSTM context

# Cache sizes
SERVICE_CACHE_SIZE = 10
CONTENT_CACHE_SIZE = 20
POPULARITY_ZIPF_ALPHA = 1.2

# LSTM Parameters
PREDICTION_SEQUENCE_LENGTH = 20
CACHE_UPDATE_INTERVAL = 20
CACHE_UPDATE_PROBABILITY = 0.2

# ========================================================================================
# G. COMMUNICATION & ENERGY MODELS
# ========================================================================================

# Bandwidth & Power
DYNAMIC_BANDWIDTH = True
TDMA_SLOTS_PER_STEP = 10
MAX_HOPS = 2
BANDWIDTH_UAV_USER = 2e6
BANDWIDTH_UAV_UAV = 3e6
BANDWIDTH_UAV_CCC = 20e6
POWER_UAV_USER = 0.5
POWER_UAV_UAV = 0.7
POWER_CCC = 5
NOISE_POWER_SPECTRAL_DENSITY = -96
CARRIER_FREQUENCY = 2e9

# Path Loss
ETA_LOS = 1.8
ETA_NLOS = 30
LOS_X0 = 11.9
LOS_Y0 = 0.13
C = 3e8

# Energy
USE_ENERGY_PENALTY = True
ENERGY_REWARD_PENALTY = 0.0001
ENERGY_HOVER_WATT = 200.0
ENERGY_COMPUTATION_JOULE_PER_GCYCLE = 10e-9
ENERGY_COMM_JOULE_PER_MBIT = 0.5

# ========================================================================================
# H. REWARDS & COSTS
# ========================================================================================

BETA_MAINTENANCE = 10.0
BETA_COMPUTATION = 1e-7
DELTA_LATENCY = 5.0
DELTA_SIZE = 2.0
DELTA_COMPUTATION = 0.5
REWARD_SCALING_FACTOR = 1.0

# ========================================================================================
# I. RL HYPERPARAMETERS
# ========================================================================================

# DDQN (Outer Loop)
DDQN_ACTION_SPACE = 50
DDQN_STATE_DIM = 6
DDQN_LEARNING_RATE = 0.0005
DDQN_BUFFER_SIZE = 50000
DDQN_BATCH_SIZE = 64
DDQN_GAMMA = 0.95
DDQN_EPSILON_START = 0.9
DDQN_EPSILON_END = 0.01
DDQN_EPSILON_DECAY = 0.995
DDQN_TAU = 0.005

# MADDPG (Inner Loop)
USE_UAV_STATUS = True
MADDPG_STATE_DIM = 7 if USE_UAV_STATUS else 6
MADDPG_ACTION_DIM = 2
MADDPG_LEARNING_RATE_ACTOR = 0.0005
MADDPG_LEARNING_RATE_CRITIC = 0.0005
MADDPG_BUFFER_SIZE = 100000
MADDPG_BATCH_SIZE = 128
MADDPG_GAMMA = 0.95
MADDPG_TAU = 0.01

# ========================================================================================
# J. I/O & EVALUATION SETTINGS
# ========================================================================================

def get_experiment_name():
    """Generate experiment name with simulation mode, caching mode, and timestamp."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_mode = "PRED" if USE_PREDICTIVE_CACHING else "REAC"
    sim_mode = SIMULATION_MODE.replace('_', '').lower()  # e.g., 'sumo' or 'pythonkinematic'
    return f"{sim_mode}_{cache_mode}_{timestamp}"

MODEL_SAVE_PATH = "models/"
IMAGE_SAVE_PATH = "visualization_snapshots/"
EPISODES_TO_SNAPSHOT = [1, 250, 500, 750, 1000]
STEPS_TO_SNAPSHOT = [1, 25, 50, 75, 100]

# --- Evaluation Params ---
EVAL_EPISODES = 5
EVAL_SCENARIO_VEHICLES = range(50, 121, 10)  # [50, 60, ..., 120]

# Offloading Logic
USE_SIMPLIFIED_OFFLOADING = True
LATENCY_SCALING_FACTOR = 0.000001
CLOUD_COMPUTE_LATENCY = 30
