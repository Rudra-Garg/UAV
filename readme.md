# MUCEDS: Multi-UAV Cost-Efficient Deployment Scheme

### Cost-Efficient Deployment and Predictive Caching Optimization in Multi-UAV-Assisted Vehicular Edge Networks using HRL and LSTM

---

## About This Project

**MUCEDS** (Multi-UAV Cost-Efficient Deployment Scheme) is a comprehensive research implementation accompanying the technical report:

> **"Cost-Efficient Deployment and Predictive Caching Optimization in Multi-UAV-Assisted Vehicular Edge Computing Networks using Hierarchical Reinforcement Learning and LSTM"**

This project presents a novel **dual-optimization framework** for Vehicular Edge Computing Networks (VECNs) that simultaneously addresses:

1. **Physical Layer Optimization** — Dynamic UAV fleet sizing and positioning using Hierarchical Reinforcement Learning (HRL)
2. **Logical Layer Optimization** — Intelligent content caching using Spatial-Temporal LSTM prediction

---

## Problem Statement

Modern vehicular networks face critical challenges:

- **Dynamic Traffic Patterns**: Vehicle density varies significantly across time and space
- **Limited UAV Resources**: Battery capacity, computational power, and storage are constrained
- **Latency Requirements**: Edge computing tasks have strict deadline constraints
- **Cost-Performance Trade-off**: Deploying more UAVs improves coverage but increases operational costs

MUCEDS addresses these challenges through a joint optimization approach that balances **service quality**, **energy efficiency**, and **operational costs**.

---

## Key Features

### Hierarchical Reinforcement Learning Framework

| Layer | Algorithm | Objective | Action Space |
|-------|-----------|-----------|--------------|
| **Outer Loop** | Double DQN (DDQN) | Fleet Size Optimization | Discrete: Number of UAVs (1-50) |
| **Inner Loop** | MADDPG | Position Optimization | Continuous: 2D velocity vectors |

*   **Outer Layer (DDQN):** Strategically optimizes the *number* of UAVs to deploy, balancing coverage vs. operational costs based on global network state.
*   **Inner Layer (MADDPG):** A multi-agent continuous control policy where each UAV independently optimizes its *position* to cover dynamic vehicle hotspots while maintaining coordination.

### Predictive Caching (Spatial-Temporal LSTM)

*   **Architecture**: Dual-embedding LSTM with service and zone context
*   **Input**: Sequence of `(Service_ID, Zone_ID)` pairs from request history
*   **Output**: Top-K predicted services and content for proactive caching
*   **Benefit**: Significantly increases Cache Hit Ratio compared to reactive (Zipf/LRU) baselines

### Dual Simulation Modes

| Mode | Use Case | Speed | Fidelity |
|------|----------|-------|----------|
| **SUMO** | Final evaluation, real-world validation | Slower | High (real road networks) |
| **Python Kinematic** | Prototyping, hyperparameter tuning | Fast | Medium (simplified physics) |

*   **SUMO Mode:** Uses TraCI to interface with real-world road networks from OpenStreetMap for 8 global cities: Delhi, Mumbai, Bangalore, Guwahati, Paris, London, NYC, Tokyo.
*   **Python Kinematic Mode:** Fast, lightweight simulation for algorithmic debugging and rapid experimentation.

### Additional Features

*   **Real-World Workload Integration:** Adapts the Azure Functions Trace 2019 dataset to simulate realistic edge computing task requests
*   **Interactive Dashboard:** Full-featured Streamlit web dashboard for training monitoring, TensorBoard visualization, and configuration management
*   **Comprehensive Baselines:** 5 benchmark algorithms for comparative evaluation
*   **Multi-City Support:** 8 real-world city networks from OpenStreetMap data

---

## Project Structure

```text
├── agents/                 # RL Implementations
│   ├── ddqn.py             # Outer Agent (Fleet Sizing)
│   ├── maddpg.py           # Inner Agents (Positioning)
│   └── networks.py         # PyTorch Neural Architectures
├── analysis/               # Evaluation Tools
│   ├── evaluate.py         # Comparative evaluation script
│   ├── run_sensitivity.py  # Economic parameter sensitivity analysis
│   └── benchmark_agents.py # Baselines (Random, K-Means, etc.)
├── config.py               # Central Configuration (Simulation, RL, Caching)
├── dashboard.py            # Streamlit Web Dashboard
├── data/                   # Workload Datasets (Azure Traces)
├── main.py                 # Main Training Entry Point
├── main_parallel.py        # Multi-core Training Script
├── models/                 # Saved Model Checkpoints (.pth)
├── prediction/             # Caching Logic
│   ├── model.py            # LSTM Architecture
│   └── generator.py        # Task Demand Generator
├── scripts/                # Helper Bash Scripts (Start/Stop services)
├── simulation/             # Environment Logic
│   ├── environment.py      # OpenAI Gym-style Wrapper
│   ├── physics.py          # Bridge to SUMO and Python Physics
│   └── tasks.py            # Task Offloading & Lifecycle Manager
├── sumo_scenario/          # SUMO Network & Route Files
├── tools/                  # Data Preprocessing Scripts
└── visualization/          # Pygame Renderer
```

---

## Installation

### 1. Prerequisites
*   Python 3.8+
*   [Eclipse SUMO](https://eclipse.dev/sumo/) (Optional, required only if using `SIMULATION_MODE='SUMO'`)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
If using SUMO, ensure the `SUMO_HOME` environment variable is set.
*   **Linux:** `export SUMO_HOME=/usr/share/sumo`
*   **Windows:** `set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo`

---

## Data Pipeline Setup

Before training, you must generate the necessary traffic and task data.

**1. Generate Traffic Scenarios (SUMO)**
Converts OpenStreetMap data in `osm_data/` to SUMO network files.
```bash
python tools/generate_traffic.py
```

**2. Process Azure Workloads**
Processes raw Azure traces. *Note: If raw data is missing, the system falls back to synthetic generation.*
```bash
python tools/process_azure_data.py
```

**3. Generate Task Sequences**
Creates sequential task data with spatial context for the LSTM.
```bash
python tools/generate_task_data.py
```

**4. Pre-train LSTM Predictor**
Trains the caching model offline before the RL agent starts.
```bash
python tools/train_cache_predictor.py
```

---

## Usage

### 1. Configuration
Open `config.py` to set your simulation parameters. Key switches:

```python
# Choose Physics Engine
SIMULATION_MODE = 'SUMO'  # or 'PYTHON_KINEMATIC'

# Enable/Disable Smart Caching
USE_PREDICTIVE_CACHING = True

# Simulation Scale
NUM_VEHICLES = 100
TOTAL_EPISODES = 1000
```

### 2. Start Training
You can run training in single-core mode or parallel mode.

**Standard Training:**
```bash
python main.py
```

**Parallel Training (Faster):**
```bash
bash scripts/start_parallel_training.sh
```

### 3. Monitoring (Dashboard)
Launch the web interface to view logs, TensorBoard metrics, and system status.
```bash
bash scripts/start_dashboard.sh
```
*Access at: http://localhost:8501*

---

## Evaluation & Results

Once models are trained (saved in `models/`), you can benchmark MUCEDS against baseline algorithms.

**Run Comparative Evaluation:**
```bash
python analysis/evaluate.py --model_path "models/experiment_your_timestamp"
```

**Generate Final Report Plots:**
This script compares Python-Kinematic vs. SUMO vs. LSTM-enabled variants.
```bash
bash scripts/run_report_evaluation.sh
```

**Run Sensitivity Analysis:**
```bash
python analysis/run_sensitivity.py
```

### Baseline Algorithms

| Algorithm | Fleet Sizing | Positioning Strategy | Description |
|-----------|--------------|---------------------|-------------|
| **OUPRS** | Fixed (K=1) | Random | Single UAV with random movement |
| **OUPOS** | Fixed (K=1) | Center of Mass | Single UAV targeting vehicle centroid |
| **MRUPRS** | Random (K∈[3,7]) | Random | Multiple UAVs with random movement |
| **MRUPOS** | Random (K∈[3,7]) | K-Means Clustering | Multiple UAVs positioned at cluster centers |
| **MOUPRS** | DDQN-Optimized | Random | Learned fleet size, random positioning |
| **MUCEDS** | DDQN-Optimized | MADDPG-Optimized | **Full HRL optimization (Ours)** |

### Key Performance Metrics

- **Task Completion Rate**: Percentage of tasks completed within latency constraints
- **Cache Hit Ratio**: Percentage of requests served from local UAV cache
- **Energy Efficiency**: Tasks completed per Joule of energy consumed
- **Average Latency**: Mean task completion time
- **Total Profit**: Revenue from completed tasks minus operational costs

---

## Architecture Details

### System Model

### Physical Optimization (HRL)

#### Outer Loop - DDQN Agent
- **State Space (6D):**
  - Global vehicle density
  - Average task latency
  - Total accumulated profit
  - Current UAV count
  - Energy utilization ratio
  - Task completion rate

- **Action Space:** Discrete (1 to 50 UAVs)

- **Network Architecture:** 
  ```
  Input(6) → FC(256) → ReLU → FC(256) → ReLU → FC(50)
  ```

#### Inner Loop - MADDPG Agents
- **State Space (7D per UAV):**
  - Local vehicle density (within communication range)
  - UAV energy level
  - Relative position to nearest hotspot (x, y)
  - Distance to nearest neighbor UAV
  - Current velocity (x, y)

- **Action Space:** Continuous 2D velocity vector

- **Network Architecture:**
  ```
  Actor:  Input(7) → FC(256) → ReLU → FC(256) → ReLU → Tanh → Output(2)
  Critic: Input(7*K + 2*K) → FC(256) → ReLU → FC(256) → ReLU → Output(1)
  ```

### Reward Function

$$R_{total} = R_{task} - C_{energy} - P_{latency}$$

Where:
- $R_{task} = \alpha \cdot \sum_{i} \mathbb{1}[\text{task}_i \text{ completed}]$
- $C_{energy} = \beta \cdot \sum_{u} E_u^{hover} + E_u^{compute} + E_u^{comm}$
- $P_{latency} = \gamma \cdot \sum_{i} \max(0, t_i - t_i^{deadline})$

### Logical Optimization (LSTM)

#### Architecture
```
Input: (service_seq, zone_seq) → [Service Embedding(64)] ⊕ [Zone Embedding(16)]
                                              ↓
                               LSTM(hidden=128, layers=2, dropout=0.2)
                                              ↓
                            [Service Head] + [Content Head]
                                              ↓
                               Top-K Services & Content
```

#### Training
- **Loss**: Cross-entropy on next-item prediction
- **Optimizer**: Adam (lr=0.001)
- **Sequence Length**: 20 timesteps

---

## Supported Cities

The framework includes pre-processed SUMO networks for 8 global cities:

| City | Region | Network Size | Road Segments |
|------|--------|--------------|---------------|
| Delhi | India | 10km × 10km | ~2,500 |
| Mumbai | India | 10km × 10km | ~2,200 |
| Bangalore | India | 10km × 10km | ~1,800 |
| Guwahati | India | 10km × 10km | ~1,200 |
| Paris | France | 10km × 10km | ~3,100 |
| London | UK | 10km × 10km | ~2,800 |
| New York City | USA | 10km × 10km | ~3,500 |
| Tokyo | Japan | 10km × 10km | ~2,900 |

To add a new city:
1. Download OSM data to `osm_data/`
2. Run `python tools/generate_traffic.py --city your_city`
3. Add city name to `SUMO_SCENARIO_POOL` in `config.py`

---

## Configuration Reference

Key parameters in `config.py`:

### Simulation Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `SIMULATION_MODE` | `'SUMO'` | Physics engine: `'SUMO'` or `'PYTHON_KINEMATIC'` |
| `USE_PREDICTIVE_CACHING` | `False` | Enable LSTM-based predictive caching |
| `TOTAL_EPISODES` | `1000` | Training episodes |
| `INNER_STEPS` | `100` | Steps per episode (inner loop) |
| `NUM_VEHICLES` | `200` | Target vehicle count |

### UAV Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `UAV_ALTITUDE` | `50m` | Fixed flight altitude |
| `UAV_COMMUNICATION_RANGE` | `1000m` | Service coverage radius |
| `UAV_MAX_SPEED` | `50 m/s` | Maximum velocity |
| `UAV_COMPUTATIONAL_RESOURCES` | `10 GFLOPS` | Processing capacity |
| `UAV_ENERGY_CAPACITY_JOULES` | `(800k, 1M)` | Battery capacity range |

### RL Hyperparameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `DDQN_LEARNING_RATE` | `0.0005` | Outer loop learning rate |
| `MADDPG_LEARNING_RATE_ACTOR` | `0.0005` | Inner loop actor LR |
| `MADDPG_LEARNING_RATE_CRITIC` | `0.0005` | Inner loop critic LR |
| `DDQN_GAMMA` | `0.95` | Discount factor |
| `DDQN_EPSILON_DECAY` | `0.995` | Exploration decay |

---

## Output Files

After training, the following outputs are generated:

```
models/
└── experiment_{mode}_{cache}_{timestamp}/
    ├── ddqn_model.pth           # Outer loop weights
    ├── ddqn_target_model.pth    # Target network weights
    └── maddpg_{k}_agents/       # Inner loop weights for K UAVs
        ├── maddpg_actor_0.pth
        ├── maddpg_critic_0.pth
        └── ...

runs/
└── experiment_{mode}_{cache}_{timestamp}/
    └── events.out.tfevents.*    # TensorBoard logs

logs/
├── training_log_{experiment}.log
└── pids/                        # Process IDs for services
```

---

## Troubleshooting

### Common Issues

**SUMO Connection Error**
```
Error: Could not connect to SUMO
```
Solution: Ensure `SUMO_HOME` is set and SUMO is installed correctly.

**CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
Solution: Reduce `MADDPG_BATCH_SIZE` or set `DEVICE = 'cpu'` in config.

**TraCI Port Conflict**
```
Error: Port already in use
```
Solution: Run `bash scripts/stop_services.sh` to kill existing SUMO instances.

**Missing Azure Data**
```
Warning: Azure data not found, using synthetic generation
```
This is normal if you haven't downloaded the Azure traces. The system auto-generates synthetic workload data.

---

