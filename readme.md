# Cost-Efficient Deployment and Predictive Caching Optimization in Multi-UAV-Assisted VECNs

## 📌 Project Overview

This repository contains the implementation of **MUCEDS (Multi-UAV Cost-Efficient Deployment Scheme)**, a novel framework for Vehicular Edge Computing Networks (VECNs). This project addresses the challenges of dynamic resource allocation and latency minimization in intelligent transportation systems by integrating:

1.  **Physical Optimization (HRL):** A Hierarchical Reinforcement Learning framework combining **DDQN** (strategic UAV numbering) and **MADDPG** (tactical UAV positioning) to adapt to realistic traffic flows.
2.  **Logical Optimization (LSTM):** A Context-Aware **LSTM** predictive caching mechanism that pre-fetches content based on spatiotemporal user request patterns.
3.  **Realistic Simulation:** Integration with **SUMO (Simulation of Urban MObility)** to model real-world road topologies (e.g., Delhi, Mumbai, Paris) and complex vehicular dynamics.

## 📂 Directory Structure

```text
├── agents/                 # Deep RL Agent implementations
│   ├── ddqn.py             # Outer Layer: Optimization of UAV quantity
│   └── maddpg.py           # Inner Layer: Optimization of UAV positioning
├── analysis/               # Benchmarking and Evaluation modules
│   ├── evaluate.py         # Main script for testing trained models
│   └── benchmark_agents.py # Baseline algorithms (OUPRS, MRUPOS, etc.)
├── data/                   # Dataset storage (Azure traces & Generated tasks)
├── osm_data/               # Raw OpenStreetMap files for target cities
├── prediction/             # Deep Learning module for Content Caching
│   └── model.py            # Spatial-Temporal LSTM architecture
├── simulation/             # Core Environment Logic
│   ├── environment.py      # VECN Environment wrapper
│   ├── physics.py          # Interface for SUMO/TraCI and Kinematics
│   └── entities.py         # UAV, Vehicle, and Task class definitions
├── sumo_scenario/          # Generated SUMO network and route files
├── tools/                  # Pre-processing and Training utilities
│   ├── generate_traffic.py # Converts OSM data to SUMO scenarios
│   ├── process_azure_data.py # Processes Azure Function traces
│   └── train_cache_predictor.py # Offline training for LSTM
├── config.py               # Centralized configuration and hyperparameters
└── main.py                 # Main entry point for HRL training
```

## 🛠️ Prerequisites & Installation

### 1. System Requirements
*   **Python:** 3.8 or higher
*   **OS:** Windows, Linux, or macOS
*   **SUMO:** You must have [Eclipse SUMO](https://eclipse.dev/sumo/) installed.
    *   Set the `SUMO_HOME` environment variable (e.g., `C:\Program Files (x86)\Eclipse\Sumo`).

### 2. Python Dependencies
Install the required libraries:

```bash
pip install -r requirements.txt
```
*Key libraries: `torch`, `traci`, `numpy`, `pandas`, `scipy`, `tensorboard`, `pygame`.*

---

## ⚙️ Data Preparation Pipeline

Before running the main simulation, you must generate the traffic scenarios and the workload datasets. Run the following commands in order:

### Step 1: Generate Traffic Scenarios (SUMO)
Converts the `.osm` files in `osm_data/` into routable SUMO networks and generates random vehicle trips.

```bash
python tools/generate_traffic.py
```

### Step 2: Process Workload Traces
Processes the raw Azure Functions Trace 2019 dataset to create a probability matrix for service invocation.
*Note: Ensure the raw Azure CSV path is correctly set in `tools/process_azure_data.py`.*

```bash
python tools/process_azure_data.py
```

### Step 3: Generate Synthetic Task Sequences
Uses the processed workload matrix to generate a sequence of task requests with spatial context (Zones) for LSTM training.

```bash
python tools/generate_task_data.py
```

### Step 4: Train Predictive Caching Model
Trains the LSTM model offline on the generated task data. This saves the weights to `models/lstm_cache_predictor.pth`.

```bash
python tools/train_cache_predictor.py
```

---

## 🚀 Running the Simulation (Training)

The core training loop involves the DDQN and MADDPG agents interacting with the environment while the LSTM module manages caching.

### 1. Configuration
Open `config.py` to adjust simulation parameters:
*   **`SIMULATION_MODE`**: Set to `'SUMO'` for full physics or `'PYTHON_KINEMATIC'` for rapid prototyping.
*   **`USE_PREDICTIVE_CACHING`**: Set to `True` to enable the LSTM module.
*   **`VISUALIZATION`**: Set to `True` to view the Pygame or SUMO-GUI interface.

### 2. Execute Training
Run the main script. This will create a timestamped folder in `models/` for checkpoints and `logs/` for execution details.

```bash
python main.py
```

**Monitoring:**
Track training progress (Reward, Loss, Latency) using TensorBoard:
```bash
tensorboard --logdir=runs
```

---

## 📊 Evaluation & Benchmarking

Once training is complete, evaluate the trained model against heuristic baselines (Random, Center-of-Mass, K-Means).

### Run Evaluation Script
You must provide the path to the specific experiment folder generated during training.

```bash
python analysis/evaluate.py --model_path "models/experiment_YYYY-MM-DD_HH-MM-SS" --output_dir "results/final_test"
```

### Baselines Compared
The evaluation script compares **MUCEDS** against:
*   **OUPRS/OUPOS:** Single UAV (Random vs. Optimized Position).
*   **MRUPRS:** Multi-UAV Random Position.
*   **MRUPOS:** Multi-UAV Position Optimized (K-Means Clustering).
*   **MOUPRS:** Multi-UAV (DDQN Number Optimization) with Random Positioning.

---

## 🔬 Sensitivity Analysis

To analyze how the system performs under varying economic parameters (e.g., Energy Cost, Latency Penalty):

```bash
python analysis/run_sensitivity.py
```
*Note: Ensure `run_sensitivity.py` points to a valid trained model path internally or via arguments.*

## 📚 References

This project is based on the system model and problem formulation described in:

> Y. Liu, C. Yang, Y. Tang, H. Zhao, Y. Liu and S. Xie, "Cost-Efficient Deployment Optimization for Multi-UAV-Assisted Vehicular Edge Computing Networks," in *IEEE Internet of Things Journal*, vol. 12, no. 6, pp. 6158-6169, 15 March 2025.