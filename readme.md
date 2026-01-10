# MUCEDS: Multi-UAV Cost-Efficient Deployment Scheme
### Cost-Efficient Deployment and Predictive Caching Optimization in Multi-UAV-Assisted Vehicular Edge Networks using HRL and LSTM


**MUCEDS** is a holistic optimization framework for Vehicular Edge Computing Networks (VECNs). It addresses the challenges of dynamic traffic, limited UAV battery/storage, and latency constraints by coupling **Hierarchical Reinforcement Learning (HRL)** for physical UAV deployment with **LSTM-based Predictive Caching** for logical content management.

This project offers a dual-simulation environment: a realistic traffic physics engine using **SUMO (Simulation of Urban MObility)** and a lightweight **Python Kinematic** simulator for rapid prototyping.

---

## 🌟 Key Features

*   **Hierarchical RL Framework:**
    *   **Outer Layer (DDQN):** Strategically optimizes the *number* of UAVs to deploy to balance coverage vs. operational costs.
    *   **Inner Layer (MADDPG):** A multi-agent continuous control policy that optimizes UAV *positioning* to cover dynamic vehicle hotspots.
*   **Predictive Caching (Spatial-Temporal LSTM):**
    *   Analyzes user request sequences and spatial zones to pre-fetch content.
    *   Significantly increases Cache Hit Ratio compared to reactive (Zipf/LRU) baselines.
*   **Dual Simulation Modes:**
    *   **SUMO Mode:** Uses TraCI to interface with real-world road networks (OSM data) for Delhi, Mumbai, Paris, NYC, etc.
    *   **Python Kinematic Mode:** Fast, lightweight simulation for algorithmic debugging and hyperparameter tuning.
*   **Real-World Workload Integration:** Adapts the Azure Functions Trace 2019 dataset to simulate realistic edge computing task requests.
*   **Interactive Dashboard:** A full-featured web dashboard (Streamlit) to monitor training, visualize TensorBoard metrics, and manage configurations.

---

## 📂 Project Structure

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

## 🛠️ Installation

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

## ⚙️ Data Pipeline Setup

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

## 🚀 Usage

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

## 📊 Evaluation & Results

Once models are trained (saved in `models/`), you can benchmark MUCEDS against baseline algorithms (Random, OUPOS, etc.).

**Run Comparative Evaluation:**
```bash
python analysis/evaluate.py --model_path "models/experiment_your_timestamp"
```

**Generate Final Report Plots:**
This script compares Python-Kinematic vs. SUMO vs. LSTM-enabled variants.
```bash
bash scripts/run_report_evaluation.sh
```

### Baselines Included:
1.  **OUPRS:** One UAV, Random Position.
2.  **OUPOS:** One UAV, Center of Mass Position.
3.  **MRUPRS:** Multi-UAV (Random K), Random Position.
4.  **MRUPOS:** Multi-UAV (Random K), K-Means Clustering Position.
5.  **MOUPRS:** DDQN-Optimized K, Random Position.

---

## 🧩 Architecture Details

### Physical Optimization (HRL)
*   **State Space:** Global density, average latency, total profit (Outer); Local density, UAV energy, relative position (Inner).
*   **Reward Function:** $Profit = \alpha \cdot (\text{Tasks Completed}) - \beta \cdot (\text{Energy Cost}) - \gamma \cdot (\text{Latency Penalty})$.

### Logical Optimization (LSTM)
*   **Input:** Sequence of `(Service_ID, Zone_ID)` pairs.
*   **Architecture:** Embedding Layers $\to$ LSTM (128 units) $\to$ Softmax Heads (Service & Content).
*   **Output:** Probabilities for next requested service and content type.

---
