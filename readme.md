# Cost-Efficient Deployment and Predictive Caching Optimization in Multi-UAV-Assisted VECNs

## 📌 Project Overview

This repository contains the implementation of **MUCEDS (Multi-UAV Cost-Efficient Deployment Scheme)**, a comprehensive framework for Vehicular Edge Computing Networks (VECNs). This project addresses the challenges of dynamic resource allocation, latency minimization, and cost optimization in intelligent transportation systems by integrating:

1.  **Physical Optimization (HRL):** A Hierarchical Reinforcement Learning framework combining **DDQN** (strategic UAV fleet sizing) and **MADDPG** (tactical UAV positioning) to adapt to dynamic traffic flows.
2.  **Logical Optimization (LSTM):** A Context-Aware **LSTM** predictive caching mechanism that pre-fetches content based on spatiotemporal user request patterns and workload characteristics.
3.  **Flexible Simulation Modes:** Dual simulation support with **SUMO (Simulation of Urban MObility)** for realistic road topologies and traffic patterns, plus a lightweight **Python kinematic simulator** for rapid prototyping.
4.  **Real-World Data Integration:** Utilizes Azure Functions Trace 2019 dataset for realistic workload patterns across 8 major cities worldwide.
5.  **Web-Based Monitoring:** Interactive **Streamlit dashboard** for real-time training monitoring, configuration management, and log analysis.

## 📂 Directory Structure

```text
├── agents/                 # Deep Reinforcement Learning Agents
│   ├── ddqn.py             # Outer Layer: UAV fleet size optimization
│   ├── maddpg.py           # Inner Layer: Multi-agent UAV positioning
│   ├── networks.py         # Neural network architectures (Actor, Critic, DDQN)
│   └── buffer.py           # Experience replay buffer implementation
├── analysis/               # Evaluation and Benchmarking
│   ├── evaluate.py         # Model evaluation with parallel execution
│   ├── benchmark_agents.py # Baseline algorithms (OUPRS, OUPOS, MRUPRS, MRUPOS, MOUPRS)
│   ├── run_sensitivity.py  # Sensitivity analysis for economic parameters
│   └── evaluate_predictor.py # LSTM predictor standalone evaluation
├── data/                   # Datasets and Generated Data
│   ├── azure_workload_matrix.npy     # Processed workload patterns
│   ├── azure_service_meta.csv        # Service metadata
│   └── task_request_data.csv         # Generated task sequences for LSTM
├── osm_data/               # OpenStreetMap Files for 8 Cities
│   └── {city}.osm          # Delhi, Mumbai, Guwahati, Bangaluru, Paris, London, NYC, Tokyo
├── prediction/             # Predictive Caching Module
│   ├── model.py            # Spatial-Temporal LSTM architecture
│   └── generator.py        # Demand and workload generation
├── simulation/             # Environment and Physics
│   ├── environment.py      # VECN environment wrapper
│   ├── physics.py          # Dual-mode physics connector (SUMO/Python kinematic)
│   ├── entities.py         # UAV, Vehicle, and Task entities
│   ├── comms.py            # Communication model (LoS/NLoS, path loss, bandwidth)
│   └── tasks.py            # Task management and offloading logic
├── sumo_scenario/          # SUMO Configuration Files
│   ├── {city}.net.xml      # Road network files
│   ├── {city}.rou.xml      # Route definitions
│   ├── {city}.sumocfg      # SUMO configuration
│   └── {city}.add.xml      # Additional files (UAV definitions)
├── tools/                  # Utilities and Preprocessing
│   ├── generate_traffic.py         # OSM to SUMO scenario converter
│   ├── process_azure_data.py       # Azure trace processor
│   ├── generate_task_data.py       # Synthetic task sequence generator
│   ├── train_cache_predictor.py    # LSTM offline training
│   ├── analyze_task_data.py        # Task data statistics
│   └── tensorboard_tools/          # TensorBoard utilities
├── scripts/                # Automation Scripts
│   ├── start_dashboard.sh          # Launch web dashboard
│   ├── start_services.sh           # Start all services
│   └── stop_services.sh            # Stop all services
├── visualization/          # Real-time Visualization
│   └── visualizer.py       # Pygame-based environment renderer
├── config.py               # Centralized configuration and hyperparameters
├── main.py                 # Main training entry point
├── dashboard.py            # Streamlit web dashboard
└── DASHBOARD_README.md     # Dashboard documentation
```

## 🛠️ Prerequisites & Installation

### 1. System Requirements
*   **Python:** 3.8 or higher (3.10+ recommended)
*   **OS:** Windows, Linux, or macOS
*   **CUDA:** Optional but recommended for GPU acceleration (PyTorch)
*   **SUMO (Optional):** For realistic traffic simulation
    *   Download and install [Eclipse SUMO](https://eclipse.dev/sumo/)
    *   Set the `SUMO_HOME` environment variable (e.g., `C:\Program Files (x86)\Eclipse\Sumo` or `/usr/share/sumo`)
    *   **Note:** SUMO is only required if `SIMULATION_MODE='SUMO'` in config.py

### 2. Python Dependencies
Install all required libraries:

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `torch` - Deep learning framework
- `numpy`, `scipy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib`, `plotly` - Visualization
- `tensorboard` - Training monitoring
- `streamlit` - Web dashboard
- `pygame` - Real-time visualization
- `traci` - SUMO interface (optional)
- `tqdm` - Progress bars
- `numba` - JIT compilation for performance

### 3. Hardware Recommendations
- **Minimum:** 8GB RAM, 4-core CPU
- **Recommended:** 16GB+ RAM, 8-core CPU, NVIDIA GPU with 6GB+ VRAM
- **Storage:** 2GB+ free space for models and logs

---

## ⚙️ Data Preparation Pipeline

Before running the main simulation, you must prepare the traffic scenarios and workload datasets. Follow these steps in order:

### Step 1: Generate Traffic Scenarios (SUMO)
**Purpose:** Converts `.osm` files from `osm_data/` into routable SUMO networks and generates random vehicle trips for 8 cities.

```bash
python tools/generate_traffic.py
```

**Cities processed:** Delhi, Mumbai, Guwahati, Bangaluru, Paris, London, NYC, Tokyo

**Output:** Creates `.net.xml`, `.rou.xml`, `.sumocfg`, and `.add.xml` files in `sumo_scenario/`

**Note:** Skip this step if using `SIMULATION_MODE='PYTHON_KINEMATIC'` exclusively.

### Step 2: Process Workload Traces (Azure Dataset)
**Purpose:** Processes the Azure Functions Trace 2019 dataset to create a probability matrix for realistic service invocation patterns.

1. **Download the dataset:** [Azure Functions Trace 2019](https://github.com/Azure/AzurePublicDataset)
2. **Update the path** in `tools/process_azure_data.py` (line 12):
   ```python
   INPUT_FILE = "/path/to/invocations_per_function_md.anon.d01.csv"
   ```
3. **Run the processor:**
   ```bash
   python tools/process_azure_data.py
   ```

**Output:** 
- `data/azure_workload_matrix.npy` - 20×1440 matrix of invocation patterns
- `data/azure_service_meta.csv` - Service metadata with trigger types

**Fallback:** If Azure data is unavailable, the system uses synthetic workload generation.

### Step 3: Generate Synthetic Task Sequences
**Purpose:** Creates labeled sequences of task requests with spatial context (zones) for LSTM training.

```bash
python tools/generate_task_data.py
```

**Configuration:** Adjust parameters in the script:
- `NUM_SAMPLES` - Number of task sequences to generate (default: 10,000)
- Uses `NUM_ZONES` and `NUM_SERVICE_TYPES` from config.py

**Output:** `data/task_request_data.csv` with columns: `service`, `content`, `zone`

### Step 4: Train Predictive Caching Model (LSTM)
**Purpose:** Trains the spatial-temporal LSTM model offline on generated task data.

```bash
python tools/train_cache_predictor.py
```

**Configuration:**
- `TRAINING_EPOCHS` - Number of training epochs (default: 20)
- `BATCH_SIZE` - Batch size for training (default: 1024)

**Output:** Saves trained model to `models/lstm_cache_predictor.pth`

**Note:** This is required if `USE_PREDICTIVE_CACHING=True` in config.py

**Optional: Analyze generated data**
```bash
python tools/analyze_task_data.py
```

---

## 🚀 Running the Simulation

### Training Configuration

Before starting training, configure the simulation in [config.py](config.py):

#### Core Simulation Settings
```python
# Simulation Mode Selection
SIMULATION_MODE = 'PYTHON_KINEMATIC'  # Options: 'SUMO', 'PYTHON_KINEMATIC'

# Caching Strategy
USE_PREDICTIVE_CACHING = True  # True: LSTM predictor, False: Reactive caching

# Training Parameters
TOTAL_EPISODES = 1000  # Total training episodes
INNER_STEPS = 100      # Steps per episode

# Visualization
VISUALIZATION = False  # Enable Pygame/SUMO-GUI visualization
SAVE_VISUALIZATION_IMAGES = True  # Save snapshots at key episodes
```

#### Simulation Mode Comparison

| Mode | Description | Use Case | Requirements |
|------|-------------|----------|--------------|
| **PYTHON_KINEMATIC** | Lightweight kinematic simulator | Fast prototyping, debugging, testing | None |
| **SUMO** | Realistic traffic simulation | Final evaluation, realistic scenarios | SUMO installation |

#### Advanced Configuration
- **Entity Parameters:** UAV/Vehicle fleet size, communication range, speeds
- **Traffic Scenarios:** Uniform, Single Congestion, Multi-Hotspot patterns
- **RL Hyperparameters:** DDQN/MADDPG learning rates, buffer sizes, exploration
- **Communication Models:** Bandwidth, path loss, energy consumption
- **Economic Parameters:** Maintenance costs, latency penalties, computation costs

See [config.py](config.py) for all available parameters (220+ lines of documentation).

### Starting Training

Run the main training script:

```bash
python main.py
```

**What happens during training:**
1. **Initialization:** Creates timestamped experiment folder in `models/experiment_{mode}_{cache}_{timestamp}/`
2. **Outer Loop (DDQN):** Determines optimal UAV fleet size
3. **Inner Loop (MADDPG):** Optimizes UAV positions using multi-agent learning
4. **Caching:** LSTM predictor pre-fetches content (if enabled)
5. **Logging:** Saves checkpoints, metrics, and logs automatically

**Output Structure:**
```
models/experiment_pythonkinematic_pred_20241230_143000/
├── ddqn_checkpoint_ep1000.pth
├── maddpg_1_uavs_ep1000.pth
├── maddpg_2_uavs_ep1000.pth
└── ...

logs/
└── training_log_pythonkinematic_pred_20241230_143000.log

runs/experiment_pythonkinematic_pred_20241230_143000/
└── events.out.tfevents...  # TensorBoard logs
```

### Monitoring Training Progress

#### Option 1: TensorBoard (Recommended)
Track training metrics in real-time:

```bash
tensorboard --logdir=runs
```

Open browser to `http://localhost:6006` to view:
- **Profit Metrics:** Total profit, maintenance costs, latency penalties
- **System Metrics:** Success rate, failure rate, average latency
- **UAV Metrics:** Active UAVs, energy consumption, positions
- **Cache Metrics:** Hit rate, prediction accuracy
- **Loss Metrics:** Actor/Critic losses, DDQN loss

#### Option 2: Web Dashboard
Launch the interactive Streamlit dashboard:

```bash
# Using script
bash scripts/start_dashboard.sh

# Or directly
streamlit run dashboard.py
```

**Dashboard Features:**
- 🎮 Start/Stop training with one click
- 📈 Real-time TensorBoard metric visualization
- 📋 Live log tailing and searching
- ⚙️ Edit configuration with auto-backup
- 📊 System status and environment info
- 💾 Download logs and model checkpoints

Access at `http://localhost:8501`

See [DASHBOARD_README.md](DASHBOARD_README.md) for detailed dashboard documentation.

### Visualization During Training

**Python Kinematic Mode:**
Set `VISUALIZATION = True` to see Pygame rendering:
- Blue circles: Vehicles
- Green/Red circles: UAVs (green = healthy energy, red = low energy)
- Lines: Communication links
- Real-time metrics overlay

**SUMO Mode:**
SUMO-GUI automatically displays realistic traffic and UAV overlays.

**Save Snapshots:**
Set `SAVE_VISUALIZATION_IMAGES = True` to capture frames at episodes: 1, 250, 500, 750, 1000
Saved to `visualization_snapshots/`

---

## 📊 Evaluation & Benchmarking

Once training is complete, evaluate the trained model against heuristic baselines and analyze performance.

### Running Model Evaluation

Evaluate trained models with parallel execution support:

```bash
python analysis/evaluate.py --model_path "models/experiment_pythonkinematic_pred_20241230_143000" \
                           --output_dir "results/final_test"
```

**Parameters:**
- `--model_path`: Path to the experiment folder containing trained checkpoints
- `--output_dir`: Directory to save evaluation results (optional)

**Evaluation Process:**
1. Loads trained DDQN and MADDPG models
2. Runs multiple episodes with varying vehicle densities (50-120 vehicles)
3. Compares MUCEDS against baseline algorithms
4. Generates comprehensive performance metrics and plots

**Output:**
- Performance comparison plots
- Average latency, success rate, costs
- Statistical analysis across scenarios

### Baseline Algorithms

The evaluation compares **MUCEDS** against these benchmarks:

| Algorithm | UAV Fleet | Positioning Strategy | Description |
|-----------|-----------|---------------------|-------------|
| **OUPRS** | 1 (Fixed) | Random | One UAV with Pure Random positioning |
| **OUPOS** | 1 (Fixed) | Center of Mass | One UAV, Position Optimized |
| **MRUPRS** | Random (3-7) | Random | Multi-UAV Random Position Random Scheme |
| **MRUPOS** | Random (3-7) | K-Means Clustering | Multi-UAV Random-count Position Optimized |
| **MOUPRS** | DDQN Optimized | Random | Multi-UAV Optimized-count with Random positioning |
| **MUCEDS** | DDQN Optimized | MADDPG Optimized | **Full HRL framework** (Our method) |

### Sensitivity Analysis

Analyze system behavior under varying economic parameters:

```bash
python analysis/run_sensitivity.py
```

**Analyzed Parameters:**
- Energy costs per joule
- Latency penalty weights
- Maintenance costs per UAV
- Computation costs
- Bandwidth allocation

**Output:** Plots showing performance sensitivity to each parameter

### LSTM Predictor Evaluation

Evaluate the caching predictor independently:

```bash
python analysis/evaluate_predictor.py
```

**Metrics:**
- Service prediction accuracy
- Content prediction accuracy
- Cache hit rate improvement
- Prediction latency

---

## � Advanced Usage

### Custom Scenarios

#### Creating New Traffic Scenarios
1. Add `.osm` file to `osm_data/` directory
2. Update `SUMO_SCENARIO_POOL` in [config.py](config.py):
   ```python
   SUMO_SCENARIO_POOL = ['delhi', 'mumbai', ..., 'your_city']
   ```
3. Run traffic generator:
   ```bash
   python tools/generate_traffic.py
   ```

#### Configuring Traffic Patterns
Edit `TRAFFIC_SCENARIOS` in [config.py](config.py):
```python
TRAFFIC_SCENARIOS = {
    'SCENARIO_NAMES': ['UNIFORM', 'SINGLE_CONGESTION', 'MULTI_CONGESTION'],
    'SCENARIO_WEIGHTS': [0.2, 0.4, 0.4],  # Probability distribution
    'SCENARIOS': {
        'UNIFORM': {
            'num_hotspots': 0,
            'hotspot_radius': 0,
            'hotspot_ratio': 0.0
        },
        'SINGLE_CONGESTION': {
            'num_hotspots': 1,
            'hotspot_radius': 2000,
            'hotspot_ratio': 0.7
        },
        # Add custom scenarios here
    }
}
```

### Hyperparameter Tuning

#### DDQN Parameters
```python
DDQN_ACTION_SPACE = 50          # Maximum UAV fleet size
DDQN_LEARNING_RATE = 0.0005     # Learning rate
DDQN_EPSILON_START = 0.9        # Initial exploration
DDQN_EPSILON_END = 0.01         # Final exploration
DDQN_EPSILON_DECAY = 0.995      # Decay rate per episode
DDQN_GAMMA = 0.95               # Discount factor
DDQN_TAU = 0.005                # Target network update rate
```

#### MADDPG Parameters
```python
MADDPG_LEARNING_RATE_ACTOR = 0.0005
MADDPG_LEARNING_RATE_CRITIC = 0.0005
MADDPG_GAMMA = 0.95
MADDPG_TAU = 0.01
MADDPG_BUFFER_SIZE = 100000
MADDPG_BATCH_SIZE = 128
```

### TensorBoard Analysis Tools

Extract and compare multiple training runs:

```bash
# Extract metrics from TensorBoard logs
python tools/tensorboard_tools/extract_tensorboard_graphs.py

# Compare different runs
python tools/tensorboard_tools/compare_tensorboard_runs.py

# Batch extract from multiple experiments
python tools/tensorboard_tools/batch_extract_tensorboard.py
```

### Process Management Scripts

Use provided shell scripts for easy management:

```bash
# Start all services (dashboard + training)
bash scripts/start_services.sh

# Stop all running services
bash scripts/stop_services.sh

# Check service status
bash scripts/check_services.sh

# Start dashboard only
bash scripts/start_dashboard.sh
```

### GPU Acceleration

Enable CUDA for faster training:
1. Install PyTorch with CUDA support
2. Verify GPU availability:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should return True
   ```
3. Config automatically uses GPU if available (check `DEVICE` in config.py)

---

## 🏗️ Architecture Details

### Hierarchical RL Framework

```
┌─────────────────────────────────────────────────────────┐
│                    DDQN (Outer Agent)                   │
│  State: [avg_latency, num_vehicles, congestion, ...]   │
│  Action: Select number of UAVs to deploy (0-50)        │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              MADDPG (Inner Agent Pool)                  │
│  State: [position, vehicle_density, energy, ...]       │
│  Action: Velocity vector (vx, vy) for each UAV         │
│  Coordination: Shared critic, decentralized actors     │
└─────────────────────────────────────────────────────────┘
```

### LSTM Caching Architecture

```
Input: Service Sequence + Spatial Zone Context
   │
   ▼
┌──────────────────┐    ┌──────────────────┐
│ Service Embedding│    │ Zone Embedding   │
│   (64-dim)       │    │   (16-dim)       │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │   LSTM (128 hidden)   │
         │   2 layers, dropout   │
         └──────────┬────────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
┌────────────────┐    ┌────────────────┐
│ Service Output │    │ Content Output │
│   (20 classes) │    │   (50 classes) │
└────────────────┘    └────────────────┘
```

### Communication Model

**Path Loss Calculation:**
- LoS (Line-of-Sight) probability based on distance and angle
- NLoS (Non-Line-of-Sight) with additional attenuation
- Free space path loss with carrier frequency consideration
- Dynamic bandwidth allocation via TDMA

**Energy Model:**
- Hovering energy: 200W constant
- Communication energy: 0.5 J/Mbit
- Computation energy: 10 nJ/Gcycle
- Energy capacity: 800-1000 kJ per UAV

---

## 🐛 Troubleshooting

### Common Issues

#### SUMO Not Found
**Error:** `TraCI could not connect to SUMO`
**Solution:**
1. Ensure SUMO is installed: `sumo --version`
2. Set environment variable:
   ```bash
   # Linux/Mac
   export SUMO_HOME=/usr/share/sumo
   
   # Windows
   set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
   ```
3. Alternative: Use `SIMULATION_MODE='PYTHON_KINEMATIC'` in config.py

#### LSTM Model Not Found
**Error:** `Predictor model not found at models/lstm_cache_predictor.pth`
**Solution:**
1. Run training: `python tools/train_cache_predictor.py`
2. Or disable caching: `USE_PREDICTIVE_CACHING = False` in config.py

#### Out of Memory (GPU)
**Error:** `CUDA out of memory`
**Solution:**
1. Reduce batch sizes in config.py:
   ```python
   DDQN_BATCH_SIZE = 32  # Default: 64
   MADDPG_BATCH_SIZE = 64  # Default: 128
   ```
2. Switch to CPU: `DEVICE = torch.device("cpu")`

#### Training Too Slow
**Solutions:**
1. Use `PYTHON_KINEMATIC` mode instead of SUMO
2. Disable visualization: `VISUALIZATION = False`
3. Reduce episodes: `TOTAL_EPISODES = 100`
4. Enable GPU acceleration
5. Reduce vehicle count: `NUM_VEHICLES = 100`

#### Dashboard Won't Start
**Error:** `streamlit: command not found`
**Solution:**
```bash
pip install streamlit plotly
streamlit run dashboard.py
```

#### Module Import Errors
**Error:** `ModuleNotFoundError: No module named 'xxx'`
**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Performance Optimization Tips

1. **Fast Prototyping:**
   ```python
   SIMULATION_MODE = 'PYTHON_KINEMATIC'
   VISUALIZATION = False
   TOTAL_EPISODES = 100
   NUM_VEHICLES = 50
   ```

2. **Production Training:**
   ```python
   SIMULATION_MODE = 'SUMO'
   USE_PREDICTIVE_CACHING = True
   TOTAL_EPISODES = 1000
   DEVICE = torch.device("cuda")  # GPU
   ```

3. **Quick Evaluation:**
   ```python
   EVAL_EPISODES = 3
   EVAL_SCENARIO_VEHICLES = range(50, 101, 25)  # [50, 75, 100]
   ```

---

## 📁 Project Statistics

**Lines of Code:** ~8,000+ lines
**Languages:** Python 100%
**Key Technologies:**
- PyTorch (Deep RL)
- SUMO/TraCI (Traffic Simulation)
- Streamlit (Web Dashboard)
- TensorBoard (Monitoring)
- Pygame (Visualization)

**Supported Cities:** 8 (Delhi, Mumbai, Guwahati, Bangaluru, Paris, London, NYC, Tokyo)
**RL Algorithms:** DDQN, MADDPG, LSTM
**Baseline Methods:** 6 comparative algorithms

---

## 📚 References

This project implements the system model and algorithms described in:

> Y. Liu, C. Yang, Y. Tang, H. Zhao, Y. Liu and S. Xie, "Cost-Efficient Deployment Optimization for Multi-UAV-Assisted Vehicular Edge Computing Networks," in *IEEE Internet of Things Journal*, vol. 12, no. 6, pp. 6158-6169, 15 March 2025, doi: 10.1109/JIOT.2024.3510691.

### Related Work

**Multi-Agent Reinforcement Learning:**
- [MADDPG Paper](https://arxiv.org/abs/1706.02275) - Lowe et al., 2017
- Deep Q-Networks (DQN) - Mnih et al., 2015

**Vehicular Edge Computing:**
- UAV-assisted MEC networks
- Task offloading optimization
- Content caching strategies

**Datasets:**
- [Azure Functions Trace 2019](https://github.com/Azure/AzurePublicDataset)
- [OpenStreetMap](https://www.openstreetmap.org/)

---

## 📄 License

This project is provided for research and educational purposes. Please cite the reference paper if you use this code in your research.

---

## 👥 Contributors

This implementation is based on research work from the IEEE IoT Journal paper. For questions or collaboration:
- Check the [Issues](https://github.com/Rudra-Garg/UAV/issues) page
- See the original paper for theoretical background

---

## 🆕 Recent Updates

- ✅ Dual simulation mode support (SUMO + Python Kinematic)
- ✅ Web-based monitoring dashboard with Streamlit
- ✅ Parallel evaluation for faster benchmarking
- ✅ TensorBoard integration for comprehensive metrics
- ✅ Automated process management scripts
- ✅ Enhanced configuration system with 220+ parameters
- ✅ Multi-city support (8 major cities worldwide)
- ✅ Energy-aware UAV management
- ✅ Real-time visualization with Pygame

---

## 🚦 Quick Start Checklist

- [ ] Install Python 3.8+
- [ ] Run `pip install -r requirements.txt`
- [ ] (Optional) Install SUMO for realistic simulation
- [ ] Run `python tools/generate_traffic.py` (if using SUMO)
- [ ] Run `python tools/process_azure_data.py` (if have Azure data)
- [ ] Run `python tools/generate_task_data.py`
- [ ] Run `python tools/train_cache_predictor.py`
- [ ] Configure [config.py](config.py) settings
- [ ] Run `python main.py` to start training
- [ ] Launch dashboard: `streamlit run dashboard.py`
- [ ] Monitor with TensorBoard: `tensorboard --logdir=runs`
- [ ] Evaluate: `python analysis/evaluate.py --model_path <path>`

**Estimated setup time:** 30-60 minutes
**First training run:** 1-4 hours (depending on configuration)

---

**For detailed dashboard usage, see [DASHBOARD_README.md](DASHBOARD_README.md)**