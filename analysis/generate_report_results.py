"""
Generate Report Results for MUCEDS Framework
Compares three model configurations:
1. MUCEDS (Python w/o LSTM) - Python kinematic trained model with reactive caching (tested on Python env)
2. MUCEDS (Python w/ LSTM) - Python kinematic trained model with predictive caching (tested on Python env)
3. MUCEDS (SUMO-trained w/o LSTM) - SUMO trained model with reactive caching (tested on SUMO env)

Produces plots matching the report figures:
- System Profit Analysis
- Average Task Latency
- Total Tasks Completed (Throughput)
- Cache Hit Ratio Analysis
- Task Offloading Distribution
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from agents.ddqn import DDQNAgent
from agents.maddpg import MADDPGController
from simulation.environment import VECNEnvironment
from prediction.model import LSTMCachePredictor


def load_model(model_path, simulation_mode, use_lstm):
    """Load DDQN agent and MADDPG controllers from model path."""
    # Set configuration
    import config
    config.SIMULATION_MODE = simulation_mode
    config.USE_PREDICTIVE_CACHING = use_lstm
    
    # Load DDQN
    ddqn_agent = DDQNAgent(state_dim=DDQN_STATE_DIM, action_space=DDQN_ACTION_SPACE)
    ddqn_path = os.path.join(model_path, "ddqn_policy_net.pth")
    
    if not os.path.exists(ddqn_path):
        raise FileNotFoundError(f"DDQN model not found at {ddqn_path}")
    
    ddqn_agent.load(model_path)
    
    # Load MADDPG controllers
    # Check if using directory-based format (Python kinematic) or flat format (SUMO)
    maddpg_controllers = {}
    
    # First, check for directory-based format with single directory (e.g., maddpg_44_agents)
    # Find any maddpg_*_agents directory
    maddpg_dir = None
    maddpg_num_agents = None
    for item in os.listdir(model_path):
        item_path = os.path.join(model_path, item)
        if os.path.isdir(item_path) and item.startswith('maddpg_') and item.endswith('_agents'):
            # Extract number of agents from directory name (e.g., maddpg_44_agents -> 44)
            try:
                maddpg_num_agents = int(item.replace('maddpg_', '').replace('_agents', ''))
                maddpg_dir = item_path
                break
            except ValueError:
                continue
    
    # Also check for flat format (SUMO) - files like maddpg_actor_0_10uavs.pth
    has_flat_format = any(
        f.startswith('maddpg_actor_0_') and f.endswith('uavs.pth')
        for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))
    )
    
    if maddpg_dir is not None:
        # Directory-based format (Python kinematic models) - single directory with all agents
        print(f"Loading MADDPG controllers (directory format) from {maddpg_dir}")
        print(f"Found model trained with {maddpg_num_agents} agents")
        
        # Check if actor files exist in the directory
        if os.path.exists(os.path.join(maddpg_dir, 'maddpg_actor_0.pth')):
            # Create controller for the number of agents this model was trained with
            controller = MADDPGController(num_agents=maddpg_num_agents, state_dim=MADDPG_STATE_DIM, action_dim=MADDPG_ACTION_DIM)
            controller.load(maddpg_dir)
            # Store with all possible UAV counts (the model can handle variable number by using subset)
            for i in range(1, DDQN_ACTION_SPACE + 1):
                maddpg_controllers[i] = controller
    elif has_flat_format:
        # Flat format with suffix (SUMO models)
        print(f"Loading MADDPG controllers (flat format) from {model_path}")
        # Check which UAV counts have models
        for num_uavs in range(1, DDQN_ACTION_SPACE + 1):
            # Check if files exist for this UAV count
            actor_0_file = os.path.join(model_path, f"maddpg_actor_0_{num_uavs}uavs.pth")
            critic_file = os.path.join(model_path, f"maddpg_critic_{num_uavs}uavs.pth")
            
            if os.path.exists(actor_0_file) and os.path.exists(critic_file):
                controller = MADDPGController(num_agents=num_uavs, state_dim=MADDPG_STATE_DIM, action_dim=MADDPG_ACTION_DIM)
                # Load with suffix
                controller.load(model_path, suffix=f"_{num_uavs}uavs")
                maddpg_controllers[num_uavs] = controller
    else:
        print(f"WARNING: No MADDPG controllers found in {model_path}")
    
    print(f"Loaded {len(maddpg_controllers)} MADDPG controllers")
    
    # Load LSTM predictor if using predictive caching
    predictor = None
    if use_lstm:
        predictor_path = "models/lstm_cache_predictor.pth"
        if os.path.exists(predictor_path):
            predictor = LSTMCachePredictor()
            predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
            predictor.eval()
            print(f"Loaded LSTM predictor from {predictor_path}")
    
    return ddqn_agent, maddpg_controllers, predictor


def run_evaluation_episode(ddqn_agent, maddpg_controllers, predictor, 
                           num_vehicles, episode_seed, simulation_mode='PYTHON_KINEMATIC'):
    """Run a single evaluation episode and collect metrics."""
    # Set configuration based on simulation mode
    import config
    config.SIMULATION_MODE = simulation_mode
    config.USE_PREDICTIVE_CACHING = (predictor is not None)
    
    # Create environment with the specified simulation mode
    env = VECNEnvironment()
    if predictor:
        env.set_predictor(predictor)
    
    # Set seed
    np.random.seed(episode_seed)
    
    # Select number of UAVs using DDQN
    outer_state = np.zeros(DDQN_STATE_DIM)
    num_uavs = ddqn_agent.select_action(outer_state, evaluation=True) + 1
    
    # Get MADDPG controller for selected number of UAVs
    maddpg_controller = maddpg_controllers.get(num_uavs)
    
    # Reset environment
    inner_states = env.reset(num_uavs=num_uavs, num_vehicles=num_vehicles)
    
    if not env.uavs:
        return {
            'profit': 0,
            'tasks_completed': 0,
            'avg_latency': INNER_STEPS,
            'cache_hits': 0,
            'cache_total': 0,
            'local_tasks': 0,
            'relay_tasks': 0,
            'cloud_tasks': 0,
            'num_uavs': 0
        }
    
    # Track cache statistics
    cache_hits = 0
    cache_requests = 0
    
    # Run episode
    for step in range(INNER_STEPS):
        # Select actions using MADDPG
        if maddpg_controller:
            actions = maddpg_controller.select_actions(inner_states, evaluation=True)
        else:
            # Hover if no controller available
            actions = [np.zeros(MADDPG_ACTION_DIM) for _ in range(num_uavs)]
        
        # Update cache if using predictor
        if predictor and step % CACHE_UPDATE_INTERVAL == 0:
            recent_requests = env.get_recent_requests(PREDICTION_SEQUENCE_LENGTH)
            if len(recent_requests) >= PREDICTION_SEQUENCE_LENGTH:
                for uav in env.uavs:
                    zone_id = env._get_zone_id(uav.position)
                    top_services, top_content = predictor.predict_top_k(
                        recent_requests, zone_id,
                        k_services=SERVICE_CACHE_SIZE,
                        k_content=CONTENT_CACHE_SIZE
                    )
                    uav.update_cache_from_prediction(top_services, top_content)
        
        # Step environment
        next_inner_states, _, done = env.step(actions)
        inner_states = next_inner_states
        
        if done:
            break
    
    # Get final statistics
    final_state = env.get_ddqn_state()
    tasks_completed = final_state[0]
    avg_latency = final_state[4] if tasks_completed > 0 else INNER_STEPS
    profit = final_state[3]
    
    # Get offloading and cache statistics
    stats = env.get_episode_statistics()
    local_tasks = stats.get('Offloading/local', 0)
    cloud_tasks = stats.get('Offloading/cloud', 0)
    relay_tasks = stats.get('Offloading/relay', 0)
    cache_hits = stats.get('cache_hits', 0)
    cache_requests = stats.get('cache_requests', 0)
    
    # Calculate cache hit ratio
    cache_hit_ratio = (cache_hits / cache_requests * 100) if cache_requests > 0 else 0
    
    env.close()
    
    return {
        'profit': profit,
        'tasks_completed': tasks_completed,
        'avg_latency': avg_latency,
        'cache_hits': cache_hits,
        'cache_total': cache_requests,
        'cache_hit_ratio': cache_hit_ratio,
        'local_tasks': local_tasks,
        'relay_tasks': relay_tasks,
        'cloud_tasks': cloud_tasks,
        'num_uavs': num_uavs
    }


def aggregate_results(results_list):
    """Aggregate results from multiple episodes."""
    if not results_list:
        return None
    
    aggregated = {}
    for key in results_list[0].keys():
        values = [r[key] for r in results_list if r is not None]
        if values:
            aggregated[key] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
        else:
            aggregated[key] = 0
            aggregated[f'{key}_std'] = 0
    
    return aggregated


def plot_system_profit(all_results, vehicle_scenarios, output_dir):
    """Generate System Profit Analysis plot (Figure 3)."""
    plt.figure(figsize=(10, 6))
    
    colors = {'Python w/o LSTM': 'gray', 'Python w/ LSTM': 'blue', 'SUMO-trained w/o LSTM': 'red'}
    linestyles = {'Python w/o LSTM': '--', 'Python w/ LSTM': '-', 'SUMO-trained w/o LSTM': '-.'}
    markers = {'Python w/o LSTM': 'o', 'Python w/ LSTM': 's', 'SUMO-trained w/o LSTM': '^'}
    
    for model_name, results in all_results.items():
        x_values = []
        y_values = []
        
        for num_v in vehicle_scenarios:
            if num_v in results and results[num_v] is not None:
                x_values.append(num_v)
                y_values.append(results[num_v]['profit'])
        
        if x_values:
            plt.plot(x_values, y_values, 
                    marker=markers[model_name],
                    linestyle=linestyles[model_name],
                    color=colors[model_name],
                    label=f'MUCEDS ({model_name})',
                    linewidth=2,
                    markersize=8)
    
    plt.xlabel('Number of Vehicle Users', fontsize=14)
    plt.ylabel('Average System Profit', fontsize=14)
    plt.title('System Profit Analysis', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'system_profit_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_task_latency(all_results, vehicle_scenarios, output_dir):
    """Generate Average Task Latency plot (Figure 4)."""
    plt.figure(figsize=(10, 6))
    
    colors = {'Python w/o LSTM': 'gray', 'Python w/ LSTM': 'blue', 'SUMO-trained w/o LSTM': 'red'}
    linestyles = {'Python w/o LSTM': '--', 'Python w/ LSTM': '-', 'SUMO-trained w/o LSTM': '-.'}
    markers = {'Python w/o LSTM': 'o', 'Python w/ LSTM': 's', 'SUMO-trained w/o LSTM': '^'}
    
    for model_name, results in all_results.items():
        x_values = []
        y_values = []
        
        for num_v in vehicle_scenarios:
            if num_v in results and results[num_v] is not None:
                x_values.append(num_v)
                y_values.append(results[num_v]['avg_latency'])
        
        if x_values:
            plt.plot(x_values, y_values,
                    marker=markers[model_name],
                    linestyle=linestyles[model_name],
                    color=colors[model_name],
                    label=f'MUCEDS ({model_name})',
                    linewidth=2,
                    markersize=8)
    
    plt.xlabel('Number of Vehicle Users', fontsize=14)
    plt.ylabel('Latency (time steps)', fontsize=14)
    plt.title('Average Task Latency', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'average_task_latency.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_tasks_completed(all_results, vehicle_scenarios, output_dir):
    """Generate Total Tasks Completed plot (Figure 5)."""
    plt.figure(figsize=(10, 6))
    
    colors = {'Python w/o LSTM': 'gray', 'Python w/ LSTM': 'blue', 'SUMO-trained w/o LSTM': 'red'}
    linestyles = {'Python w/o LSTM': '--', 'Python w/ LSTM': '-', 'SUMO-trained w/o LSTM': '-.'}
    markers = {'Python w/o LSTM': 'o', 'Python w/ LSTM': 's', 'SUMO-trained w/o LSTM': '^'}
    
    for model_name, results in all_results.items():
        x_values = []
        y_values = []
        
        for num_v in vehicle_scenarios:
            if num_v in results and results[num_v] is not None:
                x_values.append(num_v)
                y_values.append(results[num_v]['tasks_completed'])
        
        if x_values:
            plt.plot(x_values, y_values,
                    marker=markers[model_name],
                    linestyle=linestyles[model_name],
                    color=colors[model_name],
                    label=f'MUCEDS ({model_name})',
                    linewidth=2,
                    markersize=8)
    
    plt.xlabel('Number of Vehicle Users', fontsize=14)
    plt.ylabel('Tasks Completed', fontsize=14)
    plt.title('Total Tasks Completed', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'total_tasks_completed.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_cache_hit_ratio(all_results, vehicle_scenarios, output_dir):
    """Generate Cache Hit Ratio Analysis plot (Figure 6)."""
    plt.figure(figsize=(10, 6))
    
    # Only plot models with LSTM capability
    models_to_plot = {
        'Python w/o LSTM': {'color': 'gray', 'linestyle': '--', 'marker': 'o'},
        'Python w/ LSTM': {'color': 'blue', 'linestyle': '-', 'marker': 's'}
    }
    
    for model_name, style in models_to_plot.items():
        if model_name in all_results:
            results = all_results[model_name]
            x_values = []
            y_values = []
            
            for num_v in vehicle_scenarios:
                if num_v in results and results[num_v] is not None:
                    x_values.append(num_v)
                    y_values.append(results[num_v]['cache_hit_ratio'])
            
            if x_values:
                plt.plot(x_values, y_values,
                        marker=style['marker'],
                        linestyle=style['linestyle'],
                        color=style['color'],
                        label=f'MUCEDS ({model_name})',
                        linewidth=2,
                        markersize=8)
    
    plt.xlabel('Number of Vehicle Users', fontsize=14)
    plt.ylabel('Cache Hit Percentage (%)', fontsize=14)
    plt.title('Cache Hit Ratio Analysis', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'cache_hit_ratio_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_offloading_distribution(all_results, vehicle_scenarios, output_dir):
    """Generate Task Offloading Distribution plot (Figure 7)."""
    # Select a representative vehicle density (e.g., 100 vehicles)
    target_vehicles = 100
    
    # Find closest available scenario
    available_scenarios = list(vehicle_scenarios)
    if target_vehicles in available_scenarios:
        selected_scenario = target_vehicles
    else:
        selected_scenario = min(available_scenarios, key=lambda x: abs(x - target_vehicles))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Models to compare
    model_names = ['Python w/o LSTM', 'Python w/ LSTM']
    colors = {'Local (Cache Hit)': 'green', 'Relay (Partial)': 'orange', 'Cloud (Offload)': 'red'}
    
    for idx, model_name in enumerate(model_names):
        if model_name not in all_results:
            continue
        
        results = all_results[model_name]
        if selected_scenario not in results or results[selected_scenario] is None:
            continue
        
        data = results[selected_scenario]
        total_tasks = data['tasks_completed']
        
        if total_tasks == 0:
            continue
        
        # Calculate percentages
        local_pct = (data['local_tasks'] / total_tasks) * 100
        relay_pct = (data['relay_tasks'] / total_tasks) * 100
        cloud_pct = (data['cloud_tasks'] / total_tasks) * 100
        
        # Create stacked bar
        ax = ax1 if idx == 0 else ax2
        
        bottom = 0
        for category, pct in [('Local (Cache Hit)', local_pct), 
                              ('Relay (Partial)', relay_pct),
                              ('Cloud (Offload)', cloud_pct)]:
            ax.bar(0, pct, bottom=bottom, color=colors[category], 
                   label=category if idx == 1 else '', width=0.6)
            # Add percentage label
            if pct > 5:  # Only show label if significant
                ax.text(0, bottom + pct/2, f'{pct:.1f}%', 
                       ha='center', va='center', fontweight='bold', fontsize=11)
            bottom += pct
        
        ax.set_title(f'MUCEDS\n({model_name})', fontsize=13, fontweight='bold')
        ax.set_ylabel('Percentage of Tasks (%)', fontsize=12)
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.grid(axis='y', alpha=0.3)
    
    # Add shared legend
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
               ncol=3, fontsize=11, frameon=True)
    
    plt.suptitle('Task Offloading Distribution (LSTM Impact)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    output_path = os.path.join(output_dir, 'task_offloading_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Report Results for MUCEDS Framework")
    
    # Model paths
    parser.add_argument('--python_no_lstm', type=str, required=True,
                       help='Path to Python kinematic trained model without LSTM')
    parser.add_argument('--python_lstm', type=str, required=True,
                       help='Path to Python kinematic trained model with LSTM')
    parser.add_argument('--sumo_no_lstm', type=str, required=True,
                       help='Path to SUMO trained model without LSTM (tested on SUMO env)')
    
    # Evaluation settings
    parser.add_argument('--output_dir', type=str, default='report_results',
                       help='Directory to save results')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes per scenario')
    parser.add_argument('--vehicles', type=int, nargs='+', default=[50, 60, 70, 80, 90, 100, 110, 120],
                       help='Vehicle densities to evaluate')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Model configurations with their respective simulation modes
    model_configs = {
        'SUMO-trained w/o LSTM': {
            'path': args.sumo_no_lstm,
            'use_lstm': False,
            'simulation_mode': 'SUMO'
        },
        'Python w/o LSTM': {
            'path': args.python_no_lstm,
            'use_lstm': False,
            'simulation_mode': 'PYTHON_KINEMATIC'
        },
        'Python w/ LSTM': {
            'path': args.python_lstm,
            'use_lstm': True,
            'simulation_mode': 'PYTHON_KINEMATIC'
        }
    }
    
    # Results storage: {model_name: {num_vehicles: aggregated_metrics}}
    all_results = {}
    
    # Evaluate each model
    for model_name, config in model_configs.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        # Load model
        print(f"Loading model from: {config['path']}")
        print(f"Simulation mode: {config['simulation_mode']}")
        ddqn_agent, maddpg_controllers, predictor = load_model(
            config['path'],
            config['simulation_mode'],
            config['use_lstm']
        )
        
        model_results = {}
        
        # Evaluate across vehicle scenarios
        for num_vehicles in args.vehicles:
            print(f"\nEvaluating with {num_vehicles} vehicles...")
            episode_results = []
            
            for episode in tqdm(range(args.episodes), desc=f"Episodes"):
                seed = num_vehicles * 1000 + episode
                
                try:
                    result = run_evaluation_episode(
                        ddqn_agent, maddpg_controllers, predictor,
                        num_vehicles, seed, config['simulation_mode']
                    )
                    episode_results.append(result)
                except Exception as e:
                    print(f"Error in episode {episode}: {e}")
                    continue
            
            # Aggregate results
            aggregated = aggregate_results(episode_results)
            model_results[num_vehicles] = aggregated
            
            if aggregated:
                print(f"  Profit: {aggregated['profit']:.2f}")
                print(f"  Tasks: {aggregated['tasks_completed']:.1f}")
                print(f"  Latency: {aggregated['avg_latency']:.2f}")
                print(f"  Cache Hit: {aggregated['cache_hit_ratio']:.2f}%")
        
        all_results[model_name] = model_results
    
    # Generate plots
    print(f"\n{'='*60}")
    print("Generating Report Plots")
    print(f"{'='*60}")
    
    plot_system_profit(all_results, args.vehicles, args.output_dir)
    plot_task_latency(all_results, args.vehicles, args.output_dir)
    plot_tasks_completed(all_results, args.vehicles, args.output_dir)
    plot_cache_hit_ratio(all_results, args.vehicles, args.output_dir)
    plot_offloading_distribution(all_results, args.vehicles, args.output_dir)
    
    print(f"\nAll plots saved to: {args.output_dir}")
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
