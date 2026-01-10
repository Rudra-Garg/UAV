#!/usr/bin/env python3
"""
Quick Evaluation Script
Simplified script to quickly evaluate models and generate report figures.
"""

import os
import subprocess
import sys

# Model paths - UPDATE THESE to match your trained models
MODEL_PATHS = {
    'python_no_lstm': 'models/experiment_pythonkinematic_REAC_20251230_114547',
    'python_lstm': 'models/experiment_pythonkinematic_PRED_20251231_075839',
    'sumo_no_lstm': 'models/experiment_sumo_REAC_20260102_100720'
}

# Evaluation settings
EPISODES = 5  # Number of episodes per scenario
VEHICLES = [50, 60, 70, 80, 90, 100, 110, 120]  # Vehicle densities
OUTPUT_DIR = 'report_results'
SUMO_PORT = 8813


def check_models():
    """Check if all model paths exist."""
    missing = []
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            missing.append((name, path))
    
    if missing:
        print("ERROR: The following models were not found:")
        for name, path in missing:
            print(f"  {name}: {path}")
        print("\nPlease update the MODEL_PATHS in this script to match your trained models.")
        sys.exit(1)
    
    print("✓ All model paths verified")


def run_evaluation():
    """Run the evaluation script."""
    print("\n" + "="*60)
    print("Starting Report Evaluation")
    print("="*60)
    print(f"Episodes per scenario: {EPISODES}")
    print(f"Vehicle densities: {VEHICLES}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Construct command
    cmd = [
        'python', 'analysis/generate_report_results.py',
        '--python_no_lstm', MODEL_PATHS['python_no_lstm'],
        '--python_lstm', MODEL_PATHS['python_lstm'],
        '--sumo_no_lstm', MODEL_PATHS['sumo_no_lstm'],
        '--output_dir', OUTPUT_DIR,
        '--episodes', str(EPISODES),
        '--sumo_port', str(SUMO_PORT),
        '--vehicles'
    ] + [str(v) for v in VEHICLES]
    
    # Run evaluation
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Evaluation failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nGenerated plots:")
    plots = [
        'system_profit_analysis.png',
        'average_task_latency.png',
        'total_tasks_completed.png',
        'cache_hit_ratio_analysis.png',
        'task_offloading_distribution.png'
    ]
    for plot in plots:
        print(f"  ✓ {plot}")


def main():
    print("="*60)
    print("MUCEDS Report Results Generator")
    print("="*60)
    
    # Check if models exist
    check_models()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run evaluation
    run_evaluation()


if __name__ == "__main__":
    main()
