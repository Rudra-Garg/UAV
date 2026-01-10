#!/usr/bin/env python3
"""
Model Verification Script
Checks if model directories have the required structure for evaluation.
"""

import os
import sys


def check_model_structure(model_path, model_name):
    """Verify that a model directory has the required files."""
    print(f"\nChecking: {model_name}")
    print(f"Path: {model_path}")
    print("-" * 60)
    
    issues = []
    
    # Check if directory exists
    if not os.path.exists(model_path):
        print("❌ Directory does not exist!")
        return False
    
    print("✓ Directory exists")
    
    # Check for DDQN model
    ddqn_path = os.path.join(model_path, "ddqn_policy_net.pth")
    if os.path.exists(ddqn_path):
        size_mb = os.path.getsize(ddqn_path) / (1024 * 1024)
        print(f"✓ DDQN model found ({size_mb:.2f} MB)")
    else:
        print("❌ DDQN model (ddqn_policy_net.pth) NOT FOUND")
        issues.append("Missing DDQN model")
    
    # Check for MADDPG controllers
    maddpg_dirs = [d for d in os.listdir(model_path) 
                   if d.startswith('maddpg_') and d.endswith('_agents')]
    
    # Also check for flat format (SUMO models)
    maddpg_flat_files = [f for f in os.listdir(model_path)
                         if f.startswith('maddpg_actor_') and 'uavs.pth' in f]
    
    if maddpg_dirs:
        # Directory-based format (Python kinematic)
        print(f"✓ Found {len(maddpg_dirs)} MADDPG controller directory(ies) (directory format)")
        
        # Check a few for completeness
        for maddpg_dir in sorted(maddpg_dirs)[:3]:  # Check first 3
            maddpg_path = os.path.join(model_path, maddpg_dir)
            actor_0 = os.path.join(maddpg_path, 'maddpg_actor_0.pth')
            # Check for either individual critic files or single critic file
            critic_0 = os.path.join(maddpg_path, 'maddpg_critic_0.pth')
            critic_single = os.path.join(maddpg_path, 'maddpg_critic.pth')
            
            has_actor = os.path.exists(actor_0)
            has_critic = os.path.exists(critic_0) or os.path.exists(critic_single)
            
            if has_actor and has_critic:
                print(f"  ✓ {maddpg_dir}")
            elif has_actor:
                print(f"  ⚠ {maddpg_dir} (missing critic)")
                issues.append(f"Incomplete MADDPG controller: {maddpg_dir}")
            else:
                print(f"  ⚠ {maddpg_dir} (missing actor)")
                issues.append(f"Incomplete MADDPG controller: {maddpg_dir}")
    
    elif maddpg_flat_files:
        # Flat format (SUMO models)
        # Count unique UAV configurations
        uav_counts = set()
        for f in maddpg_flat_files:
            # Extract UAV count from filename like "maddpg_actor_0_10uavs.pth"
            if 'uavs.pth' in f:
                # Split by underscore and get the part before "uavs.pth"
                parts = f.replace('uavs.pth', '').split('_')
                # The last part should be the UAV count
                try:
                    uav_count = int(parts[-1])
                    uav_counts.add(uav_count)
                except (ValueError, IndexError):
                    pass
        
        print(f"✓ Found MADDPG controllers for {len(uav_counts)} UAV configuration(s) (flat format)")
        
        # Check a few configurations
        for uav_count in sorted(list(uav_counts))[:3]:
            actor_0 = os.path.join(model_path, f'maddpg_actor_0_{uav_count}uavs.pth')
            critic = os.path.join(model_path, f'maddpg_critic_{uav_count}uavs.pth')
            
            if os.path.exists(actor_0) and os.path.exists(critic):
                print(f"  ✓ {uav_count} UAVs configuration")
            else:
                print(f"  ⚠ {uav_count} UAVs configuration (incomplete)")
                issues.append(f"Incomplete MADDPG controller for {uav_count} UAVs")
    
    else:
        print("❌ No MADDPG controllers found")
        issues.append("Missing MADDPG controllers")
    
    # Summary
    if issues:
        print("\n⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✅ Model structure is valid!")
        return True


def check_lstm_predictor():
    """Check if LSTM cache predictor exists."""
    print("\nChecking: LSTM Cache Predictor")
    print("-" * 60)
    
    lstm_path = "models/lstm_cache_predictor.pth"
    
    if os.path.exists(lstm_path):
        size_mb = os.path.getsize(lstm_path) / (1024 * 1024)
        print(f"✓ LSTM predictor found ({size_mb:.2f} MB)")
        print("  This will be used for predictive caching models")
        return True
    else:
        print("⚠ LSTM predictor not found")
        print("  Predictive caching (Python w/ LSTM) will not work without this")
        print("  Train the predictor using: tools/train_cache_predictor.py")
        return False


def main():
    print("="*60)
    print("Model Structure Verification Tool")
    print("="*60)
    
    # Model paths - these should match your actual models
    models_to_check = {
        'Python w/o LSTM (Reactive)': 'models/experiment_pythonkinematic_REAC_20251230_114547',
        'Python w/ LSTM (Predictive)': 'models/experiment_pythonkinematic_PRED_20251231_075839',
        'SUMO w/o LSTM (Realistic)': 'models/experiment_sumo_REAC_20260102_100720'
    }
    
    # You can override these with command-line arguments
    if len(sys.argv) > 1:
        print("\nUsing custom model paths from command line:")
        models_to_check = {}
        for i, path in enumerate(sys.argv[1:], 1):
            models_to_check[f'Model {i}'] = path
    
    results = {}
    
    # Check each model
    for name, path in models_to_check.items():
        results[name] = check_model_structure(path, name)
    
    # Check LSTM predictor
    lstm_ok = check_lstm_predictor()
    
    # Final summary
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)
    
    all_ok = all(results.values())
    
    for name, status in results.items():
        status_str = "✅ PASS" if status else "❌ FAIL"
        print(f"{status_str} - {name}")
    
    if lstm_ok:
        print("✅ PASS - LSTM Cache Predictor")
    else:
        print("⚠ WARN - LSTM Cache Predictor (optional)")
    
    print("="*60)
    
    if all_ok:
        print("\n🎉 All models are ready for evaluation!")
        print("\nNext steps:")
        print("1. Run: python scripts/quick_evaluate.py")
        print("   OR")
        print("2. Run: ./scripts/run_report_evaluation.sh")
        return 0
    else:
        print("\n⚠ Some models have issues. Please fix them before running evaluation.")
        print("\nCommon fixes:")
        print("- Ensure models are fully trained and saved")
        print("- Check that model paths in the scripts match your actual model directories")
        print("- Re-train models if files are missing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
