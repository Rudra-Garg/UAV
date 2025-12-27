# batch_extract_tensorboard.py
"""
Batch process multiple TensorBoard runs to extract all graphs.
Automatically discovers all run directories and processes them.

Usage:
    python batch_extract_tensorboard.py --runs_dir runs --output_base_dir all_tensorboard_plots
"""

import argparse
import os
import subprocess


def find_tensorboard_logs(runs_dir):
    """
    Find all directories containing TensorBoard event files.

    Args:
        runs_dir: Base directory to search

    Returns:
        List of tuples (directory_path, directory_name)
    """
    log_dirs = []

    for root, dirs, files in os.walk(runs_dir):
        # Check if this directory contains TensorBoard event files
        has_events = any(f.startswith('events.out.tfevents') for f in files)
        if has_events:
            rel_path = os.path.relpath(root, runs_dir)
            log_dirs.append((root, rel_path))

    return log_dirs


def sanitize_dirname(name):
    """Convert a path to a safe directory name."""
    return name.replace('\\', '_').replace('/', '_').replace(':', '')


def main():
    parser = argparse.ArgumentParser(
        description='Batch process multiple TensorBoard runs'
    )
    parser.add_argument(
        '--runs_dir',
        type=str,
        default='runs',
        help='Base directory containing TensorBoard runs (default: runs)'
    )
    parser.add_argument(
        '--output_base_dir',
        type=str,
        default='all_tensorboard_plots',
        help='Base directory for all output plots (default: all_tensorboard_plots)'
    )
    parser.add_argument(
        '--smooth',
        type=float,
        default=0.9,
        help='Smoothing factor (default: 0.9)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for output images (default: 300)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='png',
        choices=['png', 'pdf', 'svg'],
        help='Output format (default: png)'
    )

    args = parser.parse_args()

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extract_script = os.path.join(script_dir, 'extract_tensorboard_graphs.py')
    compare_script = os.path.join(script_dir, 'compare_tensorboard_runs.py')

    # Validate runs directory
    if not os.path.exists(args.runs_dir):
        print(f"Error: Runs directory not found: {args.runs_dir}")
        return

    # Find all TensorBoard log directories
    print(f"Searching for TensorBoard logs in: {args.runs_dir}\n")
    log_dirs = find_tensorboard_logs(args.runs_dir)

    if not log_dirs:
        print("No TensorBoard logs found!")
        return

    print(f"Found {len(log_dirs)} TensorBoard run(s):\n")
    for path, name in log_dirs:
        print(f"  - {name}")
    print()

    # Create base output directory
    os.makedirs(args.output_base_dir, exist_ok=True)

    # Process each run
    print("="*60)
    print("Starting batch processing...")
    print("="*60 + "\n")

    success_count = 0
    failed_runs = []

    for idx, (log_path, log_name) in enumerate(log_dirs, 1):
        print(f"[{idx}/{len(log_dirs)}] Processing: {log_name}")

        # Create output directory for this run
        safe_name = sanitize_dirname(log_name)
        output_dir = os.path.join(args.output_base_dir, safe_name)

        # Build command
        cmd = [
            'python', extract_script,
            '--logdir', log_path,
            '--output_dir', output_dir,
            '--smooth', str(args.smooth),
            '--dpi', str(args.dpi),
            '--format', args.format
        ]

        try:
            # Run extraction
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                print(f"  [OK] Success: {output_dir}\n")
                success_count += 1
            else:
                print(f"  [FAIL] Failed with error:\n{result.stderr}\n")
                failed_runs.append(log_name)

        except subprocess.TimeoutExpired:
            print(f"  [FAIL] Timeout (>120s)\n")
            failed_runs.append(log_name)
        except Exception as e:
            print(f"  [FAIL] Error: {e}\n")
            failed_runs.append(log_name)

    # Summary of individual extractions
    print("="*60)
    print("INDIVIDUAL EXTRACTION COMPLETE")
    print("="*60)
    print(f"Total runs: {len(log_dirs)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_runs)}")

    if failed_runs:
        print(f"\nFailed runs:")
        for name in failed_runs:
            print(f"  - {name}")

    print(f"\nIndividual plots location: {os.path.abspath(args.output_base_dir)}")
    print("="*60 + "\n")

    # Generate comparison plots if we have 2 or more successful runs
    comparison_dir = os.path.join(args.output_base_dir, '_comparison_all_runs')

    if success_count >= 2:
        print("="*60)
        print("GENERATING COMPARISON PLOTS")
        print("="*60 + "\n")

        # Get successful log directories
        successful_logs = []
        successful_labels = []

        for log_path, log_name in log_dirs:
            if log_name not in failed_runs:
                successful_logs.append(log_path)
                successful_labels.append(log_name)

        print(f"Comparing {len(successful_logs)} runs:\n")
        for label in successful_labels:
            print(f"  - {label}")
        print()

        # Create comparison output directory (already defined above)

        # Build comparison command
        cmd = ['python', compare_script]
        cmd.extend(['--logdirs'] + successful_logs)
        cmd.extend(['--labels'] + successful_labels)
        cmd.extend(['--output_dir', comparison_dir])
        cmd.extend(['--smooth', str(args.smooth)])
        cmd.extend(['--dpi', str(args.dpi)])

        try:
            print("Running comparison (this may take a moment)...\n")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

            if result.returncode == 0:
                print(f"[OK] Comparison plots generated successfully!")
                print(f"[OK] Location: {os.path.abspath(comparison_dir)}\n")
            else:
                print(f"[FAIL] Comparison failed with error:\n{result.stderr}\n")

        except subprocess.TimeoutExpired:
            print(f"[FAIL] Comparison timeout (>180s)\n")
        except Exception as e:
            print(f"[FAIL] Comparison error: {e}\n")

        print("="*60)
    else:
        print("\nSkipping comparison (need at least 2 successful runs)\n")
        print("="*60)

    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"Individual plots: {os.path.abspath(args.output_base_dir)}")
    if success_count >= 2:
        print(f"Comparison plots: {os.path.abspath(comparison_dir)}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

