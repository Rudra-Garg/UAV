# compare_tensorboard_runs.py
"""
Compare multiple TensorBoard runs side-by-side on the same plots.
Generates comparison plots with automatic scaling and smoothing.

Usage:
    python compare_tensorboard_runs.py --logdirs runs/sumo_final runs/muceds_experiment_2025-11-16_15-32-15 --labels "SUMO Final" "MUCEDS 11-16" --output_dir comparison_plots
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def exponential_moving_average(data, alpha=0.9):
    """Apply exponential moving average smoothing."""
    if len(data) == 0:
        return data

    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]

    for i in range(1, len(data)):
        smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * data[i]

    return smoothed


def load_tensorboard_data(logdir):
    """Load all scalar data from TensorBoard event files."""
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    metrics = {}
    scalar_tags = ea.Tags()['scalars']

    for tag in scalar_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        metrics[tag] = (np.array(steps), np.array(values))

    return metrics


def group_metrics_by_category(metrics):
    """Group metrics into logical categories based on their prefix."""
    categories = defaultdict(list)

    for metric_name in metrics.keys():
        if '/' in metric_name:
            category = metric_name.split('/')[0]
        else:
            category = 'Other'
        categories[category].append(metric_name)

    return dict(categories)


def normalize_metric_name(metric_name):
    """Normalize metric names to handle SUMO/ vs Sim/ differences."""
    return metric_name.replace('SUMO/', 'Sim/')


def get_common_metrics(all_runs_metrics):
    """Find metrics that exist in all runs."""
    if not all_runs_metrics:
        return set()

    # Normalize metric names for comparison
    normalized_sets = []
    for metrics in all_runs_metrics:
        normalized = {normalize_metric_name(m) for m in metrics.keys()}
        normalized_sets.append(normalized)

    # Find intersection
    common = set.intersection(*normalized_sets)
    return common


def plot_comparison(all_runs_data, labels, output_path, smooth_alpha=0.9, dpi=300):
    """
    Create comparison plots for all runs.

    Args:
        all_runs_data: List of dictionaries containing metrics for each run
        labels: List of labels for each run
        output_path: Path to save the plot
        smooth_alpha: Smoothing factor
        dpi: DPI for output
    """
    # Find common metrics across all runs
    common_metrics = get_common_metrics(all_runs_data)

    if not common_metrics:
        print("No common metrics found across all runs!")
        return

    # Group metrics by category
    categories = defaultdict(list)
    for metric in common_metrics:
        if '/' in metric:
            category = metric.split('/')[0]
        else:
            category = 'Other'
        categories[category].append(metric)

    # Color palette for different runs
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Create plots for each category
    for category, metric_list in sorted(categories.items()):
        num_metrics = len(metric_list)

        fig_height = 4 * num_metrics
        fig, axes = plt.subplots(num_metrics, 1, figsize=(14, fig_height), squeeze=False)
        axes = axes.flatten()

        fig.suptitle(f'{category} Metrics - Comparison', fontsize=16, fontweight='bold', y=0.995)

        for idx, metric_name in enumerate(sorted(metric_list)):
            ax = axes[idx]

            # Plot each run
            for run_idx, (run_data, label) in enumerate(zip(all_runs_data, labels)):
                # Try to find the metric (with or without SUMO/Sim normalization)
                original_metric = None
                for key in run_data.keys():
                    if normalize_metric_name(key) == metric_name:
                        original_metric = key
                        break

                if original_metric is None:
                    continue

                steps, values = run_data[original_metric]

                # Apply smoothing
                smoothed_values = exponential_moving_average(values, alpha=smooth_alpha)

                color = colors[run_idx % len(colors)]

                # Plot smoothed data
                ax.plot(steps, smoothed_values,
                       alpha=0.9,
                       color=color,
                       linewidth=2,
                       label=label)

            # Format plot
            metric_display_name = metric_name.split('/')[-1].replace('_', ' ').title()
            ax.set_title(metric_display_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Episode', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=9)

        plt.tight_layout()

        output_filename = os.path.join(output_path, f"{category.lower()}_comparison.png")
        plt.savefig(output_filename, dpi=dpi, bbox_inches='tight')
        plt.close()

        print(f"[OK] Saved: {output_filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare multiple TensorBoard runs side-by-side'
    )
    parser.add_argument(
        '--logdirs',
        type=str,
        nargs='+',
        required=True,
        help='Paths to TensorBoard log directories to compare'
    )
    parser.add_argument(
        '--labels',
        type=str,
        nargs='+',
        help='Labels for each run (default: Run 1, Run 2, etc.)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='comparison_plots',
        help='Directory to save comparison plots (default: comparison_plots)'
    )
    parser.add_argument(
        '--smooth',
        type=float,
        default=0.9,
        help='Smoothing factor for exponential moving average (0-1, default: 0.9)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for output images (default: 300)'
    )

    args = parser.parse_args()

    # Validate inputs
    for logdir in args.logdirs:
        if not os.path.exists(logdir):
            print(f"Error: Log directory not found: {logdir}")
            return

    # Generate default labels if not provided
    if args.labels is None:
        args.labels = [f"Run {i+1}" for i in range(len(args.logdirs))]
    elif len(args.labels) != len(args.logdirs):
        print(f"Error: Number of labels ({len(args.labels)}) must match number of log directories ({len(args.logdirs)})")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Comparing {len(args.logdirs)} TensorBoard runs:")
    for label, logdir in zip(args.labels, args.logdirs):
        print(f"  - {label}: {logdir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Smoothing factor: {args.smooth}")
    print(f"DPI: {args.dpi}")
    print(f"{'='*60}\n")

    # Load data from all runs
    all_runs_data = []
    for logdir, label in zip(args.logdirs, args.labels):
        print(f"Loading data from: {label}")
        metrics = load_tensorboard_data(logdir)
        print(f"  Found {len(metrics)} metrics\n")
        all_runs_data.append(metrics)

    # Generate comparison plots
    print("Generating comparison plots...\n")
    plot_comparison(all_runs_data, args.labels, args.output_dir, args.smooth, args.dpi)

    print(f"\n{'='*60}")
    print(f"[OK] All comparison plots generated successfully!")
    print(f"[OK] Output location: {os.path.abspath(args.output_dir)}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

