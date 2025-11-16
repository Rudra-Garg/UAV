# extract_tensorboard_graphs.py
"""
Extracts and visualizes TensorBoard logs with automatic scaling and smoothing.
Generates high-quality, publication-ready plots with one image per metric section.

Usage:
    python extract_tensorboard_graphs.py --logdir runs/muceds_experiment_2025-11-16_15-32-15
    python extract_tensorboard_graphs.py --logdir runs/sumo_final --output_dir tensorboard_plots
    python extract_tensorboard_graphs.py --logdir runs/sumo_final --smooth 0.9 --dpi 300
"""

import argparse
import os
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def exponential_moving_average(data, alpha=0.9):
    """
    Apply exponential moving average smoothing.

    Args:
        data: Array of values to smooth
        alpha: Smoothing factor (0-1). Higher = smoother

    Returns:
        Smoothed array
    """
    if len(data) == 0:
        return data

    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]

    for i in range(1, len(data)):
        smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * data[i]

    return smoothed


def load_tensorboard_data(logdir):
    """
    Load all scalar data from TensorBoard event files.

    Args:
        logdir: Path to TensorBoard log directory

    Returns:
        Dictionary mapping metric names to (steps, values) tuples
    """
    print(f"Loading TensorBoard data from: {logdir}")

    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    metrics = {}
    scalar_tags = ea.Tags()['scalars']

    print(f"Found {len(scalar_tags)} metrics:")
    for tag in scalar_tags:
        print(f"  - {tag}")
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        metrics[tag] = (np.array(steps), np.array(values))

    return metrics


def group_metrics_by_category(metrics):
    """
    Group metrics into logical categories based on their prefix.

    Args:
        metrics: Dictionary of metric name -> (steps, values)

    Returns:
        Dictionary of category -> list of metric names
    """
    categories = defaultdict(list)

    for metric_name in metrics.keys():
        if '/' in metric_name:
            category = metric_name.split('/')[0]
        else:
            category = 'Other'
        categories[category].append(metric_name)

    return dict(categories)


def plot_metric_category(metrics_dict, category_metrics, output_path, smooth_alpha=0.9, dpi=300):
    """
    Create a high-quality plot for a single metric category.

    Args:
        metrics_dict: Dictionary of all metrics
        category_metrics: List of metric names in this category
        output_path: Path to save the plot
        smooth_alpha: Smoothing factor for exponential moving average
        dpi: DPI for output image
    """
    num_metrics = len(category_metrics)

    if num_metrics == 0:
        return

    # Create figure with appropriate size
    fig_height = 4 * num_metrics
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, fig_height), squeeze=False)
    axes = axes.flatten()

    category_name = category_metrics[0].split('/')[0] if '/' in category_metrics[0] else 'Metrics'
    fig.suptitle(f'{category_name} Metrics', fontsize=16, fontweight='bold', y=0.995)

    for idx, metric_name in enumerate(sorted(category_metrics)):
        ax = axes[idx]
        steps, values = metrics_dict[metric_name]

        # Apply smoothing
        smoothed_values = exponential_moving_average(values, alpha=smooth_alpha)

        # Plot both raw (faint) and smoothed (bold) data
        ax.plot(steps, values, alpha=0.2, color='blue', linewidth=0.5, label='Raw')
        ax.plot(steps, smoothed_values, alpha=0.9, color='blue', linewidth=2, label='Smoothed')

        # Format plot
        metric_display_name = metric_name.split('/')[-1].replace('_', ' ').title()
        ax.set_title(metric_display_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=8)

        # Automatic scaling with 5% padding
        y_min, y_max = np.min(values), np.max(values)
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"[OK] Saved: {output_path}")


def save_raw_data(metrics, output_dir):
    """
    Save raw extracted data to JSON for further analysis.

    Args:
        metrics: Dictionary of metrics
        output_dir: Directory to save JSON file
    """
    output_data = {}

    for metric_name, (steps, values) in metrics.items():
        output_data[metric_name] = {
            'steps': steps.tolist(),
            'values': values.tolist()
        }

    json_path = os.path.join(output_dir, 'tensorboard_data.json')
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"[OK] Saved raw data: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract and visualize TensorBoard logs with automatic scaling and smoothing'
    )
    parser.add_argument(
        '--logdir',
        type=str,
        required=True,
        help='Path to TensorBoard log directory (e.g., runs/muceds_experiment_2025-11-16_15-32-15)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='tensorboard_plots',
        help='Directory to save output plots (default: tensorboard_plots)'
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
    parser.add_argument(
        '--format',
        type=str,
        default='png',
        choices=['png', 'pdf', 'svg'],
        help='Output file format (default: png)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.logdir):
        print(f"Error: Log directory not found: {args.logdir}")
        return

    if not 0 <= args.smooth <= 1:
        print(f"Error: Smoothing factor must be between 0 and 1")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Smoothing factor: {args.smooth}")
    print(f"DPI: {args.dpi}\n")

    # Load data
    metrics = load_tensorboard_data(args.logdir)

    if not metrics:
        print("No metrics found in the log directory!")
        return

    # Group metrics by category
    categories = group_metrics_by_category(metrics)

    print(f"\nGenerating plots for {len(categories)} categories...\n")

    # Create one plot per category
    for category, metric_names in sorted(categories.items()):
        output_filename = f"{category.lower()}_metrics.{args.format}"
        output_path = os.path.join(args.output_dir, output_filename)

        plot_metric_category(
            metrics,
            metric_names,
            output_path,
            smooth_alpha=args.smooth,
            dpi=args.dpi
        )

    # Save raw data
    save_raw_data(metrics, args.output_dir)

    print(f"\n{'='*60}")
    print(f"[OK] All plots generated successfully!")
    print(f"[OK] Output location: {os.path.abspath(args.output_dir)}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

