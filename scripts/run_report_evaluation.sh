#!/bin/bash
# Batch evaluation script to generate report results
# Compares three model configurations (all tested on Python kinematic environment)

set -e

# Default configuration
EPISODES=5
VEHICLES="50 60 70 80 90 100 110 120"
OUTPUT_DIR="report_results"

# Model paths (update these to match your trained models)
PYTHON_NO_LSTM="models/experiment_pythonkinematic_REAC_20251230_114547"
PYTHON_LSTM="models/experiment_pythonkinematic_PRED_20251231_075839"
SUMO_NO_LSTM="models/experiment_sumo_REAC_20260102_100720"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --python-no-lstm)
            PYTHON_NO_LSTM="$2"
            shift 2
            ;;
        --python-lstm)
            PYTHON_LSTM="$2"
            shift 2
            ;;
        --sumo-no-lstm)
            SUMO_NO_LSTM="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --episodes N              Number of episodes per scenario (default: 5)"
            echo "  --output-dir DIR          Output directory for results (default: report_results)"
            echo "  --python-no-lstm PATH     Path to Python w/o LSTM model"
            echo "  --python-lstm PATH        Path to Python w/ LSTM model"
            echo "  --sumo-no-lstm PATH       Path to SUMO-trained model (tested on Python env)"
            echo "  --help                    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "=========================================="
echo "Report Evaluation Configuration"
echo "=========================================="
echo "Episodes per scenario: $EPISODES"
echo "Output directory: $OUTPUT_DIR"
echo "Vehicle densities: $VEHICLES"
echo ""
echo "Model paths:"
echo "  Python w/o LSTM:       $PYTHON_NO_LSTM"
echo "  Python w/ LSTM:        $PYTHON_LSTM"
echo "  SUMO-trained w/o LSTM: $SUMO_NO_LSTM"
echo "=========================================="
echo ""

# Check if model paths exist
if [ ! -d "$PYTHON_NO_LSTM" ]; then
    echo "Error: Python w/o LSTM model not found at $PYTHON_NO_LSTM"
    exit 1
fi

if [ ! -d "$PYTHON_LSTM" ]; then
    echo "Error: Python w/ LSTM model not found at $PYTHON_LSTM"
    exit 1
fi

if [ ! -d "$SUMO_NO_LSTM" ]; then
    echo "Error: SUMO-trained model not found at $SUMO_NO_LSTM"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
echo "Starting evaluation..."
echo ""

python analysis/generate_report_results.py \
    --python_no_lstm "$PYTHON_NO_LSTM" \
    --python_lstm "$PYTHON_LSTM" \
    --sumo_no_lstm "$SUMO_NO_LSTM" \
    --output_dir "$OUTPUT_DIR" \
    --episodes "$EPISODES" \
    --vehicles $VEHICLES

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "Generated plots:"
echo "  - system_profit_analysis.png"
echo "  - average_task_latency.png"
echo "  - total_tasks_completed.png"
echo "  - cache_hit_ratio_analysis.png"
echo "  - task_offloading_distribution.png"
