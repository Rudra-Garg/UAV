#!/bin/bash
# start_parallel_training.sh
# Quick script to start parallel training with proper logging

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  UAV-VECN Parallel Training Launcher${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if training is already running
if [ -f .training_pid ]; then
    PID=$(head -n 1 .training_pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Training is already running (PID: $PID)${NC}"
        echo "Stop the current training first with: kill $PID"
        exit 1
    else
        rm .training_pid
    fi
fi

# Check if main_parallel.py exists
if [ ! -f "main_parallel.py" ]; then
    echo -e "${YELLOW}⚠️  main_parallel.py not found!${NC}"
    exit 1
fi

# Display system info
echo -e "${GREEN}System Information:${NC}"
echo "  CPU Cores: $(nproc)"
echo "  Available Memory: $(free -h | awk '/^Mem:/ {print $7}')"
echo "  Python: $(python3 --version)"
echo ""

# Ask for confirmation
read -p "Start parallel training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Start training
echo -e "${GREEN}Starting parallel training...${NC}"
echo ""

nohup python3 main_parallel.py > logs/parallel_training_$(date +%Y%m%d_%H%M%S).out 2>&1 &
PID=$!

# Save PID
echo -e "$PID\nmain_parallel.py" > .training_pid

echo -e "${GREEN}✅ Training started successfully!${NC}"
echo "  PID: $PID"
echo "  Log file: logs/parallel_training_*.out"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/training_log_*.log"
echo "  tensorboard --logdir=runs/"
echo ""
echo "Stop training with:"
echo "  kill $PID"
echo "  or: bash scripts/stop_services.sh"
echo ""
