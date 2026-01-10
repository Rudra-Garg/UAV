#!/bin/bash
# Stop UAV VECN Simulation Dashboard and TensorBoard services

# Activate virtual environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

PID_DIR="./logs/pids"
DASHBOARD_PID_FILE="$PID_DIR/dashboard.pid"
TENSORBOARD_PID_FILE="$PID_DIR/tensorboard.pid"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}🛑 Stopping UAV VECN Simulation Services${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Function to stop a service
stop_service() {
    local service_name=$1
    local pid_file=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo -e "${YELLOW}Stopping $service_name (PID: $pid)...${NC}"
            kill "$pid" 2>/dev/null
            sleep 2
            if ps -p "$pid" > /dev/null 2>&1; then
                echo -e "${RED}Process still running, force killing...${NC}"
                kill -9 "$pid" 2>/dev/null
                sleep 1
            fi
            if ! ps -p "$pid" > /dev/null 2>&1; then
                echo -e "${GREEN}✓ $service_name stopped${NC}"
            else
                echo -e "${RED}✗ Failed to stop $service_name${NC}"
            fi
        else
            echo -e "${YELLOW}$service_name is not running (stale PID file)${NC}"
        fi
        rm -f "$pid_file"
    else
        echo -e "${YELLOW}$service_name PID file not found${NC}"
    fi
}

stop_service "Dashboard" "$DASHBOARD_PID_FILE"
stop_service "TensorBoard" "$TENSORBOARD_PID_FILE"

# Stop training if running
if [ -f ".training_pid" ]; then
    echo ""
    echo -e "${YELLOW}Found running training process...${NC}"
    PID=$(head -n 1 .training_pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo -e "${YELLOW}Stopping training (PID: $PID)...${NC}"
        kill -TERM $PID 2>/dev/null
        sleep 2
        if ps -p $PID > /dev/null 2>&1; then
            kill -9 $PID 2>/dev/null
        fi
        if ! ps -p $PID > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Training stopped${NC}"
        fi
    fi
    rm -f .training_pid
fi

# Kill any remaining SUMO processes
SUMO_PIDS=$(pgrep -f "sumo" 2>/dev/null)
if [ ! -z "$SUMO_PIDS" ]; then
    echo ""
    echo -e "${YELLOW}Cleaning up SUMO processes...${NC}"
    pkill -9 sumo 2>/dev/null
    echo -e "${GREEN}✓ SUMO processes terminated${NC}"
fi

echo ""
echo -e "${GREEN}All services stopped${NC}"
echo ""
