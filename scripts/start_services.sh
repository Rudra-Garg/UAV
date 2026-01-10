#!/bin/bash
# Start UAV VECN Simulation Dashboard and TensorBoard services in background

# Activate virtual environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -d "$SCRIPT_DIR/../venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "✓ Virtual environment activated"
else
    echo "⚠ Warning: venv directory not found at $SCRIPT_DIR/venv"
fi

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Directories for logs and PIDs
LOG_DIR="./logs"
PID_DIR="./logs/pids"
mkdir -p "$LOG_DIR"
mkdir -p "$PID_DIR"

# PID files
DASHBOARD_PID_FILE="$PID_DIR/dashboard.pid"
TENSORBOARD_PID_FILE="$PID_DIR/tensorboard.pid"

# Log files
DASHBOARD_LOG="$LOG_DIR/dashboard.log"
TENSORBOARD_LOG="$LOG_DIR/tensorboard.log"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}🚁  UAV VECN Simulation Services Launcher${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Function to check if a process is running
is_running() {
    local pid_file=$1
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$pid_file"
            return 1
        fi
    fi
    return 1
}

# Function to stop a service
stop_service() {
    local service_name=$1
    local pid_file=$2
    
    if is_running "$pid_file"; then
        local pid=$(cat "$pid_file")
        echo -e "${YELLOW}Stopping existing $service_name (PID: $pid)...${NC}"
        kill "$pid" 2>/dev/null
        sleep 2
        if ps -p "$pid" > /dev/null 2>&1; then
            echo -e "${RED}Process still running, force killing...${NC}"
            kill -9 "$pid" 2>/dev/null
        fi
        rm -f "$pid_file"
        echo -e "${GREEN}✓ $service_name stopped${NC}"
    fi
}

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo -e "${YELLOW}⚠ Streamlit is not installed.${NC}"
    echo ""
    echo "Installing required packages..."
    pip install streamlit plotly
    echo ""
fi

# Check if tensorboard is installed
if ! command -v tensorboard &> /dev/null; then
    echo -e "${YELLOW}⚠ TensorBoard is not installed.${NC}"
    echo ""
    echo "Installing TensorBoard..."
    pip install tensorboard
    echo ""
fi

# Stop any existing services
echo -e "${BLUE}Checking for existing services...${NC}"
stop_service "Dashboard" "$DASHBOARD_PID_FILE"
stop_service "TensorBoard" "$TENSORBOARD_PID_FILE"
echo ""

# Start Dashboard
echo -e "${GREEN}🌐 Starting Dashboard...${NC}"
nohup streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0 > "$DASHBOARD_LOG" 2>&1 &
DASHBOARD_PID=$!
echo $DASHBOARD_PID > "$DASHBOARD_PID_FILE"
echo -e "${GREEN}✓ Dashboard started (PID: $DASHBOARD_PID)${NC}"
echo -e "   URL: ${BLUE}http://localhost:8501${NC}"
echo -e "   Log: ${DASHBOARD_LOG}"
echo ""

# Start TensorBoard
echo -e "${GREEN}📊 Starting TensorBoard...${NC}"
TENSORBOARD_DIR="./runs"
if [ ! -d "$TENSORBOARD_DIR" ]; then
    echo -e "${YELLOW}⚠ Creating runs directory for TensorBoard logs...${NC}"
    mkdir -p "$TENSORBOARD_DIR"
fi
nohup tensorboard --logdir="$TENSORBOARD_DIR" --port=6006 --host=0.0.0.0 > "$TENSORBOARD_LOG" 2>&1 &
TENSORBOARD_PID=$!
echo $TENSORBOARD_PID > "$TENSORBOARD_PID_FILE"
echo -e "${GREEN}✓ TensorBoard started (PID: $TENSORBOARD_PID)${NC}"
echo -e "   URL: ${BLUE}http://localhost:6006${NC}"
echo -e "   Log: ${TENSORBOARD_LOG}"
echo ""

# Wait a moment for services to initialize
sleep 3

# Check if services are running
echo -e "${BLUE}Verifying services...${NC}"
ALL_GOOD=true

if is_running "$DASHBOARD_PID_FILE"; then
    echo -e "${GREEN}✓ Dashboard is running${NC}"
else
    echo -e "${RED}✗ Dashboard failed to start${NC}"
    echo -e "  Check log: ${DASHBOARD_LOG}"
    ALL_GOOD=false
fi

if is_running "$TENSORBOARD_PID_FILE"; then
    echo -e "${GREEN}✓ TensorBoard is running${NC}"
else
    echo -e "${RED}✗ TensorBoard failed to start${NC}"
    echo -e "  Check log: ${TENSORBOARD_LOG}"
    ALL_GOOD=false
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [ "$ALL_GOOD" = true ]; then
    echo -e "${GREEN}✓ All services started successfully!${NC}"
else
    echo -e "${RED}⚠ Some services failed to start. Check the logs above.${NC}"
fi
echo ""
echo -e "${YELLOW}Service Management:${NC}"
echo -e "  • View Dashboard logs:    tail -f ${DASHBOARD_LOG}"
echo -e "  • View TensorBoard logs:  tail -f ${TENSORBOARD_LOG}"
echo -e "  • Stop all services:      ./stop_services.sh"
echo -e "  • Check service status:   ./check_services.sh"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
