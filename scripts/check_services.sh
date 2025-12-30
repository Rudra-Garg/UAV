#!/bin/bash
# Check status of UAV VECN Simulation services

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
LOG_DIR="./logs"
DASHBOARD_PID_FILE="$PID_DIR/dashboard.pid"
TENSORBOARD_PID_FILE="$PID_DIR/tensorboard.pid"
DASHBOARD_LOG="$LOG_DIR/dashboard.log"
TENSORBOARD_LOG="$LOG_DIR/tensorboard.log"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}📋 UAV VECN Simulation Services Status${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Function to check service status
check_service() {
    local service_name=$1
    local pid_file=$2
    local log_file=$3
    local port=$4
    local url=$5
    
    echo -e "${BLUE}$service_name:${NC}"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            # Get process info
            local uptime=$(ps -p "$pid" -o etime= | xargs)
            local mem=$(ps -p "$pid" -o rss= | xargs)
            local mem_mb=$((mem / 1024))
            
            echo -e "  Status:  ${GREEN}● Running${NC}"
            echo -e "  PID:     $pid"
            echo -e "  Uptime:  $uptime"
            echo -e "  Memory:  ${mem_mb} MB"
            echo -e "  URL:     ${BLUE}$url${NC}"
            
            # Check if port is actually listening
            if command -v netstat &> /dev/null; then
                if netstat -tuln 2>/dev/null | grep -q ":$port "; then
                    echo -e "  Port:    ${GREEN}$port (listening)${NC}"
                else
                    echo -e "  Port:    ${YELLOW}$port (not listening yet)${NC}"
                fi
            elif command -v ss &> /dev/null; then
                if ss -tuln 2>/dev/null | grep -q ":$port "; then
                    echo -e "  Port:    ${GREEN}$port (listening)${NC}"
                else
                    echo -e "  Port:    ${YELLOW}$port (not listening yet)${NC}"
                fi
            fi
            
            # Show last log line
            if [ -f "$log_file" ]; then
                local last_log=$(tail -n 1 "$log_file" 2>/dev/null)
                if [ -n "$last_log" ]; then
                    echo -e "  Last log: ${last_log:0:60}..."
                fi
            fi
        else
            echo -e "  Status:  ${RED}● Not running${NC} (stale PID file)"
            rm -f "$pid_file"
        fi
    else
        echo -e "  Status:  ${RED}● Not running${NC}"
    fi
    echo ""
}

check_service "Dashboard" "$DASHBOARD_PID_FILE" "$DASHBOARD_LOG" "8501" "http://localhost:8501"
check_service "TensorBoard" "$TENSORBOARD_PID_FILE" "$TENSORBOARD_LOG" "6006" "http://localhost:6006"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Commands:${NC}"
echo -e "  • Start services:     ./start_services.sh"
echo -e "  • Stop services:      ./stop_services.sh"
echo -e "  • View logs:          tail -f logs/dashboard.log"
echo -e "                        tail -f logs/tensorboard.log"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
