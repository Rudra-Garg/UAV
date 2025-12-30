#!/bin/bash
# Start the UAV VECN Simulation Dashboard

echo "🚁 Starting UAV VECN Simulation Dashboard..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit is not installed."
    echo ""
    echo "Installing required packages..."
    pip install streamlit plotly --break-system-packages
    echo ""
fi

# Start the dashboard
echo "🌐 Dashboard will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

nohup streamlit run dashboard.py &
