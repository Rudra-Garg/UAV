# UAV VECN Simulation Dashboard

A comprehensive web-based dashboard for monitoring and controlling UAV VECN simulations.

## Features

### 🎮 Process Control
- **Start/Stop Training**: Launch or terminate training runs with a single click
- **Real-time Status**: Monitor whether training is currently running
- **Process Management**: Automatic PID tracking and cleanup

### 📈 Metrics & Plots
- **TensorBoard Integration**: View all TensorBoard metrics directly in the dashboard
- **Interactive Plots**: Plotly-based interactive visualizations
- **Metric Categories**: Organized by Profit, System, Time, Tasks, UAV, Energy, and Cache metrics
- **Statistics**: View latest, mean, max, and min values for each metric
- **Multiple Runs**: Switch between different training runs

### 📋 Log Monitoring
- **Real-time Logs**: View and tail log files
- **Search Functionality**: Search through logs for specific terms
- **File Info**: Display file size, modification time, and line count
- **Download Logs**: Export log files for offline analysis

### ⚙️ Configuration Editor
- **Live Editing**: Edit `config.py` directly in the dashboard
- **Auto-Backup**: Automatic backup creation before saving
- **Restore**: Restore from previous configuration backups
- **Syntax Highlighting**: Python syntax highlighting for better readability

### ℹ️ System Information
- **Project Overview**: View directory structure and file counts
- **Environment Info**: Check Python, PyTorch, and CUDA versions
- **Recent Activity**: See the latest runs and logs

## Installation

1. Install the required dependencies:
```bash
pip install streamlit plotly tensorboard
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Dashboard

Run the dashboard with:
```bash
streamlit run dashboard.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

### Alternative Port

To run on a different port:
```bash
streamlit run dashboard.py --server.port 8502
```

### Running on a Remote Server

If running on a remote server, you can access it by:
```bash
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501
```

Then access via: `http://<your-server-ip>:8501`

## Dashboard Sections

### 1. Control Panel (Sidebar)
- View current training status
- Start/Stop training with buttons
- Navigate between different views
- Enable auto-refresh (10-second intervals)

### 2. Metrics & Plots
- Select from available TensorBoard runs
- View categorized metrics
- Select specific metrics to display
- See interactive plots with zoom, pan, and hover capabilities
- View statistical summaries

### 3. Logs
- Select from available log files
- View file metadata (size, modification time, line count)
- Adjust number of lines to display
- Search within logs
- Download log files

### 4. Configuration
- Edit the entire `config.py` file
- Save changes with automatic backup
- Reload from file or reset changes
- View and restore from previous backups

### 5. System Info
- View project structure and file counts
- Check Python environment and dependencies
- See recent activity
- View current working directory

## Tips

### Auto-Refresh
Enable auto-refresh in the sidebar to automatically update the dashboard every 10 seconds - useful for monitoring active training runs.

### Multiple Metrics
In the Metrics & Plots view, you can select multiple metrics within each category to compare them side by side.

### Log Search
Use the search functionality in the Logs view to quickly find specific events like errors, warnings, or episode markers.

### Configuration Backups
The dashboard automatically creates timestamped backups of `config.py` every time you save changes. These backups are stored in the same directory and can be restored at any time.

### Process Management
The dashboard uses a PID file (`.training_pid`) to track running training processes. If you manually stop a training process, the dashboard will detect it and update the status accordingly.

## Troubleshooting

### Dashboard won't start
- Make sure Streamlit is installed: `pip install streamlit`
- Check if port 8501 is already in use

### TensorBoard metrics not showing
- Ensure TensorBoard is installed: `pip install tensorboard`
- Verify that runs exist in the `runs/` directory
- Check that the run contains event files

### Can't start training
- Verify `main.py` exists in the current directory
- Check that all dependencies are installed
- Ensure no other training process is already running

### Configuration changes not taking effect
- Configuration changes only affect NEW training runs
- Stop current training and restart for changes to take effect

## File Structure

```
UAV/
├── dashboard.py              # Main dashboard application
├── main.py                   # Training script
├── config.py                 # Configuration file
├── logs/                     # Training logs
│   └── training_log_*.log
├── runs/                     # TensorBoard event files
│   └── experiment_*/
└── models/                   # Saved model checkpoints
```

## Dependencies

- `streamlit` - Web dashboard framework
- `plotly` - Interactive plotting
- `tensorboard` - For reading event files
- `pandas` - Data manipulation
- `numpy` - Numerical operations

## Security Notes

⚠️ **Warning**: This dashboard is designed for local use or trusted environments. It allows:
- Starting/stopping system processes
- Editing configuration files
- Viewing log files

Do not expose this dashboard to untrusted networks without proper authentication and security measures.

## License

Same as the main UAV VECN project.
