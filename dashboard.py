#!/usr/bin/env python3
"""
UAV VECN Simulation Dashboard
=================================
A comprehensive web-based dashboard for monitoring and controlling UAV simulations.

Features:
- Start/Stop training runs
- Monitor TensorBoard metrics in real-time
- View and tail log files
- Edit configuration settings
- Display system status
"""

import os
import sys
import time
import subprocess
import signal
import glob
import re
from pathlib import Path
from datetime import datetime
import threading

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# TensorBoard event file parsing
try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    event_accumulator = None

# Page configuration
st.set_page_config(
    page_title="UAV VECN Dashboard",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================
MAIN_SCRIPT = "main.py"
MAIN_SCRIPT_PARALLEL = "main_parallel.py"
CONFIG_FILE = "config.py"
LOGS_DIR = "logs"
RUNS_DIR = "runs"
MODELS_DIR = "models"
PID_FILE = ".training_pid"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_timestamp(timestamp, include_date=True, include_time=True):
    """Format a timestamp consistently across the dashboard."""
    dt = datetime.fromtimestamp(timestamp)
    if include_date and include_time:
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    elif include_date:
        return dt.strftime('%Y-%m-%d')
    elif include_time:
        return dt.strftime('%H:%M:%S')
    return str(dt)

def format_seconds_to_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    if pd.isna(seconds) or seconds < 0:
        return "N/A"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def is_time_metric(metric_name):
    """Check if a metric represents time values (in seconds)."""
    time_keywords = ['time', 'duration', 'seconds', 'latency', 'delay', 'remaining']
    metric_lower = metric_name.lower()
    return any(keyword in metric_lower for keyword in time_keywords)

def is_process_running(pid):
    """Check if a process with given PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False

def get_training_pid():
    """Get the PID of running training process if it exists."""
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                lines = f.read().strip().split('\n')
                pid = int(lines[0])
            if is_process_running(pid):
                return pid
            else:
                os.remove(PID_FILE)
        except (ValueError, FileNotFoundError, IndexError):
            pass
    return None

def start_training(script_name=MAIN_SCRIPT):
    """Start the training process in background."""
    try:
        # Start the process
        process = subprocess.Popen(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            preexec_fn=os.setsid  # Create new process group
        )
        
        # Save PID and script name
        with open(PID_FILE, 'w') as f:
            f.write(f"{process.pid}\n{script_name}")
        
        mode = "Parallel" if "parallel" in script_name else "Single-core"
        return process.pid, f"{mode} training started successfully!"
    except Exception as e:
        return None, f"Error starting training: {str(e)}"

def stop_training(pid):
    """Stop the training process."""
    try:
        # Kill the entire process group
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(1)
        
        # Clean up PID file
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        
        return True, "Training stopped successfully!"
    except Exception as e:
        return False, f"Error stopping training: {str(e)}"

def get_log_files():
    """Get list of log files sorted by modification time."""
    if not os.path.exists(LOGS_DIR):
        return []
    
    log_files = glob.glob(os.path.join(LOGS_DIR, "*.log"))
    log_files.sort(key=os.path.getmtime, reverse=True)
    return log_files

def get_tensorboard_runs():
    """Get list of TensorBoard run directories."""
    if not os.path.exists(RUNS_DIR):
        return []
    
    run_dirs = [d for d in glob.glob(os.path.join(RUNS_DIR, "*")) if os.path.isdir(d)]
    run_dirs.sort(key=os.path.getmtime, reverse=True)
    return run_dirs

def read_tensorboard_events(run_dir):
    """Read TensorBoard event files and extract metrics."""
    if event_accumulator is None:
        return None, "TensorBoard not installed. Install with: pip install tensorboard"
    
    try:
        ea = event_accumulator.EventAccumulator(run_dir)
        ea.Reload()
        
        # Get all scalar tags
        tags = ea.Tags().get('scalars', [])
        
        metrics = {}
        for tag in tags:
            events = ea.Scalars(tag)
            metrics[tag] = pd.DataFrame([
                {'step': e.step, 'value': e.value, 'wall_time': e.wall_time}
                for e in events
            ])
        
        return metrics, None
    except Exception as e:
        return None, f"Error reading TensorBoard events: {str(e)}"

def tail_file(filepath, n=100):
    """Read last n lines from a file."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            return ''.join(lines[-n:])
    except Exception as e:
        return f"Error reading file: {str(e)}"

def load_config():
    """Load configuration file content."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return f.read()
    except Exception as e:
        return f"# Error loading config: {str(e)}"

def save_config(content):
    """Save configuration file content."""
    try:
        # Backup original config
        backup_file = f"{CONFIG_FILE}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                backup_content = f.read()
            with open(backup_file, 'w') as f:
                f.write(backup_content)
        
        # Save new config
        with open(CONFIG_FILE, 'w') as f:
            f.write(content)
        
        return True, f"Configuration saved! Backup created: {backup_file}"
    except Exception as e:
        return False, f"Error saving config: {str(e)}"

# ============================================================================
# DASHBOARD UI
# ============================================================================

def main():
    # Header
    st.title("🚁 UAV VECN Simulation Dashboard")
    st.markdown("---")
    
    # Sidebar - Control Panel
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        # Check if training is running
        training_pid = get_training_pid()
        is_running = training_pid is not None
        
        if is_running:
            st.success(f"✅ Training is running (PID: {training_pid})")
            
            if st.button("🛑 Stop Training", type="primary", use_container_width=True):
                success, message = stop_training(training_pid)
                if success:
                    st.success(message)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(message)
        else:
            st.info("⏸️ Training is not running")
            
            # Training mode selection
            st.markdown("**Select Training Mode:**")
            training_mode = st.radio(
                "Mode",
                ["Single-Core", "Parallel (Multi-Core)"],
                help="Parallel mode uses multiple CPU cores for faster training",
                label_visibility="collapsed"
            )
            
            # Show info about selected mode
            if training_mode == "Parallel (Multi-Core)":
                import multiprocessing as mp
                num_cores = mp.cpu_count()
                st.info(f"ℹ️ Will use up to {num_cores-1} of {num_cores} CPU cores")
            
            script = MAIN_SCRIPT_PARALLEL if training_mode == "Parallel (Multi-Core)" else MAIN_SCRIPT
            
            if st.button("▶️ Start Training", type="primary", use_container_width=True):
                pid, message = start_training(script)
                if pid:
                    st.success(message)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(message)
        
        st.markdown("---")
        
        # Navigation
        st.header("📊 Navigation")
        page = st.radio(
            "Select View:",
            ["📈 Metrics & Plots", "📋 Logs", "⚙️ Configuration", "ℹ️ System Info"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Refresh button
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
        
        # Auto-refresh
        st.markdown("---")
        auto_refresh = st.checkbox("Auto-refresh (10s)", value=False)
        if auto_refresh:
            time.sleep(10)
            st.rerun()
    
    # Main content area
    if page == "📈 Metrics & Plots":
        show_metrics_page()
    elif page == "📋 Logs":
        show_logs_page()
    elif page == "⚙️ Configuration":
        show_config_page()
    elif page == "ℹ️ System Info":
        show_system_info_page()

def show_metrics_page():
    """Display TensorBoard metrics and plots."""
    st.header("📈 Training Metrics & Plots")
    
    # Get available runs
    runs = get_tensorboard_runs()
    
    if not runs:
        st.warning("No TensorBoard runs found. Start training to generate metrics.")
        return
    
    # Run selector
    run_names = [os.path.basename(r) for r in runs]
    selected_run_name = st.selectbox("Select Run:", run_names)
    selected_run = runs[run_names.index(selected_run_name)]
    
    st.info(f"📁 Run Directory: `{selected_run}`")
    
    # Load metrics
    with st.spinner("Loading TensorBoard events..."):
        metrics, error = read_tensorboard_events(selected_run)
    
    if error:
        st.error(error)
        return
    
    if not metrics:
        st.warning("No metrics found in this run.")
        return
    
    # Display metrics
    st.subheader("Available Metrics")
    
    # Group metrics by category
    metric_categories = {
        'Profit': [],
        'System': [],
        'Time': [],
        'Tasks': [],
        'UAV': [],
        'Energy': [],
        'Cache': [],
        'Other': []
    }
    
    for tag in metrics.keys():
        categorized = False
        for category in metric_categories.keys():
            if tag.startswith(category):
                metric_categories[category].append(tag)
                categorized = True
                break
        if not categorized:
            metric_categories['Other'].append(tag)
    
    # Display metrics in tabs
    tab_names = [cat for cat in metric_categories.keys() if metric_categories[cat]]
    tabs = st.tabs(tab_names)
    
    for tab, category in zip(tabs, tab_names):
        with tab:
            category_metrics = metric_categories[category]
            
            # Create multi-metric selector
            selected_metrics = st.multiselect(
                f"Select {category} metrics to display:",
                category_metrics,
                default=category_metrics[:min(3, len(category_metrics))],
                key=f"select_{category}"
            )
            
            if not selected_metrics:
                st.info(f"Select metrics to display {category} plots")
                continue
            
            # Plot metrics
            for metric in selected_metrics:
                df = metrics[metric]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['step'],
                    y=df['value'],
                    mode='lines',
                    name=metric,
                    line=dict(width=2)
                ))
                
                fig.update_layout(
                    title=metric,
                    xaxis_title="Episode",
                    yaxis_title="Value",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show statistics
                col1, col2, col3, col4 = st.columns(4)
                
                # Check if this is a time-based metric
                is_time = is_time_metric(metric)
                
                with col1:
                    latest_val = df['value'].iloc[-1]
                    st.metric("Latest", format_seconds_to_time(latest_val) if is_time else f"{latest_val:.4f}")
                with col2:
                    mean_val = df['value'].mean()
                    st.metric("Mean", format_seconds_to_time(mean_val) if is_time else f"{mean_val:.4f}")
                with col3:
                    max_val = df['value'].max()
                    st.metric("Max", format_seconds_to_time(max_val) if is_time else f"{max_val:.4f}")
                with col4:
                    min_val = df['value'].min()
                    st.metric("Min", format_seconds_to_time(min_val) if is_time else f"{min_val:.4f}")

def show_logs_page():
    """Display log files."""
    st.header("📋 Training Logs")
    
    log_files = get_log_files()
    
    if not log_files:
        st.warning("No log files found. Start training to generate logs.")
        return
    
    # Log file selector
    log_names = [os.path.basename(f) for f in log_files]
    selected_log_name = st.selectbox("Select Log File:", log_names)
    selected_log = log_files[log_names.index(selected_log_name)]
    
    # File info
    file_stat = os.stat(selected_log)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File Size", f"{file_stat.st_size / 1024:.2f} KB")
    with col2:
        st.metric("Modified", format_timestamp(file_stat.st_mtime))
    with col3:
        # Count lines
        with open(selected_log, 'r') as f:
            line_count = sum(1 for _ in f)
        st.metric("Lines", line_count)
    
    # Number of lines to display
    num_lines = st.slider("Number of lines to display:", 10, 1000, 100)
    
    # Display log content
    st.subheader("Log Content")
    log_content = tail_file(selected_log, num_lines)
    
    # Search functionality
    search_term = st.text_input("🔍 Search in logs:", "")
    
    if search_term:
        # Highlight search terms
        lines = log_content.split('\n')
        filtered_lines = [line for line in lines if search_term.lower() in line.lower()]
        st.info(f"Found {len(filtered_lines)} matching lines")
        log_content = '\n'.join(filtered_lines)
    
    st.code(log_content, language="log")
    
    # Download button
    st.download_button(
        label="📥 Download Log File",
        data=log_content,
        file_name=selected_log_name,
        mime="text/plain"
    )

def show_config_page():
    """Display and edit configuration."""
    st.header("⚙️ Configuration Editor")
    
    st.warning("⚠️ Warning: Changes to configuration will affect the next training run. Make sure to backup important configurations!")
    
    # Load current config
    if 'config_content' not in st.session_state:
        st.session_state.config_content = load_config()
    
    # Editor
    edited_config = st.text_area(
        "Edit config.py:",
        value=st.session_state.config_content,
        height=600,
        key="config_editor"
    )
    
    # Buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("💾 Save Configuration", type="primary"):
            success, message = save_config(edited_config)
            if success:
                st.success(message)
                st.session_state.config_content = edited_config
            else:
                st.error(message)
    
    with col2:
        if st.button("🔄 Reload from File"):
            st.session_state.config_content = load_config()
            st.rerun()
    
    with col3:
        if st.button("⚠️ Reset to Current"):
            st.rerun()
    
    # Show config backups
    st.markdown("---")
    st.subheader("📦 Configuration Backups")
    
    backup_files = glob.glob(f"{CONFIG_FILE}.backup_*")
    backup_files.sort(reverse=True)
    
    if backup_files:
        backup_names = [os.path.basename(f) for f in backup_files[:10]]  # Show last 10
        selected_backup = st.selectbox("Available Backups:", backup_names)
        
        if selected_backup:
            backup_path = os.path.join(os.path.dirname(CONFIG_FILE), selected_backup)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👁️ Preview Backup"):
                    with open(backup_path, 'r') as f:
                        st.code(f.read(), language="python")
            
            with col2:
                if st.button("♻️ Restore from Backup"):
                    try:
                        with open(backup_path, 'r') as f:
                            backup_content = f.read()
                        st.session_state.config_content = backup_content
                        st.success("Backup loaded! Click 'Save Configuration' to apply.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading backup: {str(e)}")
    else:
        st.info("No backups available yet.")

def show_system_info_page():
    """Display system information."""
    st.header("ℹ️ System Information")
    
    # Project structure
    st.subheader("📁 Project Structure")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Directories:**")
        dirs_info = {
            "Logs": (LOGS_DIR, len(get_log_files())),
            "Models": (MODELS_DIR, len(glob.glob(os.path.join(MODELS_DIR, "*.pth"))) if os.path.exists(MODELS_DIR) else 0),
            "Runs": (RUNS_DIR, len(get_tensorboard_runs())),
        }
        
        for name, (path, count) in dirs_info.items():
            exists = os.path.exists(path)
            st.metric(
                f"{name}",
                f"{count} files" if exists else "Not found",
                f"📁 {path}"
            )
    
    with col2:
        st.markdown("**Files:**")
        files_info = {
            "Main Script (Single)": MAIN_SCRIPT,
            "Main Script (Parallel)": MAIN_SCRIPT_PARALLEL,
            "Configuration": CONFIG_FILE,
        }
        
        for name, path in files_info.items():
            if os.path.exists(path):
                size = os.path.getsize(path)
                st.metric(name, f"{size / 1024:.2f} KB", f"✅ {path}")
            else:
                st.metric(name, "Not found", f"❌ {path}")
    
    # Environment info
    st.markdown("---")
    st.subheader("🖥️ Environment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    with col2:
        # Check PyTorch
        try:
            import torch
            st.metric("PyTorch", torch.__version__)
        except ImportError:
            st.metric("PyTorch", "Not installed")
    
    with col3:
        # Check CUDA
        try:
            import torch
            if torch.cuda.is_available():
                st.metric("CUDA", f"✅ {torch.cuda.get_device_name(0)}")
            else:
                st.metric("CUDA", "Not available")
        except:
            st.metric("CUDA", "Unknown")
    
    # Recent activity
    st.markdown("---")
    st.subheader("📊 Recent Activity")
    
    # Get most recent run
    runs = get_tensorboard_runs()
    if runs:
        latest_run = runs[0]
        st.success(f"Latest Run: {os.path.basename(latest_run)}")
        st.info(f"Created: {format_timestamp(os.path.getctime(latest_run))}")
    
    # Get most recent log
    logs = get_log_files()
    if logs:
        latest_log = logs[0]
        st.success(f"Latest Log: {os.path.basename(latest_log)}")
        st.info(f"Modified: {format_timestamp(os.path.getmtime(latest_log))}")
    
    # Current directory
    st.markdown("---")
    st.subheader("📍 Current Working Directory")
    st.code(os.getcwd())

if __name__ == "__main__":
    main()
