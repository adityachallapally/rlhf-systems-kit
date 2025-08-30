"""
RLHF Training Monitor

A Streamlit-based dashboard for monitoring RLHF training metrics, stage times, and profiler data.
"""

import os
import sys
import argparse
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np


def load_metrics(run_path: str) -> Optional[pd.DataFrame]:
    """Load metrics from metrics.jsonl file."""
    metrics_file = Path(run_path) / "metrics.jsonl"
    if not metrics_file.exists():
        return None
    
    try:
        # Read JSONL file
        data = []
        with open(metrics_file, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        if not data:
            return None
            
        df = pd.DataFrame(data)
        
        # Convert numeric columns
        numeric_cols = ['loss', 'reward_mean', 'reward_var', 'kl', 'entropy', 
                       'clip_frac', 'grad_norm', 'lr', 'time_ms']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return None


def load_stage_times(run_path: str) -> Optional[list]:
    """Load stage timing information."""
    stage_file = Path(run_path) / "stage_times.json"
    if not stage_file.exists():
        return None
    
    try:
        with open(stage_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading stage times: {e}")
        return None


def load_sysinfo(run_path: str) -> Optional[Dict[str, Any]]:
    """Load system information."""
    sysinfo_file = Path(run_path) / "sysinfo.json"
    if not sysinfo_file.exists():
        return None
    
    try:
        with open(sysinfo_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading system info: {e}")
        return None


def check_alerts(df: pd.DataFrame) -> list:
    """Check for training alerts based on metrics."""
    alerts = []
    
    if df.empty:
        return alerts
    
    # Alert if KL divergence slope is too high (last 200 steps)
    if 'kl' in df.columns and len(df) >= 10:
        recent_kl = df['kl'].tail(200).dropna()
        if len(recent_kl) >= 10:
            # Calculate rolling slope
            x = np.arange(len(recent_kl))
            slope = np.polyfit(x, recent_kl, 1)[0]
            if abs(slope) > 0.01:  # Threshold for KL slope
                alerts.append({
                    'type': 'warning',
                    'message': f'KL divergence slope is high: {slope:.4f} (last 200 steps)'
                })
    
    # Alert if reward variance is too high
    if 'reward_var' in df.columns:
        recent_var = df['reward_var'].tail(100).dropna()
        if len(recent_var) > 0:
            mean_var = recent_var.mean()
            if mean_var > 2.0:  # Threshold for reward variance
                alerts.append({
                    'type': 'warning',
                    'message': f'High reward variance: {mean_var:.4f} (last 100 steps)'
                })
    
    # Alert if loss is NaN or infinite
    if 'loss' in df.columns:
        if df['loss'].isna().any() or np.isinf(df['loss']).any():
            alerts.append({
                'type': 'error',
                'message': 'Loss contains NaN or infinite values'
            })
    
    return alerts


def create_metrics_plots(df: pd.DataFrame):
    """Create interactive plots for training metrics."""
    if df.empty:
        st.warning("No metrics data available for plotting.")
        return
    
    # Filter out NaN values for plotting
    plot_df = df.dropna(subset=['step'])
    
    if plot_df.empty:
        st.warning("No valid step data for plotting.")
        return
    
    # KL Divergence
    if 'kl' in plot_df.columns:
        st.subheader("KL Divergence")
        kl_data = plot_df[['step', 'kl']].dropna()
        if not kl_data.empty:
            fig = px.line(kl_data, x='step', y='kl', title='KL Divergence Over Time')
            fig.update_layout(xaxis_title='Step', yaxis_title='KL Divergence')
            st.plotly_chart(fig, use_container_width=True)
    
    # Reward Metrics
    if 'reward_mean' in plot_df.columns:
        st.subheader("Reward Metrics")
        reward_data = plot_df[['step', 'reward_mean', 'reward_var']].dropna()
        if not reward_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=reward_data['step'], y=reward_data['reward_mean'], 
                                   mode='lines', name='Reward Mean'))
            if 'reward_var' in reward_data.columns:
                fig.add_trace(go.Scatter(x=reward_data['step'], y=reward_data['reward_var'], 
                                       mode='lines', name='Reward Variance'))
            fig.update_layout(title='Reward Metrics Over Time', xaxis_title='Step', yaxis_title='Reward')
            st.plotly_chart(fig, use_container_width=True)
    
    # Loss
    if 'loss' in plot_df.columns:
        st.subheader("Training Loss")
        loss_data = plot_df[['step', 'loss']].dropna()
        if not loss_data.empty:
            fig = px.line(loss_data, x='step', y='loss', title='Training Loss Over Time')
            fig.update_layout(xaxis_title='Step', yaxis_title='Loss')
            st.plotly_chart(fig, use_container_width=True)
    
    # Other metrics
    other_metrics = ['entropy', 'clip_frac', 'grad_norm']
    available_metrics = [m for m in other_metrics if m in plot_df.columns]
    
    if available_metrics:
        st.subheader("Other Metrics")
        for metric in available_metrics:
            metric_data = plot_df[['step', metric]].dropna()
            if not metric_data.empty:
                fig = px.line(metric_data, x='step', y=metric, title=f'{metric.replace("_", " ").title()} Over Time')
                fig.update_layout(xaxis_title='Step', yaxis_title=metric.replace("_", " ").title())
                st.plotly_chart(fig, use_container_width=True)


def display_stage_times(stage_times: list):
    """Display stage timing information."""
    if not stage_times:
        st.warning("No stage timing data available.")
        return
    
    st.subheader("Stage Timing")
    
    # Convert to DataFrame for better display
    stage_df = pd.DataFrame(stage_times)
    
    # Format memory in MB
    if 'peak_mem_bytes' in stage_df.columns:
        stage_df['peak_mem_mb'] = stage_df['peak_mem_bytes'] / (1024 * 1024)
        stage_df['peak_mem_mb'] = stage_df['peak_mem_mb'].round(2)
    
    # Display table
    st.dataframe(stage_df, use_container_width=True)
    
    # Create bar chart for stage durations
    if 'seconds' in stage_df.columns:
        fig = px.bar(stage_df, x='stage', y='seconds', title='Stage Durations')
        fig.update_layout(xaxis_title='Stage', yaxis_title='Duration (seconds)')
        st.plotly_chart(fig, use_container_width=True)


def display_sysinfo(sysinfo: Dict[str, Any]):
    """Display system information."""
    if not sysinfo:
        st.warning("No system information available.")
        return
    
    st.subheader("System Information")
    
    # Display key system info
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Environment:**")
        st.write(f"- Python: {sysinfo.get('python_version', 'N/A')}")
        st.write(f"- PyTorch: {sysinfo.get('torch_version', 'N/A')}")
        st.write(f"- Device: {sysinfo.get('device', 'N/A')}")
        st.write(f"- CUDA Available: {sysinfo.get('cuda_available', 'N/A')}")
    
    with col2:
        st.write("**Hardware:**")
        st.write(f"- Platform: {sysinfo.get('platform', 'N/A')}")
        st.write(f"- Processor: {sysinfo.get('processor', 'N/A')}")
        if sysinfo.get('cuda_available'):
            st.write(f"- GPU: {sysinfo.get('gpu_name', 'N/A')}")
            st.write(f"- GPU Memory: {sysinfo.get('gpu_memory_total_gb', 'N/A')} GB")
    
    # Display seed and timing
    st.write("**Training:**")
    st.write(f"- Seed: {sysinfo.get('seed', 'N/A')}")
    st.write(f"- Start Time: {sysinfo.get('start_time', 'N/A')}")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="RLHF Training Monitor",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("RLHF Training Monitor")
    st.markdown("Monitor your RLHF training runs with real-time metrics and alerts.")
    
    # Get run path from query parameters or default
    query_params = st.experimental_get_query_params()
    default_run = query_params.get("run", ["runs/latest"])[0]
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Run path input
    run_path = st.sidebar.text_input(
        "Run Path",
        value=default_run,
        help="Path to the training run directory"
    )
    
    # Check if run exists
    if not run_path or not Path(run_path).exists():
        st.error(f"Run directory not found: {run_path}")
        st.info("Please specify a valid run directory path.")
        return
    
    st.sidebar.success(f"Monitoring run: {run_path}")
    
    # Load data
    metrics_df = load_metrics(run_path)
    stage_times = load_stage_times(run_path)
    sysinfo = load_sysinfo(run_path)
    
    # Check for alerts
    alerts = check_alerts(metrics_df) if metrics_df is not None else []
    
    # Display alerts
    if alerts:
        st.subheader("🚨 Alerts")
        for alert in alerts:
            if alert['type'] == 'error':
                st.error(alert['message'])
            else:
                st.warning(alert['message'])
    
    # Main content
    if metrics_df is not None:
        # Summary statistics
        st.subheader("�� Training Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Steps", len(metrics_df))
        with col2:
            if 'reward_mean' in metrics_df.columns:
                recent_reward = metrics_df['reward_mean'].tail(10).mean()
                st.metric("Recent Avg Reward", f"{recent_reward:.4f}" if not pd.isna(recent_reward) else "N/A")
        with col3:
            if 'kl' in metrics_df.columns:
                recent_kl = metrics_df['kl'].tail(10).mean()
                st.metric("Recent Avg KL", f"{recent_kl:.4f}" if not pd.isna(recent_kl) else "N/A")
        with col4:
            if 'loss' in metrics_df.columns:
                recent_loss = metrics_df['loss'].tail(10).mean()
                st.metric("Recent Avg Loss", f"{recent_loss:.4f}" if not pd.isna(recent_loss) else "N/A")
        
        # Metrics plots
        create_metrics_plots(metrics_df)
        
        # Raw metrics data
        st.subheader("📋 Raw Metrics Data")
        st.dataframe(metrics_df, use_container_width=True)
    
    # Stage times
    if stage_times:
        display_stage_times(stage_times)
    
    # System information
    if sysinfo:
        display_sysinfo(sysinfo)
    
    # Profiler artifacts
    st.subheader("🔍 Profiler Artifacts")
    
    trace_file = Path(run_path) / "trace.json"
    if trace_file.exists():
        st.success("✅ Chrome trace available")
        st.download_button(
            label="Download trace.json",
            data=trace_file.read_bytes(),
            file_name="trace.json",
            mime="application/json"
        )
    else:
        st.info("ℹ️ No trace.json found (profiler may not have been enabled)")
    
    op_stats_file = Path(run_path) / "op_stats.csv"
    if op_stats_file.exists():
        st.success("✅ Operation statistics available")
        st.download_button(
            label="Download op_stats.csv",
            data=op_stats_file.read_bytes(),
            file_name="op_stats.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("**RLHF Systems Kit Monitor** - Built with Streamlit and Plotly")


if __name__ == "__main__":
    # Handle command line arguments for CLI usage
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='RLHF Training Monitor')
        parser.add_argument('--run', type=str, default='runs/latest',
                          help='Path to the training run directory')
        args = parser.parse_args()
        
        # Set the run path for Streamlit
        os.environ['STREAMLIT_RUN_PATH'] = args.run
    
    main()
