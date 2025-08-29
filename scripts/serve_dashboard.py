#!/usr/bin/env python3
"""
Live RLHF Training Stability Dashboard Server

This script provides a real-time web dashboard for monitoring RLHF training stability.
It tails the stability.jsonl log file and serves live-updating plots with warning banners.
"""

import os
import json
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import io
import base64

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn


class StabilityDashboard:
    """Live stability dashboard that monitors training logs."""
    
    def __init__(self, log_file: str = "runs/latest/logs/stability.jsonl"):
        """Initialize the dashboard.
        
        Args:
            log_file: Path to stability.jsonl file to monitor
        """
        self.log_file = log_file
        self.last_modified = 0
        self.cached_data = []
        self.cached_plots = {}
        self.plot_update_interval = 5  # Update plots every 5 seconds
        
        # Instability thresholds
        self.thresholds = {
            'kl': 0.2,           # KL > 0.2 indicates runaway
            'entropy': 0.1,      # Entropy < 0.1 indicates collapse
            'grad_norm': 1e3,    # Gradient norm > 1000 indicates exploding gradients
            'reward_std': 2.0,   # High reward variance indicates instability
            'kl_target_err': 0.15  # KL target error > 0.15 indicates poor control
        }
        
        # Warning messages
        self.warnings = []
        
    def find_latest_log(self) -> str:
        """Find the most recent stability log file."""
        # Try the specified path first
        if os.path.exists(self.log_file):
            return self.log_file
        
        # Look for latest run
        runs_dir = Path("runs")
        if runs_dir.exists():
            latest_link = runs_dir / "latest"
            if latest_link.exists() and latest_link.is_symlink():
                latest_run = latest_link.resolve()
                log_path = latest_run / "logs" / "stability.jsonl"
                if log_path.exists():
                    return str(log_path)
            
            # Find most recent run directory
            run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
            if run_dirs:
                latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
                log_path = latest_run / "logs" / "stability.jsonl"
                if log_path.exists():
                    return str(log_path)
        
        raise FileNotFoundError("No stability log file found. Run training first.")
    
    def load_log_data(self) -> List[Dict[str, Any]]:
        """Load and parse the stability log file."""
        try:
            log_file = self.find_latest_log()
            self.log_file = log_file
            
            # Check if file has been modified
            current_mtime = os.path.getmtime(log_file)
            if current_mtime <= self.last_modified:
                return self.cached_data
            
            self.last_modified = current_mtime
            
            data = []
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            
            self.cached_data = data
            self.update_warnings(data)
            return data
            
        except Exception as e:
            print(f"Error loading log data: {e}")
            return self.cached_data
    
    def update_warnings(self, data: List[Dict[str, Any]]) -> None:
        """Update warning messages based on current data."""
        if not data:
            self.warnings = []
            return
        
        warnings = []
        latest = data[-1]
        
        # Check KL divergence
        if latest.get('kl', 0) > self.thresholds['kl']:
            warnings.append(f"⚠️ KL Divergence ({latest['kl']:.3f}) above threshold ({self.thresholds['kl']})")
        
        # Check entropy collapse
        if latest.get('entropy', 1.0) < self.thresholds['entropy']:
            warnings.append(f"⚠️ Policy Entropy ({latest['entropy']:.3f}) below threshold ({self.thresholds['entropy']})")
        
        # Check gradient explosion
        if latest.get('grad_norm', 0) > self.thresholds['grad_norm']:
            warnings.append(f"⚠️ Gradient Norm ({latest['grad_norm']:.2e}) above threshold ({self.thresholds['grad_norm']})")
        
        # Check reward instability
        if latest.get('reward_std', 0) > self.thresholds['reward_std']:
            warnings.append(f"⚠️ High Reward Variance ({latest['reward_std']:.3f}) above threshold ({self.thresholds['reward_std']})")
        
        # Check KL target error
        if abs(latest.get('kl_target_err', 0)) > self.thresholds['kl_target_err']:
            warnings.append(f"⚠️ KL Target Error ({latest['kl_target_err']:.3f}) above threshold ({self.thresholds['kl_target_err']})")
        
        self.warnings = warnings
    
    def create_live_plot(self, metric: str, title: str, ylabel: str, 
                        log_scale: bool = False) -> str:
        """Create a live plot for a specific metric and return as base64 image."""
        data = self.load_log_data()
        if not data:
            return ""
        
        df = pd.DataFrame(data)
        if 'step' not in df.columns or metric not in df.columns:
            return ""
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['step'], df[metric], 'b-', linewidth=2, label=metric)
        
        # Add threshold line if applicable
        if metric in self.thresholds:
            threshold = self.thresholds[metric]
            if metric == 'entropy':
                # For entropy, threshold is minimum
                plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, 
                           label=f'Threshold ({threshold})')
                # Highlight unstable regions
                unstable = df[df[metric] < threshold]
                if not unstable.empty:
                    plt.scatter(unstable['step'], unstable[metric], color='red', s=50, alpha=0.7)
            else:
                # For other metrics, threshold is maximum
                plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, 
                           label=f'Threshold ({threshold})')
                # Highlight unstable regions
                unstable = df[df[metric] > threshold]
                if not unstable.empty:
                    plt.scatter(unstable['step'], unstable[metric], color='red', s=50, alpha=0.7)
        
        plt.title(f'{title} - Live Update')
        plt.xlabel('Training Step')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if log_scale:
            plt.yscale('log')
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_str
    
    def get_dashboard_html(self) -> str:
        """Generate the main dashboard HTML."""
        data = self.load_log_data()
        latest_metrics = data[-1] if data else {}
        
        # Create live plots
        kl_plot = self.create_live_plot('kl', 'KL Divergence', 'KL Divergence')
        entropy_plot = self.create_live_plot('entropy', 'Policy Entropy', 'Entropy')
        grad_plot = self.create_live_plot('grad_norm', 'Gradient Norm', 'Gradient Norm', log_scale=True)
        reward_plot = self.create_live_plot('reward_mean', 'Reward Mean', 'Reward')
        
        # Warning banners
        warning_html = ""
        if self.warnings:
            warning_html = '<div class="warnings">'
            for warning in self.warnings:
                warning_html += f'<div class="warning">{warning}</div>'
            warning_html += '</div>'
        
        # Latest metrics table
        metrics_table = ""
        if latest_metrics:
            metrics_table = '<table class="metrics-table"><tr><th>Metric</th><th>Value</th></tr>'
            for key, value in latest_metrics.items():
                if key not in ['timestamp', 'step', 'batch_size'] and isinstance(value, (int, float)):
                    if isinstance(value, float):
                        display_value = f"{value:.4f}"
                    else:
                        display_value = str(value)
                    metrics_table += f'<tr><td>{key}</td><td>{display_value}</td></tr>'
            metrics_table += '</table>'
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RLHF Training Stability Dashboard</title>
            <meta charset="utf-8">
            <meta http-equiv="refresh" content="10">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    text-align: center;
                }}
                .warnings {{
                    margin: 20px 0;
                }}
                .warning {{
                    background-color: #ffeb3b;
                    color: #333;
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 5px;
                    border-left: 4px solid #ff9800;
                    font-weight: bold;
                }}
                .plots-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin: 20px 0;
                }}
                .plot-container {{
                    background: white;
                    padding: 15px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .plot-container img {{
                    width: 100%;
                    height: auto;
                }}
                .metrics-section {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    margin: 20px 0;
                }}
                .metrics-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 15px;
                }}
                .metrics-table th, .metrics-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .metrics-table th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                .status {{
                    text-align: center;
                    margin: 20px 0;
                    padding: 10px;
                    background-color: #e8f5e8;
                    border-radius: 5px;
                    border: 1px solid #4caf50;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    color: #666;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 RLHF Training Stability Dashboard</h1>
                <p>Real-time monitoring of training stability metrics</p>
                <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            {warning_html}
            
            <div class="status">
                📊 Monitoring: {self.log_file} | 
                📈 Total Steps: {len(data)} | 
                🔄 Auto-refresh every 10 seconds
            </div>
            
            <div class="plots-grid">
                <div class="plot-container">
                    <h3>KL Divergence</h3>
                    <img src="data:image/png;base64,{kl_plot}" alt="KL Divergence Plot">
                </div>
                <div class="plot-container">
                    <h3>Policy Entropy</h3>
                    <img src="data:image/png;base64,{entropy_plot}" alt="Entropy Plot">
                </div>
                <div class="plot-container">
                    <h3>Gradient Norm</h3>
                    <img src="data:image/png;base64,{grad_plot}" alt="Gradient Plot">
                </div>
                <div class="plot-container">
                    <h3>Reward Mean</h3>
                    <img src="data:image/png;base64,{reward_plot}" alt="Reward Plot">
                </div>
            </div>
            
            <div class="metrics-section">
                <h3>Latest Metrics</h3>
                {metrics_table}
            </div>
            
            <div class="footer">
                <p>RLHF Systems Kit - Stability Dashboard</p>
                <p>Run 'make dashboard' to start this server</p>
            </div>
        </body>
        </html>
        """
        
        return html


# Create FastAPI app
app = FastAPI(title="RLHF Stability Dashboard", version="1.0.0")

# Create dashboard instance
dashboard = StabilityDashboard()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard page."""
    return dashboard.get_dashboard_html()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        data = dashboard.load_log_data()
        return {
            "status": "healthy",
            "log_file": dashboard.log_file,
            "data_points": len(data),
            "last_updated": dashboard.last_modified,
            "warnings": len(dashboard.warnings)
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/metrics")
async def get_metrics():
    """Get current metrics as JSON."""
    data = dashboard.load_log_data()
    if not data:
        raise HTTPException(status_code=404, detail="No metrics data available")
    
    return {
        "latest": data[-1] if data else {},
        "total_steps": len(data),
        "warnings": dashboard.warnings,
        "thresholds": dashboard.thresholds
    }


@app.get("/plot/{metric}")
async def get_plot(metric: str):
    """Get a specific metric plot as an image."""
    if metric == "kl":
        plot_data = dashboard.create_live_plot('kl', 'KL Divergence', 'KL Divergence')
    elif metric == "entropy":
        plot_data = dashboard.create_live_plot('entropy', 'Policy Entropy', 'Entropy')
    elif metric == "grad_norm":
        plot_data = dashboard.create_live_plot('grad_norm', 'Gradient Norm', 'Gradient Norm', log_scale=True)
    elif metric == "reward":
        plot_data = dashboard.create_live_plot('reward_mean', 'Reward Mean', 'Reward')
    else:
        raise HTTPException(status_code=400, detail="Invalid metric")
    
    if not plot_data:
        raise HTTPException(status_code=404, detail="Plot data not available")
    
    # Return base64 image data
    return {"metric": metric, "image_data": plot_data}


def main():
    """Main function to run the dashboard server."""
    print("🚀 Starting RLHF Training Stability Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8000/")
    print("📈 Monitoring stability metrics in real-time...")
    print("🔄 Auto-refresh every 10 seconds")
    print("⏹️  Press Ctrl+C to stop")
    
    # Try to find and load initial data
    try:
        dashboard.find_latest_log()
        print(f"✅ Found log file: {dashboard.log_file}")
    except FileNotFoundError as e:
        print(f"⚠️  Warning: {e}")
        print("   Run 'make train_smoke' first to generate training data")
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()