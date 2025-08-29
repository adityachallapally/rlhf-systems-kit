#!/usr/bin/env python3
"""
RLHF Training Stability Dashboard

FastAPI app providing real-time monitoring of training metrics with automated warnings.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(
    title="RLHF Training Stability Dashboard",
    description="Real-time monitoring with automated warning systems",
    version="1.0.0"
)

# Global state for latest metrics
latest_metrics: Dict[str, Any] = {}
last_update = None

def load_latest_metrics():
    """Load the latest metrics from the most recent training run."""
    global latest_metrics, last_update
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.parent
    latest_link = script_dir / "runs" / "latest"
    
    if not latest_link.exists():
        return {}
    
    # Resolve symlink to actual path
    if latest_link.is_symlink():
        actual_path = latest_link.resolve()
    else:
        actual_path = latest_link
    
    log_file = actual_path / "logs" / "train.jsonl"
    
    if not log_file.exists():
        return {}
    
    try:
        # Read the last few lines to get latest metrics
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if lines:
                # Get the last non-empty line
                for line in reversed(lines):
                    line = line.strip()
                    if line:
                        try:
                            latest_metrics = json.loads(line)
                            last_update = datetime.now()
                            break
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        print(f"Error loading metrics: {e}")
    
    return latest_metrics

def format_metric(value, default="N/A", precision=4):
    """Helper function to format metric values safely."""
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    return default

def check_alerts(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Check for training stability alerts."""
    alerts = []
    
    # KL divergence alerts
    if 'kl' in metrics:
        kl_value = metrics['kl']
        if isinstance(kl_value, (int, float)) and kl_value > 0.3:  # High KL
            alerts.append({
                'type': 'warning',
                'metric': 'kl',
                'value': kl_value,
                'message': f'KL divergence ({kl_value:.4f}) is high - training may be unstable'
            })
    
    # Entropy collapse alerts
    if 'entropy' in metrics:
        entropy_value = metrics['entropy']
        if isinstance(entropy_value, (int, float)) and entropy_value < 0.01:  # Low entropy
            alerts.append({
                'type': 'warning',
                'metric': 'entropy',
                'value': entropy_value,
                'message': f'Entropy ({entropy_value:.4f}) is very low - policy may be collapsing'
            })
    
    # Gradient norm alerts
    if 'grad_norm' in metrics:
        grad_norm_value = metrics['grad_norm']
        if isinstance(grad_norm_value, (int, float)) and grad_norm_value > 10.0:  # High gradient norm
            alerts.append({
                'type': 'warning',
                'metric': 'grad_norm',
                'value': grad_norm_value,
                'message': f'Gradient norm ({grad_norm_value:.4f}) is very high - potential exploding gradients'
            })
    
    # Reward variance alerts
    if 'reward_var' in metrics:
        reward_var_value = metrics['reward_var']
        if isinstance(reward_var_value, (int, float)) and reward_var_value < 0.001:  # Low reward variance
            alerts.append({
                'type': 'info',
                'metric': 'reward_var',
                'value': reward_var_value,
                'message': f'Reward variance ({reward_var_value:.6f}) is very low - limited exploration'
            })
    
    return alerts

@app.get("/")
async def root():
    """Root endpoint with dashboard info."""
    return {
        "service": "RLHF Training Stability Dashboard",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/metrics": "Latest training metrics",
            "/alerts": "Current training alerts",
            "/dashboard": "HTML dashboard view"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "rlhf-monitor"
    }

@app.get("/metrics")
async def get_metrics():
    """Get the latest training metrics."""
    load_latest_metrics()
    
    if not latest_metrics:
        raise HTTPException(status_code=404, detail="No training metrics found")
    
    return {
        "metrics": latest_metrics,
        "last_update": last_update.isoformat() if last_update else None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/alerts")
async def get_alerts(test_alerts: bool = Query(False, description="Enable test alerts for debugging")):
    """Get current training alerts."""
    load_latest_metrics()
    
    alerts = check_alerts(latest_metrics)
    
    # Add test alerts if requested
    if test_alerts:
        test_alerts_list = [
            {
                'type': 'warning',
                'metric': 'kl_test',
                'value': 0.9,
                'message': 'TEST: KL divergence is 3x target (0.9 > 0.3)'
            },
            {
                'type': 'warning', 
                'metric': 'entropy_test',
                'value': 0.005,
                'message': 'TEST: Entropy collapse detected (0.005 < 0.01)'
            },
            {
                'type': 'warning',
                'metric': 'grad_norm_test', 
                'value': 25.0,
                'message': 'TEST: Gradient norm spike (25.0 > 10.0)'
            }
        ]
        alerts.extend(test_alerts_list)
    
    return {
        "alerts": alerts,
        "count": len(alerts),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """HTML dashboard view."""
    load_latest_metrics()
    alerts = check_alerts(latest_metrics)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RLHF Training Stability Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
            .metric-card {{ background: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
            .alerts {{ margin: 20px 0; }}
            .alert {{ padding: 10px; margin: 5px 0; border-radius: 3px; }}
            .warning {{ background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }}
            .info {{ background: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; }}
            .refresh {{ background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 3px; cursor: pointer; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 RLHF Training Stability Dashboard</h1>
            <p>Real-time monitoring of training metrics with automated warning systems</p>
            <button class="refresh" onclick="location.reload()">🔄 Refresh</button>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <h3>📊 Training Status</h3>
                <p><strong>Last Update:</strong> {last_update.strftime('%H:%M:%S') if last_update else 'Never'}</p>
                <p><strong>Metrics Count:</strong> {len(latest_metrics)}</p>
            </div>
            
            <div class="metric-card">
                <h3>🎯 Key Metrics</h3>
                <p><strong>KL Divergence:</strong> {format_metric(latest_metrics.get('kl'))}</p>
                <p><strong>Reward Mean:</strong> {format_metric(latest_metrics.get('reward_mean'))}</p>
                <p><strong>Entropy:</strong> {format_metric(latest_metrics.get('entropy'))}</p>
            </div>
        </div>
        
        <div class="alerts">
            <h2>🚨 Training Alerts ({len(alerts)})</h2>
            {''.join([f'<div class="alert {alert["type"]}"><strong>{alert["metric"]}:</strong> {alert["message"]}</div>' for alert in alerts])}
            {f'<p><em>No alerts at this time. Training appears stable.</em></p>' if not alerts else ''}
        </div>
        
        <script>
            // Auto-refresh every 30 seconds
            setTimeout(() => location.reload(), 30000);
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
