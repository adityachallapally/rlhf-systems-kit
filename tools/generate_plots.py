#!/usr/bin/env python3
"""
Training Plot Generator

Generates PNG plots from existing training run metrics. This can be used to generate
plots for training runs that completed before plot generation was implemented.

Usage:
    python tools/generate_plots.py --run runs/latest
    python tools/generate_plots.py --run runs/run_20250829_024038 --output-dir custom_plots
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import visualization functions without importing the full rlhf_core package
# This avoids the torch dependency issue
import json

# Try to import optional dependencies
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_training_metrics_simple(jsonl_path: str) -> list:
    """Load training metrics from JSONL file without pandas."""
    if not os.path.exists(jsonl_path):
        return []
    
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    return data


def load_training_metrics(jsonl_path: str):
    """Load training metrics from JSONL file."""
    if not PANDAS_AVAILABLE:
        return load_training_metrics_simple(jsonl_path)
    
    if not os.path.exists(jsonl_path):
        return pd.DataFrame()
    
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Convert numeric columns
    numeric_cols = ['step', 'loss', 'total_loss', 'policy_loss', 'kl_loss', 
                   'reward_mean', 'reward_std', 'kl_mean', 'kl_std',
                   'clip_fraction', 'advantage_mean', 'advantage_std',
                   'sequence_length_mean', 'learning_rate', 'epoch']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def generate_training_plots(metrics_df: pd.DataFrame, output_dir: str) -> list:
    """Generate PNG plots from training metrics."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.style as style
    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation")
        return []
    
    if metrics_df.empty:
        print("Warning: No metrics data available for plotting")
        return []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style for better looking plots
    plt.style.use('default')
    
    generated_plots = []
    
    # Plot 1: Training Loss Over Time
    if 'step' in metrics_df.columns and 'total_loss' in metrics_df.columns:
        loss_data = metrics_df[['step', 'total_loss']].dropna()
        if not loss_data.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(loss_data['step'], loss_data['total_loss'], 'b-', linewidth=2, alpha=0.8)
            plt.title('Training Loss Over Time', fontsize=16, fontweight='bold')
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Total Loss', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, 'training_loss.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            generated_plots.append(plot_path)
            print(f"Generated training loss plot: {plot_path}")
    
    # Plot 2: KL Divergence Over Time
    if 'step' in metrics_df.columns and 'kl_mean' in metrics_df.columns:
        kl_data = metrics_df[['step', 'kl_mean']].dropna()
        if not kl_data.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(kl_data['step'], kl_data['kl_mean'], 'r-', linewidth=2, alpha=0.8)
            plt.title('KL Divergence Over Time', fontsize=16, fontweight='bold')
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('KL Divergence', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, 'kl_divergence.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            generated_plots.append(plot_path)
            print(f"Generated KL divergence plot: {plot_path}")
    
    # Plot 3: Reward Metrics Over Time
    if 'step' in metrics_df.columns and 'reward_mean' in metrics_df.columns:
        reward_data = metrics_df[['step', 'reward_mean', 'reward_std']].dropna()
        if not reward_data.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(reward_data['step'], reward_data['reward_mean'], 'g-', linewidth=2, alpha=0.8, label='Mean Reward')
            if 'reward_std' in reward_data.columns:
                plt.plot(reward_data['step'], reward_data['reward_std'], 'g--', linewidth=2, alpha=0.6, label='Reward Std')
            plt.title('Reward Metrics Over Time', fontsize=16, fontweight='bold')
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Reward', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, 'reward_metrics.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            generated_plots.append(plot_path)
            print(f"Generated reward metrics plot: {plot_path}")
    
    # Plot 4: Training Overview (Multi-metric dashboard)
    if 'step' in metrics_df.columns:
        metrics_to_plot = []
        if 'total_loss' in metrics_df.columns:
            metrics_to_plot.append(('total_loss', 'Total Loss', 'blue'))
        if 'kl_mean' in metrics_df.columns:
            metrics_to_plot.append(('kl_mean', 'KL Divergence', 'red'))
        if 'reward_mean' in metrics_df.columns:
            metrics_to_plot.append(('reward_mean', 'Mean Reward', 'green'))
        if 'clip_fraction' in metrics_df.columns:
            metrics_to_plot.append(('clip_fraction', 'Clip Fraction', 'orange'))
        
        if len(metrics_to_plot) >= 2:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, (metric, label, color) in enumerate(metrics_to_plot[:4]):
                if i < len(axes):
                    metric_data = metrics_df[['step', metric]].dropna()
                    if not metric_data.empty:
                        axes[i].plot(metric_data['step'], metric_data[metric], 
                                   color=color, linewidth=2, alpha=0.8)
                        axes[i].set_title(label, fontsize=12, fontweight='bold')
                        axes[i].set_xlabel('Training Step')
                        axes[i].set_ylabel(label)
                        axes[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(metrics_to_plot), len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle('Training Metrics Overview', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, 'training_overview.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            generated_plots.append(plot_path)
            print(f"Generated training overview plot: {plot_path}")
    
    return generated_plots


def generate_plots_from_run(run_dir: str) -> list:
    """Generate plots from a training run directory."""
    # Look for metrics file
    metrics_file = os.path.join(run_dir, 'logs', 'train.jsonl')
    if not os.path.exists(metrics_file):
        print(f"Warning: No training metrics found at {metrics_file}")
        return []
    
    # Load metrics
    metrics_df = load_training_metrics(metrics_file)
    if metrics_df.empty:
        print("Warning: No valid metrics data found")
        return []
    
    # Create plots directory
    plots_dir = os.path.join(run_dir, 'plots')
    
    # Generate plots
    generated_plots = generate_training_plots(metrics_df, plots_dir)
    
    return generated_plots


def main():
    parser = argparse.ArgumentParser(description='Generate PNG plots from training metrics')
    parser.add_argument('--run', type=str, default='runs/latest', 
                       help='Path to training run directory')
    parser.add_argument('--output-dir', type=str, 
                       help='Custom output directory for plots (default: run_dir/plots)')
    parser.add_argument('--metrics-file', type=str,
                       help='Direct path to metrics JSONL file')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Training Plot Generator")
    print("="*60)
    
    if args.metrics_file:
        # Generate plots from specific metrics file
        if not os.path.exists(args.metrics_file):
            print(f"Error: Metrics file not found: {args.metrics_file}")
            return 1
        
        print(f"Loading metrics from: {args.metrics_file}")
        metrics_df = load_training_metrics(args.metrics_file)
        
        if metrics_df.empty:
            print("Error: No valid metrics data found")
            return 1
        
        output_dir = args.output_dir or os.path.join(os.path.dirname(args.metrics_file), '..', 'plots')
        print(f"Output directory: {output_dir}")
        
        plot_files = generate_training_plots(metrics_df, output_dir)
        
    else:
        # Generate plots from run directory
        run_path = args.run
        
        # Resolve symlinks
        if os.path.islink(run_path):
            run_path = os.readlink(run_path)
            print(f"Resolved symlink: {run_path}")
        
        if not os.path.exists(run_path):
            print(f"Error: Run directory not found: {run_path}")
            return 1
        
        print(f"Processing run: {run_path}")
        
        if args.output_dir:
            # Custom output directory - need to load metrics and generate manually
            metrics_file = os.path.join(run_path, 'logs', 'train.jsonl')
            if not os.path.exists(metrics_file):
                print(f"Error: Training metrics not found: {metrics_file}")
                return 1
            
            metrics_df = load_training_metrics(metrics_file)
            if metrics_df.empty:
                print("Error: No valid metrics data found")
                return 1
            
            plot_files = generate_training_plots(metrics_df, args.output_dir)
        else:
            # Use default output directory in run folder
            plot_files = generate_plots_from_run(run_path)
    
    # Report results
    if plot_files:
        print(f"\n✅ Successfully generated {len(plot_files)} plots:")
        for plot_file in plot_files:
            print(f"  - {plot_file}")
        print(f"\nPlots are ready for viewing!")
    else:
        print("❌ No plots were generated. Check that:")
        print("  - matplotlib is installed")
        print("  - Training metrics file exists and contains data")
        print("  - Output directory is writable")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())