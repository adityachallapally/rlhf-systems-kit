"""
Visualization utilities for RLHF training metrics.

Provides utilities to generate PNG plots from training metrics data.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_training_metrics(jsonl_path: str) -> pd.DataFrame:
    """Load training metrics from JSONL file.
    
    Args:
        jsonl_path: Path to the training metrics JSONL file
        
    Returns:
        DataFrame containing the metrics data
    """
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


def generate_training_plots(metrics_df: pd.DataFrame, output_dir: str) -> List[str]:
    """Generate PNG plots from training metrics.
    
    Args:
        metrics_df: DataFrame containing training metrics
        output_dir: Directory to save plots
        
    Returns:
        List of generated plot file paths
    """
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
    
    # Plot 5: Loss Components Breakdown
    loss_components = []
    if 'policy_loss' in metrics_df.columns:
        loss_components.append(('policy_loss', 'Policy Loss', 'blue'))
    if 'kl_loss' in metrics_df.columns:
        loss_components.append(('kl_loss', 'KL Loss', 'red'))
    
    if len(loss_components) >= 2 and 'step' in metrics_df.columns:
        plt.figure(figsize=(10, 6))
        for metric, label, color in loss_components:
            component_data = metrics_df[['step', metric]].dropna()
            if not component_data.empty:
                plt.plot(component_data['step'], component_data[metric], 
                        color=color, linewidth=2, alpha=0.8, label=label)
        
        plt.title('Loss Components Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'loss_components.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        generated_plots.append(plot_path)
        print(f"Generated loss components plot: {plot_path}")
    
    return generated_plots


def generate_plots_from_run(run_dir: str) -> List[str]:
    """Generate plots from a training run directory.
    
    Args:
        run_dir: Path to the training run directory
        
    Returns:
        List of generated plot file paths
    """
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


def create_training_summary_plot(run_dir: str) -> Optional[str]:
    """Create a single summary plot with key training metrics.
    
    Args:
        run_dir: Path to the training run directory
        
    Returns:
        Path to the generated summary plot, or None if failed
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation")
        return None
    
    metrics_file = os.path.join(run_dir, 'logs', 'train.jsonl')
    if not os.path.exists(metrics_file):
        return None
    
    metrics_df = load_training_metrics(metrics_file)
    if metrics_df.empty:
        return None
    
    # Create a 2x2 summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Total Loss
    if 'step' in metrics_df.columns and 'total_loss' in metrics_df.columns:
        loss_data = metrics_df[['step', 'total_loss']].dropna()
        if not loss_data.empty:
            ax1.plot(loss_data['step'], loss_data['total_loss'], 'b-', linewidth=2)
            ax1.set_title('Training Loss', fontweight='bold')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
    
    # Plot 2: KL Divergence
    if 'step' in metrics_df.columns and 'kl_mean' in metrics_df.columns:
        kl_data = metrics_df[['step', 'kl_mean']].dropna()
        if not kl_data.empty:
            ax2.plot(kl_data['step'], kl_data['kl_mean'], 'r-', linewidth=2)
            ax2.set_title('KL Divergence', fontweight='bold')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('KL')
            ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rewards
    if 'step' in metrics_df.columns and 'reward_mean' in metrics_df.columns:
        reward_data = metrics_df[['step', 'reward_mean']].dropna()
        if not reward_data.empty:
            ax3.plot(reward_data['step'], reward_data['reward_mean'], 'g-', linewidth=2)
            ax3.set_title('Mean Reward', fontweight='bold')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Reward')
            ax3.grid(True, alpha=0.3)
    
    # Plot 4: Clip Fraction
    if 'step' in metrics_df.columns and 'clip_fraction' in metrics_df.columns:
        clip_data = metrics_df[['step', 'clip_fraction']].dropna()
        if not clip_data.empty:
            ax4.plot(clip_data['step'], clip_data['clip_fraction'], 'orange', linewidth=2)
            ax4.set_title('Clip Fraction', fontweight='bold')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Clip Fraction')
            ax4.grid(True, alpha=0.3)
    
    plt.suptitle('RLHF Training Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, 'training_summary.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Generated training summary plot: {plot_path}")
    return plot_path