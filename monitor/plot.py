"""
Plotting utilities for RLHF training stability metrics.

This module provides functions to create time-series plots for all stability metrics
and highlight instability with simple thresholds.
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


def load_stability_logs(log_file: str) -> pd.DataFrame:
    """Load stability metrics from JSONL log file.
    
    Args:
        log_file: Path to stability.jsonl file
        
    Returns:
        DataFrame with stability metrics
    """
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    data = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    if not data:
        raise ValueError("No valid log entries found")
    
    df = pd.DataFrame(data)
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by step
    if 'step' in df.columns:
        df = df.sort_values('step')
    
    return df


def create_stability_plots(df: pd.DataFrame, 
                          output_dir: str = "report/figures",
                          figsize: tuple = (15, 12)) -> None:
    """Create comprehensive stability plots and save them.
    
    Args:
        df: DataFrame with stability metrics
        output_dir: Directory to save plots
        figsize: Figure size for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define instability thresholds
    thresholds = {
        'kl': 0.2,           # KL > 0.2 indicates runaway
        'entropy': 0.1,      # Entropy < 0.1 indicates collapse
        'grad_norm': 1e3,    # Gradient norm > 1000 indicates exploding gradients
        'reward_std': 2.0,   # High reward variance indicates instability
        'kl_target_err': 0.15  # KL target error > 0.15 indicates poor control
    }
    
    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle('RLHF Training Stability Dashboard', fontsize=16, fontweight='bold')
    
    # 1. KL Divergence and Target Error
    ax1 = axes[0, 0]
    ax1.plot(df['step'], df['kl'], 'b-', label='KL Divergence', linewidth=2)
    ax1.axhline(y=thresholds['kl'], color='r', linestyle='--', alpha=0.7, label=f'Threshold ({thresholds["kl"]})')
    ax1.set_title('KL Divergence')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('KL Divergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Highlight instability
    unstable_kl = df[df['kl'] > thresholds['kl']]
    if not unstable_kl.empty:
        ax1.scatter(unstable_kl['step'], unstable_kl['kl'], color='red', s=50, alpha=0.7, label='Unstable')
    
    # 2. KL Target Error
    ax2 = axes[0, 1]
    ax2.plot(df['step'], df['kl_target_err'], 'g-', label='KL Target Error', linewidth=2)
    ax2.axhline(y=thresholds['kl_target_err'], color='r', linestyle='--', alpha=0.7, label=f'Threshold ({thresholds["kl_target_err"]})')
    ax2.axhline(y=-thresholds['kl_target_err'], color='r', linestyle='--', alpha=0.7)
    ax2.set_title('KL Target Error')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('KL Target Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Policy Entropy
    ax3 = axes[1, 0]
    ax3.plot(df['step'], df['entropy'], 'm-', label='Policy Entropy', linewidth=2)
    ax3.axhline(y=thresholds['entropy'], color='r', linestyle='--', alpha=0.7, label=f'Threshold ({thresholds["entropy"]})')
    ax3.set_title('Policy Entropy')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Entropy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Highlight instability
    unstable_entropy = df[df['entropy'] < thresholds['entropy']]
    if not unstable_entropy.empty:
        ax3.scatter(unstable_entropy['step'], unstable_entropy['entropy'], color='red', s=50, alpha=0.7, label='Unstable')
    
    # 4. Reward Statistics
    ax4 = axes[1, 1]
    ax4.plot(df['step'], df['reward_mean'], 'c-', label='Reward Mean', linewidth=2)
    ax4.fill_between(df['step'], 
                     df['reward_mean'] - df['reward_std'], 
                     df['reward_mean'] + df['reward_std'], 
                     alpha=0.3, color='c', label='±1 Std Dev')
    ax4.set_title('Reward Statistics')
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Reward')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Gradient Norm
    ax5 = axes[2, 0]
    ax5.plot(df['step'], df['grad_norm'], 'orange', label='Gradient Norm', linewidth=2)
    ax5.axhline(y=thresholds['grad_norm'], color='r', linestyle='--', alpha=0.7, label=f'Threshold ({thresholds["grad_norm"]})')
    ax5.set_title('Gradient Norm')
    ax5.set_xlabel('Training Step')
    ax5.set_ylabel('Gradient Norm')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # Highlight instability
    unstable_grad = df[df['grad_norm'] > thresholds['grad_norm']]
    if not unstable_grad.empty:
        ax5.scatter(unstable_grad['step'], unstable_grad['grad_norm'], color='red', s=50, alpha=0.7, label='Unstable')
    
    # 6. Throughput and PPO Metrics
    ax6 = axes[2, 1]
    ax6_twin = ax6.twinx()
    
    # Throughput
    ax6.plot(df['step'], df['tokens_per_second'], 'b-', label='Tokens/sec', linewidth=2)
    ax6.set_xlabel('Training Step')
    ax6.set_ylabel('Tokens per Second', color='b')
    ax6.tick_params(axis='y', labelcolor='b')
    
    # PPO clip fraction
    ax6_twin.plot(df['step'], df['ppo_clip_fraction'], 'r-', label='PPO Clip Fraction', linewidth=2)
    ax6_twin.set_ylabel('PPO Clip Fraction', color='r')
    ax6_twin.tick_params(axis='y', labelcolor='r')
    
    ax6.set_title('Throughput and PPO Metrics')
    ax6.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    plot_path = os.path.join(output_dir, 'stability_dashboard.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved stability dashboard to: {plot_path}")
    
    # Create individual metric plots
    create_individual_plots(df, output_dir, thresholds)
    
    plt.close()


def create_individual_plots(df: pd.DataFrame, 
                           output_dir: str, 
                           thresholds: Dict[str, float]) -> None:
    """Create individual metric plots for detailed analysis.
    
    Args:
        df: DataFrame with stability metrics
        output_dir: Directory to save plots
        thresholds: Dictionary of instability thresholds
    """
    # 1. KL Divergence detailed plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['step'], df['kl'], 'b-', linewidth=2, label='KL Divergence')
    plt.axhline(y=thresholds['kl'], color='r', linestyle='--', alpha=0.7, label=f'Unstable Threshold ({thresholds["kl"]})')
    plt.fill_between(df['step'], 0, thresholds['kl'], alpha=0.1, color='green', label='Stable Region')
    plt.fill_between(df['step'], thresholds['kl'], df['kl'].max(), alpha=0.1, color='red', label='Unstable Region')
    plt.title('KL Divergence Stability Analysis')
    plt.xlabel('Training Step')
    plt.ylabel('KL Divergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'kl_stability.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Entropy collapse analysis
    plt.figure(figsize=(12, 6))
    plt.plot(df['step'], df['entropy'], 'm-', linewidth=2, label='Policy Entropy')
    plt.axhline(y=thresholds['entropy'], color='r', linestyle='--', alpha=0.7, label=f'Collapse Threshold ({thresholds["entropy"]})')
    plt.fill_between(df['step'], thresholds['entropy'], df['entropy'].max(), alpha=0.1, color='green', label='Healthy Entropy')
    plt.fill_between(df['step'], 0, thresholds['entropy'], alpha=0.1, color='red', label='Entropy Collapse')
    plt.title('Policy Entropy Stability Analysis')
    plt.xlabel('Training Step')
    plt.ylabel('Entropy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'entropy_stability.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Gradient explosion analysis
    plt.figure(figsize=(12, 6))
    plt.plot(df['step'], df['grad_norm'], 'orange', linewidth=2, label='Gradient Norm')
    plt.axhline(y=thresholds['grad_norm'], color='r', linestyle='--', alpha=0.7, label=f'Explosion Threshold ({thresholds["grad_norm"]})')
    plt.fill_between(df['step'], 0, thresholds['grad_norm'], alpha=0.1, color='green', label='Stable Gradients')
    plt.fill_between(df['step'], thresholds['grad_norm'], df['grad_norm'].max(), alpha=0.1, color='red', label='Exploding Gradients')
    plt.title('Gradient Norm Stability Analysis')
    plt.xlabel('Training Step')
    plt.ylabel('Gradient Norm')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'gradient_stability.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_stability_report(df: pd.DataFrame, 
                            output_dir: str = "report") -> str:
    """Generate a text report summarizing stability issues.
    
    Args:
        df: DataFrame with stability metrics
        output_dir: Directory to save report
        
    Returns:
        Path to the generated report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define thresholds
    thresholds = {
        'kl': 0.2,
        'entropy': 0.1,
        'grad_norm': 1e3,
        'reward_std': 2.0,
        'kl_target_err': 0.15
    }
    
    report_lines = [
        "RLHF Training Stability Report",
        "=" * 40,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Steps: {len(df)}",
        f"Training Duration: {df['step'].max() - df['step'].min()} steps",
        "",
        "Stability Analysis:",
        "-" * 20
    ]
    
    # Check KL divergence
    unstable_kl = df[df['kl'] > thresholds['kl']]
    if not unstable_kl.empty:
        report_lines.extend([
            f"⚠️  KL Divergence Instability: {len(unstable_kl)} steps above threshold {thresholds['kl']}",
            f"   Max KL: {unstable_kl['kl'].max():.4f}",
            f"   Steps: {unstable_kl['step'].tolist()}"
        ])
    else:
        report_lines.append("✅ KL Divergence: Stable throughout training")
    
    # Check entropy collapse
    unstable_entropy = df[df['entropy'] < thresholds['entropy']]
    if not unstable_entropy.empty:
        report_lines.extend([
            f"⚠️  Entropy Collapse: {len(unstable_entropy)} steps below threshold {thresholds['entropy']}",
            f"   Min Entropy: {unstable_entropy['entropy'].min():.4f}",
            f"   Steps: {unstable_entropy['step'].tolist()}"
        ])
    else:
        report_lines.append("✅ Policy Entropy: Stable throughout training")
    
    # Check gradient explosion
    unstable_grad = df[df['grad_norm'] > thresholds['grad_norm']]
    if not unstable_grad.empty:
        report_lines.extend([
            f"⚠️  Gradient Explosion: {len(unstable_grad)} steps above threshold {thresholds['grad_norm']}",
            f"   Max Gradient Norm: {unstable_grad['grad_norm'].max():.2e}",
            f"   Steps: {unstable_grad['step'].tolist()}"
        ])
    else:
        report_lines.append("✅ Gradient Norms: Stable throughout training")
    
    # Summary statistics
    report_lines.extend([
        "",
        "Summary Statistics:",
        "-" * 20,
        f"Average KL Divergence: {df['kl'].mean():.4f} ± {df['kl'].std():.4f}",
        f"Average Entropy: {df['entropy'].mean():.4f} ± {df['entropy'].std():.4f}",
        f"Average Gradient Norm: {df['grad_norm'].mean():.2e} ± {df['grad_norm'].std():.2e}",
        f"Average Reward: {df['reward_mean'].mean():.4f} ± {df['reward_std'].mean():.4f}",
        f"Average Throughput: {df['tokens_per_second'].mean():.1f} tokens/sec"
    ])
    
    # Save report
    report_path = os.path.join(output_dir, 'stability_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Generated stability report: {report_path}")
    return report_path