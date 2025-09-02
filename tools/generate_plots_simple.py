#!/usr/bin/env python3
"""
Simple Training Plot Generator

Generates PNG plots from existing training run metrics, with fallback to text reports
when matplotlib is not available.

Usage:
    python tools/generate_plots_simple.py --run runs/latest
    python tools/generate_plots_simple.py --run runs/run_20250829_024038
"""

import os
import sys
import json
import argparse
from pathlib import Path


def load_training_metrics(jsonl_path: str) -> list:
    """Load training metrics from JSONL file."""
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


def generate_text_report(metrics_data: list, output_dir: str) -> list:
    """Generate a text-based training report."""
    if not metrics_data:
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate summary report
    report_path = os.path.join(output_dir, 'training_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("RLHF Training Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic statistics
        f.write(f"Total Steps: {len(metrics_data)}\n")
        
        if metrics_data:
            first_step = metrics_data[0]
            last_step = metrics_data[-1]
            
            f.write(f"First Step: {first_step.get('step', 'N/A')}\n")
            f.write(f"Last Step: {last_step.get('step', 'N/A')}\n")
            f.write(f"Epochs: {last_step.get('epoch', 'N/A')}\n\n")
            
            # Metrics summary
            f.write("Metrics Summary:\n")
            f.write("-" * 30 + "\n")
            
            metrics_to_summarize = ['total_loss', 'kl_mean', 'reward_mean', 'clip_fraction']
            
            for metric in metrics_to_summarize:
                values = [step.get(metric) for step in metrics_data if step.get(metric) is not None]
                if values:
                    try:
                        values = [float(v) for v in values if v is not None]
                        if values:
                            f.write(f"{metric}:\n")
                            f.write(f"  Min: {min(values):.6f}\n")
                            f.write(f"  Max: {max(values):.6f}\n")
                            f.write(f"  Mean: {sum(values)/len(values):.6f}\n")
                            f.write(f"  Final: {values[-1]:.6f}\n\n")
                    except (ValueError, TypeError):
                        continue
            
            # Step-by-step data
            f.write("Step-by-Step Data:\n")
            f.write("-" * 30 + "\n")
            f.write("Step | Loss     | KL       | Reward   | Clip Frac\n")
            f.write("-----|----------|----------|----------|----------\n")
            
            for step_data in metrics_data:
                step = step_data.get('step', 'N/A')
                loss = step_data.get('total_loss', 0)
                kl = step_data.get('kl_mean', 0)
                reward = step_data.get('reward_mean', 0)
                clip = step_data.get('clip_fraction', 0)
                
                try:
                    f.write(f"{step:4} | {loss:8.6f} | {kl:8.6f} | {reward:8.6f} | {clip:8.6f}\n")
                except (ValueError, TypeError):
                    f.write(f"{step:4} | {'N/A':8} | {'N/A':8} | {'N/A':8} | {'N/A':8}\n")
    
    print(f"Generated training report: {report_path}")
    return [report_path]


def generate_matplotlib_plots(metrics_data: list, output_dir: str) -> list:
    """Generate PNG plots using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, falling back to text report")
        return generate_text_report(metrics_data, output_dir)
    
    if not metrics_data:
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    generated_plots = []
    
    # Extract data
    steps = []
    losses = []
    kl_values = []
    rewards = []
    clip_fractions = []
    
    for step_data in metrics_data:
        step = step_data.get('step')
        if step is not None:
            steps.append(step)
            losses.append(step_data.get('total_loss', 0))
            kl_values.append(step_data.get('kl_mean', 0))
            rewards.append(step_data.get('reward_mean', 0))
            clip_fractions.append(step_data.get('clip_fraction', 0))
    
    if not steps:
        return generate_text_report(metrics_data, output_dir)
    
    # Plot 1: Training Loss
    if losses and any(l != 0 for l in losses):
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, 'b-', linewidth=2, alpha=0.8)
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
    
    # Plot 2: KL Divergence
    if kl_values and any(k != 0 for k in kl_values):
        plt.figure(figsize=(10, 6))
        plt.plot(steps, kl_values, 'r-', linewidth=2, alpha=0.8)
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
    
    # Plot 3: Overview Dashboard
    if len(steps) > 1:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        ax1.plot(steps, losses, 'b-', linewidth=2)
        ax1.set_title('Training Loss', fontweight='bold')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        # KL Divergence
        ax2.plot(steps, kl_values, 'r-', linewidth=2)
        ax2.set_title('KL Divergence', fontweight='bold')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('KL')
        ax2.grid(True, alpha=0.3)
        
        # Rewards
        ax3.plot(steps, rewards, 'g-', linewidth=2)
        ax3.set_title('Mean Reward', fontweight='bold')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Reward')
        ax3.grid(True, alpha=0.3)
        
        # Clip Fraction
        ax4.plot(steps, clip_fractions, 'orange', linewidth=2)
        ax4.set_title('Clip Fraction', fontweight='bold')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Clip Fraction')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('RLHF Training Overview', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'training_overview.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        generated_plots.append(plot_path)
        print(f"Generated training overview plot: {plot_path}")
    
    # Also generate text report
    text_files = generate_text_report(metrics_data, output_dir)
    generated_plots.extend(text_files)
    
    return generated_plots


def main():
    parser = argparse.ArgumentParser(description='Generate plots/reports from training metrics')
    parser.add_argument('--run', type=str, default='runs/latest', 
                       help='Path to training run directory')
    parser.add_argument('--output-dir', type=str, 
                       help='Custom output directory for plots (default: run_dir/plots)')
    parser.add_argument('--text-only', action='store_true',
                       help='Generate text report only, skip matplotlib plots')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Training Plot/Report Generator")
    print("="*60)
    
    run_path = args.run
    
    # Resolve symlinks
    if os.path.islink(run_path):
        original_path = run_path
        run_path = os.path.realpath(run_path)
        print(f"Resolved symlink: {original_path} -> {run_path}")
    
    if not os.path.exists(run_path):
        print(f"Error: Run directory not found: {run_path}")
        return 1
    
    print(f"Processing run: {run_path}")
    
    # Look for metrics file
    metrics_file = os.path.join(run_path, 'logs', 'train.jsonl')
    if not os.path.exists(metrics_file):
        print(f"Error: Training metrics not found: {metrics_file}")
        return 1
    
    print(f"Loading metrics from: {metrics_file}")
    
    # Load metrics
    metrics_data = load_training_metrics(metrics_file)
    if not metrics_data:
        print("Error: No valid metrics data found")
        return 1
    
    print(f"Loaded {len(metrics_data)} training steps")
    
    # Determine output directory
    output_dir = args.output_dir or os.path.join(run_path, 'plots')
    print(f"Output directory: {output_dir}")
    
    # Generate plots/reports
    if args.text_only:
        generated_files = generate_text_report(metrics_data, output_dir)
    else:
        generated_files = generate_matplotlib_plots(metrics_data, output_dir)
    
    # Report results
    if generated_files:
        print(f"\n✅ Successfully generated {len(generated_files)} files:")
        for file_path in generated_files:
            print(f"  - {file_path}")
        print(f"\nFiles are ready for viewing!")
    else:
        print("❌ No files were generated.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())