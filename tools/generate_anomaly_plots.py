#!/usr/bin/env python3
"""
Anomaly Detection Plot Generator

Generates visualizations for anomaly detection results from RLHF training.

Usage:
    python tools/generate_anomaly_plots.py --run runs/latest
    python tools/generate_anomaly_plots.py --metrics-file runs/run_123/logs/train.jsonl
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


def detect_anomalies(metrics_data: list) -> dict:
    """Simple anomaly detection on training metrics."""
    anomalies = {
        'lr_changes': [],
        'loss_spikes': [],
        'kl_spikes': [],
        'gradient_anomalies': []
    }
    
    if len(metrics_data) < 2:
        return anomalies
    
    # Detect learning rate changes
    prev_lr = None
    for i, step in enumerate(metrics_data):
        lr = step.get('learning_rate')
        if lr is not None and prev_lr is not None:
            if abs(lr - prev_lr) / max(prev_lr, 1e-8) > 0.1:  # 10% change
                anomalies['lr_changes'].append({
                    'step': step.get('step', i),
                    'old_lr': prev_lr,
                    'new_lr': lr,
                    'change_pct': (lr - prev_lr) / max(prev_lr, 1e-8) * 100
                })
        prev_lr = lr
    
    # Detect loss spikes
    losses = [step.get('total_loss', 0) for step in metrics_data if step.get('total_loss') is not None]
    if len(losses) > 5:
        # Simple outlier detection - values beyond 2 standard deviations
        import statistics
        mean_loss = statistics.mean(losses)
        std_loss = statistics.stdev(losses) if len(losses) > 1 else 0
        
        for i, step in enumerate(metrics_data):
            loss = step.get('total_loss')
            if loss is not None and std_loss > 0:
                z_score = abs(loss - mean_loss) / std_loss
                if z_score > 2.0:  # 2 standard deviations
                    anomalies['loss_spikes'].append({
                        'step': step.get('step', i),
                        'loss': loss,
                        'z_score': z_score,
                        'mean_loss': mean_loss
                    })
    
    # Detect KL divergence spikes
    kl_values = [step.get('kl_mean', 0) for step in metrics_data if step.get('kl_mean') is not None]
    if len(kl_values) > 5:
        import statistics
        mean_kl = statistics.mean([abs(k) for k in kl_values])  # Use absolute values
        
        for i, step in enumerate(metrics_data):
            kl = step.get('kl_mean')
            if kl is not None and abs(kl) > mean_kl * 5:  # 5x the mean
                anomalies['kl_spikes'].append({
                    'step': step.get('step', i),
                    'kl': kl,
                    'threshold': mean_kl * 5
                })
    
    return anomalies


def generate_anomaly_report(metrics_data: list, output_dir: str) -> list:
    """Generate anomaly detection report."""
    if not metrics_data:
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect anomalies
    anomalies = detect_anomalies(metrics_data)
    
    # Generate report
    report_path = os.path.join(output_dir, 'anomaly_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("RLHF Training Anomaly Detection Report\n")
        f.write("=" * 60 + "\n\n")
        
        total_anomalies = sum(len(v) for v in anomalies.values())
        f.write(f"Total Anomalies Detected: {total_anomalies}\n")
        f.write(f"Training Steps Analyzed: {len(metrics_data)}\n\n")
        
        # Learning rate changes
        if anomalies['lr_changes']:
            f.write("Learning Rate Changes:\n")
            f.write("-" * 30 + "\n")
            for change in anomalies['lr_changes']:
                f.write(f"Step {change['step']}: {change['old_lr']:.2e} -> {change['new_lr']:.2e} "
                       f"({change['change_pct']:+.1f}%)\n")
            f.write("\n")
        else:
            f.write("✓ No learning rate anomalies detected\n\n")
        
        # Loss spikes
        if anomalies['loss_spikes']:
            f.write("Loss Spikes (>2 std deviations):\n")
            f.write("-" * 30 + "\n")
            for spike in anomalies['loss_spikes']:
                f.write(f"Step {spike['step']}: Loss={spike['loss']:.6f}, "
                       f"Z-score={spike['z_score']:.2f}\n")
            f.write("\n")
        else:
            f.write("✓ No loss spikes detected\n\n")
        
        # KL spikes
        if anomalies['kl_spikes']:
            f.write("KL Divergence Spikes:\n")
            f.write("-" * 30 + "\n")
            for spike in anomalies['kl_spikes']:
                f.write(f"Step {spike['step']}: KL={spike['kl']:.6f}, "
                       f"Threshold={spike['threshold']:.6f}\n")
            f.write("\n")
        else:
            f.write("✓ No KL divergence spikes detected\n\n")
        
        # Summary
        if total_anomalies == 0:
            f.write("🎉 No anomalies detected - training appears stable!\n")
        else:
            f.write(f"⚠️  {total_anomalies} anomalies detected - review recommended\n")
    
    print(f"Generated anomaly report: {report_path}")
    return [report_path]


def generate_anomaly_plots(metrics_data: list, output_dir: str) -> list:
    """Generate anomaly detection plots using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, falling back to text report")
        return generate_anomaly_report(metrics_data, output_dir)
    
    if not metrics_data:
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    # Extract data
    steps = []
    losses = []
    kl_values = []
    learning_rates = []
    
    for step_data in metrics_data:
        step = step_data.get('step')
        if step is not None:
            steps.append(step)
            losses.append(step_data.get('total_loss', 0))
            kl_values.append(step_data.get('kl_mean', 0))
            learning_rates.append(step_data.get('learning_rate', 0))
    
    if not steps:
        return generate_anomaly_report(metrics_data, output_dir)
    
    # Detect anomalies
    anomalies = detect_anomalies(metrics_data)
    
    # Plot 1: Loss with anomaly markers
    if losses:
        plt.figure(figsize=(12, 6))
        plt.plot(steps, losses, 'b-', linewidth=2, alpha=0.7, label='Loss')
        
        # Mark anomalies
        for spike in anomalies['loss_spikes']:
            spike_step = spike['step']
            if spike_step in steps:
                idx = steps.index(spike_step)
                plt.scatter(spike_step, losses[idx], color='red', s=100, marker='x', 
                           label='Loss Spike' if spike == anomalies['loss_spikes'][0] else "")
        
        plt.title('Training Loss with Anomaly Detection', fontsize=14, fontweight='bold')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'loss_anomalies.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        generated_files.append(plot_path)
        print(f"Generated loss anomaly plot: {plot_path}")
    
    # Plot 2: KL Divergence with anomaly markers
    if kl_values:
        plt.figure(figsize=(12, 6))
        plt.plot(steps, kl_values, 'r-', linewidth=2, alpha=0.7, label='KL Divergence')
        
        # Mark anomalies
        for spike in anomalies['kl_spikes']:
            spike_step = spike['step']
            if spike_step in steps:
                idx = steps.index(spike_step)
                plt.scatter(spike_step, kl_values[idx], color='orange', s=100, marker='x',
                           label='KL Spike' if spike == anomalies['kl_spikes'][0] else "")
        
        plt.title('KL Divergence with Anomaly Detection', fontsize=14, fontweight='bold')
        plt.xlabel('Training Step')
        plt.ylabel('KL Divergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'kl_anomalies.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        generated_files.append(plot_path)
        print(f"Generated KL anomaly plot: {plot_path}")
    
    # Plot 3: Learning rate changes
    if learning_rates and anomalies['lr_changes']:
        plt.figure(figsize=(12, 6))
        plt.plot(steps, learning_rates, 'g-', linewidth=2, alpha=0.7, label='Learning Rate')
        
        # Mark LR changes
        for change in anomalies['lr_changes']:
            change_step = change['step']
            if change_step in steps:
                idx = steps.index(change_step)
                plt.scatter(change_step, learning_rates[idx], color='purple', s=100, marker='o',
                           label='LR Change' if change == anomalies['lr_changes'][0] else "")
        
        plt.title('Learning Rate with Change Detection', fontsize=14, fontweight='bold')
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'lr_anomalies.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        generated_files.append(plot_path)
        print(f"Generated LR anomaly plot: {plot_path}")
    
    # Also generate text report
    text_files = generate_anomaly_report(metrics_data, output_dir)
    generated_files.extend(text_files)
    
    return generated_files


def main():
    parser = argparse.ArgumentParser(description='Generate anomaly detection plots/reports')
    parser.add_argument('--run', type=str, default='runs/latest', 
                       help='Path to training run directory')
    parser.add_argument('--metrics-file', type=str,
                       help='Direct path to metrics JSONL file')
    parser.add_argument('--output-dir', type=str, 
                       help='Custom output directory for plots (default: run_dir/anomaly_plots)')
    parser.add_argument('--text-only', action='store_true',
                       help='Generate text report only, skip matplotlib plots')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Anomaly Detection Plot/Report Generator")
    print("="*60)
    
    # Determine metrics file
    if args.metrics_file:
        metrics_file = args.metrics_file
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.join(os.path.dirname(metrics_file), '..', 'anomaly_plots')
    else:
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
        
        metrics_file = os.path.join(run_path, 'logs', 'train.jsonl')
        output_dir = args.output_dir or os.path.join(run_path, 'anomaly_plots')
    
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
    print(f"Output directory: {output_dir}")
    
    # Generate plots/reports
    if args.text_only:
        generated_files = generate_anomaly_report(metrics_data, output_dir)
    else:
        generated_files = generate_anomaly_plots(metrics_data, output_dir)
    
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