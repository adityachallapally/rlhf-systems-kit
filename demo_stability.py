#!/usr/bin/env python3
"""
Demo script for RLHF Stability Dashboard.

This script demonstrates the stability monitoring capabilities
without requiring the full training environment.
"""

import json
import os
from datetime import datetime, timedelta
import random

def create_sample_stability_logs():
    """Create sample stability logs for demonstration."""
    print("📊 Creating sample stability logs...")
    
    # Create runs directory structure
    os.makedirs("runs/demo_run/logs", exist_ok=True)
    os.makedirs("runs/demo_run/tb", exist_ok=True)
    
    # Generate sample data
    data = []
    base_time = datetime.now()
    
    for step in range(1, 25):  # 24 training steps
        # Simulate some instability around step 15
        if step > 15:
            kl = 0.25 + random.uniform(-0.05, 0.05)  # Above threshold
            entropy = 0.08 + random.uniform(-0.02, 0.02)  # Below threshold
        else:
            kl = 0.12 + random.uniform(-0.03, 0.03)  # Normal
            entropy = 0.85 + random.uniform(-0.1, 0.1)  # Normal
        
        # Simulate gradient explosion around step 20
        if step > 20:
            grad_norm = 1500 + random.uniform(-200, 200)  # Above threshold
        else:
            grad_norm = 200 + random.uniform(-50, 50)  # Normal
        
        entry = {
            "timestamp": (base_time + timedelta(minutes=step)).isoformat(),
            "step": step,
            "kl": round(kl, 4),
            "kl_target_err": round(kl - 0.1, 4),  # Target KL is 0.1
            "entropy": round(entropy, 4),
            "reward_mean": round(0.7 + random.uniform(-0.2, 0.2), 4),
            "reward_std": round(0.8 + random.uniform(-0.1, 0.3), 4),
            "advantage_mean": round(0.3 + random.uniform(-0.1, 0.1), 4),
            "advantage_std": round(0.4 + random.uniform(-0.1, 0.1), 4),
            "grad_norm": round(grad_norm, 2),
            "ppo_clip_fraction": round(random.uniform(0.05, 0.15), 4),
            "tokens_per_second": round(150 + random.uniform(-20, 20), 1),
            "batch_size": 4,
            "policy_loss": round(0.6 + random.uniform(-0.1, 0.1), 4),
            "learning_rate": 1e-5
        }
        data.append(entry)
    
    # Write to stability.jsonl
    log_file = "runs/demo_run/logs/stability.jsonl"
    with open(log_file, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"✅ Created sample logs: {log_file}")
    print(f"   - {len(data)} training steps")
    print(f"   - Instability at step 15+ (KL > 0.2)")
    print(f"   - Entropy collapse at step 15+ (entropy < 0.1)")
    print(f"   - Gradient explosion at step 20+ (grad_norm > 1000)")
    
    return log_file

def analyze_stability_logs(log_file):
    """Analyze the stability logs and identify issues."""
    print("\n🔍 Analyzing stability logs...")
    
    data = []
    with open(log_file, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    # Define thresholds
    thresholds = {
        'kl': 0.2,
        'entropy': 0.1,
        'grad_norm': 1000,
        'reward_std': 2.0,
        'kl_target_err': 0.15
    }
    
    # Check for issues
    issues = []
    
    for entry in data:
        step = entry['step']
        
        if entry['kl'] > thresholds['kl']:
            issues.append(f"Step {step}: KL divergence ({entry['kl']:.3f}) above threshold ({thresholds['kl']})")
        
        if entry['entropy'] < thresholds['entropy']:
            issues.append(f"Step {step}: Policy entropy ({entry['entropy']:.3f}) below threshold ({thresholds['entropy']})")
        
        if entry['grad_norm'] > thresholds['grad_norm']:
            issues.append(f"Step {step}: Gradient norm ({entry['grad_norm']:.2e}) above threshold ({thresholds['grad_norm']})")
        
        if entry['reward_std'] > thresholds['reward_std']:
            issues.append(f"Step {step}: High reward variance ({entry['reward_std']:.3f}) above threshold ({thresholds['reward_std']})")
        
        if abs(entry['kl_target_err']) > thresholds['kl_target_err']:
            issues.append(f"Step {step}: KL target error ({entry['kl_target_err']:.3f}) above threshold ({thresholds['kl_target_err']})")
    
    if issues:
        print("⚠️  Stability issues detected:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("✅ No stability issues detected")
    
    # Summary statistics
    print(f"\n📈 Summary Statistics:")
    print(f"   Total steps: {len(data)}")
    print(f"   Average KL: {sum(e['kl'] for e in data) / len(data):.4f}")
    print(f"   Average Entropy: {sum(e['entropy'] for e in data) / len(data):.4f}")
    print(f"   Average Gradient Norm: {sum(e['grad_norm'] for e in data) / len(data):.2e}")
    print(f"   Average Throughput: {sum(e['tokens_per_second'] for e in data) / len(data):.1f} tokens/sec")

def show_dashboard_info():
    """Show information about the live dashboard."""
    print("\n🚀 Live Dashboard Information:")
    print("=" * 40)
    print("The stability dashboard provides real-time monitoring at:")
    print("   http://localhost:8000/")
    print("\nFeatures:")
    print("   ✅ Live-updating metric plots")
    print("   ✅ Warning banners for threshold violations")
    print("   ✅ Real-time stability analysis")
    print("   ✅ Auto-refresh every 10 seconds")
    print("\nTo launch the dashboard:")
    print("   make dashboard")
    print("\nOr run directly:")
    print("   python3 scripts/serve_dashboard.py")

def show_offline_analysis_info():
    """Show information about offline analysis."""
    print("\n📊 Offline Analysis Information:")
    print("=" * 40)
    print("The offline notebook provides comprehensive analysis:")
    print("   notebooks/stability_dashboard.ipynb")
    print("\nFeatures:")
    print("   ✅ Time-series plots for all metrics")
    print("   ✅ Correlation analysis")
    print("   ✅ Detailed instability analysis")
    print("   ✅ Export capabilities")
    print("\nTo use the notebook:")
    print("   jupyter notebook notebooks/stability_dashboard.ipynb")

def main():
    """Main demo function."""
    print("🎯 RLHF Stability Dashboard Demo")
    print("=" * 50)
    print("This demo shows the stability monitoring capabilities")
    print("without requiring the full training environment.\n")
    
    # Create sample logs
    log_file = create_sample_stability_logs()
    
    # Analyze the logs
    analyze_stability_logs(log_file)
    
    # Show dashboard information
    show_dashboard_info()
    
    # Show offline analysis information
    show_offline_analysis_info()
    
    print("\n🎉 Demo completed!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run real training: make train_smoke")
    print("3. Launch live dashboard: make dashboard")
    print("4. View offline analysis: jupyter notebook notebooks/stability_dashboard.ipynb")

if __name__ == "__main__":
    main()