#!/usr/bin/env python3
"""
Dashboard Metrics Checker for M3 Acceptance

This script verifies that all required metrics exist in the training logs.
"""

import json
import sys
import os
from pathlib import Path

def check_dashboard_metrics():
    """Check that all required dashboard metrics exist in the logs."""
    # Required metrics for M3 acceptance
    required_metrics = {
        'kl', 'kl_target_error', 'entropy', 'reward_mean', 'reward_var',
        'grad_norm', 'clip_frac', 'adv_var', 'tokens_per_second'
    }
    
    # Find the latest run logs
    latest_link = Path("runs/latest")
    if not latest_link.exists():
        print("ERROR: runs/latest symlink not found")
        print("Run training first to generate logs")
        return False
    
    # Check if it's a symlink and resolve it
    if latest_link.is_symlink():
        actual_path = latest_link.resolve()
        print(f"Latest run: {actual_path}")
    else:
        actual_path = latest_link
        print(f"Latest run: {actual_path}")
    
    # Look for train.jsonl in the logs directory
    log_file = actual_path / "logs" / "train.jsonl"
    if not log_file.exists():
        print(f"ERROR: Training logs not found at {log_file}")
        return False
    
    print(f"Checking metrics in: {log_file}")
    
    # Read and analyze the logs
    found_metrics = set()
    total_lines = 0
    
    try:
        with open(log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    found_metrics.update(data.keys())
                    total_lines += 1
                    
                    # Stop after checking first 100 lines for efficiency
                    if total_lines >= 100:
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"WARNING: Invalid JSON at line {line_num}: {e}")
                    continue
        
        print(f"Analyzed {total_lines} log entries")
        
        # Check which metrics are missing
        missing_metrics = required_metrics - found_metrics
        found_required = required_metrics & found_metrics
        
        print(f"\nFound {len(found_required)}/{len(required_metrics)} required metrics:")
        for metric in sorted(found_required):
            print(f"  ✓ {metric}")
        
        if missing_metrics:
            print(f"\nMissing {len(missing_metrics)} required metrics:")
            for metric in sorted(missing_metrics):
                print(f"  ❌ {metric}")
            return False
        else:
            print(f"\n✓ All required metrics found!")
            return True
            
    except Exception as e:
        print(f"ERROR: Failed to read log file: {e}")
        return False

def main():
    """Main function to run the check."""
    print("Checking M3 Dashboard Metrics...")
    print("=" * 40)
    
    success = check_dashboard_metrics()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 M3 Dashboard Metrics: PASSED")
        return 0
    else:
        print("❌ M3 Dashboard Metrics: FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
