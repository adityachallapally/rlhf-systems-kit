#!/usr/bin/env python3
"""
Test script to verify determinism fixes for M1.
"""

import os
import subprocess
import tempfile
import json
from pathlib import Path

def test_determinism():
    """Test that two runs with the same seed produce identical logs."""
    print("Testing M1: Determinism...")
    
    # Set environment variables
    env = os.environ.copy()
    env['SEED'] = '123'
    
    # Create temporary directories for test runs
    with tempfile.TemporaryDirectory() as temp_dir:
        run1_dir = Path(temp_dir) / "run1"
        run2_dir = Path(temp_dir) / "run2"
        
        # Run training twice with same seed
        print("Running first training run...")
        result1 = subprocess.run([
            'python', 'train.py', 
            '--seed', '123',
            '--epochs', '1',
            '--steps_per_epoch', '2',
            '--batch_size', '2',
            '--max_new_tokens', '5',
            '--output_dir', str(run1_dir)
        ], env=env, capture_output=True, text=True)
        
        if result1.returncode != 0:
            print(f"First run failed: {result1.stderr}")
            return False
        
        print("Running second training run...")
        result2 = subprocess.run([
            'python', 'train.py', 
            '--seed', '123',
            '--epochs', '1',
            '--steps_per_epoch', '2',
            '--batch_size', '2',
            '--max_new_tokens', '5',
            '--output_dir', str(run2_dir)
        ], env=env, capture_output=True, text=True)
        
        if result2.returncode != 0:
            print(f"Second run failed: {result2.stderr}")
            return False
        
        # Check if log files exist
        log1 = run1_dir / "run_*/logs/train.jsonl"
        log2 = run2_dir / "run_*/logs/train.jsonl"
        
        log1_files = list(run1_dir.glob("run_*/logs/train.jsonl"))
        log2_files = list(run2_dir.glob("run_*/logs/train.jsonl"))
        
        if not log1_files or not log2_files:
            print("Log files not found")
            return False
        
        log1_path = log1_files[0]
        log2_path = log2_files[0]
        
        print(f"Comparing logs:")
        print(f"  Run 1: {log1_path}")
        print(f"  Run 2: {log2_path}")
        
        # Read and compare logs
        with open(log1_path, 'r') as f1, open(log2_path, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
        
        if len(lines1) != len(lines2):
            print(f"Log lengths differ: {len(lines1)} vs {len(lines2)}")
            return False
        
        # Compare each line (excluding any remaining timestamps)
        for i, (line1, line2) in enumerate(zip(lines1, lines2)):
            try:
                data1 = json.loads(line1.strip())
                data2 = json.loads(line2.strip())
                
                # Remove any timestamp fields for comparison
                if 'timestamp' in data1:
                    del data1['timestamp']
                if 'timestamp' in data2:
                    del data2['timestamp']
                
                if data1 != data2:
                    print(f"Log entries differ at line {i+1}:")
                    print(f"  Run 1: {data1}")
                    print(f"  Run 2: {data2}")
                    return False
                    
            except json.JSONDecodeError as e:
                print(f"JSON decode error at line {i+1}: {e}")
                return False
        
        print("✅ Determinism test PASSED: Logs are identical")
        return True

if __name__ == "__main__":
    success = test_determinism()
    exit(0 if success else 1)
