#!/usr/bin/env python3
"""
Test script to verify CPU-only training works without CUDA errors.
"""

import os
import sys
import subprocess

def test_cpu_training():
    """Test that training can run on CPU without CUDA errors."""
    print("Testing CPU-only training...")
    
    # Set environment variables
    env = os.environ.copy()
    env['SEED'] = '123'
    
    # Test with explicit CPU device
    try:
        result = subprocess.run([
            'python', 'train.py', 
            '--seed', '123',
            '--epochs', '1',
            '--steps_per_epoch', '2',
            '--batch_size', '2',
            '--max_new_tokens', '5',
            '--device', 'cpu',
            '--profiler', 'off'  # Disable profiler for simple test
        ], env=env, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ CPU training test PASSED")
            return True
        else:
            print(f"❌ CPU training test FAILED")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ CPU training test TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ CPU training test ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_cpu_training()
    sys.exit(0 if success else 1)