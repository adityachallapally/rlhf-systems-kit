#!/usr/bin/env python3
"""
Test script for divergence analysis functionality.

This script creates synthetic training data and tests the divergence analysis
to ensure it works correctly.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Import the divergence analysis module
from rlhf_core.divergence import (
    DivergenceReport, 
    first_divergence, 
    generate_drift_card, 
    analyze_multiple_runs
)


def create_synthetic_training_data(output_dir: str = "test_data"):
    """Create synthetic training data for testing divergence analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create two training runs with different characteristics
    # Run 1: Stable training
    run1_data = []
    for step in range(1, 101):
        # Stable metrics with some noise
        run1_data.append({
            'step': step,
            'loss': 2.0 + 0.1 * np.sin(step * 0.1) + 0.05 * np.random.randn(),
            'reward_mean': 0.5 + 0.02 * np.sin(step * 0.05) + 0.01 * np.random.randn(),
            'kl': 0.1 + 0.01 * np.sin(step * 0.08) + 0.005 * np.random.randn(),
            'entropy': 0.8 + 0.02 * np.sin(step * 0.06) + 0.01 * np.random.randn(),
            'grad_norm': 0.5 + 0.1 * np.sin(step * 0.04) + 0.05 * np.random.randn()
        })
    
    # Run 2: Diverges after step 50
    run2_data = []
    for step in range(1, 101):
        if step <= 50:
            # Same as run 1 for first 50 steps
            run2_data.append({
                'step': step,
                'loss': 2.0 + 0.1 * np.sin(step * 0.1) + 0.05 * np.random.randn(),
                'reward_mean': 0.5 + 0.02 * np.sin(step * 0.05) + 0.01 * np.random.randn(),
                'kl': 0.1 + 0.01 * np.sin(step * 0.08) + 0.005 * np.random.randn(),
                'entropy': 0.8 + 0.02 * np.sin(step * 0.06) + 0.01 * np.random.randn(),
                'grad_norm': 0.5 + 0.1 * np.sin(step * 0.04) + 0.05 * np.random.randn()
            })
        else:
            # Diverges after step 50
            divergence_factor = (step - 50) / 50.0  # Gradual divergence
            run2_data.append({
                'step': step,
                'loss': 2.0 + 0.1 * np.sin(step * 0.1) + 0.05 * np.random.randn() + divergence_factor * 0.5,
                'reward_mean': 0.5 + 0.02 * np.sin(step * 0.05) + 0.01 * np.random.randn() - divergence_factor * 0.2,
                'kl': 0.1 + 0.01 * np.sin(step * 0.08) + 0.005 * np.random.randn() + divergence_factor * 0.3,
                'entropy': 0.8 + 0.02 * np.sin(step * 0.06) + 0.01 * np.random.randn() - divergence_factor * 0.4,
                'grad_norm': 0.5 + 0.1 * np.sin(step * 0.04) + 0.05 * np.random.randn() + divergence_factor * 1.0
            })
    
    # Save the data
    run1_path = os.path.join(output_dir, "run1_stable.jsonl")
    run2_path = os.path.join(output_dir, "run2_divergent.jsonl")
    
    with open(run1_path, 'w') as f:
        for entry in run1_data:
            f.write(json.dumps(entry) + '\n')
    
    with open(run2_path, 'w') as f:
        for entry in run2_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created synthetic training data:")
    print(f"  Run 1 (stable): {run1_path}")
    print(f"  Run 2 (divergent): {run2_path}")
    
    return run1_path, run2_path


def test_divergence_analysis():
    """Test the divergence analysis functionality."""
    print("="*60)
    print("Testing Divergence Analysis")
    print("="*60)
    
    # Create test data
    run1_path, run2_path = create_synthetic_training_data()
    
    # Test 1: Basic divergence analysis
    print("\n1. Testing basic divergence analysis...")
    report = first_divergence(
        run1_logs=run1_path,
        run2_logs=run2_path,
        window_size=20,
        z_score_threshold=2.0,
        min_overlapping_steps=10
    )
    
    print(f"   Diverged: {report.diverged}")
    print(f"   Overlapping steps: {report.overlapping_steps}")
    if report.diverged:
        print(f"   Divergence step: {report.divergence_step}")
        if report.divergence_z_scores:
            print("   Diverged metrics:")
            for metric, details in report.divergence_z_scores.items():
                print(f"     {metric}: diff={details['difference']:.3f}")
    
    # Test 2: Generate drift card
    print("\n2. Testing drift card generation...")
    try:
        card_path = generate_drift_card(report, "test_output")
        print(f"   Drift card generated: {card_path}")
    except Exception as e:
        print(f"   Error generating drift card: {e}")
    
    # Test 3: Test with different parameters
    print("\n3. Testing with different parameters...")
    report2 = first_divergence(
        run1_logs=run1_path,
        run2_logs=run2_path,
        window_size=10,
        z_score_threshold=1.5,
        min_overlapping_steps=5
    )
    
    print(f"   Diverged: {report2.diverged}")
    print(f"   Overlapping steps: {report2.overlapping_steps}")
    
    # Test 4: Test multiple run analysis
    print("\n4. Testing multiple run analysis...")
    try:
        reports = analyze_multiple_runs(
            [run1_path, run2_path],
            window_size=20,
            z_score_threshold=2.0
        )
        print(f"   Generated {len(reports)} reports")
        for i, r in enumerate(reports):
            print(f"   Report {i+1}: diverged={r.diverged}")
    except Exception as e:
        print(f"   Error in multiple run analysis: {e}")
    
    # Test 5: Test with insufficient data
    print("\n5. Testing with insufficient data...")
    # Create a very short run
    short_run_data = [{'step': i, 'loss': 1.0 + i*0.1} for i in range(1, 6)]
    short_run_path = "test_data/short_run.jsonl"
    
    with open(short_run_path, 'w') as f:
        for entry in short_run_data:
            f.write(json.dumps(entry) + '\n')
    
    report3 = first_divergence(
        run1_logs=short_run_path,
        run2_logs=run2_path,
        window_size=20,
        min_overlapping_steps=10
    )
    
    print(f"   Diverged: {report3.diverged}")
    print(f"   Overlapping steps: {report3.overlapping_steps}")
    if 'error' in report3.summary:
        print(f"   Error: {report3.summary['error']}")
    
    # Cleanup
    os.remove(short_run_path)
    
    print("\n" + "="*60)
    print("Divergence Analysis Tests Complete!")
    print("="*60)
    
    return report


if __name__ == "__main__":
    try:
        report = test_divergence_analysis()
        print(f"\nFinal test result: {'PASSED' if report.diverged else 'FAILED'}")
        print("Note: The test is designed so that the runs should diverge after step 50.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()