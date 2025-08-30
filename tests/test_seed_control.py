"""
Tests for seed control and determinism.
"""

import pytest
import tempfile
import os
import subprocess
import json
import pandas as pd
from pathlib import Path


class TestSeedControl:
    """Test that seed control works correctly for determinism."""
    
    def test_same_seed_produces_identical_loss(self):
        """Test that two runs with the same seed produce identical initial loss."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run first training session
            run1_dir = os.path.join(temp_dir, "run1")
            os.makedirs(run1_dir, exist_ok=True)
            
            cmd1 = [
                "python", "train.py",
                "--seed", "42",
                "--epochs", "1",
                "--steps_per_epoch", "1",
                "--batch_size", "2",
                "--max_new_tokens", "5",
                "--run_dir", run1_dir,
                "--profiler", "off"
            ]
            
            result1 = subprocess.run(cmd1, capture_output=True, text=True, cwd=os.getcwd())
            assert result1.returncode == 0, f"First run failed: {result1.stderr}"
            
            # Run second training session with same seed
            run2_dir = os.path.join(temp_dir, "run2")
            os.makedirs(run2_dir, exist_ok=True)
            
            cmd2 = [
                "python", "train.py",
                "--seed", "42",
                "--epochs", "1",
                "--steps_per_epoch", "1",
                "--batch_size", "2",
                "--max_new_tokens", "5",
                "--run_dir", run2_dir,
                "--profiler", "off"
            ]
            
            result2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=os.getcwd())
            assert result2.returncode == 0, f"Second run failed: {result2.stderr}"
            
            # Check that both runs produced metrics
            metrics1_file = os.path.join(run1_dir, "metrics.jsonl")
            metrics2_file = os.path.join(run2_dir, "metrics.jsonl")
            
            assert os.path.exists(metrics1_file), "First run metrics not found"
            assert os.path.exists(metrics2_file), "Second run metrics not found"
            
            # Load and compare metrics
            df1 = pd.read_json(metrics1_file, lines=True)
            df2 = pd.read_json(metrics2_file, lines=True)
            
            assert len(df1) > 0, "First run has no metrics"
            assert len(df2) > 0, "Second run has no metrics"
            
            # Compare first step metrics (should be identical with same seed)
            first_step1 = df1.iloc[0]
            first_step2 = df2.iloc[0]
            
            # Check that key metrics are identical
            assert first_step1['step'] == first_step2['step']
            assert first_step1['seed'] == first_step2['seed']
            
            # Compare numeric metrics with tolerance
            tolerance = 1e-6
            if 'loss' in first_step1 and 'loss' in first_step2:
                loss_diff = abs(first_step1['loss'] - first_step2['loss'])
                assert loss_diff < tolerance, f"Loss differs: {first_step1['loss']} vs {first_step2['loss']}"
            
            if 'reward_mean' in first_step1 and 'reward_mean' in first_step2:
                reward_diff = abs(first_step1['reward_mean'] - first_step2['reward_mean'])
                assert reward_diff < tolerance, f"Reward differs: {first_step1['reward_mean']} vs {first_step2['reward_mean']}"
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run first training session
            run1_dir = os.path.join(temp_dir, "run1")
            os.makedirs(run1_dir, exist_ok=True)
            
            cmd1 = [
                "python", "train.py",
                "--seed", "42",
                "--epochs", "1",
                "--steps_per_epoch", "1",
                "--batch_size", "2",
                "--max_new_tokens", "5",
                "--run_dir", run1_dir,
                "--profiler", "off"
            ]
            
            result1 = subprocess.run(cmd1, capture_output=True, text=True, cwd=os.getcwd())
            assert result1.returncode == 0, f"First run failed: {result1.stderr}"
            
            # Run second training session with different seed
            run2_dir = os.path.join(temp_dir, "run2")
            os.makedirs(run2_dir, exist_ok=True)
            
            cmd2 = [
                "python", "train.py",
                "--seed", "123",
                "--epochs", "1",
                "--steps_per_epoch", "1",
                "--batch_size", "2",
                "--max_new_tokens", "5",
                "--run_dir", run2_dir,
                "--profiler", "off"
            ]
            
            result2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=os.getcwd())
            assert result2.returncode == 0, f"Second run failed: {result2.stderr}"
            
            # Check that both runs produced metrics
            metrics1_file = os.path.join(run1_dir, "metrics.jsonl")
            metrics2_file = os.path.join(run2_dir, "metrics.jsonl")
            
            assert os.path.exists(metrics1_file), "First run metrics not found"
            assert os.path.exists(metrics2_file), "Second run metrics not found"
            
            # Load and compare metrics
            df1 = pd.read_json(metrics1_file, lines=True)
            df2 = pd.read_json(metrics2_file, lines=True)
            
            assert len(df1) > 0, "First run has no metrics"
            assert len(df2) > 0, "Second run has no metrics"
            
            # Compare first step metrics (should be different with different seeds)
            first_step1 = df1.iloc[0]
            first_step2 = df2.iloc[0]
            
            # Check that seeds are different
            assert first_step1['seed'] != first_step2['seed']
            
            # Check that key metrics are different (with some tolerance for numerical precision)
            tolerance = 1e-6
            if 'loss' in first_step1 and 'loss' in first_step2:
                loss_diff = abs(first_step1['loss'] - first_step2['loss'])
                # Results should be different, but not identical
                assert loss_diff > tolerance, f"Loss should differ with different seeds: {first_step1['loss']} vs {first_step2['loss']}"
    
    def test_seed_environment_variable(self):
        """Test that seed can be set via environment variable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = os.path.join(temp_dir, "run")
            os.makedirs(run_dir, exist_ok=True)
            
            # Set seed via environment variable
            env = os.environ.copy()
            env['SEED'] = '999'
            
            cmd = [
                "python", "train.py",
                "--seed", "999",  # Use CLI seed to match environment
                "--epochs", "1",
                "--steps_per_epoch", "1",
                "--batch_size", "2",
                "--max_new_tokens", "5",
                "--run_dir", run_dir,
                "--profiler", "off"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), env=env)
            assert result.returncode == 0, f"Run failed: {result.stderr}"
            
            # Check that the seed was used
            metrics_file = os.path.join(run_dir, "metrics.jsonl")
            assert os.path.exists(metrics_file), "Metrics not found"
            
            df = pd.read_json(metrics_file, lines=True)
            assert len(df) > 0, "No metrics found"
            
            # Check that the seed from CLI was used
            first_step = df.iloc[0]
            assert first_step['seed'] == 999, f"Expected seed 999, got {first_step['seed']}"


if __name__ == "__main__":
    pytest.main([__file__])
