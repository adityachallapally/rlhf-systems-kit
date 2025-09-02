#!/usr/bin/env python3
"""
Tests for TRL Integration

This module tests the TRL integration functionality including:
- Training callbacks
- PPO-specific monitoring
- Checkpoint analysis
- Reward model integration
"""

import os
import sys
import tempfile
import shutil
import unittest
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from datetime import datetime

# Import TRL integration components
from rlhf_core.trl_integration import (
    TRLIntegrationConfig,
    PPOMonitoringCallback,
    CheckpointAnalyzer,
    RewardModelIntegrator,
    TRLIntegrationManager
)


class TestPPOMonitoringCallback(unittest.TestCase):
    """Test PPO monitoring callback functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.callback = PPOMonitoringCallback(
            anomaly_threshold=2.0,
            log_dir=self.temp_dir,
            enable_detailed_logging=True
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_step_monitoring(self):
        """Test step monitoring functionality."""
        # Test step begin
        self.callback.on_step_begin(0, {"step": 0})
        
        # Test step end with normal metrics
        logs = {
            "kl_div": 0.1,
            "policy_loss": 0.5,
            "value_loss": 0.3,
            "reward": 2.0,
            "clip_ratio": 0.2,
            "entropy": 0.8
        }
        self.callback.on_step_end(0, logs)
        
        # Verify metrics are tracked
        self.assertIn("kl_div", self.callback.metric_history)
        self.assertEqual(len(self.callback.metric_history["kl_div"]), 1)
        self.assertEqual(self.callback.metric_history["kl_div"][0], 0.1)
    
    def test_anomaly_detection(self):
        """Test anomaly detection functionality."""
        # Add some normal values
        for i in range(10):
            logs = {
                "kl_div": 0.1 + np.random.normal(0, 0.01),
                "policy_loss": 0.5 + np.random.normal(0, 0.01),
                "value_loss": 0.3 + np.random.normal(0, 0.01),
                "reward": 2.0 + np.random.normal(0, 0.1),
                "clip_ratio": 0.2 + np.random.normal(0, 0.01),
                "entropy": 0.8 + np.random.normal(0, 0.01)
            }
            self.callback.on_step_end(i, logs)
        
        # Add an anomalous value
        anomalous_logs = {
            "kl_div": 1.0,  # Much higher than normal
            "policy_loss": 0.5,
            "value_loss": 0.3,
            "reward": 2.0,
            "clip_ratio": 0.2,
            "entropy": 0.8
        }
        self.callback.on_step_end(10, anomalous_logs)
        
        # Check that anomaly was detected
        # Note: The actual anomaly detection happens in the callback
        # We can verify the metric history was updated
        self.assertEqual(len(self.callback.metric_history["kl_div"]), 11)
    
    def test_epoch_monitoring(self):
        """Test epoch monitoring functionality."""
        # Test epoch begin
        self.callback.on_epoch_begin(0, {"epoch": 0})
        
        # Test epoch end
        self.callback.on_epoch_end(0, {"epoch": 0})
        
        # Verify epoch summary was created
        summary_path = os.path.join(self.temp_dir, "epoch_0_summary.json")
        self.assertTrue(os.path.exists(summary_path))


class TestCheckpointAnalyzer(unittest.TestCase):
    """Test checkpoint analysis functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = CheckpointAnalyzer(log_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_checkpoint_analysis(self):
        """Test checkpoint analysis functionality."""
        # Create a dummy checkpoint
        dummy_checkpoint = {
            "model": {
                "layer1.weight": torch.randn(10, 5),
                "layer1.bias": torch.randn(5),
                "layer2.weight": torch.randn(5, 2),
                "layer2.bias": torch.randn(2)
            },
            "trainer_state": {
                "global_step": 100,
                "learning_rate": 1e-5,
                "epoch": 5
            }
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.temp_dir, "test_checkpoint.pt")
        torch.save(dummy_checkpoint, checkpoint_path)
        
        # Analyze checkpoint
        analysis = self.analyzer.analyze_checkpoint(checkpoint_path, step=100)
        
        # Verify analysis results
        self.assertIn("health_score", analysis)
        self.assertIn("weight_stats", analysis)
        self.assertIn("training_metrics", analysis)
        self.assertIn("issues", analysis)
        self.assertIn("recommendations", analysis)
        
        # Health score should be between 0 and 1
        self.assertGreaterEqual(analysis["health_score"], 0.0)
        self.assertLessEqual(analysis["health_score"], 1.0)
    
    def test_checkpoint_comparison(self):
        """Test checkpoint comparison functionality."""
        # Create two dummy checkpoints
        checkpoint1 = {
            "model": {
                "layer1.weight": torch.randn(10, 5),
                "layer1.bias": torch.randn(5)
            }
        }
        
        checkpoint2 = {
            "model": {
                "layer1.weight": torch.randn(10, 5) + 0.1,  # Slightly different
                "layer1.bias": torch.randn(5) + 0.1
            }
        }
        
        # Save checkpoints
        path1 = os.path.join(self.temp_dir, "checkpoint1.pt")
        path2 = os.path.join(self.temp_dir, "checkpoint2.pt")
        torch.save(checkpoint1, path1)
        torch.save(checkpoint2, path2)
        
        # Analyze with comparison
        analysis = self.analyzer.analyze_checkpoint(path2, step=100, reference_checkpoint=path1)
        
        # Verify comparison results
        self.assertIn("comparison", analysis)
        self.assertIn("weight_differences", analysis["comparison"])
        self.assertIn("overall_drift", analysis["comparison"])


class TestRewardModelIntegrator(unittest.TestCase):
    """Test reward model integration functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.integrator = RewardModelIntegrator(log_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_reward_monitoring(self):
        """Test reward model monitoring functionality."""
        # Test with normal rewards
        rewards = [2.1, 2.3, 1.9, 2.0, 2.2]
        analysis = self.integrator.monitor_reward_model(rewards, step=0)
        
        # Verify analysis results
        self.assertIn("reward_stats", analysis)
        self.assertIn("reliability_metrics", analysis)
        self.assertIn("anomalies", analysis)
        self.assertIn("recommendations", analysis)
        
        # Check reward statistics
        stats = analysis["reward_stats"]
        self.assertAlmostEqual(stats["mean"], np.mean(rewards), places=2)
        self.assertAlmostEqual(stats["std"], np.std(rewards), places=2)
        self.assertEqual(stats["min"], min(rewards))
        self.assertEqual(stats["max"], max(rewards))
    
    def test_anomaly_detection(self):
        """Test reward anomaly detection."""
        # Add some normal rewards
        for i in range(5):
            normal_rewards = [2.0 + np.random.normal(0, 0.1) for _ in range(5)]
            self.integrator.monitor_reward_model(normal_rewards, step=i)
        
        # Add anomalous rewards
        anomalous_rewards = [2.0, 2.1, 10.0, 2.0, 1.9]  # One outlier
        analysis = self.integrator.monitor_reward_model(anomalous_rewards, step=5)
        
        # Check that anomaly was detected
        self.assertGreater(len(analysis["anomalies"]), 0)
    
    def test_reliability_metrics(self):
        """Test reliability metrics calculation."""
        # Add consistent rewards
        for i in range(10):
            consistent_rewards = [2.0, 2.0, 2.0, 2.0, 2.0]
            analysis = self.integrator.monitor_reward_model(consistent_rewards, step=i)
        
        # Check reliability metrics
        reliability = analysis["reliability_metrics"]
        self.assertIn("stability", reliability)
        self.assertIn("consistency", reliability)
        self.assertIn("variance", reliability)
        
        # For consistent rewards, stability and consistency should be high
        self.assertGreater(reliability["stability"], 0.8)
        self.assertGreater(reliability["consistency"], 0.8)


class TestTRLIntegrationConfig(unittest.TestCase):
    """Test TRL integration configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TRLIntegrationConfig()
        
        # Test default values
        self.assertEqual(config.model_name, "gpt2")
        self.assertEqual(config.learning_rate, 1e-5)
        self.assertEqual(config.batch_size, 4)
        self.assertTrue(config.enable_profiling)
        self.assertTrue(config.enable_checkpoint_analysis)
        self.assertTrue(config.enable_reward_monitoring)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TRLIntegrationConfig(
            model_name="gpt2-medium",
            learning_rate=2e-5,
            batch_size=8,
            enable_profiling=False,
            anomaly_detection_threshold=5.0
        )
        
        # Test custom values
        self.assertEqual(config.model_name, "gpt2-medium")
        self.assertEqual(config.learning_rate, 2e-5)
        self.assertEqual(config.batch_size, 8)
        self.assertFalse(config.enable_profiling)
        self.assertEqual(config.anomaly_detection_threshold, 5.0)


class TestTRLIntegrationManager(unittest.TestCase):
    """Test TRL integration manager."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TRLIntegrationConfig(
            logging_dir=self.temp_dir,
            enable_profiling=False,  # Disable for testing
            enable_checkpoint_analysis=True,
            enable_reward_monitoring=True
        )
        self.manager = TRLIntegrationManager(self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        # Verify components are initialized
        self.assertIsNotNone(self.manager.ppo_callback)
        self.assertIsNotNone(self.manager.checkpoint_analyzer)
        self.assertIsNotNone(self.manager.reward_integrator)
        self.assertIsNone(self.manager.profiler)  # Disabled in config
        self.assertIsNone(self.manager.trl_trainer)  # Not set up yet
    
    def test_configuration(self):
        """Test configuration handling."""
        self.assertEqual(self.manager.config.model_name, "gpt2")
        self.assertEqual(self.manager.config.learning_rate, 1e-5)
        self.assertEqual(self.manager.log_dir, self.temp_dir)


def run_integration_tests():
    """Run all integration tests."""
    print("🧪 Running TRL Integration Tests")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestPPOMonitoringCallback))
    test_suite.addTest(unittest.makeSuite(TestCheckpointAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestRewardModelIntegrator))
    test_suite.addTest(unittest.makeSuite(TestTRLIntegrationConfig))
    test_suite.addTest(unittest.makeSuite(TestTRLIntegrationManager))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return True
    else:
        print("\n❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)