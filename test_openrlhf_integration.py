#!/usr/bin/env python3
"""
OpenRLHF Integration Test Script

This script tests the OpenRLHF integration functionality and demonstrates
its capabilities matching the TRL integration.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rlhf_core.openrlhf_integration import (
    OpenRLHFIntegrationManager,
    OpenRLHFIntegrationConfig,
    create_openrlhf_integration_example
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_openrlhf_config():
    """Test OpenRLHF configuration."""
    logger.info("Testing OpenRLHF configuration...")
    
    config = OpenRLHFIntegrationConfig(
        model_name="gpt2",
        learning_rate=1e-5,
        batch_size=4,
        advantage_estimator="reinforce_baseline",
        normalize_reward=True,
        packing_samples=True,
        vllm_num_engines=8,
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.6,
        zero_stage=3,
        enable_profiling=True,
        enable_checkpoint_analysis=True,
        enable_reward_monitoring=True,
        logging_dir="./test_openrlhf_logs"
    )
    
    # Verify configuration
    assert config.model_name == "gpt2"
    assert config.advantage_estimator == "reinforce_baseline"
    assert config.normalize_reward == True
    assert config.packing_samples == True
    assert config.vllm_num_engines == 8
    assert config.zero_stage == 3
    
    logger.info("✅ OpenRLHF configuration test passed")
    return config


def test_openrlhf_integration_manager():
    """Test OpenRLHF integration manager initialization."""
    logger.info("Testing OpenRLHF integration manager...")
    
    config = OpenRLHFIntegrationConfig(
        model_name="gpt2",
        logging_dir="./test_openrlhf_logs"
    )
    
    manager = OpenRLHFIntegrationManager(config)
    
    # Verify components are initialized
    assert manager.config == config
    assert manager.ppo_callback is not None
    assert manager.checkpoint_analyzer is not None
    assert manager.reward_integrator is not None
    assert manager.openrlhf_trainer is None  # Not initialized yet
    
    logger.info("✅ OpenRLHF integration manager test passed")
    return manager


def test_openrlhf_trainer_setup():
    """Test OpenRLHF trainer setup."""
    logger.info("Testing OpenRLHF trainer setup...")
    
    config = OpenRLHFIntegrationConfig(
        model_name="gpt2",
        logging_dir="./test_openrlhf_logs"
    )
    
    manager = OpenRLHFIntegrationManager(config)
    
    # Setup trainer
    trainer = manager.setup_openrlhf_trainer(
        model_name="gpt2",
        dataset_name="imdb"
    )
    
    # Verify trainer is initialized
    assert trainer is not None
    assert manager.openrlhf_trainer is not None
    assert trainer.config == config
    
    logger.info("✅ OpenRLHF trainer setup test passed")
    return manager, trainer


def test_openrlhf_training_step():
    """Test OpenRLHF training step."""
    logger.info("Testing OpenRLHF training step...")
    
    config = OpenRLHFIntegrationConfig(
        model_name="gpt2",
        logging_dir="./test_openrlhf_logs"
    )
    
    manager = OpenRLHFIntegrationManager(config)
    trainer = manager.setup_openrlhf_trainer("gpt2", "imdb")
    
    # Test training step
    logs = trainer.step()
    
    # Verify logs contain expected metrics
    expected_metrics = [
        "kl_div", "policy_loss", "value_loss", "reward", 
        "clip_ratio", "entropy", "advantage"
    ]
    
    for metric in expected_metrics:
        assert metric in logs
        assert isinstance(logs[metric], (int, float))
    
    # Test OpenRLHF specific metrics
    openrlhf_metrics = ["vllm_latency", "vllm_throughput", "ray_actor_utilization"]
    for metric in openrlhf_metrics:
        assert metric in logs
        assert isinstance(logs[metric], (int, float))
    
    logger.info("✅ OpenRLHF training step test passed")
    return logs


def test_openrlhf_checkpoint_saving():
    """Test OpenRLHF checkpoint saving."""
    logger.info("Testing OpenRLHF checkpoint saving...")
    
    config = OpenRLHFIntegrationConfig(
        model_name="gpt2",
        logging_dir="./test_openrlhf_logs"
    )
    
    manager = OpenRLHFIntegrationManager(config)
    trainer = manager.setup_openrlhf_trainer("gpt2", "imdb")
    
    # Save checkpoint
    checkpoint_path = "./test_openrlhf_logs/test_checkpoint.pt"
    trainer.save_model(checkpoint_path)
    
    # Verify checkpoint exists
    assert os.path.exists(checkpoint_path)
    
    # Test checkpoint analysis
    analysis = manager.checkpoint_analyzer.analyze_checkpoint(
        checkpoint_path, step=1
    )
    
    # Verify analysis contains expected fields
    expected_fields = [
        "step", "timestamp", "framework", "model_path", 
        "health_score", "issues", "recommendations"
    ]
    
    for field in expected_fields:
        assert field in analysis
    
    assert analysis["framework"] == "openrlhf"
    assert analysis["health_score"] >= 0.0 and analysis["health_score"] <= 1.0
    
    logger.info("✅ OpenRLHF checkpoint saving test passed")
    return analysis


def test_openrlhf_reward_monitoring():
    """Test OpenRLHF reward monitoring."""
    logger.info("Testing OpenRLHF reward monitoring...")
    
    config = OpenRLHFIntegrationConfig(
        model_name="gpt2",
        logging_dir="./test_openrlhf_logs"
    )
    
    manager = OpenRLHFIntegrationManager(config)
    
    # Test reward monitoring
    reward_scores = [0.1, 0.2, 0.15, 0.3, 0.25]
    advantage_scores = [0.05, 0.1, 0.08, 0.12, 0.1]
    
    analysis = manager.reward_integrator.monitor_reward_model(
        reward_scores, step=1, advantage_scores=advantage_scores
    )
    
    # Verify analysis contains expected fields
    expected_fields = [
        "step", "timestamp", "framework", "reward_stats", 
        "advantage_stats", "reliability_metrics", "anomalies", "recommendations"
    ]
    
    for field in expected_fields:
        assert field in analysis
    
    assert analysis["framework"] == "openrlhf"
    assert "mean" in analysis["reward_stats"]
    assert "mean" in analysis["advantage_stats"]
    
    logger.info("✅ OpenRLHF reward monitoring test passed")
    return analysis


def test_openrlhf_full_training():
    """Test full OpenRLHF training with monitoring."""
    logger.info("Testing full OpenRLHF training with monitoring...")
    
    config = OpenRLHFIntegrationConfig(
        model_name="gpt2",
        learning_rate=1e-5,
        batch_size=4,
        save_freq=10,  # Save every 10 steps for testing
        enable_profiling=True,
        enable_checkpoint_analysis=True,
        enable_reward_monitoring=True,
        logging_dir="./test_openrlhf_logs"
    )
    
    manager = OpenRLHFIntegrationManager(config)
    trainer = manager.setup_openrlhf_trainer("gpt2", "imdb")
    
    # Run training with monitoring
    results = manager.train_with_monitoring(
        num_steps=20,  # Small number for testing
        save_checkpoints=True
    )
    
    # Verify training results
    expected_fields = [
        "total_steps", "start_time", "framework", "checkpoints_saved",
        "anomalies_detected", "final_metrics"
    ]
    
    for field in expected_fields:
        assert field in results
    
    assert results["framework"] == "openrlhf"
    assert results["total_steps"] == 20
    assert len(results["checkpoints_saved"]) >= 1  # At least one checkpoint saved
    
    # Generate training report
    report_path = manager.generate_training_report(results)
    assert os.path.exists(report_path)
    
    logger.info("✅ Full OpenRLHF training test passed")
    return results


def test_openrlhf_vs_trl_comparison():
    """Test comparison between OpenRLHF and TRL integrations."""
    logger.info("Testing OpenRLHF vs TRL comparison...")
    
    # Test that both integrations have similar structure
    from rlhf_core.trl_integration import TRLIntegrationManager, TRLIntegrationConfig
    
    # OpenRLHF config
    openrlhf_config = OpenRLHFIntegrationConfig(
        model_name="gpt2",
        logging_dir="./test_openrlhf_logs"
    )
    
    # TRL config
    trl_config = TRLIntegrationConfig(
        model_name="gpt2",
        logging_dir="./test_trl_logs"
    )
    
    # Initialize both managers
    openrlhf_manager = OpenRLHFIntegrationManager(openrlhf_config)
    trl_manager = TRLIntegrationManager(trl_config)
    
    # Verify both have similar components
    assert openrlhf_manager.ppo_callback is not None
    assert trl_manager.ppo_callback is not None
    assert openrlhf_manager.checkpoint_analyzer is not None
    assert trl_manager.checkpoint_analyzer is not None
    assert openrlhf_manager.reward_integrator is not None
    assert trl_manager.reward_integrator is not None
    
    # Test that OpenRLHF has additional features
    assert hasattr(openrlhf_config, 'advantage_estimator')
    assert hasattr(openrlhf_config, 'normalize_reward')
    assert hasattr(openrlhf_config, 'packing_samples')
    assert hasattr(openrlhf_config, 'vllm_num_engines')
    assert hasattr(openrlhf_config, 'zero_stage')
    
    logger.info("✅ OpenRLHF vs TRL comparison test passed")


def run_all_tests():
    """Run all OpenRLHF integration tests."""
    logger.info("Starting OpenRLHF integration tests...")
    
    test_results = {
        "start_time": datetime.now().isoformat(),
        "tests_passed": 0,
        "tests_failed": 0,
        "test_details": []
    }
    
    tests = [
        ("Configuration Test", test_openrlhf_config),
        ("Integration Manager Test", test_openrlhf_integration_manager),
        ("Trainer Setup Test", test_openrlhf_trainer_setup),
        ("Training Step Test", test_openrlhf_training_step),
        ("Checkpoint Saving Test", test_openrlhf_checkpoint_saving),
        ("Reward Monitoring Test", test_openrlhf_reward_monitoring),
        ("Full Training Test", test_openrlhf_full_training),
        ("OpenRLHF vs TRL Comparison Test", test_openrlhf_vs_trl_comparison),
    ]
    
    for test_name, test_func in tests:
        try:
            logger.info(f"Running {test_name}...")
            result = test_func()
            test_results["tests_passed"] += 1
            test_results["test_details"].append({
                "test": test_name,
                "status": "PASSED",
                "result": str(result) if result else "No return value"
            })
            logger.info(f"✅ {test_name} PASSED")
        except Exception as e:
            test_results["tests_failed"] += 1
            test_results["test_details"].append({
                "test": test_name,
                "status": "FAILED",
                "error": str(e)
            })
            logger.error(f"❌ {test_name} FAILED: {e}")
    
    test_results["end_time"] = datetime.now().isoformat()
    
    # Save test results
    results_path = "./test_openrlhf_logs/test_results.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("OpenRLHF Integration Test Summary")
    logger.info("=" * 60)
    logger.info(f"Tests Passed: {test_results['tests_passed']}")
    logger.info(f"Tests Failed: {test_results['tests_failed']}")
    logger.info(f"Total Tests: {test_results['tests_passed'] + test_results['tests_failed']}")
    logger.info(f"Success Rate: {test_results['tests_passed'] / (test_results['tests_passed'] + test_results['tests_failed']) * 100:.1f}%")
    logger.info(f"Test results saved to: {results_path}")
    
    if test_results["tests_failed"] == 0:
        logger.info("🎉 All OpenRLHF integration tests passed!")
        return True
    else:
        logger.error(f"❌ {test_results['tests_failed']} tests failed")
        return False


def demo_openrlhf_integration():
    """Demonstrate OpenRLHF integration capabilities."""
    logger.info("Running OpenRLHF integration demonstration...")
    
    try:
        # Run the example
        integration_manager, results = create_openrlhf_integration_example()
        
        logger.info("OpenRLHF integration demonstration completed successfully!")
        logger.info(f"Training completed with {results['total_steps']} steps")
        logger.info(f"Checkpoints saved: {len(results['checkpoints_saved'])}")
        logger.info(f"Anomalies detected: {len(results['anomalies_detected'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"OpenRLHF integration demonstration failed: {e}")
        return False


if __name__ == "__main__":
    print("OpenRLHF Integration Test Suite")
    print("=" * 50)
    
    # Run tests
    tests_passed = run_all_tests()
    
    if tests_passed:
        print("\n" + "=" * 50)
        print("Running OpenRLHF Integration Demonstration...")
        demo_success = demo_openrlhf_integration()
        
        if demo_success:
            print("\n🎉 OpenRLHF Integration is working correctly!")
            print("The integration matches TRL functionality with additional OpenRLHF-specific features.")
        else:
            print("\n❌ OpenRLHF Integration demonstration failed.")
    else:
        print("\n❌ Some tests failed. Please check the logs for details.")
    
    print("\nOpenRLHF Integration Test Suite completed.")