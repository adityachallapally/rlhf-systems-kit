#!/usr/bin/env python3
"""
Test script to verify the RLHF implementation works correctly.
"""

import torch
import numpy as np
from rlhf_core.policy import PolicyModel
from rlhf_core.reward import ToyRewardModel
from rlhf_core.ppo import PPOTrainer


def test_policy_model():
    """Test the policy model."""
    print("Testing PolicyModel...")
    
    # Create model
    model = PolicyModel(device="cpu")
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 10))
    outputs = model(input_ids)
    assert 'logits' in outputs
    assert outputs['logits'].shape == (2, 10, 50257)  # GPT-2 vocab size
    
    # Test sampling
    prompt_ids = torch.randint(0, 1000, (1, 5))
    sequences, logprobs = model.sample(prompt_ids, max_new_tokens=3)
    assert sequences.shape[1] == 8  # 5 + 3
    assert logprobs.shape[1] == 3
    
    print("✓ PolicyModel tests passed")


def test_reward_model():
    """Test the reward model."""
    print("Testing ToyRewardModel...")
    
    # Create model
    model = ToyRewardModel(device="cpu")
    
    # Test sentiment reward
    input_ids = torch.randint(0, 1000, (2, 10))
    rewards = model.compute_reward(input_ids, reward_type="sentiment")
    assert rewards.shape == (2,)
    assert torch.all(rewards >= -5) and torch.all(rewards <= 5)
    
    # Test preference reward
    rewards = model.compute_reward(input_ids, reward_type="preference")
    assert rewards.shape == (2,)
    
    # Test length reward
    rewards = model.compute_reward(input_ids, reward_type="length")
    assert rewards.shape == (2,)
    
    print("✓ ToyRewardModel tests passed")


def test_ppo_trainer():
    """Test the PPO trainer."""
    print("Testing PPOTrainer...")
    
    # Create models
    policy_model = PolicyModel(device="cpu")
    reference_model = PolicyModel(device="cpu")
    reward_model = ToyRewardModel(device="cpu")
    
    # Create trainer
    trainer = PPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        reward_model=reward_model,
        device="cpu"
    )
    
    # Test single training step
    prompts = ["The weather today is", "I really enjoyed"]
    metrics = trainer.train_step(prompts, max_new_tokens=5, batch_size=2)
    
    # Check metrics
    required_keys = ['total_loss', 'policy_loss', 'kl_loss', 'reward_mean', 'kl_mean']
    for key in required_keys:
        assert key in metrics
        assert isinstance(metrics[key], (int, float))
    
    print("✓ PPOTrainer tests passed")


def test_reproducibility():
    """Test that the same seed produces identical results."""
    print("Testing reproducibility...")
    
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create model and sample
    model = PolicyModel(device="cpu")
    prompt_ids = torch.randint(0, 1000, (1, 5))
    sequences1, logprobs1 = model.sample(prompt_ids, max_new_tokens=3)
    
    # Reset seed and sample again
    torch.manual_seed(42)
    np.random.seed(42)
    
    model2 = PolicyModel(device="cpu")
    prompt_ids2 = torch.randint(0, 1000, (1, 5))
    sequences2, logprobs2 = model2.sample(prompt_ids2, max_new_tokens=3)
    
    # Check reproducibility
    assert torch.allclose(sequences1, sequences2)
    assert torch.allclose(logprobs1, logprobs2)
    
    print("✓ Reproducibility tests passed")


def main():
    """Run all tests."""
    print("Running RLHF implementation tests...\n")
    
    try:
        test_policy_model()
        test_reward_model()
        test_ppo_trainer()
        test_reproducibility()
        
        print("\n🎉 All tests passed! The RLHF implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()