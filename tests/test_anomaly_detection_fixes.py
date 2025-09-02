#!/usr/bin/env python3
"""
Test script to verify the anomaly detection fixes.

Tests:
1. AnomalyDetectionHook.after_step receives proper training context
2. Learning rate zero division is handled correctly
3. StepProfiler.end_step passes training context to hooks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytest
from profiler.anomaly_detection import (
    AnomalyDetectionHook, 
    StepProfiler, 
    register_anomaly_hook, 
    end_step
)


class TestAnomalyDetectionFixes:
    """Test the fixes for anomaly detection issues."""
    
    def test_after_step_receives_training_context(self):
        """Test that after_step receives model, optimizer, and loss."""
        hook = AnomalyDetectionHook()
        
        # Track if analyze_training_step was called
        analyze_called = False
        original_analyze = hook.analyze_training_step
        
        def mock_analyze(*args, **kwargs):
            nonlocal analyze_called
            analyze_called = True
            # Verify we received the expected arguments
            assert 'model' in kwargs
            assert 'optimizer' in kwargs
            assert 'loss' in kwargs
            return original_analyze(*args, **kwargs)
        
        hook.analyze_training_step = mock_analyze
        
        # Create test objects
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss = torch.tensor(1.0)
        
        # Call after_step with training context
        hook.after_step(
            step=1,
            step_duration=0.1,
            model=model,
            optimizer=optimizer,
            loss=loss
        )
        
        # Verify analyze_training_step was called
        assert analyze_called, "analyze_training_step should have been called"
    
    def test_after_step_skips_without_context(self):
        """Test that after_step skips analysis when context is missing."""
        hook = AnomalyDetectionHook()
        
        # Track if analyze_training_step was called
        analyze_called = False
        original_analyze = hook.analyze_training_step
        
        def mock_analyze(*args, **kwargs):
            nonlocal analyze_called
            analyze_called = True
            return original_analyze(*args, **kwargs)
        
        hook.analyze_training_step = mock_analyze
        
        # Call after_step without training context (old behavior)
        hook.after_step(step=1, step_duration=0.1)
        
        # Verify analyze_training_step was NOT called
        assert not analyze_called, "analyze_training_step should not have been called without context"
    
    def test_learning_rate_zero_division_fix(self):
        """Test that learning rate change detection doesn't crash on zero LR."""
        hook = AnomalyDetectionHook(lr_change_threshold=0.1)
        
        # Create optimizer
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Simulate LR history with zero values
        hook.lr_history[0] = [0.01, 0.005, 0.0, 0.0, 0.001]
        
        # This should not raise ZeroDivisionError
        anomalies = hook._check_learning_rate_anomalies(optimizer, step=1)
        
        # Should return empty list (no anomalies detected due to zero values being skipped)
        assert isinstance(anomalies, list)
        assert len(anomalies) == 0  # No anomalies because zero values are skipped
    
    def test_learning_rate_change_detection_works(self):
        """Test that learning rate change detection works for non-zero values."""
        hook = AnomalyDetectionHook(lr_change_threshold=0.1)
        
        # Create optimizer
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Simulate LR history with significant change
        hook.lr_history[0] = [0.01, 0.005, 0.02]  # 100% increase from 0.01 to 0.02
        
        # This should detect the anomaly
        anomalies = hook._check_learning_rate_anomalies(optimizer, step=1)
        
        # Should detect the large change
        assert len(anomalies) == 1
        assert anomalies[0][0] == 'lr_change'
        assert anomalies[0][1]['change_ratio'] > 0.1
    
    def test_step_profiler_passes_context(self):
        """Test that StepProfiler.end_step passes training context to hooks."""
        profiler = StepProfiler()
        hook = AnomalyDetectionHook()
        
        # Track if after_step was called with proper context
        after_step_called = False
        original_after_step = hook.after_step
        
        def mock_after_step(step, step_duration, **kwargs):
            nonlocal after_step_called
            after_step_called = True
            # Verify we received the training context
            assert 'model' in kwargs
            assert 'optimizer' in kwargs
            assert 'loss' in kwargs
            return original_after_step(step, step_duration, **kwargs)
        
        hook.after_step = mock_after_step
        profiler.register_hook(hook)
        
        # Create test objects
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss = torch.tensor(1.0)
        
        # Call end_step with training context
        profiler.end_step(
            step=1,
            step_duration=0.1,
            model=model,
            optimizer=optimizer,
            loss=loss
        )
        
        # Verify after_step was called
        assert after_step_called, "after_step should have been called with training context"
    
    def test_global_end_step_function(self):
        """Test that the global end_step function works correctly."""
        hook = AnomalyDetectionHook()
        
        # Track if after_step was called
        after_step_called = False
        original_after_step = hook.after_step
        
        def mock_after_step(step, step_duration, **kwargs):
            nonlocal after_step_called
            after_step_called = True
            assert 'model' in kwargs
            assert 'optimizer' in kwargs
            assert 'loss' in kwargs
            return original_after_step(step, step_duration, **kwargs)
        
        hook.after_step = mock_after_step
        register_anomaly_hook(hook)
        
        # Create test objects
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss = torch.tensor(1.0)
        
        # Call global end_step function
        end_step(
            step=1,
            step_duration=0.1,
            model=model,
            optimizer=optimizer,
            loss=loss
        )
        
        # Verify after_step was called
        assert after_step_called, "after_step should have been called via global end_step"


def test_integration_scenario():
    """Integration test simulating the original bug scenario."""
    # This test simulates the exact scenario described in the bug report
    
    # Create hook (this would be registered but never receive proper context)
    hook = AnomalyDetectionHook()
    
    # Track if analyze_training_step was called
    analyze_called = False
    original_analyze = hook.analyze_training_step
    
    def mock_analyze(*args, **kwargs):
        nonlocal analyze_called
        analyze_called = True
        return original_analyze(*args, **kwargs)
    
    hook.analyze_training_step = mock_analyze
    
    # Simulate the old broken behavior (only step and step_duration)
    hook.after_step(step=1, step_duration=0.1)
    
    # This should NOT call analyze_training_step (the bug)
    assert not analyze_called, "Old behavior should not call analyze_training_step"
    
    # Now simulate the fixed behavior with proper context
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss = torch.tensor(1.0)
    
    hook.after_step(
        step=1,
        step_duration=0.1,
        model=model,
        optimizer=optimizer,
        loss=loss
    )
    
    # This SHOULD call analyze_training_step (the fix)
    assert analyze_called, "Fixed behavior should call analyze_training_step"


if __name__ == "__main__":
    # Run the tests
    test_instance = TestAnomalyDetectionFixes()
    
    print("Running anomaly detection fix tests...")
    
    try:
        test_instance.test_after_step_receives_training_context()
        print("✅ test_after_step_receives_training_context passed")
        
        test_instance.test_after_step_skips_without_context()
        print("✅ test_after_step_skips_without_context passed")
        
        test_instance.test_learning_rate_zero_division_fix()
        print("✅ test_learning_rate_zero_division_fix passed")
        
        test_instance.test_learning_rate_change_detection_works()
        print("✅ test_learning_rate_change_detection_works passed")
        
        test_instance.test_step_profiler_passes_context()
        print("✅ test_step_profiler_passes_context passed")
        
        test_instance.test_global_end_step_function()
        print("✅ test_global_end_step_function passed")
        
        test_integration_scenario()
        print("✅ test_integration_scenario passed")
        
        print("\n🎉 All tests passed! The fixes are working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()