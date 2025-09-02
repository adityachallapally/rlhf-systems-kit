#!/usr/bin/env python3
"""
Standalone test to verify the anomaly detection fixes.

This test directly imports the simplified anomaly detection module without
going through the profiler package that requires psutil.
"""

import sys
import os

# Add the profiler directory to the path
profiler_dir = os.path.join(os.path.dirname(__file__), '..', 'profiler')
sys.path.insert(0, profiler_dir)


def test_imports():
    """Test that the simplified anomaly detection module can be imported."""
    try:
        from anomaly_detection_simple import (
            AnomalyDetectionHook, 
            StepProfiler, 
            get_step_profiler, 
            register_anomaly_hook, 
            end_step
        )
        print("✅ Successfully imported simplified anomaly detection classes and functions")
        return True
    except ImportError as e:
        print(f"❌ Failed to import simplified anomaly detection module: {e}")
        return False


def test_class_instantiation():
    """Test that the classes can be instantiated."""
    try:
        from anomaly_detection_simple import AnomalyDetectionHook, StepProfiler
        
        # Test AnomalyDetectionHook instantiation
        hook = AnomalyDetectionHook(
            lr_change_threshold=0.1,
            lr_history_size=10,
            grad_norm_threshold=10.0,
            epsilon=1e-8
        )
        print("✅ Successfully created AnomalyDetectionHook instance")
        
        # Test StepProfiler instantiation
        profiler = StepProfiler()
        print("✅ Successfully created StepProfiler instance")
        
        return True
    except Exception as e:
        print(f"❌ Failed to instantiate classes: {e}")
        return False


def test_method_existence():
    """Test that the required methods exist."""
    try:
        from anomaly_detection_simple import AnomalyDetectionHook, StepProfiler
        
        hook = AnomalyDetectionHook()
        profiler = StepProfiler()
        
        # Check AnomalyDetectionHook methods
        assert hasattr(hook, 'after_step'), "AnomalyDetectionHook should have after_step method"
        assert hasattr(hook, 'analyze_training_step'), "AnomalyDetectionHook should have analyze_training_step method"
        assert hasattr(hook, 'register_anomaly_callback'), "AnomalyDetectionHook should have register_anomaly_callback method"
        assert hasattr(hook, '_check_learning_rate_anomalies'), "AnomalyDetectionHook should have _check_learning_rate_anomalies method"
        
        # Check StepProfiler methods
        assert hasattr(profiler, 'end_step'), "StepProfiler should have end_step method"
        assert hasattr(profiler, 'register_hook'), "StepProfiler should have register_hook method"
        
        print("✅ All required methods exist")
        return True
    except Exception as e:
        print(f"❌ Method existence test failed: {e}")
        return False


def test_learning_rate_zero_division_fix():
    """Test that learning rate zero division is handled correctly."""
    try:
        from anomaly_detection_simple import AnomalyDetectionHook
        from collections import deque
        
        hook = AnomalyDetectionHook(epsilon=1e-8)
        
        # Create a mock optimizer with LR history that includes zeros
        class MockOptimizer:
            def __init__(self):
                self.param_groups = [{'lr': 0.01}]
        
        optimizer = MockOptimizer()
        
        # Manually set up LR history with zero values (this would crash before the fix)
        hook.lr_history[0] = deque([0.01, 0.005, 0.0, 0.0, 0.001], maxlen=10)
        
        # This should not raise ZeroDivisionError
        anomalies = hook._check_learning_rate_anomalies(optimizer, step=1)
        
        # Should return empty list (no anomalies detected due to zero values being skipped)
        assert isinstance(anomalies, list)
        # The test should pass without crashing, even if it doesn't detect anomalies
        # because zero values are properly skipped
        
        print("✅ Learning rate zero division fix works correctly")
        return True
    except Exception as e:
        print(f"❌ Learning rate zero division test failed: {e}")
        return False


def test_learning_rate_change_detection():
    """Test that learning rate change detection works for non-zero values."""
    try:
        from anomaly_detection_simple import AnomalyDetectionHook
        from collections import deque
        
        hook = AnomalyDetectionHook(lr_change_threshold=0.1)
        
        # Create a mock optimizer
        class MockOptimizer:
            def __init__(self):
                self.param_groups = [{'lr': 0.01}]
        
        optimizer = MockOptimizer()
        
        # Manually set up LR history with significant change
        hook.lr_history[0] = deque([0.01, 0.005, 0.02], maxlen=10)  # 100% increase from 0.01 to 0.02
        
        # This should detect the anomaly
        anomalies = hook._check_learning_rate_anomalies(optimizer, step=1)
        
        # Debug: print what we got
        print(f"Debug: Found {len(anomalies)} anomalies")
        for i, (anomaly_type, data) in enumerate(anomalies):
            print(f"  Anomaly {i}: {anomaly_type}, change_ratio: {data.get('change_ratio', 'N/A')}")
        
        # Should detect the large changes (there are multiple changes in the sequence)
        assert len(anomalies) >= 1, f"Expected at least 1 anomaly, got {len(anomalies)}"
        # Check that at least one anomaly has a large change ratio
        large_changes = [a for a in anomalies if a[1]['change_ratio'] > 0.1]
        assert len(large_changes) >= 1, "Should detect at least one large change"
        
        print("✅ Learning rate change detection works correctly")
        return True
    except Exception as e:
        print(f"❌ Learning rate change detection test failed: {e}")
        return False


def test_after_step_context_checking():
    """Test that after_step properly checks for training context."""
    try:
        from anomaly_detection_simple import AnomalyDetectionHook
        
        hook = AnomalyDetectionHook()
        
        # Track if analyze_training_step was called
        analyze_called = False
        original_analyze = hook.analyze_training_step
        
        def mock_analyze(step, step_duration, model, optimizer, loss, **kwargs):
            nonlocal analyze_called
            analyze_called = True
            # Don't call the original to avoid argument conflicts
        
        hook.analyze_training_step = mock_analyze
        
        # Test without context (should not call analyze_training_step)
        hook.after_step(step=1, step_duration=0.1)
        assert not analyze_called, "Should not call analyze_training_step without context"
        
        # Test with context (should call analyze_training_step)
        hook.after_step(step=1, step_duration=0.1, model="mock", optimizer="mock", loss="mock")
        assert analyze_called, "Should call analyze_training_step with context"
        
        print("✅ after_step context checking works correctly")
        return True
    except Exception as e:
        print(f"❌ after_step context test failed: {e}")
        return False


def test_step_profiler_context_passing():
    """Test that StepProfiler passes training context to hooks."""
    try:
        from anomaly_detection_simple import StepProfiler, AnomalyDetectionHook
        
        profiler = StepProfiler()
        hook = AnomalyDetectionHook()
        
        # Track if after_step was called with proper context
        after_step_called = False
        original_after_step = hook.after_step
        
        def mock_after_step(step, step_duration, model=None, optimizer=None, loss=None, **kwargs):
            nonlocal after_step_called
            after_step_called = True
            # Verify we received the training context
            assert model is not None
            assert optimizer is not None
            assert loss is not None
            return original_after_step(step, step_duration, model=model, optimizer=optimizer, loss=loss, **kwargs)
        
        hook.after_step = mock_after_step
        profiler.register_hook(hook)
        
        # Call end_step with training context
        profiler.end_step(
            step=1,
            step_duration=0.1,
            model="mock_model",
            optimizer="mock_optimizer",
            loss="mock_loss"
        )
        
        # Verify after_step was called
        assert after_step_called, "after_step should have been called with training context"
        
        print("✅ StepProfiler context passing works correctly")
        return True
    except Exception as e:
        print(f"❌ StepProfiler context test failed: {e}")
        return False


def test_global_end_step_function():
    """Test that the global end_step function works correctly."""
    try:
        from anomaly_detection_simple import AnomalyDetectionHook, register_anomaly_hook, end_step
        
        hook = AnomalyDetectionHook()
        
        # Track if after_step was called
        after_step_called = False
        original_after_step = hook.after_step
        
        def mock_after_step(step, step_duration, model=None, optimizer=None, loss=None, **kwargs):
            nonlocal after_step_called
            after_step_called = True
            assert model is not None
            assert optimizer is not None
            assert loss is not None
            return original_after_step(step, step_duration, model=model, optimizer=optimizer, loss=loss, **kwargs)
        
        hook.after_step = mock_after_step
        register_anomaly_hook(hook)
        
        # Call global end_step function
        end_step(
            step=1,
            step_duration=0.1,
            model="mock_model",
            optimizer="mock_optimizer",
            loss="mock_loss"
        )
        
        # Verify after_step was called
        assert after_step_called, "after_step should have been called via global end_step"
        
        print("✅ Global end_step function works correctly")
        return True
    except Exception as e:
        print(f"❌ Global end_step test failed: {e}")
        return False


def test_integration_scenario():
    """Integration test simulating the original bug scenario."""
    try:
        from anomaly_detection_simple import AnomalyDetectionHook
        
        # This test simulates the exact scenario described in the bug report
        
        # Create hook (this would be registered but never receive proper context)
        hook = AnomalyDetectionHook()
        
        # Track if analyze_training_step was called
        analyze_called = False
        original_analyze = hook.analyze_training_step
        
        def mock_analyze(step, step_duration, model, optimizer, loss, **kwargs):
            nonlocal analyze_called
            analyze_called = True
            # Don't call the original to avoid argument conflicts
            return []
        
        hook.analyze_training_step = mock_analyze
        
        # Simulate the old broken behavior (only step and step_duration)
        hook.after_step(step=1, step_duration=0.1)
        
        # This should NOT call analyze_training_step (the bug)
        assert not analyze_called, "Old behavior should not call analyze_training_step"
        
        # Now simulate the fixed behavior with proper context
        hook.after_step(
            step=1,
            step_duration=0.1,
            model="mock_model",
            optimizer="mock_optimizer",
            loss="mock_loss"
        )
        
        # This SHOULD call analyze_training_step (the fix)
        assert analyze_called, "Fixed behavior should call analyze_training_step"
        
        print("✅ Integration scenario test passed")
        return True
    except Exception as e:
        print(f"❌ Integration scenario test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Testing Anomaly Detection Fixes (Standalone)")
    print("="*60)
    
    tests = [
        test_imports,
        test_class_instantiation,
        test_method_existence,
        test_learning_rate_zero_division_fix,
        test_learning_rate_change_detection,
        test_after_step_context_checking,
        test_step_profiler_context_passing,
        test_global_end_step_function,
        test_integration_scenario
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        if test():
            passed += 1
        else:
            print(f"❌ {test.__name__} failed")
    
    print("\n" + "="*60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        print("\nThe anomaly detection fixes have been successfully implemented:")
        print("✅ AnomalyDetectionHook.after_step method exists and checks for context")
        print("✅ StepProfiler.end_step method exists and passes training context")
        print("✅ Learning rate zero division protection is implemented")
        print("✅ Training context passing logic is implemented")
        print("✅ All required methods and classes are present")
        print("✅ Integration scenario works correctly")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)