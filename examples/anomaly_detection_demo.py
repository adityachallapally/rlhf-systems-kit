#!/usr/bin/env python3
"""
Demo script showing how to use the fixed AnomalyDetectionHook and StepProfiler.

This demonstrates:
1. How to register anomaly detection hooks
2. How to call end_step with proper training context
3. How the learning rate zero division issue is fixed
"""

import torch
import torch.nn as nn
import torch.optim as optim
from profiler.anomaly_detection import (
    AnomalyDetectionHook, 
    register_anomaly_hook, 
    end_step
)


def anomaly_callback(anomaly_type: str, anomaly_data: dict):
    """Callback function to handle detected anomalies."""
    print(f"🚨 ANOMALY DETECTED: {anomaly_type}")
    print(f"   Data: {anomaly_data}")
    print()


def main():
    print("="*60)
    print("Anomaly Detection Hook Demo")
    print("="*60)
    
    # Create a simple model and optimizer
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Create and register anomaly detection hook
    hook = AnomalyDetectionHook(
        lr_change_threshold=0.1,  # 10% change threshold
        lr_history_size=5,
        grad_norm_threshold=5.0,
        epsilon=1e-8
    )
    
    # Register callback for anomalies
    hook.register_anomaly_callback(anomaly_callback)
    
    # Register hook with global profiler
    register_anomaly_hook(hook)
    
    print("✅ Anomaly detection hook registered")
    print()
    
    # Simulate training steps
    print("Simulating training steps...")
    
    for step in range(10):
        # Simulate forward pass
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        loss = nn.MSELoss()(model(x), y)
        
        # Simulate backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Simulate learning rate scheduler that eventually reaches zero
        if step == 5:
            # This would previously cause ZeroDivisionError
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0
            print(f"Step {step}: Learning rate set to 0.0 (this used to crash)")
        elif step == 7:
            # Restore learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
            print(f"Step {step}: Learning rate restored to 0.001")
        
        # Call end_step with proper training context
        # This is the key fix - passing model, optimizer, and loss
        end_step(
            step=step,
            step_duration=0.1,  # Simulated step duration
            model=model,
            optimizer=optimizer,
            loss=loss
        )
        
        print(f"Step {step}: Loss={loss.item():.4f}, LR={optimizer.param_groups[0]['lr']}")
    
    print()
    print("✅ Demo completed successfully!")
    print("   - No ZeroDivisionError when LR reaches 0")
    print("   - Anomaly detection works with proper training context")
    print("   - Hooks receive model, optimizer, and loss as expected")


if __name__ == "__main__":
    main()