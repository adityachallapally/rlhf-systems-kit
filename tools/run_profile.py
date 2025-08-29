#!/usr/bin/env python3
"""
RLHF Profiling Runner

Runs a short RLHF training job with profiling enabled to generate
timing breakdowns, memory analysis, and torch profiler traces.
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rlhf_core.policy import PolicyModel
from rlhf_core.reward import ToyRewardModel
from rlhf_core.ppo import PPOTrainer
from profiler.hooks import get_profiler_registry
from profiler.trace import create_torch_profiler
from profiler.report import create_profiler_report


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_sample_prompts() -> list:
    """Create sample prompts for training."""
    prompts = [
        "The weather today is",
        "I really enjoyed",
        "This movie was",
        "The food tasted",
        "I feel very",
        "The book was",
        "My day was",
        "The music sounds",
        "I think that",
        "The game was"
    ]
    return prompts


def main():
    parser = argparse.ArgumentParser(description='RLHF Profiling Runner')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--steps', type=int, default=1, help='Number of training steps to profile')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--seq_len', type=int, default=10, help='Maximum new tokens to generate')
    parser.add_argument('--output_dir', type=str, default='profiles', help='Output directory for profiles')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--kl_coef', type=float, default=0.1, help='KL penalty coefficient')
    
    args = parser.parse_args()
    
    print("="*60)
    print("RLHF Profiling Runner")
    print("="*60)
    print(f"Steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Output directory: {args.output_dir}")
    
    # Auto-detect device
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Device: {device}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    trace_dir = os.path.join(args.output_dir, "trace")
    os.makedirs(trace_dir, exist_ok=True)
    
    # Clear any existing profiling results
    registry = get_profiler_registry()
    registry.clear()
    
    # Create models
    print("\nInitializing models...")
    policy_model = PolicyModel(device=device)
    reference_model = PolicyModel(device=device)
    reference_model.set_train_mode(False)
    reward_model = ToyRewardModel(device=device)
    print("Models initialized successfully")
    
    # Create PPO trainer
    trainer = PPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        reward_model=reward_model,
        device=device,
        learning_rate=args.learning_rate,
        kl_coef=args.kl_coef
    )
    
    # Create sample prompts
    prompts = create_sample_prompts()
    print(f"Created {len(prompts)} sample prompts")
    
    # Start profiling
    print("\nStarting profiling...")
    start_time = time.time()
    
    try:
        # Create torch profiler
        torch_profiler = create_torch_profiler(trace_dir)
        
        with torch_profiler:
            # Start torch profiler
            torch_profiler.start_profiling()
            
            # Run training steps
            for step in range(args.steps):
                print(f"Running training step {step + 1}/{args.steps}...")
                
                # Sample random subset of prompts
                if len(prompts) > args.batch_size:
                    import numpy as np
                    batch_prompts = np.random.choice(prompts, args.batch_size, replace=False).tolist()
                else:
                    batch_prompts = prompts
                
                # Run training step
                step_metrics = trainer.train_step(
                    batch_prompts,
                    max_new_tokens=args.seq_len,
                    batch_size=args.batch_size
                )
                
                # Step the torch profiler
                torch_profiler.step()
                
                print(f"  Step {step + 1} completed: Loss={step_metrics['total_loss']:.4f}")
            
            # Stop torch profiler
            torch_profiler.stop_profiling()
        
        # Get profiling results
        profiling_results = trainer.get_profiling_results()
        print(f"\nCollected {len(profiling_results)} profiling results")
        
        # Generate reports
        print("\nGenerating profiling reports...")
        report_generator = create_profiler_report(args.output_dir)
        
        report_files = report_generator.generate_full_report(
            profiling_results,
            summary_filename="summary.csv",
            figure_filename="stage_breakdown.png"
        )
        
        # Print file locations
        print("\nProfiling artifacts generated:")
        for file_type, file_path in report_files.items():
            if file_path:
                print(f"  {file_type}: {file_path}")
        
        # Check for required files
        required_files = [
            os.path.join(args.output_dir, "summary", "summary.csv"),
            os.path.join(args.output_dir, "trace", "ops.csv"),
            os.path.join(args.output_dir, "trace", "trace.json")
        ]
        
        print("\nRequired files check:")
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"  ✓ {file_path}")
            else:
                print(f"  ✗ {file_path} (missing)")
        
        # Check for optional figure
        figure_path = os.path.join(args.output_dir, "figures", "stage_breakdown.png")
        if os.path.exists(figure_path):
            print(f"  ✓ {figure_path} (optional)")
        else:
            print(f"  - {figure_path} (optional, not generated)")
        
        total_time = time.time() - start_time
        print(f"\nProfiling completed in {total_time:.2f} seconds")
        
        # Verify acceptance criteria
        print("\nAcceptance criteria check:")
        
        # Check if summary CSV has required data
        summary_path = os.path.join(args.output_dir, "summary", "summary.csv")
        if os.path.exists(summary_path):
            import csv
            with open(summary_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if len(rows) >= 6:  # Should have at least 6 stages
                    print(f"  ✓ Summary CSV has {len(rows)} stage rows")
                else:
                    print(f"  ✗ Summary CSV has only {len(rows)} stage rows (expected 6+)")
        else:
            print("  ✗ Summary CSV not found")
        
        # Check if trace file exists
        trace_path = os.path.join(args.output_dir, "trace", "trace.json")
        if os.path.exists(trace_path):
            print("  ✓ Trace file exists")
        else:
            print("  ✗ Trace file not found")
        
        # Check if ops CSV exists
        ops_path = os.path.join(args.output_dir, "trace", "ops.csv")
        if os.path.exists(ops_path):
            print("  ✓ Operations CSV exists")
        else:
            print("  ✗ Operations CSV not found")
        
        print("\nProfiling run completed successfully!")
        
    except Exception as e:
        print(f"\nError during profiling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()