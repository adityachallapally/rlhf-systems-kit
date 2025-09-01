#!/usr/bin/env python3
"""
CLI tool for analyzing divergence between RLHF training runs.

This tool uses the divergence analysis module to detect when training runs
start to diverge significantly.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rlhf_core.divergence import first_divergence, generate_drift_card, analyze_multiple_runs


def main():
    parser = argparse.ArgumentParser(
        description='Analyze divergence between RLHF training runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze divergence between two specific runs
  python tools/analyze_divergence.py runs/run1/logs/train.jsonl runs/run2/logs/train.jsonl
  
  # Analyze with custom parameters
  python tools/analyze_divergence.py runs/run1/logs/train.jsonl runs/run2/logs/train.jsonl \\
    --window-size 30 --z-score-threshold 2.5 --metrics loss,reward_mean,kl
  
  # Analyze all runs in a directory
  python tools/analyze_divergence.py runs/*/logs/train.jsonl --output-dir drift_analysis
        """
    )
    
    parser.add_argument(
        'log_files',
        nargs='+',
        help='Paths to training log files (JSONL format)'
    )
    
    parser.add_argument(
        '--metrics',
        type=str,
        default='loss,reward_mean,kl,entropy,grad_norm',
        help='Comma-separated list of metrics to analyze'
    )
    
    parser.add_argument(
        '--window-size',
        type=int,
        default=20,
        help='Size of rolling window for z-score calculation (default: 20)'
    )
    
    parser.add_argument(
        '--z-score-threshold',
        type=float,
        default=3.0,
        help='Z-score threshold for considering runs diverged (default: 3.0)'
    )
    
    parser.add_argument(
        '--min-overlapping-steps',
        type=int,
        default=10,
        help='Minimum number of overlapping steps required (default: 10)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='drift_analysis',
        help='Output directory for drift analysis cards (default: drift_analysis)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Parse metrics
    metrics = [m.strip() for m in args.metrics.split(',')]
    
    # Validate inputs
    if len(args.log_files) < 2:
        print("Error: Need at least 2 log files to analyze divergence")
        sys.exit(1)
    
    # Check if log files exist
    for log_file in args.log_files:
        if not Path(log_file).exists():
            print(f"Error: Log file not found: {log_file}")
            sys.exit(1)
    
    print("="*60)
    print("RLHF Training Divergence Analysis")
    print("="*60)
    print(f"Log files: {len(args.log_files)}")
    print(f"Metrics: {', '.join(metrics)}")
    print(f"Window size: {args.window_size}")
    print(f"Z-score threshold: {args.z_score_threshold}")
    print(f"Min overlapping steps: {args.min_overlapping_steps}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    if len(args.log_files) == 2:
        # Analyze two specific runs
        run1_logs = args.log_files[0]
        run2_logs = args.log_files[1]
        
        print(f"Analyzing divergence between:")
        print(f"  Run 1: {Path(run1_logs).parent.parent.name}")
        print(f"  Run 2: {Path(run2_logs).parent.parent.name}")
        print()
        
        # Perform divergence analysis
        report = first_divergence(
            run1_logs=run1_logs,
            run2_logs=run2_logs,
            metrics=metrics,
            window_size=args.window_size,
            z_score_threshold=args.z_score_threshold,
            min_overlapping_steps=args.min_overlapping_steps
        )
        
        # Display results
        print("Analysis Results:")
        print("-" * 30)
        print(f"Diverged: {'Yes' if report.diverged else 'No'}")
        print(f"Overlapping steps: {report.overlapping_steps}")
        
        if report.diverged:
            print(f"Divergence detected at step: {report.divergence_step}")
            if report.divergence_z_scores:
                print("Diverged metrics:")
                for metric, details in report.divergence_z_scores.items():
                    print(f"  {metric}: Run1 Z={details['run1_z_score']:.3f}, "
                          f"Run2 Z={details['run2_z_score']:.3f}, "
                          f"Diff={details['difference']:.3f}")
        else:
            if 'error' in report.summary:
                print(f"Analysis issue: {report.summary['error']}")
            else:
                print("Runs remained consistent throughout training")
        
        # Generate drift card
        try:
            card_path = generate_drift_card(report, args.output_dir)
            print(f"\nDrift analysis card saved to: {card_path}")
        except Exception as e:
            print(f"Warning: Failed to generate drift card: {e}")
        
        # Display summary
        if report.summary and 'analysis_complete' in report.summary:
            print("\nSummary:")
            print(f"  Total steps Run 1: {report.summary.get('total_steps_run1', 'N/A')}")
            print(f"  Total steps Run 2: {report.summary.get('total_steps_run2', 'N/A')}")
            print(f"  Analysis complete: {report.summary.get('analysis_complete', False)}")
    
    else:
        # Analyze multiple runs
        print(f"Analyzing divergence between {len(args.log_files)} runs...")
        print()
        
        try:
            reports = analyze_multiple_runs(
                args.log_files,
                metrics=metrics,
                window_size=args.window_size,
                z_score_threshold=args.z_score_threshold
            )
            
            # Display summary of all analyses
            print("Analysis Summary:")
            print("-" * 30)
            
            diverged_pairs = 0
            total_pairs = len(reports)
            
            for i, report in enumerate(reports):
                run1_name = Path(args.log_files[i // (len(args.log_files) - 1)]).parent.parent.name
                run2_name = Path(args.log_files[(i // (len(args.log_files) - 1)) + 1]).parent.parent.name
                
                if report.diverged:
                    diverged_pairs += 1
                    print(f"  {run1_name} vs {run2_name}: DIVERGED at step {report.divergence_step}")
                else:
                    print(f"  {run1_name} vs {run2_name}: Consistent")
                
                # Generate drift card for each pair
                try:
                    pair_output_dir = Path(args.output_dir) / f"{run1_name}_vs_{run2_name}"
                    card_path = generate_drift_card(report, str(pair_output_dir))
                    if args.verbose:
                        print(f"    Drift card: {card_path}")
                except Exception as e:
                    if args.verbose:
                        print(f"    Warning: Failed to generate drift card: {e}")
            
            print(f"\nOverall: {diverged_pairs}/{total_pairs} run pairs diverged")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            sys.exit(1)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()