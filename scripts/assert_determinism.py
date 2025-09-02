#!/usr/bin/env python3
"""
Assert determinism between two RLHF training runs.

Compares metrics.jsonl files from two runs and ensures they are identical within tolerance.
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys


def load_metrics(run_path: str) -> pd.DataFrame:
    """Load metrics from metrics.jsonl file."""
    metrics_file = Path(run_path) / "metrics.jsonl"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    # Read JSONL file
    data = []
    with open(metrics_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    if not data:
        raise ValueError(f"No data found in metrics file: {metrics_file}")
    
    df = pd.DataFrame(data)
    
    # Convert numeric columns
    numeric_cols = ['loss', 'reward_mean', 'reward_var', 'kl', 'entropy', 
                   'clip_frac', 'grad_norm', 'lr', 'time_ms']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def align_metrics(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align two metrics DataFrames by step and phase."""
    # Ensure both have step column
    if 'step' not in df_a.columns or 'step' not in df_b.columns:
        raise ValueError("Both DataFrames must have 'step' column")
    
    # Sort by step
    df_a = df_a.sort_values('step').reset_index(drop=True)
    df_b = df_b.sort_values('step').reset_index(drop=True)
    
    # Find common steps
    common_steps = set(df_a['step']) & set(df_b['step'])
    if not common_steps:
        raise ValueError("No common steps found between runs")
    
    # Filter to common steps
    df_a_aligned = df_a[df_a['step'].isin(common_steps)].sort_values('step').reset_index(drop=True)
    df_b_aligned = df_b[df_b['step'].isin(common_steps)].sort_values('step').reset_index(drop=True)
    
    return df_a_aligned, df_b_aligned


def compare_metrics(df_a: pd.DataFrame, df_b: pd.DataFrame, tolerance: float = 1e-6) -> Dict[str, Any]:
    """Compare metrics between two aligned DataFrames."""
    if len(df_a) != len(df_b):
        raise ValueError(f"DataFrames have different lengths: {len(df_a)} vs {len(df_b)}")
    
    # Columns to compare (exclude metadata columns and timing)
    exclude_cols = {'step', 'run_id', 'seed', 'phase', 'time_ms'}
    numeric_cols = [col for col in df_a.columns if col not in exclude_cols]
    
    differences = []
    max_diff = 0.0
    max_diff_col = None
    max_diff_step = None
    
    for col in numeric_cols:
        if col not in df_b.columns:
            differences.append(f"Column '{col}' missing in run_b")
            continue
        
        # Get values for comparison
        val_a = df_a[col]
        val_b = df_b[col]
        
        # Handle NaN values
        both_nan = val_a.isna() & val_b.isna()
        either_nan = val_a.isna() | val_b.isna()
        
        # Compare non-NaN values
        valid_mask = ~either_nan
        if valid_mask.any():
            diff = np.abs(val_a[valid_mask] - val_b[valid_mask])
            max_col_diff = diff.max()
            
            if max_col_diff > max_diff:
                max_diff = max_col_diff
                max_diff_col = col
                max_diff_step = df_a.loc[valid_mask & (diff == max_col_diff), 'step'].iloc[0]
            
            # Check tolerance
            if max_col_diff > tolerance:
                differences.append(f"Column '{col}' exceeds tolerance: max diff = {max_col_diff:.2e} > {tolerance}")
        
        # Check for NaN mismatches
        nan_mismatch = either_nan & ~both_nan
        if nan_mismatch.any():
            differences.append(f"Column '{col}' has NaN mismatches at steps: {df_a.loc[nan_mismatch, 'step'].tolist()}")
    
    return {
        'differences': differences,
        'max_diff': max_diff,
        'max_diff_col': max_diff_col,
        'max_diff_step': max_diff_step,
        'total_steps': len(df_a),
        'tolerance': tolerance
    }


def print_comparison_summary(comparison: Dict[str, Any], run_a: str, run_b: str):
    """Print a summary of the comparison results."""
    print(f"\n=== Determinism Comparison ===")
    print(f"Run A: {run_a}")
    print(f"Run B: {run_b}")
    print(f"Total steps compared: {comparison['total_steps']}")
    print(f"Tolerance: {comparison['tolerance']}")
    
    if comparison['max_diff_col']:
        print(f"Maximum difference: {comparison['max_diff']:.2e}")
        print(f"  Column: {comparison['max_diff_col']}")
        print(f"  Step: {comparison['max_diff_step']}")
    
    if comparison['differences']:
        print(f"\n❌ FAILED: Found {len(comparison['differences'])} differences:")
        for diff in comparison['differences']:
            print(f"  - {diff}")
        return False
    else:
        print(f"\n✅ PASSED: All metrics within tolerance")
        return True


def main():
    parser = argparse.ArgumentParser(description='Assert determinism between two RLHF training runs')
    parser.add_argument('--run_a', type=str, required=True, help='Path to first run directory')
    parser.add_argument('--run_b', type=str, required=True, help='Path to second run directory')
    parser.add_argument('--tolerance', type=float, default=1e-6, 
                       help='Tolerance for numeric differences (default: 1e-6)')
    
    args = parser.parse_args()
    
    # Validate run directories (resolve symlinks)
    run_a_path = Path(args.run_a).resolve()
    run_b_path = Path(args.run_b).resolve()
    
    print(f"Run A path: {args.run_a} -> {run_a_path}")
    print(f"Run B path: {args.run_b} -> {run_b_path}")
    
    if not run_a_path.exists():
        print(f"Error: Run directory A does not exist: {args.run_a} (resolved to {run_a_path})")
        sys.exit(1)
    
    if not run_b_path.exists():
        print(f"Error: Run directory B does not exist: {args.run_b} (resolved to {run_b_path})")
        sys.exit(1)
    
    try:
        # Load metrics from both runs
        print(f"Loading metrics from run A: {run_a_path}")
        df_a = load_metrics(str(run_a_path))
        
        print(f"Loading metrics from run B: {run_b_path}")
        df_b = load_metrics(str(run_b_path))
        
        # Align metrics by step
        print("Aligning metrics by step...")
        df_a_aligned, df_b_aligned = align_metrics(df_a, df_b)
        
        # Compare metrics
        print("Comparing metrics...")
        comparison = compare_metrics(df_a_aligned, df_b_aligned, args.tolerance)
        
        # Print summary and exit with appropriate code
        success = print_comparison_summary(comparison, args.run_a, args.run_b)
        
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during comparison: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
