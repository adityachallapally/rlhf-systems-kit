"""
Divergence analysis utilities for RLHF training.

This module provides functions to analyze divergence between training runs
and detect when runs start to diverge significantly.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DivergenceReport:
    """Report containing divergence analysis results."""
    
    # Whether the runs have diverged
    diverged: bool
    
    # Step where divergence was first detected
    divergence_step: Optional[int]
    
    # Z-scores at divergence point
    divergence_z_scores: Optional[Dict[str, float]]
    
    # Rolling z-scores for all overlapping steps
    rolling_z_scores: Optional[pd.DataFrame]
    
    # Number of overlapping steps analyzed
    overlapping_steps: int
    
    # Window size used for rolling calculations
    window_size: int
    
    # Threshold for considering runs diverged
    z_score_threshold: float
    
    # Metrics that were analyzed
    metrics: List[str]
    
    # Summary statistics
    summary: Dict[str, Union[float, str]]


def load_training_logs(log_file: str) -> pd.DataFrame:
    """Load training logs from JSONL file.
    
    Args:
        log_file: Path to training log file
        
    Returns:
        DataFrame with training metrics
    """
    if not Path(log_file).exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    data = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except (json.JSONDecodeError, pd.errors.EmptyDataError):
                continue
    
    if not data:
        raise ValueError("No valid log entries found")
    
    df = pd.DataFrame(data)
    
    # Ensure step column exists and is sorted
    if 'step' not in df.columns:
        raise ValueError("Log file must contain 'step' column")
    
    df = df.sort_values('step').reset_index(drop=True)
    return df


def compute_rolling_z_scores(df: pd.DataFrame, 
                            metrics: List[str], 
                            window_size: int = 20) -> pd.DataFrame:
    """Compute rolling z-scores for specified metrics.
    
    Args:
        df: DataFrame with training metrics
        metrics: List of metric names to analyze
        window_size: Size of rolling window for z-score calculation
        
    Returns:
        DataFrame with rolling z-scores
    """
    if window_size < 2:
        raise ValueError("Window size must be at least 2")
    
    if len(df) < window_size:
        raise ValueError(f"DataFrame must have at least {window_size} rows")
    
    result_df = df[['step']].copy()
    
    for metric in metrics:
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in data, skipping")
            continue
        
        # Compute rolling mean and std
        rolling_mean = df[metric].rolling(window=window_size, center=True).mean()
        rolling_std = df[metric].rolling(window=window_size, center=True).std()
        
        # Compute z-scores (avoid division by zero)
        z_scores = np.where(
            rolling_std > 0,
            (df[metric] - rolling_mean) / rolling_std,
            0.0
        )
        
        result_df[f'{metric}_z_score'] = z_scores
        result_df[f'{metric}_rolling_mean'] = rolling_mean
        result_df[f'{metric}_rolling_std'] = rolling_std
    
    return result_df


def first_divergence(run1_logs: str, 
                    run2_logs: str,
                    metrics: Optional[List[str]] = None,
                    window_size: int = 20,
                    z_score_threshold: float = 3.0,
                    min_overlapping_steps: int = 10) -> DivergenceReport:
    """Analyze divergence between two training runs.
    
    This function computes rolling z-scores for overlapping training steps
    and detects when the runs start to diverge significantly.
    
    Args:
        run1_logs: Path to first run's training logs
        run2_logs: Path to second run's training logs
        metrics: List of metrics to analyze (default: common training metrics)
        window_size: Size of rolling window for z-score calculation
        z_score_threshold: Z-score threshold for considering runs diverged
        min_overlapping_steps: Minimum number of overlapping steps required
        
    Returns:
        DivergenceReport with analysis results
    """
    
    # Default metrics to analyze if none specified
    if metrics is None:
        metrics = ['loss', 'reward_mean', 'kl', 'entropy', 'grad_norm']
    
    # Load training logs
    try:
        df1 = load_training_logs(run1_logs)
        df2 = load_training_logs(run2_logs)
    except Exception as e:
        # Return a report indicating loading failure
        return DivergenceReport(
            diverged=False,
            divergence_step=None,
            divergence_z_scores=None,
            rolling_z_scores=None,
            overlapping_steps=0,
            window_size=window_size,
            z_score_threshold=z_score_threshold,
            metrics=metrics,
            summary={'error': f'Failed to load logs: {str(e)}'}
        )
    
    # Find overlapping steps
    common_steps = set(df1['step']) & set(df2['step'])
    if len(common_steps) < min_overlapping_steps:
        # Return report for insufficient overlapping steps
        return DivergenceReport(
            diverged=False,
            divergence_step=None,
            divergence_z_scores=None,
            rolling_z_scores=None,
            overlapping_steps=len(common_steps),
            window_size=window_size,
            z_score_threshold=z_score_threshold,
            metrics=metrics,
            summary={
                'error': f'Insufficient overlapping steps: {len(common_steps)} < {min_overlapping_steps}',
                'overlapping_steps': len(common_steps),
                'required_steps': min_overlapping_steps
            }
        )
    
    # Filter to common steps and sort
    common_steps = sorted(list(common_steps))
    df1_common = df1[df1['step'].isin(common_steps)].sort_values('step').reset_index(drop=True)
    df2_common = df2[df2['step'].isin(common_steps)].sort_values('step').reset_index(drop=True)
    
    # Ensure we have enough data for rolling calculations
    if len(common_steps) < window_size:
        # Return report for insufficient data
        return DivergenceReport(
            diverged=False,
            divergence_step=None,
            divergence_z_scores=None,
            rolling_z_scores=None,
            overlapping_steps=len(common_steps),
            window_size=window_size,
            z_score_threshold=z_score_threshold,
            metrics=metrics,
            summary={
                'error': f'Insufficient data for window size {window_size}: {len(common_steps)} steps',
                'overlapping_steps': len(common_steps),
                'window_size': window_size
            }
        )
    
    # Compute rolling z-scores for each run
    z_scores_1 = compute_rolling_z_scores(df1_common, metrics, window_size)
    z_scores_2 = compute_rolling_z_scores(df2_common, metrics, window_size)
    
    # Find divergence point by comparing z-scores
    divergence_step = None
    divergence_z_scores = None
    
    for i, step in enumerate(common_steps):
        if i < window_size // 2:  # Skip early steps where rolling stats aren't fully populated
            continue
            
        # Check if any metric has diverged
        step_diverged = False
        step_z_scores = {}
        
        for metric in metrics:
            z_score_col = f'{metric}_z_score'
            if z_score_col in z_scores_1.columns and z_score_col in z_scores_2.columns:
                z1 = z_scores_1.loc[i, z_score_col]
                z2 = z_scores_2.loc[i, z_score_col]
                
                # Check if z-scores differ significantly
                if abs(z1 - z2) > z_score_threshold:
                    step_diverged = True
                    step_z_scores[metric] = {
                        'run1_z_score': z1,
                        'run2_z_score': z2,
                        'difference': abs(z1 - z2)
                    }
        
        if step_diverged:
            divergence_step = step
            divergence_z_scores = step_z_scores
            break
    
    # Create rolling z-scores DataFrame for analysis
    rolling_data = []
    for i, step in enumerate(common_steps):
        if i < window_size // 2:
            continue
            
        row_data = {'step': step}
        for metric in metrics:
            z_score_col = f'{metric}_z_score'
            if z_score_col in z_scores_1.columns and z_score_col in z_scores_2.columns:
                row_data[f'{metric}_run1_z'] = z_scores_1.loc[i, z_score_col]
                row_data[f'{metric}_run2_z'] = z_scores_2.loc[i, z_score_col]
                row_data[f'{metric}_z_diff'] = abs(
                    z_scores_1.loc[i, z_score_col] - z_scores_2.loc[i, z_score_col]
                )
        
        rolling_data.append(row_data)
    
    rolling_z_scores = pd.DataFrame(rolling_data) if rolling_data else None
    
    # Create summary statistics
    summary = {
        'total_steps_run1': len(df1),
        'total_steps_run2': len(df2),
        'overlapping_steps': len(common_steps),
        'window_size': window_size,
        'z_score_threshold': z_score_threshold,
        'analysis_complete': True
    }
    
    if divergence_step is not None:
        summary['divergence_detected'] = True
        summary['divergence_step'] = divergence_step
        summary['diverged_metrics'] = list(divergence_z_scores.keys())
    else:
        summary['divergence_detected'] = False
        summary['runs_consistent'] = True
    
    # Create and return the divergence report
    return DivergenceReport(
        diverged=divergence_step is not None,
        divergence_step=divergence_step,
        divergence_z_scores=divergence_z_scores,
        rolling_z_scores=rolling_z_scores,
        overlapping_steps=len(common_steps),
        window_size=window_size,
        z_score_threshold=z_score_threshold,
        metrics=metrics,
        summary=summary
    )


def generate_drift_card(report: DivergenceReport, output_dir: str = "drift_analysis") -> str:
    """Generate a drift analysis card from a DivergenceReport.
    
    Args:
        report: DivergenceReport to analyze
        output_dir: Directory to save the drift card
        
    Returns:
        Path to the generated drift card
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create drift card content
    card_lines = [
        "RLHF Training Drift Analysis Card",
        "=" * 50,
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Analysis Parameters:",
        f"  Window Size: {report.window_size}",
        f"  Z-Score Threshold: {report.z_score_threshold}",
        f"  Metrics Analyzed: {', '.join(report.metrics)}",
        f"  Overlapping Steps: {report.overlapping_steps}",
        "",
        "Results:",
        f"  Runs Diverged: {'Yes' if report.diverged else 'No'}",
    ]
    
    if report.diverged:
        card_lines.extend([
            f"  Divergence Detected at Step: {report.divergence_step}",
            f"  Diverged Metrics: {', '.join(report.divergence_z_scores.keys()) if report.divergence_z_scores else 'None'}",
            "",
            "Divergence Details:"
        ])
        
        if report.divergence_z_scores:
            for metric, details in report.divergence_z_scores.items():
                card_lines.extend([
                    f"  {metric}:",
                    f"    Run 1 Z-Score: {details['run1_z_score']:.3f}",
                    f"    Run 2 Z-Score: {details['run2_z_score']:.3f}",
                    f"    Difference: {details['difference']:.3f}"
                ])
    else:
        card_lines.extend([
            "  Runs remained consistent throughout training",
            "  No significant divergence detected"
        ])
    
    # Add summary statistics
    if report.summary:
        card_lines.extend([
            "",
            "Summary Statistics:",
            "-" * 20
        ])
        
        for key, value in report.summary.items():
            if key not in ['error', 'analysis_complete']:
                card_lines.append(f"  {key}: {value}")
    
    # Add error information if present
    if report.summary and 'error' in report.summary:
        card_lines.extend([
            "",
            "Errors/Warnings:",
            "-" * 20,
            f"  {report.summary['error']}"
        ])
    
    # Save drift card
    card_path = Path(output_dir) / "drift_analysis_card.txt"
    with open(card_path, 'w') as f:
        f.write('\n'.join(card_lines))
    
    print(f"Generated drift analysis card: {card_path}")
    return str(card_path)


def analyze_multiple_runs(run_logs: List[str],
                         metrics: Optional[List[str]] = None,
                         window_size: int = 20,
                         z_score_threshold: float = 3.0) -> List[DivergenceReport]:
    """Analyze divergence between multiple training runs.
    
    Args:
        run_logs: List of paths to training log files
        metrics: List of metrics to analyze
        window_size: Size of rolling window for z-score calculation
        z_score_threshold: Z-score threshold for considering runs diverged
        
    Returns:
        List of DivergenceReport objects for each pair of runs
    """
    if len(run_logs) < 2:
        raise ValueError("Need at least 2 runs to analyze divergence")
    
    reports = []
    
    # Analyze each pair of runs
    for i in range(len(run_logs)):
        for j in range(i + 1, len(run_logs)):
            run1 = run_logs[i]
            run2 = run_logs[j]
            
            print(f"Analyzing divergence between {Path(run1).name} and {Path(run2).name}")
            
            report = first_divergence(
                run1, run2, metrics, window_size, z_score_threshold
            )
            reports.append(report)
    
    return reports