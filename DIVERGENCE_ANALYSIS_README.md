# Divergence Analysis - Restored Functionality

## Overview

This document describes the restored divergence analysis functionality that was missing from the recent refactor. The divergence analysis module provides tools to detect when RLHF training runs start to diverge significantly.

## What Was Restored

The recent refactor had removed the bulk of the `first_divergence` function, causing it to exit early when runs shared fewer than `window` steps. This meant that for normal inputs with enough overlapping steps, the function would fall off the end and return `None` instead of a `DivergenceReport`, causing callers like the new CLI (`report.diverged`) and `generate_drift_card` to raise `AttributeError`.

## Restored Components

### 1. `DivergenceReport` Class

A comprehensive data class that contains:
- `diverged`: Boolean indicating if runs have diverged
- `divergence_step`: Step where divergence was first detected
- `divergence_z_scores`: Z-scores at divergence point
- `rolling_z_scores`: Rolling z-scores for all overlapping steps
- `overlapping_steps`: Number of overlapping steps analyzed
- `window_size`: Window size used for rolling calculations
- `z_score_threshold`: Threshold for considering runs diverged
- `metrics`: List of metrics analyzed
- `summary`: Summary statistics

### 2. `first_divergence()` Function

The core function that:
- Loads training logs from two runs
- Computes rolling z-scores for overlapping training steps
- Detects when runs start to diverge significantly
- Returns a populated `DivergenceReport` for all cases
- Handles edge cases gracefully (insufficient data, loading errors)

### 3. `generate_drift_card()` Function

Generates detailed drift analysis cards that can be saved to files for further analysis.

### 4. `analyze_multiple_runs()` Function

Analyzes divergence between multiple training runs, generating reports for each pair.

## Key Features

### Rolling Z-Score Analysis
- Uses configurable window size (default: 20 steps)
- Computes rolling mean and standard deviation
- Calculates z-scores for each metric
- Detects divergence when z-score differences exceed threshold

### Robust Error Handling
- Returns valid `DivergenceReport` objects even when analysis fails
- Handles insufficient overlapping steps gracefully
- Provides detailed error messages in the report summary

### Configurable Parameters
- `window_size`: Rolling window size for z-score calculation
- `z_score_threshold`: Threshold for divergence detection
- `min_overlapping_steps`: Minimum steps required for analysis
- `metrics`: Customizable list of metrics to analyze

## Usage Examples

### Basic Divergence Analysis

```python
from rlhf_core.divergence import first_divergence

# Analyze divergence between two runs
report = first_divergence(
    run1_logs="runs/run1/logs/train.jsonl",
    run2_logs="runs/run2/logs/train.jsonl",
    window_size=20,
    z_score_threshold=3.0
)

# Check if runs diverged
if report.diverged:
    print(f"Runs diverged at step {report.divergence_step}")
    print(f"Diverged metrics: {list(report.divergence_z_scores.keys())}")
else:
    print("Runs remained consistent")
```

### CLI Usage

```bash
# Analyze two specific runs
python tools/analyze_divergence.py runs/run1/logs/train.jsonl runs/run2/logs/train.jsonl

# Analyze with custom parameters
python tools/analyze_divergence.py runs/run1/logs/train.jsonl runs/run2/logs/train.jsonl \
  --window-size 30 --z-score-threshold 2.5 --metrics loss,reward_mean,kl

# Analyze all runs in a directory
python tools/analyze_divergence.py runs/*/logs/train.jsonl --output-dir drift_analysis
```

### Generate Drift Cards

```python
from rlhf_core.divergence import generate_drift_card

# Generate a drift analysis card
card_path = generate_drift_card(report, output_dir="drift_analysis")
print(f"Drift card saved to: {card_path}")
```

## File Structure

```
rlhf_core/
├── divergence.py          # Main divergence analysis module
└── __init__.py           # Updated to export divergence functions

tools/
├── analyze_divergence.py # CLI tool for divergence analysis
└── ...

test_divergence.py        # Test script for verification
```

## Dependencies

The divergence analysis module requires:
- `numpy`: For numerical computations
- `pandas`: For data manipulation and rolling statistics
- `json`: For loading training logs
- `pathlib`: For file path handling

These are already included in the project's `requirements.txt` and `pyproject.toml`.

## Testing

Run the test script to verify functionality:

```bash
python3 test_divergence.py
```

The test creates synthetic training data where one run diverges after step 50, allowing verification that the divergence detection works correctly.

## Integration Points

The restored divergence analysis integrates with:

1. **CLI Tools**: `tools/analyze_divergence.py` provides command-line access
2. **Monitoring Systems**: Can be used in real-time training monitoring
3. **Reporting**: Generates drift cards for documentation and analysis
4. **Automation**: Can be integrated into CI/CD pipelines for training stability

## Configuration

Default parameters can be customized:

- **Window Size**: 20 steps (adjust based on training dynamics)
- **Z-Score Threshold**: 3.0 (adjust for sensitivity)
- **Min Overlapping Steps**: 10 (adjust based on expected training length)
- **Metrics**: `['loss', 'reward_mean', 'kl', 'entropy', 'grad_norm']`

## Troubleshooting

### Common Issues

1. **Insufficient Overlapping Steps**: Ensure runs have enough common training steps
2. **Window Size Too Large**: Reduce window size for shorter training runs
3. **Z-Score Threshold Too High**: Lower threshold for more sensitive detection
4. **Missing Metrics**: Verify that log files contain the expected metric columns

### Error Handling

The module provides detailed error messages in the report summary:
- File loading failures
- Insufficient data for analysis
- Missing required columns
- Analysis completion status

## Future Enhancements

Potential improvements for the divergence analysis:

1. **Real-time Monitoring**: Integration with live training loops
2. **Advanced Statistics**: Additional divergence metrics and visualizations
3. **Machine Learning**: ML-based divergence detection
4. **Alerting**: Automated notifications when divergence is detected
5. **Historical Analysis**: Trend analysis across multiple training sessions

## Conclusion

The restored divergence analysis functionality provides a robust foundation for detecting training instability in RLHF systems. It addresses the critical issue where the previous implementation would fail silently, ensuring that training stability can be properly monitored and analyzed.