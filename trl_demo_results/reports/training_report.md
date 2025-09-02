# TRL Integration Training Report

**Training Period**: 2025-09-02T02:39:43.736304 - 2025-09-02T02:39:43.736311
**Total Steps**: 30

## Summary

- Checkpoints Saved: 3
- Anomalies Detected: 2

## Final Training Metrics

### kl_div
- Mean: 0.12
- Std: 0.05
- Trend: stable

### policy_loss
- Mean: 0.45
- Std: 0.08
- Trend: decreasing

### value_loss
- Mean: 0.32
- Std: 0.06
- Trend: stable

### reward
- Mean: 2.1
- Std: 0.3
- Trend: increasing

### entropy
- Mean: 0.78
- Std: 0.05
- Trend: stable

## Detected Anomalies

- Step 5: High KL divergence
  - Details: {'kl_div': 0.8, 'threshold': 0.5}
- Step 15: Reward anomaly
  - Details: {'reward': 5.2, 'z_score': 3.5}

## Recommendations

1. Review anomaly logs for potential issues
2. Check checkpoint health scores
3. Monitor reward model reliability metrics
4. Consider adjusting hyperparameters based on trends
