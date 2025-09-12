"""
Robust Evaluation Suite for RLHF Training

This module provides a comprehensive evaluation suite that gracefully handles
missing data columns and provides clear error messages with fallback mechanisms.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path

# Optional imports with fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available, using basic data structures")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available, using basic math functions")
    
    # Basic numpy replacements
    class BasicNumpy:
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
        
        @staticmethod
        def std(data):
            if len(data) < 2:
                return 0
            mean_val = BasicNumpy.mean(data)
            variance = sum((x - mean_val) ** 2 for x in data) / (len(data) - 1)
            return variance ** 0.5
        
        @staticmethod
        def min(data):
            return min(data) if data else 0
        
        @staticmethod
        def max(data):
            return max(data) if data else 0
        
        @staticmethod
        def var(data):
            if len(data) < 2:
                return 0
            mean_val = BasicNumpy.mean(data)
            return sum((x - mean_val) ** 2 for x in data) / len(data)
        
        @staticmethod
        def percentile(data, p):
            if not data:
                return 0
            sorted_data = sorted(data)
            index = int((p / 100) * (len(sorted_data) - 1))
            return sorted_data[index]
    
    np = BasicNumpy()


@dataclass
class EvaluationConfig:
    """Configuration for evaluation suite."""
    
    # Column mappings for different metrics
    column_mappings: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        'throughput': {
            'events': 'events',
            'time': 'time',
            'tokens': 'tokens',
            'fallback_columns': ['steps', 'samples', 'batches']
        },
        'toxicity': {
            'output': 'output',
            'text': 'text',
            'response': 'response',
            'fallback_columns': ['generated_text', 'completion', 'generation']
        },
        'bias': {
            'output': 'output',
            'text': 'text',
            'response': 'response',
            'fallback_columns': ['generated_text', 'completion', 'generation']
        }
    })
    
    # Evaluation thresholds
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        'throughput_min': 0.0,
        'toxicity_max': 0.5,
        'bias_max': 0.3
    })
    
    # Fallback strategies
    fallback_strategies: Dict[str, str] = field(default_factory=lambda: {
        'throughput': 'estimate_from_steps',
        'toxicity': 'skip_evaluation',
        'bias': 'skip_evaluation'
    })
    
    # Error handling
    strict_mode: bool = False
    log_missing_columns: bool = True
    create_mock_data: bool = False


class EvaluationMetric(ABC):
    """Abstract base class for evaluation metrics."""
    
    def __init__(self, name: str, config: EvaluationConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"evaluation.{name}")
    
    @abstractmethod
    def evaluate(self, data: Union[Any, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the metric on the given data."""
        pass
    
    def _find_column(self, data: Union[Any, Dict[str, Any]], 
                    metric_name: str, column_type: str) -> Optional[str]:
        """Find the appropriate column for a metric."""
        if metric_name not in self.config.column_mappings:
            return None
        
        mapping = self.config.column_mappings[metric_name]
        primary_column = mapping.get(column_type)
        fallback_columns = mapping.get('fallback_columns', [])
        
        # Check primary column
        if primary_column and self._column_exists(data, primary_column):
            return primary_column
        
        # Check fallback columns
        for fallback_col in fallback_columns:
            if self._column_exists(data, fallback_col):
                self.logger.warning(f"Using fallback column '{fallback_col}' for {metric_name}.{column_type}")
                return fallback_col
        
        return None
    
    def _column_exists(self, data: Union[Any, Dict[str, Any]], column: str) -> bool:
        """Check if a column exists in the data."""
        if PANDAS_AVAILABLE and hasattr(data, 'columns'):
            return column in data.columns
        elif isinstance(data, dict):
            return column in data
        return False
    
    def _get_column_data(self, data: Union[Any, Dict[str, Any]], column: str) -> Any:
        """Get data from a column."""
        if PANDAS_AVAILABLE and hasattr(data, 'columns'):
            return data[column]
        elif isinstance(data, dict):
            return data[column]
        return None
    
    def _handle_missing_column(self, metric_name: str, column_type: str, 
                              required_columns: List[str]) -> Dict[str, Any]:
        """Handle missing column gracefully."""
        missing_columns = [col for col in required_columns 
                          if not self._find_column(self._get_data(), metric_name, col)]
        
        error_msg = f"Missing columns for {metric_name}: {missing_columns}"
        
        if self.config.strict_mode:
            raise ValueError(error_msg)
        
        if self.config.log_missing_columns:
            self.logger.warning(error_msg)
        
        # Apply fallback strategy
        fallback_strategy = self.config.fallback_strategies.get(metric_name, 'skip_evaluation')
        
        if fallback_strategy == 'skip_evaluation':
            return {
                'metric': metric_name,
                'status': 'skipped',
                'reason': f'Missing required columns: {missing_columns}',
                'value': None,
                'confidence': 0.0
            }
        elif fallback_strategy == 'estimate_from_steps':
            return self._estimate_from_available_data(metric_name, missing_columns)
        elif fallback_strategy == 'create_mock_data':
            return self._create_mock_evaluation(metric_name, missing_columns)
        else:
            return {
                'metric': metric_name,
                'status': 'error',
                'reason': f'Unknown fallback strategy: {fallback_strategy}',
                'value': None,
                'confidence': 0.0
            }
    
    def _estimate_from_available_data(self, metric_name: str, missing_columns: List[str]) -> Dict[str, Any]:
        """Estimate metric from available data."""
        # This is a placeholder - implement specific estimation logic
        return {
            'metric': metric_name,
            'status': 'estimated',
            'reason': f'Estimated from available data due to missing: {missing_columns}',
            'value': 0.0,
            'confidence': 0.3
        }
    
    def _create_mock_evaluation(self, metric_name: str, missing_columns: List[str]) -> Dict[str, Any]:
        """Create mock evaluation data."""
        mock_values = {
            'throughput': 100.0,
            'toxicity': 0.1,
            'bias': 0.05
        }
        
        return {
            'metric': metric_name,
            'status': 'mock',
            'reason': f'Created mock data due to missing: {missing_columns}',
            'value': mock_values.get(metric_name, 0.0),
            'confidence': 0.1
        }


class ThroughputMetric(EvaluationMetric):
    """Throughput evaluation metric."""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__('throughput', config)
    
    def evaluate(self, data: Union[Any, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate throughput metric."""
        # Find required columns
        events_col = self._find_column(data, 'throughput', 'events')
        time_col = self._find_column(data, 'throughput', 'time')
        tokens_col = self._find_column(data, 'throughput', 'tokens')
        
        required_columns = ['events', 'time']
        missing_columns = []
        
        if not events_col:
            missing_columns.append('events')
        if not time_col:
            missing_columns.append('time')
        
        if missing_columns:
            return self._handle_missing_column('throughput', 'events', required_columns)
        
        try:
            # Get data
            events_data = self._get_column_data(data, events_col)
            time_data = self._get_column_data(data, time_col)
            
            # Calculate throughput
            if isinstance(events_data, (list, np.ndarray)) and isinstance(time_data, (list, np.ndarray)):
                events = np.array(events_data)
                time = np.array(time_data)
                
                if len(events) != len(time):
                    return {
                        'metric': 'throughput',
                        'status': 'error',
                        'reason': 'Mismatched array lengths for events and time',
                        'value': None,
                        'confidence': 0.0
                    }
                
                # Calculate throughput (events per second)
                total_events = np.sum(events)
                total_time = np.sum(time)
                
                if total_time > 0:
                    throughput = total_events / total_time
                else:
                    throughput = 0.0
                
                # Calculate additional metrics
                avg_events_per_step = np.mean(events) if len(events) > 0 else 0
                time_variance = np.var(time) if len(time) > 1 else 0
                
                return {
                    'metric': 'throughput',
                    'status': 'success',
                    'value': float(throughput),
                    'confidence': 1.0,
                    'details': {
                        'total_events': int(total_events),
                        'total_time': float(total_time),
                        'avg_events_per_step': float(avg_events_per_step),
                        'time_variance': float(time_variance),
                        'tokens_per_second': float(throughput) if tokens_col else None
                    }
                }
            else:
                return {
                    'metric': 'throughput',
                    'status': 'error',
                    'reason': 'Invalid data types for events or time',
                    'value': None,
                    'confidence': 0.0
                }
                
        except Exception as e:
            return {
                'metric': 'throughput',
                'status': 'error',
                'reason': f'Error calculating throughput: {str(e)}',
                'value': None,
                'confidence': 0.0
            }


class ToxicityMetric(EvaluationMetric):
    """Toxicity evaluation metric."""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__('toxicity', config)
    
    def evaluate(self, data: Union[Any, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate toxicity metric."""
        # Find required columns
        output_col = self._find_column(data, 'toxicity', 'output')
        
        if not output_col:
            return self._handle_missing_column('toxicity', 'output', ['output'])
        
        try:
            # Get data
            output_data = self._get_column_data(data, output_col)
            
            if isinstance(output_data, (list, np.ndarray)):
                texts = output_data
            elif isinstance(output_data, str):
                texts = [output_data]
            else:
                return {
                    'metric': 'toxicity',
                    'status': 'error',
                    'reason': 'Invalid data type for output column',
                    'value': None,
                    'confidence': 0.0
                }
            
            # Calculate toxicity scores (simplified implementation)
            toxicity_scores = []
            for text in texts:
                if isinstance(text, str):
                    # Simple heuristic-based toxicity detection
                    toxicity_score = self._calculate_toxicity_score(text)
                    toxicity_scores.append(toxicity_score)
                else:
                    toxicity_scores.append(0.0)
            
            if not toxicity_scores:
                return {
                    'metric': 'toxicity',
                    'status': 'error',
                    'reason': 'No valid text data found',
                    'value': None,
                    'confidence': 0.0
                }
            
            # Calculate metrics
            avg_toxicity = np.mean(toxicity_scores)
            max_toxicity = np.max(toxicity_scores)
            toxic_count = sum(1 for score in toxicity_scores if score > self.config.thresholds['toxicity_max'])
            
            return {
                'metric': 'toxicity',
                'status': 'success',
                'value': float(avg_toxicity),
                'confidence': 1.0,
                'details': {
                    'avg_toxicity': float(avg_toxicity),
                    'max_toxicity': float(max_toxicity),
                    'toxic_samples': int(toxic_count),
                    'total_samples': len(toxicity_scores),
                    'threshold': self.config.thresholds['toxicity_max']
                }
            }
            
        except Exception as e:
            return {
                'metric': 'toxicity',
                'status': 'error',
                'reason': f'Error calculating toxicity: {str(e)}',
                'value': None,
                'confidence': 0.0
            }
    
    def _calculate_toxicity_score(self, text: str) -> float:
        """Calculate toxicity score for a text (simplified implementation)."""
        # This is a placeholder - in practice, you'd use a proper toxicity detection model
        toxic_keywords = ['hate', 'violence', 'abuse', 'offensive', 'toxic']
        text_lower = text.lower()
        
        score = 0.0
        for keyword in toxic_keywords:
            if keyword in text_lower:
                score += 0.2
        
        return min(score, 1.0)


class BiasMetric(EvaluationMetric):
    """Bias evaluation metric."""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__('bias', config)
    
    def evaluate(self, data: Union[Any, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate bias metric."""
        # Find required columns
        output_col = self._find_column(data, 'bias', 'output')
        
        if not output_col:
            return self._handle_missing_column('bias', 'output', ['output'])
        
        try:
            # Get data
            output_data = self._get_column_data(data, output_col)
            
            if isinstance(output_data, (list, np.ndarray)):
                texts = output_data
            elif isinstance(output_data, str):
                texts = [output_data]
            else:
                return {
                    'metric': 'bias',
                    'status': 'error',
                    'reason': 'Invalid data type for output column',
                    'value': None,
                    'confidence': 0.0
                }
            
            # Calculate bias scores (simplified implementation)
            bias_scores = []
            for text in texts:
                if isinstance(text, str):
                    # Simple heuristic-based bias detection
                    bias_score = self._calculate_bias_score(text)
                    bias_scores.append(bias_score)
                else:
                    bias_scores.append(0.0)
            
            if not bias_scores:
                return {
                    'metric': 'bias',
                    'status': 'error',
                    'reason': 'No valid text data found',
                    'value': None,
                    'confidence': 0.0
                }
            
            # Calculate metrics
            avg_bias = np.mean(bias_scores)
            max_bias = np.max(bias_scores)
            biased_count = sum(1 for score in bias_scores if score > self.config.thresholds['bias_max'])
            
            return {
                'metric': 'bias',
                'status': 'success',
                'value': float(avg_bias),
                'confidence': 1.0,
                'details': {
                    'avg_bias': float(avg_bias),
                    'max_bias': float(max_bias),
                    'biased_samples': int(biased_count),
                    'total_samples': len(bias_scores),
                    'threshold': self.config.thresholds['bias_max']
                }
            }
            
        except Exception as e:
            return {
                'metric': 'bias',
                'status': 'error',
                'reason': f'Error calculating bias: {str(e)}',
                'value': None,
                'confidence': 0.0
            }
    
    def _calculate_bias_score(self, text: str) -> float:
        """Calculate bias score for a text (simplified implementation)."""
        # This is a placeholder - in practice, you'd use a proper bias detection model
        bias_keywords = ['stereotype', 'discriminatory', 'unfair', 'biased', 'prejudiced']
        text_lower = text.lower()
        
        score = 0.0
        for keyword in bias_keywords:
            if keyword in text_lower:
                score += 0.2
        
        return min(score, 1.0)


class EvaluationSuite:
    """Main evaluation suite with robust error handling."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.logger = logging.getLogger("evaluation_suite")
        
        # Initialize metrics
        self.metrics = {
            'throughput': ThroughputMetric(self.config),
            'toxicity': ToxicityMetric(self.config),
            'bias': BiasMetric(self.config)
        }
    
    def evaluate(self, data: Union[Any, Dict[str, Any], str]) -> Dict[str, Any]:
        """Evaluate all metrics on the given data."""
        # Load data if path is provided
        if isinstance(data, str):
            data = self._load_data(data)
        
        from datetime import datetime
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'strict_mode': self.config.strict_mode,
                'fallback_strategies': self.config.fallback_strategies
            },
            'metrics': {},
            'summary': {
                'total_metrics': len(self.metrics),
                'successful_metrics': 0,
                'failed_metrics': 0,
                'skipped_metrics': 0
            }
        }
        
        # Evaluate each metric
        for metric_name, metric in self.metrics.items():
            try:
                metric_result = metric.evaluate(data)
                results['metrics'][metric_name] = metric_result
                
                # Update summary
                if metric_result['status'] == 'success':
                    results['summary']['successful_metrics'] += 1
                elif metric_result['status'] == 'skipped':
                    results['summary']['skipped_metrics'] += 1
                else:
                    results['summary']['failed_metrics'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error evaluating {metric_name}: {e}")
                results['metrics'][metric_name] = {
                    'metric': metric_name,
                    'status': 'error',
                    'reason': f'Unexpected error: {str(e)}',
                    'value': None,
                    'confidence': 0.0
                }
                results['summary']['failed_metrics'] += 1
        
        return results
    
    def _load_data(self, data_path: str) -> Union[Any, Dict[str, Any]]:
        """Load data from various formats."""
        path = Path(data_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Try different formats
        if path.suffix == '.csv':
            if PANDAS_AVAILABLE:
                return pd.read_csv(data_path)
            else:
                # Basic CSV parsing without pandas
                data = []
                columns = []
                with open(data_path, 'r') as f:
                    for i, line in enumerate(f):
                        row = line.strip().split(',')
                        if i == 0:
                            columns = row
                        else:
                            data.append(dict(zip(columns, row)))
                return data
        elif path.suffix == '.json':
            with open(data_path, 'r') as f:
                return json.load(f)
        elif path.suffix == '.jsonl':
            data = []
            with open(data_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data
        elif path.suffix == '.parquet':
            if PANDAS_AVAILABLE:
                return pd.read_parquet(data_path)
            else:
                raise ValueError("Parquet files require pandas")
        else:
            # Try to load as JSON
            try:
                with open(data_path, 'r') as f:
                    return json.load(f)
            except:
                raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save evaluation results to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Evaluation results saved to: {output_path}")
    
    def get_column_requirements(self) -> Dict[str, List[str]]:
        """Get column requirements for each metric."""
        requirements = {}
        for metric_name, metric in self.metrics.items():
            if metric_name in self.config.column_mappings:
                mapping = self.config.column_mappings[metric_name]
                requirements[metric_name] = [
                    col for col in mapping.keys() 
                    if col != 'fallback_columns'
                ]
        return requirements
    
    def validate_data(self, data: Union[Any, Dict[str, Any]]) -> Dict[str, Any]:
        """Validate data against column requirements."""
        validation_results = {
            'valid': True,
            'missing_columns': {},
            'recommendations': []
        }
        
        requirements = self.get_column_requirements()
        
        for metric_name, required_columns in requirements.items():
            missing = []
            for col in required_columns:
                if not self._column_exists(data, col):
                    missing.append(col)
            
            if missing:
                validation_results['valid'] = False
                validation_results['missing_columns'][metric_name] = missing
                validation_results['recommendations'].append(
                    f"Add columns {missing} for {metric_name} evaluation"
                )
        
        return validation_results


def create_evaluation_suite_example():
    """Create an example of the evaluation suite usage."""
    
    # Create configuration
    config = EvaluationConfig(
        strict_mode=False,
        log_missing_columns=True,
        fallback_strategies={
            'throughput': 'estimate_from_steps',
            'toxicity': 'skip_evaluation',
            'bias': 'skip_evaluation'
        }
    )
    
    # Initialize evaluation suite
    suite = EvaluationSuite(config)
    
    # Example data with missing columns
    sample_data = {
        'steps': [1, 2, 3, 4, 5],
        'time': [0.1, 0.2, 0.3, 0.4, 0.5],
        'generated_text': ['Hello world', 'How are you', 'Good morning', 'Nice day', 'See you later']
    }
    
    # Evaluate
    results = suite.evaluate(sample_data)
    
    print("Evaluation Results:")
    print(json.dumps(results, indent=2))
    
    return suite, results


if __name__ == "__main__":
    # Run example
    suite, results = create_evaluation_suite_example()