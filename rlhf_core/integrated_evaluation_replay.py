"""
Integrated Evaluation and Replay System

This module demonstrates the integration of the robust evaluation suite
and flexible replay system for comprehensive RLHF training analysis.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime

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

from .evaluation_suite import EvaluationSuite, EvaluationConfig
from .replay_system import ReplaySystem, ReplayConfig


class IntegratedEvaluationReplay:
    """Integrated system combining evaluation suite and replay system."""
    
    def __init__(self, 
                 evaluation_config: Optional[EvaluationConfig] = None,
                 replay_config: Optional[ReplayConfig] = None):
        self.evaluation_suite = EvaluationSuite(evaluation_config)
        self.replay_system = ReplaySystem(replay_config)
        self.logger = logging.getLogger("integrated_evaluation_replay")
    
    def analyze_training_run(self, run_directory: Union[str, Path]) -> Dict[str, Any]:
        """Analyze a complete training run using both systems."""
        run_path = Path(run_directory)
        
        if not run_path.exists():
            raise FileNotFoundError(f"Training run directory not found: {run_path}")
        
        self.logger.info(f"Analyzing training run: {run_path}")
        
        analysis_results = {
            'run_directory': str(run_path),
            'timestamp': datetime.now().isoformat(),
            'replay_results': {},
            'evaluation_results': {},
            'integration_summary': {}
        }
        
        # Step 1: Replay the training data
        try:
            replay_result = self.replay_system.replay(run_path)
            analysis_results['replay_results'] = replay_result
            
            if replay_result['status'] in ['success', 'success_fallback', 'mock_data']:
                self.logger.info("Successfully replayed training data")
            else:
                self.logger.warning(f"Replay failed: {replay_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.logger.error(f"Error during replay: {e}")
            analysis_results['replay_results'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Step 2: Extract data for evaluation
        evaluation_data = self._extract_evaluation_data(analysis_results['replay_results'])
        
        # Step 3: Run evaluation suite
        try:
            evaluation_result = self.evaluation_suite.evaluate(evaluation_data)
            analysis_results['evaluation_results'] = evaluation_result
            
            if evaluation_result['summary']['successful_metrics'] > 0:
                self.logger.info("Successfully evaluated training metrics")
            else:
                self.logger.warning("No metrics could be evaluated successfully")
                
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            analysis_results['evaluation_results'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Step 4: Generate integration summary
        analysis_results['integration_summary'] = self._generate_integration_summary(
            analysis_results['replay_results'],
            analysis_results['evaluation_results']
        )
        
        return analysis_results
    
    def _extract_evaluation_data(self, replay_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data suitable for evaluation from replay results."""
        if replay_result['status'] not in ['success', 'success_fallback', 'mock_data']:
            return {}
        
        data = replay_result.get('data', {})
        
        # Try to extract common training data patterns
        evaluation_data = {}
        
        # Look for throughput data
        if 'data' in data:
            if isinstance(data['data'], list):
                # Try to extract events and time data
                events = []
                time_data = []
                outputs = []
                
                for item in data['data']:
                    if isinstance(item, dict):
                        # Look for common column names
                        if 'events' in item:
                            events.append(item['events'])
                        elif 'steps' in item:
                            events.append(item['steps'])
                        
                        if 'time' in item:
                            time_data.append(item['time'])
                        elif 'duration' in item:
                            time_data.append(item['duration'])
                        
                        if 'output' in item:
                            outputs.append(item['output'])
                        elif 'text' in item:
                            outputs.append(item['text'])
                        elif 'response' in item:
                            outputs.append(item['response'])
                
                if events:
                    evaluation_data['events'] = events
                if time_data:
                    evaluation_data['time'] = time_data
                if outputs:
                    evaluation_data['output'] = outputs
        
        # If no data found, create mock data for demonstration
        if not evaluation_data:
            evaluation_data = {
                'events': [1, 2, 3, 4, 5],
                'time': [0.1, 0.2, 0.3, 0.4, 0.5],
                'output': ['Sample output 1', 'Sample output 2', 'Sample output 3']
            }
            self.logger.info("Created mock evaluation data")
        
        return evaluation_data
    
    def _generate_integration_summary(self, replay_result: Dict[str, Any], 
                                    evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of integrated analysis."""
        summary = {
            'replay_status': replay_result.get('status', 'unknown'),
            'evaluation_status': evaluation_result.get('summary', {}).get('successful_metrics', 0),
            'adapter_used': replay_result.get('adapter_used', 'unknown'),
            'metrics_evaluated': evaluation_result.get('summary', {}).get('successful_metrics', 0),
            'total_metrics': evaluation_result.get('summary', {}).get('total_metrics', 0),
            'recommendations': []
        }
        
        # Generate recommendations
        if replay_result.get('status') == 'error':
            summary['recommendations'].append("Fix data format issues for proper replay")
        
        if evaluation_result.get('summary', {}).get('successful_metrics', 0) == 0:
            summary['recommendations'].append("Ensure data contains required columns for evaluation")
        
        if replay_result.get('adapter_used') == 'generic':
            summary['recommendations'].append("Consider converting data to a more specific format")
        
        if evaluation_result.get('summary', {}).get('skipped_metrics', 0) > 0:
            summary['recommendations'].append("Add missing columns to enable all evaluations")
        
        return summary
    
    def batch_analyze_runs(self, runs_directory: Union[str, Path]) -> Dict[str, Any]:
        """Analyze multiple training runs in batch."""
        runs_path = Path(runs_directory)
        
        if not runs_path.exists():
            raise FileNotFoundError(f"Runs directory not found: {runs_path}")
        
        self.logger.info(f"Batch analyzing runs in: {runs_path}")
        
        batch_results = {
            'runs_directory': str(runs_path),
            'timestamp': datetime.now().isoformat(),
            'runs_analyzed': [],
            'summary': {
                'total_runs': 0,
                'successful_runs': 0,
                'failed_runs': 0,
                'common_issues': []
            }
        }
        
        # Find all run directories
        run_directories = []
        for item in runs_path.iterdir():
            if item.is_dir():
                run_directories.append(item)
        
        batch_results['summary']['total_runs'] = len(run_directories)
        
        # Analyze each run
        for run_dir in run_directories:
            try:
                run_analysis = self.analyze_training_run(run_dir)
                batch_results['runs_analyzed'].append(run_analysis)
                
                if run_analysis['replay_results'].get('status') in ['success', 'success_fallback']:
                    batch_results['summary']['successful_runs'] += 1
                else:
                    batch_results['summary']['failed_runs'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error analyzing run {run_dir}: {e}")
                batch_results['runs_analyzed'].append({
                    'run_directory': str(run_dir),
                    'error': str(e),
                    'status': 'failed'
                })
                batch_results['summary']['failed_runs'] += 1
        
        # Identify common issues
        common_issues = self._identify_common_issues(batch_results['runs_analyzed'])
        batch_results['summary']['common_issues'] = common_issues
        
        return batch_results
    
    def _identify_common_issues(self, runs_analyzed: List[Dict[str, Any]]) -> List[str]:
        """Identify common issues across multiple runs."""
        issues = []
        
        # Count adapter usage
        adapter_usage = {}
        for run in runs_analyzed:
            if 'replay_results' in run:
                adapter = run['replay_results'].get('adapter_used', 'unknown')
                adapter_usage[adapter] = adapter_usage.get(adapter, 0) + 1
        
        if adapter_usage.get('generic', 0) > len(runs_analyzed) * 0.5:
            issues.append("Many runs using generic adapter - consider standardizing data format")
        
        # Count evaluation failures
        evaluation_failures = 0
        for run in runs_analyzed:
            if 'evaluation_results' in run:
                eval_summary = run['evaluation_results'].get('summary', {})
                if eval_summary.get('successful_metrics', 0) == 0:
                    evaluation_failures += 1
        
        if evaluation_failures > len(runs_analyzed) * 0.3:
            issues.append("Many runs have evaluation failures - check data columns")
        
        return issues
    
    def generate_comprehensive_report(self, analysis_results: Dict[str, Any], 
                                    output_path: str) -> str:
        """Generate a comprehensive report from analysis results."""
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("# Comprehensive Training Analysis Report\n\n")
            f.write(f"**Generated**: {analysis_results['timestamp']}\n")
            f.write(f"**Run Directory**: {analysis_results['run_directory']}\n\n")
            
            # Replay Results
            f.write("## Replay Results\n\n")
            replay_results = analysis_results['replay_results']
            f.write(f"- **Status**: {replay_results.get('status', 'unknown')}\n")
            f.write(f"- **Adapter Used**: {replay_results.get('adapter_used', 'unknown')}\n")
            
            if replay_results.get('status') == 'error':
                f.write(f"- **Error**: {replay_results.get('error', 'Unknown error')}\n")
            
            f.write("\n")
            
            # Evaluation Results
            f.write("## Evaluation Results\n\n")
            evaluation_results = analysis_results['evaluation_results']
            if 'summary' in evaluation_results:
                summary = evaluation_results['summary']
                f.write(f"- **Successful Metrics**: {summary.get('successful_metrics', 0)}\n")
                f.write(f"- **Failed Metrics**: {summary.get('failed_metrics', 0)}\n")
                f.write(f"- **Skipped Metrics**: {summary.get('skipped_metrics', 0)}\n")
                
                # Individual metric results
                if 'metrics' in evaluation_results:
                    f.write("\n### Metric Details\n\n")
                    for metric_name, metric_result in evaluation_results['metrics'].items():
                        f.write(f"#### {metric_name}\n")
                        f.write(f"- **Status**: {metric_result.get('status', 'unknown')}\n")
                        f.write(f"- **Value**: {metric_result.get('value', 'N/A')}\n")
                        f.write(f"- **Confidence**: {metric_result.get('confidence', 'N/A')}\n")
                        if 'reason' in metric_result:
                            f.write(f"- **Reason**: {metric_result['reason']}\n")
                        f.write("\n")
            
            # Integration Summary
            f.write("## Integration Summary\n\n")
            integration_summary = analysis_results['integration_summary']
            f.write(f"- **Replay Status**: {integration_summary.get('replay_status', 'unknown')}\n")
            f.write(f"- **Metrics Evaluated**: {integration_summary.get('metrics_evaluated', 0)}\n")
            f.write(f"- **Total Metrics**: {integration_summary.get('total_metrics', 0)}\n")
            
            if integration_summary.get('recommendations'):
                f.write("\n### Recommendations\n\n")
                for i, rec in enumerate(integration_summary['recommendations'], 1):
                    f.write(f"{i}. {rec}\n")
                f.write("\n")
        
        self.logger.info(f"Comprehensive report generated: {report_path}")
        return str(report_path)


def create_integrated_example():
    """Create an example of the integrated system."""
    
    # Create configurations
    evaluation_config = EvaluationConfig(
        strict_mode=False,
        log_missing_columns=True,
        fallback_strategies={
            'throughput': 'estimate_from_steps',
            'toxicity': 'skip_evaluation',
            'bias': 'skip_evaluation'
        }
    )
    
    replay_config = ReplayConfig(
        strict_mode=False,
        log_adapter_selection=True,
        create_mock_data_on_failure=True
    )
    
    # Initialize integrated system
    integrated_system = IntegratedEvaluationReplay(evaluation_config, replay_config)
    
    # Example training run data
    sample_run_data = {
        'events': [1, 2, 3, 4, 5],
        'time': [0.1, 0.2, 0.3, 0.4, 0.5],
        'output': ['Hello world', 'How are you', 'Good morning', 'Nice day', 'See you later']
    }
    
    # Create a mock replay result
    mock_replay_result = {
        'status': 'success',
        'adapter_used': 'json',
        'data': {
            'type': 'json',
            'data': [
                {'events': 1, 'time': 0.1, 'output': 'Hello world'},
                {'events': 2, 'time': 0.2, 'output': 'How are you'},
                {'events': 3, 'time': 0.3, 'output': 'Good morning'},
                {'events': 4, 'time': 0.4, 'output': 'Nice day'},
                {'events': 5, 'time': 0.5, 'output': 'See you later'}
            ]
        }
    }
    
    # Extract evaluation data
    evaluation_data = integrated_system._extract_evaluation_data(mock_replay_result)
    
    # Run evaluation
    evaluation_result = integrated_system.evaluation_suite.evaluate(evaluation_data)
    
    # Generate analysis results
    analysis_results = {
        'run_directory': './test_runs/sample_run',
        'timestamp': pd.Timestamp.now().isoformat(),
        'replay_results': mock_replay_result,
        'evaluation_results': evaluation_result,
        'integration_summary': integrated_system._generate_integration_summary(
            mock_replay_result, evaluation_result
        )
    }
    
    # Generate report
    report_path = integrated_system.generate_comprehensive_report(
        analysis_results, './reports/comprehensive_analysis_report.md'
    )
    
    print("Integrated Analysis Results:")
    print(json.dumps(analysis_results, indent=2))
    print(f"\nReport generated: {report_path}")
    
    return integrated_system, analysis_results


if __name__ == "__main__":
    # Run example
    integrated_system, results = create_integrated_example()