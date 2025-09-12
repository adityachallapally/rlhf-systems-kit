#!/usr/bin/env python3
"""
Demonstration of Robust Evaluation Suite and Replay System

This script demonstrates the improved evaluation suite and replay system
with graceful fallbacks for missing columns and flexible data format handling.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from rlhf_core.evaluation_suite import EvaluationSuite, EvaluationConfig
from rlhf_core.replay_system import ReplaySystem, ReplayConfig
from rlhf_core.integrated_evaluation_replay import IntegratedEvaluationReplay


def setup_logging():
    """Setup logging for the demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('demo_evaluation_replay.log')
        ]
    )


def demonstrate_evaluation_suite():
    """Demonstrate the robust evaluation suite."""
    print("\n" + "="*60)
    print("🔍 DEMONSTRATING ROBUST EVALUATION SUITE")
    print("="*60)
    
    # Create configuration with graceful fallbacks
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
    
    print("✅ Evaluation suite initialized with graceful fallbacks")
    
    # Test case 1: Data with missing columns (should handle gracefully)
    print("\n📊 Test Case 1: Data with missing columns")
    incomplete_data = {
        'steps': [1, 2, 3, 4, 5],
        'time': [0.1, 0.2, 0.3, 0.4, 0.5],
        'generated_text': ['Hello world', 'How are you', 'Good morning', 'Nice day', 'See you later']
    }
    
    results1 = suite.evaluate(incomplete_data)
    print(f"   Status: {results1['summary']['successful_metrics']} successful, {results1['summary']['skipped_metrics']} skipped")
    
    # Test case 2: Data with correct columns
    print("\n📊 Test Case 2: Data with correct columns")
    complete_data = {
        'events': [1, 2, 3, 4, 5],
        'time': [0.1, 0.2, 0.3, 0.4, 0.5],
        'output': ['Hello world', 'How are you', 'Good morning', 'Nice day', 'See you later']
    }
    
    results2 = suite.evaluate(complete_data)
    print(f"   Status: {results2['summary']['successful_metrics']} successful, {results2['summary']['skipped_metrics']} skipped")
    
    # Test case 3: Empty data (should handle gracefully)
    print("\n📊 Test Case 3: Empty data")
    empty_data = {}
    
    results3 = suite.evaluate(empty_data)
    print(f"   Status: {results3['summary']['successful_metrics']} successful, {results3['summary']['skipped_metrics']} skipped")
    
    # Show column requirements
    print("\n📋 Column Requirements:")
    requirements = suite.get_column_requirements()
    for metric, cols in requirements.items():
        print(f"   {metric}: {cols}")
    
    return suite


def demonstrate_replay_system():
    """Demonstrate the flexible replay system."""
    print("\n" + "="*60)
    print("🔄 DEMONSTRATING FLEXIBLE REPLAY SYSTEM")
    print("="*60)
    
    # Create configuration with flexible adapter selection
    config = ReplayConfig(
        strict_mode=False,
        log_adapter_selection=True,
        create_mock_data_on_failure=True
    )
    
    # Initialize replay system
    replay_system = ReplaySystem(config)
    
    print("✅ Replay system initialized with flexible adapter selection")
    
    # Show available adapters
    print("\n🔧 Available Adapters:")
    adapters = replay_system.get_available_adapters()
    for adapter in adapters:
        info = replay_system.get_adapter_info(adapter)
        print(f"   {adapter}: Priority {info['priority']}, Patterns {info['file_patterns']}")
    
    # Test case 1: JSON data
    print("\n📁 Test Case 1: JSON data")
    json_data = {
        'events': [1, 2, 3, 4, 5],
        'time': [0.1, 0.2, 0.3, 0.4, 0.5],
        'output': ['Hello world', 'How are you', 'Good morning', 'Nice day', 'See you later']
    }
    
    # Save JSON data to file
    os.makedirs('./test_data', exist_ok=True)
    with open('./test_data/sample.json', 'w') as f:
        json.dump(json_data, f)
    
    result1 = replay_system.replay('./test_data/sample.json')
    print(f"   Status: {result1['status']}")
    print(f"   Adapter: {result1.get('adapter_used', 'unknown')}")
    
    # Test case 2: JSONL data
    print("\n📁 Test Case 2: JSONL data")
    jsonl_data = [
        {'events': 1, 'time': 0.1, 'output': 'Hello world'},
        {'events': 2, 'time': 0.2, 'output': 'How are you'},
        {'events': 3, 'time': 0.3, 'output': 'Good morning'},
        {'events': 4, 'time': 0.4, 'output': 'Nice day'},
        {'events': 5, 'time': 0.5, 'output': 'See you later'}
    ]
    
    with open('./test_data/sample.jsonl', 'w') as f:
        for item in jsonl_data:
            f.write(json.dumps(item) + '\n')
    
    result2 = replay_system.replay('./test_data/sample.jsonl')
    print(f"   Status: {result2['status']}")
    print(f"   Adapter: {result2.get('adapter_used', 'unknown')}")
    
    # Test case 3: Non-existent file (should handle gracefully)
    print("\n📁 Test Case 3: Non-existent file")
    result3 = replay_system.replay('./test_data/nonexistent.json')
    print(f"   Status: {result3['status']}")
    if 'error' in result3:
        print(f"   Error: {result3['error']}")
    
    # Test case 4: Directory (should try multiple adapters)
    print("\n📁 Test Case 4: Directory")
    os.makedirs('./test_runs/sample_run', exist_ok=True)
    with open('./test_runs/sample_run/config.json', 'w') as f:
        json.dump({'model': 'gpt2', 'steps': 100}, f)
    
    result4 = replay_system.replay('./test_runs/sample_run')
    print(f"   Status: {result4['status']}")
    print(f"   Adapter: {result4.get('adapter_used', 'unknown')}")
    
    return replay_system


def demonstrate_integrated_system():
    """Demonstrate the integrated evaluation and replay system."""
    print("\n" + "="*60)
    print("🚀 DEMONSTRATING INTEGRATED SYSTEM")
    print("="*60)
    
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
    
    print("✅ Integrated system initialized")
    
    # Create sample training run data
    print("\n📊 Creating sample training run data")
    os.makedirs('./test_runs/sample_training_run', exist_ok=True)
    
    # Create training logs
    training_logs = []
    for step in range(10):
        log_entry = {
            'step': step,
            'events': step + 1,
            'time': 0.1 * (step + 1),
            'output': f'Generated text for step {step}',
            'reward': 2.0 + (step * 0.1),
            'kl_div': 0.1 + (step * 0.01)
        }
        training_logs.append(log_entry)
    
    # Save as JSONL
    with open('./test_runs/sample_training_run/training_logs.jsonl', 'w') as f:
        for log in training_logs:
            f.write(json.dumps(log) + '\n')
    
    # Save as JSON
    with open('./test_runs/sample_training_run/training_data.json', 'w') as f:
        json.dump(training_logs, f)
    
    # Save config
    with open('./test_runs/sample_training_run/config.json', 'w') as f:
        json.dump({
            'model': 'gpt2',
            'learning_rate': 1e-5,
            'batch_size': 4,
            'total_steps': 10
        }, f)
    
    print("   ✅ Sample training run data created")
    
    # Analyze the training run
    print("\n🔍 Analyzing training run")
    analysis_results = integrated_system.analyze_training_run('./test_runs/sample_training_run')
    
    print(f"   Replay Status: {analysis_results['replay_results']['status']}")
    print(f"   Adapter Used: {analysis_results['replay_results'].get('adapter_used', 'unknown')}")
    print(f"   Metrics Evaluated: {analysis_results['evaluation_results']['summary']['successful_metrics']}")
    
    # Generate comprehensive report
    print("\n📋 Generating comprehensive report")
    report_path = integrated_system.generate_comprehensive_report(
        analysis_results, './reports/training_analysis_report.md'
    )
    print(f"   ✅ Report generated: {report_path}")
    
    # Show integration summary
    print("\n📊 Integration Summary:")
    summary = analysis_results['integration_summary']
    print(f"   Replay Status: {summary['replay_status']}")
    print(f"   Metrics Evaluated: {summary['metrics_evaluated']}/{summary['total_metrics']}")
    print(f"   Adapter Used: {summary['adapter_used']}")
    
    if summary['recommendations']:
        print("   Recommendations:")
        for i, rec in enumerate(summary['recommendations'], 1):
            print(f"     {i}. {rec}")
    
    return integrated_system, analysis_results


def demonstrate_error_handling():
    """Demonstrate error handling and fallback mechanisms."""
    print("\n" + "="*60)
    print("⚠️  DEMONSTRATING ERROR HANDLING & FALLBACKS")
    print("="*60)
    
    # Test evaluation suite with various error conditions
    print("\n🔍 Testing Evaluation Suite Error Handling")
    
    evaluation_config = EvaluationConfig(
        strict_mode=False,
        log_missing_columns=True,
        fallback_strategies={
            'throughput': 'estimate_from_steps',
            'toxicity': 'skip_evaluation',
            'bias': 'skip_evaluation'
        }
    )
    
    suite = EvaluationSuite(evaluation_config)
    
    # Test with completely missing data
    print("   Testing with missing data...")
    result = suite.evaluate({})
    print(f"   Result: {result['summary']['successful_metrics']} successful, {result['summary']['skipped_metrics']} skipped")
    
    # Test with invalid data types
    print("   Testing with invalid data types...")
    invalid_data = {
        'events': 'not_a_list',
        'time': None,
        'output': 123
    }
    result = suite.evaluate(invalid_data)
    print(f"   Result: {result['summary']['successful_metrics']} successful, {result['summary']['failed_metrics']} failed")
    
    # Test replay system with various error conditions
    print("\n🔄 Testing Replay System Error Handling")
    
    replay_config = ReplayConfig(
        strict_mode=False,
        log_adapter_selection=True,
        create_mock_data_on_failure=True
    )
    
    replay_system = ReplaySystem(replay_config)
    
    # Test with non-existent file
    print("   Testing with non-existent file...")
    result = replay_system.replay('./nonexistent_file.json')
    print(f"   Result: {result['status']}")
    
    # Test with invalid file format
    print("   Testing with invalid file format...")
    with open('./test_data/invalid.txt', 'w') as f:
        f.write("This is not valid JSON or any supported format")
    
    result = replay_system.replay('./test_data/invalid.txt')
    print(f"   Result: {result['status']}")
    print(f"   Adapter: {result.get('adapter_used', 'unknown')}")
    
    print("\n✅ Error handling demonstration completed")


def main():
    """Main demonstration function."""
    print("🎯 Robust Evaluation Suite and Replay System Demonstration")
    print("="*80)
    print("This demonstration shows:")
    print("1. Graceful handling of missing columns in evaluation metrics")
    print("2. Flexible data format detection and adapter selection")
    print("3. Clear error messages and fallback mechanisms")
    print("4. Integrated analysis of training runs")
    print("="*80)
    
    # Setup logging
    setup_logging()
    
    try:
        # Demonstrate evaluation suite
        suite = demonstrate_evaluation_suite()
        
        # Demonstrate replay system
        replay_system = demonstrate_replay_system()
        
        # Demonstrate integrated system
        integrated_system, analysis_results = demonstrate_integrated_system()
        
        # Demonstrate error handling
        demonstrate_error_handling()
        
        print("\n" + "="*80)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("Key improvements demonstrated:")
        print("✅ Graceful fallbacks for missing columns")
        print("✅ Flexible data format detection")
        print("✅ Clear error messages and recommendations")
        print("✅ Multiple adapter support with fallback mechanisms")
        print("✅ Integrated analysis capabilities")
        print("✅ Robust error handling")
        
        print("\n📁 Generated files:")
        print("   - demo_evaluation_replay.log (detailed logs)")
        print("   - ./test_data/ (sample data files)")
        print("   - ./test_runs/ (sample training runs)")
        print("   - ./reports/ (analysis reports)")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        logging.error(f"Demonstration failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())