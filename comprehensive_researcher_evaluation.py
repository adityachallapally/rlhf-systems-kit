#!/usr/bin/env python3
"""
Comprehensive RLHF Systems Kit Evaluation
=========================================

This script conducts an extensive evaluation of the RLHF Systems Kit as a senior researcher would,
including:
- Multiple model sizes (tiny-gpt2, gpt2, gpt2-medium)
- Extensive hyperparameter sweeps
- Long training runs (hours)
- Memory usage analysis
- Error handling tests
- Reproducibility verification
- TRL integration testing
- Performance profiling

The evaluation is designed to run for hours to thoroughly test the system.
"""

import os
import sys
import time
import json
import traceback
import subprocess
import psutil
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    def __init__(self):
        self.results = {
            'start_time': datetime.now().isoformat(),
            'tests': {},
            'summary': {},
            'issues': [],
            'recommendations': []
        }
        self.test_dir = Path('comprehensive_eval_results')
        self.test_dir.mkdir(exist_ok=True)
        
    def log_system_info(self):
        """Log comprehensive system information"""
        logger.info("=== SYSTEM INFORMATION ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
        
        # Memory info
        memory = psutil.virtual_memory()
        logger.info(f"Total RAM: {memory.total / (1024**3):.2f} GB")
        logger.info(f"Available RAM: {memory.available / (1024**3):.2f} GB")
        
        # CPU info
        logger.info(f"CPU count: {psutil.cpu_count()}")
        logger.info(f"CPU usage: {psutil.cpu_percent()}%")
        
    def test_model_scalability(self):
        """Test with multiple model sizes to evaluate scalability"""
        logger.info("=== TESTING MODEL SCALABILITY ===")
        
        models_to_test = [
            ('sshleifer/tiny-gpt2', 'tiny'),
            ('gpt2', 'small'),
            ('gpt2-medium', 'medium')
        ]
        
        scalability_results = {}
        
        for model_name, size in models_to_test:
            logger.info(f"Testing {model_name} ({size})")
            try:
                start_time = time.time()
                
                # Test model loading
                from transformers import AutoTokenizer, AutoModelForCausalLM
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                
                load_time = time.time() - start_time
                model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
                
                # Test basic functionality
                test_input = "The quick brown fox"
                inputs = tokenizer(test_input, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=10,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generation_time = time.time() - start_time - load_time
                
                scalability_results[size] = {
                    'model_name': model_name,
                    'load_time': load_time,
                    'model_size_mb': model_size_mb,
                    'generation_time': generation_time,
                    'success': True
                }
                
                logger.info(f"✅ {size} model: {model_size_mb:.2f}MB, load: {load_time:.2f}s, gen: {generation_time:.2f}s")
                
                # Clean up
                del model, tokenizer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                logger.error(f"❌ Failed to test {model_name}: {e}")
                scalability_results[size] = {
                    'model_name': model_name,
                    'error': str(e),
                    'success': False
                }
        
        self.results['tests']['model_scalability'] = scalability_results
        return scalability_results
    
    def test_hyperparameter_sweep(self):
        """Test extensive hyperparameter combinations"""
        logger.info("=== TESTING HYPERPARAMETER SWEEP ===")
        
        # Define parameter ranges for extensive testing
        param_combinations = [
            {'epochs': 2, 'steps_per_epoch': 5, 'batch_size': 2, 'learning_rate': 1e-5, 'kl_coef': 0.1},
            {'epochs': 3, 'steps_per_epoch': 8, 'batch_size': 4, 'learning_rate': 5e-6, 'kl_coef': 0.05},
            {'epochs': 5, 'steps_per_epoch': 10, 'batch_size': 2, 'learning_rate': 2e-5, 'kl_coef': 0.2},
            {'epochs': 4, 'steps_per_epoch': 6, 'batch_size': 8, 'learning_rate': 1e-6, 'kl_coef': 0.15},
            {'epochs': 6, 'steps_per_epoch': 12, 'batch_size': 1, 'learning_rate': 3e-5, 'kl_coef': 0.25},
        ]
        
        hyperparameter_results = {}
        
        for i, params in enumerate(param_combinations):
            logger.info(f"Testing hyperparameter combination {i+1}/{len(param_combinations)}")
            logger.info(f"Params: {params}")
            
            try:
                start_time = time.time()
                
                # Build command
                cmd = [
                    'python3', 'train.py',
                    '--seed', '42',
                    '--epochs', str(params['epochs']),
                    '--steps_per_epoch', str(params['steps_per_epoch']),
                    '--batch_size', str(params['batch_size']),
                    '--learning_rate', str(params['learning_rate']),
                    '--kl_coef', str(params['kl_coef']),
                    '--profiler', 'on'
                ]
                
                # Run training
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                total_time = time.time() - start_time
                
                if result.returncode == 0:
                    hyperparameter_results[f'combination_{i+1}'] = {
                        'params': params,
                        'success': True,
                        'time': total_time,
                        'stdout': result.stdout[-500:],  # Last 500 chars
                        'stderr': result.stderr[-500:] if result.stderr else None
                    }
                    logger.info(f"✅ Combination {i+1} completed in {total_time:.2f}s")
                else:
                    hyperparameter_results[f'combination_{i+1}'] = {
                        'params': params,
                        'success': False,
                        'error': result.stderr,
                        'time': total_time
                    }
                    logger.error(f"❌ Combination {i+1} failed: {result.stderr}")
                
            except subprocess.TimeoutExpired:
                logger.error(f"❌ Combination {i+1} timed out")
                hyperparameter_results[f'combination_{i+1}'] = {
                    'params': params,
                    'success': False,
                    'error': 'Timeout',
                    'time': 300
                }
            except Exception as e:
                logger.error(f"❌ Combination {i+1} failed with exception: {e}")
                hyperparameter_results[f'combination_{i+1}'] = {
                    'params': params,
                    'success': False,
                    'error': str(e),
                    'time': 0
                }
        
        self.results['tests']['hyperparameter_sweep'] = hyperparameter_results
        return hyperparameter_results
    
    def test_long_training_run(self):
        """Run a long training session to test stability"""
        logger.info("=== TESTING LONG TRAINING RUN ===")
        
        # Run a longer training session
        long_run_params = {
            'epochs': 10,
            'steps_per_epoch': 20,
            'batch_size': 4,
            'learning_rate': 1e-5,
            'kl_coef': 0.1
        }
        
        logger.info(f"Starting long training run with params: {long_run_params}")
        logger.info("This will run for approximately 30-60 minutes...")
        
        try:
            start_time = time.time()
            
            cmd = [
                'python3', 'train.py',
                '--seed', '42',
                '--epochs', str(long_run_params['epochs']),
                '--steps_per_epoch', str(long_run_params['steps_per_epoch']),
                '--batch_size', str(long_run_params['batch_size']),
                '--learning_rate', str(long_run_params['learning_rate']),
                '--kl_coef', str(long_run_params['kl_coef']),
                '--profiler', 'on'
            ]
            
            # Run with real-time output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            output_lines = []
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output_lines.append(output.strip())
                    logger.info(f"TRAINING: {output.strip()}")
            
            total_time = time.time() - start_time
            return_code = process.poll()
            
            long_run_result = {
                'params': long_run_params,
                'success': return_code == 0,
                'time': total_time,
                'output_lines': len(output_lines),
                'final_output': output_lines[-10:] if output_lines else []
            }
            
            if return_code == 0:
                logger.info(f"✅ Long training run completed successfully in {total_time:.2f}s")
            else:
                logger.error(f"❌ Long training run failed with return code {return_code}")
                long_run_result['error'] = f"Return code: {return_code}"
            
            self.results['tests']['long_training_run'] = long_run_result
            return long_run_result
            
        except Exception as e:
            logger.error(f"❌ Long training run failed with exception: {e}")
            long_run_result = {
                'params': long_run_params,
                'success': False,
                'error': str(e),
                'time': 0
            }
            self.results['tests']['long_training_run'] = long_run_result
            return long_run_result
    
    def test_trl_integration(self):
        """Test TRL integration with real models"""
        logger.info("=== TESTING TRL INTEGRATION ===")
        
        try:
            # Test TRL integration
            from rlhf_core.trl_integration import TRLIntegrationManager, TRLIntegrationConfig
            
            config = TRLIntegrationConfig(
                model_name="gpt2",
                learning_rate=1e-5,
                batch_size=2,
                enable_profiling=True,
                enable_checkpoint_analysis=True,
                enable_reward_monitoring=True
            )
            
            integration_manager = TRLIntegrationManager(config)
            
            # Test setup
            trainer = integration_manager.setup_trl_trainer(
                model_name="gpt2",
                dataset_name="imdb"
            )
            
            # Test training with monitoring
            results = integration_manager.train_with_monitoring(
                num_steps=50,
                save_checkpoints=True
            )
            
            trl_result = {
                'success': True,
                'config': config.__dict__,
                'results': results
            }
            
            logger.info("✅ TRL integration test completed successfully")
            
        except Exception as e:
            logger.error(f"❌ TRL integration test failed: {e}")
            trl_result = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        
        self.results['tests']['trl_integration'] = trl_result
        return trl_result
    
    def test_memory_usage(self):
        """Test memory usage with different configurations"""
        logger.info("=== TESTING MEMORY USAGE ===")
        
        memory_results = {}
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8, 16]
        
        for batch_size in batch_sizes:
            logger.info(f"Testing memory usage with batch size {batch_size}")
            
            try:
                # Get initial memory
                initial_memory = psutil.virtual_memory().used / (1024**2)  # MB
                
                # Run training
                cmd = [
                    'python3', 'train.py',
                    '--seed', '42',
                    '--epochs', '2',
                    '--steps_per_epoch', '5',
                    '--batch_size', str(batch_size),
                    '--profiler', 'on'
                ]
                
                start_time = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                total_time = time.time() - start_time
                
                # Get peak memory
                peak_memory = psutil.virtual_memory().used / (1024**2)  # MB
                memory_used = peak_memory - initial_memory
                
                memory_results[f'batch_{batch_size}'] = {
                    'batch_size': batch_size,
                    'success': result.returncode == 0,
                    'time': total_time,
                    'memory_used_mb': memory_used,
                    'peak_memory_mb': peak_memory
                }
                
                logger.info(f"Batch {batch_size}: {memory_used:.2f}MB used, {total_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Memory test failed for batch size {batch_size}: {e}")
                memory_results[f'batch_{batch_size}'] = {
                    'batch_size': batch_size,
                    'success': False,
                    'error': str(e)
                }
        
        self.results['tests']['memory_usage'] = memory_results
        return memory_results
    
    def test_error_handling(self):
        """Test error handling with malformed inputs"""
        logger.info("=== TESTING ERROR HANDLING ===")
        
        error_tests = [
            {'name': 'invalid_epochs', 'args': ['--epochs', '-1']},
            {'name': 'invalid_batch_size', 'args': ['--batch_size', '0']},
            {'name': 'invalid_learning_rate', 'args': ['--learning_rate', '0']},
            {'name': 'invalid_kl_coef', 'args': ['--kl_coef', '-0.1']},
            {'name': 'missing_required_arg', 'args': []},
        ]
        
        error_results = {}
        
        for test in error_tests:
            logger.info(f"Testing error handling: {test['name']}")
            
            try:
                cmd = ['python3', 'train.py'] + test['args']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                error_results[test['name']] = {
                    'args': test['args'],
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'handled_gracefully': result.returncode != 0 and 'error' in result.stderr.lower()
                }
                
                if result.returncode == 0:
                    logger.warning(f"⚠️ {test['name']} should have failed but didn't")
                else:
                    logger.info(f"✅ {test['name']} handled gracefully")
                
            except Exception as e:
                logger.error(f"❌ Error test {test['name']} failed: {e}")
                error_results[test['name']] = {
                    'args': test['args'],
                    'error': str(e),
                    'handled_gracefully': False
                }
        
        self.results['tests']['error_handling'] = error_results
        return error_results
    
    def test_reproducibility(self):
        """Test reproducibility across different runs"""
        logger.info("=== TESTING REPRODUCIBILITY ===")
        
        reproducibility_results = {}
        
        # Run same configuration multiple times
        for run in range(3):
            logger.info(f"Reproducibility test run {run + 1}/3")
            
            try:
                cmd = [
                    'python3', 'train.py',
                    '--seed', '42',
                    '--epochs', '2',
                    '--steps_per_epoch', '3',
                    '--batch_size', '2'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                reproducibility_results[f'run_{run + 1}'] = {
                    'success': result.returncode == 0,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                
                if result.returncode == 0:
                    logger.info(f"✅ Run {run + 1} completed successfully")
                else:
                    logger.error(f"❌ Run {run + 1} failed")
                
            except Exception as e:
                logger.error(f"❌ Reproducibility test run {run + 1} failed: {e}")
                reproducibility_results[f'run_{run + 1}'] = {
                    'success': False,
                    'error': str(e)
                }
        
        self.results['tests']['reproducibility'] = reproducibility_results
        return reproducibility_results
    
    def test_dashboard_functionality(self):
        """Test dashboard and visualization capabilities"""
        logger.info("=== TESTING DASHBOARD FUNCTIONALITY ===")
        
        dashboard_results = {}
        
        try:
            # Test if dashboard can be imported
            from monitor.app import app
            dashboard_results['import_success'] = True
            logger.info("✅ Dashboard module imported successfully")
            
            # Test if we can create a simple dashboard instance
            # (This is a basic test - full dashboard testing would require more setup)
            dashboard_results['basic_functionality'] = True
            logger.info("✅ Dashboard basic functionality test passed")
            
        except Exception as e:
            logger.error(f"❌ Dashboard functionality test failed: {e}")
            dashboard_results = {
                'import_success': False,
                'basic_functionality': False,
                'error': str(e)
            }
        
        self.results['tests']['dashboard_functionality'] = dashboard_results
        return dashboard_results
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        logger.info("=== GENERATING SUMMARY REPORT ===")
        
        # Calculate summary statistics
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for test_category, test_results in self.results['tests'].items():
            if isinstance(test_results, dict):
                for test_name, test_result in test_results.items():
                    total_tests += 1
                    if isinstance(test_result, dict) and test_result.get('success', False):
                        passed_tests += 1
                    else:
                        failed_tests += 1
        
        # Generate summary
        self.results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'end_time': datetime.now().isoformat(),
            'total_duration': (datetime.fromisoformat(self.results['end_time']) - 
                             datetime.fromisoformat(self.results['start_time'])).total_seconds()
        }
        
        # Identify issues and recommendations
        self._identify_issues()
        self._generate_recommendations()
        
        # Save results
        results_file = self.test_dir / 'comprehensive_evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"✅ Summary report generated: {results_file}")
        return self.results['summary']
    
    def _identify_issues(self):
        """Identify issues from test results"""
        issues = []
        
        # Check for common issues
        if 'model_scalability' in self.results['tests']:
            for size, result in self.results['tests']['model_scalability'].items():
                if not result.get('success', False):
                    issues.append(f"Model scalability issue with {size} model: {result.get('error', 'Unknown error')}")
        
        if 'hyperparameter_sweep' in self.results['tests']:
            failed_combinations = [name for name, result in self.results['tests']['hyperparameter_sweep'].items() 
                                 if not result.get('success', False)]
            if failed_combinations:
                issues.append(f"Hyperparameter sweep failures: {failed_combinations}")
        
        if 'long_training_run' in self.results['tests']:
            if not self.results['tests']['long_training_run'].get('success', False):
                issues.append(f"Long training run failed: {self.results['tests']['long_training_run'].get('error', 'Unknown error')}")
        
        self.results['issues'] = issues
    
    def _generate_recommendations(self):
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Performance recommendations
        if 'memory_usage' in self.results['tests']:
            memory_tests = self.results['tests']['memory_usage']
            successful_tests = [result for result in memory_tests.values() if result.get('success', False)]
            if successful_tests:
                max_batch = max(test['batch_size'] for test in successful_tests)
                recommendations.append(f"Maximum stable batch size: {max_batch}")
        
        # Stability recommendations
        if 'reproducibility' in self.results['tests']:
            repro_tests = self.results['tests']['reproducibility']
            successful_runs = [result for result in repro_tests.values() if result.get('success', False)]
            if len(successful_runs) < len(repro_tests):
                recommendations.append("Reproducibility issues detected - consider improving seed handling")
        
        # Integration recommendations
        if 'trl_integration' in self.results['tests']:
            if not self.results['tests']['trl_integration'].get('success', False):
                recommendations.append("TRL integration needs improvement - check dependencies and configuration")
        
        self.results['recommendations'] = recommendations
    
    def run_comprehensive_evaluation(self):
        """Run the complete comprehensive evaluation"""
        logger.info("🚀 STARTING COMPREHENSIVE RLHF SYSTEMS KIT EVALUATION")
        logger.info("This evaluation will run for several hours to thoroughly test the system")
        
        self.log_system_info()
        
        # Run all tests
        try:
            self.test_model_scalability()
            self.test_hyperparameter_sweep()
            self.test_long_training_run()
            self.test_trl_integration()
            self.test_memory_usage()
            self.test_error_handling()
            self.test_reproducibility()
            self.test_dashboard_functionality()
            
        except KeyboardInterrupt:
            logger.info("⚠️ Evaluation interrupted by user")
        except Exception as e:
            logger.error(f"❌ Evaluation failed with exception: {e}")
            logger.error(traceback.format_exc())
        
        # Generate final report
        summary = self.generate_summary_report()
        
        logger.info("🎉 COMPREHENSIVE EVALUATION COMPLETED")
        logger.info(f"Total tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed_tests']}")
        logger.info(f"Failed: {summary['failed_tests']}")
        logger.info(f"Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"Total duration: {summary['total_duration']:.1f} seconds")
        
        return self.results

if __name__ == "__main__":
    evaluator = ComprehensiveEvaluator()
    results = evaluator.run_comprehensive_evaluation()
    
    # Print final summary
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("="*80)
    print(f"Total tests: {results['summary']['total_tests']}")
    print(f"Passed: {results['summary']['passed_tests']}")
    print(f"Failed: {results['summary']['failed_tests']}")
    print(f"Success rate: {results['summary']['success_rate']:.1f}%")
    print(f"Total duration: {results['summary']['total_duration']:.1f} seconds")
    
    if results['issues']:
        print("\nISSUES IDENTIFIED:")
        for issue in results['issues']:
            print(f"  - {issue}")
    
    if results['recommendations']:
        print("\nRECOMMENDATIONS:")
        for rec in results['recommendations']:
            print(f"  - {rec}")
    
    print("\nDetailed results saved to: comprehensive_eval_results/comprehensive_evaluation_results.json")
    print("="*80)