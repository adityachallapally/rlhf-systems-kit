"""
Tests for logging and timing functionality.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
import pandas as pd

from rlhf_core.logging import JSONLLogger, write_sysinfo
from rlhf_core.profiler import stage_timer


class TestJSONLLogger:
    """Test JSONLLogger functionality."""
    
    def test_jsonl_logger_creation(self):
        """Test JSONLLogger can be created and writes to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.jsonl")
            logger = JSONLLogger(log_file)
            
            # Test logging
            test_metrics = {
                'step': 1,
                'loss': 0.5,
                'reward_mean': 0.8,
                'kl': 0.1
            }
            logger.log(test_metrics)
            logger.close()
            
            # Verify file was created and contains data
            assert os.path.exists(log_file)
            
            with open(log_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1
                
                # Parse the logged JSON
                logged_data = json.loads(lines[0])
                assert logged_data['step'] == 1
                assert logged_data['loss'] == 0.5
                assert logged_data['reward_mean'] == 0.8
                assert logged_data['kl'] == 0.1
    
    def test_jsonl_logger_context_manager(self):
        """Test JSONLLogger works as a context manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.jsonl")
            
            with JSONLLogger(log_file) as logger:
                logger.log({'step': 1, 'test': 'value'})
                logger.log({'step': 2, 'test': 'value2'})
            
            # Verify file was created and contains data
            assert os.path.exists(log_file)
            
            with open(log_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 2
    
    def test_jsonl_logger_float_rounding(self):
        """Test that floats are properly rounded to avoid minor differences."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.jsonl")
            
            with JSONLLogger(log_file) as logger:
                # Log a float with many decimal places
                logger.log({'step': 1, 'value': 0.123456789012345})
            
            # Verify the logged value is rounded
            with open(log_file, 'r') as f:
                logged_data = json.loads(f.readline())
                assert logged_data['value'] == 0.12345679  # Should be rounded to 8 decimal places


class TestStageTimer:
    """Test stage timing functionality."""
    
    def test_stage_timer_basic(self):
        """Test basic stage timer functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with stage_timer("test_stage", temp_dir):
                # Simulate some work
                import time
                time.sleep(0.1)
            
            # Verify stage_times.json was created
            stage_file = os.path.join(temp_dir, "stage_times.json")
            assert os.path.exists(stage_file)
            
            # Verify content
            with open(stage_file, 'r') as f:
                stage_data = json.load(f)
                assert len(stage_data) == 1
                assert stage_data[0]['stage'] == 'test_stage'
                assert stage_data[0]['seconds'] > 0.09  # Should be at least 0.1 seconds
                assert 'device' in stage_data[0]
    
    def test_stage_timer_multiple_stages(self):
        """Test that multiple stages can be timed and appended."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First stage
            with stage_timer("stage1", temp_dir):
                import time
                time.sleep(0.05)
            
            # Second stage
            with stage_timer("stage2", temp_dir):
                time.sleep(0.05)
            
            # Verify both stages are recorded
            stage_file = os.path.join(temp_dir, "stage_times.json")
            with open(stage_file, 'r') as f:
                stage_data = json.load(f)
                assert len(stage_data) == 2
                assert stage_data[0]['stage'] == 'stage1'
                assert stage_data[1]['stage'] == 'stage2'


class TestSystemInfo:
    """Test system information functionality."""
    
    def test_write_sysinfo(self):
        """Test that system information is written correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sysinfo = write_sysinfo(temp_dir, seed=42)
            
            # Verify sysinfo.json was created
            sysinfo_file = os.path.join(temp_dir, "sysinfo.json")
            assert os.path.exists(sysinfo_file)
            
            # Verify content
            with open(sysinfo_file, 'r') as f:
                loaded_sysinfo = json.load(f)
                assert loaded_sysinfo['seed'] == 42
                assert 'python_version' in loaded_sysinfo
                assert 'torch_version' in loaded_sysinfo
                assert 'device' in loaded_sysinfo
                assert 'start_time' in loaded_sysinfo


class TestMetricsSchema:
    """Test that metrics follow the expected schema."""
    
    def test_metrics_schema_validation(self):
        """Test that logged metrics follow the expected schema."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "metrics.jsonl")
            
            with JSONLLogger(log_file) as logger:
                # Log metrics in the expected format
                metrics = {
                    'step': 1,
                    'phase': 'ppo_update',
                    'loss': 0.5,
                    'reward_mean': 0.8,
                    'reward_var': 0.1,
                    'kl': 0.05,
                    'entropy': 2.0,
                    'clip_frac': 0.1,
                    'grad_norm': 1.0,
                    'lr': 1e-5,
                    'time_ms': 100.0,
                    'seed': 42,
                    'run_id': 'test_run'
                }
                logger.log(metrics)
            
            # Verify the file can be loaded as a DataFrame
            df = pd.read_json(log_file, lines=True)
            assert len(df) == 1
            
            # Verify all expected columns are present
            expected_columns = {
                'step', 'phase', 'loss', 'reward_mean', 'reward_var', 
                'kl', 'entropy', 'clip_frac', 'grad_norm', 'lr', 
                'time_ms', 'seed', 'run_id'
            }
            assert set(df.columns) == expected_columns
            
            # Verify data types
            assert df['step'].dtype in ['int64', 'int32']
            assert df['phase'].dtype == 'object'  # string
            assert df['loss'].dtype in ['float64', 'float32']
            assert df['seed'].dtype in ['int64', 'int32']


if __name__ == "__main__":
    pytest.main([__file__])
