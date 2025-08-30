"""
Profiling and timing utilities for RLHF training.

Provides context managers for profiling with torch.profiler and timing training stages.
"""

import os
import json
import time
import torch
import csv
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
from pathlib import Path


class ProfilerManager:
    """Context manager for PyTorch profiling with trace and stats output."""
    
    def __init__(self, run_dir: str, enabled: bool = True):
        self.run_dir = Path(run_dir)
        self.enabled = enabled
        self.profiler = None
        self.trace_path = self.run_dir / "trace.json"
        self.op_stats_path = self.run_dir / "op_stats.csv"
        
    def __enter__(self):
        if not self.enabled:
            return self
            
        # Create run directory if it doesn't exist
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure profiler
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=0,
                active=1,
                repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.run_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        self.profiler.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler is not None:
            try:
                self.profiler.stop()
                self._export_trace()
                self._export_op_stats()
            except Exception as e:
                # Log error but don't fail the training
                print(f"Warning: Profiler export failed: {e}")
            
    def _export_trace(self):
        """Export chrome trace to JSON file."""
        if self.profiler is not None:
            try:
                # Export chrome trace
                self.profiler.export_chrome_trace(str(self.trace_path))
            except Exception as e:
                print(f"Warning: Failed to export trace: {e}")
                
        # Also try to find and copy any trace files that were created
        try:
            trace_files = list(self.run_dir.glob("*.trace.json"))
            if trace_files:
                # Copy the first trace file to our expected location
                import shutil
                shutil.copy2(trace_files[0], self.trace_path)
                print(f"Copied trace file: {trace_files[0]} -> {self.trace_path}")
        except Exception as e:
            print(f"Warning: Failed to copy trace file: {e}")
            
    def _export_op_stats(self):
        """Extract and export operation statistics to CSV."""
        if self.profiler is None:
            return
            
        try:
            # Get profiler events
            events = self.profiler.key_averages()
            
            # Aggregate stats by operation name
            op_stats: Dict[str, Dict[str, Any]] = {}
            
            for event in events:
                name = event.key
                if name not in op_stats:
                    op_stats[name] = {
                        'name': name,
                        'cpu_time_total_us': 0,
                        'cuda_time_total_us': 0,
                        'cpu_time_avg_us': 0,
                        'cuda_time_avg_us': 0,
                        'calls': 0
                    }
                
                stats = op_stats[name]
                stats['calls'] += 1
                stats['cpu_time_total_us'] += event.cpu_time_total
                # Handle case where CUDA time might not be available
                if hasattr(event, 'cuda_time_total'):
                    stats['cuda_time_total_us'] += event.cuda_time_total
                
            # Calculate averages
            for stats in op_stats.values():
                if stats['calls'] > 0:
                    stats['cpu_time_avg_us'] = stats['cpu_time_total_us'] / stats['calls']
                    stats['cuda_time_avg_us'] = stats['cuda_time_total_us'] / stats['calls']
            
            # Write to CSV
            with open(self.op_stats_path, 'w', newline='') as f:
                if op_stats:
                    fieldnames = ['name', 'cpu_time_total_us', 'cuda_time_total_us', 
                                'cpu_time_avg_us', 'cuda_time_avg_us', 'calls']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for stats in op_stats.values():
                        writer.writerow(stats)
        except Exception as e:
            print(f"Warning: Failed to export op stats: {e}")


class StageTimer:
    """Context manager for timing training stages and recording memory usage."""
    
    def __init__(self, name: str, run_dir: str):
        self.name = name
        self.run_dir = Path(run_dir)
        self.start_time = None
        self.start_memory = None
        self.stage_times_path = self.run_dir / "stage_times.json"
        
    def __enter__(self):
        self.start_time = time.time()
        
        # Record initial memory state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.start_memory = torch.cuda.memory_allocated()
        else:
            self.start_memory = 0
            
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Record peak memory
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            device = "cuda"
        else:
            peak_memory = 0
            device = "cpu"
            
        # Load existing stage times or create new
        stage_times = []
        if self.stage_times_path.exists():
            try:
                with open(self.stage_times_path, 'r') as f:
                    stage_times = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                stage_times = []
        
        # Add new stage record
        stage_record = {
            'stage': self.name,
            'seconds': round(duration, 6),
            'peak_mem_bytes': peak_memory,
            'device': device
        }
        stage_times.append(stage_record)
        
        # Write updated stage times
        with open(self.stage_times_path, 'w') as f:
            json.dump(stage_times, f, indent=2)


def stage_timer(name: str, run_dir: str):
    """Convenience function for stage timing."""
    return StageTimer(name, run_dir)
