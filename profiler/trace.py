"""
Torch profiler integration for RLHF training.

Provides wrapper around torch.profiler.profile to capture timeline data
and NVTX ranges for training stages.
"""

import os
import json
import torch
from typing import Optional, Dict, Any, List
from contextlib import contextmanager


class TorchProfiler:
    """Wrapper around torch.profiler.profile for RLHF training stages."""
    
    def __init__(self, output_dir: str = "profiles/trace"):
        self.output_dir = output_dir
        self.profiler = None
        self.stage_stack = []
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    @contextmanager
    def profile_stage(self, stage_name: str):
        """Context manager for profiling a training stage with torch.profiler.
        
        Args:
            stage_name: Name of the stage being profiled
        
        Yields:
            None
        """
        # Add NVTX range for the stage
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(stage_name)
        
        # Add to stage stack for nested profiling
        self.stage_stack.append(stage_name)
        
        try:
            yield
        finally:
            # Remove from stage stack
            if self.stage_stack:
                self.stage_stack.pop()
            
            # End NVTX range
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
    
    def start_profiling(self, 
                        activities: Optional[List] = None,
                        schedule: Optional[torch.profiler.schedule] = None,
                        on_trace_ready: Optional[torch.profiler.tensorboard_trace_handler] = None,
                        record_shapes: bool = True,
                        profile_memory: bool = True,
                        with_stack: bool = False,
                        with_flops: bool = False,
                        with_modules: bool = False):
        """Start torch profiler with specified configuration.
        
        Args:
            activities: List of profiler activities (default: CPU and CUDA)
            schedule: Profiling schedule
            on_trace_ready: Callback for when trace is ready
            record_shapes: Whether to record tensor shapes
            profile_memory: Whether to profile memory usage
            with_stack: Whether to record stack traces
            with_flops: Whether to record FLOPS
            with_modules: Whether to record module information
        """
        if activities is None:
            activities = [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ]
        
        if schedule is None:
            schedule = torch.profiler.schedule(
                wait=0,
                warmup=0,
                active=1,
                repeat=1
            )
        
        if on_trace_ready is None:
            trace_file = os.path.join(self.output_dir, "trace.json")
            on_trace_ready = torch.profiler.tensorboard_trace_handler(
                self.output_dir,
                worker_name="rlhf_profiler"
            )
        
        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=on_trace_ready,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules
        )
        
        self.profiler.start()
        return self.profiler
    
    def stop_profiling(self):
        """Stop torch profiler and save results."""
        if self.profiler is not None:
            self.profiler.stop()
            
            # Export key metrics to CSV
            self._export_ops_csv()
            
            # Export trace data
            self._export_trace()
            
            self.profiler = None
    
    def _export_ops_csv(self):
        """Export key operator statistics to CSV."""
        if self.profiler is None:
            return
        
        try:
            # Get key metrics from profiler
            key_averages = self.profiler.key_averages()
            
            # Prepare CSV data
            csv_data = []
            for avg in key_averages:
                try:
                    # Handle different attribute names for CPU vs CUDA
                    self_time = getattr(avg, 'self_cpu_time_total', 0) / 1000
                    total_time = getattr(avg, 'cpu_time_total', 0) / 1000
                    calls = getattr(avg, 'count', 0)
                    
                    # Handle CUDA attributes safely
                    cuda_time = 0
                    cuda_calls = 0
                    if hasattr(avg, 'cuda_time_total') and avg.cuda_time_total > 0:
                        cuda_time = avg.cuda_time_total / 1000
                    if hasattr(avg, 'cuda_event_count'):
                        cuda_calls = avg.cuda_event_count
                    
                    row = {
                        'op_name': avg.key,
                        'self_time_ms': self_time,
                        'total_time_ms': total_time,
                        'calls': calls,
                        'cuda_time_ms': cuda_time,
                        'cuda_calls': cuda_calls
                    }
                    csv_data.append(row)
                except Exception as attr_error:
                    print(f"Warning: Could not process operator {avg.key}: {attr_error}")
                    continue
            
            if csv_data:
                # Write to CSV
                csv_file = os.path.join(self.output_dir, "ops.csv")
                with open(csv_file, 'w') as f:
                    # Write header
                    f.write("op_name,self_time_ms,total_time_ms,calls,cuda_time_ms,cuda_calls\n")
                    
                    # Write data
                    for row in csv_data:
                        f.write(f"{row['op_name']},{row['self_time_ms']:.3f},"
                               f"{row['total_time_ms']:.3f},{row['calls']},"
                               f"{row['cuda_time_ms']:.3f},{row['cuda_calls']}\n")
                
                print(f"Exported operator statistics to {csv_file}")
            else:
                print("Warning: No operator statistics available to export")
            
        except Exception as e:
            print(f"Warning: Could not export operator statistics: {e}")
            # Create a minimal ops.csv with placeholder data
            try:
                csv_file = os.path.join(self.output_dir, "ops.csv")
                with open(csv_file, 'w') as f:
                    f.write("op_name,self_time_ms,total_time_ms,calls,cuda_time_ms,cuda_calls\n")
                    f.write("placeholder,0.0,0.0,0,0.0,0\n")
                print(f"Created placeholder ops.csv: {csv_file}")
            except Exception as placeholder_error:
                print(f"Could not create placeholder ops.csv: {placeholder_error}")
    
    def _export_trace(self):
        """Export trace data in Chrome trace format."""
        if self.profiler is None:
            return
        
        try:
            # The tensorboard_trace_handler should have already created the trace
            # Check if trace file exists
            trace_files = [f for f in os.listdir(self.output_dir) if f.endswith('.json')]
            
            if trace_files:
                # Find the most recent trace file
                trace_file = max(trace_files, key=lambda x: os.path.getmtime(os.path.join(self.output_dir, x)))
                trace_path = os.path.join(self.output_dir, trace_file)
                
                # Create a symlink to trace.json for consistency
                symlink_path = os.path.join(self.output_dir, "trace.json")
                if os.path.exists(symlink_path):
                    os.remove(symlink_path)
                
                try:
                    os.symlink(trace_file, symlink_path)
                    print(f"Created trace symlink: {symlink_path} -> {trace_file}")
                except OSError:
                    # On Windows or if symlink fails, copy the file
                    import shutil
                    shutil.copy2(trace_path, symlink_path)
                    print(f"Copied trace file: {symlink_path}")
                
            else:
                print("Warning: No trace files found in output directory")
                
        except Exception as e:
            print(f"Warning: Could not export trace data: {e}")
    
    def step(self):
        """Step the profiler (call this after each training step)."""
        if self.profiler is not None:
            self.profiler.step()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_profiling()


def create_torch_profiler(output_dir: str = "profiles/trace") -> TorchProfiler:
    """Create a torch profiler instance.
    
    Args:
        output_dir: Directory to save trace files
    
    Returns:
        TorchProfiler instance
    """
    return TorchProfiler(output_dir)