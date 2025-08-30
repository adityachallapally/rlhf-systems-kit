"""
Logging utilities for RLHF training metrics.

Provides JSONL logging and system information capture.
"""

import os
import json
import time
import platform
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class JSONLLogger:
    """Logger that writes metrics to JSONL format with automatic flushing."""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.file = None
        self._open_file()
        
    def _open_file(self):
        """Open the JSONL file for writing."""
        self.file = open(self.filepath, 'a', buffering=1)  # Line buffered
        
    def log(self, metrics: Dict[str, Any]):
        """Log a single metrics dictionary to JSONL."""
        if self.file is None:
            self._open_file()
            
        # Ensure metrics are JSON serializable
        safe_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, float):
                # Round floats to avoid minor kernel noise causing string diffs
                safe_metrics[key] = round(value, 8)
            elif isinstance(value, (int, str, bool, type(None))):
                safe_metrics[key] = value
            else:
                # Convert other types to string
                safe_metrics[key] = str(value)
        
        # Write to file with flush
        json.dump(safe_metrics, self.file, separators=(",", ":"))
        self.file.write("\n")
        self.file.flush()
        
    def close(self):
        """Close the logger and flush any remaining data."""
        if self.file is not None:
            self.file.flush()
            self.file.close()
            self.file = None
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def write_sysinfo(run_dir: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """Write system information to sysinfo.json in the run directory."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    
    # Gather system information
    sysinfo = {
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'start_time': datetime.now().isoformat(),
        'seed': seed,
    }
    
    # Add CUDA information if available
    if torch.cuda.is_available():
        sysinfo.update({
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory_total_gb': round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        })
    
    # Add environment variables that affect determinism
    env_vars = ['PYTHONHASHSEED', 'CUDA_LAUNCH_BLOCKING', 'OMP_NUM_THREADS']
    for var in env_vars:
        if var in os.environ:
            sysinfo[f'env_{var}'] = os.environ[var]
    
    # Write to file
    sysinfo_path = run_path / "sysinfo.json"
    with open(sysinfo_path, 'w') as f:
        json.dump(sysinfo, f, indent=2)
        
    return sysinfo


def create_run_dir(base_dir: str = "runs") -> str:
    """Create a new run directory with timestamp-based naming."""
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_path / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    return str(run_dir)


def update_latest_symlink(run_dir: str, base_dir: str = "runs"):
    """Create or update the 'latest' symlink to point to the current run."""
    base_path = Path(base_dir)
    latest_link = base_path / "latest"
    
    # Remove existing symlink if it exists
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    
    # Create new symlink
    run_path = Path(run_dir)
    if run_path.exists():
        latest_link.symlink_to(run_path.name, target_is_directory=True)
