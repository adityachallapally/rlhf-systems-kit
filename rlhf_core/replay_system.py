"""
Flexible Replay System for RLHF Training

This module provides a robust replay system that can handle various data formats
and automatically selects the appropriate adapter with fallback mechanisms.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import glob

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
    
    np = BasicNumpy()


@dataclass
class ReplayConfig:
    """Configuration for replay system."""
    
    # Adapter priorities (higher number = higher priority)
    adapter_priorities: Dict[str, int] = field(default_factory=lambda: {
        'trl': 100,
        'huggingface': 90,
        'json': 80,
        'jsonl': 70,
        'csv': 60,
        'generic': 50
    })
    
    # File format detection patterns
    file_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        'trl': ['*.pt', '*.pth', '*.bin', '*.safetensors'],
        'huggingface': ['*.json', '*.jsonl', '*.txt'],
        'json': ['*.json'],
        'jsonl': ['*.jsonl'],
        'csv': ['*.csv'],
        'generic': ['*']
    })
    
    # Fallback strategies
    fallback_strategies: List[str] = field(default_factory=lambda: [
        'try_all_adapters',
        'use_generic_adapter',
        'create_mock_data'
    ])
    
    # Error handling
    strict_mode: bool = False
    log_adapter_selection: bool = True
    create_mock_data_on_failure: bool = True


class DataAdapter(ABC):
    """Abstract base class for data adapters."""
    
    def __init__(self, name: str, config: ReplayConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"replay.adapter.{name}")
    
    @abstractmethod
    def can_handle(self, source: Union[str, Path, Dict[str, Any]]) -> bool:
        """Check if this adapter can handle the given source."""
        pass
    
    @abstractmethod
    def load_data(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Load data from the source."""
        pass
    
    @abstractmethod
    def get_metadata(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Get metadata about the data source."""
        pass


class JSONAdapter(DataAdapter):
    """Adapter for JSON data files."""
    
    def __init__(self, config: ReplayConfig):
        super().__init__('json', config)
    
    def can_handle(self, source: Union[str, Path, Dict[str, Any]]) -> bool:
        """Check if this is a JSON file."""
        if isinstance(source, (str, Path)):
            return Path(source).suffix == '.json'
        return False
    
    def load_data(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Load JSON data."""
        try:
            with open(source, 'r') as f:
                data = json.load(f)
            
            return {
                'type': 'json',
                'path': str(source),
                'data': data,
                'metadata': {
                    'size': os.path.getsize(source),
                    'keys': list(data.keys()) if isinstance(data, dict) else None
                }
            }
        except Exception as e:
            self.logger.error(f"Error loading JSON data: {e}")
            raise
    
    def get_metadata(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Get metadata about JSON data."""
        metadata = {
            'adapter': 'json',
            'type': 'file',
            'size': 0
        }
        
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            metadata['path'] = str(source_path)
            metadata['size'] = source_path.stat().st_size if source_path.exists() else 0
        
        return metadata


class JSONLAdapter(DataAdapter):
    """Adapter for JSONL data files."""
    
    def __init__(self, config: ReplayConfig):
        super().__init__('jsonl', config)
    
    def can_handle(self, source: Union[str, Path, Dict[str, Any]]) -> bool:
        """Check if this is a JSONL file."""
        if isinstance(source, (str, Path)):
            return Path(source).suffix == '.jsonl'
        return False
    
    def load_data(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Load JSONL data."""
        try:
            data = []
            with open(source, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Invalid JSON at line {line_num}: {e}")
                            continue
            
            return {
                'type': 'jsonl',
                'path': str(source),
                'data': data,
                'metadata': {
                    'size': os.path.getsize(source),
                    'lines': len(data)
                }
            }
        except Exception as e:
            self.logger.error(f"Error loading JSONL data: {e}")
            raise
    
    def get_metadata(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Get metadata about JSONL data."""
        metadata = {
            'adapter': 'jsonl',
            'type': 'file',
            'size': 0,
            'lines': 0
        }
        
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            metadata['path'] = str(source_path)
            metadata['size'] = source_path.stat().st_size if source_path.exists() else 0
            
            # Count lines
            try:
                with open(source_path, 'r') as f:
                    metadata['lines'] = sum(1 for _ in f)
            except:
                pass
        
        return metadata


class GenericAdapter(DataAdapter):
    """Generic adapter for unknown data formats."""
    
    def __init__(self, config: ReplayConfig):
        super().__init__('generic', config)
    
    def can_handle(self, source: Union[str, Path, Dict[str, Any]]) -> bool:
        """Generic adapter can handle any source as fallback."""
        return True
    
    def load_data(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Load data using generic methods."""
        try:
            if isinstance(source, (str, Path)):
                source_path = Path(source)
                
                if source_path.is_dir():
                    return self._load_directory_generic(source_path)
                else:
                    return self._load_file_generic(source_path)
            else:
                return {
                    'type': 'generic_dict',
                    'data': source,
                    'metadata': {'source_type': 'dict'}
                }
        except Exception as e:
            self.logger.error(f"Error loading data with generic adapter: {e}")
            raise
    
    def _load_directory_generic(self, directory: Path) -> Dict[str, Any]:
        """Load directory using generic methods."""
        data = {
            'type': 'generic_directory',
            'path': str(directory),
            'files': [],
            'data': {}
        }
        
        # List all files
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                data['files'].append(str(file_path.relative_to(directory)))
        
        # Try to load common file types
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                try:
                    if file_path.suffix == '.json':
                        with open(file_path, 'r') as f:
                            data['data'][str(file_path.relative_to(directory))] = json.load(f)
                    elif file_path.suffix == '.txt':
                        with open(file_path, 'r') as f:
                            data['data'][str(file_path.relative_to(directory))] = f.read()
                except Exception as e:
                    self.logger.warning(f"Could not load {file_path}: {e}")
        
        return data
    
    def _load_file_generic(self, file_path: Path) -> Dict[str, Any]:
        """Load file using generic methods."""
        data = {
            'type': 'generic_file',
            'path': str(file_path),
            'data': None,
            'metadata': {
                'size': file_path.stat().st_size if file_path.exists() else 0,
                'extension': file_path.suffix
            }
        }
        
        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data['data'] = json.load(f)
            elif file_path.suffix == '.txt':
                with open(file_path, 'r') as f:
                    data['data'] = f.read()
            else:
                # Try to read as text
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    data['data'] = f.read()
        except Exception as e:
            self.logger.warning(f"Could not load {file_path}: {e}")
            data['data'] = None
        
        return data
    
    def get_metadata(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Get metadata about generic data."""
        metadata = {
            'adapter': 'generic',
            'type': 'unknown',
            'size': 0
        }
        
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            metadata['path'] = str(source_path)
            metadata['type'] = 'directory' if source_path.is_dir() else 'file'
            metadata['size'] = source_path.stat().st_size if source_path.exists() else 0
        
        return metadata


class ReplaySystem:
    """Main replay system with flexible adapter selection."""
    
    def __init__(self, config: Optional[ReplayConfig] = None):
        self.config = config or ReplayConfig()
        self.logger = logging.getLogger("replay_system")
        
        # Initialize adapters
        self.adapters = {
            'json': JSONAdapter(self.config),
            'jsonl': JSONLAdapter(self.config),
            'generic': GenericAdapter(self.config)
        }
    
    def replay(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Replay data from the given source."""
        self.logger.info(f"Starting replay for source: {source}")
        
        # Try to find the best adapter
        adapter_name, adapter = self._select_adapter(source)
        
        if adapter_name is None:
            return self._handle_no_adapter_found(source)
        
        try:
            # Load data using selected adapter
            data = adapter.load_data(source)
            metadata = adapter.get_metadata(source)
            
            result = {
                'status': 'success',
                'adapter_used': adapter_name,
                'source': str(source),
                'data': data,
                'metadata': metadata,
                'timestamp': self._get_timestamp()
            }
            
            self.logger.info(f"Successfully replayed data using {adapter_name} adapter")
            return result
            
        except Exception as e:
            self.logger.error(f"Error replaying data with {adapter_name} adapter: {e}")
            return self._handle_replay_error(source, adapter_name, str(e))
    
    def _select_adapter(self, source: Union[str, Path, Dict[str, Any]]) -> tuple:
        """Select the best adapter for the given source."""
        # Sort adapters by priority
        sorted_adapters = sorted(
            self.adapters.items(),
            key=lambda x: self.config.adapter_priorities.get(x[0], 0),
            reverse=True
        )
        
        for adapter_name, adapter in sorted_adapters:
            try:
                if adapter.can_handle(source):
                    if self.config.log_adapter_selection:
                        self.logger.info(f"Selected adapter: {adapter_name}")
                    return adapter_name, adapter
            except Exception as e:
                self.logger.warning(f"Error checking adapter {adapter_name}: {e}")
                continue
        
        return None, None
    
    def _handle_no_adapter_found(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Handle case when no adapter can handle the source."""
        error_msg = f"No suitable adapter found for source: {source}"
        
        if self.config.strict_mode:
            raise ValueError(error_msg)
        
        self.logger.error(error_msg)
        
        # Try fallback strategies
        for strategy in self.config.fallback_strategies:
            if strategy == 'use_generic_adapter':
                try:
                    generic_adapter = self.adapters['generic']
                    data = generic_adapter.load_data(source)
                    return {
                        'status': 'success_fallback',
                        'adapter_used': 'generic',
                        'source': str(source),
                        'data': data,
                        'metadata': generic_adapter.get_metadata(source),
                        'timestamp': self._get_timestamp(),
                        'fallback_reason': 'generic_adapter'
                    }
                except Exception as e:
                    self.logger.warning(f"Generic adapter failed: {e}")
                    continue
            
            elif strategy == 'create_mock_data':
                if self.config.create_mock_data_on_failure:
                    return self._create_mock_data(source)
        
        return {
            'status': 'error',
            'source': str(source),
            'error': error_msg,
            'timestamp': self._get_timestamp()
        }
    
    def _handle_replay_error(self, source: Union[str, Path, Dict[str, Any]], 
                           adapter_name: str, error: str) -> Dict[str, Any]:
        """Handle replay errors."""
        error_msg = f"Error replaying data with {adapter_name} adapter: {error}"
        
        if self.config.strict_mode:
            raise ValueError(error_msg)
        
        self.logger.error(error_msg)
        
        # Try fallback strategies
        for strategy in self.config.fallback_strategies:
            if strategy == 'create_mock_data':
                if self.config.create_mock_data_on_failure:
                    return self._create_mock_data(source)
        
        return {
            'status': 'error',
            'source': str(source),
            'adapter_used': adapter_name,
            'error': error_msg,
            'timestamp': self._get_timestamp()
        }
    
    def _create_mock_data(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Create mock data when all adapters fail."""
        mock_data = {
            'type': 'mock_data',
            'source': str(source),
            'data': {
                'message': 'Mock data created due to adapter failure',
                'original_source': str(source),
                'sample_metrics': {
                    'throughput': 100.0,
                    'toxicity': 0.1,
                    'bias': 0.05
                }
            },
            'metadata': {
                'adapter': 'mock',
                'created_at': self._get_timestamp(),
                'reason': 'adapter_failure'
            }
        }
        
        self.logger.warning(f"Created mock data for source: {source}")
        
        return {
            'status': 'mock_data',
            'adapter_used': 'mock',
            'source': str(source),
            'data': mock_data,
            'metadata': mock_data['metadata'],
            'timestamp': self._get_timestamp(),
            'fallback_reason': 'mock_data_created'
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_available_adapters(self) -> List[str]:
        """Get list of available adapters."""
        return list(self.adapters.keys())
    
    def get_adapter_info(self, adapter_name: str) -> Dict[str, Any]:
        """Get information about a specific adapter."""
        if adapter_name not in self.adapters:
            return {'error': f'Adapter {adapter_name} not found'}
        
        adapter = self.adapters[adapter_name]
        return {
            'name': adapter_name,
            'priority': self.config.adapter_priorities.get(adapter_name, 0),
            'file_patterns': self.config.file_patterns.get(adapter_name, []),
            'description': f'{adapter_name} adapter for data replay'
        }
    
    def validate_source(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Validate if a source can be handled by any adapter."""
        validation_results = {
            'can_handle': False,
            'suitable_adapters': [],
            'recommendations': []
        }
        
        for adapter_name, adapter in self.adapters.items():
            try:
                if adapter.can_handle(source):
                    validation_results['suitable_adapters'].append(adapter_name)
                    validation_results['can_handle'] = True
            except Exception as e:
                self.logger.warning(f"Error validating with {adapter_name}: {e}")
        
        if not validation_results['can_handle']:
            validation_results['recommendations'].append(
                "Try converting data to a supported format (JSON, JSONL)"
            )
            validation_results['recommendations'].append(
                "Use the generic adapter as a fallback"
            )
        
        return validation_results


def create_replay_system_example():
    """Create an example of the replay system usage."""
    
    # Create configuration
    config = ReplayConfig(
        strict_mode=False,
        log_adapter_selection=True,
        create_mock_data_on_failure=True
    )
    
    # Initialize replay system
    replay_system = ReplaySystem(config)
    
    # Example sources to test
    test_sources = [
        "./test_data/sample.json",   # JSON file
        "./test_data/sample.jsonl",  # JSONL file
    ]
    
    results = []
    for source in test_sources:
        print(f"\nTesting source: {source}")
        result = replay_system.replay(source)
        results.append(result)
        
        print(f"Status: {result['status']}")
        if 'adapter_used' in result:
            print(f"Adapter: {result['adapter_used']}")
        if 'error' in result:
            print(f"Error: {result['error']}")
    
    return replay_system, results


if __name__ == "__main__":
    # Run example
    replay_system, results = create_replay_system_example()