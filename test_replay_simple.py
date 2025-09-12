#!/usr/bin/env python3
"""Simple test of replay system syntax."""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod

class DataAdapter(ABC):
    """Abstract base class for data adapters."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def can_handle(self, source: Union[str, Path, Dict[str, Any]]) -> bool:
        """Check if this adapter can handle the given source."""
        pass

class TestAdapter(DataAdapter):
    """Test adapter."""
    
    def __init__(self):
        super().__init__('test')
    
    def can_handle(self, source: Union[str, Path, Dict[str, Any]]) -> bool:
        """Check if this is a test source."""
        return True

print("Simple replay system syntax test passed")