# data/__init__.py
"""Data handling package for sycophancy analysis.

This package provides modular data operations including:
- Prompt generation (prompts.py)
- Response collection (collection.py) 
- Data persistence and loading (persistence.py, loading.py)
- Metadata generation (metadata.py)
"""

from .prompts import build_sycophancy_battery
from .collection import collect_responses
from .persistence import (
    save_responses, save_sss, save_vectors, save_matrices, 
    save_scored_rows, save_metadata
)
from .loading import (
    load_all_responses, get_latest_responses, load_sss, 
    load_vectors, load_matrices, load_metadata
)
from .metadata import create_comprehensive_metadata

# Export main functions
__all__ = [
    # Prompt generation
    'build_sycophancy_battery',
    
    # Response collection
    'collect_responses',
    
    # Data persistence
    'save_responses', 'save_sss', 'save_vectors', 'save_matrices', 
    'save_scored_rows', 'save_metadata',
    
    # Data loading
    'load_all_responses', 'get_latest_responses', 'load_sss', 
    'load_vectors', 'load_matrices', 'load_metadata',
    
    # Metadata generation
    'create_comprehensive_metadata'
]
