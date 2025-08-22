# api/__init__.py
"""API package for OpenRouter client functionality.

This package provides OpenRouter-specific API client operations including:
- OpenRouter client functions (openrouter_client.py)
- Model configurations and scoring settings (config.py)
- Model capability detection and enrichment utilities
"""

from .openrouter_client import (
    get_models, model_supports_reasoning, model_supports_structured_outputs,
    chat_one, CallMetadata
)
from .config import (
    SCORING_CONFIG, MODEL_CONFIGS, OUTPUT_FORMAT,
    resolve_judge_overrides, enrich_model_configs,
    annotate_reasoning_support, annotate_reasoning_support_inplace
)

# Export main functions
__all__ = [
    # OpenRouter client
    'get_models', 'model_supports_reasoning', 'model_supports_structured_outputs',
    'chat_one', 'CallMetadata',
    
    # Configuration
    'SCORING_CONFIG', 'MODEL_CONFIGS', 'OUTPUT_FORMAT',
    'resolve_judge_overrides', 'enrich_model_configs',
    'annotate_reasoning_support', 'annotate_reasoning_support_inplace'
]
