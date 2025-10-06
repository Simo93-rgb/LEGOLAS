"""
Config module per LEGOLAS
"""

from .paths import (
    PROJECT_ROOT,
    DATA_DIR,
    OUTPUT_DIR,
    STORIES_DIR,
    MODELS_DIR,
    EVALUATION_DIR,
    PREDICTION_DIR,
    MODEL_CONFIG_FILE,
    TRANSLATION_CACHE_FILE,
    get_story_file_path,
    get_model_path,
    get_prediction_path,
    ensure_directories,
)

__all__ = [
    'PROJECT_ROOT',
    'DATA_DIR',
    'OUTPUT_DIR',
    'STORIES_DIR',
    'MODELS_DIR',
    'EVALUATION_DIR',
    'PREDICTION_DIR',
    'MODEL_CONFIG_FILE',
    'TRANSLATION_CACHE_FILE',
    'get_story_file_path',
    'get_model_path',
    'get_prediction_path',
    'ensure_directories',
]
