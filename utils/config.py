"""
Configuration management for the GALADRIEL pipeline.
Handles loading and validation of configuration files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

from .types import LLMProvider, LLMConfig


class DataConfig(BaseModel):
    """Configuration for data processing."""
    xes_file_path: str
    output_dir: str = "data/processed"
    max_trace_length: Optional[int] = None
    min_trace_length: int = 1


class BatchProcessingConfig(BaseModel):
    """Configuration for batch processing optimization."""
    enabled: bool = True
    batch_size: int = 5
    checkpoint_every: int = 100
    async_pipeline: bool = True


class OllamaOptimizationConfig(BaseModel):
    """Configuration for Ollama optimization."""
    keep_alive: str = "24h"
    num_thread: str = "auto"
    num_parallel: int = 1
    numa: bool = False
    num_ctx: int = 4096


class NarrativeConfig(BaseModel):
    """Configuration for story generation."""
    template_path: str = "config/narrative_templates.yaml"
    include_timestamps: bool = True
    include_patient_metadata: bool = True
    max_story_length: int = 2000
    batch_processing: BatchProcessingConfig = BatchProcessingConfig()
    ollama_optimization: OllamaOptimizationConfig = OllamaOptimizationConfig()


class TranslationConfig(BaseModel):
    """Configuration for translation."""
    source_language: str = "it"
    target_language: str = "en"
    llm_config: LLMConfig


# ClassificationConfig removed - use src.classification.config.ClassificationConfig instead


class ExplainabilityConfig(BaseModel):
    """Configuration for explainability."""
    method: str = "integrated_gradients"
    n_steps: int = 50
    visualize_top_k: int = 20


class GALADRIELConfig(BaseModel):
    """Main configuration class for the GALADRIEL pipeline."""
    data: DataConfig
    narrative: NarrativeConfig
    translation: TranslationConfig
    classification: Optional[Dict[str, Any]] = None  # ClassificationConfig removed - use src.classification.config.ClassificationConfig instead
    explainability: ExplainabilityConfig
    
    # General settings
    random_seed: int = 42
    verbose: bool = True
    use_gpu: bool = True


def load_config(config_path: str = "config/config.yaml") -> GALADRIELConfig:
    """
    Load configuration from YAML file and environment variables.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        Validated GALADRIELConfig object
        
    Raises:
        FileNotFoundError: If config file is not found
        ValidationError: If config validation fails
    """
    # Load environment variables
    load_dotenv()
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML configuration
    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Override with environment variables if present
    _override_with_env_vars(config_dict)
    
    try:
        return GALADRIELConfig(**config_dict)
    except ValidationError as e:
        raise ValidationError(f"Configuration validation failed: {e}")


def _override_with_env_vars(config_dict: Dict[str, Any]) -> None:
    """Override configuration values with environment variables."""
    
    # API keys
    if os.getenv("GEMINI_API_KEY"):
        if "translation" in config_dict and "llm_config" in config_dict["translation"]:
            config_dict["translation"]["llm_config"]["api_key"] = os.getenv("GEMINI_API_KEY")
        if "classification" in config_dict and "llm_config" in config_dict["classification"]:
            config_dict["classification"]["llm_config"]["api_key"] = os.getenv("GEMINI_API_KEY")
    
    if os.getenv("OPENAI_API_KEY"):
        # Similar logic for OpenAI
        pass
    
    # File paths
    if os.getenv("XES_FILE_PATH"):
        if "data" in config_dict:
            config_dict["data"]["xes_file_path"] = os.getenv("XES_FILE_PATH")
    
    # Other environment overrides can be added here


def get_default_config() -> GALADRIELConfig:
    """
    Get a default configuration for testing or quick setup.
    
    Returns:
        Default GALADRIELConfig object
    """
    default_llm_config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="mistral:7b",
        base_url="http://localhost:11434",
        max_tokens=2048,
        temperature=0.1
    )
    
    return GALADRIELConfig(
        data=DataConfig(
            xes_file_path="data/ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025.xes"
        ),
        narrative=NarrativeConfig(),
        translation=TranslationConfig(llm_config=default_llm_config),
        classification=None, # ClassificationConfig removed - use src.classification.config.ClassificationConfig instead
        explainability=ExplainabilityConfig()
    )
