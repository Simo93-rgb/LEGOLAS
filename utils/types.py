"""
Type definitions for the LEGOLAS pipeline.
Defines all custom types used across the project for better type safety.
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Literal
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ClassificationTarget(Enum):
    """Classification target values."""
    ADMITTED = 1
    DISCHARGED = 0


class LLMProvider(Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    GEMINI = "gemini"
    OPENAI = "openai"


@dataclass
class Event:
    """Represents a single event in a patient trace."""
    activity: str
    timestamp: datetime
    case_id: str
    additional_attributes: Optional[Dict[str, Any]] = None


@dataclass
class PatientTrace:
    """Represents a complete patient trace with events and metadata."""
    case_id: str
    events: List[Event]
    classification: Optional[ClassificationTarget] = None
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    additional_metadata: Optional[Dict[str, Any]] = None


@dataclass
class PatientStory:
    """Represents a patient story in natural language."""
    case_id: str
    story_text: str
    language: str  # Changed from Literal to str for flexibility
    classification: Optional[ClassificationTarget] = None
    original_trace: Optional[PatientTrace] = None


@dataclass
class ClassificationResult:
    """Result of LLM classification."""
    case_id: str
    predicted_class: ClassificationTarget
    confidence: float
    attention_weights: Optional[Dict[str, float]] = None
    explanation: Optional[str] = None


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 3072
    temperature: float = 0.0
    additional_params: Optional[Dict[str, Any]] = None


# Type aliases for better readability
TraceDict = Dict[str, List[Event]]
StoryDict = Dict[str, PatientStory]
ClassificationResults = List[ClassificationResult]
