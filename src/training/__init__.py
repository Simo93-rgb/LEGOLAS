"""
LEGOLAS Training Module
Modulo per il training avanzato dei modelli con K-Fold CV, Focal Loss e Early Stopping
"""

# Configuration
from .config import (
    TrainingConfig,
    create_default_config,
    create_kfold_config
)

# Loss Functions
from .focal_loss import (
    FocalLoss,
    create_focal_loss,
    create_loss_from_config
)

# Utilities
from .utils import (
    # Dataset splitting
    stratified_train_val_test_split,
    create_stratified_kfold,
    
    # Class weights
    compute_class_weights,
    
    # Metrics
    compute_metrics,
    compute_detailed_metrics,
    
    # Saving/Loading
    save_metrics,
    load_metrics,
    
    # Analysis
    analyze_class_distribution
)

__all__ = [
    # Configuration
    "TrainingConfig",
    "create_default_config",
    "create_kfold_config",
    
    # Loss Functions
    "FocalLoss",
    "create_focal_loss",
    "create_loss_from_config",
    
    # Utilities - Splitting
    "stratified_train_val_test_split",
    "create_stratified_kfold",
    
    # Utilities - Weights
    "compute_class_weights",
    
    # Utilities - Metrics
    "compute_metrics",
    "compute_detailed_metrics",
    "save_metrics",
    "load_metrics",
    
    # Utilities - Analysis
    "analyze_class_distribution",
]
