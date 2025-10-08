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

# Checkpointing
from .checkpoint import (
    ModelCheckpoint,
    create_checkpoint_from_config
)

# Early Stopping
from .early_stopping import (
    EarlyStopping,
    create_early_stopping_from_config
)

# K-Fold Training
from .kfold_trainer import (
    KFoldTrainer,
    save_kfold_summary
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
    
    # Checkpointing
    "ModelCheckpoint",
    "create_checkpoint_from_config",
    
    # Early Stopping
    "EarlyStopping",
    "create_early_stopping_from_config",
    
    # K-Fold Training
    "KFoldTrainer",
    "save_kfold_summary",
    
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
