"""
Test per FASE 4.3.1 - CLI Arguments e TrainingConfig Integration
Verifica che argparse e TrainingConfig funzionino correttamente.
"""

import pytest
import sys
from unittest.mock import patch
from pathlib import Path

# Importa le funzioni da train_llm
import importlib.util
spec = importlib.util.spec_from_file_location(
    "train_llm_module",
    Path(__file__).parent.parent / "src" / "training" / "train_llm.py"
)
train_llm_module = importlib.util.module_from_spec(spec)

from src.training.config import TrainingConfig


class TestCLIArguments:
    """Test per parsing argomenti CLI."""
    
    def test_parse_args_minimal(self):
        """Test con argomenti minimi richiesti."""
        test_args = [
            'train_llm.py',
            '--model_name', 'bertm',
            '--story_format', 'narrativo'
        ]
        
        with patch.object(sys, 'argv', test_args):
            # Esegui il modulo per importare parse_args
            spec.loader.exec_module(train_llm_module)
            args = train_llm_module.parse_args()
            
            assert args.model_name == 'bertm'
            assert args.story_format == 'narrativo'
            assert args.use_kfold is False  # Default
            assert args.n_folds == 10  # Default
            assert args.use_focal_loss is False  # Default
            assert args.epochs == 50  # Default
            assert args.batch_size == 16  # Default
            assert args.learning_rate == 1e-5  # Default
            assert args.patience == 5  # Default
            assert args.seed == 42  # Default
    
    def test_parse_args_kfold(self):
        """Test con K-Fold abilitato."""
        test_args = [
            'train_llm.py',
            '--model_name', 'cbert',
            '--story_format', 'clinical',
            '--use_kfold',
            '--n_folds', '5'
        ]
        
        with patch.object(sys, 'argv', test_args):
            spec.loader.exec_module(train_llm_module)
            args = train_llm_module.parse_args()
            
            assert args.model_name == 'cbert'
            assert args.story_format == 'clinical'
            assert args.use_kfold is True
            assert args.n_folds == 5
    
    def test_parse_args_focal_loss(self):
        """Test con Focal Loss abilitato."""
        test_args = [
            'train_llm.py',
            '--model_name', 'roberta',
            '--story_format', 'bullet',
            '--use_focal_loss',
            '--focal_alpha', '0.3', '0.7',
            '--focal_gamma', '3.0'
        ]
        
        with patch.object(sys, 'argv', test_args):
            spec.loader.exec_module(train_llm_module)
            args = train_llm_module.parse_args()
            
            assert args.use_focal_loss is True
            assert args.focal_alpha == [0.3, 0.7]
            assert args.focal_gamma == 3.0
    
    def test_parse_args_full(self):
        """Test con tutti gli argomenti."""
        test_args = [
            'train_llm.py',
            '--model_name', 'gpt2',
            '--story_format', 'narrativo',
            '--use_kfold',
            '--n_folds', '8',
            '--use_focal_loss',
            '--focal_alpha', '0.25', '0.75',
            '--focal_gamma', '2.5',
            '--epochs', '100',
            '--batch_size', '32',
            '--learning_rate', '2e-5',
            '--patience', '10',
            '--seed', '123'
        ]
        
        with patch.object(sys, 'argv', test_args):
            spec.loader.exec_module(train_llm_module)
            args = train_llm_module.parse_args()
            
            assert args.model_name == 'gpt2'
            assert args.story_format == 'narrativo'
            assert args.use_kfold is True
            assert args.n_folds == 8
            assert args.use_focal_loss is True
            assert args.focal_alpha == [0.25, 0.75]
            assert args.focal_gamma == 2.5
            assert args.epochs == 100
            assert args.batch_size == 32
            assert args.learning_rate == 2e-5
            assert args.patience == 10
            assert args.seed == 123


class TestCreateTrainingConfig:
    """Test per creazione TrainingConfig da CLI args."""
    
    def test_create_config_simple_training(self):
        """Test config per training semplice (no K-Fold)."""
        test_args = [
            'train_llm.py',
            '--model_name', 'bertm',
            '--story_format', 'narrativo',
            '--epochs', '30'
        ]
        
        with patch.object(sys, 'argv', test_args):
            spec.loader.exec_module(train_llm_module)
            args = train_llm_module.parse_args()
            config = train_llm_module.create_training_config(args)
            
            assert isinstance(config, TrainingConfig)
            assert config.model_name == 'bertm'
            assert config.story_format == 'narrativo'
            assert config.num_epochs == 30  # Corretto: num_epochs
            assert config.use_kfold is False
            assert config.n_folds == 1  # Non K-Fold
            assert config.loss_function == 'ce'  # Corretto: loss_function (non use_focal_loss)
    
    def test_create_config_kfold_training(self):
        """Test config per K-Fold training."""
        test_args = [
            'train_llm.py',
            '--model_name', 'cbert',
            '--story_format', 'clinical',
            '--use_kfold',
            '--n_folds', '10'
        ]
        
        with patch.object(sys, 'argv', test_args):
            spec.loader.exec_module(train_llm_module)
            args = train_llm_module.parse_args()
            config = train_llm_module.create_training_config(args)
            
            assert config.use_kfold is True
            assert config.n_folds == 10
    
    def test_create_config_focal_loss(self):
        """Test config con Focal Loss."""
        test_args = [
            'train_llm.py',
            '--model_name', 'roberta',
            '--story_format', 'bullet',
            '--use_focal_loss',
            '--focal_alpha', '0.3', '0.7',
            '--focal_gamma', '2.0'
        ]
        
        with patch.object(sys, 'argv', test_args):
            spec.loader.exec_module(train_llm_module)
            args = train_llm_module.parse_args()
            config = train_llm_module.create_training_config(args)
            
            assert config.loss_function == 'focal'  # Corretto: loss_function
            assert config.focal_alpha == [0.3, 0.7]
            assert config.focal_gamma == 2.0
    
    def test_create_config_validation(self):
        """Test che validate() viene chiamato."""
        test_args = [
            'train_llm.py',
            '--model_name', 'bertm',
            '--story_format', 'narrativo'
        ]
        
        with patch.object(sys, 'argv', test_args):
            spec.loader.exec_module(train_llm_module)
            args = train_llm_module.parse_args()
            
            # Questo non dovrebbe sollevare eccezioni
            config = train_llm_module.create_training_config(args)
            assert config is not None
    
    def test_create_config_invalid_story_format_raises(self):
        """Test che story_format invalido viene bloccato da argparse."""
        test_args = [
            'train_llm.py',
            '--model_name', 'bertm',
            '--story_format', 'invalid_format'  # Non in choices
        ]
        
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit):  # argparse fa exit(2) su errore
                spec.loader.exec_module(train_llm_module)
                train_llm_module.parse_args()


class TestBackwardCompatibility:
    """Test per backward compatibility con training semplice."""
    
    def test_no_kfold_flag_means_simple_training(self):
        """Test che senza --use_kfold si usa training semplice."""
        test_args = [
            'train_llm.py',
            '--model_name', 'bertm',
            '--story_format', 'narrativo'
        ]
        
        with patch.object(sys, 'argv', test_args):
            spec.loader.exec_module(train_llm_module)
            args = train_llm_module.parse_args()
            config = train_llm_module.create_training_config(args)
            
            assert config.use_kfold is False
            assert config.n_folds == 1
    
    def test_no_focal_loss_flag_means_cross_entropy(self):
        """Test che senza --use_focal_loss si usa Cross Entropy."""
        test_args = [
            'train_llm.py',
            '--model_name', 'bertm',
            '--story_format', 'narrativo'
        ]
        
        with patch.object(sys, 'argv', test_args):
            spec.loader.exec_module(train_llm_module)
            args = train_llm_module.parse_args()
            config = train_llm_module.create_training_config(args)
            
            assert config.loss_function == 'ce'  # Corretto: loss_function


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
