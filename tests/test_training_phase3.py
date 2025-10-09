"""
Test Suite for Training Phase 3: Early Stopping

Testa:
- EarlyStopping (src/training/early_stopping.py)

Run tests:
    uv run python -m pytest tests/test_training_phase3.py -v
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.training.early_stopping import EarlyStopping, create_early_stopping_from_config
from src.training.config import TrainingConfig


class DummyModel(nn.Module):
    """Modello dummy per test"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)


class TestEarlyStopping:
    """Test EarlyStopping functionality"""
    
    def test_early_stopping_creation(self):
        """Test creazione early stopping"""
        early_stop = EarlyStopping(
            patience=5,
            min_delta=0.001,
            use_loss_ratio=True,
            verbose=False
        )
        
        assert early_stop.patience == 5
        assert early_stop.min_delta == 0.001
        assert early_stop.use_loss_ratio is True
        assert early_stop.best_val_loss == np.inf
        assert early_stop.stopped is False
    
    def test_early_stopping_improvement(self):
        """Test aggiornamento con miglioramento"""
        early_stop = EarlyStopping(patience=3, min_delta=0.01, verbose=False)
        model = DummyModel()
        
        # Val loss migliora
        improved = early_stop.update(val_loss=0.5, model=model)
        
        assert improved is True
        assert early_stop.best_val_loss == 0.5
        assert early_stop.wait_count == 0
        assert early_stop.should_stop() is False
    
    def test_early_stopping_no_improvement(self):
        """Test aggiornamento senza miglioramento"""
        early_stop = EarlyStopping(patience=3, min_delta=0.01, verbose=False)
        model = DummyModel()
        
        # Prima epoca
        early_stop.update(val_loss=0.5, model=model)
        
        # Val loss peggiora
        improved = early_stop.update(val_loss=0.52, model=model)
        
        assert improved is False
        assert early_stop.wait_count == 1
        assert early_stop.should_stop() is False
    
    def test_early_stopping_patience_exhausted(self):
        """Test stop per patience esaurita"""
        early_stop = EarlyStopping(patience=3, min_delta=0.01, verbose=False)
        model = DummyModel()
        
        # Val loss non migliora per 4 epoche
        val_losses = [0.5, 0.51, 0.52, 0.53]
        
        for val_loss in val_losses:
            early_stop.update(val_loss=val_loss, model=model)
            if early_stop.should_stop():
                break
        
        assert early_stop.stopped is True
        assert early_stop.wait_count >= early_stop.patience
    
    def test_early_stopping_ratio_violations(self):
        """Test stop per ratio violations (overfitting)"""
        early_stop = EarlyStopping(
            patience=5,
            min_delta=0.01,
            use_loss_ratio=True,
            loss_ratio_threshold=0.3,  # Threshold molto basso
            loss_ratio_patience=2,  # Solo 2 consecutivi
            verbose=False
        )
        model = DummyModel()
        
        # Simula overfitting estremo: train molto basso, val alto
        # Simula overfitting: train_loss scende, val_loss sale
        train_losses = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1]
        val_losses = [0.5, 0.48, 0.50, 0.52, 0.55, 0.60]
        
        for train_loss, val_loss in zip(train_losses, val_losses):
            early_stop.update(val_loss=val_loss, train_loss=train_loss, model=model)
            if early_stop.should_stop():
                break
        
        assert early_stop.stopped is True
        assert early_stop.ratio_violations >= early_stop.loss_ratio_patience
    
    def test_early_stopping_model_buffer(self):
        """Test buffer degli stati modello"""
        early_stop = EarlyStopping(patience=3, verbose=False)
        model = DummyModel()
        
        # Aggiorna 5 volte
        for epoch in range(5):
            early_stop.update(val_loss=0.5 + epoch * 0.01, model=model)
        
        # Buffer mantiene solo ultimi 3 (patience)
        assert len(early_stop.model_states_buffer) == 3
    
    def test_early_stopping_restore_best(self):
        """Test ripristino best weights"""
        early_stop = EarlyStopping(
            patience=3,
            min_delta=0.01,
            restore_best_weights=True,
            verbose=False
        )
        model = DummyModel()
        
        # Val loss migliora poi peggiora
        val_losses = [0.5, 0.4, 0.45, 0.50, 0.55]
        
        for val_loss in val_losses:
            early_stop.update(val_loss=val_loss, model=model)
            if early_stop.should_stop():
                break
        
        # Ripristina
        restored_epoch = early_stop.restore_weights(model)
        
        assert restored_epoch == early_stop.best_epoch
        assert early_stop.best_epoch == 1  # Epoca con val_loss=0.4
    
    def test_early_stopping_history(self):
        """Test tracking history"""
        early_stop = EarlyStopping(patience=3, verbose=False)
        model = DummyModel()
        
        # Aggiorna 3 volte
        for epoch in range(3):
            early_stop.update(
                val_loss=0.5 - epoch * 0.05,
                train_loss=0.4 - epoch * 0.05,
                model=model
            )
        
        assert len(early_stop.val_loss_history) == 3
        assert len(early_stop.train_loss_history) == 3
        assert len(early_stop.ratio_history) == 3
    
    def test_early_stopping_get_state(self):
        """Test recupero stato"""
        early_stop = EarlyStopping(patience=3, verbose=False)
        model = DummyModel()
        
        early_stop.update(val_loss=0.5, model=model)
        
        state = early_stop.get_state()
        
        assert "best_val_loss" in state
        assert "best_epoch" in state
        assert "wait_count" in state
        assert "stopped" in state
        assert state["best_val_loss"] == 0.5
        assert state["best_epoch"] == 0
    
    def test_early_stopping_continuous_improvement(self):
        """Test nessun stop se migliora sempre"""
        early_stop = EarlyStopping(patience=3, min_delta=0.01, verbose=False)
        model = DummyModel()
        
        # Val loss migliora continuamente
        val_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
        
        for val_loss in val_losses:
            early_stop.update(val_loss=val_loss, model=model)
            if early_stop.should_stop():
                break
        
        assert early_stop.stopped is False
        assert early_stop.best_val_loss == 0.1
    
    def test_early_stopping_from_config(self):
        """Test creazione da TrainingConfig"""
        config = TrainingConfig(
            early_stopping_patience=7,
            early_stopping_min_delta=0.005,
            use_loss_ratio_monitoring=True,
            loss_ratio_patience=4,
            loss_function="focal"
        )
        
        early_stop = create_early_stopping_from_config(config)
        
        assert early_stop.patience == 7
        assert early_stop.min_delta == 0.005
        assert early_stop.use_loss_ratio is True
        assert early_stop.loss_ratio_patience == 4
        assert early_stop.loss_ratio_threshold == config.focal_loss_ratio_threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
