"""
Test Suite for Training Phase 2: Model Checkpoint

Testa:
- ModelCheckpoint (src/training/checkpoint.py)

Run tests:
    uv run python -m pytest tests/test_training_phase2.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json

from src.training.checkpoint import ModelCheckpoint, create_checkpoint_from_config
from src.training.config import TrainingConfig


class DummyModel(nn.Module):
    """Modello dummy per test"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)


class TestModelCheckpoint:
    """Test ModelCheckpoint functionality"""
    
    def test_checkpoint_creation(self, tmp_path):
        """Test creazione checkpoint"""
        checkpoint = ModelCheckpoint(
            save_dir=tmp_path,
            metric="balanced_accuracy",
            mode="max",
            fold=1
        )
        
        assert checkpoint.metric == "balanced_accuracy"
        assert checkpoint.mode == "max"
        assert checkpoint.fold == 1
        assert checkpoint.best_value == -np.inf
        assert checkpoint.best_epoch == -1
    
    def test_checkpoint_improvement(self, tmp_path):
        """Test salvataggio quando metrica migliora"""
        model = DummyModel()
        optimizer = optim.Adam(model.parameters())
        
        checkpoint = ModelCheckpoint(
            save_dir=tmp_path,
            metric="balanced_accuracy",
            mode="max",
            fold=0,
            verbose=False
        )
        
        # Epoch 0: prima metrica
        metrics_0 = {"balanced_accuracy": 0.75, "f1": 0.70}
        saved = checkpoint.update(model, optimizer, 0, metrics_0)
        
        assert saved is True
        assert checkpoint.best_value == 0.75
        assert checkpoint.best_epoch == 0
        assert checkpoint.best_path.exists()
    
    def test_checkpoint_no_improvement(self, tmp_path):
        """Test nessun salvataggio quando metrica peggiora"""
        model = DummyModel()
        optimizer = optim.Adam(model.parameters())
        
        checkpoint = ModelCheckpoint(
            save_dir=tmp_path,
            metric="balanced_accuracy",
            mode="max",
            fold=0,
            verbose=False
        )
        
        # Epoch 0: metrica iniziale
        metrics_0 = {"balanced_accuracy": 0.75}
        checkpoint.update(model, optimizer, 0, metrics_0)
        
        # Epoch 1: metrica peggiore
        metrics_1 = {"balanced_accuracy": 0.70}
        saved = checkpoint.update(model, optimizer, 1, metrics_1)
        
        assert saved is False
        assert checkpoint.best_value == 0.75
        assert checkpoint.best_epoch == 0
    
    def test_checkpoint_multiple_improvements(self, tmp_path):
        """Test salvataggio multiplo con miglioramenti"""
        model = DummyModel()
        optimizer = optim.Adam(model.parameters())
        
        checkpoint = ModelCheckpoint(
            save_dir=tmp_path,
            metric="balanced_accuracy",
            mode="max",
            fold=0,
            verbose=False
        )
        
        metrics_sequence = [
            {"balanced_accuracy": 0.75},
            {"balanced_accuracy": 0.78},  # Migliora
            {"balanced_accuracy": 0.77},  # Peggiora
            {"balanced_accuracy": 0.82},  # Migliora
        ]
        
        improvements = 0
        for epoch, metrics in enumerate(metrics_sequence):
            if checkpoint.update(model, optimizer, epoch, metrics):
                improvements += 1
        
        assert improvements == 3  # Epoch 0, 1, 3
        assert checkpoint.best_value == 0.82
        assert checkpoint.best_epoch == 3
        assert len(checkpoint.history) == 4
    
    def test_checkpoint_mode_min(self, tmp_path):
        """Test modalità minimizzazione (per loss)"""
        model = DummyModel()
        optimizer = optim.Adam(model.parameters())
        
        checkpoint = ModelCheckpoint(
            save_dir=tmp_path,
            metric="val_loss",
            mode="min",
            fold=0,
            verbose=False
        )
        
        # Loss che diminuisce (miglioramento)
        checkpoint.update(model, optimizer, 0, {"val_loss": 0.5})
        saved = checkpoint.update(model, optimizer, 1, {"val_loss": 0.3})
        
        assert saved is True
        assert checkpoint.best_value == 0.3
    
    def test_checkpoint_load(self, tmp_path):
        """Test caricamento checkpoint"""
        model = DummyModel()
        optimizer = optim.Adam(model.parameters())
        
        checkpoint = ModelCheckpoint(
            save_dir=tmp_path,
            metric="balanced_accuracy",
            mode="max",
            fold=0,
            verbose=False
        )
        
        # Salva checkpoint
        metrics = {"balanced_accuracy": 0.85, "f1": 0.80}
        checkpoint.update(model, optimizer, 5, metrics)
        
        # Carica in nuovo modello
        new_model = DummyModel()
        new_optimizer = optim.Adam(new_model.parameters())
        loaded = checkpoint.load_best(new_model, new_optimizer)
        
        assert loaded["epoch"] == 5
        assert loaded["best_value"] == 0.85
        assert "model_state_dict" in loaded
        assert "optimizer_state_dict" in loaded
    
    def test_checkpoint_history(self, tmp_path):
        """Test salvataggio history"""
        model = DummyModel()
        optimizer = optim.Adam(model.parameters())
        
        checkpoint = ModelCheckpoint(
            save_dir=tmp_path,
            metric="balanced_accuracy",
            mode="max",
            fold=0,
            verbose=False
        )
        
        # Aggiorna più epoche
        for epoch in range(3):
            metrics = {"balanced_accuracy": 0.7 + epoch * 0.05}
            checkpoint.update(model, optimizer, epoch, metrics)
        
        # Salva history
        history_path = checkpoint.save_history()
        
        assert history_path.exists()
        
        with open(history_path, 'r') as f:
            history_data = json.load(f)
        
        assert history_data["metric"] == "balanced_accuracy"
        assert len(history_data["history"]) == 3
        assert history_data["best_epoch"] == 2
    
    def test_checkpoint_get_best_info(self, tmp_path):
        """Test recupero info best model"""
        model = DummyModel()
        optimizer = optim.Adam(model.parameters())
        
        checkpoint = ModelCheckpoint(
            save_dir=tmp_path,
            metric="balanced_accuracy",
            mode="max",
            fold=2,
            verbose=False
        )
        
        checkpoint.update(model, optimizer, 0, {"balanced_accuracy": 0.75})
        
        info = checkpoint.get_best_info()
        
        assert info["metric"] == "balanced_accuracy"
        assert info["best_value"] == 0.75
        assert info["best_epoch"] == 0
        assert info["fold"] == 2
        assert info["num_updates"] == 1
    
    def test_checkpoint_from_config(self, tmp_path):
        """Test creazione da TrainingConfig"""
        config = TrainingConfig(
            checkpoint_metric="f1",
            checkpoint_mode="max",
            model_name="bert-base",
            story_format="narrativo"
        )
        # Override models_dir per test
        config.__dict__["models_dir"] = tmp_path
        
        checkpoint = create_checkpoint_from_config(config, fold=3)
        
        assert checkpoint.metric == "f1"
        assert checkpoint.mode == "max"
        assert checkpoint.fold == 3
        assert checkpoint.model_name == "bert-base"
        assert checkpoint.story_format == "narrativo"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
