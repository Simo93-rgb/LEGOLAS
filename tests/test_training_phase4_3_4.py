"""
Test per FASE 4.3.4 - Checkpoint e Model Saving
Verifica che il checkpoint salvi correttamente modelli e history.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json

from src.training.config import TrainingConfig
from src.training.checkpoint import ModelCheckpoint
from src.training.early_stopping import EarlyStopping


class SimpleModel(nn.Module):
    """Modello semplice per test."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


class TestCheckpointMetrics:
    """Test per tracking e salvataggio metriche."""
    
    def test_checkpoint_uses_balanced_accuracy(self, tmp_path):
        """Test che checkpoint usa balanced_accuracy come metrica."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint = ModelCheckpoint(
            save_dir=tmp_path,
            metric='balanced_accuracy',
            mode='max',
            model_name='test_model',
            story_format='narrativo'
        )
        
        # Simula epoche con miglioramenti
        metrics_sequence = [
            {'balanced_accuracy': 0.75, 'accuracy': 0.80, 'f1': 0.70},
            {'balanced_accuracy': 0.78, 'accuracy': 0.82, 'f1': 0.73},
            {'balanced_accuracy': 0.82, 'accuracy': 0.85, 'f1': 0.78},
        ]
        
        for epoch, metrics in enumerate(metrics_sequence):
            checkpoint.update(model, optimizer, epoch, metrics)
        
        # Verifica best metric
        best_info = checkpoint.get_best_info()
        assert best_info['metric'] == 'balanced_accuracy'
        assert best_info['best_value'] == 0.82
        assert best_info['best_epoch'] == 2
    
    def test_checkpoint_saves_all_metrics(self, tmp_path):
        """Test che checkpoint salva tutte le metriche nel history."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint = ModelCheckpoint(
            save_dir=tmp_path,
            metric='balanced_accuracy',
            mode='max',
            model_name='test_model',
            story_format='narrativo'
        )
        
        # Simula training con metriche complete
        metrics = {
            'epoch': 0,
            'train_loss': 0.5,
            'val_loss': 0.6,
            'balanced_accuracy': 0.75,
            'accuracy': 0.80,
            'precision': 0.78,
            'recall': 0.72,
            'f1': 0.75
        }
        
        checkpoint.update(model, optimizer, 0, metrics)
        
        # Verifica history
        assert len(checkpoint.history) == 1
        history_entry = checkpoint.history[0]
        
        # Verifica tutte le metriche sono salvate
        assert 'epoch' in history_entry
        assert 'train_loss' in history_entry
        assert 'val_loss' in history_entry
        assert 'balanced_accuracy' in history_entry
        assert 'accuracy' in history_entry
        assert 'precision' in history_entry
        assert 'recall' in history_entry
        assert 'f1' in history_entry
    
    def test_checkpoint_history_file_creation(self, tmp_path):
        """Test che checkpoint.save_history() crea file JSON."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint = ModelCheckpoint(
            save_dir=tmp_path,
            metric='balanced_accuracy',
            mode='max',
            model_name='test_model',
            story_format='narrativo',
            fold=1
        )
        
        # Simula alcune epoche
        for epoch in range(3):
            metrics = {
                'balanced_accuracy': 0.7 + epoch * 0.05,
                'accuracy': 0.75 + epoch * 0.03
            }
            checkpoint.update(model, optimizer, epoch, metrics)
        
        # Salva history
        history_path = checkpoint.save_history()
        
        # Verifica file creato
        assert history_path.exists()
        assert history_path.name == 'checkpoint_history_fold1.json'
        
        # Verifica contenuto
        with open(history_path) as f:
            history_data = json.load(f)
        
        assert 'metric' in history_data
        assert 'mode' in history_data
        assert 'best_value' in history_data
        assert 'best_epoch' in history_data
        assert 'fold' in history_data
        assert 'history' in history_data
        
        assert history_data['metric'] == 'balanced_accuracy'
        assert history_data['mode'] == 'max'
        assert history_data['fold'] == 1
        assert len(history_data['history']) == 3


class TestCheckpointModelSaving:
    """Test per salvataggio modelli."""
    
    def test_checkpoint_saves_only_improvements(self, tmp_path):
        """Test che checkpoint salva solo quando la metrica migliora."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint = ModelCheckpoint(
            save_dir=tmp_path,
            metric='balanced_accuracy',
            mode='max',
            model_name='test_model',
            story_format='narrativo'
        )
        
        # Simula epoche: migliora, peggiora, migliora
        metrics_sequence = [
            {'balanced_accuracy': 0.75},  # Salva (primo)
            {'balanced_accuracy': 0.78},  # Salva (migliora)
            {'balanced_accuracy': 0.76},  # Non salva (peggiora)
            {'balanced_accuracy': 0.82},  # Salva (migliora)
        ]
        
        saved_count = 0
        for epoch, metrics in enumerate(metrics_sequence):
            improved = checkpoint.update(model, optimizer, epoch, metrics)
            if improved:
                saved_count += 1
        
        # Dovrebbe aver salvato 3 volte (epoch 0, 1, 3)
        assert saved_count == 3
        assert checkpoint.best_epoch == 3
        assert checkpoint.best_value == 0.82
    
    def test_checkpoint_file_path_format(self, tmp_path):
        """Test formato path file checkpoint."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Test senza fold
        checkpoint_no_fold = ModelCheckpoint(
            save_dir=tmp_path / 'no_fold',
            metric='balanced_accuracy',
            mode='max',
            model_name='bertm',
            story_format='narrativo'
        )
        
        checkpoint_no_fold.update(model, optimizer, 0, {'balanced_accuracy': 0.8})
        
        assert checkpoint_no_fold.best_path is not None
        assert 'narrativo' in checkpoint_no_fold.best_path.name
        assert 'bertm' in checkpoint_no_fold.best_path.name
        assert 'fold' not in checkpoint_no_fold.best_path.name  # No fold in filename
        
        # Test con fold
        checkpoint_with_fold = ModelCheckpoint(
            save_dir=tmp_path / 'with_fold',
            metric='balanced_accuracy',
            mode='max',
            model_name='cbert',
            story_format='clinical',
            fold=3
        )
        
        checkpoint_with_fold.update(model, optimizer, 0, {'balanced_accuracy': 0.85})
        
        assert checkpoint_with_fold.best_path is not None
        assert 'clinical' in checkpoint_with_fold.best_path.name
        assert 'cbert' in checkpoint_with_fold.best_path.name
        assert 'fold3' in checkpoint_with_fold.best_path.name  # Fold in filename
    
    def test_checkpoint_model_loadable(self, tmp_path):
        """Test che modello salvato può essere ricaricato."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint = ModelCheckpoint(
            save_dir=tmp_path,
            metric='balanced_accuracy',
            mode='max',
            model_name='test_model',
            story_format='narrativo'
        )
        
        # Salva modello
        metrics = {'balanced_accuracy': 0.85}
        checkpoint.update(model, optimizer, 0, metrics)
        
        # Crea nuovo modello e carica pesi
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        checkpoint_data = checkpoint.load_best(new_model, new_optimizer)
        
        # Verifica che checkpoint è stato caricato
        assert 'metrics' in checkpoint_data
        assert checkpoint_data['metrics']['balanced_accuracy'] == 0.85
        assert 'epoch' in checkpoint_data


class TestCheckpointIntegration:
    """Test integrazione checkpoint con training loop."""
    
    def test_checkpoint_tracks_epoch_progression(self, tmp_path):
        """Test che checkpoint traccia correttamente progressione epoche."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint = ModelCheckpoint(
            save_dir=tmp_path,
            metric='balanced_accuracy',
            mode='max',
            model_name='test_model',
            story_format='narrativo'
        )
        
        # Simula training con 5 epoche
        for epoch in range(5):
            metrics = {
                'epoch': epoch,
                'balanced_accuracy': 0.70 + epoch * 0.02,
                'train_loss': 0.5 - epoch * 0.05,
                'val_loss': 0.6 - epoch * 0.04
            }
            checkpoint.update(model, optimizer, epoch, metrics)
        
        # Verifica history completa
        assert len(checkpoint.history) == 5
        
        # Verifica ordine epoche
        for i, entry in enumerate(checkpoint.history):
            assert entry['epoch'] == i
    
    def test_checkpoint_with_early_stopping_scenario(self, tmp_path):
        """Test checkpoint in scenario con early stopping."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint = ModelCheckpoint(
            save_dir=tmp_path,
            metric='balanced_accuracy',
            mode='max',
            model_name='test_model',
            story_format='narrativo'
        )
        
        early_stopping = EarlyStopping(
            patience=3,
            min_delta=0.01,
            use_loss_ratio=False
        )
        
        # Simula training: migliora, poi peggiora (trigger early stop)
        val_losses = [0.6, 0.5, 0.4, 0.41, 0.42, 0.43]
        bal_accs = [0.70, 0.75, 0.80, 0.79, 0.78, 0.77]
        
        stopped_at = -1
        for epoch, (val_loss, bal_acc) in enumerate(zip(val_losses, bal_accs)):
            metrics = {'balanced_accuracy': bal_acc}
            checkpoint.update(model, optimizer, epoch, metrics)
            
            early_stopping.update(val_loss=val_loss, train_loss=0.3, model=model)
            
            if early_stopping.should_stop():
                stopped_at = epoch
                break
        
        # Verifica che early stopping è stato triggerato
        assert stopped_at > 0
        
        # Verifica che checkpoint ha salvato best model (epoch 2, bal_acc=0.80)
        assert checkpoint.best_epoch == 2
        assert checkpoint.best_value == 0.80


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
