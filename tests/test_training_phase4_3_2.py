"""
Test per FASE 4.3.2 - Refactor pre_train() con Checkpoint e Early Stopping
Verifica che la nuova signature e logica funzionino correttamente.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys

from src.training.config import TrainingConfig
from src.training.checkpoint import ModelCheckpoint
from src.training.early_stopping import EarlyStopping

# Import pre_train per testarlo (lo importiamo come modulo)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "train_llm_module",
    Path(__file__).parent.parent / "src" / "training" / "train_llm.py"
)
train_llm_module = importlib.util.module_from_spec(spec)


class SimpleTestModel(nn.Module):
    """Modello semplice per test."""
    def __init__(self, input_size=10, hidden_size=5, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask=None):
        x = torch.relu(self.fc1(input_ids))
        return self.fc2(x)


class TestPreTrainRefactor:
    """Test per nuova signature di pre_train()."""
    
    def test_pre_train_accepts_new_parameters(self, tmp_path):
        """Test che pre_train() accetta i nuovi parametri."""
        # Crea dati mock
        X = torch.randn(50, 10)
        y = torch.randint(0, 2, (50,))
        
        # Crea dataset mock con formato corretto (dict con 'input_ids', 'attention_mask', 'labels')
        class MockDataset(torch.utils.data.Dataset):
            def __init__(self, X, y):
                self.X = X
                self.y = y
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                return {
                    'input_ids': self.X[idx],
                    'attention_mask': torch.ones(10),  # Mock attention mask
                    'labels': self.y[idx]
                }
        
        train_dataset = MockDataset(X[:40], y[:40])
        val_dataset = MockDataset(X[40:], y[40:])
        
        train_loader = DataLoader(train_dataset, batch_size=8)
        val_loader = DataLoader(val_dataset, batch_size=8)
        
        # Crea modello
        model = SimpleTestModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        criterion = nn.CrossEntropyLoss()
        
        # Crea config temporaneo
        config = TrainingConfig(
            model_name='test_model',
            story_format='narrativo',
            num_epochs=2,  # Solo 2 epoch per test veloce
            batch_size=8,
            early_stopping_patience=3,
            device='cpu'
        )
        
        # Crea checkpoint con save_dir temporaneo
        checkpoint = ModelCheckpoint(
            save_dir=tmp_path / 'models',
            fold=None,
            metric='balanced_accuracy',
            mode='max',
            model_name='test_model',
            story_format='narrativo'
        )
        
        # Crea EarlyStopping
        early_stopping = EarlyStopping(
            patience=3,
            min_delta=0.0,
            use_loss_ratio=True,
            loss_ratio_threshold=config.get_loss_ratio_threshold(),
            loss_ratio_patience=2,
            restore_best_weights=True
        )
        
        # Mock accelerator semplice (senza distributed)
        class MockAccelerator:
            def __init__(self):
                self.device = torch.device('cpu')
            
            def backward(self, loss):
                loss.backward()
            
            def wait_for_everyone(self):
                pass
            
            @property
            def is_main_process(self):
                return True
            
            def unwrap_model(self, model):
                return model
        
        accelerator = MockAccelerator()
        
        # Importa e esegui pre_train
        spec.loader.exec_module(train_llm_module)
        
        # Questa chiamata NON deve sollevare eccezioni
        train_llm_module.pre_train(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            scheduler=scheduler,
            criterion=criterion,
            accelerator=accelerator,
            config=config,
            checkpoint=checkpoint,
            early_stopping=early_stopping,
            fold=None
        )
    
    def test_pre_train_creates_checkpoint(self, tmp_path):
        """Test che pre_train() crea file checkpoint."""
        # Setup identico al test precedente ma controllo creazione file
        X = torch.randn(50, 10)
        y = torch.randint(0, 2, (50,))
        
        class MockDataset(torch.utils.data.Dataset):
            def __init__(self, X, y):
                self.X = X
                self.y = y
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                return {
                    'input_ids': self.X[idx],
                    'attention_mask': torch.ones(10),
                    'labels': self.y[idx]
                }
        
        train_dataset = MockDataset(X[:40], y[:40])
        val_dataset = MockDataset(X[40:], y[40:])
        
        train_loader = DataLoader(train_dataset, batch_size=8)
        val_loader = DataLoader(val_dataset, batch_size=8)
        
        model = SimpleTestModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        criterion = nn.CrossEntropyLoss()
        
        config = TrainingConfig(
            model_name='test_model',
            story_format='narrativo',
            num_epochs=2,
            batch_size=8,
            early_stopping_patience=5,  # Alto per non triggerare
            device='cpu'
        )
        
        checkpoint = ModelCheckpoint(
            save_dir=tmp_path / 'models',
            fold=None,
            metric='balanced_accuracy',
            mode='max',
            model_name='test_model',
            story_format='narrativo'
        )
        
        early_stopping = EarlyStopping(
            patience=10,  # Alto per completare training
            min_delta=0.0,
            use_loss_ratio=False,  # Disabilita per completare training
            loss_ratio_threshold=100.0,
            loss_ratio_patience=10,
            restore_best_weights=True
        )
        
        class MockAccelerator:
            def __init__(self):
                self.device = torch.device('cpu')
            
            def backward(self, loss):
                loss.backward()
            
            def wait_for_everyone(self):
                pass
            
            @property
            def is_main_process(self):
                return True
            
            def unwrap_model(self, model):
                return model
        
        accelerator = MockAccelerator()
        
        spec.loader.exec_module(train_llm_module)
        
        # Esegui training
        train_llm_module.pre_train(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            scheduler=scheduler,
            criterion=criterion,
            accelerator=accelerator,
            config=config,
            checkpoint=checkpoint,
            early_stopping=early_stopping,
            fold=None
        )
        
        # Verifica che checkpoint sia stato creato
        assert checkpoint.best_path is not None, "Checkpoint dovrebbe avere un best_path"
        assert checkpoint.best_path.exists(), f"File checkpoint dovrebbe esistere: {checkpoint.best_path}"
        
        # Verifica che ci siano metriche salvate
        best_info = checkpoint.get_best_info()
        assert 'best_epoch' in best_info
        assert 'best_value' in best_info
        assert best_info['best_value'] > 0.0  # balanced_accuracy dovrebbe essere > 0
    
    def test_pre_train_with_fold_number(self, tmp_path):
        """Test che pre_train() gestisce correttamente il numero di fold."""
        X = torch.randn(30, 10)
        y = torch.randint(0, 2, (30,))
        
        class MockDataset(torch.utils.data.Dataset):
            def __init__(self, X, y):
                self.X = X
                self.y = y
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                return {
                    'input_ids': self.X[idx],
                    'attention_mask': torch.ones(10),
                    'labels': self.y[idx]
                }
        
        train_dataset = MockDataset(X[:24], y[:24])
        val_dataset = MockDataset(X[24:], y[24:])
        
        train_loader = DataLoader(train_dataset, batch_size=8)
        val_loader = DataLoader(val_dataset, batch_size=8)
        
        model = SimpleTestModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        criterion = nn.CrossEntropyLoss()
        
        config = TrainingConfig(
            model_name='test_model',
            story_format='narrativo',
            num_epochs=1,
            batch_size=8,
            use_kfold=True,
            n_folds=3,
            device='cpu'
        )
        
        # Testa con fold=2 (terzo fold)
        checkpoint = ModelCheckpoint(
            save_dir=tmp_path / 'models',
            fold=2,  # Terzo fold
            metric='balanced_accuracy',
            mode='max',
            model_name='test_model',
            story_format='narrativo'
        )
        
        early_stopping = EarlyStopping(
            patience=10,
            min_delta=0.0,
            use_loss_ratio=False,
            loss_ratio_threshold=100.0,
            loss_ratio_patience=10,
            restore_best_weights=True
        )
        
        class MockAccelerator:
            def __init__(self):
                self.device = torch.device('cpu')
            
            def backward(self, loss):
                loss.backward()
            
            def wait_for_everyone(self):
                pass
            
            @property
            def is_main_process(self):
                return True
            
            def unwrap_model(self, model):
                return model
        
        accelerator = MockAccelerator()
        
        spec.loader.exec_module(train_llm_module)
        
        # Esegui con fold=2
        train_llm_module.pre_train(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            scheduler=scheduler,
            criterion=criterion,
            accelerator=accelerator,
            config=config,
            checkpoint=checkpoint,
            early_stopping=early_stopping,
            fold=2  # Specifica fold
        )
        
        # Verifica che checkpoint include fold nel filename
        assert checkpoint.best_path is not None
        assert 'fold2' in str(checkpoint.best_path), "Filename checkpoint dovrebbe includere fold number"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
