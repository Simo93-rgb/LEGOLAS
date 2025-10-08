"""
Test Suite for Training Phase 4: K-Fold Trainer

Testa:
- KFoldTrainer (src/training/kfold_trainer.py)

Run tests:
    uv run python -m pytest tests/test_training_phase4.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset

from src.training.kfold_trainer import KFoldTrainer, save_kfold_summary
from src.training.config import TrainingConfig


class DummyModel(nn.Module):
    """Modello dummy per test"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    
    def forward(self, x):
        return self.fc(x)


def create_dummy_dataset(n_samples=100, n_features=10, n_classes=2):
    """Crea dataset dummy per test"""
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    dataset = TensorDataset(X, y)
    labels = y.numpy()
    return dataset, labels


def dummy_train_func(model, train_dataset, val_dataset, fold, checkpoint, early_stopping, config):
    """Funzione di training dummy per test"""
    device = config.get_device()
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8)
    
    # Solo 2 epoche per test veloce
    for epoch in range(2):
        # Training
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())
        
        # Metriche semplici
        from src.training.utils import compute_metrics
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        
        # Update checkpoint
        checkpoint.update(model, optimizer, epoch, metrics)
    
    return metrics


class TestKFoldTrainer:
    """Test KFoldTrainer functionality"""
    
    def test_kfold_trainer_creation(self):
        """Test creazione KFoldTrainer"""
        dataset, labels = create_dummy_dataset(n_samples=50)
        config = TrainingConfig(use_kfold=True, n_folds=3, num_epochs=2)
        
        kfold_trainer = KFoldTrainer(
            config=config,
            train_func=dummy_train_func,
            model_factory=DummyModel,
            dataset=dataset,
            labels=labels,
            verbose=False
        )
        
        assert kfold_trainer.config.n_folds == 3
        assert len(kfold_trainer.labels) == 50
        assert kfold_trainer.kfold is not None
    
    def test_kfold_trainer_requires_kfold_config(self):
        """Test errore se config non ha use_kfold=True"""
        dataset, labels = create_dummy_dataset(n_samples=50)
        config = TrainingConfig(use_kfold=False)  # Errore!
        
        with pytest.raises(ValueError, match="use_kfold=True"):
            KFoldTrainer(
                config=config,
                train_func=dummy_train_func,
                model_factory=DummyModel,
                dataset=dataset,
                labels=labels
            )
    
    def test_kfold_trainer_mismatch_labels(self):
        """Test errore se labels e dataset hanno dimensioni diverse"""
        dataset, labels = create_dummy_dataset(n_samples=50)
        wrong_labels = labels[:30]  # Solo 30 labels per 50 samples
        config = TrainingConfig(use_kfold=True, n_folds=3)
        
        with pytest.raises(ValueError, match="Mismatch"):
            KFoldTrainer(
                config=config,
                train_func=dummy_train_func,
                model_factory=DummyModel,
                dataset=dataset,
                labels=wrong_labels
            )
    
    def test_kfold_trainer_run(self, tmp_path):
        """Test esecuzione completa K-Fold"""
        dataset, labels = create_dummy_dataset(n_samples=60)
        config = TrainingConfig(use_kfold=True, n_folds=3, num_epochs=2)
        
        # Override paths per test
        config.__dict__["models_dir"] = tmp_path / "models"
        config.__dict__["reports_dir"] = tmp_path / "reports"
        config.models_dir.mkdir(parents=True, exist_ok=True)
        config.reports_dir.mkdir(parents=True, exist_ok=True)
        
        kfold_trainer = KFoldTrainer(
            config=config,
            train_func=dummy_train_func,
            model_factory=DummyModel,
            dataset=dataset,
            labels=labels,
            verbose=False
        )
        
        results = kfold_trainer.run()
        
        # Verifica struttura risultati
        assert "n_folds" in results
        assert "fold_results" in results
        assert "mean" in results
        assert "std" in results
        assert "min" in results
        assert "max" in results
        
        # Verifica numero fold
        assert results["n_folds"] == 3
        assert len(results["fold_results"]) == 3
        
        # Verifica metriche aggregate
        assert "accuracy" in results["mean"]
        assert "balanced_accuracy" in results["mean"]
        assert "precision" in results["mean"]
        assert "recall" in results["mean"]
        assert "f1" in results["mean"]
    
    def test_kfold_trainer_fold_models(self, tmp_path):
        """Test salvataggio modelli per fold"""
        dataset, labels = create_dummy_dataset(n_samples=60)
        config = TrainingConfig(use_kfold=True, n_folds=3, num_epochs=2)
        
        config.__dict__["models_dir"] = tmp_path / "models"
        config.__dict__["reports_dir"] = tmp_path / "reports"
        config.models_dir.mkdir(parents=True, exist_ok=True)
        config.reports_dir.mkdir(parents=True, exist_ok=True)
        
        kfold_trainer = KFoldTrainer(
            config=config,
            train_func=dummy_train_func,
            model_factory=DummyModel,
            dataset=dataset,
            labels=labels,
            verbose=False
        )
        
        results = kfold_trainer.run()
        
        # Verifica modelli salvati
        assert len(kfold_trainer.fold_models) == 3
        for model_path in kfold_trainer.fold_models:
            assert model_path.exists()
            assert "fold" in model_path.name
    
    def test_kfold_trainer_load_fold_model(self, tmp_path):
        """Test caricamento modello da fold specifico"""
        dataset, labels = create_dummy_dataset(n_samples=60)
        config = TrainingConfig(use_kfold=True, n_folds=3, num_epochs=2)
        
        config.__dict__["models_dir"] = tmp_path / "models"
        config.__dict__["reports_dir"] = tmp_path / "reports"
        config.models_dir.mkdir(parents=True, exist_ok=True)
        config.reports_dir.mkdir(parents=True, exist_ok=True)
        
        kfold_trainer = KFoldTrainer(
            config=config,
            train_func=dummy_train_func,
            model_factory=DummyModel,
            dataset=dataset,
            labels=labels,
            verbose=False
        )
        
        kfold_trainer.run()
        
        # Carica modello fold 0
        model_fold0 = kfold_trainer.load_fold_model(fold=0)
        assert isinstance(model_fold0, nn.Module)
    
    def test_kfold_trainer_get_best_fold(self, tmp_path):
        """Test identificazione best fold"""
        dataset, labels = create_dummy_dataset(n_samples=60)
        config = TrainingConfig(use_kfold=True, n_folds=3, num_epochs=2)
        
        config.__dict__["models_dir"] = tmp_path / "models"
        config.__dict__["reports_dir"] = tmp_path / "reports"
        config.models_dir.mkdir(parents=True, exist_ok=True)
        config.reports_dir.mkdir(parents=True, exist_ok=True)
        
        kfold_trainer = KFoldTrainer(
            config=config,
            train_func=dummy_train_func,
            model_factory=DummyModel,
            dataset=dataset,
            labels=labels,
            verbose=False
        )
        
        kfold_trainer.run()
        
        # Trova best fold
        best_fold = kfold_trainer.get_best_fold(metric="balanced_accuracy")
        assert 0 <= best_fold < 3
    
    def test_kfold_trainer_metrics_files(self, tmp_path):
        """Test salvataggio file metriche per fold"""
        dataset, labels = create_dummy_dataset(n_samples=60)
        config = TrainingConfig(use_kfold=True, n_folds=3, num_epochs=2)
        
        config.__dict__["models_dir"] = tmp_path / "models"
        config.__dict__["reports_dir"] = tmp_path / "reports"
        config.models_dir.mkdir(parents=True, exist_ok=True)
        config.reports_dir.mkdir(parents=True, exist_ok=True)
        
        kfold_trainer = KFoldTrainer(
            config=config,
            train_func=dummy_train_func,
            model_factory=DummyModel,
            dataset=dataset,
            labels=labels,
            verbose=False
        )
        
        kfold_trainer.run()
        
        # Verifica file metriche
        for fold in range(3):
            metrics_file = config.reports_dir / f"fold_{fold}_metrics.json"
            assert metrics_file.exists()
        
        # Verifica aggregated results
        aggregated_file = config.reports_dir / "kfold_aggregated_results.json"
        assert aggregated_file.exists()
    
    def test_save_kfold_summary(self, tmp_path):
        """Test salvataggio summary K-Fold"""
        config = TrainingConfig(use_kfold=True, n_folds=5)
        
        kfold_results = {
            "n_folds": 5,
            "mean": {"accuracy": 0.85},
            "std": {"accuracy": 0.05}
        }
        
        summary_path = tmp_path / "kfold_summary.json"
        save_kfold_summary(kfold_results, summary_path, config)
        
        assert summary_path.exists()
        
        import json
        with open(summary_path) as f:
            summary = json.load(f)
        
        assert "timestamp" in summary
        assert "config" in summary
        assert "results" in summary
        assert summary["config"]["n_folds"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
