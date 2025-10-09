"""
Test Suite for Training Phase 1: Configuration and Utilities

Testa:
- TrainingConfig (src/training/config.py)
- FocalLoss (src/training/focal_loss.py)
- Training utilities (src/training/utils.py)

Run tests:
    pytest tests/test_training_phase1.py -v
    pytest tests/test_training_phase1.py::test_config_creation -v
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys



from src.training.config import TrainingConfig, create_default_config, create_kfold_config
from src.training.focal_loss import FocalLoss, create_focal_loss, create_loss_from_config
from src.training.utils import (
    stratified_train_val_test_split,
    create_stratified_kfold,
    compute_class_weights,
    compute_metrics,
    compute_detailed_metrics,
    analyze_class_distribution
)


# =============================================================================
# 🔧 TRAINING CONFIG TESTS
# =============================================================================

class TestTrainingConfig:
    """Test TrainingConfig creation and validation"""
    
    def test_config_creation(self):
        """Test creazione configurazione di base"""
        config = TrainingConfig()
        
        assert config.model_name == "clinical-bert"
        assert config.story_format == "narrativo"
        assert config.num_classes == 2
        assert config.batch_size == 8
        assert config.learning_rate == 2e-5
    
    def test_config_device(self):
        """Test device detection"""
        config = TrainingConfig()
        
        # Device dovrebbe essere 'cuda' se disponibile, altrimenti 'cpu'
        if torch.cuda.is_available():
            assert config.device == "cuda"
        else:
            assert config.device == "cpu"
        
        device = config.get_device()
        assert isinstance(device, torch.device)
    
    def test_config_paths(self):
        """Test path properties"""
        config = TrainingConfig()
        
        assert config.project_root.exists()
        assert config.project_root.is_dir()
        
        # Directory vengono create automaticamente
        assert config.output_dir.exists()
        assert config.models_dir.exists()
        assert config.logs_dir.exists()
        assert config.reports_dir.exists()
    
    def test_config_kfold(self):
        """Test configurazione K-Fold"""
        config = TrainingConfig(use_kfold=True, n_folds=5)
        
        assert config.use_kfold is True
        assert config.n_folds == 5
        assert config.kfold_stratified is True
    
    def test_config_focal_loss(self):
        """Test configurazione Focal Loss"""
        config = TrainingConfig(loss_function="focal")
        
        assert config.loss_function == "focal"
        assert len(config.focal_alpha) == 2
        assert config.focal_gamma == 2.0
        
        # Verifica threshold corretto
        threshold = config.get_loss_ratio_threshold()
        assert threshold == config.focal_loss_ratio_threshold
    
    def test_config_cross_entropy(self):
        """Test configurazione CrossEntropy"""
        config = TrainingConfig(loss_function="ce")
        
        assert config.loss_function == "ce"
        assert config.use_class_weights is True
        
        # Verifica threshold corretto
        threshold = config.get_loss_ratio_threshold()
        assert threshold == config.loss_ratio_threshold
    
    def test_config_model_filename(self):
        """Test generazione nome file modello"""
        config = TrainingConfig(
            model_name="bert-base-uncased",
            story_format="narrativo"
        )
        
        # Senza fold
        filename = config.get_model_filename()
        assert "narrativo" in filename
        assert "bert-base-uncased" in filename
        assert "fold" not in filename
        
        # Con fold
        filename_fold = config.get_model_filename(fold=3)
        assert "fold3" in filename_fold
        
        # Con suffix
        filename_suffix = config.get_model_filename(suffix="epoch_10")
        assert "epoch_10" in filename_suffix
    
    def test_config_validation(self):
        """Test validazione configurazione"""
        # Configurazione valida
        config_valid = TrainingConfig()
        assert config_valid.validate() is True
        
        # Configurazione invalida: batch_size negativo
        config_invalid = TrainingConfig(batch_size=-1)
        assert config_invalid.validate() is False
        
        # Configurazione invalida: test_size > 1
        config_invalid2 = TrainingConfig(test_size=1.5)
        assert config_invalid2.validate() is False
    
    def test_factory_functions(self):
        """Test factory functions"""
        # create_default_config
        config_default = create_default_config(
            model_name="bert-base-uncased",
            use_kfold=False
        )
        assert config_default.model_name == "bert-base-uncased"
        assert config_default.use_kfold is False
        
        # create_kfold_config
        config_kfold = create_kfold_config(n_folds=5)
        assert config_kfold.use_kfold is True
        assert config_kfold.n_folds == 5


# =============================================================================
# 🎲 FOCAL LOSS TESTS
# =============================================================================

class TestFocalLoss:
    """Test FocalLoss implementation"""
    
    def test_focal_loss_creation(self):
        """Test creazione Focal Loss"""
        focal = FocalLoss(alpha=[0.25, 0.75], gamma=2.0)
        
        assert focal.gamma == 2.0
        assert focal.reduction == "mean"
        assert torch.allclose(focal.alpha, torch.tensor([0.25, 0.75]))
    
    def test_focal_loss_forward(self):
        """Test forward pass"""
        focal = FocalLoss(alpha=[0.25, 0.75], gamma=2.0)
        
        batch_size = 8
        num_classes = 2
        logits = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        loss = focal(logits, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Loss deve essere non negativa
    
    def test_focal_loss_backward(self):
        """Test backward pass (gradient flow)"""
        focal = FocalLoss(alpha=[0.25, 0.75], gamma=2.0)
        
        logits = torch.randn(8, 2, requires_grad=True)
        targets = torch.randint(0, 2, (8,))
        
        loss = focal(logits, targets)
        loss.backward()
        
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape
    
    def test_focal_loss_vs_ce(self):
        """Test che Focal Loss riduce peso su esempi facili"""
        focal = FocalLoss(alpha=None, gamma=2.0)
        ce = nn.CrossEntropyLoss()
        
        # Esempi facili (alta confidenza)
        easy_logits = torch.tensor([
            [5.0, -5.0],
            [-5.0, 5.0]
        ])
        labels = torch.tensor([0, 1])
        
        focal_loss = focal(easy_logits, labels)
        ce_loss = ce(easy_logits, labels)
        
        # Focal Loss dovrebbe essere < CrossEntropy per esempi facili
        assert focal_loss < ce_loss
    
    def test_focal_loss_alpha_weighting(self):
        """Test che alpha aumenta peso su classe minoritaria"""
        # Senza alpha
        focal_no_alpha = FocalLoss(alpha=None, gamma=2.0)
        
        # Con alpha (più peso su classe 1)
        focal_with_alpha = FocalLoss(alpha=[0.25, 0.75], gamma=2.0)
        
        # Dataset sbilanciato: 9 esempi classe 0, 1 esempio classe 1
        logits = torch.tensor([
            [2.0, -2.0],  # Classe 0
            [2.0, -2.0],  # Classe 0
            [2.0, -2.0],  # Classe 0
            [2.0, -2.0],  # Classe 0
            [2.0, -2.0],  # Classe 0
            [2.0, -2.0],  # Classe 0
            [2.0, -2.0],  # Classe 0
            [2.0, -2.0],  # Classe 0
            [2.0, -2.0],  # Classe 0
            [-2.0, 2.0],  # Classe 1
        ])
        labels = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        
        loss_no_alpha = focal_no_alpha(logits, labels)
        loss_with_alpha = focal_with_alpha(logits, labels)
        
        # Con alpha, loss dovrebbe essere diversa (più peso su classe 1)
        assert not torch.allclose(loss_no_alpha, loss_with_alpha)
    
    def test_focal_loss_reduction(self):
        """Test different reduction modes"""
        logits = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))
        
        # Mean reduction
        focal_mean = FocalLoss(reduction="mean")
        loss_mean = focal_mean(logits, targets)
        assert loss_mean.dim() == 0
        
        # Sum reduction
        focal_sum = FocalLoss(reduction="sum")
        loss_sum = focal_sum(logits, targets)
        assert loss_sum.dim() == 0
        
        # None reduction
        focal_none = FocalLoss(reduction="none")
        loss_none = focal_none(logits, targets)
        assert loss_none.dim() == 1
        assert loss_none.shape[0] == 8
    
    def test_focal_loss_factory(self):
        """Test factory function"""
        focal = create_focal_loss(alpha=[0.3, 0.7], gamma=1.5)
        
        assert focal.gamma == 1.5
        assert torch.allclose(focal.alpha, torch.tensor([0.3, 0.7]))
    
    def test_create_loss_from_config(self):
        """Test loss creation from config"""
        # Focal Loss
        config_focal = TrainingConfig(loss_function="focal")
        loss_focal = create_loss_from_config(config_focal)
        assert isinstance(loss_focal, FocalLoss)
        
        # CrossEntropy
        config_ce = TrainingConfig(loss_function="ce")
        loss_ce = create_loss_from_config(config_ce)
        assert isinstance(loss_ce, nn.CrossEntropyLoss)


# =============================================================================
# 📊 TRAINING UTILS TESTS
# =============================================================================

class TestTrainingUtils:
    """Test training utility functions"""
    
    def test_stratified_split(self):
        """Test stratified train/val/test split"""
        # Dataset simulato: 89% classe 0, 11% classe 1
        n_samples = 1000
        X = np.arange(n_samples)
        y = np.array([0] * 890 + [1] * 110)
        np.random.shuffle(y)
        
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
            X, y, test_size=0.2, val_size=0.1, random_state=42
        )
        
        # Verifica dimensioni
        assert len(X_train) + len(X_val) + len(X_test) == n_samples
        assert len(y_train) == len(X_train)
        assert len(y_val) == len(X_val)
        assert len(y_test) == len(X_test)
        
        # Verifica split percentages (con tolleranza)
        assert abs(len(X_test) / n_samples - 0.2) < 0.01
        assert abs(len(X_val) / n_samples - 0.1) < 0.01
        
        # Verifica stratificazione (proporzioni simili)
        prop_train = (y_train == 0).sum() / len(y_train)
        prop_val = (y_val == 0).sum() / len(y_val)
        prop_test = (y_test == 0).sum() / len(y_test)
        
        # Tutte le proporzioni dovrebbero essere ~0.89 (con tolleranza)
        assert abs(prop_train - 0.89) < 0.05
        assert abs(prop_val - 0.89) < 0.05
        assert abs(prop_test - 0.89) < 0.05
    
    def test_stratified_kfold(self):
        """Test stratified K-Fold creation"""
        skf = create_stratified_kfold(n_splits=5, random_state=42)
        
        # Test split
        X = np.arange(100)
        y = np.array([0] * 80 + [1] * 20)
        
        fold_count = 0
        for train_idx, val_idx in skf.split(X, y):
            fold_count += 1
            
            # Verifica che non ci siano overlap
            assert len(set(train_idx) & set(val_idx)) == 0
            
            # Verifica stratificazione
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
            prop_train = (y_train_fold == 0).sum() / len(y_train_fold)
            prop_val = (y_val_fold == 0).sum() / len(y_val_fold)
            
            # Proporzioni simili
            assert abs(prop_train - 0.8) < 0.15
            assert abs(prop_val - 0.8) < 0.15
        
        assert fold_count == 5
    
    def test_class_weights_computation(self):
        """Test class weights calculation"""
        # Dataset sbilanciato
        y = np.array([0] * 890 + [1] * 110)
        
        weights = compute_class_weights(y, num_classes=2, method="balanced")
        
        assert len(weights) == 2
        assert weights[1] > weights[0]  # Classe minoritaria ha peso maggiore
        
        # Verifica formula: weight_i = n_samples / (n_classes * count_i)
        expected_w0 = 1000 / (2 * 890)
        expected_w1 = 1000 / (2 * 110)
        
        # Normalizzazione: somma = num_classes
        norm_factor = 2 / (expected_w0 + expected_w1)
        expected_w0 *= norm_factor
        expected_w1 *= norm_factor
        
        assert abs(weights[0] - expected_w0) < 0.01
        assert abs(weights[1] - expected_w1) < 0.01
    
    def test_compute_metrics(self):
        """Test metrics computation"""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1])  # 1 errore
        
        metrics = compute_metrics(y_true, y_pred)
        
        assert "accuracy" in metrics
        assert "balanced_accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        
        # Accuracy = 7/8 = 0.875
        assert abs(metrics["accuracy"] - 0.875) < 0.01
    
    def test_compute_detailed_metrics(self):
        """Test detailed metrics computation"""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        
        detailed = compute_detailed_metrics(
            y_true, y_pred,
            class_names=["DISCHARGED", "ADMITTED"]
        )
        
        assert "overall" in detailed
        assert "confusion_matrix" in detailed
        assert "per_class" in detailed
        assert "classification_report" in detailed
        
        # Verifica per-class
        assert "DISCHARGED" in detailed["per_class"]
        assert "ADMITTED" in detailed["per_class"]
        
        # Verifica confusion matrix shape
        cm = np.array(detailed["confusion_matrix"])
        assert cm.shape == (2, 2)
    
    def test_analyze_class_distribution(self):
        """Test class distribution analysis"""
        y = np.array([0] * 890 + [1] * 110)
        
        dist = analyze_class_distribution(
            y,
            class_names=["DISCHARGED", "ADMITTED"],
            print_results=False
        )
        
        assert dist["total_samples"] == 1000
        assert dist["num_classes"] == 2
        assert "DISCHARGED" in dist["classes"]
        assert "ADMITTED" in dist["classes"]
        
        # Verifica percentages
        discharged = dist["classes"]["DISCHARGED"]
        admitted = dist["classes"]["ADMITTED"]
        
        assert abs(discharged["percentage"] - 89.0) < 0.1
        assert abs(admitted["percentage"] - 11.0) < 0.1
        
        # Verifica imbalance ratio
        assert discharged["imbalance_ratio"] == 1.0  # Classe maggioritaria
        assert admitted["imbalance_ratio"] > 7.0  # ~8.1x


# =============================================================================
# 🧪 TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
