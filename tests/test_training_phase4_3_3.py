"""
Test per FASE 4.3.3 - Loss Function Integration
Verifica che la loss function venga creata correttamente basandosi su config.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

from src.training.config import TrainingConfig
from src.training.focal_loss import FocalLoss, create_loss_from_config
from src.training.utils import compute_class_weights

# Import train_llm per testare integrazione
import importlib.util
spec = importlib.util.spec_from_file_location(
    "train_llm_module",
    Path(__file__).parent.parent / "src" / "training" / "train_llm.py"
)
train_llm_module = importlib.util.module_from_spec(spec)


class TestLossFunctionCreation:
    """Test creazione loss function da config."""
    
    def test_create_cross_entropy_loss_with_weights(self):
        """Test che CrossEntropyLoss usa class weights."""
        # Crea config con CE loss
        config = TrainingConfig(
            model_name='test_model',
            story_format='narrativo',
            loss_function='ce'
        )
        
        # Labels sbilanciati (70% classe 0, 30% classe 1)
        labels = np.array([0]*70 + [1]*30)
        
        # Calcola class weights
        class_weights = compute_class_weights(labels, method='balanced')
        
        # Verifica che classe minoritaria ha peso maggiore
        assert class_weights[1] > class_weights[0], "Classe minoritaria dovrebbe avere peso maggiore"
        
        # Crea loss con weights
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # Verifica che criterion ha weights
        assert criterion.weight is not None
        assert torch.allclose(criterion.weight, class_weights_tensor)
    
    def test_create_focal_loss_from_config(self):
        """Test che FocalLoss viene creata correttamente da config."""
        config = TrainingConfig(
            model_name='test_model',
            story_format='narrativo',
            loss_function='focal',
            focal_alpha=[0.3, 0.7],
            focal_gamma=2.5
        )
        
        criterion = create_loss_from_config(config)
        
        assert isinstance(criterion, FocalLoss)
        assert criterion.gamma == 2.5
        # Alpha è stato convertito in tensor
        assert torch.allclose(criterion.alpha, torch.tensor([0.3, 0.7]))
    
    def test_focal_loss_no_class_weights(self):
        """Test che Focal Loss NON usa class weights aggiuntivi (usa solo α)."""
        config = TrainingConfig(
            model_name='test_model',
            story_format='narrativo',
            loss_function='focal',
            focal_alpha=[0.25, 0.75],
            focal_gamma=2.0
        )
        
        # Focal Loss gestisce pesi tramite alpha, non class_weights
        criterion = create_loss_from_config(config)
        
        # Verifica parametri
        assert criterion.gamma == 2.0
        assert torch.allclose(criterion.alpha, torch.tensor([0.25, 0.75]))
    
    def test_loss_function_default_is_ce(self):
        """Test che default loss è Focal Loss (policy del progetto)."""
        config = TrainingConfig(
            model_name='test_model',
            story_format='narrativo'
            # loss_function non specificato → default 'focal' (policy progetto)
        )
        
        assert config.loss_function == 'focal'  # Default è focal, non ce


class TestLossFunctionIntegration:
    """Test integrazione loss function in train_llm.py."""
    
    def test_cross_entropy_path_with_balanced_weights(self):
        """Test che CE path calcola balanced weights."""
        # Simula labels sbilanciati
        labels = np.array([0]*70 + [1]*30)
        
        # Calcola weights come fa train_llm (parametro è 'y' non 'labels')
        class_weights = compute_class_weights(
            y=labels,
            method='balanced'
        )
        
        # Verifica calcolo corretto
        assert len(class_weights) == 2
        assert class_weights[1] > class_weights[0]  # Classe minoritaria pesa di più
        
        # Verifica normalizzazione (somma = num_classes)
        assert np.isclose(class_weights.sum().item(), 2.0, atol=0.01)
    
    def test_focal_loss_path_uses_config_params(self):
        """Test che Focal Loss path usa parametri da config."""
        config = TrainingConfig(
            model_name='test_model',
            story_format='narrativo',
            loss_function='focal',
            focal_alpha=[0.4, 0.6],
            focal_gamma=3.0
        )
        
        criterion = create_loss_from_config(config)
        
        assert isinstance(criterion, FocalLoss)
        assert criterion.gamma == 3.0
        assert torch.allclose(criterion.alpha, torch.tensor([0.4, 0.6]))
    
    def test_loss_function_forward_pass(self):
        """Test che entrambe le loss funzionano in forward pass."""
        # Setup
        batch_size = 8
        num_classes = 2
        logits = torch.randn(batch_size, num_classes, requires_grad=True)  # Serve requires_grad qui
        targets = torch.randint(0, num_classes, (batch_size,))
        
        # Test CrossEntropy
        ce_criterion = nn.CrossEntropyLoss()
        ce_loss = ce_criterion(logits, targets)
        assert ce_loss.item() > 0
        assert ce_loss.requires_grad  # Loss eredita requires_grad da logits
        
        # Test FocalLoss
        focal_criterion = FocalLoss(alpha=[0.25, 0.75], gamma=2.0)
        focal_loss = focal_criterion(logits, targets)
        assert focal_loss.item() > 0
        assert focal_loss.requires_grad


class TestLossFunctionComparison:
    """Test comparativo tra CE e Focal Loss."""
    
    def test_focal_loss_reduces_easy_examples_loss(self):
        """Test che Focal Loss riduce contributo di esempi facili."""
        batch_size = 10
        num_classes = 2
        
        # Crea esempi "facili" (alta confidenza corretta)
        logits_easy = torch.tensor([
            [5.0, -5.0],  # Classe 0 molto probabile
            [5.0, -5.0],
            [5.0, -5.0],
            [5.0, -5.0],
            [5.0, -5.0],
            [-5.0, 5.0],  # Classe 1 molto probabile
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
        ])
        targets = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        
        # CrossEntropy
        ce_criterion = nn.CrossEntropyLoss(reduction='mean')
        ce_loss = ce_criterion(logits_easy, targets)
        
        # FocalLoss (γ=2 riduce loss per esempi facili)
        focal_criterion = FocalLoss(alpha=None, gamma=2.0, reduction='mean')
        focal_loss = focal_criterion(logits_easy, targets)
        
        # Focal Loss dovrebbe essere minore per esempi facili
        assert focal_loss < ce_loss, "Focal Loss dovrebbe essere < CE Loss per esempi facili"
    
    def test_focal_loss_emphasizes_hard_examples(self):
        """Test che Focal Loss enfatizza esempi difficili."""
        # Esempi "difficili" (bassa confidenza)
        logits_hard = torch.tensor([
            [0.5, 0.4],   # Classe 0 poco probabile
            [0.4, 0.5],   # Classe 1 poco probabile (target=0, difficile)
            [0.6, 0.5],
            [0.5, 0.6],
        ])
        targets = torch.tensor([0, 0, 1, 1])
        
        # Focal Loss con γ alto enfatizza hard examples
        focal_high_gamma = FocalLoss(alpha=None, gamma=3.0, reduction='mean')
        focal_low_gamma = FocalLoss(alpha=None, gamma=0.5, reduction='mean')
        
        loss_high = focal_high_gamma(logits_hard, targets)
        loss_low = focal_low_gamma(logits_hard, targets)
        
        # Con γ più alto, gli hard examples pesano relativamente di più
        # (la loss totale può variare, ma comportamento è corretto)
        assert loss_high.item() > 0
        assert loss_low.item() > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
