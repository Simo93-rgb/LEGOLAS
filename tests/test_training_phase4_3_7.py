import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import tempfile

class TestDataLoadingAndPaths:
    """Test creazione label_mapping.json e paths"""
    
    def test_label_mapping_json_created(self, tmp_path):
        """Verifica che label_mapping.json viene creato"""
        # Mock config con tmp_path
        # Mock pickle.load per train/test data
        # Esegui data loading section
        # Assert: label_mapping.json esiste
        
    def test_label_mapping_format(self, tmp_path):
        """Verifica formato JSON corretto"""
        # Carica label_mapping.json
        # Assert: "label2id", "id2label", "num_classes" presenti
        # Assert: format "CLS_0", "CLS_1"

class TestSimpleTrainingIntegration:
    """Test workflow simple training (no K-Fold)"""
    
    @patch('src.training.train_llm.pickle.load')
    @patch('src.training.train_llm.pre_train')
    def test_simple_training_workflow(self, mock_pre_train, mock_pickle):
        """Test workflow completo simple training"""
        # Mock config.use_kfold = False
        # Mock pickle files
        # Mock pre_train
        # Esegui main section
        # Assert: pre_train chiamato con fold=None

class TestKFoldIntegration:
    """Test integrazione K-Fold"""
    
    @patch('src.training.train_llm.KFoldTrainer')
    def test_kfold_routing(self, mock_kfold_trainer):
        """Verifica routing a KFoldTrainer quando use_kfold=True"""
        # Mock config.use_kfold = True
        # Mock KFoldTrainer
        # Esegui main section
        # Assert: KFoldTrainer.run() chiamato

class TestGPUHandling:
    """Test gestione GPU/CPU"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_device_selection(self):
        """Verifica selezione GPU quando disponibile"""
        # Mock config con device='cuda'
        # Assert: model.to('cuda') chiamato