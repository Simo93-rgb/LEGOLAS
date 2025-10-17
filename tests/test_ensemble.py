"""
Test Suite for FASE 5: Ensemble Model

Testa:
- EnsembleModel (src/models/ensemble.py)
- Static methods: find_kfold_models, load_fold_metrics, get_best_fold
- Core methods: predict_single, predict
- Integration: con modelli bert-base-uncased narrativo reali

Run tests:
    pytest tests/test_ensemble.py -v
    pytest tests/test_ensemble.py::test_find_kfold_models -v
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModel

from src.models.ensemble import EnsembleModel
from src.models.neural_network import LongFormerMultiClassificationHeads


# =============================================================================
# 🔧 FIXTURES
# =============================================================================

@pytest.fixture
def story_format():
    """Story format per test"""
    return 'narrativo'


@pytest.fixture
def model_name():
    """Model name per test"""
    return 'bert-base-uncased'


@pytest.fixture
def device():
    """Device per test"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def num_classes(model_name, story_format):
    """
    Determina num_classes dal primo checkpoint disponibile
    """
    models_dir = Path('output/models')
    pattern = f"best_model_{story_format}_{model_name}_fold*.pth"
    fold_files = list(models_dir.glob(pattern))
    
    if not fold_files:
        pytest.skip(f"No fold models found for {model_name} {story_format}")
    
    checkpoint = torch.load(fold_files[0], map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Trova num_classes dalla forma dell'output layer
    # Cerca: output_layer, classifier, fc, o qualsiasi layer finale
    for key, tensor in state_dict.items():
        if tensor.dim() == 2:  # Layer lineare
            # Check per layer di output comuni
            if any(name in key.lower() for name in ['output_layer', 'classifier', 'fc', 'head']):
                # Il primo dim è num_classes, secondo è hidden_size
                if 'weight' in key:
                    return tensor.shape[0]
    
    pytest.skip("Could not determine num_classes from checkpoint")


@pytest.fixture
def model_factory(model_name, num_classes):
    """
    Factory per creare modelli BERT
    """
    def create_model():
        bert_model = AutoModel.from_pretrained(model_name)
        model = LongFormerMultiClassificationHeads(
            longformer=bert_model,
            num_classes=num_classes
        )
        return model
    
    return create_model


@pytest.fixture
def ensemble_model(story_format, model_name, model_factory, device):
    """
    EnsembleModel instance per test
    """
    try:
        ensemble = EnsembleModel(
            story_format=story_format,
            model_name=model_name,
            model_factory=model_factory,
            device=device
        )
        return ensemble
    except FileNotFoundError:
        pytest.skip(f"No fold models found for {model_name} {story_format}")


# =============================================================================
# 🧪 STATIC METHODS TESTS
# =============================================================================

class TestStaticMethods:
    """Test static methods di EnsembleModel"""
    
    def test_find_kfold_models(self, story_format, model_name):
        """Test auto-discovery di fold models"""
        try:
            model_paths = EnsembleModel.find_kfold_models(
                story_format=story_format,
                model_name=model_name
            )
            
            # Verifica che abbiamo trovato almeno 2 fold
            assert len(model_paths) >= 2, "Expected at least 2 folds"
            
            # Verifica che siano Path objects
            assert all(isinstance(p, Path) for p in model_paths)
            
            # Verifica che i file esistano
            assert all(p.exists() for p in model_paths)
            
            # Verifica naming pattern
            for i, path in enumerate(model_paths):
                expected_pattern = f"best_model_{story_format}_{model_name}_fold{i}.pth"
                assert path.name == expected_pattern, f"Wrong filename: {path.name}"
            
            print(f"✅ Found {len(model_paths)} fold models")
            
        except FileNotFoundError:
            pytest.skip(f"No fold models found for {model_name} {story_format}")
    
    def test_find_kfold_models_nonexistent(self):
        """Test con modello inesistente"""
        with pytest.raises(FileNotFoundError):
            EnsembleModel.find_kfold_models(
                story_format='narrativo',
                model_name='nonexistent-model-xyz'
            )
    
    def test_load_fold_metrics(self, story_format, model_name):
        """Test caricamento metriche fold"""
        try:
            # Prima trova quanti fold ci sono
            model_paths = EnsembleModel.find_kfold_models(
                story_format=story_format,
                model_name=model_name
            )
            n_folds = len(model_paths)
            
            # Carica metriche
            fold_metrics = EnsembleModel.load_fold_metrics(
                story_format=story_format,
                model_name=model_name,
                n_folds=n_folds
            )
            
            # Verifica struttura
            assert len(fold_metrics) == n_folds
            
            for i, metrics in enumerate(fold_metrics):
                # Verifica campi obbligatori
                assert 'metric' in metrics
                assert 'mode' in metrics
                assert 'best_value' in metrics
                assert 'best_epoch' in metrics
                assert 'fold' in metrics
                
                # Verifica fold index
                assert metrics['fold'] == i
                
                # Verifica metric type
                assert metrics['metric'] == 'balanced_accuracy'
                assert metrics['mode'] == 'max'
                
                # Verifica best_value è valido
                assert 0.0 <= metrics['best_value'] <= 1.0
                
            print(f"✅ Loaded metrics for {n_folds} folds")
            
        except FileNotFoundError:
            pytest.skip(f"No metrics found for {model_name} {story_format}")
    
    def test_get_best_fold(self, story_format, model_name):
        """Test identificazione best fold"""
        try:
            # Carica metriche
            model_paths = EnsembleModel.find_kfold_models(
                story_format=story_format,
                model_name=model_name
            )
            fold_metrics = EnsembleModel.load_fold_metrics(
                story_format=story_format,
                model_name=model_name,
                n_folds=len(model_paths)
            )
            
            # Trova best fold
            best_fold = EnsembleModel.get_best_fold(fold_metrics)
            
            # Verifica che sia un indice valido
            assert 0 <= best_fold < len(fold_metrics)
            
            # Verifica che sia effettivamente il migliore
            best_value = fold_metrics[best_fold]['best_value']
            all_values = [m['best_value'] for m in fold_metrics]
            assert best_value == max(all_values)
            
            print(f"✅ Best fold: {best_fold} (accuracy: {best_value:.4f})")
            
        except FileNotFoundError:
            pytest.skip(f"No metrics found for {model_name} {story_format}")


# =============================================================================
# 🧪 ENSEMBLE MODEL TESTS
# =============================================================================

class TestEnsembleModel:
    """Test EnsembleModel initialization e core methods"""
    
    def test_ensemble_initialization(self, ensemble_model):
        """Test inizializzazione ensemble"""
        assert ensemble_model is not None
        assert len(ensemble_model.models) >= 2
        assert len(ensemble_model.model_paths) == len(ensemble_model.models)
        assert len(ensemble_model.fold_metrics) == len(ensemble_model.models)
        
        # Verifica che tutti i modelli siano in eval mode
        for model in ensemble_model.models:
            assert not model.training
        
        print(f"✅ Ensemble initialized with {len(ensemble_model.models)} models")
    
    def test_predict_single(self, ensemble_model, device, num_classes):
        """Test predizione con singolo modello"""
        batch_size = 4
        seq_len = 128
        
        # Crea dummy input
        input_ids = torch.randint(0, 30522, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long).to(device)
        
        # Predict con primo modello
        probs = ensemble_model.predict_single(
            ensemble_model.models[0],
            input_ids,
            attention_mask
        )
        
        # Verifica output shape
        assert probs.shape == (batch_size, num_classes)
        
        # Verifica che siano probabilità valide
        assert torch.all(probs >= 0.0)
        assert torch.all(probs <= 1.0)
        
        # Verifica che sommino a 1 (softmax)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        
        print(f"✅ predict_single() output shape: {probs.shape}")
    
    def test_predict_ensemble(self, ensemble_model, device):
        """Test predizione ensemble"""
        batch_size = 4
        seq_len = 128
        
        # Crea dummy input
        input_ids = torch.randint(0, 30522, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        
        # Predict ensemble
        probs = ensemble_model.predict(input_ids, attention_mask)
        
        # Verifica output shape
        assert probs.shape[0] == batch_size
        assert probs.dim() == 2
        
        # Verifica che siano probabilità valide
        assert torch.all(probs >= 0.0)
        assert torch.all(probs <= 1.0)
        
        # Verifica che sommino a 1
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        
        # Verifica cache
        assert ensemble_model.probs is not None
        assert torch.allclose(ensemble_model.probs, probs)
        
        print(f"✅ predict() ensemble output shape: {probs.shape}")
    
    def test_predict_with_return_individual(self, ensemble_model, device):
        """Test predizione con return_individual=True"""
        batch_size = 4
        seq_len = 128
        
        # Crea dummy input
        input_ids = torch.randint(0, 30522, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        
        # Predict con return_individual
        ensemble_probs, individual_probs = ensemble_model.predict(
            input_ids,
            attention_mask,
            return_individual=True
        )
        
        # Verifica ensemble shape
        assert ensemble_probs.shape[0] == batch_size
        assert ensemble_probs.dim() == 2
        
        # Verifica individual shape
        n_folds = len(ensemble_model.models)
        assert individual_probs.shape == (n_folds, batch_size, ensemble_probs.shape[1])
        
        # Verifica che ensemble sia media di individual
        manual_mean = individual_probs.mean(dim=0)
        assert torch.allclose(ensemble_probs, manual_mean, atol=1e-5)
        
        print(f"✅ predict(return_individual=True)")
        print(f"   Ensemble shape: {ensemble_probs.shape}")
        print(f"   Individual shape: {individual_probs.shape}")
    
    def test_predictions_consistency(self, ensemble_model, device):
        """Test consistenza delle predizioni"""
        batch_size = 4
        seq_len = 128
        
        # Crea dummy input
        input_ids = torch.randint(0, 30522, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        
        # Esegui predict due volte
        probs1 = ensemble_model.predict(input_ids, attention_mask)
        probs2 = ensemble_model.predict(input_ids, attention_mask)
        
        # Dovrebbero essere identiche (no randomness)
        assert torch.allclose(probs1, probs2, atol=1e-7)
        
        print(f"✅ Predictions are consistent")


# =============================================================================
# 🧪 INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Test integrazione con modelli reali"""
    
    def test_with_real_bert_model(self, story_format, model_name, model_factory, device):
        """Test completo con modello bert-base-uncased reale"""
        try:
            # Initialize ensemble
            ensemble = EnsembleModel(
                story_format=story_format,
                model_name=model_name,
                model_factory=model_factory,
                device=device
            )
            
            print(f"\n{'='*60}")
            print(f"Integration Test: {model_name} {story_format}")
            print(f"{'='*60}")
            
            # Info ensemble
            print(f"\n📊 Ensemble Info:")
            print(f"   Models: {len(ensemble.models)}")
            print(f"   Device: {ensemble.device}")
            
            # Best fold info
            best_fold = EnsembleModel.get_best_fold(ensemble.fold_metrics)
            print(f"\n🏆 Best Fold: {best_fold}")
            print(f"   Accuracy: {ensemble.fold_metrics[best_fold]['best_value']:.4f}")
            print(f"   Epoch: {ensemble.fold_metrics[best_fold]['best_epoch']}")
            
            # All folds info
            print(f"\n📈 All Folds Performance:")
            for i, metrics in enumerate(ensemble.fold_metrics):
                marker = "⭐" if i == best_fold else "  "
                print(f"   {marker} Fold {i}: {metrics['best_value']:.4f} (epoch {metrics['best_epoch']})")
            
            # Test prediction
            print(f"\n🔄 Testing prediction...")
            batch_size = 2
            seq_len = 64
            
            input_ids = torch.randint(0, 30522, (batch_size, seq_len))
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
            
            probs = ensemble.predict(input_ids, attention_mask)
            predictions = probs.argmax(dim=-1)
            
            print(f"   Input shape: {input_ids.shape}")
            print(f"   Output shape: {probs.shape}")
            print(f"   Predictions: {predictions.cpu().tolist()}")
            print(f"   Confidence: {probs.max(dim=-1)[0].cpu().tolist()}")
            
            print(f"\n✅ Integration test PASSED")
            print(f"{'='*60}\n")
            
        except FileNotFoundError as e:
            pytest.skip(f"Models not found: {e}")


if __name__ == "__main__":
    # Per eseguire i test direttamente
    pytest.main([__file__, "-v", "-s"])
