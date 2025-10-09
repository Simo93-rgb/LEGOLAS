"""
Test per verificare refactoring num_classes
Verifica che LongFormerMultiClassificationHeads e SimpleGPT2SequenceClassifier
supportino numero classi configurabile (FASE 4.3.9)
"""

import pytest
import torch
from transformers import AutoModel
from src.models.neural_network import LongFormerMultiClassificationHeads, SimpleGPT2SequenceClassifier


class TestLongFormerNumClasses:
    """Test LongFormerMultiClassificationHeads con diverse num_classes"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        """Fixture: carica bert-base-uncased una volta per tutti i test"""
        return AutoModel.from_pretrained("bert-base-uncased")
    
    @pytest.mark.parametrize("num_classes", [2, 3, 8, 10])
    def test_output_layer_size(self, base_model, num_classes):
        """Verifica che output layer abbia dimensione corretta"""
        model = LongFormerMultiClassificationHeads(
            longformer=base_model,
            num_classes=num_classes
        )
        
        assert model.num_classes == num_classes
        assert model.output_layer.out_features == num_classes
    
    @pytest.mark.parametrize("num_classes", [2, 3, 8, 10])
    def test_forward_pass_shape(self, base_model, num_classes):
        """Verifica shape output del forward pass"""
        model = LongFormerMultiClassificationHeads(
            longformer=base_model,
            num_classes=num_classes
        )
        
        # Dummy input
        dummy_input = torch.randint(0, 1000, (2, 10))  # batch=2, seq_len=10
        dummy_mask = torch.ones((2, 10))
        
        output = model(dummy_input, dummy_mask)
        
        assert output.shape == (2, num_classes)
    
    def test_default_num_classes(self, base_model):
        """Verifica backward compatibility: default num_classes=8"""
        model = LongFormerMultiClassificationHeads(longformer=base_model)
        
        assert model.num_classes == 8
        assert model.output_layer.out_features == 8


class TestGPT2NumClasses:
    """Test SimpleGPT2SequenceClassifier con diverse num_classes"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        """Fixture: carica gpt2 una volta per tutti i test"""
        return AutoModel.from_pretrained("gpt2")
    
    @pytest.mark.parametrize("num_classes", [2, 3, 8, 10])
    def test_fc1_layer_size(self, base_model, num_classes):
        """Verifica che FC1 layer abbia dimensione corretta"""
        model = SimpleGPT2SequenceClassifier(
            hidden_size=768,
            num_classes=num_classes,
            max_seq_len=512,
            gpt_model=base_model
        )
        
        expected_in = 768 * 512
        assert model.fc1.in_features == expected_in
        assert model.fc1.out_features == num_classes
    
    @pytest.mark.parametrize("num_classes", [2, 3, 8, 10])
    def test_forward_pass_shape(self, base_model, num_classes):
        """Verifica shape output del forward pass"""
        model = SimpleGPT2SequenceClassifier(
            hidden_size=768,
            num_classes=num_classes,
            max_seq_len=512,
            gpt_model=base_model
        )
        
        # Dummy input
        dummy_input = torch.randint(0, 1000, (2, 512))  # batch=2, seq_len=512
        dummy_mask = torch.ones((2, 512))
        
        output = model(dummy_input, dummy_mask)
        
        assert output.shape == (2, num_classes)


class TestTrainingConfigIntegration:
    """Test integrazione con TrainingConfig"""
    
    def test_config_num_classes_binary(self):
        """Verifica TrainingConfig per classificazione binaria"""
        from src.training.config import TrainingConfig
        
        config = TrainingConfig(num_classes=2)
        assert config.num_classes == 2
    
    def test_config_num_classes_multiclass(self):
        """Verifica TrainingConfig per classificazione multi-classe"""
        from src.training.config import TrainingConfig
        
        config = TrainingConfig(num_classes=5)
        assert config.num_classes == 5
    
    def test_focal_alpha_validation(self):
        """Verifica validazione focal_alpha con num_classes"""
        from src.training.config import TrainingConfig
        
        # Deve fallire: focal_alpha ha 2 elementi, num_classes=3
        config = TrainingConfig(
            num_classes=3,
            loss_function='focal',
            focal_alpha=[0.25, 0.75]  # solo 2 elementi!
        )
        
        # validate() dovrebbe rilevare mismatch
        assert not config.validate()
    
    def test_focal_alpha_correct(self):
        """Verifica focal_alpha corretto per num_classes"""
        from src.training.config import TrainingConfig
        
        config = TrainingConfig(
            num_classes=3,
            loss_function='focal',
            focal_alpha=[0.2, 0.3, 0.5]  # 3 elementi
        )
        
        assert config.validate()
