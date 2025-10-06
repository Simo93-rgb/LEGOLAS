"""
Model Configuration Loader per LEGOLAS
Gestisce caricamento e validazione configurazioni modelli da YAML
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configurazione per un singolo modello"""
    name: str
    hf_model_id: str
    type: str
    max_length: int
    num_labels: int
    description: str
    recommended_batch_size: int
    recommended_lr: float
    
    @property
    def display_name(self) -> str:
        """Nome user-friendly per display"""
        return self.name.replace('-', ' ').title()


class ModelConfigLoader:
    """Carica e gestisce configurazioni modelli da YAML"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inizializza il loader
        
        Args:
            config_path: Path al file YAML (default: config/model_configs.yaml)
        """
        if config_path is None:
            # Cerca config in diverse posizioni
            possible_paths = [
                Path("config/model_configs.yaml"),
                Path("../config/model_configs.yaml"),
                Path("../../config/model_configs.yaml"),
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break
        
        if config_path is None or not Path(config_path).exists():
            raise FileNotFoundError(
                f"Config file non trovato. Cercato in: {possible_paths}"
            )
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.models = self._parse_models()
    
    def _load_config(self) -> Dict:
        """Carica il file YAML"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _parse_models(self) -> Dict[str, ModelConfig]:
        """Parse configurazioni modelli"""
        models = {}
        
        for model_id, config in self.config.get('models', {}).items():
            models[model_id] = ModelConfig(
                name=model_id,
                hf_model_id=config['name'],
                type=config['type'],
                max_length=config['max_length'],
                num_labels=config['num_labels'],
                description=config.get('description', ''),
                recommended_batch_size=config.get('recommended_batch_size', 12),
                recommended_lr=config.get('recommended_lr', 2e-5)
            )
        
        return models
    
    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Ottieni configurazione per un modello specifico"""
        return self.models.get(model_id)
    
    def get_all_models(self) -> Dict[str, ModelConfig]:
        """Ottieni tutte le configurazioni modelli"""
        return self.models
    
    def list_model_ids(self) -> List[str]:
        """Lista ID di tutti i modelli disponibili"""
        return list(self.models.keys())
    
    def list_models_by_type(self, model_type: str) -> List[ModelConfig]:
        """
        Lista modelli filtrati per tipo
        
        Args:
            model_type: Tipo modello (es: 'bert', 'longformer')
        """
        return [
            model for model in self.models.values()
            if model.type == model_type
        ]
    
    def get_biomedical_models(self) -> List[ModelConfig]:
        """Ottieni solo modelli biomedici specializzati"""
        biomedical_keywords = ['bio', 'clinical', 'pubmed', 'medical', 'blue']
        
        return [
            model for model in self.models.values()
            if any(kw in model.name.lower() for kw in biomedical_keywords)
        ]
    
    def get_training_config(self) -> Dict[str, Any]:
        """Ottieni configurazione training di default"""
        return self.config.get('training_defaults', {})
    
    def get_recommended_params(self, model_id: str) -> Dict[str, Any]:
        """
        Ottieni parametri raccomandati per un modello
        
        Returns:
            Dict con batch_size, learning_rate, max_length
        """
        model = self.get_model(model_id)
        if not model:
            return {}
        
        return {
            'batch_size': model.recommended_batch_size,
            'learning_rate': model.recommended_lr,
            'max_length': model.max_length,
            'num_labels': model.num_labels
        }
    
    def print_available_models(self, model_type: Optional[str] = None):
        """
        Stampa lista modelli disponibili
        
        Args:
            model_type: Filtra per tipo (opzionale)
        """
        if model_type:
            models = self.list_models_by_type(model_type)
            print(f"\n📋 Modelli disponibili (tipo: {model_type}):")
        else:
            models = list(self.models.values())
            print("\n📋 Modelli disponibili:")
        
        print("=" * 80)
        
        for model in models:
            print(f"\n🤖 {model.name}")
            print(f"   HuggingFace: {model.hf_model_id}")
            print(f"   Tipo: {model.type}")
            print(f"   Descrizione: {model.description}")
            print(f"   Batch consigliato: {model.recommended_batch_size}")
            print(f"   Learning rate: {model.recommended_lr}")


# Funzione di comodo per uso rapido
def load_model_config(model_id: str, config_path: Optional[str] = None) -> Optional[ModelConfig]:
    """
    Carica configurazione per un modello specifico
    
    Args:
        model_id: ID del modello
        config_path: Path al file config (opzionale)
    
    Returns:
        ModelConfig o None se non trovato
    """
    try:
        loader = ModelConfigLoader(config_path)
        return loader.get_model(model_id)
    except FileNotFoundError as e:
        print(f"⚠️  Errore caricamento config: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Test del loader
    loader = ModelConfigLoader()
    
    print("=" * 80)
    print("TEST MODEL CONFIG LOADER")
    print("=" * 80)
    
    # Lista tutti i modelli
    loader.print_available_models()
    
    # Test modello specifico
    print("\n" + "=" * 80)
    print("TEST MODELLO SPECIFICO: clinical-bert")
    print("=" * 80)
    
    model = loader.get_model('clinical-bert')
    if model:
        print(f"\n✅ Modello trovato:")
        print(f"   Nome: {model.name}")
        print(f"   HF ID: {model.hf_model_id}")
        print(f"   Descrizione: {model.description}")
        
        params = loader.get_recommended_params('clinical-bert')
        print(f"\n📊 Parametri raccomandati:")
        print(f"   {params}")
    
    # Test modelli biomedici
    print("\n" + "=" * 80)
    print("MODELLI BIOMEDICI")
    print("=" * 80)
    
    bio_models = loader.get_biomedical_models()
    print(f"\nTrovati {len(bio_models)} modelli biomedici:")
    for model in bio_models:
        print(f"  • {model.name}: {model.description}")
