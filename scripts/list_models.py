#!/usr/bin/env python3
"""
Helper script per listare modelli disponibili da model_configs.yaml
Usato dagli script bash per menu dinamici
"""

import sys
from pathlib import Path

# Aggiungi parent directory al path per import
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_config_loader import ModelConfigLoader


def list_models_menu():
    """Stampa menu modelli per script bash"""
    try:
        loader = ModelConfigLoader()
        models = loader.get_all_models()
        
        # Separa modelli per categoria
        biomedical = []
        general = []
        
        for model_id, config in models.items():
            bio_keywords = ['bio', 'clinical', 'pubmed', 'medical', 'blue']
            is_bio = any(kw in model_id.lower() for kw in bio_keywords)
            
            if is_bio:
                biomedical.append((model_id, config))
            else:
                general.append((model_id, config))
        
        # Stampa formato menu bash
        menu_items = []
        counter = 1
        
        if biomedical:
            for model_id, config in sorted(biomedical):
                desc = config.description[:60] + "..." if len(config.description) > 60 else config.description
                menu_items.append(f"{counter}|{model_id}|{desc} [BIOMEDICAL]")
                counter += 1
        
        if general:
            for model_id, config in sorted(general):
                desc = config.description[:60] + "..." if len(config.description) > 60 else config.description
                menu_items.append(f"{counter}|{model_id}|{desc}")
                counter += 1
        
        # Output per bash (pipe-separated)
        for item in menu_items:
            print(item)
        
        return 0
        
    except FileNotFoundError:
        # Nessun config file - usa legacy
        print("1|bertm|BERT Medium - General purpose (LEGACY)")
        print("2|roberta|RoBERTa Base - Robust BERT (LEGACY)")
        print("3|cbert|Clinical BERT - Biomedical (LEGACY)")
        print("4|gpt2|GPT-2 - Generative (LEGACY)")
        return 1


def get_model_count():
    """Conta modelli disponibili"""
    try:
        loader = ModelConfigLoader()
        print(len(loader.list_model_ids()))
        return 0
    except FileNotFoundError:
        print("4")  # Legacy count
        return 1


def get_model_info(model_id):
    """Info su modello specifico"""
    try:
        loader = ModelConfigLoader()
        model = loader.get_model(model_id)
        
        if model:
            print(f"NAME={model.name}")
            print(f"HF_ID={model.hf_model_id}")
            print(f"TYPE={model.type}")
            print(f"DESC={model.description}")
            print(f"BATCH={model.recommended_batch_size}")
            print(f"LR={model.recommended_lr}")
            return 0
        else:
            print(f"ERROR=Model '{model_id}' not found")
            return 1
            
    except FileNotFoundError:
        print("ERROR=Config file not found")
        return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  list_models.py menu       - Print menu items")
        print("  list_models.py count      - Print model count")
        print("  list_models.py info <id>  - Print model info")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "menu":
        sys.exit(list_models_menu())
    elif command == "count":
        sys.exit(get_model_count())
    elif command == "info" and len(sys.argv) >= 3:
        sys.exit(get_model_info(sys.argv[2]))
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
