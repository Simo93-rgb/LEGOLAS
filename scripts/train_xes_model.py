"""
Esempio di adattamento di train_llm.py per usare le storie generate dalla pipeline XES.

Questo file mostra come modificare train_llm.py per caricare i file pickle
generati da generate_stories.py invece dei file MIMIC originali.
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from history_dataset import TextDataset
from neural_network import LongFormerMultiClassificationHeads
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path


def load_xes_generated_stories(prefix: str = "narrativo", output_dir: str = "output"):
    """
    Carica le storie generate dalla pipeline XES.
    
    Args:
        prefix: Prefisso dei file (narrativo, bullet, clinical, etc.)
        output_dir: Directory contenente i file pickle
        
    Returns:
        Tuple (train_stories, test_stories, train_labels, test_labels)
    """
    output_path = Path(output_dir)
    
    print(f"📖 Caricamento storie generate (prefisso: {prefix})")
    
    with open(output_path / f'{prefix}_train.pkl', 'rb') as f:
        train = pickle.load(f)
    
    with open(output_path / f'{prefix}_test.pkl', 'rb') as f:
        test = pickle.load(f)
    
    with open(output_path / f'{prefix}_label_train.pkl', 'rb') as f:
        label_train = pickle.load(f)
    
    with open(output_path / f'{prefix}_label_test.pkl', 'rb') as f:
        label_test = pickle.load(f)
    
    print(f"✅ Caricate {len(train)} storie di training")
    print(f"✅ Caricate {len(test)} storie di test")
    
    return train, test, label_train, label_test


def train_model_on_xes_stories(
    prefix: str = "narrativo",
    model_name: str = "bertm",
    learning_rate: float = 1e-5,
    batch_size: int = 256,
    epochs: int = 5,
    seed: int = 42
):
    """
    Addestra un modello sulle storie generate dalla pipeline XES.
    
    Args:
        prefix: Prefisso dei file di storie generate
        model_name: Nome del modello ('bertm', 'roberta', 'gpt2', 'cbert')
        learning_rate: Learning rate per l'ottimizzatore
        batch_size: Dimensione del batch
        epochs: Numero di epoche di training
        seed: Random seed per riproducibilità
    """
    # Set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Carica storie generate
    train, test, label_train, label_test = load_xes_generated_stories(prefix)
    
    # Crea mapping label→id
    label2id = {}
    id2label = {}
    for i, label in enumerate(sorted(np.unique(label_train))):
        label2id[label] = i
        id2label[i] = label
    
    print(f"\n📊 Label mapping:")
    print(f"   {label2id}")
    
    # Converti label in interi
    label_train_int = [label2id[l] for l in label_train]
    label_test_int = [label2id[l] for l in label_test]
    
    # Split validation set
    X_train, X_val, y_train, y_val = train_test_split(
        train, label_train_int,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
        stratify=label_train_int
    )
    
    print(f"\n📊 Dataset split:")
    print(f"   - Training: {len(X_train)}")
    print(f"   - Validation: {len(X_val)}")
    print(f"   - Test: {len(test)}")
    
    # Carica modello e tokenizer
    print(f"\n🤖 Caricamento modello: {model_name}")
    
    from train_llm import get_weight_dir
    
    model_paths = {
        'bertm': 'prajjwal1/bert-medium',
        'roberta': 'FacebookAI/roberta-base',
        'cbert': 'emilyalsentzer/Bio_ClinicalBERT'
    }
    
    if model_name not in model_paths:
        raise ValueError(f"Modello non supportato: {model_name}")
    
    weights_dir = get_weight_dir(model_paths[model_name])
    
    tokenizer = AutoTokenizer.from_pretrained(weights_dir, truncation_side='left')
    model = AutoModel.from_pretrained(weights_dir)
    model = LongFormerMultiClassificationHeads(model)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"   Device: {device}")
    
    # Crea dataset e dataloader
    train_dataset = TextDataset(X_train, y_train, tokenizer, 512)
    val_dataset = TextDataset(X_val, y_val, tokenizer, 512)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    print(f"\n🚀 Inizio training...")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Epochs: {epochs}")
    
    # Import pre_train da train_llm.py
    from train_llm import pre_train
    from accelerate import Accelerator
    
    accelerator = Accelerator(split_batches=True)
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    # Training
    pre_train(
        model, optimizer, train_loader, val_loader, scheduler,
        epochs=epochs,
        patience=5,
        best_val_loss=float('inf'),
        model_name=f'xes_{prefix}_{model_name}'
    )
    
    print(f"\n✅ Training completato!")
    print(f"   Modello salvato come: xes_{prefix}_{model_name}")


if __name__ == '__main__':
    """
    Esempio di utilizzo:
    
    # Usa storie narrative
    python train_xes_model.py
    
    # Oppure modifica i parametri:
    """
    
    # CONFIGURAZIONE
    STORY_PREFIX = "narrativo"  # Cambia in "bullet" o "clinical" se necessario
    MODEL_NAME = "bertm"        # 'bertm', 'roberta', 'cbert'
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 256
    EPOCHS = 5
    SEED = 42
    
    print("═" * 60)
    print("  LEGOLAS - Training su Storie XES Generate")
    print("═" * 60)
    
    train_model_on_xes_stories(
        prefix=STORY_PREFIX,
        model_name=MODEL_NAME,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        seed=SEED
    )
