import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from src.data.history_dataset import TextDataset
from src.models.neural_network import LongFormerMultiClassificationHeads, SimpleGPT2SequenceClassifier
from src.utils.model_config_loader import ModelConfigLoader
from src.config.paths import get_story_file_path, get_model_path, ensure_directories
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from transformers import (TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
from transformers import BertModel, BertConfig

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using CUDA
    np.random.seed(seed)
    random.seed(seed)


import os
from pathlib import Path
from typing import Optional

HF_DEFAULT_HOME = os.environ.get("HF_HOME", "~/.cache/huggingface/hub")


def get_weight_dir(
        model_ref: str,
        *,
        model_dir=HF_DEFAULT_HOME,
        revision: str = "main", ) -> Path:
    """
    Parse model name to locally stored weights.
    Args:
        model_ref (str) : Model reference containing org_name/model_name such as 'meta-llama/Llama-2-7b-chat-hf'.
        revision (str): Model revision branch. Defaults to 'main'.
        model_dir (str | os.PathLike[Any]): Path to directory where models are stored. Defaults to value of $HF_HOME (or present directory)

    Returns:
        str: path to model weights within model directory
    """
    model_dir = Path(model_dir)
    assert model_dir.is_dir()
    model_path = model_dir / "--".join(["models", *model_ref.split("/")])
    assert model_path.is_dir()
    snapshot_hash = (model_path / "refs" / revision).read_text()
    weight_dir = model_path / "snapshots" / snapshot_hash
    assert weight_dir.is_dir()
    return weight_dir

def pre_train(model, optimizer, train_dataloader, val_dataloader, scheduler, patience, epochs, best_loss, model_name):
        # Training loop
        patience_counter = 0
        for epoch in range(epochs):  # number of epochs
            model.train()
            for batch in train_dataloader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)  # Assuming labels are 0 or 1
                output = model(input_ids, attention_mask)
                loss = criterion(output, labels)
                accelerator.backward(loss)
                optimizer.step()

            # Validation phase
            model.eval()
            with torch.no_grad():
                cumulated_loss = torch.as_tensor([0.0]).to(accelerator.device)

                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)  # Assuming labels are 0 or 1
                    output = model(input_ids, attention_mask)
                    loss = criterion(output, labels)
                    cumulated_loss = cumulated_loss + loss

            cumulated_loss = accelerator.gather(cumulated_loss).cpu().mean().item()

            if accelerator.is_main_process:
                avg_cumulated_loss = cumulated_loss / len(val_dataloader)
                print(f"Epoch {epoch + 1}, Validation Loss: {avg_cumulated_loss:.4f}")
                if avg_cumulated_loss < best_loss:
                    accelerator.set_trigger()
                    best_loss = avg_cumulated_loss
                    patience_counter = 0
                    unwrapped_model = accelerator.unwrap_model(model)
                    # Usa path centralizzato per salvataggio modello
                    from src.config.paths import MODELS_DIR
                    model_save_path = MODELS_DIR / (model_name + str(epoch + 1) + '.pth')
                    accelerator.save(unwrapped_model.state_dict(), str(model_save_path))
                else:
                    patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            accelerator.wait_for_everyone()



if __name__ == '__main__':
    seed = 42
    LEARNING_RATE = 1e-5
    BATCH = 32  # Ridotto da 256 per evitare OOM su GPU
    
    # ============================================================
    # CONFIGURAZIONE: Scegli il formato delle storie generate
    # ============================================================
    # Opzioni disponibili: 'narrativo', 'bullet', 'clinical'
    STORY_FORMAT = 'narrativo'  # <-- CAMBIA QUI per usare formato diverso
    
    print(f"\n{'='*60}")
    print(f"  LEGOLAS - Training su Storie XES Generate")
    print(f"  Formato: {STORY_FORMAT}")
    print(f"{'='*60}\n")
    
    set_seed(seed)
    
    # Assicura che le directory esistano
    ensure_directories()
    
    # Carica file generati dalla pipeline XES
    print(f"📖 Caricamento storie formato '{STORY_FORMAT}'...")
    
    train_path = get_story_file_path(STORY_FORMAT, 'train')
    with open(train_path, 'rb') as f:
        train = pickle.load(f)
    print(f"   ✅ Training stories: {len(train)} campioni")

    test_path = get_story_file_path(STORY_FORMAT, 'test')
    with open(test_path, 'rb') as f:
        test = pickle.load(f)
    print(f"   ✅ Test stories: {len(test)} campioni")

    label_train_path = get_story_file_path(STORY_FORMAT, 'label_train')
    with open(label_train_path, 'rb') as f:
        label_train = pickle.load(f)

    label_test_path = get_story_file_path(STORY_FORMAT, 'label_test')
    with open(label_test_path, 'rb') as f:
        label_test = pickle.load(f)

    label2id = {}
    id2label = {}
    i = 0
    for l in list(np.unique(label_train)):
        label2id[l] = i
        id2label[i] = l
        i = i + 1

    label_train_int = []
    for l in label_train:
        label_train_int.append(label2id[l])

    label_test_int = []
    for l in label_test:
        label_test_int.append(label2id[l])


    X_train, X_val, y_train, y_val = train_test_split(
                                                    train, label_train_int,
                                                      test_size=0.2, random_state=42, shuffle=True,
                                                      stratify=label_train_int)


    model_name = 'bert-base-uncased'

    # Carica configurazioni modelli da YAML
    print(f'\n📋 Caricamento configurazioni modelli da YAML...')
    try:
        config_loader = ModelConfigLoader()
        available_models = config_loader.list_model_ids()
        print(f'✅ {len(available_models)} modelli caricati')
        
        # Mostra modelli biomedici disponibili
        bio_models = config_loader.get_biomedical_models()
        print(f'   • {len(bio_models)} modelli biomedici specializzati')
        
    except FileNotFoundError as e:
        print(f'⚠️  Config YAML non trovato: {e}')
        print(f'   Uso configurazioni legacy hardcoded')
        
        # Fallback a mappa legacy
        model_map = {
            'bertm': 'prajjwal1/bert-medium',
            'roberta': 'FacebookAI/roberta-base',
            'gpt2': 'openai-community/gpt2',
            'cbert': 'emilyalsentzer/Bio_ClinicalBERT'
        }
        config_loader = None
        available_models = list(model_map.keys())
    
    # Verifica che il modello richiesto esista
    if config_loader:
        model_config = config_loader.get_model(model_name)
        if not model_config:
            print(f'❌ Modello non trovato: {model_name}')
            print(f'   Modelli disponibili:')
            for model_id in available_models:
                m = config_loader.get_model(model_id)
                print(f'      • {model_id}: {m.description}')
            exit(1)
        
        model_ref = model_config.hf_model_id
        recommended_params = config_loader.get_recommended_params(model_name)
        
        print(f'\n🤖 Modello selezionato: {model_name}')
        print(f'   HuggingFace ID: {model_ref}')
        print(f'   Tipo: {model_config.type}')
        print(f'   Descrizione: {model_config.description}')
        print(f'   📊 Parametri raccomandati:')
        print(f'      Batch size: {recommended_params["batch_size"]}')
        print(f'      Learning rate: {recommended_params["learning_rate"]}')
        print(f'      Max length: {recommended_params["max_length"]}')
        
    else:
        # Usa legacy model_map
        if model_name not in model_map:
            print(f'❌ Modello non trovato: {model_name}')
            print(f'   Modelli disponibili: {list(model_map.keys())}')
            exit(1)
        model_ref = model_map[model_name]
    
    # Prova a usare cache locale, altrimenti scarica da HuggingFace
    try:
        weights_dir = get_weight_dir(model_ref)
        print(f'✅ Modello trovato in cache locale')
    except (AssertionError, FileNotFoundError):
        print(f'⚠️  Modello non in cache, verrà scaricato da HuggingFace')
        weights_dir = model_ref  # Usa direttamente il model ID
    
    print(f'\n🚀 TRAINING START...')
    print(f'   Modello: {model_name} ({model_ref})')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from accelerate import Accelerator

    accelerator = Accelerator(split_batches=True)
    device = accelerator.device

    if model_name =='gpt2':

        model = AutoModel.from_pretrained(weights_dir)
        model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=weights_dir, num_labels=8)
            # Get model's tokenizer.
        print('Loading tokenizer...')
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=weights_dir, truncation_side='left')
            # default to left padding
        tokenizer.padding_side = "left"
            # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))
        # fix model padding token id
        model.config.pad_token_id = model.config.eos_token_id
        model = SimpleGPT2SequenceClassifier(hidden_size=768, num_classes=8, max_seq_len=512, gpt_model=model).to(device)

    else:
        tokenizer = AutoTokenizer.from_pretrained(weights_dir, truncation_side='left')
        model = AutoModel.from_pretrained(weights_dir)
        model = LongFormerMultiClassificationHeads(model)
        model = model.to(device)


    train_dataset = TextDataset(X_train, y_train, tokenizer, 512)
    val_dataset = TextDataset(X_val, y_val, tokenizer, 512)

    # print(train_dataset.__getitem__(5))

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False)

    print('device-->', device)
    print(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    startTime = time.time()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)
    
    # Nome modello include formato delle storie (usa solo basename senza path)
    model_output_basename = f'xes_{STORY_FORMAT}_{model_name}'
    print(f"\n💾 Modello verrà salvato come: {model_output_basename}\n")
    
    # Il path completo sarà generato da get_model_path nella funzione pre_train
    pre_train(model, optimizer, train_dataloader, val_dataloader, scheduler, 1, 5, float('inf'), model_output_basename)



