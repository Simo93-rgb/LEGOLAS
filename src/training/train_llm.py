import pickle
import pandas as pd
import json
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
import argparse
from pathlib import Path
from typing import Optional
from transformers import (TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
from transformers import BertModel, BertConfig

# Import nuove classi training
from src.training.config import TrainingConfig
from src.training.checkpoint import ModelCheckpoint
from src.training.early_stopping import EarlyStopping
from src.training.focal_loss import create_loss_from_config
from src.training.utils import compute_class_weights, compute_metrics, analyze_class_distribution
from src.training.kfold_trainer import KFoldTrainer

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using CUDA
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description='LEGOLAS - Training LLM su storie XES',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Nome del modello (es: bertm, cbert, roberta, gpt2)'
    )
    parser.add_argument(
        '--story_format',
        type=str,
        required=True,
        choices=['narrativo', 'bullet', 'clinical'],
        help='Formato delle storie generate'
    )
    
    # Training mode
    parser.add_argument(
        '--use_kfold',
        action='store_true',
        help='Abilita K-Fold Cross Validation'
    )
    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='Numero di fold per K-Fold CV'
    )
    
    # Loss function
    parser.add_argument(
        '--use_focal_loss',
        action='store_true',
        help='Usa Focal Loss invece di Cross Entropy'
    )
    parser.add_argument(
        '--focal_alpha',
        type=float,
        nargs='+',
        default=[0.25, 0.75],
        help='Pesi alpha per Focal Loss (uno per classe o scalare)'
    )
    parser.add_argument(
        '--focal_gamma',
        type=float,
        default=2.0,
        help='Parametro gamma per Focal Loss'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Numero massimo di epoch'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size per training e validation'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5e-5,
        help='Learning rate per optimizer'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=3,
        help='Patience per early stopping'
    )
    
    # Seed
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed per riproducibilità'
    )
    
    return parser.parse_args()


def create_training_config(args) -> TrainingConfig:
    """Crea TrainingConfig da argomenti CLI."""
    config = TrainingConfig(
        model_name=args.model_name,
        story_format=args.story_format,
        num_epochs=args.epochs,  # CLI usa 'epochs', config usa 'num_epochs'
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_kfold=args.use_kfold,
        n_folds=args.n_folds if args.use_kfold else 1,
        loss_function='focal' if args.use_focal_loss else 'ce',  # CLI flag → loss_function
        focal_alpha=args.focal_alpha if args.use_focal_loss else [0.25, 0.75],
        focal_gamma=args.focal_gamma if args.use_focal_loss else 2.0,
        early_stopping_patience=args.patience,  # CLI usa 'patience', config usa 'early_stopping_patience'
        seed=args.seed
    )
    
    # Valida configurazione
    config.validate()
    
    return config


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

def pre_train(
    model,
    optimizer,
    train_dataloader,
    val_dataloader,
    scheduler,
    criterion,  # Loss function
    accelerator,  # Accelerator per distributed training
    config: TrainingConfig,
    checkpoint: ModelCheckpoint,
    early_stopping: EarlyStopping,
    fold: Optional[int] = None
):
    """
    Training loop principale con Early Stopping e Model Checkpoint.
    
    Args:
        model: Modello da trainare
        optimizer: Optimizer
        train_dataloader: DataLoader per training
        val_dataloader: DataLoader per validation
        scheduler: Learning rate scheduler
        criterion: Loss function (CrossEntropyLoss o FocalLoss)
        accelerator: Accelerator per distributed training
        config: TrainingConfig con tutti i parametri
        checkpoint: ModelCheckpoint per salvare best model
        early_stopping: EarlyStopping per fermare training
        fold: Numero fold (opzionale, per K-Fold CV)
    """
    from sklearn.metrics import balanced_accuracy_score
    from src.training.utils import compute_metrics
    
    device = accelerator.device
    
    print(f"\n{'='*60}")
    if fold is not None:
        print(f"  Training Fold {fold + 1}/{config.n_folds}")
    else:
        print(f"  Training Semplice")
    print(f"{'='*60}\n")
    
    # Training loop
    for epoch in range(config.num_epochs):
        # ===== TRAINING PHASE =====
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            output = model(input_ids, attention_mask)
            
            # Usa criterion globale (verrà creato nel main)
            loss = criterion(output, labels)
            accelerator.backward(loss)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Raccogli predizioni per calcolare balanced_accuracy
            preds = torch.argmax(output, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Medie training
        avg_train_loss = train_loss / len(train_dataloader)
        train_metrics = compute_metrics(train_labels, train_preds)
        
        # ===== VALIDATION PHASE =====
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                output = model(input_ids, attention_mask)
                loss = criterion(output, labels)
                
                val_loss += loss.item()
                
                preds = torch.argmax(output, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Medie validation
        avg_val_loss = val_loss / len(val_dataloader)
        val_metrics = compute_metrics(val_labels, val_preds)
        
        # ===== LOGGING =====
        if accelerator.is_main_process:
            fold_info = f"[Fold {fold + 1}/{config.n_folds}] " if fold is not None else ""
            print(f"{fold_info}Epoch {epoch + 1}/{config.num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f} | Train Bal.Acc: {train_metrics['balanced_accuracy']:.4f}")
            print(f"  Val Loss:   {avg_val_loss:.4f} | Val Bal.Acc:   {val_metrics['balanced_accuracy']:.4f}")
            
            # Flush stdout per vedere output in real-time (importante per K-Fold)
            import sys
            sys.stdout.flush()
            
            # ===== MODEL CHECKPOINT =====
            # Usa balanced_accuracy come metrica principale
            metrics_for_checkpoint = {
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'balanced_accuracy': val_metrics['balanced_accuracy'],
                'accuracy': val_metrics['accuracy'],
                'precision': val_metrics['precision'],
                'recall': val_metrics['recall'],
                'f1': val_metrics['f1']
            }
            
            # Unwrap model per salvataggio
            unwrapped_model = accelerator.unwrap_model(model)
            improved = checkpoint.update(
                model=unwrapped_model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics_for_checkpoint
            )
            
            if improved:
                print(f"  ✅ Best model saved! Bal.Acc: {val_metrics['balanced_accuracy']:.4f}")
                import sys
                sys.stdout.flush()
            
            # ===== EARLY STOPPING =====
            # Passa val_loss, train_loss e model per controllo
            early_stopping.update(
                val_loss=avg_val_loss,
                train_loss=avg_train_loss,
                model=unwrapped_model
            )
            
            # Verifica se fermare il training
            if early_stopping.should_stop():
                stop_reason = "Patience esaurita" if early_stopping.wait_count >= early_stopping.patience else "Loss ratio violation"
                print(f"\n⚠️  Early stopping attivato: {stop_reason}")
                print(f"   Epoca trigger: {early_stopping.trigger_epoch}")
                
                # Ripristina weights del best model
                if early_stopping.restore_best_weights:
                    restored_epoch = early_stopping.restore_weights(unwrapped_model)
                    print(f"   ✅ Ripristinati pesi epoca {restored_epoch}")
                
                import sys
                sys.stdout.flush()
                break
            
            # ===== SCHEDULER STEP =====
            scheduler.step(avg_val_loss)
        
        accelerator.wait_for_everyone()
    
    # Fine training
    if accelerator.is_main_process:
        fold_info = f"[Fold {fold + 1}/{config.n_folds}] " if fold is not None else ""
        print(f"\n{'='*60}")
        print(f"  {fold_info}Training completato!")
        best_info = checkpoint.get_best_info()
        print(f"  Best epoch: {best_info['best_epoch']}")
        print(f"  Best Bal.Acc: {best_info['best_value']:.4f}")
        print(f"  Modello salvato: {best_info['best_path']}")
        
        # Salva history delle metriche
        history_path = checkpoint.save_history()
        print(f"  History salvata: {history_path}")
        
        # Salva anche early stopping state se disponibile
        if early_stopping.stopped:
            print(f"\n  📊 Early Stopping Info:")
            print(f"     Trigger epoch: {early_stopping.trigger_epoch}")
            print(f"     Best val loss: {early_stopping.best_val_loss:.4f}")
            print(f"     Wait count: {early_stopping.wait_count}/{early_stopping.patience}")
        
        print(f"{'='*60}\n")
        
        # Flush finale
        import sys
        sys.stdout.flush()
        
        print(f"{'='*60}\n")


if __name__ == '__main__':
    # Parse CLI arguments
    args = parse_args()
    
    # Crea TrainingConfig da CLI args
    config = create_training_config(args)
    
    # Set seed per riproducibilità
    set_seed(config.seed)
    
    print(f"\n{'='*60}")
    print(f"  LEGOLAS - Training su Storie XES Generate")
    print(f"  Formato: {config.story_format}")
    print(f"  Modello: {config.model_name}")
    print(f"  Modalità: {'K-Fold CV' if config.use_kfold else 'Training Semplice'}")
    print(f"  Loss: {'Focal Loss' if config.loss_function == 'focal' else 'Cross Entropy'}")
    print(f"{'='*60}\n")
    
    # Stampa configurazione completa
    config.print_config()
    
    # Assicura che le directory esistano
    ensure_directories()
    
    # Carica file generati dalla pipeline XES
    print(f"\n📖 Caricamento storie formato '{config.story_format}'...")
    
    train_path = get_story_file_path(config.story_format, 'train')
    with open(train_path, 'rb') as f:
        train = pickle.load(f)
    print(f"   ✅ Training stories: {len(train)} campioni")

    test_path = get_story_file_path(config.story_format, 'test')
    with open(test_path, 'rb') as f:
        test = pickle.load(f)
    print(f"   ✅ Test stories: {len(test)} campioni")

    label_train_path = get_story_file_path(config.story_format, 'label_train')
    with open(label_train_path, 'rb') as f:
        label_train = pickle.load(f)

    label_test_path = get_story_file_path(config.story_format, 'label_test')
    with open(label_test_path, 'rb') as f:
        label_test = pickle.load(f)

    # Crea label mapping generico (CLS_0, CLS_1, ...)
    unique_labels = list(np.unique(label_train))
    num_classes = len(unique_labels)
    
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    
    # Crea label mapping generico per export (CLS_0, CLS_1, ...)
    label2id_export = {f"CLS_{i}": i for i in range(num_classes)}
    id2label_export = {str(i): f"CLS_{i}" for i in range(num_classes)}
    
    # Salva label mapping in JSON (per eval_model.py e extract_explainability.py)
    label_mapping_path = config.reports_dir / 'label_mapping.json'
    label_mapping = {
        "label2id": label2id_export,
        "id2label": id2label_export,
        "num_classes": num_classes
    }
    
    with open(label_mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    print(f"\n💾 Label mapping salvato: {label_mapping_path}")
    print(f"   Num classes: {num_classes}")
    for cls_name, cls_id in label2id_export.items():
        print(f"   {cls_name} → {cls_id}")

    label_train_int = []
    for l in label_train:
        label_train_int.append(label2id[l])

    label_test_int = []
    for l in label_test:
        label_test_int.append(label2id[l])


    X_train, X_val, y_train, y_val = train_test_split(
                                                    train, label_train_int,
                                                      test_size=0.2, random_state=config.seed, shuffle=True,
                                                      stratify=label_train_int)

    # Analizza e stampa distribuzione classi
    print(f"\n📊 Distribuzione Classi dopo Split:")
    print(f"\n   Training Set ({len(y_train)} samples):")
    train_dist = analyze_class_distribution(np.array(y_train), print_results=False)
    for class_name, info in train_dist["classes"].items():
        print(f"      {class_name}: {info['count']:4d} samples ({info['percentage']:.1f}%)")
    
    print(f"\n   Validation Set ({len(y_val)} samples):")
    val_dist = analyze_class_distribution(np.array(y_val), print_results=False)
    for class_name, info in val_dist["classes"].items():
        print(f"      {class_name}: {info['count']:4d} samples ({info['percentage']:.1f}%)")
    
    print(f"\n   Test Set ({len(label_test_int)} samples):")
    test_dist = analyze_class_distribution(np.array(label_test_int), print_results=False)
    for class_name, info in test_dist["classes"].items():
        print(f"      {class_name}: {info['count']:4d} samples ({info['percentage']:.1f}%)")
    print()


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
        model_config_data = config_loader.get_model(config.model_name)
        if not model_config_data:
            print(f'❌ Modello non trovato: {config.model_name}')
            print(f'   Modelli disponibili:')
            for model_id in available_models:
                m = config_loader.get_model(model_id)
                print(f'      • {model_id}: {m.description}')
            exit(1)
        
        model_ref = model_config_data.hf_model_id
        recommended_params = config_loader.get_recommended_params(config.model_name)
        
        print(f'\n🤖 Modello selezionato: {config.model_name}')
        print(f'   HuggingFace ID: {model_ref}')
        print(f'   Tipo: {model_config_data.type}')
        print(f'   Descrizione: {model_config_data.description}')
        print(f'   📊 Parametri raccomandati:')
        print(f'      Batch size: {recommended_params["batch_size"]}')
        print(f'      Learning rate: {recommended_params["learning_rate"]}')
        print(f'      Max length: {recommended_params["max_length"]}')
        
    else:
        # Usa legacy model_map
        if config.model_name not in model_map:
            print(f'❌ Modello non trovato: {config.model_name}')
            print(f'   Modelli disponibili: {list(model_map.keys())}')
            exit(1)
        model_ref = model_map[config.model_name]
    
    # Prova a usare cache locale, altrimenti scarica da HuggingFace
    try:
        weights_dir = get_weight_dir(model_ref)
        print(f'✅ Modello trovato in cache locale')
    except (AssertionError, FileNotFoundError):
        print(f'⚠️  Modello non in cache, verrà scaricato da HuggingFace')
        weights_dir = model_ref  # Usa direttamente il model ID
    
    print(f'\n🚀 TRAINING START...')
    print(f'   Modello: {config.model_name} ({model_ref})')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from accelerate import Accelerator

    accelerator = Accelerator(split_batches=True)
    device = accelerator.device

    if config.model_name == 'gpt2':

        model = AutoModel.from_pretrained(weights_dir)
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
        model = SimpleGPT2SequenceClassifier(hidden_size=768, num_classes=config.num_classes, max_seq_len=512, gpt_model=model).to(device)

    else:
        tokenizer = AutoTokenizer.from_pretrained(weights_dir, truncation_side='left')
        model = AutoModel.from_pretrained(weights_dir)
        model = LongFormerMultiClassificationHeads(longformer=model, num_classes=config.num_classes)
        model = model.to(device)


    train_dataset = TextDataset(X_train, y_train, tokenizer, 512)
    val_dataset = TextDataset(X_val, y_val, tokenizer, 512)

    # print(train_dataset.__getitem__(5))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    print('device-->', device)
    print(model)

    # ===== LOSS FUNCTION =====
    print(f"\n📊 Configurazione Loss Function:")
    print(f"   Loss: {config.loss_function.upper()}")
    
    if config.loss_function == 'focal':
        # Focal Loss - usa α/γ da config
        print(f"   Focal α: {config.focal_alpha}")
        print(f"   Focal γ: {config.focal_gamma}")
        criterion = create_loss_from_config(config)
    else:
        # Cross Entropy Loss - calcola class weights automaticamente
        print(f"   Calcolo class weights (method='balanced')...")
        class_weights = compute_class_weights(
            y=np.array(label_train_int),
            method='balanced'
        )
        print(f"   Class weights: {class_weights}")
        
        # Converti in tensor e sposta su device
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    print(f"   ✅ Loss function configurata!\n")

    # ===== CHECK K-FOLD VS SIMPLE TRAINING =====
    if config.use_kfold:
        # ========== K-FOLD CROSS VALIDATION ==========
        print(f"\n{'='*80}")
        print(f"  🔄 MODALITÀ K-FOLD CROSS VALIDATION")
        print(f"{'='*80}\n")
        
        # Combina X_train + X_val e y_train + y_val per K-Fold
        combined_texts = X_train + X_val
        combined_labels = y_train + y_val
        
        # Crea dataset combinato
        combined_dataset = TextDataset(
            combined_texts,
            combined_labels,
            tokenizer,
            max_len=512
        )
        
        # Model factory per creare modelli freschi
        def model_factory():
            """Crea nuovo modello con config.num_classes"""
            if config.model_name == 'gpt2':
                base_model = AutoModel.from_pretrained(weights_dir)
                return SimpleGPT2SequenceClassifier(
                    hidden_size=768,
                    num_classes=config.num_classes,
                    max_seq_len=512,
                    gpt_model=base_model
                )
            else:
                # BERT-based models (BERT, ClinicalBERT, RoBERTa, etc.)
                base_model = AutoModel.from_pretrained(weights_dir)
                return LongFormerMultiClassificationHeads(longformer=base_model, num_classes=config.num_classes)
        
        # Training function per KFoldTrainer
        def train_fold_fn(model, train_dataset, val_dataset, fold, checkpoint, early_stopping, config):
            """Wrapper di pre_train per KFoldTrainer"""
            
            # Crea dataloader
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
            
            # Setup optimizer e scheduler
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            
            # Prepare con accelerator
            model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
                model, optimizer, train_loader, val_loader, scheduler
            )
            
            # Training
            pre_train(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                scheduler=scheduler,
                criterion=criterion,
                accelerator=accelerator,
                config=config,
                checkpoint=checkpoint,
                early_stopping=early_stopping,
                fold=fold
            )
            
            # Ritorna metriche del best model
            best_info = checkpoint.get_best_info()
            return {
                'balanced_accuracy': best_info['best_value'],
                'best_epoch': best_info['best_epoch']
            }
        
        # Crea KFoldTrainer
        kfold_trainer = KFoldTrainer(
            config=config,
            train_func=train_fold_fn,
            model_factory=model_factory,
            dataset=combined_dataset,
            labels=np.array(combined_labels),
            verbose=True
        )
        
        # Esegui K-Fold training
        kfold_results = kfold_trainer.run()
        
        print(f"\n✅ K-Fold Training completato!")
        print(f"   Mean Balanced Accuracy: {kfold_results['mean']['balanced_accuracy']:.4f} ± {kfold_results['std']['balanced_accuracy']:.4f}")
        
    else:
        # ========== SIMPLE TRAINING (NO K-FOLD) ==========
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        startTime = time.time()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)
        
        # ===== CHECKPOINT E EARLY STOPPING =====
        checkpoint = ModelCheckpoint(
            save_dir=config.models_dir,
            fold=None,
            metric='balanced_accuracy',
            mode='max',
            model_name=config.model_name,
            story_format=config.story_format
        )
        
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            use_loss_ratio=config.use_loss_ratio_monitoring,
            loss_ratio_threshold=config.get_loss_ratio_threshold(),
            loss_ratio_patience=config.loss_ratio_patience,
            restore_best_weights=True  # Sempre ripristina best weights
        )
        
        print(f"\n📊 Checkpoint dir: {checkpoint.save_dir}")
        print(f"📊 Early Stopping: patience={config.early_stopping_patience}, ratio_threshold={config.get_loss_ratio_threshold()}")
        
        # ===== TRAINING =====
        pre_train(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            scheduler=scheduler,
            criterion=criterion,
            accelerator=accelerator,
            config=config,
            checkpoint=checkpoint,
            early_stopping=early_stopping,
            fold=None
        )

