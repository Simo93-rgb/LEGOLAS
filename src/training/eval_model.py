from transformers import AutoModel, AutoTokenizer, AutoConfig
from src.data.history_dataset import TextDataset
from src.models.neural_network import LongFormerMultiClassificationHeads, SimpleGPT2SequenceClassifier
from src.utils.model_config_loader import ModelConfigLoader
from src.config.paths import (
    get_story_file_path,
    get_model_path,
    get_prediction_path,
    MODELS_DIR,
    PREDICTION_DIR,
    ensure_directories
)
from torch.utils.data import DataLoader
import torch
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, RocCurveDisplay, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from transformers import (TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
import os
import argparse
from pathlib import Path
from typing import Optional, Union
from transformers import BertModel, BertConfig

# ============================================================
# HELPER FUNCTIONS
# ============================================================


def load_model_for_eval(
    story_format: str,
    model_name: str,
    num_classes: int,
    device: str = 'cuda',
    use_ensemble: bool = False,
    models_dir: Path = None,
    metrics_dir: Path = None
):
    """
    Load model for evaluation: either best fold or ensemble.
    
    Loads models from local checkpoints ONLY (no HuggingFace download).
    Extracts model architecture from checkpoint metadata.
    
    Args:
        story_format: Story format ('narrativo', etc.)
        model_name: Model name ('bert-base-uncased', etc.)
        num_classes: Number of output classes
        device: Device for inference
        use_ensemble: If True, load EnsembleModel; if False, load best fold
        models_dir: Directory where fold models are stored (default: output/models)
        metrics_dir: Directory where metrics are stored (default: output/metrics)
        
    Returns:
        model: Either single model or EnsembleModel instance
        is_ensemble: Boolean flag indicating if returned model is ensemble
        hf_model_id: HuggingFace model ID for tokenizer (extracted from checkpoint)
        
    Example:
        >>> model, is_ensemble, hf_id = load_model_for_eval(
        ...     story_format='narrativo',
        ...     model_name='bert-base-uncased',
        ...     num_classes=2,
        ...     use_ensemble=True
        ... )
    """
    from src.models.ensemble import EnsembleModel
    
    # Prima trova i checkpoint per estrarre il model_name HuggingFace
    model_paths = EnsembleModel.find_kfold_models(
        story_format=story_format,
        model_name=model_name,
        models_dir=models_dir
    )
    
    # Carica primo checkpoint per estrarre metadata
    first_ckpt = torch.load(model_paths[0], map_location='cpu')
    
    # Estrai HuggingFace model ID dai metadati checkpoint
    # (salvato durante training in ModelCheckpoint)
    if 'model_name' in first_ckpt:
        checkpoint_model_name = first_ckpt['model_name']
        print(f"📋 Model architecture from checkpoint: {checkpoint_model_name}")
    else:
        # Fallback: usa model_name passato come argomento
        checkpoint_model_name = model_name
        print(f"⚠️  No model_name in checkpoint, using: {checkpoint_model_name}")
    
    # Determina HuggingFace ID dal model_name
    # Per questo abbiamo bisogno del ModelConfigLoader
    config_loader = ModelConfigLoader()
    model_config = config_loader.get_model(checkpoint_model_name)
    
    if model_config:
        hf_model_id = model_config.hf_model_id
        print(f"✅ HuggingFace ID: {hf_model_id}")
    else:
        # Se non trovato in YAML, usa direttamente il checkpoint_model_name
        hf_model_id = checkpoint_model_name
        print(f"⚠️  Model not in YAML, using: {hf_model_id}")
    
    # Factory per creare modelli
    # NOTA: Crea modello vuoto dalla struttura dedotta dai checkpoint (NO download)
    def create_model():
        # Non scaricare nulla da HuggingFace!
        # La struttura del modello è dedotta dai pesi nel checkpoint
        # Creiamo un modello placeholder che verrà popolato con load_state_dict()
        
        # Carica un checkpoint di esempio per dedurre la struttura
        sample_ckpt = torch.load(model_paths[0], map_location='cpu')
        state_dict = sample_ckpt['model_state_dict'] if 'model_state_dict' in sample_ckpt else sample_ckpt
        
        # Deduce config dai pesi
        # Esempio: longformer.embeddings.word_embeddings.weight ha shape [vocab_size, hidden_size]
        hidden_size = state_dict['longformer.embeddings.word_embeddings.weight'].shape[1]
        vocab_size = state_dict['longformer.embeddings.word_embeddings.weight'].shape[0]
        
        # Conta i layer dell'encoder per dedurre num_hidden_layers
        num_layers = len([k for k in state_dict.keys() if k.startswith('longformer.encoder.layer.')])
        num_hidden_layers = len(set([k.split('.')[3] for k in state_dict.keys() if k.startswith('longformer.encoder.layer.')]))
        
        print(f"   Deduced config: hidden_size={hidden_size}, vocab_size={vocab_size}, layers={num_hidden_layers}")
        
        # Crea config manualmente
        from transformers import BertConfig
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=12,  # Standard per BERT-base
            intermediate_size=hidden_size * 4,  # Standard ratio
            max_position_embeddings=512,
        )
        
        # Crea modello vuoto
        base_model = AutoModel.from_config(config)
        model = LongFormerMultiClassificationHeads(
            longformer=base_model,
            num_classes=num_classes
        )
        return model
    
    if use_ensemble:
        # Load ensemble di tutti i fold
        print(f"\n🔄 Loading K-Fold Ensemble...")
        ensemble = EnsembleModel(
            story_format=story_format,
            model_name=model_name,
            model_factory=create_model,
            models_dir=models_dir,
            metrics_dir=metrics_dir,
            device=device
        )
        
        # Info ensemble
        best_fold = EnsembleModel.get_best_fold(ensemble.fold_metrics)
        print(f"   Ensemble: {len(ensemble.models)} folds")
        print(f"   Best fold: {best_fold} (accuracy: {ensemble.fold_metrics[best_fold]['best_value']:.4f})")
        
        return ensemble, True, hf_model_id
        
    else:
        # Load solo best fold
        print(f"\n📥 Loading best fold model...")
        
        # Load metrics e trova best
        fold_metrics = EnsembleModel.load_fold_metrics(
            story_format=story_format,
            model_name=model_name,
            n_folds=len(model_paths),
            metrics_dir=metrics_dir
        )
        
        best_fold = EnsembleModel.get_best_fold(fold_metrics)
        best_model_path = model_paths[best_fold]
        
        print(f"   Best fold: {best_fold}")
        print(f"   Accuracy: {fold_metrics[best_fold]['best_value']:.4f}")
        print(f"   Model: {best_model_path.name}")
        
        # Load model
        model = create_model()
        checkpoint = torch.load(best_model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        print(f"   ✅ Model loaded successfully\n")
        
        return model, False, hf_model_id


# ============================================================
# CONFIGURAZIONE EVAL
# ============================================================

# Parse CLI arguments
parser = argparse.ArgumentParser(description='LEGOLAS - Model Evaluation')
parser.add_argument('--story_format', type=str, default='narrativo',
                    help='Story format (default: narrativo)')
parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                    help='Model name (default: bert-base-uncased)')
parser.add_argument('--use_ensemble', action='store_true',
                    help='Use K-Fold ensemble instead of best fold only')
args = parser.parse_args()

STORY_FORMAT = args.story_format
model_name = args.model_name
USE_ENSEMBLE = args.use_ensemble

print(f"\n{'='*60}")
print(f"  LEGOLAS - Evaluation Modello XES")
print(f"  Formato: {STORY_FORMAT}")
print(f"  Modello: {model_name}")
print(f"  Mode: {'K-Fold Ensemble' if USE_ENSEMBLE else 'Best Fold Only'}")
print(f"{'='*60}\n")

# Carica configurazioni modelli da YAML
print(f'📋 Caricamento configurazioni modelli da YAML...')
try:
    config_loader = ModelConfigLoader()
    available_models = config_loader.list_model_ids()
    print(f'✅ {len(available_models)} modelli caricati')
    
except FileNotFoundError as e:
    print(f'⚠️  Config YAML non trovato: {e}')
    print(f'   Evaluation richiede config YAML per identificare modelli')
    exit(1)

# Verifica che il modello richiesto esista
model_config = config_loader.get_model(model_name)
if not model_config:
    print(f'❌ Modello non trovato: {model_name}')
    print(f'   Modelli disponibili:')
    for model_id in available_models:
        m = config_loader.get_model(model_id)
        print(f'      • {model_id}: {m.description}')
    exit(1)

print(f'\n🤖 Modello selezionato: {model_name}')
print(f'   HuggingFace ID: {model_config.hf_model_id}')
print(f'   Tipo: {model_config.type}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assicura che le directory esistano
ensure_directories()

# Carica storie dal formato specificato
print(f"\n📖 Caricamento storie formato '{STORY_FORMAT}'...")

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

label_test_int = []
for l in label_test:
    label_test_int.append(label2id[l])

# Determine num_classes
num_classes = len(label2id)

# Load model using helper function (estrae HF model ID dai checkpoint)
test_model, is_ensemble, hf_model_id = load_model_for_eval(
    story_format=STORY_FORMAT,
    model_name=model_name,
    num_classes=num_classes,
    device=device,
    use_ensemble=USE_ENSEMBLE
)

# Load tokenizer usando HF model ID estratto dai checkpoint
print(f"\n📝 Loading tokenizer: {hf_model_id}")
tokenizer = AutoTokenizer.from_pretrained(hf_model_id, truncation_side='left')


test_dataset = TextDataset(test, label_test_int, tokenizer, 512)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Set model to eval mode (no-op for ensemble, but consistent API)
if not is_ensemble:
    test_model.eval()

all_targets = []
all_predictions = []
pred_prob = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Handle ensemble vs regular model
        if is_ensemble:
            # EnsembleModel uses .predict() method
            output = test_model.predict(input_ids, attention_mask)
        else:
            # Regular model uses direct call
            output = test_model(input_ids, attention_mask)
        
        pred_prob.append(output.cpu())
        predicted = output.argmax(dim=1)
        all_targets.extend(batch['labels'].to('cpu').numpy())
        all_predictions.extend(predicted.to('cpu').numpy())

all_targets = [int(x) for x in all_targets]
all_predictions = [int(x) for x in all_predictions]

# Assicura che prediction directory esista
ensure_directories()

# Prefix per nomi file risultati (include ensemble info)
ensemble_suffix = '_ensemble' if USE_ENSEMBLE else ''
result_prefix = f'{STORY_FORMAT}_{model_name}{ensemble_suffix}'

print(f"\n💾 Salvataggio risultati...")

# Costruisci path manualmente per supportare suffisso ensemble
if USE_ENSEMBLE:
    prob_path = PREDICTION_DIR / f'{result_prefix}_prob.pkl'
    target_path = PREDICTION_DIR / f'{result_prefix}_all_target.pkl'
    prediction_path = PREDICTION_DIR / f'{result_prefix}_all_prediction.pkl'
    report_path = PREDICTION_DIR / f'{result_prefix}_report.txt'
else:
    # Usa helper per consistency
    prob_path = get_prediction_path(STORY_FORMAT, model_name, 'prob')
    target_path = get_prediction_path(STORY_FORMAT, model_name, 'all_target')
    prediction_path = get_prediction_path(STORY_FORMAT, model_name, 'all_prediction')
    report_path = get_prediction_path(STORY_FORMAT, model_name, 'report')

# Salva predictions
with open(prob_path, 'wb') as file:
    pickle.dump(pred_prob, file)

with open(target_path, 'wb') as file:
    pickle.dump(all_targets, file)

with open(prediction_path, 'wb') as file:
    pickle.dump(all_predictions, file)

print(f"\n{'='*60}")
print("  CLASSIFICATION REPORT")
print(f"{'='*60}\n")

report = classification_report(all_targets, all_predictions, output_dict=False, digits=4)
print(report)

# Salva report
result = open(report_path, 'w')
result.write(f"Formato Storie: {STORY_FORMAT}\n")
result.write(f"Modello: {model_name}\n")
if USE_ENSEMBLE:
    result.write(f"Mode: K-Fold Ensemble\n")
result.write(f"{'='*60}\n\n")
result.write(report)
result.write('\n\n')

print(f"\n{'='*60}")
print("  CONFUSION MATRIX")
print(f"{'='*60}\n")

conf_m = confusion_matrix(all_targets, all_predictions)
print(conf_m)
result.write(str(conf_m))
result.close()

print(f"\n✅ Risultati salvati in: {PREDICTION_DIR / result_prefix}_*")
print(f"\n{'='*60}")