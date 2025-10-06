from transformers import AutoModel, AutoTokenizer
from src.data.history_dataset import TextDataset
from src.models.neural_network import LongFormerMultiClassificationHeads, SimpleGPT2SequenceClassifier
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
from pathlib import Path
from typing import Optional
from transformers import BertModel, BertConfig

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

# ============================================================
# CONFIGURAZIONE EVAL
# ============================================================
# Deve corrispondere al formato usato in train_llm.py
STORY_FORMAT = 'narrativo'
model_name = 'cbert'

print(f"\n{'='*60}")
print(f"  LEGOLAS - Evaluation Modello XES")
print(f"  Formato: {STORY_FORMAT}")
print(f"  Modello: {model_name}")
print(f"{'='*60}\n")

# Mappa nome modello a HuggingFace model ID
model_map = {
    'bertm': 'prajjwal1/bert-medium',
    'roberta': 'FacebookAI/roberta-base',
    'gpt2': 'openai-community/gpt2',
    'cbert': 'emilyalsentzer/Bio_ClinicalBERT'
}

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

print(f'   Modello: {model_name} ({model_ref})')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica storie dal formato specificato
print(f"📖 Caricamento storie formato '{STORY_FORMAT}'...")

with open(f'output/stories/{STORY_FORMAT}_test.pkl', 'rb') as f:
    test = pickle.load(f)
print(f"   ✅ Test stories: {len(test)} campioni")

with open(f'output/stories/{STORY_FORMAT}_label_train.pkl', 'rb') as f:
    label_train = pickle.load(f)

with open(f'output/stories/{STORY_FORMAT}_label_test.pkl', 'rb') as f:
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
    test_model = SimpleGPT2SequenceClassifier(hidden_size=768, num_classes=8, max_seq_len=512, gpt_model=model).to(device)

else:
    tokenizer = AutoTokenizer.from_pretrained(weights_dir, truncation_side='left')
    test_model = AutoModel.from_pretrained(weights_dir)
    test_model = LongFormerMultiClassificationHeads(test_model)
    
    # Carica modello addestrato - cerca ultimo modello salvato
    base_model_name = f'output/models/xes_{STORY_FORMAT}_{model_name}'
    print(f"\n📥 Caricamento modello: {base_model_name}")
    
    # Cerca modelli con numero epoca (es: xes_narrativo_bertm1.pth, xes_narrativo_bertm2.pth, ...)
    import glob
    model_files = glob.glob(f'{base_model_name}*.pth')
    
    if model_files:
        # Ordina per numero epoca (l'ultimo è il migliore di solito)
        model_files.sort()
        model_file = model_files[-1]  # Prendi l'ultimo
        print(f"   ✅ Trovati {len(model_files)} modelli, uso: {model_file}")
    else:
        # Fallback: cerca senza numero epoca
        model_file = f'{base_model_name}.pth'
        if not os.path.exists(model_file):
            print(f"❌ ERRORE: Nessun modello trovato!")
            print(f"   Cercato: {base_model_name}*.pth")
            print(f"   Assicurati di aver eseguito train_llm.py prima!")
            exit(1)
        print(f"   ✅ Uso modello: {model_file}")
    
    test_model.load_state_dict(torch.load(model_file))
    test_model = test_model.to(device)
    print(f"   ✅ Modello caricato con successo\n")


test_dataset = TextDataset(test, label_test_int, tokenizer, 512)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)


test_model.eval()

all_targets = []
all_predictions = []
pred_prob = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = test_model(input_ids, attention_mask)
        pred_prob.append(output.cpu())
        predicted = output.argmax(dim=1)
        all_targets.extend(batch['labels'].to('cpu').numpy())
        all_predictions.extend(predicted.to('cpu').numpy())

all_targets = [int(x) for x in all_targets]
all_predictions = [int(x) for x in all_predictions]

# Crea directory prediction se non esiste
os.makedirs('prediction', exist_ok=True)

# Salva risultati con nome che include formato storie
result_prefix = f'xes_{STORY_FORMAT}_{model_name}'

print(f"\n💾 Salvataggio risultati...")

with open(f'prediction/{result_prefix}_prob.pkl', 'wb') as file:
    pickle.dump(pred_prob, file)

with open(f'prediction/{result_prefix}_all_target.pkl', 'wb') as file:
    pickle.dump(all_targets, file)

with open(f'prediction/{result_prefix}_all_prediction.pkl', 'wb') as file:
    pickle.dump(all_predictions, file)

print(f"\n{'='*60}")
print("  CLASSIFICATION REPORT")
print(f"{'='*60}\n")

report = classification_report(all_targets, all_predictions, output_dict=False, digits=4)
print(report)

result = open(f'prediction/{result_prefix}_report.txt', 'w')
result.write(f"Formato Storie: {STORY_FORMAT}\n")
result.write(f"Modello: {model_name}\n")
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

print(f"\n✅ Risultati salvati in: prediction/{result_prefix}_*")
print(f"\n{'='*60}")