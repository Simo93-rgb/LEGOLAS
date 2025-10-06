# 🚀 Guida Rapida - Model Configuration System

## Overview

LEGOLAS ora supporta 15+ modelli biomedici configurabili tramite YAML invece dei 4 modelli hardcoded precedenti.

## Modelli Disponibili

### 🏥 Modelli Biomedici Specializzati (8)
- **clinical-bert** - ClinicalBERT specializzato in note cliniche (default consigliato)
- **pubmedbert-base** - Addestrato su letteratura PubMed
- **biobert-base** - Pre-addestrato su testi biomedici
- **bluebert-base** - Specializzato in dominio clinico
- **clinicalbert-base** - Ottimizzato per record medici
- **clinical-modernbert** - Architettura moderna per dati clinici
- **bioclinical-modernbert** - Variante avanzata
- **scibert-base** - Specializzato in letteratura scientifica

### 📚 Modelli Longformer per Sequenze Lunghe (2)
- **clinical-longformer** - Clinical Longformer (fino a 4096 token)
- **longformer-base** - Longformer base (sequenze lunghe)

### 🌍 Modelli Generici (3)
- **bert-base-uncased** - BERT base standard
- **bert-large-uncased** - BERT large (maggiore capacità)
- **bert-italian** - BERT italiano (per testi non tradotti)

## Uso Rapido

### Opzione 1: Menu Interattivo
```bash
cd /home/simon/GitHub/LEGOLAS
./scripts/launch_run_train_eval
```

Il menu ora mostra automaticamente tutti i 13+ modelli da `config/model_configs.yaml` con:
- ID modello
- Descrizione
- Tag [BIOMEDICAL] per modelli specializzati
- Parametri raccomandati (batch size, learning rate)

### Opzione 2: Modifica Diretta train_llm.py
```python
# In src/training/train_llm.py, cambia la variabile:
model_name = 'clinical-bert'  # O qualsiasi altro ID da model_configs.yaml
```

### Opzione 3: Script Python
```python
from src.utils.model_config_loader import ModelConfigLoader

# Carica tutte le configurazioni
loader = ModelConfigLoader()

# Lista modelli disponibili
loader.print_available_models()

# Ottieni configurazione specifica
model = loader.get_model('clinical-bert')
print(f"HF ID: {model.hf_model_id}")
print(f"Batch consigliato: {model.recommended_batch_size}")

# Solo modelli biomedici
bio_models = loader.get_biomedical_models()
```

## Struttura Config File

`config/model_configs.yaml` ha questa struttura:

```yaml
models:
  clinical-bert:
    name: emilyalsentzer/Bio_ClinicalBERT
    type: bert
    max_length: 512
    num_labels: 8
    description: "ClinicalBERT, specializzato in note cliniche"
    recommended_batch_size: 12
    recommended_lr: 2.0e-5
```

## Aggiungere Nuovi Modelli

1. Apri `config/model_configs.yaml`
2. Aggiungi entry sotto `models:`:
   ```yaml
   mio-nuovo-modello:
     name: organization/model-id-on-huggingface
     type: bert  # o longformer, gpt2, ecc.
     max_length: 512
     num_labels: 8
     description: "Descrizione del modello"
     recommended_batch_size: 12
     recommended_lr: 2.0e-5
   ```
3. Il modello sarà immediatamente disponibile nel menu!

## Parametri Raccomandati per Tipo

### BERT Standard (512 token)
- Batch size: 12-16
- Learning rate: 2e-5

### BERT Large (più parametri)
- Batch size: 8
- Learning rate: 1e-5

### Longformer (4096 token)
- Batch size: 4-6
- Learning rate: 1e-5

### Note GPU
Con 24GB VRAM:
- BERT base: batch 12-16 ✅
- BERT large: batch 8 ✅
- Longformer: batch 4-6 ✅

Con meno memoria, riduci batch size di conseguenza.

## Esempio Workflow Completo

```bash
# 1. Genera storie da XES (se non fatto)
./scripts/run_xes_pipeline.sh

# 2. Training interattivo con nuovo modello
./scripts/launch_run_train_eval

# Menu:
# - Scegli formato: narrativo
# - Scegli modello: pubmedbert-base (opzione 8)
# - Azione: Training + Evaluation

# 3. Risultati salvati in:
#    output/models/xes_narrativo_pubmedbert-base.pth
#    prediction/xes_narrativo_pubmedbert-base_*.pkl
```

## Script Helper

### list_models.py
Utility per uso da bash:

```bash
# Lista menu
./scripts/list_models.py menu

# Conta modelli
./scripts/list_models.py count

# Info modello specifico
./scripts/list_models.py info clinical-bert
```

Output formato KEY=VALUE per parsing in bash.

## Fallback Legacy

Se `config/model_configs.yaml` non viene trovato, il sistema usa automaticamente i 4 modelli legacy:
- bertm (BERT Medium)
- roberta (RoBERTa Base)
- cbert (Clinical BERT)
- gpt2 (GPT-2)

## Migrazione da Legacy

Vecchio codice:
```python
model_name = 'cbert'
```

Nuovo codice (stesso ID):
```python
model_name = 'clinical-bert'  # Nuovo ID da YAML
```

Mappatura legacy → nuovo:
- `bertm` → NON PIÙ SUPPORTATO (usa `bert-base-uncased`)
- `roberta` → NON PIÙ SUPPORTATO (aggiungi a YAML se necessario)
- `cbert` → `clinical-bert` (stesso modello)
- `gpt2` → NON PIÙ SUPPORTATO (aggiungi a YAML se necessario)

## Testing

```bash
# Test loader
uv run python src/utils/model_config_loader.py

# Test in train pipeline
uv run python -c "from src.utils.model_config_loader import ModelConfigLoader; ModelConfigLoader().print_available_models()"

# Test menu bash
./scripts/list_models.py menu
```

## Troubleshooting

### "Config file non trovato"
- Assicurati che esista `config/model_configs.yaml`
- Path cercati: `config/`, `../config/`, `../../config/`

### "Modello non trovato"
- Verifica ID in `config/model_configs.yaml`
- Usa `list_models.py menu` per lista completa

### "Out of memory"
- Riduci batch size nel config YAML
- Usa modelli più piccoli (bert-base vs bert-large)
- Considera Longformer con gradient checkpointing

## Prossimi Sviluppi

- [ ] Supporto modelli GPT-based
- [ ] Auto-tuning batch size basato su GPU
- [ ] Profili configurazione per dataset diversi
- [ ] Model ensemble configuration
- [ ] Integration con Ray Tune per hyperparameter search
