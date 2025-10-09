# Refactoring num_classes - 9 Ottobre 2025

## 📋 Obiettivo
Rendere il progetto LEGOLAS generico per N classi, non più limitato a 8 classi hardcoded.

## ✅ Modifiche Implementate

### 1. **Architettura Modelli** (`src/models/neural_network.py`)

#### `LongFormerMultiClassificationHeads`
```python
# PRIMA (8 classi hardcoded)
class LongFormerMultiClassificationHeads(nn.Module):
    def __init__(self, longformer):
        self.output_layer = nn.Linear(longformer.config.hidden_size, 8)

# DOPO (num_classes configurabile)
class LongFormerMultiClassificationHeads(nn.Module):
    def __init__(self, longformer, num_classes: int = 8):
        self.num_classes = num_classes
        self.output_layer = nn.Linear(longformer.config.hidden_size, num_classes)
```

**Backward compatibility**: Default `num_classes=8` per codice legacy.

### 2. **Training Script** (`src/training/train_llm.py`)

#### Training Semplice (linee 543-556)
```python
# PRIMA
model = SimpleGPT2SequenceClassifier(..., num_classes=8, ...)
model = LongFormerMultiClassificationHeads(model)

# DOPO
model = SimpleGPT2SequenceClassifier(..., num_classes=config.num_classes, ...)
model = LongFormerMultiClassificationHeads(model, num_classes=config.num_classes)
```

#### K-Fold model_factory (linee 620-633)
```python
# PRIMA
return SimpleGPT2SequenceClassifier(..., num_classes=8, ...)
return LongFormerMultiClassificationHeads(longformer=base_model)

# DOPO
return SimpleGPT2SequenceClassifier(..., num_classes=config.num_classes, ...)
return LongFormerMultiClassificationHeads(longformer=base_model, num_classes=config.num_classes)
```

#### Cleanup
- ❌ Rimosso `gpt2_config` inutilizzato (linee 546, 624)
- ✅ Corretto typo: `resize_token_embedings` → `resize_token_embeddings`

### 3. **Logging K-Fold** (`src/training/train_llm.py`)

#### Problemi Risolti
- Output non visibile durante K-Fold training
- Mancanza di context (quale fold sta trainando)

#### Soluzioni
1. **Fold info in ogni stampa**:
```python
# PRIMA
print(f"Epoch {epoch + 1}/{config.num_epochs}")

# DOPO
fold_info = f"[Fold {fold + 1}/{config.n_folds}] " if fold is not None else ""
print(f"{fold_info}Epoch {epoch + 1}/{config.num_epochs}")
```

2. **Flush stdout dopo stampe critiche**:
```python
import sys
sys.stdout.flush()  # dopo ogni print importante
```

Posizioni flush:
- Dopo ogni epoca (riga ~296)
- Dopo best model saved (riga ~313)
- Dopo early stopping (riga ~334)
- Dopo training completato (riga ~377)

### 4. **Test Suite** (`tests/test_num_classes_refactor.py`)

#### Test Creati
1. **TestLongFormerNumClasses**: 
   - `test_output_layer_size`: Verifica dimensione layer con 2, 3, 8, 10 classi
   - `test_forward_pass_shape`: Verifica shape output
   - `test_default_num_classes`: Backward compatibility (default=8)

2. **TestGPT2NumClasses**:
   - `test_fc1_layer_size`: Verifica dimensione FC1
   - `test_forward_pass_shape`: Verifica shape output

3. **TestTrainingConfigIntegration**:
   - `test_config_num_classes_binary`: Config per binario (2 classi)
   - `test_config_num_classes_multiclass`: Config per multi-classe (5 classi)
   - `test_focal_alpha_validation`: Verifica mismatch focal_alpha/num_classes
   - `test_focal_alpha_correct`: Verifica focal_alpha corretto

#### Esecuzione
```bash
pytest tests/test_num_classes_refactor.py -v
```

## 📊 Impatto

### Prima del Refactoring
- ❌ Solo 8 classi possibili
- ❌ `num_classes` in `TrainingConfig` ignorato
- ❌ Impossibile training binario o multi-classe arbitrario
- ❌ Output K-Fold non visibile in real-time

### Dopo il Refactoring
- ✅ N classi configurabili (2, 3, 8, 10, ...)
- ✅ `config.num_classes` usato ovunque
- ✅ Training binario, multi-classe, qualsiasi N
- ✅ Backward compatible (default 8 classi)
- ✅ Output K-Fold visibile e contestualizzato

## 🧪 Verifica

### Test Automatici
```bash
# Test refactoring num_classes
pytest tests/test_num_classes_refactor.py -v

# Tutti i test FASE 1-4
pytest tests/ -v
```

### Test Manuale
```bash
# Training binario (2 classi)
uv run python src/training/train_llm.py \
  --story_format narrativo \
  --model bert-base-uncased \
  --epochs 5 \
  --use_kfold \
  --n_folds 3

# Dovrebbe mostrare:
# [Fold 1/3] Epoch 1/5
# [Fold 1/3] Epoch 2/5
# ...
# [Fold 2/3] Epoch 1/5
# ...
```

## 📝 Documentazione Aggiornata
- ✅ `docs/PIANO_LAVORI_FUNZIONALITÀ_TRAINING.md` - Sezione refactoring marcata COMPLETATO
- ✅ `tests/test_num_classes_refactor.py` - Test suite completa
- ✅ Questo file (`REFACTORING_NUM_CLASSES.md`)

## 🔄 Prossimi Passi
1. Eseguire test suite completa
2. Testare training K-Fold con output visibile
3. Verificare che `config.num_classes=2` funzioni end-to-end
4. Commit: "refactor: support configurable num_classes, improve K-Fold logging"
5. Procedere con FASE 5 (Ensemble Prediction)

---

**Data**: 9 Ottobre 2025  
**Branch**: `advanced-training`  
**Stato**: ✅ COMPLETATO - Pronto per testing
