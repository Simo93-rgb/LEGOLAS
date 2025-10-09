# 🚀 Training Avanzato - Guida Completa

## 📋 Panoramica

Questa guida documenta l'integrazione completa del sistema di training avanzato in `train_llm.py`, che include:

- ✅ **K-Fold Cross Validation** con stratificazione classi
- ✅ **Focal Loss** per gestione classi sbilanciate
- ✅ **Early Stopping** avanzato con loss ratio monitoring
- ✅ **Best Model Checkpoint** basato su balanced_accuracy
- ✅ **Label Mapping** generico per evaluation/explainability
- ✅ **CLI completa** per configurazione flessibile

---

## 🎯 Funzionalità Implementate

### 1. K-Fold Cross Validation (FASE 4.3.5)

Training con K-Fold stratificato per validazione robusta:
- Split stratificato per mantenere distribuzione classi
- Modello fresco per ogni fold (evita contaminazione pesi)
- Aggregazione risultati: mean ± std ± min ± max
- Salvataggio modello per fold: `best_model_{format}_{model}_fold{k}.pth`
- Report JSON: `kfold_aggregated_results.json`

### 2. Focal Loss (FASE 4.3.3)

Loss function specializzata per classi sbilanciate:
- Formula: `FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)`
- Parametri α (class weights) e γ (focusing) configurabili da CLI
- Riduce peso esempi facili, enfatizza esempi difficili
- Alternative: Cross Entropy con class weights automatici

### 3. Early Stopping Avanzato (FASE 4.3.2)

Monitoraggio intelligente per stop training:
- **Delta monitoring**: Stop se val_loss non migliora per N epoche
- **Ratio monitoring**: Stop se train_loss/val_loss < threshold (overfitting)
- **Restore weights**: Ripristina pesi epoca `trigger - patience`
- Configurable patience, min_delta, ratio_threshold

### 4. Best Model Checkpoint (FASE 4.3.4)

Salvataggio solo miglior modello:
- Metrica: `balanced_accuracy` (gestisce classi sbilanciate)
- Rimuove vecchio best quando migliora
- History salvata: `checkpoint_history.json` (o `_fold{k}.json`)
- Include: model_state, optimizer_state, metrics, timestamp

### 5. Label Mapping Generico (FASE 4.3.6)

Export label per eval/explainability:
- Formato generico: `CLS_0`, `CLS_1` (riusabile)
- File JSON: `output/reports/label_mapping.json`
- Usato da `eval_model.py` e `extract_explainability.py`
- Logging distribuzione classi train/val/test

---

## 🛠️ Utilizzo

### Metodo 1: Script Bash Helper (RACCOMANDATO)

```bash
# Training semplice (no K-Fold)
./scripts/launch_run_train_eval
```

Il menu interattivo guida nella scelta di:
1. Formato storie (narrativo/bullet/clinical)
2. Modello (bertm/roberta/cbert/gpt2)
3. Modalità training (semplice/K-Fold)
4. Loss function (focal/cross-entropy)

### Metodo 2: CLI Diretta

#### Training Semplice (no K-Fold)
```bash
uv run python src/training/train_llm.py \
    --story_format narrativo \
    --model bertm \
    --epochs 10 \
    --patience 5
```

#### Training con K-Fold (10 folds)
```bash
uv run python src/training/train_llm.py \
    --story_format narrativo \
    --model bertm \
    --use_kfold \
    --n_folds 10 \
    --epochs 10 \
    --patience 5
```

#### Training con Focal Loss
```bash
uv run python src/training/train_llm.py \
    --story_format narrativo \
    --model bertm \
    --use_focal_loss \
    --focal_alpha 0.25 0.75 \
    --focal_gamma 2.0 \
    --epochs 10
```

#### Training K-Fold + Focal Loss (Full Power!)
```bash
uv run python src/training/train_llm.py \
    --story_format narrativo \
    --model bertm \
    --use_kfold \
    --n_folds 5 \
    --use_focal_loss \
    --focal_alpha 0.25 0.75 \
    --focal_gamma 2.0 \
    --epochs 15 \
    --patience 7
```

---

## 📝 Parametri CLI

### Parametri Obbligatori
| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `--story_format` | `narrativo` | Formato storie: `narrativo`, `bullet`, `clinical` |
| `--model` | `bertm` | Modello: `bertm`, `roberta`, `cbert`, `gpt2` |

### Parametri K-Fold
| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `--use_kfold` | `False` | Attiva K-Fold Cross Validation |
| `--n_folds` | `10` | Numero fold (solo se `--use_kfold`) |

### Parametri Focal Loss
| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `--use_focal_loss` | `False` | Usa Focal Loss invece di Cross Entropy |
| `--focal_alpha` | `0.25 0.75` | Pesi classi (2 valori per binario) |
| `--focal_gamma` | `2.0` | Focusing parameter (γ) |

### Parametri Training
| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `--epochs` | `10` | Numero epoche massime |
| `--patience` | `5` | Patience early stopping |
| `--batch_size` | `256` | Batch size training |
| `--learning_rate` | `1e-5` | Learning rate optimizer |

---

## 📊 Output e File Generati

### Training Semplice (no K-Fold)

```
output/
├── models/
│   └── best_model_narrativo_bertm.pth         # Best model
├── reports/
│   ├── checkpoint_history.json                # History metriche
│   └── label_mapping.json                     # Label mapping CLS_0, CLS_1
└── stories/
    ├── narrativo_train.pkl                    # Dati training
    └── narrativo_test.pkl                     # Dati test
```

### Training K-Fold (5 folds)

```
output/
├── models/
│   ├── best_model_narrativo_bertm_fold0.pth   # Modello fold 0
│   ├── best_model_narrativo_bertm_fold1.pth   # Modello fold 1
│   ├── best_model_narrativo_bertm_fold2.pth   # Modello fold 2
│   ├── best_model_narrativo_bertm_fold3.pth   # Modello fold 3
│   └── best_model_narrativo_bertm_fold4.pth   # Modello fold 4
├── reports/
│   ├── checkpoint_history_fold0.json          # History fold 0
│   ├── checkpoint_history_fold1.json          # History fold 1
│   ├── checkpoint_history_fold2.json          # History fold 2
│   ├── checkpoint_history_fold3.json          # History fold 3
│   ├── checkpoint_history_fold4.json          # History fold 4
│   ├── kfold_aggregated_results.json          # Risultati aggregati
│   ├── fold_0_metrics.json                    # Metriche fold 0
│   ├── fold_1_metrics.json                    # Metriche fold 1
│   ├── fold_2_metrics.json                    # Metriche fold 2
│   ├── fold_3_metrics.json                    # Metriche fold 3
│   ├── fold_4_metrics.json                    # Metriche fold 4
│   └── label_mapping.json                     # Label mapping CLS_0, CLS_1
└── stories/
    └── ...
```

### Formato `label_mapping.json`

```json
{
  "label2id": {
    "CLS_0": 0,
    "CLS_1": 1
  },
  "id2label": {
    "0": "CLS_0",
    "1": "CLS_1"
  },
  "num_classes": 2
}
```

### Formato `kfold_aggregated_results.json`

```json
{
  "n_folds": 5,
  "fold_results": [
    {
      "fold": 0,
      "train_size": 800,
      "val_size": 200,
      "balanced_accuracy": 0.8234,
      "best_epoch": 7
    },
    ...
  ],
  "fold_models": [
    "output/models/best_model_narrativo_bertm_fold0.pth",
    ...
  ],
  "mean": {
    "balanced_accuracy": 0.8123,
    "best_epoch": 6.4
  },
  "std": {
    "balanced_accuracy": 0.0234,
    "best_epoch": 1.2
  },
  "min": {
    "balanced_accuracy": 0.7789,
    "best_epoch": 5
  },
  "max": {
    "balanced_accuracy": 0.8456,
    "best_epoch": 8
  }
}
```

---

## 🔄 Workflow Completo

### 1. Preparazione Dati

```bash
# Genera storie da file XES
./scripts/run_xes_pipeline.sh
```

Output: `output/stories/{formato}_train.pkl` e `{formato}_test.pkl`

### 2. Training

#### Opzione A: Training Semplice (sviluppo rapido)
```bash
uv run python src/training/train_llm.py \
    --story_format narrativo \
    --model bertm \
    --epochs 10
```

**Quando usare:**
- Test rapidi durante sviluppo
- Debugging modello
- Risorse limitate

#### Opzione B: Training K-Fold (validazione robusta)
```bash
uv run python src/training/train_llm.py \
    --story_format narrativo \
    --model bertm \
    --use_kfold \
    --n_folds 10 \
    --epochs 15
```

**Quando usare:**
- Validazione finale modello
- Confronto modelli diversi
- Paper/pubblicazioni (metriche robuste)

### 3. Evaluation

```bash
# Valuta modello su test set
uv run python src/training/eval_model.py \
    --story_format narrativo \
    --model bertm
```

Output: `prediction/xes_{formato}_{model}_*.pkl`

### 4. Explainability

```bash
# Genera SHAP explanations
uv run python extract_explainability.py \
    --story_format narrativo \
    --model bertm
```

Output: `output/explainability/{formato}_{model}_*.pkl`

---

## 🎓 Esempi Pratici

### Esempio 1: Quick Test con BERT

```bash
# Training veloce per test (no K-Fold)
uv run python src/training/train_llm.py \
    --story_format narrativo \
    --model bertm \
    --epochs 5 \
    --patience 3
```

**Tempo stimato:** ~10-15 minuti  
**Output:** 1 modello, checkpoint history, label mapping

### Esempio 2: Validazione Robusta con K-Fold

```bash
# Training con 5 fold per validazione
uv run python src/training/train_llm.py \
    --story_format narrativo \
    --model bertm \
    --use_kfold \
    --n_folds 5 \
    --epochs 10 \
    --patience 5
```

**Tempo stimato:** ~50-75 minuti (5x il tempo simple)  
**Output:** 5 modelli, aggregated results, fold metrics

### Esempio 3: Focal Loss per Classi Sbilanciate

```bash
# Training con Focal Loss (es. 70% CLS_0, 30% CLS_1)
uv run python src/training/train_llm.py \
    --story_format narrativo \
    --model bertm \
    --use_focal_loss \
    --focal_alpha 0.3 0.7 \
    --focal_gamma 2.0 \
    --epochs 10
```

**Quando usare:**
- Dataset molto sbilanciato (es. 80%-20%)
- Minoranza class più importante (es. eventi rari)

### Esempio 4: Full Power (K-Fold + Focal)

```bash
# Best practice per produzione
uv run python src/training/train_llm.py \
    --story_format narrativo \
    --model bertm \
    --use_kfold \
    --n_folds 10 \
    --use_focal_loss \
    --focal_alpha 0.25 0.75 \
    --focal_gamma 2.0 \
    --epochs 15 \
    --patience 7
```

**Tempo stimato:** ~2-3 ore  
**Output:** 10 modelli, metriche aggregate robuste

---

## 🔍 Interpretazione Output

### Output Console - Training Semplice

```
============================================================
  LEGOLAS - Training su Storie XES Generate
  Formato: narrativo
  Modello: bertm
  Modalità: Training Semplice
  Loss: Cross Entropy
============================================================

📊 Configurazione Loss Function:
   Loss: CROSS ENTROPY
   Calcolo class weights (method='balanced')...
   Class weights: [0.8, 1.2]
   ✅ Loss function configurata!

💾 Label mapping salvato: output/reports/label_mapping.json
   Num classes: 2
   CLS_0 → 0
   CLS_1 → 1

📊 Distribuzione Classi dopo Split:

   Training Set (800 samples):
      Class 0 (CLS_0):  520 samples (65.0%)
      Class 1 (CLS_1):  280 samples (35.0%)

   Validation Set (200 samples):
      Class 0 (CLS_0):  130 samples (65.0%)
      Class 1 (CLS_1):   70 samples (35.0%)

============================================================
  🏋️  TRAINING LOOP
============================================================

Epoch 1/10:
  Train Loss: 0.5234 | Val Loss: 0.4567
  Balanced Accuracy: 0.7234
  ✅ New best model! (0.7234 > 0.0000)

Epoch 2/10:
  Train Loss: 0.4123 | Val Loss: 0.3891
  Balanced Accuracy: 0.7891
  ✅ New best model! (0.7891 > 0.7234)

...

Epoch 8/10:
  Train Loss: 0.2345 | Val Loss: 0.2678
  Balanced Accuracy: 0.8567
  🛑 Early stopping triggered at epoch 8

============================================================
  📊 TRAINING SUMMARY
============================================================
  Best Epoch: 5
  Best Balanced Accuracy: 0.8567
  Model saved: output/models/best_model_narrativo_bertm.pth
============================================================
```

### Output Console - Training K-Fold

```
============================================================
  🔄 K-FOLD CROSS VALIDATION
============================================================

================================================================================
  📊 FOLD 1/5
================================================================================
   Train samples: 800
   Val samples:   200
   Train class 0: 65.0%, class 1: 35.0%
   Val   class 0: 65.0%, class 1: 35.0%

[... training fold 1 ...]

✅ Fold 1 completato!
   Best Bal.Acc: 0.8234 @ Epoch 7

================================================================================
  📊 FOLD 2/5
================================================================================
[... fold 2 ...]

[... fold 3, 4, 5 ...]

================================================================================
  📊 K-FOLD RISULTATI AGGREGATI
================================================================================

Metric                     Mean           Std            Min            Max
--------------------------------------------------------------------------------
balanced_accuracy          0.8123         0.0234         0.7789         0.8456
best_epoch                 6.4000         1.2000         5.0000         8.0000

================================================================================
  💾 Modelli salvati: 5
     Fold 0: best_model_narrativo_bertm_fold0.pth
     Fold 1: best_model_narrativo_bertm_fold1.pth
     Fold 2: best_model_narrativo_bertm_fold2.pth
     Fold 3: best_model_narrativo_bertm_fold3.pth
     Fold 4: best_model_narrativo_bertm_fold4.pth
================================================================================

✅ K-Fold Training completato!
   Mean Balanced Accuracy: 0.8123 ± 0.0234
```

---

## 🧠 Best Practices

### 1. Scelta Training Mode

| Scenario | Mode | Motivo |
|----------|------|--------|
| **Sviluppo/Debug** | Simple | Veloce, iterate rapidamente |
| **Test modello** | Simple | Baseline performance |
| **Paper/Pubblicazione** | K-Fold (10 folds) | Metriche robuste, statisticamente valide |
| **Confronto modelli** | K-Fold (5 folds) | Comparazione fair |
| **Produzione** | K-Fold + Ensemble | Best performance |

### 2. Scelta Loss Function

| Scenario | Loss | Config |
|----------|------|--------|
| **Dataset bilanciato (50%-50%)** | Cross Entropy | Default (class weights auto) |
| **Sbilanciato moderato (60%-40%)** | Cross Entropy | Class weights auto |
| **Sbilanciato forte (70%-30%)** | Focal Loss | α=0.3,0.7 γ=2.0 |
| **Sbilanciato estremo (80%-20%)** | Focal Loss | α=0.2,0.8 γ=3.0 |

### 3. Tuning Hyperparameters

**Patience (Early Stopping):**
- Dataset piccolo (<1000): patience=3
- Dataset medio (1000-10k): patience=5
- Dataset grande (>10k): patience=7

**Epochs:**
- Sviluppo rapido: 5-10 epochs
- Training normale: 10-15 epochs
- Training completo: 20-30 epochs

**Learning Rate:**
- BERT-based: 1e-5 (default)
- GPT2: 5e-5
- Fine-tuning: 1e-6

### 4. Gestione Risorse

**Training Semplice:**
- RAM: ~8GB
- GPU: ~4GB VRAM
- Tempo: ~10-15 min

**Training K-Fold (10 folds):**
- RAM: ~8GB
- GPU: ~4GB VRAM
- Tempo: ~100-150 min (10x simple)
- Disk: ~500MB per fold (5GB totale)

**Tip:** Usa `--n_folds 5` per bilanciare tempo/robustezza

---

## 🐛 Troubleshooting

### Problema 1: Out of Memory (GPU)

**Sintomo:** `RuntimeError: CUDA out of memory`

**Soluzione:**
```bash
# Riduci batch size
uv run python src/training/train_llm.py \
    --story_format narrativo \
    --model bertm \
    --batch_size 128  # invece di 256
```

### Problema 2: Training Troppo Lento

**Sintomo:** Training impiega ore

**Soluzione:**
```bash
# Riduci epochs e usa GPU
uv run python src/training/train_llm.py \
    --story_format narrativo \
    --model bertm \
    --epochs 10 \
    --patience 3  # stop early se non migliora
```

### Problema 3: File `label_mapping.json` Non Trovato

**Sintomo:** `eval_model.py` non trova label mapping

**Soluzione:**
```bash
# Assicurati di aver eseguito training prima
ls -la output/reports/label_mapping.json

# Se non esiste, ri-esegui training
uv run python src/training/train_llm.py --story_format narrativo --model bertm
```

### Problema 4: Overfitting (Train Loss << Val Loss)

**Sintomo:** Train loss 0.1, Val loss 0.5

**Soluzione 1 - Early Stopping più aggressivo:**
```bash
uv run python src/training/train_llm.py \
    --story_format narrativo \
    --model bertm \
    --patience 3  # più basso
```

**Soluzione 2 - Focal Loss:**
```bash
uv run python src/training/train_llm.py \
    --story_format narrativo \
    --model bertm \
    --use_focal_loss \
    --focal_gamma 3.0  # più alto = più focus su hard examples
```

---

## 📚 Riferimenti

### Documentazione Correlata
- [PIANO_LAVORI_FUNZIONALITÀ_TRAINING.md](./PIANO_LAVORI_FUNZIONALITÀ_TRAINING.md) - Piano implementazione completo
- [TRAINING_EVALUATION_GUIDE.md](./TRAINING_EVALUATION_GUIDE.md) - Guida training base (legacy)

### Paper/References
- **Focal Loss**: Lin et al. (2017) "Focal Loss for Dense Object Detection"
- **Balanced Accuracy**: Brodersen et al. (2010) "The Balanced Accuracy and Its Posterior Distribution"
- **K-Fold CV**: Kohavi (1995) "A study of cross-validation and bootstrap for accuracy estimation"

### Codice Sorgente
- `src/training/train_llm.py` - Main training script
- `src/training/config.py` - TrainingConfig class
- `src/training/kfold_trainer.py` - KFoldTrainer class
- `src/training/focal_loss.py` - FocalLoss implementation
- `src/training/checkpoint.py` - ModelCheckpoint class
- `src/training/early_stopping.py` - EarlyStopping class

---

## 🎉 Conclusione

Il sistema di training avanzato offre:

✅ **Flessibilità**: CLI completa per ogni scenario  
✅ **Robustezza**: K-Fold CV per metriche affidabili  
✅ **Efficienza**: Early stopping e best model tracking  
✅ **Specializzazione**: Focal Loss per classi sbilanciate  
✅ **Integrabilità**: Label mapping per eval/XAI  

**Quick Start:**
```bash
# Training semplice per test
uv run python src/training/train_llm.py --story_format narrativo --model bertm --epochs 5

# Training robusto per produzione
uv run python src/training/train_llm.py --story_format narrativo --model bertm --use_kfold --n_folds 10 --epochs 15
```

**Per domande o problemi:**
- Consulta questa guida
- Verifica `docs/PIANO_LAVORI_FUNZIONALITÀ_TRAINING.md` per dettagli implementazione
- Check `tests/test_training_phase*.py` per esempi uso API

---

*Documento aggiornato: 9 Ottobre 2025*
