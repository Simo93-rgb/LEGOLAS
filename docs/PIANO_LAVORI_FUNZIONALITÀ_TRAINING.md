📋 Piano Lavori - Training Avanzato

## 📊 STATO AVANZAMENTO
- **Ultimo aggiornamento**: 8 Ottobre 2025
- **Branch**: `advanced-training`
- **Fase corrente**: FASE 1 ✅ COMPLETATA

---

## ✅ FASE 1: Setup Configurazione e Utilities [COMPLETATA]
**Obiettivo**: Creare infrastruttura base

### Task Completati:
- ✅ **1.1**: Creato `src/training/config.py` con classe `TrainingConfig`
  - Dataclass con tutti i parametri di training
  - K-Fold CV: `use_kfold`, `n_folds=10`, stratificato
  - Focal Loss: `focal_alpha=[0.25, 0.75]`, `focal_gamma=2.0`
  - Early Stopping: patience=5, delta+ratio monitoring
  - Best model tracking: balanced_accuracy
  - Path management integrato con struttura LEGOLAS
  - Factory functions: `create_default_config()`, `create_kfold_config()`
  - Validazione configurazione con `validate()`
  - File: **~350 righe**

- ✅ **1.2**: Creato `src/training/focal_loss.py` con implementazione Focal Loss
  - Classe `FocalLoss(nn.Module)` completa
  - Formula: `FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)`
  - Supporto alpha (class weights) e gamma (focusing parameter)
  - Reduction modes: mean, sum, none
  - Factory function: `create_focal_loss()`
  - Integration helper: `create_loss_from_config()`
  - Test comparativi Focal vs CrossEntropy
  - File: **~360 righe**

- ✅ **1.3**: Creato `src/training/utils.py` con funzioni utility
  - `stratified_train_val_test_split()`: Split stratificato train/val/test
  - `create_stratified_kfold()`: StratifiedKFold per CV
  - `compute_class_weights()`: Calcolo pesi per classi sbilanciate
  - `compute_metrics()`: Accuracy, balanced_accuracy, precision, recall, F1
  - `compute_detailed_metrics()`: Metriche + confusion matrix + report
  - `save_metrics()` / `load_metrics()`: I/O metriche JSON
  - `analyze_class_distribution()`: Analisi distribuzione classi
  - File: **~470 righe**

- ✅ **1.4**: Creato `tests/test_training_phase1.py` con test suite pytest
  - Test `TrainingConfig`: creazione, validazione, paths, factory
  - Test `FocalLoss`: forward, backward, alpha weighting, reduction
  - Test utilities: split stratificato, class weights, metriche
  - File: **~450 righe**, 30+ test cases

- ✅ **1.5**: Aggiornato `src/training/__init__.py` con exports
  - Tutti i moduli esportati correttamente
  - `__all__` completo per import puliti

### Deliverable:
✅ Moduli riutilizzabili pronti per l'integrazione  
✅ Test suite completa per validazione  
✅ Documentazione inline completa con esempi

### Note Tecniche:
- Usare `uv run python` per tutti i comandi Python
- Test imports con path assoluti (no modifica sys.path)
- Focal Loss threshold separato per ratio monitoring (20.0 vs 1.15)
- Class weights normalizzati (somma = num_classes)

---

## 🔄 FASE 2: Best Model Tracking [PROSSIMA]
**Obiettivo**: Salvare solo il miglior modello basato su balanced_accuracy

### Task:
- ⚠️ **2.1**: Creare classe `ModelCheckpoint` in `src/training/checkpoint.py`
  - Traccia best balanced_accuracy
  - Salva modello solo se migliora
  - Gestisce path `best_model_{format}_{model}_fold{k}.pth`
- ⚠️ **2.2**: Integrare ModelCheckpoint nel training loop esistente
- ⚠️ **2.3**: Rimuovere salvataggio ad ogni epoca (o renderlo opzionale)

### Deliverable:
Training salva solo best model per fold

---

## 🛑 FASE 3: Early Stopping Avanzato [PIANIFICATA]
**Obiettivo**: Implementare early stopping con recupero pesi epoca trigger - patience

### Task:
- ⏸️ **3.1**: Creare classe `EarlyStopping` in `src/training/early_stopping.py`
  - Delta loss + Train/Val ratio check
  - Buffer di stati modello (ultimi N stati dove N=patience)
  - Ripristino pesi a epoca `trigger_epoch - patience`
- ⏸️ **3.2**: Integrare nel training loop
- ⏸️ **3.3**: Test: verificare ripristino corretto dei pesi

### Deliverable:
Early stopping funzionante con recupero pesi corretti

---

## 🔀 FASE 4: K-Fold Cross Validation con Stratified Split [PIANIFICATA]
**Obiettivo**: K-fold mantenendo distribuzione classi, test set separato

### Task:
- ⏸️ **4.1**: ~~Creare funzione stratified_train_val_test_split() in utils.py~~ ✅ GIÀ IN FASE 1
- ⏸️ **4.2**: Creare `KFoldTrainer` in `src/training/kfold_trainer.py`
  - Loop su K folds
  - Salva modello per fold: `best_model_{format}_{model}_fold{k}.pth`
  - Aggrega metriche: mean ± std
- ⏸️ **4.3**: Modificare `train_llm.py` per usare KFoldTrainer

### Deliverable:
Training con k-fold, k modelli salvati, metriche aggregate

---

## 🎭 FASE 5: Ensemble Prediction [PIANIFICATA]
**Obiettivo**: Usare ensemble dei k modelli per predizione/explainability

### Task:
- ⏸️ **5.1**: Creare `EnsembleModel` in `src/models/ensemble.py`
  - Carica k modelli da fold
  - Averaging o voting per predizione
- ⏸️ **5.2**: Modificare `eval_model.py` per usare ensemble
- ⏸️ **5.3**: Modificare `extract_explainability.py` per usare ensemble (o best fold)

### Deliverable:
Sistema ensemble funzionante

---

## 🎯 FASE 6: Focal Loss Integration [PIANIFICATA]
**Obiettivo**: Focal Loss come opzione da command line

### Task:
- ⏸️ **6.1**: ~~Implementare Focal Loss~~ ✅ GIÀ IN FASE 1
- ⏸️ **6.2**: Aggiungere parametro `--loss-function` a `train_llm.py` (choices: focal, ce)
- ⏸️ **6.3**: Factory function per creare loss appropriata ✅ GIÀ IN FASE 1
- ⏸️ **6.4**: Test confronto focal vs crossentropy ✅ GIÀ IN FASE 1

### Deliverable:
Focal loss selezionabile, default focal

---

## 📝 FASE 7: Logging e Script Updates [PIANIFICATA]
**Obiettivo**: Log completi e redirect su file negli script bash

### Task:
- ⏸️ **7.1**: Creare logger testuale in `src/training/logger.py`
  - Salva in `output/logs/training_{timestamp}.log`
  - Print a schermo + file simultaneo
- ⏸️ **7.2**: Aggiornare script bash per redirect output
  - `launch_run_train_eval` → tee `output/logs/train_{timestamp}.log`
  - `run_explainability.sh` → tee `output/logs/xai_{timestamp}.log`

### Deliverable:
Logging completo, tutto tracciato su file

---

## 🧪 FASE 8: Testing e Validation [PIANIFICATA]
**Obiettivo**: Verificare tutto funziona end-to-end

### Task:
- ⏸️ **8.1**: Test completo k-fold con focal loss
- ⏸️ **8.2**: Verificare early stopping e recupero pesi
- ⏸️ **8.3**: Test ensemble prediction
- ⏸️ **8.4**: Documentazione in `docs/ADVANCED_TRAINING.md`

### Deliverable:
Sistema completo testato e documentato

---

## 🎯 Ordine di Esecuzione
1. ✅ **FASE 1** (Setup) → Base per tutto [COMPLETATA]
2. ⚠️ **FASE 2** (Best Model) → Riduce spazio disco subito [PROSSIMA]
3. ⏸️ **FASE 3** (Early Stop) → Migliora qualità training
4. ⏸️ **FASE 4** (K-Fold) → Core functionality
5. ⏸️ **FASE 6** (Focal Loss) → Parallelo a Fase 4, indipendente
6. ⏸️ **FASE 5** (Ensemble) → Dipende da Fase 4
7. ⏸️ **FASE 7** (Logging) → Polish
8. ⏸️ **FASE 8** (Testing) → Validazione finale

---

## ✅ Conferme Finali
- ✅ K-fold: k=10 default, test stratificato separato (20%)
- ✅ Best model: Solo balanced_accuracy, un file per fold
- ✅ Early stopping: epoca = trigger - patience
- ✅ Focal loss: Default, parametri standard (α=0.25, γ=2.0)
- ✅ Ensemble: Media predizioni dei k modelli
- ✅ Mantieni opzione training semplice (non k-fold) per sviluppo rapido
- ✅ Test pytest per ogni fase

---

## 📦 File Creati in FASE 1
```
src/training/
├── __init__.py          [Aggiornato con exports]
├── config.py            [~350 righe] TrainingConfig + factory
├── focal_loss.py        [~360 righe] FocalLoss + tests
└── utils.py             [~470 righe] Split, metrics, weights

tests/
└── test_training_phase1.py  [~450 righe] 30+ test cases
```

**Totale**: ~1630 righe di codice + test, infrastruttura completa pronta!

---

Vuoi mantenere anche l'opzione di training "semplice" (senza k-fold) per test rapidi?
Parti da quale FASE? Suggerisco FASE 1 per avere le basi pronte.
Dammi il via libera e partiamo dalla FASE 1! 🚀