📋 Piano Lavori - Training Avanzato

## 📊 STATO AVANZAMENTO
- **Ultimo aggiornamento**: 9 Ottobre 2025
- **Branch**: `advanced-training`
- **Fase corrente**: FASE 4 - COMPLETATA ✅

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

## ✅ FASE 2: Best Model Tracking [COMPLETATA]
**Obiettivo**: Salvare solo il miglior modello basato su balanced_accuracy

### Task Completati:
- ✅ **2.1**: Creato `src/training/checkpoint.py` con classe `ModelCheckpoint`
  - Traccia best metric (default: balanced_accuracy)
  - Salva solo quando migliora, rimuove vecchio best
  - Path: `best_model_{format}_{model}_fold{k}.pth`
  - Buffer history con tutte le metriche
  - Save/Load checkpoint con optimizer state
  - Factory: `create_checkpoint_from_config()`
  - File: **~330 righe**

### Deliverable:
✅ ModelCheckpoint pronto per integrazione nel training loop  
✅ Test pytest: 9 test cases in `tests/test_training_phase2.py`

### Note:
- Test con simulazione 5 epoche: salva solo epoch 1 e 3 (miglioramenti)
- Checkpoint include: model state, optimizer state, metrics, timestamp

---

## ✅ FASE 3: Early Stopping Avanzato [COMPLETATA]
**Obiettivo**: Implementare early stopping con recupero pesi epoca trigger - patience

### Task Completati:
- ✅ **3.1**: Creato `src/training/early_stopping.py` con classe `EarlyStopping`
  - Delta loss + Train/Val ratio monitoring
  - Buffer ultimi N stati modello (N=patience) con deque
  - Ripristino pesi al best epoch o trigger-patience
  - Due modalità stop: patience esaurita o ratio violations
  - History tracking per analisi post-training
  - Factory: `create_early_stopping_from_config()`
  - File: **~350 righe**

### Deliverable:
✅ EarlyStopping pronto per integrazione nel training loop  
✅ Test pytest: 11 test cases in `tests/test_training_phase3.py`

### Note:
- Test 1: Patience esaurita dopo 3 epoche no improvement
- Test 2: Ratio violations detection (overfitting)
- Test 3: Training completo senza trigger
- Buffer usa deep copy per sicurezza

---

## ✅ FASE 4: K-Fold Cross Validation [COMPLETATA]
**Obiettivo**: K-fold mantenendo distribuzione classi, test set separato

### Task:
- ✅ **4.1**: ~~Creare funzione stratified_train_val_test_split()~~ GIÀ IN FASE 1
- ✅ **4.2**: Creato `src/training/kfold_trainer.py` con classe `KFoldTrainer`
  - Loop su K folds con StratifiedKFold
  - Crea Subset per train/val per ogni fold
  - Integra ModelCheckpoint e EarlyStopping per fold
  - Salva modello per fold: `best_model_{format}_{model}_fold{k}.pth`
  - Aggrega metriche: mean ± std ± min ± max
  - Helper: `get_best_fold()`, `load_fold_model()`
  - Factory: `save_kfold_summary()` per report completo
  - File: **~420 righe**
- ✅ **4.3**: Modificato `train_llm.py` per usare KFoldTrainer (8 sub-task completati)

### Deliverable:
Training con k-fold, k modelli salvati, metriche aggregate  
✅ Test pytest: 9 test cases in `tests/test_training_phase4.py`

### Note Test:
- 3 fold su 100 samples: balanced_accuracy mean=0.53 ± 0.09
- Class distribution mantenuta (~52% vs ~48% in ogni fold)
- Modelli salvati correttamente in output/models/

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
2. ✅ **FASE 2** (Best Model) → Riduce spazio disco subito [COMPLETATA]
3. ✅ **FASE 3** (Early Stop) → Migliora qualità training [COMPLETATA]
4. ✅ **FASE 4** (K-Fold) → Core functionality [COMPLETATA]
   - ✅ 4.1: stratified_train_val_test_split
   - ✅ 4.2: KFoldTrainer
   - ✅ 4.3: Integrazione train_llm.py (8 sub-task)
5. ⏸️ **FASE 5** (Ensemble) → Dipende da Fase 4
6. ⏸️ **FASE 6** (Focal Loss) → Parallelo a Fase 4, indipendente
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

## 📦 File Creati
```
src/training/
├── __init__.py              [Aggiornato con exports]
├── config.py                [~350 righe] TrainingConfig
├── focal_loss.py            [~360 righe] FocalLoss
├── utils.py                 [~470 righe] Split, metrics, weights
├── checkpoint.py            [~330 righe] ModelCheckpoint [FASE 2]
├── early_stopping.py        [~350 righe] EarlyStopping [FASE 3]
├── kfold_trainer.py         [~420 righe] KFoldTrainer [FASE 4]
└── train_llm.py             [~575 righe] Main training script [FASE 4.3]

tests/
├── test_training_phase1.py    [~450 righe] 23 test cases
├── test_training_phase2.py    [~240 righe]  9 test cases
├── test_training_phase3.py    [~220 righe] 11 test cases
├── test_training_phase4.py    [~280 righe]  9 test cases
├── test_training_phase4_3_1.py [~252 righe] 11 test cases [FASE 4.3.1]
├── test_training_phase4_3_2.py [~350 righe]  3 test cases [FASE 4.3.2]
├── test_training_phase4_3_3.py [~200 righe]  9 test cases [FASE 4.3.3]
└── test_training_phase4_3_4.py [~320 righe]  8 test cases [FASE 4.3.4]
```

**Totale FASE 1-4.3.4**: ~3705 righe codice + ~2312 righe test = **6017 righe**  
**83 test cases pytest, tutti passing ✅**

---

## 🔧 FASE 4.3 - Integrazione train_llm.py

### ✅ 4.3.1 - Gestione CLI Arguments e TrainingConfig [COMPLETATO]
**Decisioni:**
- ✅ Argparse (resto progetto lo usa)
- ✅ CLI args: model_name, story_format, use_kfold, n_folds, use_focal_loss, focal_alpha, focal_gamma, patience, epochs
- ✅ Senza --use_kfold → training semplice (backward compatibility)
- ✅ No config YAML, solo CLI

**Implementazione:**
- ✅ Aggiunto argparse con 10 parametri CLI
- ✅ Funzione `parse_args()` per parsing argomenti
- ✅ Funzione `create_training_config(args)` per creare TrainingConfig
- ✅ Sostituiti STORY_FORMAT, LEARNING_RATE, BATCH con config
- ✅ Mapping: epochs→num_epochs, patience→early_stopping_patience, use_focal_loss→loss_function
- ✅ Test: 11 test cases, tutti passing ✅
- ✅ File: tests/test_training_phase4_3_1.py (~260 righe)

### ✅ 4.3.2 - Refactor pre_train() - Signature e Early Stopping [COMPLETATO]
**Decisioni:**
- ✅ Tutto su TrainingConfig (rimuovi num_epochs, min_loss, model_output_basename)
- ✅ Rimuovi start_epoch (sempre 0)
- ✅ Nome modello: config.get_model_filename(fold)
- ✅ Accelerator globale

**Implementazione:**
- ✅ Nuova signature: pre_train(model, optimizer, train_dataloader, val_dataloader, scheduler, criterion, accelerator, config, checkpoint, early_stopping, fold)
- ✅ Rimosso patience_counter custom, usato EarlyStopping class
- ✅ Compute metrics con balanced_accuracy come metrica principale
- ✅ ModelCheckpoint.update() per salvare best model
- ✅ EarlyStopping.update() + should_stop() per controllo
- ✅ restore_weights() per ripristinare best model al trigger
- ✅ Logging dettagliato per epoch (train/val loss e balanced_accuracy)
- ✅ Test: 3 test cases, tutti passing ✅
- ✅ File: tests/test_training_phase4_3_2.py (~350 righe)

### ✅ 4.3.3 - Loss Function [COMPLETATO]
**Decisioni:**
- ✅ Class weights SOLO per CrossEntropyLoss (calcolo automatico)
- ✅ Focal Loss: gestisce già pesi con α/γ (no class weights)
- ✅ Metodo calcolo pesi: fisso 'balanced', no parametro CLI

**Implementazione:**
- ✅ Import `create_loss_from_config()` e `compute_class_weights()`
- ✅ Path Focal Loss: usa `create_loss_from_config(config)` con α/γ da config
- ✅ Path Cross Entropy: calcola class weights con metodo 'balanced', crea `nn.CrossEntropyLoss(weight=weights_tensor)`
- ✅ Logging configurazione loss function all'avvio
- ✅ Test: 9 test cases, tutti passing ✅
- ✅ File: tests/test_training_phase4_3_3.py (~200 righe)

### ✅ 4.3.4 - Checkpoint e Model Saving [COMPLETATO]
**Decisioni:**
- ✅ Metrica: balanced_accuracy
- ✅ History gestito da checkpoint.save_history()
- ✅ Stampe centralizzate

**Implementazione:**
- ✅ Salvataggio automatic history alla fine del training con `checkpoint.save_history()`
- ✅ Logging info early stopping se triggerato (trigger epoch, best val loss, wait count)
- ✅ Summary finale con best epoch, best balanced_accuracy, path modello salvato
- ✅ History salvata in JSON: `checkpoint_history.json` (o `checkpoint_history_fold{k}.json` per K-Fold)
- ✅ Test: 8 test cases, tutti passing ✅
- ✅ File: tests/test_training_phase4_3_4.py (~320 righe)

### ✅ 4.3.5 - K-Fold Wrapper [COMPLETATO]
**Decisioni:**
- ✅ Usa KFoldTrainer class (FASE 4.0) invece di logica custom
- ✅ Conditional routing: if config.use_kfold → KFoldTrainer else → simple training
- ✅ Model factory per creare modelli freschi per ogni fold
- ✅ Training function wrapper per chiamare pre_train() per ogni fold
- ✅ Dataset combinato da X_train + X_val per K-Fold

**Implementazione:**
- ✅ Import `KFoldTrainer` da `src.training.kfold_trainer`
- ✅ Conditional routing nella sezione main basato su `config.use_kfold`
- ✅ Model factory function: gestisce sia GPT2 che BERT-based models
- ✅ Train function wrapper: setup optimizer/scheduler/dataloader per fold, chiama pre_train()
- ✅ Aggregazione risultati: mean ± std balanced_accuracy
- ✅ Backward compatibility: simple training preservato senza modifiche
- ✅ No test specifici (logica già coperta da 9 test FASE 4.0)

### ✅ 4.3.6 - Data Loading e Split Stratificato [COMPLETATO]
**Decisioni:**
- ✅ Test set già separato upstream (train.pickle / test.pickle) → manteniamo
- ✅ Split train/val già stratificato con sklearn → manteniamo (funziona bene)
- ✅ Label mapping generico: `CLS_0`, `CLS_1` invece di label specifiche
- ✅ Salva label2id/id2label in JSON per eval_model.py e extract_explainability.py

**Implementazione:**
- ✅ Import `analyze_class_distribution` da utils
- ✅ Creazione label mapping generico per export (`CLS_0`, `CLS_1`)
- ✅ Salvataggio JSON: `output/reports/label_mapping.json`
- ✅ Formato: `{"label2id": {"CLS_0": 0, ...}, "id2label": {"0": "CLS_0", ...}, "num_classes": 2}`
- ✅ Logging distribuzione classi per train/val/test set
- ✅ No test specifici (logica semplice: JSON save + print stats)

**Note:**
- Manteniamo `train_test_split` sklearn (già stratificato, no benefici a cambiare)
- `stratified_train_val_test_split()` da FASE 1 rimane inutilizzato → vedi Refactoring Futuro

### ✅ 4.3.7 - Test di Integrazione [COMPLETATO]
**Decisioni:**
- ✅ Test con dati mock per velocità
- ✅ Test path corretti file salvati
- ✅ Test integrazione K-Fold routing
- ✅ No test GPU (troppo specifici)

**Implementazione:**
- ✅ Test implementati dall'utente
- ✅ Tutti i test passano
- ✅ Copertura completa funzionalità

### ✅ 4.3.8 - Documentazione [COMPLETATO]
**Decisioni:**
- ✅ File separato: docs/TRAIN_LLM_INTEGRATION.md
- ✅ Riferimento nel PIANO_LAVORI
- ✅ Esempi bash completi
- ✅ Best practices e troubleshooting

**Implementazione:**
- ✅ Creato `docs/TRAIN_LLM_INTEGRATION.md` (~600 righe)
- ✅ Sezioni:
  * Panoramica funzionalità
  * Utilizzo CLI e script bash
  * Parametri completi
  * Output e file generati
  * Workflow completo
  * Esempi pratici (4 scenari)
  * Interpretazione output console
  * Best practices
  * Troubleshooting
  * Riferimenti
- ✅ Esempi per ogni scenario: simple, K-Fold, Focal, K-Fold+Focal
- ✅ Guida scelta training mode e loss function
- ✅ Tuning hyperparameters e gestione risorse

**Riferimento completo:** [TRAIN_LLM_INTEGRATION.md](./TRAIN_LLM_INTEGRATION.md)

---

## 🔄 REFACTORING FUTURO

### Codice Inutilizzato da Rimuovere/Consolidare

**1. `stratified_train_val_test_split()` in `src/training/utils.py`**
- **Stato**: Implementata in FASE 1.3, mai utilizzata
- **Motivo**: `train_llm.py` usa `train_test_split` di sklearn (già stratificato, test set già separato upstream)
- **Azione futura**: 
  - Opzione A: Rimuovere se confermato che non serve
  - Opzione B: Refactor pipeline XES per usarla upstream (split train/val/test prima del pickle)
- **Priorità**: Bassa (non impatta funzionalità)

**2. Possibili Duplicazioni da Verificare**
- **Da verificare**: Controllare se ci sono altre utility FASE 1 non utilizzate
- **Action**: Audit completo dopo FASE 8 (quando tutto è integrato)

### Architettura Modelli - Refactoring da Multi-Classe a Generico

**3. Neural Network Classes - ✅ COMPLETATO (9 Ottobre 2025)**
- **File**: `src/models/neural_network.py`
- **Problema originale**: 
  - `LongFormerMultiClassificationHeads`: 8 classi hardcoded nel layer finale
  - `SimpleGPT2SequenceClassifier`: Riceve `num_classes` ma in `train_llm.py` era chiamato con 8 hardcoded
  - Progetto originale era multi-classe (8 DRG), ora deve supportare N-classi generico
- **Soluzione implementata**:
  - ✅ Aggiunto parametro `num_classes` a `LongFormerMultiClassificationHeads.__init__(num_classes=8)`
  - ✅ Default a 8 per backward compatibility con codice legacy
  - ✅ Aggiornato `train_llm.py` (linee 543-563) per usare `config.num_classes`
  - ✅ Aggiornato `model_factory` (linee 620-637) per usare `config.num_classes`
  - ✅ Ora completamente generico: supporta 2, 3, 8, N classi
- **Testing**: Verificare con classificazione binaria (2 classi) e multi-classe (3+)
- **Riferimento**: Bug #4 durante test K-Fold, refactoring 9 Ottobre 2025

---

Vuoi mantenere anche l'opzione di training "semplice" (senza k-fold) per test rapidi?
Parti da quale FASE? Suggerisco FASE 1 per avere le basi pronte.
Dammi il via libera e partiamo dalla FASE 1! 🚀