📋 Piano Lavori - Training Avanzato

## 📊 STATO AVANZAMENTO
- **Ultimo aggiornamento**: 10 Ottobre 2025
- **Branch**: `advanced-training`
- **Fase corrente**: FASE 4 - COMPLETATA ✅
- **Prossima**: FASE 5 - Ensemble Prediction

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
- ✅ **4.3**: Modificato `train_llm.py` per usare KFoldTrainer
  - **Dettagli completi**: Vedi sezione [FASE 4.3 - Integrazione train_llm.py](#-fase-43---integrazione-train_llmpy) in fondo al documento
  - **8 sub-task completati**: CLI args, pre_train refactor, loss function, checkpoint, K-Fold wrapper, data loading, test, documentazione
  - **Bug fixes**: 4 bug risolti durante test real-world
  - **Refactoring**: num_classes configurabile, logging K-Fold migliorato

### Deliverable:
✅ Training con k-fold, k modelli salvati, metriche aggregate  
✅ Test pytest: 9 test cases in `tests/test_training_phase4.py`  
✅ Integration completa train_llm.py con tutte le nuove features

### Note Test:
- 3 fold su 100 samples: balanced_accuracy mean=0.53 ± 0.09
- Class distribution mantenuta (~52% vs ~48% in ogni fold)
- Modelli salvati correttamente in output/models/
- Test real-world con 5-fold completato con successo
- Early stopping triggera correttamente su alcuni fold

---

## 🎭 FASE 5: Ensemble Prediction [IN PIANIFICAZIONE]
**Obiettivo**: Usare ensemble dei k modelli per predizione e explainability tramite averaging

### Motivazione Tecnica
**Perché Averaging e non Voting?**
- ✅ **Compatibile con XAI**: Averaging dei pesi permette di estrarre feature importance media
- ✅ **Integrated Gradients**: Algoritmo IG già implementato lavora su pesi → media pesi dei k modelli
- ✅ **Consistenza**: Predizione e XAI usano stesso meccanismo (averaging)
- ❌ **Voting**: Non compatibile con estrazione pesi per XAI → scartato

### Task:
- ⏸️ **5.1**: Creare `EnsembleModel` in `src/models/ensemble.py`
  - Carica k modelli da fold (paths in `output/models/{format}_{model}_kfold/fold_*/`)
  - Metodo `predict()`: averaging delle probabilità (softmax) dei k modelli
  - Metodo `get_averaged_weights()`: media dei pesi per XAI
  - Helper: `load_best_fold()` per caricare solo il miglior fold
- ⏸️ **5.2**: Modificare `eval_model.py` per usare ensemble
  - Parametro `--use_ensemble` per abilitare ensemble vs singolo modello
  - Se ensemble: carica tutti i fold, fa averaging
  - Se singolo: usa best fold (massima balanced_accuracy)
- ⏸️ **5.3**: Modificare `extract_explainability.py` per usare ensemble
  - Parametro `--use_ensemble` per abilitare ensemble
  - Averaging pesi modelli prima di applicare Integrated Gradients
  - Estrazione feature importance su modello medio

### Deliverable:
✅ EnsembleModel class funzionante  
✅ Eval e XAI con supporto ensemble  
✅ Test pytest per ensemble averaging  
✅ Documentazione integrazione ensemble

### Note Implementative:
- Averaging: `prob_avg = mean([model_i.predict(x) for i in range(k)])`
- Weights averaging: `W_avg = mean([model_i.state_dict() for i in range(k)])`
- Compatibilità IG: media pesi → single model equivalente → IG standard

---

## ✅ FASE 6: Focal Loss Integration [COMPLETATA]
**Obiettivo**: Focal Loss come opzione da command line

### Task Completati:
- ✅ **6.1**: Implementare Focal Loss → Completato in FASE 1.2
- ✅ **6.2**: Parametro `--use_focal_loss` in `train_llm.py` → Completato in FASE 4.3.1
- ✅ **6.3**: Factory function `create_loss_from_config()` → Completato in FASE 1.2
- ✅ **6.4**: Test confronto focal vs crossentropy → Completato in FASE 1.4

### Implementazione:
- CLI: `--use_focal_loss --focal_alpha 0.25 0.75 --focal_gamma 2.0`
- Default: Focal Loss abilitata per gestione classi sbilanciate
- CrossEntropy: `--use_focal_loss` NON specificato, calcola class weights automatici
- Parametri configurabili: α (alpha per classe), γ (focusing parameter)

### Deliverable:
✅ Focal loss selezionabile da CLI  
✅ Default focal, fallback CrossEntropy con class weights  
✅ Test completi (9 test cases in test_training_phase4_3_3.py)

### Riferimenti:
- Implementazione: `src/training/focal_loss.py` (~360 righe)
- Integrazione: `src/training/train_llm.py` (FASE 4.3.3)
- Documentazione: [TRAIN_LLM_INTEGRATION.md](./TRAIN_LLM_INTEGRATION.md)

---

## 📝 FASE 7: Logging su File (Script Bash) [DA FARE]
**Obiettivo**: Redirect output Python su file tramite script bash, mantenendo output console

### Motivazione
- ❌ **NO Logger Python**: Evita codice aggiuntivo e complessità
- ✅ **Bash `tee`**: Soluzione semplice, output duplicato (console + file)
- ✅ **Mantiene stampe real-time**: Nessuna modifica al codice Python

### Task:
- ⏸️ **7.1**: Aggiornare `scripts/launch_run_train_eval`
  - Aggiungere `tee` per salvare in `output/logs/train_${TIMESTAMP}.log`
  - Formato: `uv run python ... | tee output/logs/train_$(date +%Y%m%d_%H%M%S).log`
  - Mantiene output colorato e real-time a schermo
  
- ⏸️ **7.2**: Aggiornare `scripts/run_explainability.sh`
  - Aggiungere `tee` per salvare in `output/logs/xai_${TIMESTAMP}.log`
  - Stesso pattern: `uv run python ... | tee output/logs/xai_$(date +%Y%m%d_%H%M%S).log`

- ⏸️ **7.3**: Creare directory `output/logs/` se non esiste
  - Aggiungere `mkdir -p output/logs` nei script

### Esempio Implementazione:
```bash
# PRIMA
eval $CMD

# DOPO
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p output/logs
eval $CMD 2>&1 | tee "output/logs/train_${TIMESTAMP}.log"
```

### Deliverable:
✅ Output salvato automaticamente in file con timestamp  
✅ Console output preservato (nessuna modifica esperienza utente)  
✅ No codice Python aggiuntivo

### Note:
- `tee`: duplica output (stdout + file)
- `2>&1`: redirect stderr a stdout (cattura anche errori)
- Timestamp: formato `YYYYMMDD_HHMMSS` per sorting automatico

---

## 🧪 FASE 8: Testing, Validation e Documentazione [IN CORSO]
**Obiettivo**: Verificare funzionamento end-to-end e documentazione completa

### ✅ Task Completati:
- ✅ **8.1**: Test completo K-Fold con Focal Loss
  - Training eseguito con successo
  - Modelli salvati per ogni fold in `output/models/`
  - JSON metriche per ogni fold con history completo
  - **Risultato**: Funzionante, early stopping triggera correttamente

- ✅ **8.2**: Verificare Early Stopping e recupero pesi
  - Alcuni addestramenti hanno triggerato patience
  - Early stopping attivato correttamente
  - Ripristino pesi best epoch funzionante
  - **Risultato**: Funzionante, sistema robusto

### ⏸️ Task Rimanenti:
- ⏸️ **8.3**: Test Ensemble Prediction
  - Dipende da completamento FASE 5
  - Test averaging predizioni
  - Test averaging pesi per XAI

- ⏸️ **8.4**: Documentazione Completa (4 sub-task)

### 📚 8.4 - Documentazione Completa

#### ⏸️ **8.4.1**: Audit Documentazione Esistente
**Obiettivo**: Capire cosa c'è nella cartella `docs/`

**Task**:
1. Leggere tutti i documenti in `docs/`
2. Creare summary per ogni documento:
   - Titolo e scopo
   - Contenuto principale
   - Stato (aggiornato/obsoleto/incompleto)
   - Dipendenze/riferimenti ad altri doc
3. Creare matrice documenti:
   - Quali si sovrappongono?
   - Quali hanno info obsolete?
   - Quali mancano?

**Output**: `docs/DOCUMENTATION_AUDIT.md` con summary completo

#### ⏸️ **8.4.2**: Pulizia Documentazione
**Obiettivo**: Rimuovere documenti inutili o duplicati

**Task**:
1. Identificare documenti da eliminare:
   - Obsoleti (info superate da refactoring)
   - Duplicati (stesse info in più file)
   - Temporary (guide intermedie non più rilevanti)
2. Eliminare file inutili
3. Archiviare in `docs/archive/` se necessario

**Output**: Cartella `docs/` pulita e organizzata

#### ⏸️ **8.4.3**: Aggiornamento Documentazione
**Obiettivo**: Portare in pari i documenti utili con lo stato del progetto

**Task**:
1. Per ogni documento utile:
   - Verificare accuratezza tecnica
   - Aggiornare comandi/path modificati
   - Aggiungere info mancanti (es: K-Fold, Focal Loss)
   - Correggere esempi obsoleti
2. Creare nuovi documenti se necessario:
   - User guide completa
   - API reference
   - Troubleshooting guide

**Output**: Documentazione aggiornata e accurata

#### ⏸️ **8.4.4**: README.md Principale
**Obiettivo**: README snello con link a documentazione dettagliata

**Struttura README**:
```markdown
# LEGOLAS Fork - Advanced Training & Explainability

## 📖 Descrizione
Fork del progetto LEGOLAS con funzionalità avanzate...

## 👥 Autori e Crediti
### Progetto Originale: LEGOLAS
- **Citazione**: [Link articolo scientifico]
- **Repository originale**: [Link]

### Fork - Advanced Training
- **Autore**: Simone Garau ([@Simo93-rgb](https://github.com/Simo93-rgb))
- **Supervisori**:
  - Prof. Stefania Montani
  - Prof. Giorgio Leonardi
  - Prof. Manuel Striani

## 🚀 Quick Start
[Link a guida rapida]

## 📚 Documentazione
- [Training Guide](docs/TRAIN_LLM_INTEGRATION.md)
- [K-Fold Cross Validation](docs/...)
- [Explainability](docs/...)
- [API Reference](docs/...)

## 🛠️ Features
- K-Fold Cross Validation
- Focal Loss
- Integrated Gradients
- ...

## 📦 Installation
[Istruzioni]

## 🧪 Testing
[Comandi pytest]

## 📄 License
[License info]
```

**Output**: README.md professionale e completo

### Deliverable FASE 8:
✅ Sistema completo testato (8.1, 8.2)  
⏸️ Ensemble prediction validato (8.3)  
⏸️ Documentazione audit (8.4.1)  
⏸️ Documentazione pulita (8.4.2)  
⏸️ Documentazione aggiornata (8.4.3)  
⏸️ README.md professionale (8.4.4)

---

## 🎯 Ordine di Esecuzione e Stato

### ✅ Completate:
1. ✅ **FASE 1** (Setup Base) - Configurazione, Focal Loss, Utilities
2. ✅ **FASE 2** (Best Model Tracking) - ModelCheckpoint
3. ✅ **FASE 3** (Early Stopping) - EarlyStopping avanzato
4. ✅ **FASE 4** (K-Fold CV) - KFoldTrainer + integrazione train_llm.py
5. ✅ **FASE 6** (Focal Loss CLI) - Completata durante FASE 4

### ⏸️ In Corso:
6. ⏸️ **FASE 8** (Testing & Validation) - 8.1 e 8.2 completati, 8.3 e 8.4 da fare

### 📋 Da Fare:
7. ⏸️ **FASE 5** (Ensemble) - Averaging per predizione e XAI
8. ⏸️ **FASE 7** (Logging Bash) - Script `tee` per salvare output

### Dipendenze:
- **FASE 5** → Necessaria per **FASE 8.3** (test ensemble)
- **FASE 8.4** → Può iniziare subito (indipendente da FASE 5/7)
- **FASE 7** → Indipendente, può essere fatta in parallelo

### 🎯 Prossimi Step Consigliati:
1. **FASE 8.4.1-8.4.2**: Audit e pulizia documentazione (non richiede codice)
2. **FASE 5**: Implementazione ensemble (core functionality)
3. **FASE 7**: Logging bash (veloce, 1-2 modifiche)
4. **FASE 8.3**: Test ensemble (dopo FASE 5)
5. **FASE 8.4.3-8.4.4**: Aggiornamento doc + README finale

---

## � Statistiche Progetto

### Codice Produzione:
```
src/training/
├── config.py                [~350 righe] TrainingConfig
├── focal_loss.py            [~360 righe] FocalLoss
├── utils.py                 [~470 righe] Utilities
├── checkpoint.py            [~330 righe] ModelCheckpoint
├── early_stopping.py        [~350 righe] EarlyStopping
├── kfold_trainer.py         [~420 righe] KFoldTrainer
└── train_llm.py             [~760 righe] Main script (refactored)

src/models/
└── neural_network.py        [~28 righe] LongFormer + GPT2 (refactored num_classes)

Total: ~3068 righe codice produzione
```

### Test Suite:
```
tests/
├── test_training_phase1.py         [~450 righe] 23 test cases
├── test_training_phase2.py         [~240 righe]  9 test cases
├── test_training_phase3.py         [~220 righe] 11 test cases
├── test_training_phase4.py         [~280 righe]  9 test cases
├── test_training_phase4_3_1.py     [~252 righe] 11 test cases
├── test_training_phase4_3_2.py     [~350 righe]  3 test cases
├── test_training_phase4_3_3.py     [~200 righe]  9 test cases
├── test_training_phase4_3_4.py     [~320 righe]  8 test cases
└── test_num_classes_refactor.py    [~110 righe] 10 test cases

Total: ~2422 righe test, 93 test cases, tutti passing ✅
```

### Documentazione:
```
docs/
├── PIANO_LAVORI_FUNZIONALITÀ_TRAINING.md  [~600 righe] Questo documento
├── TRAIN_LLM_INTEGRATION.md               [~600 righe] Guida completa training
├── REFACTORING_NUM_CLASSES.md             [~150 righe] Doc refactoring
├── BUGFIX_MODEL_CONFIG.md
├── EVALUATION_LOGGING.md
├── EXPLAINABILITY_*.md
└── ... (altri documenti da auditare in FASE 8.4.1)

Total: ~1350+ righe documentazione (stima, da auditare)
```

### 🎉 Totale FASE 1-4 + Refactoring:
- **~3068 righe** codice produzione
- **~2422 righe** test suite (93 test cases)
- **~1350+ righe** documentazione
- **~6840+ righe TOTALI**
- **Test coverage**: 93 test passing ✅
- **Branch**: `advanced-training`
- **Commit**: Multiple incrementali con storia completa

---

## 🔧 FASE 4.3 - Integrazione train_llm.py (Dettagli Tecnici)

> **Nota**: Questa sezione contiene i dettagli tecnici completi dell'integrazione di tutte le features in `train_llm.py`. Per il riepilogo vedi [FASE 4](#-fase-4-k-fold-cross-validation-completata).

---

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
- **Testing**: 
  - ✅ Test suite completa in `tests/test_num_classes_refactor.py` (10 test cases)
  - ✅ Verificato con classificazione binaria (2 classi) e multi-classe (3, 8, 10)
- **Riferimento**: Bug #4 durante test K-Fold, refactoring 9 Ottobre 2025
- **Documentazione**: [REFACTORING_NUM_CLASSES.md](./REFACTORING_NUM_CLASSES.md)

---

## 📍 Summary e Prossimi Passi

### ✅ Cosa Abbiamo Completato (FASE 1-4, 6)
- **Infrastruttura completa**: Config, Focal Loss, Utilities, Checkpoint, Early Stopping
- **K-Fold Cross Validation**: Training robusto con stratificazione e metriche aggregate
- **CLI completo**: Parametri configurabili per tutte le features
- **Testing**: 93 test cases passing, test real-world completati
- **Refactoring**: Architettura modelli generica per N classi
- **Documentazione**: Guide complete per training e integrazione

### 🎯 Prossimi Step Immediati
1. **FASE 8.4.1**: Audit documentazione (`docs/`) - Capire cosa c'è e cosa serve
2. **FASE 5**: Implementare `EnsembleModel` per averaging predizioni e pesi (XAI compatibility)
3. **FASE 7**: Aggiungere `tee` negli script bash per logging su file
4. **FASE 8.3**: Test ensemble prediction (dopo FASE 5)
5. **FASE 8.4.2-8.4.4**: Pulizia, aggiornamento doc, README professionale

### 📋 Roadmap Finale
```
[✅ COMPLETATE] FASE 1, 2, 3, 4, 6
    ↓
[⏸️ DA FARE] FASE 8.4.1 (Audit docs) → Può iniziare subito
    ↓
[⏸️ DA FARE] FASE 5 (Ensemble) → Core functionality
    ↓
[⏸️ DA FARE] FASE 7 (Logging bash) → Quick win
    ↓
[⏸️ DA FARE] FASE 8.3 (Test ensemble) → Dopo FASE 5
    ↓
[⏸️ DA FARE] FASE 8.4.2-8.4.4 (Doc finale + README) → Polish
    ↓
[✅ MERGE] advanced-training → master
```

### 🎉 Progetto LEGOLAS Fork - Advanced Training
**Repository**: https://github.com/Simo93-rgb/LEGOLAS  
**Branch**: `advanced-training`  
**Autore Fork**: Simone Garau ([@Simo93-rgb](https://github.com/Simo93-rgb))  
**Supervisori**: Prof. Stefania Montani, Prof. Giorgio Leonardi, Prof. Manuel Striani  
**Progetto Originale**: LEGOLAS - Articolo scientifico [citazione da aggiungere in README]

---

**Fine Piano Lavori**  
*Ultimo aggiornamento: 10 Ottobre 2025*
