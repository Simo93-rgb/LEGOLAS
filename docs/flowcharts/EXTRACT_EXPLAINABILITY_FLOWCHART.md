# Extract Explainability - Diagramma di Flusso

## Panoramica
Script per l'estrazione degli attribution scores usando Integrated Gradients e generazione di visualizzazioni per l'interpretabilità dei modelli.

---

## 🔄 Flusso Principale

```mermaid
flowchart TD
    Start([Avvio Script]) --> ParseArgs[Parse Argomenti CLI]
    ParseArgs --> PrintConfig[Stampa Configurazione]
    
    PrintConfig --> EnsureDir[Crea Directory Output]
    EnsureDir --> LoadData[Load Test/Train Data]
    
    LoadData --> GetNumClasses{Determina num_classes}
    GetNumClasses --> LoadModel[Load Trained Model]
    
    LoadModel --> CheckEnsemble{use_ensemble?}
    CheckEnsemble -->|Yes| LoadEnsemble[Carica EnsembleModel<br/>tutti i fold]
    CheckEnsemble -->|No| LoadSingle[Carica Best Fold<br/>modello singolo]
    
    LoadEnsemble --> LoadTokenizer[Carica Tokenizer]
    LoadSingle --> LoadTokenizer
    
    LoadTokenizer --> LimitSamples{n_samples<br/>specificato?}
    LimitSamples -->|Yes| TruncateData[Tronca Dataset]
    LimitSamples -->|No| KeepAll[Usa Tutto il Dataset]
    
    TruncateData --> GetPredictions[Genera Predizioni]
    KeepAll --> GetPredictions
    
    GetPredictions --> CalcStats[Calcola Accuracy e<br/>Distribuzione Classi]
    
    CalcStats --> ExtractIG[Estrai Integrated Gradients]
    
    ExtractIG --> CheckMode{is_ensemble?}
    
    CheckMode -->|Yes| EnsembleIG[Ensemble IG Mode]
    CheckMode -->|No| SingleIG[Single Model IG Mode]
    
    EnsembleIG --> SaveResults[Salva Risultati Raw<br/>ig_results_*.pkl]
    SingleIG --> SaveResults
    
    SaveResults --> ExtractTopWords[Estrai Top-K Words<br/>per Classe]
    
    ExtractTopWords --> AggregateActions[Aggrega Clinical Actions<br/>per Classe]
    
    AggregateActions --> SaveActions[Salva Action Results<br/>actions_*.pkl]
    
    SaveActions --> CreateViz[Crea Visualizzazioni]
    
    CreateViz --> WordViz[Word-level Plots<br/>Heatmap + Histogram]
    CreateViz --> ActionViz[Clinical Actions Plots<br/>Heatmap + Histogram]
    
    WordViz --> PrintSummary[Stampa Summary<br/>File Output]
    ActionViz --> PrintSummary
    
    PrintSummary --> End([Fine])
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style CheckEnsemble fill:#FFE4B5
    style CheckMode fill:#FFE4B5
    style LimitSamples fill:#FFE4B5
    style ExtractIG fill:#87CEEB
    style EnsembleIG fill:#DDA0DD
    style SingleIG fill:#DDA0DD
```

---

## 📊 Flusso Dettagliato: Load Data

```mermaid
flowchart TD
    LoadStart([load_test_data]) --> CheckDataset{dataset param}
    
    CheckDataset -->|'test'| LoadTest[Carica stories_test.pkl<br/>label_test.pkl]
    CheckDataset -->|'train'| LoadTrain[Carica stories_train.pkl<br/>label_train.pkl]
    CheckDataset -->|'all'| LoadBoth[Carica train + test<br/>unisci dataset]
    
    LoadTest --> LoadTrainLabels[Carica labels_train<br/>per mapping]
    LoadTrain --> LoadTrainLabels
    LoadBoth --> CreateMapping[Crea label2id<br/>da labels_train]
    
    LoadTrainLabels --> CreateMapping
    
    CreateMapping --> ConvertLabels[Converti string labels<br/>a integer IDs]
    
    ConvertLabels --> CalcDist[Calcola Class Distribution]
    
    CalcDist --> CheckImbalance{Class 0 < Class 1?}
    CheckImbalance -->|Yes| WarnImbalance[⚠️ Warning:<br/>possibile label mismatch]
    CheckImbalance -->|No| ReturnData[Return texts, true_labels,<br/>label2id]
    
    WarnImbalance --> ReturnData
    
    style LoadStart fill:#90EE90
    style CheckDataset fill:#FFE4B5
    style WarnImbalance fill:#FF6B6B
    style ReturnData fill:#87CEEB
```

---

## 🤖 Flusso Dettagliato: Load Model

```mermaid
flowchart TD
    ModelStart([load_trained_model]) --> PrintInfo[Stampa Model Info]
    
    PrintInfo --> CheckEnsemble{use_ensemble?}
    
    CheckEnsemble -->|Yes| LoadAllFolds[load_model_for_eval<br/>mode: ensemble]
    CheckEnsemble -->|No| LoadBestFold[load_model_for_eval<br/>mode: best fold]
    
    LoadAllFolds --> CreateEnsemble[Crea EnsembleModel<br/>con K fold models]
    LoadBestFold --> LoadSingleModel[Carica LongFormer<br/>best fold weights]
    
    CreateEnsemble --> LoadTok[Carica AutoTokenizer<br/>da HF model ID]
    LoadSingleModel --> LoadTok
    
    LoadTok --> PrintSuccess[✅ Stampa Success<br/>+ num folds/type]
    
    PrintSuccess --> ReturnModel[Return model,<br/>tokenizer, is_ensemble]
    
    style ModelStart fill:#90EE90
    style CheckEnsemble fill:#FFE4B5
    style CreateEnsemble fill:#DDA0DD
    style LoadSingleModel fill:#DDA0DD
    style ReturnModel fill:#87CEEB
```

---

## 🔮 Flusso Dettagliato: Get Predictions

```mermaid
flowchart TD
    PredStart([get_model_predictions]) --> CheckEnsemble{is_ensemble?}
    
    CheckEnsemble -->|Yes| EnsembleLoop[Loop: batch di testi]
    CheckEnsemble -->|No| SingleLoop[Loop: batch di testi]
    
    EnsembleLoop --> TokenizeE[Tokenize batch]
    SingleLoop --> TokenizeS[Tokenize batch]
    
    TokenizeE --> ToDeviceE[Move to device]
    TokenizeS --> ToDeviceS[Move to device]
    
    ToDeviceE --> EnsemblePredict[model.predict<br/>media K fold predictions]
    ToDeviceS --> ModelEval[model.eval<br/>no_grad context]
    
    EnsemblePredict --> ArgmaxE[Argmax per predicted class]
    ModelEval --> ForwardSingle[Forward pass<br/>single model]
    
    ForwardSingle --> SoftmaxS[Softmax per probabilità]
    SoftmaxS --> ArgmaxS[Argmax per predicted class]
    
    ArgmaxE --> CollectE[Collect probs + preds]
    ArgmaxS --> CollectS[Collect probs + preds]
    
    CollectE --> CheckMore{Altri batch?}
    CollectS --> CheckMore
    
    CheckMore -->|Yes| EnsembleLoop
    CheckMore -->|Yes| SingleLoop
    CheckMore -->|No| ReturnPreds[Return predicted_labels,<br/>predicted_probs]
    
    style PredStart fill:#90EE90
    style CheckEnsemble fill:#FFE4B5
    style EnsemblePredict fill:#DDA0DD
    style ForwardSingle fill:#DDA0DD
    style ReturnPreds fill:#87CEEB
```

---

## 🔍 Flusso Dettagliato: Ensemble IG Extraction

```mermaid
flowchart TD
    EnsStart([Ensemble IG Mode]) --> CreateTemp[Crea temp_explainer<br/>con primo fold model]
    
    CreateTemp --> InitStats[Inizializza adaptive_stats]
    
    InitStats --> LoopSamples[Loop: ogni sample]
    
    LoopSamples --> Tokenize[Tokenize text]
    
    Tokenize --> DefineCompute[Define compute_fn:<br/>model.compute_ensemble_attributions]
    
    DefineCompute --> CheckAdaptive{adaptive_steps?}
    
    CheckAdaptive -->|Yes| TrackStart[Track: started_1000++]
    CheckAdaptive -->|No| FixedSteps[Usa n_steps fisso]
    
    TrackStart --> Adaptive[compute_attributions_adaptive]
    
    Adaptive --> Try1000[Prova 1500 steps<br/>verbose=False]
    
    Try1000 --> CheckError{rel_error > 0.05?}
    
    CheckError -->|Yes| Upgrade[Upgrade a 5500 steps<br/>verbose=True]
    CheckError -->|No| KeepResult[Usa risultato 1500 steps]
    
    Upgrade --> TrackUpgrade[Track: upgraded_max++]
    TrackUpgrade --> ExtractAttr[Estrai ensemble_attributions]
    
    KeepResult --> ExtractAttr
    FixedSteps --> ExtractAttr
    
    ExtractAttr --> AggregateWords[Aggrega subwords<br/>a word attributions]
    
    AggregateWords --> StoreResult[Store in results list:<br/>text, tokens, attributions,<br/>labels, probs, diagnostics]
    
    StoreResult --> CheckMoreSamples{Altri samples?}
    
    CheckMoreSamples -->|Yes| LoopSamples
    CheckMoreSamples -->|No| ReportStats[Report Adaptive Stats]
    
    ReportStats --> AnalyzeConvergence[Analizza Convergenza Finale]
    
    AnalyzeConvergence --> CheckCritical{Campioni con<br/>errori >100%?}
    
    CheckCritical -->|Yes| WarnCritical[🔥 Warning:<br/>errori algoritmici]
    CheckCritical -->|No| CheckUnstable{Campioni con<br/>instabilità numerica?}
    
    WarnCritical --> CheckUnstable
    
    CheckUnstable -->|Yes| InfoUnstable[💤 Info:<br/>segnale troppo debole]
    CheckUnstable -->|No| ReturnResults[Return results]
    
    InfoUnstable --> ReturnResults
    
    style EnsStart fill:#90EE90
    style CheckAdaptive fill:#FFE4B5
    style CheckError fill:#FFE4B5
    style WarnCritical fill:#FF6B6B
    style InfoUnstable fill:#FFD700
    style ReturnResults fill:#87CEEB
```

---

## 🎯 Flusso Dettagliato: Single Model IG Extraction

```mermaid
flowchart TD
    SingleStart([Single Model IG Mode]) --> CreateExplainer[Crea IntegratedGradientsExplainer]
    
    CreateExplainer --> CheckAdaptive{adaptive_steps?}
    
    CheckAdaptive -->|Yes| AdaptivePath[Strategia Adattiva]
    CheckAdaptive -->|No| FixedPath[Strategia Fissa]
    
    AdaptivePath --> InitStats[Inizializza adaptive_stats]
    
    InitStats --> LoopSamples[Loop: ogni sample]
    
    LoopSamples --> Tokenize[Tokenize text]
    
    Tokenize --> GetEmbeds[Get input embeddings<br/>+ baseline zeros]
    
    GetEmbeds --> DefineForward[Define forward_func:<br/>longformer + output_layer]
    
    DefineForward --> DefineCompute[Define compute_fn:<br/>compute_ig_with_completeness_check]
    
    DefineCompute --> AdaptiveCall[compute_attributions_adaptive]
    
    AdaptiveCall --> Try1000[Prova 1000 steps]
    
    Try1000 --> CheckError{rel_error > 0.05?}
    
    CheckError -->|Yes| Upgrade[Upgrade a 5500 steps]
    CheckError -->|No| Keep1000[Usa risultato 1500 steps]
    
    Upgrade --> TrackUpgrade[Track: upgraded_max++]
    Keep1000 --> SumEmbed[Sum attributions<br/>across embedding dim]
    TrackUpgrade --> SumEmbed
    
    SumEmbed --> AggregateWords[Aggrega subwords<br/>a word attributions]
    
    AggregateWords --> StoreResult[Store in results list]
    
    StoreResult --> CheckMoreSamples{Altri samples?}
    
    CheckMoreSamples -->|Yes| LoopSamples
    CheckMoreSamples -->|No| ReportStats[Report Adaptive Stats]
    
    ReportStats --> ReturnResults[Return results]
    
    FixedPath --> FixedCall[explainer.explain_batch<br/>con n_steps fisso]
    
    FixedCall --> ReturnResults
    
    style SingleStart fill:#90EE90
    style CheckAdaptive fill:#FFE4B5
    style CheckError fill:#FFE4B5
    style ReturnResults fill:#87CEEB
```

---

## 📈 Flusso Dettagliato: Visualizations

```mermaid
flowchart TD
    VizStart([Crea Visualizzazioni]) --> ExtractTopWords[explainer.extract_top_words<br/>top_k per classe]
    
    ExtractTopWords --> AggregateActions[ClinicalActionAggregator<br/>aggregate_across_dataset]
    
    AggregateActions --> GetTopActions[get_top_actions<br/>sort by mean_score]
    
    GetTopActions --> SaveActions[Salva actions_*.pkl]
    
    SaveActions --> WordViz[Word-level Visualizations]
    
    WordViz --> WordHeatmap[plot_attention_heatmap<br/>heatmap_words_*.png]
    WordViz --> WordHistogram[plot_class_comparison<br/>histogram_words_*.png]
    
    WordHeatmap --> ActionViz[Clinical Actions Viz]
    WordHistogram --> ActionViz
    
    ActionViz --> ActionHeatmap[plot_clinical_actions_heatmap<br/>heatmap_actions_*.png]
    ActionViz --> ActionHistogram[plot_clinical_actions_comparison<br/>histogram_actions_*.png]
    
    ActionHeatmap --> Complete[Visualizzazioni Complete]
    ActionHistogram --> Complete
    
    style VizStart fill:#90EE90
    style Complete fill:#87CEEB
```

---

## 🔧 Helper Function: Adaptive Strategy

```mermaid
flowchart TD
    AdaptStart([compute_attributions_adaptive]) --> CheckMode{use_adaptive?}
    
    CheckMode -->|Yes| SetInitial[n_steps_initial = 1500<br/>n_steps_max = 5500<br/>tolerance = 0.05]
    CheckMode -->|No| UseFixed[n_steps = fixed_n_steps]
    
    SetInitial --> ComputeInitial[compute_fn<br/>n_steps=1500, verbose=False]
    
    ComputeInitial --> ExtractError[Estrai rel_error<br/>da diagnostics]
    
    ExtractError --> CheckConverge{rel_error > 0.05?}
    
    CheckConverge -->|Yes| ComputeMax[compute_fn<br/>n_steps=5500, verbose=True]
    CheckConverge -->|No| MarkKeep[upgraded = False]
    
    ComputeMax --> MarkUpgrade[upgraded = True]
    
    MarkUpgrade --> ReturnUpgraded[Return result,<br/>upgraded=True]
    MarkKeep --> ReturnKept[Return result,<br/>upgraded=False]
    
    UseFixed --> ComputeFixed[compute_fn<br/>n_steps=fixed, verbose=True]
    
    ComputeFixed --> ReturnFixed[Return result,<br/>upgraded=False]
    
    style AdaptStart fill:#90EE90
    style CheckMode fill:#FFE4B5
    style CheckConverge fill:#FFE4B5
    style ReturnUpgraded fill:#FFB6C1
    style ReturnKept fill:#87CEEB
    style ReturnFixed fill:#87CEEB
```

---

## 📝 Argomenti CLI Supportati

| Argomento | Default | Descrizione |
|-----------|---------|-------------|
| `--model` | `bert-base-uncased` | Nome modello HuggingFace |
| `--format` | `narrativo` | Formato storie (`narrativo`/`bullet`/`clinical`) |
| `--dataset` | `test` | Dataset da analizzare (`test`/`train`/`all`) |
| `--n_samples` | `None` | Limita numero samples (None = tutti) |
| `--top_k` | `20` | Top-K words/actions per visualizzazioni |
| `--device` | `cuda` (auto) | Device per computation |
| `--internal_batch_size` | `32` | Batch size per IG interpolation |
| `--n_steps` | `1500` | Steps per IG (modalità fissa) |
| `--adaptive_steps` | `False` | Abilita strategia adattiva (1500→5500) |
| `--use_ensemble` | `False` | Usa K-Fold ensemble invece di best fold |

---

## 🎯 Output Files Generati

### 1. Raw Results
- **File**: `ig_results_{format}_{model}_{mode}_{timestamp}.pkl`
- **Contenuto**: Lista di dizionari con:
  - `text`: Testo originale
  - `tokens`: Token list
  - `token_attributions`: Attribution scores per token
  - `word_attributions`: Attribution aggregati per parola
  - `true_label`, `predicted_label`, `predicted_prob`
  - `diagnostics` (solo ensemble): info convergenza per fold

### 2. Clinical Actions
- **File**: `actions_{format}_{model}_{mode}_{timestamp}.pkl`
- **Contenuto**: Dizionari con aggregazioni:
  - `class_0`: {action_text: {mean_score, count, total_score}}
  - `class_1`: {action_text: {mean_score, count, total_score}}

### 3. Visualizzazioni Word-level
- `heatmap_words_{format}_{model}_{mode}_{timestamp}.png`
- `histogram_words_{format}_{model}_{mode}_{timestamp}.png`

### 4. Visualizzazioni Clinical Actions
- `heatmap_actions_{format}_{model}_{mode}_{timestamp}.png` ⭐
- `histogram_actions_{format}_{model}_{mode}_{timestamp}.png` ⭐

*(⭐ = principali per interpretabilità clinica)*

---

## 💡 Note Implementative

### Strategia Adattiva
1. **Obiettivo**: Risparmiare tempo di computazione (~40-70% a seconda del dataset)
2. **Meccanismo**:
   - Inizia con 1500 steps (silenzioso)
   - Se `rel_error > 0.05` → ricalcola con 5500 steps (verbose)
   - Altrimenti usa risultato 1500 steps
3. **Trade-off**: Overhead 1500 steps vs risparmio su campioni convergenti
4. **Costanti configurabili** (in `extract_explainability.py`):
   - `ADAPTIVE_IG_STEPS_INITIAL = 1500`
   - `ADAPTIVE_IG_STEPS_MAX = 5500`
   - `ADAPTIVE_IG_TOLERANCE = 0.05`

### Ensemble vs Single
- **Ensemble**: Media attributions su K fold models (più robusto)
- **Single**: Usa solo best fold (più veloce)
- **Diagnostics**: Solo ensemble traccia convergenza per-fold

### Gestione Errori
- **Instabilità Numerica** (`f(x)≈f(baseline)`): Segnale troppo debole, IG non affidabile
- **Errori Algoritmici** (`rel_error > 100%`): Non-convergenza vera, serve più steps

---

## 🚀 Esempi d'Uso

```bash
# Single model, dataset test, strategia fissa
python src/explainability/extract_explainability.py \
  --model bert-base-uncased \
  --format narrativo \
  --n_steps 1500

# Ensemble, dataset completo, strategia adattiva (1500→5500)
python src/explainability/extract_explainability.py \
  --model clinical-bert \
  --format narrativo \
  --dataset all \
  --use_ensemble \
  --adaptive_steps

# Analisi rapida su 50 samples
python src/explainability/extract_explainability.py \
  --model pubmedbert-base \
  --format bullet \
  --n_samples 50 \
  --top_k 10
```

---

## 📚 Dipendenze Chiave

| Modulo | Funzione Principale |
|--------|---------------------|
| `IntegratedGradientsExplainer` | Calcolo IG per single model |
| `EnsembleModel.compute_ensemble_attributions` | Calcolo IG per ensemble |
| `ClinicalActionAggregator` | Estrazione azioni cliniche da testo |
| `compute_ig_with_completeness_check` | IG con diagnostics convergenza |
| `load_model_for_eval` | Caricamento modelli (single/ensemble) |
| Visualization functions | Heatmaps e histograms comparativi |

---

*Generato automaticamente da analisi del codice sorgente*
