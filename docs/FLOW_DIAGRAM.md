# 🔄 Diagramma di Flusso - Pipeline Integrata LEGOLAS

## Pipeline Completa

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LEGOLAS - Pipeline Integrata                     │
└─────────────────────────────────────────────────────────────────────────┘

                           ┌─────────────────┐
                           │  INPUT DATI     │
                           └────────┬────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
             ┌──────▼──────┐                 ┌─────▼──────┐
             │  FILE CSV   │                 │  FILE XES  │
             │  (Originale)│                 │  (Nuovo)   │
             └──────┬──────┘                 └─────┬──────┘
                    │                               │
             ┌──────▼──────┐                 ┌─────▼──────┐
             │  main.py    │                 │xes_parser.py│
             │ skeleton.py │                 └─────┬──────┘
             └──────┬──────┘                       │
                    │                               │
                    │                        ┌──────▼──────────┐
                    │                        │ PatientTrace[]  │
                    │                        │ (structured)    │
                    │                        └──────┬──────────┘
                    │                               │
                    │                        ┌──────▼──────────┐
                    │                        │story_generator.py│
                    │                        │                 │
                    │                        │ Uses:           │
                    │                        │ - translation_  │
                    │                        │   cache.json    │
                    │                        │ - templates     │
                    │                        └──────┬──────────┘
                    │                               │
                    └───────────┬───────────────────┘
                                │
                        ┌───────▼────────┐
                        │ NARRATIVE      │
                        │ STORIES        │
                        │ (English text) │
                        └───────┬────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
            ┌───────▼────────┐      ┌──────▼──────┐
            │ *_train.pkl    │      │ *_test.pkl  │
            │ *_label_       │      │ *_label_    │
            │   train.pkl    │      │   test.pkl  │
            └───────┬────────┘      └──────┬──────┘
                    │                      │
                    └───────────┬──────────┘
                                │
                        ┌───────▼────────┐
                        │  train_llm.py  │
                        │  (o train_xes_ │
                        │   model.py)    │
                        └───────┬────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
            ┌───────▼────────┐      ┌──────▼──────┐
            │  TRAINED LLM   │      │ EVALUATION  │
            │  (BERT)        │      │             │
            └────────────────┘      └─────────────┘
```

## Componenti Chiave

### 1. Input Layer
```
┌─────────────┐
│  CSV Data   │  → Dati strutturati MIMIC-IV (originale)
└─────────────┘     Campi fissi, lunghezza nota

┌─────────────┐
│  XES Data   │  → Dati process mining (nuovo)
└─────────────┘     Lunghezza variabile, eventi timestampati
```

### 2. Parsing Layer
```
CSV → main.py (usa skeleton.py)
      ├─ Template Jinja2
      └─ Dati strutturati

XES → xes_parser.py
      ├─ PM4Py parser
      ├─ Estrazione tracce
      └─ Metadati (età, genere, classificazione)
```

### 3. Generation Layer
```
story_generator.py
├─ Template-based generation
├─ Translation IT→EN (translation_cache.json)
├─ Activity categorization
├─ Timestamp formatting
└─ Optional: Clinical token mapping
```

### 4. Storage Layer
```
Pickle Files
├─ {prefix}_train.pkl          → Training stories
├─ {prefix}_test.pkl           → Test stories
├─ {prefix}_label_train.pkl    → Training labels
└─ {prefix}_label_test.pkl     → Test labels
```

### 5. Training Layer
```
train_llm.py / train_xes_model.py
├─ Load pickle files
├─ Tokenization
├─ Model setup (BERT/GPT2/RoBERTa)
├─ Training loop
└─ Model saving
```

## Flusso di Esecuzione

### Scenario A: Pipeline CSV (Originale)
```bash
1. python main.py
   └─ Output: mimic_*.pkl files

2. python train_llm.py
   └─ Uses: mimic_*.pkl
   └─ Output: trained model
```

### Scenario B: Pipeline XES (Nuovo)
```bash
1. python generate_stories.py --pipeline xes --input data.xes
   └─ Output: output/stories_*.pkl files

2. python train_xes_model.py
   └─ Uses: output/stories_*.pkl
   └─ Output: trained model

# Alternative: usando bash script
./run_xes_pipeline.sh
   └─ Genera tutti i formati (narrativo, bullet, clinical)
```

## File Condivisi

```
translation_cache.json
├─ Used by: story_generator.py
├─ Used by: clinical_token_mapper.py
└─ Format: {"ITALIANO": "English Translation"}

utils/types.py
├─ PatientTrace
├─ Event
├─ PatientStory
└─ ClassificationTarget
```

## Vantaggi dell'Integrazione

```
┌────────────────────────────────────────────────┐
│ ✅ Mantiene compatibilità con pipeline originale│
│ ✅ Aggiunge supporto per dati XES variabili    │
│ ✅ Riutilizza componenti comuni                │
│ ✅ Generazione parallela per performance       │
│ ✅ Facile estendibilità per nuove pipeline     │
└────────────────────────────────────────────────┘
```

## Estensibilità Futura

```
generate_stories.py (extensible)
├─ --pipeline xes    ✅ Implementato
├─ --pipeline csv    ⚠️  In sviluppo
├─ --pipeline json   🔮 Futuro
├─ --pipeline hl7    🔮 Futuro
└─ --pipeline fhir   🔮 Futuro
```

## Note Tecniche

### Thread Safety
- `story_generator.py` usa `ThreadPoolExecutor` per parallelizzazione
- Nessuna condivisione di stato mutabile tra thread
- Gestione errori per singola traccia

### Memory Management
- Processamento batch per grandi dataset
- Streaming potenziale per dataset molto grandi (TODO)

### Error Handling
- Fallback graceful per traduzioni mancanti
- Continuazione su errori singoli
- Logging dettagliato per debug

---

**Legend:**
- ✅ = Completato
- ⚠️ = In sviluppo  
- 🔮 = Pianificato
- → = Flusso dati
- ├─ = Componente/Feature
- └─ = Ultima componente/Output
