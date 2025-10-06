# Guida all'Integrazione: Pipeline XES in LEGOLAS

## 📋 Panoramica

Questo progetto ora supporta **due pipeline** per la generazione di storie narrative:

1. **Pipeline Originale (CSV)**: Usa `skeleton.py` come template per dati CSV strutturati
2. **Pipeline Nuova (XES)**: Usa `story_generator.py` per dati XES con lunghezza variabile

## 🏗️ Architettura del Sistema

```
LEGOLAS/
├── generate_stories.py        # Script unificato (NUOVO)
├── main.py                     # Pipeline originale CSV
├── train_llm.py               # Training degli LLM
│
├── xes_parser.py              # Parser per file XES (INTEGRATO)
├── story_generator.py         # Generatore storie da XES (INTEGRATO)
├── skeleton.py                # Template per pipeline CSV originale
│
├── translation_cache.json     # Dizionario IT→EN (SHARED)
├── clinical_token_mapper.py   # Mapper token clinici
│
└── utils/
    ├── types.py               # Tipi condivisi
    ├── config.py
    └── ml_utils.py
```

## 🚀 Quick Start

### 1. Generare Storie da File XES

```bash
# Formato narrativo (raccomandato per BERT)
python generate_stories.py \
    --pipeline xes \
    --input ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025.xes \
    --output-prefix my_stories \
    --format narrative

# Formato bullet points
python generate_stories.py \
    --pipeline xes \
    --input data.xes \
    --format bullet_points
```

### 2. Addestrare LLM sulle Storie Generate

```bash
# Modifica train_llm.py per caricare i file generati:
# - output/my_stories_train.pkl
# - output/my_stories_test.pkl
# - output/my_stories_label_train.pkl
# - output/my_stories_label_test.pkl

python train_llm.py
```

## 📁 Struttura dei File Generati

Lo script `generate_stories.py` crea i seguenti file nella cartella `output/`:

```
output/
├── stories_train.pkl         # Storie di training
├── stories_test.pkl          # Storie di test
├── stories_label_train.pkl   # Label di training
└── stories_label_test.pkl    # Label di test
```

Questi file sono compatibili con `train_llm.py` e possono essere usati direttamente per l'addestramento.

## 🔧 Opzioni Avanzate

### Token Clinici Atomici (SPERIMENTALE)

```bash
python generate_stories.py \
    --pipeline xes \
    --input data.xes \
    --clinical-tokens
```

Questa opzione abilita la sostituzione di procedure mediche complesse con token atomici per migliorare l'analisi XAI.

### Modificare la Proporzione Train/Test

```bash
python generate_stories.py \
    --pipeline xes \
    --input data.xes \
    --test-size 0.2  # 20% test, 80% training
```

## 🔄 Come Funziona l'Integrazione

### Pipeline XES (Nuova)

1. **XESParser** (`xes_parser.py`):
   - Carica il file XES
   - Estrae le tracce dei pazienti con eventi e timestamp
   - Estrae metadati (età, genere, classificazione)

2. **StoryGenerator** (`story_generator.py`):
   - Traduce attività da italiano a inglese usando `translation_cache.json`
   - Applica template per ogni tipo di attività clinica
   - Genera narrativa in formato discorsivo o bullet points
   - (Opzionale) Sostituisce procedure con token clinici atomici

3. **Generazione Parallela**:
   - Usa ThreadPoolExecutor per velocizzare la generazione
   - Gestisce errori per singola traccia senza interrompere il batch

### Pipeline CSV (Originale)

La pipeline originale usando `skeleton.py` rimane disponibile in `main.py` per compatibilità con il dataset MIMIC-IV originale.

## 🗂️ File Chiave

### `translation_cache.json`
Dizionario condiviso per la traduzione italiano→inglese delle attività mediche:
```json
{
  "VISITA CARDIOLOGICA": "Cardiological visit",
  "RX TORACE": "Chest X-ray",
  "ESAMI DI LABORATORIO": "Laboratory tests"
}
```

**Posizione**: Directory principale del progetto (non in `data/`)

### `story_generator.py`
Corregge automaticamente il path per trovare `translation_cache.json`:
```python
# Cerca prima nella directory principale
translation_path = Path(__file__).parent / "translation_cache.json"

# Fallback nella sottocartella data
if not translation_path.exists():
    translation_path = Path(__file__).parent / "data" / "translation_cache.json"
```

## 📊 Esempio di Output

### Input: Traccia Paziente XES
```
Case ID: 12345
Eventi:
  - ACCETTAZIONE (t=0s)
  - VISITA MEDICA (t=300s)
  - RX TORACE (t=600s)
  - ESAMI DI LABORATORIO (t=900s)
Classificazione: RICOVERATO (1)
```

### Output: Storia Narrativa
```
A patient entered the hospital for medical care.

The Admission was performed at the beginning of hospitalization, after 0 seconds.

After 300 seconds, Medical visit was performed.

After 600 seconds, Chest X-ray was performed.

After 900 seconds, Laboratory tests were performed.
```

## 🎯 Vantaggi dell'Integrazione

1. **Flessibilità**: Supporta sia dati strutturati (CSV) che non strutturati (XES)
2. **Riutilizzo**: Mantiene la pipeline originale funzionante
3. **Modularità**: Componenti separati e riutilizzabili
4. **Prestazioni**: Generazione parallela per grandi dataset
5. **Estensibilità**: Facile aggiungere nuove pipeline in futuro

## 🔮 Prossimi Passi

1. **Completare l'integrazione CSV**: Unificare completamente `main.py` in `generate_stories.py`
2. **Template personalizzabili**: Permettere definizione custom di template
3. **Validazione automatica**: Controlli di qualità sulle storie generate
4. **Export formati multipli**: JSON, CSV, oltre a pickle

## 🐛 Troubleshooting

### Errore: "Translation cache not found"
- Verifica che `translation_cache.json` sia nella directory principale
- Oppure crea una cartella `data/` e spostalo lì

### Errore: "Required columns missing from XES"
- Verifica che il file XES contenga:
  - `case:concept:name` (ID caso)
  - `concept:name` (Nome attività)
  - `time:timestamp` (Timestamp)

### Performance lente
- Usa meno worker: modifica `max_workers` in `StoryGenerator.generate_batch_stories()`
- Disabilita token clinici se non necessari

## 📚 Riferimenti

- **PM4Py**: Libreria per process mining ([docs](https://pm4py.fit.fraunhofer.de/))
- **XES Standard**: [IEEE XES Standard](http://www.xes-standard.org/)
- **BERTing**: Vocabolario medico per terminologia inglese standardizzata

## 👥 Contributi

Per domande o problemi, apri una issue su GitHub o contatta il team di sviluppo.

---

**Nota**: Questa integrazione mantiene piena compatibilità con il codice esistente mentre aggiunge nuove funzionalità per gestire dati XES con lunghezza variabile.
