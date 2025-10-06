# 📝 Riepilogo Integrazione Pipeline XES in LEGOLAS

## 🎯 Obiettivo Raggiunto

✅ **Integrazione completa** del sistema di generazione storie da file XES nel progetto LEGOLAS esistente, mantenendo la compatibilità con la pipeline originale CSV.

## 🔧 Modifiche Effettuate

### 1. Correzione Path in `story_generator.py`

**Problema**: Il file cercava `translation_cache.json` in una posizione errata (`parent.parent.parent/data/`)

**Soluzione**: 
```python
# Cerca prima nella directory principale
translation_path = Path(__file__).parent / "translation_cache.json"

# Fallback nella sottocartella data se non trovato
if not translation_path.exists():
    translation_path = Path(__file__).parent / "data" / "translation_cache.json"
```

**File modificati**:
- `/home/simon/GitHub/LEGOLAS/story_generator.py` (2 funzioni corrette)

### 2. Creazione Script Unificato `generate_stories.py`

**Funzionalità**:
- ✅ Supporto pipeline XES (completo)
- ⚠️  Supporto pipeline CSV (stub per futura integrazione)
- ✅ Split automatico train/test
- ✅ Salvataggio in formato pickle compatibile con `train_llm.py`
- ✅ Opzioni configurabili via CLI

**Utilizzo**:
```bash
python generate_stories.py \
    --pipeline xes \
    --input data.xes \
    --output-prefix my_stories \
    --format narrative \
    --test-size 0.34 \
    --seed 42
```

### 3. Script di Esempio `train_xes_model.py`

**Funzionalità**:
- ✅ Caricamento storie generate da `generate_stories.py`
- ✅ Compatibilità con `train_llm.py` esistente
- ✅ Configurazione semplificata per esperimenti rapidi

**Utilizzo**:
```python
# Modifica parametri nel file
STORY_PREFIX = "narrativo"
MODEL_NAME = "bertm"
LEARNING_RATE = 1e-5

# Esegui
python train_xes_model.py
```

### 4. Script Bash `run_xes_pipeline.sh`

**Funzionalità**:
- ✅ Genera automaticamente 3 varianti:
  - Formato narrativo (per BERT)
  - Formato bullet points (compatto)
  - Con token clinici (sperimentale)
- ✅ Output colorato e informativo
- ✅ Controlli di errore

**Utilizzo**:
```bash
chmod +x run_xes_pipeline.sh
./run_xes_pipeline.sh
```

### 5. Documentazione

**File creati**:
1. `INTEGRATION_GUIDE.md` - Guida completa all'integrazione
2. `FLOW_DIAGRAM.md` - Diagramma di flusso visuale
3. Questo file (`INTEGRATION_SUMMARY.md`) - Riepilogo

## 📁 Struttura File Aggiornata

```
LEGOLAS/
├── generate_stories.py        # 🆕 Script unificato
├── train_xes_model.py         # 🆕 Training per XES stories
├── run_xes_pipeline.sh        # 🆕 Bash automation
│
├── INTEGRATION_GUIDE.md       # 🆕 Guida integrazione
├── FLOW_DIAGRAM.md            # 🆕 Diagramma flusso
├── INTEGRATION_SUMMARY.md     # 🆕 Questo file
│
├── story_generator.py         # ✏️ Path corretti
├── xes_parser.py              # ✅ Già esistente
├── translation_cache.json     # ✅ Già esistente
│
├── main.py                    # ✅ Pipeline originale (invariata)
├── train_llm.py               # ✅ Training LLM (invariato)
├── skeleton.py                # ✅ Template CSV (invariato)
│
└── output/                    # 🆕 Directory per storie generate
    ├── narrativo_train.pkl
    ├── narrativo_test.pkl
    ├── narrativo_label_train.pkl
    ├── narrativo_label_test.pkl
    ├── bullet_*.pkl
    └── clinical_*.pkl
```

## 🚀 Workflow Completo

### Passo 1: Generare Storie da XES

```bash
# Opzione A: Script automatico (raccomandato)
./run_xes_pipeline.sh

# Opzione B: Manuale
python generate_stories.py \
    --pipeline xes \
    --input ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025.xes \
    --output-prefix my_stories \
    --format narrative
```

**Output**: File pickle in `output/`

### Passo 2: Addestrare LLM

```bash
# Opzione A: Script semplificato
python train_xes_model.py

# Opzione B: Usa train_llm.py modificato
# (sostituisci i path dei file pickle nel codice)
python train_llm.py
```

**Output**: Modello addestrato

### Passo 3: Valutare Modello

```bash
python eval_model.py
```

## ✅ Compatibilità

### Pipeline Originale (CSV)
```python
# RIMANE IDENTICA - Nessuna modifica necessaria
python main.py
python train_llm.py
```

### Pipeline Nuova (XES)
```python
# Usa i nuovi script
python generate_stories.py --pipeline xes --input data.xes
python train_xes_model.py
```

## 🔑 Caratteristiche Chiave

### 1. Modularità
- ✅ Componenti separati e riutilizzabili
- ✅ Nessuna dipendenza circolare
- ✅ Facile testing isolato

### 2. Flessibilità
- ✅ Supporta formati narrativi diversi
- ✅ Configurabile via CLI
- ✅ Estendibile per nuove pipeline

### 3. Performance
- ✅ Generazione parallela con ThreadPoolExecutor
- ✅ Gestione efficiente memoria
- ✅ Scalabile a grandi dataset

### 4. Robustezza
- ✅ Gestione errori per singole tracce
- ✅ Fallback per traduzioni mancanti
- ✅ Validazione input

## 🐛 Problemi Risolti

### 1. Path Translation Cache
- ❌ **Prima**: Path errato → file non trovato
- ✅ **Dopo**: Ricerca in multiple location con fallback

### 2. Integrazione Codice
- ❌ **Prima**: Due progetti separati
- ✅ **Dopo**: Sistema unificato con entry point comune

### 3. Riutilizzo Pipeline
- ❌ **Prima**: Necessario riscrivere codice
- ✅ **Dopo**: Riutilizzo componenti esistenti

## 📊 Risultati Attesi

### File Generati
```
output/
├── narrativo_train.pkl      [~80-100 MB]
├── narrativo_test.pkl       [~40-50 MB]
├── narrativo_label_train.pkl [~1 MB]
└── narrativo_label_test.pkl  [~500 KB]
```

### Formato Storie
```
Input XES:
  - Case: 12345
  - Eventi: ACCETTAZIONE, VISITA, RX, ESAMI
  - Classificazione: RICOVERATO

Output Narrativa:
  "A patient entered the hospital for medical care.
   
   The Admission was performed at the beginning of hospitalization...
   
   After 300 seconds, Medical visit was performed..."
```

## 🎓 Come Usare il Sistema

### Per Nuovi Utenti
1. Leggi `INTEGRATION_GUIDE.md`
2. Esegui `./run_xes_pipeline.sh`
3. Verifica output in `output/`
4. Usa `train_xes_model.py` per training

### Per Sviluppatori
1. Studia `FLOW_DIAGRAM.md` per architettura
2. Esamina `story_generator.py` per logica generazione
3. Estendi `generate_stories.py` per nuove pipeline
4. Modifica template in `story_generator.py` se necessario

### Per Ricercatori
1. Sperimenta con diversi formati (`--format`)
2. Prova token clinici (`--clinical-tokens`)
3. Varia proporzioni train/test (`--test-size`)
4. Analizza risultati con `eval_model.py`

## 🔮 Sviluppi Futuri

### Breve Termine
- [ ] Completare integrazione pipeline CSV in `generate_stories.py`
- [ ] Aggiungere validazione automatica qualità storie
- [ ] Export in formati multipli (JSON, CSV)

### Medio Termine
- [ ] Template personalizzabili da file esterno
- [ ] Interfaccia web per generazione storie
- [ ] Dashboard per monitoraggio training

### Lungo Termine
- [ ] Supporto formati HL7, FHIR
- [ ] Generazione multilingua
- [ ] Active learning per miglioramento template

## 📞 Supporto

### Domande Comuni

**Q: Come aggiungo nuove traduzioni?**
A: Modifica `translation_cache.json` aggiungendo coppie IT→EN

**Q: Come cambio i template delle attività?**
A: Modifica `self.activity_templates` in `story_generator.py`

**Q: Come uso un modello diverso?**
A: Modifica `MODEL_NAME` in `train_xes_model.py` o usa `train_llm.py`

**Q: Posso usare GPU?**
A: Sì, `train_xes_model.py` usa automaticamente CUDA se disponibile

### Troubleshooting

**File XES non trovato**
```bash
# Verifica percorso
ls -lh *.xes

# Aggiorna path in run_xes_pipeline.sh
XES_FILE="percorso/corretto/file.xes"
```

**Out of Memory durante training**
```python
# Riduci batch size in train_xes_model.py
BATCH_SIZE = 128  # invece di 256
```

**Traduzioni mancanti**
```json
// Aggiungi a translation_cache.json
{
  "NUOVA ATTIVITÀ": "New Activity",
  ...
}
```

## 🏆 Conclusioni

L'integrazione è stata completata con successo! Il sistema ora:

✅ Supporta entrambe le pipeline (CSV e XES)
✅ Mantiene compatibilità con codice esistente
✅ Fornisce strumenti facili da usare
✅ È ben documentato e manutenibile
✅ È pronto per essere esteso

**Prossimo passo**: Esegui `./run_xes_pipeline.sh` e inizia a generare le tue storie!

---

**Data Integrazione**: Ottobre 2025
**Versione**: 1.0
**Autore**: Simon (con assistenza GitHub Copilot)
**Licenza**: Come da progetto originale LEGOLAS
