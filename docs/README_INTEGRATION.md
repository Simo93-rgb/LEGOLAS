# LEGOLAS - Integrazione Pipeline XES

## 🎯 Panoramica Rapida

Questo progetto ora supporta **due pipeline** per generare storie narrative da dati clinici:

1. **Pipeline CSV (Originale)** - Dati strutturati con template fissi (`skeleton.py`)
2. **Pipeline XES (Nuovo)** - Dati variabili da file XES con generazione dinamica (`story_generator.py`)

## 🚀 Quick Start

### Test Integrazione

```bash
# Verifica che tutto sia configurato correttamente
python test_integration.py
```

### Generare Storie da XES

```bash
# Opzione A: Script automatico (raccomandato)
./run_xes_pipeline.sh

# Opzione B: Manuale
python generate_stories.py \
    --pipeline xes \
    --input IL_TUO_FILE.xes \
    --output-prefix my_stories \
    --format narrative
```

### Addestrare LLM

```bash
# Usa le storie generate
python train_xes_model.py
```

## 📁 File Creati dall'Integrazione

```
generate_stories.py          # Script unificato per generazione
train_xes_model.py           # Training su storie XES
run_xes_pipeline.sh          # Automazione bash
test_integration.py          # Test sistema

INTEGRATION_GUIDE.md         # Guida completa
FLOW_DIAGRAM.md              # Diagramma architettura
INTEGRATION_SUMMARY.md       # Riepilogo tecnico
```

## 🔧 Componenti Integrati

### Esistenti (già nel progetto)
- `xes_parser.py` - Parser per file XES
- `story_generator.py` - Generatore storie con template
- `translation_cache.json` - Dizionario IT→EN

### Modificati
- `story_generator.py` - Path corretti per `translation_cache.json`

### Nuovi
- `generate_stories.py` - Entry point unificato
- `train_xes_model.py` - Training semplificato
- Documentazione completa

## 📚 Documentazione

| File | Descrizione |
|------|-------------|
| `INTEGRATION_GUIDE.md` | Guida completa all'uso |
| `FLOW_DIAGRAM.md` | Diagramma architettura |
| `INTEGRATION_SUMMARY.md` | Riepilogo tecnico dettagliato |
| Questo `README_INTEGRATION.md` | Panoramica rapida |

## ✅ Compatibilità

### Pipeline Originale (Invariata)
```bash
python main.py          # Funziona come prima
python train_llm.py     # Funziona come prima
```

### Pipeline Nuova
```bash
python generate_stories.py --pipeline xes --input data.xes
python train_xes_model.py
```

## 🎓 Esempi d'Uso

### Esempio 1: Pipeline Base
```bash
python generate_stories.py \
    --pipeline xes \
    --input data.xes \
    --format narrative
```

### Esempio 2: Con Token Clinici
```bash
python generate_stories.py \
    --pipeline xes \
    --input data.xes \
    --format narrative \
    --clinical-tokens
```

### Esempio 3: Training Custom
```python
# In train_xes_model.py, modifica:
STORY_PREFIX = "my_stories"
MODEL_NAME = "cbert"
BATCH_SIZE = 128

# Poi esegui:
python train_xes_model.py
```

## 🔄 Workflow Tipico

```
1. Generare storie
   └─> ./run_xes_pipeline.sh

2. Verificare output
   └─> ls -lh output/*.pkl

3. Addestrare modello
   └─> python train_xes_model.py

4. Valutare risultati
   └─> python eval_model.py
```

## 🐛 Troubleshooting

### Translation cache not found
```bash
# Verifica posizione
ls -lh translation_cache.json

# Dovrebbe essere nella directory principale
```

### File XES non trovato
```bash
# Aggiorna path in run_xes_pipeline.sh
XES_FILE="percorso/corretto/tuo_file.xes"
```

### Import errors
```bash
# Verifica dipendenze
python test_integration.py
```

## 📊 Output Atteso

```
output/
├── narrativo_train.pkl      # Storie training
├── narrativo_test.pkl       # Storie test
├── narrativo_label_train.pkl # Label training
└── narrativo_label_test.pkl  # Label test
```

## 🎯 Vantaggi Integrazione

✅ Mantiene pipeline originale funzionante  
✅ Supporta dati XES variabili  
✅ Generazione parallela veloce  
✅ Configurabile via CLI  
✅ Ben documentato  
✅ Facile da estendere  

## 🔮 Prossimi Sviluppi

- [ ] Integrazione completa pipeline CSV
- [ ] Template personalizzabili
- [ ] Export multipli formati
- [ ] Interfaccia web

## 📞 Supporto

Consulta i file di documentazione:
- `INTEGRATION_GUIDE.md` per guida completa
- `FLOW_DIAGRAM.md` per architettura
- `INTEGRATION_SUMMARY.md` per dettagli tecnici

---

**Data**: Ottobre 2025  
**Versione**: 1.0  
**Status**: ✅ Produzione
