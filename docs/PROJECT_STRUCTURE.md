# 📁 Proposta Ristrutturazione Progetto LEGOLAS

## 🎯 Struttura Proposta

```
LEGOLAS/
├── README.md                           # README principale
├── pyproject.toml                      # Dipendenze e config
├── .gitignore                          # File da ignorare
│
├── docs/                               # 📚 Documentazione
│   ├── INTEGRATION_GUIDE.md
│   ├── FLOW_DIAGRAM.md
│   ├── TRAINING_EVALUATION_GUIDE.md
│   ├── INTEGRATION_SUMMARY.md
│   ├── NEXT_STEPS.md
│   ├── INDEX.md
│   ├── INTEGRATION_COMPLETE.txt
│   ├── TRAINING_COMPLETE.txt
│   └── QUICK_START.txt
│
├── scripts/                            # 🚀 Script eseguibili
│   ├── generate_stories.py
│   ├── train_xes_model.py
│   ├── test_integration.py
│   ├── run_xes_pipeline.sh
│   └── launch_run_train_eval
│
├── src/                                # 💾 Codice sorgente
│   ├── __init__.py
│   │
│   ├── models/                         # Definizioni modelli neural network
│   │   ├── __init__.py
│   │   ├── neural_network.py
│   │   └── model_config.yaml
│   │
│   ├── data/                           # Parsing e gestione dati
│   │   ├── __init__.py
│   │   ├── xes_parser.py
│   │   ├── history_dataset.py
│   │   └── clinical_token_mapper.py
│   │
│   ├── generation/                     # Generazione storie
│   │   ├── __init__.py
│   │   ├── story_generator.py
│   │   └── skeleton.py                # Template originale
│   │
│   ├── training/                       # Training e evaluation
│   │   ├── __init__.py
│   │   ├── train_llm.py
│   │   └── eval_model.py
│   │
│   └── utils/                          # Utilities
│       ├── __init__.py
│       ├── config.py
│       ├── types.py
│       └── ml_utils.py
│
├── data/                               # 📊 Dati e cache
│   ├── translation_cache.json
│   └── raw/                            # Dati raw (XES, CSV)
│       └── .gitkeep
│
├── output/                             # 📤 Output generati
│   ├── stories/                        # Storie generate
│   │   ├── narrativo_*.pkl
│   │   ├── bullet_*.pkl
│   │   └── clinical_*.pkl
│   │
│   └── models/                         # Modelli addestrati
│       └── xes_narrativo_bertm*.pth
│
├── prediction/                         # 📈 Risultati evaluation
│   ├── xes_narrativo_bertm_report.txt
│   └── xes_narrativo_bertm_*.pkl
│
└── legacy/                             # 📜 Codice legacy (opzionale)
    ├── main.py                         # Pipeline originale CSV
    └── launch_run_*                    # Script originali
```

## 🔄 Come Ristrutturare

### Opzione 1: Ristrutturazione Completa (Raccomandato)
```bash
# Esegui script di migrazione
./scripts/restructure_project.sh
```

### Opzione 2: Manuale
```bash
# 1. Crea struttura directory
mkdir -p docs scripts src/{models,data,generation,training,utils} data/raw output/{stories,models} legacy

# 2. Sposta file documentazione
mv *.md *.txt docs/

# 3. Sposta script
mv generate_stories.py train_xes_model.py test_integration.py scripts/
mv run_xes_pipeline.sh launch_run_train_eval scripts/

# 4. Sposta codice sorgente
mv neural_network.py model_config.yaml src/models/
mv xes_parser.py history_dataset.py clinical_token_mapper.py src/data/
mv story_generator.py src/generation/
mv skeleton.py src/generation/
mv train_llm.py eval_model.py src/training/
# utils già in posizione

# 5. Sposta dati
mv translation_cache.json data/
mv *.xes data/raw/ 2>/dev/null || true

# 6. Riorganizza output
mv output/*.pkl output/stories/ 2>/dev/null || true
mv *.pth output/models/ 2>/dev/null || true

# 7. Legacy (opzionale)
mv main.py launch_run_single* launch_run_eval* legacy/ 2>/dev/null || true
```

## 📝 Modifiche Necessarie agli Import

Dopo la ristrutturazione, dovrai aggiornare gli import:

### Prima:
```python
from xes_parser import XESParser
from story_generator import StoryGenerator
from neural_network import LongFormerMultiClassificationHeads
```

### Dopo:
```python
from src.data.xes_parser import XESParser
from src.generation.story_generator import StoryGenerator
from src.models.neural_network import LongFormerMultiClassificationHeads
```

## 🎯 Vantaggi

1. **Organizzazione Chiara**: Ogni componente ha il suo posto
2. **Scalabilità**: Facile aggiungere nuovi moduli
3. **Manutenibilità**: Codice facile da trovare e modificare
4. **Professionalità**: Struttura standard Python
5. **Separazione**: Codice, dati, documentazione separati

## ⚠️ Attenzione

- Aggiorna tutti gli import dopo lo spostamento
- Testa che tutto funzioni dopo la migrazione
- Mantieni backup prima di ristrutturare
- Aggiorna .gitignore per nuove directory

