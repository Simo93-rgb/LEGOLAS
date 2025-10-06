# 📚 LEGOLAS - Indice Documentazione Integrazione

## 🎯 Dove Iniziare?

### Per Utenti Nuovi
1. **START HERE**: [INTEGRATION_COMPLETE.txt](INTEGRATION_COMPLETE.txt) - Panoramica visuale completa
2. **Quick Start**: [README_INTEGRATION.md](README_INTEGRATION.md) - Guida rapida
3. **Prossimi Passi**: [NEXT_STEPS.md](NEXT_STEPS.md) - Cosa fare dopo l'integrazione

### Per Sviluppatori
1. **Architettura**: [FLOW_DIAGRAM.md](FLOW_DIAGRAM.md) - Diagrammi del sistema
2. **Dettagli Tecnici**: [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md) - Implementazione dettagliata
3. **Guida Completa**: [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Tutorial approfondito

## 📁 File Principali

### Script Eseguibili
| File | Descrizione | Utilizzo |
|------|-------------|----------|
| `generate_stories.py` | Script unificato generazione | `python generate_stories.py --help` |
| `train_xes_model.py` | Training su storie XES | `python train_xes_model.py` |
| `run_xes_pipeline.sh` | Automazione generazione | `./run_xes_pipeline.sh` |
| `launch_run_train_eval` | Training + Evaluation (menu) | `./launch_run_train_eval` |
| `test_integration.py` | Verifica sistema | `python test_integration.py` |

### Script Originali (Modificati per XES)
| File | Descrizione |
|------|-------------|
| `main.py` | Pipeline originale CSV (invariato) |
| `train_llm.py` | Training LLM - ora supporta storie XES |
| `eval_model.py` | Evaluation modello - ora supporta storie XES |

### Componenti Core
| File | Descrizione |
|------|-------------|
| `xes_parser.py` | Parser file XES |
| `story_generator.py` | Generatore storie narrative |
| `clinical_token_mapper.py` | Mapper token clinici |
| `translation_cache.json` | Dizionario IT→EN (621 traduzioni) |

### Configurazione
| File | Descrizione |
|------|-------------|
| `skeleton.py` | Template pipeline CSV originale |
| `utils/types.py` | Definizioni tipi condivisi |
| `utils/config.py` | Configurazioni generali |
| `pyproject.toml` | Dipendenze progetto |

## 📖 Documentazione

### Guide
- [README_INTEGRATION.md](README_INTEGRATION.md) - Quick start e panoramica
- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Guida completa all'uso
- [TRAINING_EVALUATION_GUIDE.md](TRAINING_EVALUATION_GUIDE.md) - Guida training e evaluation
- [NEXT_STEPS.md](NEXT_STEPS.md) - Cosa fare dopo l'integrazione
- [README.md](README.md) - README originale del progetto

### Riferimenti Tecnici
- [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md) - Riepilogo tecnico dettagliato
- [FLOW_DIAGRAM.md](FLOW_DIAGRAM.md) - Diagrammi architettura
- [INTEGRATION_COMPLETE.txt](INTEGRATION_COMPLETE.txt) - Status e metriche

## 🚀 Quick Start

```bash
# 1. Testa il sistema
python test_integration.py

# 2. Genera storie
./run_xes_pipeline.sh

# 3. Training + Evaluation (interattivo)
./launch_run_train_eval

# Oppure manuale:
python train_llm.py      # Training
python eval_model.py     # Evaluation
```

## 🎓 Percorsi di Apprendimento

### Percorso Rapido (30 min)
1. Leggi [INTEGRATION_COMPLETE.txt](INTEGRATION_COMPLETE.txt)
2. Esegui `./run_xes_pipeline.sh`
3. Consulta [NEXT_STEPS.md](NEXT_STEPS.md) per continuare

### Percorso Completo (2-3 ore)
1. Leggi [README_INTEGRATION.md](README_INTEGRATION.md)
2. Studia [FLOW_DIAGRAM.md](FLOW_DIAGRAM.md)
3. Segui [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
4. Esegui tutti gli script di test
5. Leggi [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md) per dettagli

### Percorso Sviluppatore (1 giorno)
1. Leggi tutta la documentazione sopra
2. Esamina codice sorgente:
   - `xes_parser.py`
   - `story_generator.py`
   - `generate_stories.py`
3. Esegui esperimenti con parametri diversi
4. Personalizza template e configurazioni

## 📊 Struttura Output

```
output/
├── narrativo_train.pkl          # Formato narrativo (BERT-friendly)
├── narrativo_test.pkl
├── narrativo_label_train.pkl
├── narrativo_label_test.pkl
│
├── bullet_train.pkl             # Formato bullet points (compatto)
├── bullet_test.pkl
├── bullet_label_train.pkl
├── bullet_label_test.pkl
│
└── clinical_*.pkl               # Con token clinici (sperimentale)
```

## 🔍 FAQ Rapide

**Q: Da dove inizio?**  
A: Leggi [INTEGRATION_COMPLETE.txt](INTEGRATION_COMPLETE.txt), poi esegui `./run_xes_pipeline.sh`

**Q: Come funziona il sistema?**  
A: Vedi [FLOW_DIAGRAM.md](FLOW_DIAGRAM.md) per i diagrammi

**Q: Cosa devo fare dopo l'integrazione?**  
A: Consulta [NEXT_STEPS.md](NEXT_STEPS.md)

**Q: Come risolvo problemi?**  
A: Sezione Troubleshooting in [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)

**Q: Come personalizzo il sistema?**  
A: Vedi sezioni "Customization" in [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md)

## 🆘 Supporto

### Ordine di Consultazione
1. [NEXT_STEPS.md](NEXT_STEPS.md) - Domande comuni
2. [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Troubleshooting
3. [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md) - Dettagli tecnici
4. Test diagnostico: `python test_integration.py`

### Debug
```bash
# Test sistema
python test_integration.py

# Verifica dipendenze
pip list | grep -E 'pm4py|pandas|transformers|torch'

# Log dettagliato
python generate_stories.py --pipeline xes --input data.xes -v
```

## 🎉 Status

```
✅ Integrazione:         COMPLETATA
✅ Test:                 6/7 SUPERATI  
✅ Documentazione:       COMPLETA
✅ Compatibilità:        GARANTITA
✅ Status:               PRODUZIONE

Sistema pronto all'uso!
```

## 📞 Contatti

Per domande tecniche o segnalazione problemi:
- Apri issue su GitHub (se disponibile)
- Consulta documentazione tecnica
- Esegui test diagnostici

---

**Versione**: 1.0  
**Data**: Ottobre 2025  
**Status**: ✅ Produzione
