# ⚡ LEGOLAS - Quick Reference

## 🚀 Comandi Essenziali

### Training + Evaluation
```bash
./scripts/launch_run_train_eval
```
Menu interattivo con:
- 3 formati storie (narrativo, bullet, clinical)
- 13 modelli biomedici
- 3 modalità (training, eval, completo)

**Output**:
- Modello: `output/models/xes_{formato}_{modello}*.pth`
- Log eval: `output/evaluation/eval_{formato}_{modello}_{timestamp}.log`
- Predizioni: `prediction/xes_{formato}_{modello}_*.pkl`
- Report: `prediction/xes_{formato}_{modello}_report.txt`

---

## 📊 Modelli Disponibili

### Biomedici Specializzati (consigliati)
```
clinical-bert          → Note cliniche (default)
pubmedbert-base        → Letteratura PubMed
biobert-base           → Testi biomedici
bluebert-base          → Dominio clinico
clinical-modernbert    → Architettura moderna
scibert-base           → Letteratura scientifica
```

### Longformer (sequenze lunghe)
```
clinical-longformer    → Clinical 4096 token
longformer-base        → Base 4096 token
```

### Generici
```
bert-base-uncased      → BERT standard
bert-large-uncased     → BERT large
bert-italian           → BERT italiano
```

---

## 📁 Struttura Output

```
output/
├── stories/              # Storie generate da XES
│   ├── narrativo_*.pkl
│   ├── bullet_*.pkl
│   └── clinical_*.pkl
│
├── models/              # Modelli addestrati
│   └── xes_narrativo_clinical-bert1.pth
│
└── evaluation/          # ⭐ Log evaluation
    └── eval_narrativo_clinical-bert_20251006_143022.log

prediction/              # Risultati evaluation
├── xes_narrativo_clinical-bert_prob.pkl
├── xes_narrativo_clinical-bert_all_target.pkl
├── xes_narrativo_clinical-bert_all_prediction.pkl
└── xes_narrativo_clinical-bert_report.txt
```

---

## 🔧 Configurazione

### Aggiungere Nuovo Modello
```yaml
# Edit: config/model_configs.yaml

models:
  my-new-model:
    name: organization/model-id-on-huggingface
    type: bert
    max_length: 512
    num_labels: 8
    description: "Descrizione modello"
    recommended_batch_size: 12
    recommended_lr: 2.0e-5
```

Poi lancia `./scripts/launch_run_train_eval` → appare nel menu automaticamente!

### Modificare Path
```python
# Edit: src/config/paths.py

# Esempio: spostare modelli
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
```

Tutte le modifiche si propagano automaticamente.

---

## 📈 Analisi Risultati

### Ultimo log evaluation
```bash
ls -t output/evaluation/*.log | head -1 | xargs cat
```

### Cerca accuracy
```bash
grep "accuracy" output/evaluation/*.log
```

### Confronta F1-score
```bash
grep "f1-score" output/evaluation/*.log | grep "weighted avg"
```

### Lista log per modello
```bash
ls output/evaluation/*clinical-bert*.log
```

---

## 🐛 Troubleshooting

### "File di storie non trovati"
```bash
./scripts/run_xes_pipeline.sh  # Genera storie prima
```

### "Modello non trovato"
Verifica che esista in:
```bash
cat config/model_configs.yaml | grep -A 5 "model-name"
```

### "Out of memory GPU"
Riduci batch size in `src/training/train_llm.py`:
```python
BATCH = 16  # o 8 per GPU piccole
```

### Lista modelli disponibili
```bash
./scripts/list_models.py menu
```

---

## 📚 Documentazione Completa

```
docs/
├── INTEGRATION_COMPLETE_SUMMARY.md    # ⭐ Overview completo
├── MODEL_CONFIG_GUIDE.md              # Guida modelli
├── PATH_CENTRALIZATION_COMPLETE.md    # Guida path
├── EVALUATION_LOGGING.md              # Guida logging
└── BUGFIX_MODEL_CONFIG.md             # Bugfix details
```

---

## 💡 Tips & Tricks

### Performance
- **clinical-bert**: Best trade-off accuratezza/velocità
- **pubmedbert-base**: Ottimo per terminologia medica
- **longformer**: Per note cliniche molto lunghe (>512 token)

### Formati Storie
- **narrativo**: Raccomandato per BERT (più context)
- **bullet**: Più compatto, veloce training
- **clinical**: Sperimentale con token speciali

### GPU Memory
- BERT base: batch 12-16 (OK con 12GB VRAM)
- BERT large: batch 8 (richiede 16GB+)
- Longformer: batch 4-6 (memoria intensivo)

---

## 🎯 Workflow Tipico

1. **Genera storie** (se non fatto):
   ```bash
   ./scripts/run_xes_pipeline.sh
   ```

2. **Training + Evaluation**:
   ```bash
   ./scripts/launch_run_train_eval
   ```
   - Formato: narrativo (default)
   - Modello: clinical-bert (consigliato)
   - Azione: 3 (completo)

3. **Analizza risultati**:
   ```bash
   # Log evaluation
   cat output/evaluation/eval_narrativo_clinical-bert_*.log
   
   # Report testuale
   cat prediction/xes_narrativo_clinical-bert_report.txt
   ```

4. **Prova altri modelli**:
   - Ripeti step 2 con modello diverso
   - Confronta log in `output/evaluation/`

---

## 🔗 Quick Links

- **Main script**: `scripts/launch_run_train_eval`
- **Model config**: `config/model_configs.yaml`
- **Path config**: `src/config/paths.py`
- **Training code**: `src/training/train_llm.py`
- **Eval code**: `src/training/eval_model.py`

---

## ⚡ One-Liners

```bash
# Lista modelli
./scripts/list_models.py menu

# Test paths
uv run python src/config/paths.py

# Import test
uv run python -c "from src.training.train_llm import *; print('OK')"

# Ultimi 3 log
ls -t output/evaluation/*.log | head -3

# Accuracy tutti i modelli
grep "accuracy" output/evaluation/*.log
```

---

**Need help?** Controlla `docs/INTEGRATION_COMPLETE_SUMMARY.md` per dettagli completi.

✅ **System ready!** Esegui `./scripts/launch_run_train_eval` per iniziare.
