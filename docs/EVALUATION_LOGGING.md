# 📊 Evaluation Logging System

## Overview

Sistema per salvare automaticamente l'output completo delle evaluation in file di log con timestamp.

## Struttura

```
output/
└── evaluation/
    ├── eval_narrativo_clinical-bert_20251006_143022.log
    ├── eval_narrativo_clinical-modernbert_20251006_144315.log
    └── eval_bullet_pubmedbert-base_20251006_150120.log
```

## Features

✅ **Timestamp Automatico**: Ogni log ha formato `eval_{formato}_{modello}_{timestamp}.log`  
✅ **Output Completo**: Cattura stdout e stderr (`2>&1`)  
✅ **Display + Save**: Usa `tee` per mostrare output sia a schermo che su file  
✅ **Exit Code Handling**: Controlla successo/fallimento con `${PIPESTATUS[0]}`  
✅ **Path Centralizzato**: `EVALUATION_DIR` definito in `src/config/paths.py`

## Formato Filename

```
eval_{STORY_FORMAT}_{MODEL_NAME}_{TIMESTAMP}.log
```

**Esempio**:
```
eval_narrativo_clinical-modernbert_20251006_143022.log
    │         │                      │
    │         │                      └─ Timestamp (YYYYMMDD_HHMMSS)
    │         └─ Model name
    └─ Story format
```

## Contenuto Log File

Ogni file contiene:

1. **Header Configuration**
   ```
   ============================================================
     LEGOLAS - Evaluation Modello XES
     Formato: narrativo
     Modello: clinical-modernbert
   ============================================================
   ```

2. **Model Loading Info**
   ```
   📋 Caricamento configurazioni modelli da YAML...
   ✅ 13 modelli caricati
   
   🤖 Modello selezionato: clinical-modernbert
      HuggingFace ID: Simonlee711/Clinical_ModernBERT
      Tipo: bert
   ```

3. **Data Loading**
   ```
   📖 Caricamento storie formato 'narrativo'...
      ✅ Test stories: 2514 campioni
   
   📥 Caricamento modello: clinical-modernbert
      ✅ Trovati 4 modelli, uso: output/models/xes_narrativo_clinical-modernbert4.pth
   ```

4. **Classification Report**
   ```
   ============================================================
     CLASSIFICATION REPORT
   ============================================================
   
                 precision    recall  f1-score   support
   
              0     0.9602    0.9808    0.9704      2239
              1     0.8106    0.6691    0.7331       275
   
       accuracy                         0.9467      2514
   ```

5. **Confusion Matrix**
   ```
   ============================================================
     CONFUSION MATRIX
   ============================================================
   
   [[2196   43]
    [  91  184]]
   ```

6. **Save Confirmation**
   ```
   ✅ Risultati salvati in: prediction/xes_narrativo_clinical-modernbert_*
   ```

## Uso

### Opzione 1: Solo Evaluation
```bash
./scripts/launch_run_train_eval

# Scegli:
# - Formato: narrativo
# - Modello: clinical-modernbert
# - Azione: 2 (Solo evaluation)
```

**Output**:
- Mostra evaluation a schermo (live)
- Salva in `output/evaluation/eval_narrativo_clinical-modernbert_TIMESTAMP.log`
- Messaggio finale con path log

### Opzione 2: Training + Evaluation
```bash
./scripts/launch_run_train_eval

# Scegli:
# - Formato: narrativo
# - Modello: clinical-modernbert  
# - Azione: 3 (Training + Evaluation)
```

**Output**:
- Training output a schermo (non salvato)
- Evaluation mostrata a schermo + salvata in log
- Timestamp generato DOPO training (riflette orario evaluation)

## Bash Implementation Details

### Timestamp Generation
```bash
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
```
**Formato**: YYYYMMDD_HHMMSS (es: `20251006_143022`)

### Filename Generation
```bash
EVAL_LOG="${EVALUATION_DIR}/eval_${FORMAT}_${MODEL}_${TIMESTAMP}.log"
```

### Output Redirection con tee
```bash
uv run python src/training/eval_model.py 2>&1 | tee "${EVAL_LOG}"
```

**Spiegazione**:
- `2>&1` - Redirige stderr a stdout (cattura anche errori)
- `| tee FILE` - Invia output sia a stdout (schermo) che a FILE
- Mantiene esperienza interattiva mentre salva

### Exit Code Handling
```bash
EVAL_EXIT_CODE=${PIPESTATUS[0]}

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation completato!"
else
    echo "❌ Errore durante evaluation"
fi
```

**PIPESTATUS[0]**: Exit code del primo comando nella pipe (prima di `tee`)

## Path Configuration

### Python (src/config/paths.py)
```python
EVALUATION_DIR = OUTPUT_DIR / "evaluation"

def ensure_directories():
    dirs = [
        # ...
        EVALUATION_DIR,  # ← Aggiunto
        # ...
    ]
```

### Bash (scripts/launch_run_train_eval)
```bash
EVALUATION_DIR="output/evaluation"

# Crea directory se non esiste
mkdir -p "${EVALUATION_DIR}"
```

## File Generati

Per ogni evaluation vengono creati:

1. **Log file completo**
   - Path: `output/evaluation/eval_{format}_{model}_{timestamp}.log`
   - Contiene: Output completo evaluation
   - Dimensione: ~5-10 KB

2. **Predizioni (già esistente)**
   - Path: `prediction/xes_{format}_{model}_*.pkl`
   - Files: `_prob.pkl`, `_all_target.pkl`, `_all_prediction.pkl`

3. **Report testuale (già esistente)**
   - Path: `prediction/xes_{format}_{model}_report.txt`
   - Contiene: Classification report + confusion matrix

## Vantaggi

### 🔍 **Tracciabilità**
- Ogni run ha un log timestamped
- Facile confronto tra evaluation diverse
- Storia completa delle evaluation

### 📊 **Debugging**
- Cattura messaggi di warning/error
- Output modello HuggingFace incluso
- Info loading modelli salvata

### 📈 **Analisi**
- Grep su tutti i log per metriche
- Script automatici per parsing risultati
- Confronto performance nel tempo

### 💾 **Archivio**
- Log persistenti (non persi)
- Backup automatico dei risultati
- Riproducibilità garantita

## Esempi Query

### Lista tutti i log evaluation
```bash
ls -lh output/evaluation/
```

### Trova log per modello specifico
```bash
ls output/evaluation/*clinical-bert*
```

### Cerca accuracy in tutti i log
```bash
grep "accuracy" output/evaluation/*.log
```

### Mostra ultimo log
```bash
ls -t output/evaluation/*.log | head -1 | xargs cat
```

### Confronta F1-score tra modelli
```bash
grep "f1-score" output/evaluation/*.log
```

### Conta evaluation per formato
```bash
ls output/evaluation/eval_narrativo_* | wc -l
ls output/evaluation/eval_bullet_* | wc -l
ls output/evaluation/eval_clinical_* | wc -l
```

## Cleanup

### Rimuovi log vecchi (> 30 giorni)
```bash
find output/evaluation/ -name "*.log" -mtime +30 -delete
```

### Archivia log vecchi
```bash
tar -czf evaluation_logs_archive_$(date +%Y%m).tar.gz output/evaluation/*.log
```

### Mantieni solo ultimi 10 log per modello
```bash
for model in clinical-bert clinical-modernbert pubmedbert-base; do
    ls -t output/evaluation/*${model}*.log | tail -n +11 | xargs rm -f
done
```

## Integration con Git

### .gitignore
```gitignore
# Evaluation logs (troppo grandi per git)
output/evaluation/*.log

# Ma mantieni directory structure
!output/evaluation/.gitkeep
```

### Crea .gitkeep
```bash
touch output/evaluation/.gitkeep
```

## Future Enhancements

- [ ] JSON format logs per parsing automatico
- [ ] Summary report aggregato (ultimo log per ogni modello)
- [ ] Script Python per analisi logs
- [ ] Web dashboard per visualizzazione risultati
- [ ] Email notification con summary
- [ ] Slack/Discord integration per alert

## Troubleshooting

### Log file vuoto
**Problema**: File creato ma 0 bytes  
**Causa**: Python buffering o crash immediato  
**Soluzione**: Aggiungi `PYTHONUNBUFFERED=1` prima di comando

### Timestamp duplicati
**Problema**: Due run nella stesso secondo  
**Causa**: Timestamp con secondi solo  
**Soluzione**: Già implementato (secondi inclusi)

### Permission denied
**Problema**: Impossibile scrivere in output/evaluation/  
**Causa**: Directory senza permessi write  
**Soluzione**: `chmod u+w output/evaluation/`

## Summary

✅ **Implementato**: Output evaluation salvato automaticamente  
✅ **Path**: `output/evaluation/eval_{format}_{model}_{timestamp}.log`  
✅ **Display**: Live output + save (con `tee`)  
✅ **Exit code**: Gestito correttamente  
✅ **Timestamp**: Unico per ogni run  
✅ **Documentation**: Completa  

**Ready for production!** 🚀
