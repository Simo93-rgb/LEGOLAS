# Evaluation Logs

Questa directory contiene i log completi delle evaluation dei modelli.

## Struttura File

Ogni file ha formato:
```
eval_{STORY_FORMAT}_{MODEL_NAME}_{TIMESTAMP}.log
```

**Esempio**: `eval_narrativo_clinical-bert_20251006_143022.log`

## Contenuto

Ogni log include:
- Configurazione modello
- Info caricamento dati
- Classification report completo
- Confusion matrix
- Metriche (precision, recall, F1-score)
- Output warnings/errors

## Generazione Automatica

I log vengono generati automaticamente dallo script:
```bash
./scripts/launch_run_train_eval
```

Quando selezioni opzione 2 (Solo evaluation) o 3 (Training + Evaluation).

## Utilità

### Trova ultimo log
```bash
ls -t *.log | head -1
```

### Cerca metriche
```bash
grep "accuracy" *.log
grep "f1-score" *.log
```

### Lista log per modello
```bash
ls *clinical-bert*.log
ls *pubmedbert*.log
```

## Retention

Per evitare accumulo eccessivo:
- Mantieni log delle ultime 4 settimane
- Archivia log più vecchi se necessari per audit
- Considera cleanup automatico

## Note

- File di testo UTF-8
- Include anche stderr (errori catturati)
- Timestamp format: YYYYMMDD_HHMMSS
- Path definito in: `src/config/paths.py`
