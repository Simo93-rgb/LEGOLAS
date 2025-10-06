# ✅ Training e Evaluation - Implementazione Completata

## 🎯 Modifiche Implementate

### 1. `train_llm.py` - Modificato
**Cosa è stato fatto**:
- ✅ Aggiunta configurazione `STORY_FORMAT` per scegliere il formato delle storie
- ✅ Caricamento automatico file da `output/{formato}_*.pkl`
- ✅ Nome modello salvato include formato: `xes_{formato}_{modello}.pth`
- ✅ Output informativo con emoji e separatori visivi

**Variabili configurabili**:
```python
STORY_FORMAT = 'narrativo'  # Cambia in: 'bullet', 'clinical'
model_name = 'bertm'        # Cambia in: 'roberta', 'gpt2', 'cbert'
LEARNING_RATE = 1e-5
BATCH = 256
```

### 2. `eval_model.py` - Modificato
**Cosa è stato fatto**:
- ✅ Aggiunta configurazione `STORY_FORMAT` (deve corrispondere a train_llm.py)
- ✅ Caricamento automatico file da `output/{formato}_*.pkl`
- ✅ Caricamento modello con nome corretto: `xes_{formato}_{modello}.pth`
- ✅ Risultati salvati con prefisso descrittivo: `xes_{formato}_{modello}_*`
- ✅ Output migliorato con report formattato
- ✅ Creazione automatica directory `prediction/`

**Variabili configurabili**:
```python
STORY_FORMAT = 'narrativo'  # DEVE corrispondere a train_llm.py!
model_name = 'bertm'
```

### 3. `launch_run_train_eval` - Nuovo Script Helper
**Cosa fa**:
- 🔧 Menu interattivo per scegliere formato storie
- 🔧 Menu interattivo per scegliere modello
- 🔧 Aggiorna automaticamente configurazioni in entrambi i file
- 🔧 Esegue training e/o evaluation
- 🔧 Output colorato e informativo

## 🚀 Come Usare

### Metodo 1: Script Helper (RACCOMANDATO)
```bash
./launch_run_train_eval
```

Menu interattivo ti guiderà attraverso:
1. Scelta formato storie (narrativo/bullet/clinical)
2. Scelta modello (bertm/roberta/cbert/gpt2)
3. Scelta azione (training/eval/entrambi)

### Metodo 2: Manuale

#### Step 1: Configura i file

Modifica `train_llm.py`:
```python
STORY_FORMAT = 'narrativo'  # o 'bullet', 'clinical'
model_name = 'bertm'        # o 'roberta', 'gpt2', 'cbert'
```

Modifica `eval_model.py` (usa STESSO formato!):
```python
STORY_FORMAT = 'narrativo'  # DEVE corrispondere!
model_name = 'bertm'
```

#### Step 2: Esegui Training
```bash
python train_llm.py
```

#### Step 3: Esegui Evaluation
```bash
python eval_model.py
```

## 📊 File Generati

### Training
```
xes_narrativo_bertm.pth              # Modello addestrato
xes_bullet_bertm.pth                 # (se usi formato bullet)
xes_clinical_bertm.pth               # (se usi formato clinical)
```

### Evaluation
```
prediction/
├── xes_narrativo_bertm_prob.pkl          # Probabilità predette
├── xes_narrativo_bertm_all_target.pkl    # Label vere
├── xes_narrativo_bertm_all_prediction.pkl # Label predette
└── xes_narrativo_bertm_report.txt        # Report testuale
```

## 🎓 Esempi di Utilizzo

### Esempio 1: Training Base (BERT + Narrativo)
```bash
# Lascia configurazioni di default e esegui:
python train_llm.py
python eval_model.py
```

### Esempio 2: Training con Formato Diverso
```python
# In train_llm.py E eval_model.py, cambia:
STORY_FORMAT = 'bullet'  # Usa formato bullet points

# Poi esegui:
python train_llm.py
python eval_model.py
```

### Esempio 3: Confronto Modelli
```bash
# Test 1: BERT
sed -i "s/model_name = '.*'/model_name = 'bertm'/" train_llm.py
python train_llm.py
sed -i "s/model_name = '.*'/model_name = 'bertm'/" eval_model.py
python eval_model.py

# Test 2: Clinical BERT
sed -i "s/model_name = '.*'/model_name = 'cbert'/" train_llm.py
python train_llm.py
sed -i "s/model_name = '.*'/model_name = 'cbert'/" eval_model.py
python eval_model.py

# Confronta: prediction/xes_narrativo_bertm_report.txt
#         vs prediction/xes_narrativo_cbert_report.txt
```

### Esempio 4: Confronto Formati
```bash
# Usa lo script helper tre volte con formati diversi
./launch_run_train_eval  # Scegli 'narrativo'
./launch_run_train_eval  # Scegli 'bullet'
./launch_run_train_eval  # Scegli 'clinical'

# Confronta risultati in prediction/
```

## 📈 Cosa Aspettarsi

### Tempi di Esecuzione (Approssimativi)

#### Con GPU (NVIDIA CUDA)
- Training: 30-60 minuti per epoch
- Evaluation: 2-5 minuti
- **Totale**: ~1-2 ore (5 epochs)

#### Con CPU
- Training: 3-6 ore per epoch
- Evaluation: 15-30 minuti
- **Totale**: ~15-30 ore (5 epochs)

**Nota**: I tempi dipendono da:
- Hardware (GPU/CPU, RAM)
- Dimensione dataset (7,393 casi)
- Batch size (256 default)
- Modello scelto

### Metriche Attese

Il report includerà:
```
Classification Report:
              precision    recall  f1-score   support

           0     0.xxxx    0.xxxx    0.xxxx      yyyy
           1     0.xxxx    0.xxxx    0.xxxx      yyyy

    accuracy                         0.xxxx      zzzz
   macro avg     0.xxxx    0.xxxx    0.xxxx      zzzz
weighted avg     0.xxxx    0.xxxx    0.xxxx      zzzz

Confusion Matrix:
[[tp  fp]
 [fn  tn]]
```

## 🔧 Troubleshooting

### Problema 1: "File not found: output/narrativo_train.pkl"
**Soluzione**: Esegui prima la generazione storie:
```bash
./run_xes_pipeline.sh
```

### Problema 2: "CUDA out of memory"
**Soluzione**: Riduci batch size in `train_llm.py`:
```python
BATCH = 128  # invece di 256
# oppure
BATCH = 64   # se ancora problemi
```

### Problema 3: "Model file not found: xes_narrativo_bertm.pth"
**Soluzione**: Esegui prima il training:
```bash
python train_llm.py
```

### Problema 4: STORY_FORMAT diverso tra train e eval
**Errore**: Stai valutando con formato diverso da quello usato per training!

**Soluzione**: Assicurati che `STORY_FORMAT` sia identico in entrambi i file:
```bash
# Verifica configurazione
grep "STORY_FORMAT" train_llm.py eval_model.py

# Devono essere uguali!
```

### Problema 5: "AdamW not found"
**Soluzione**: Usa script aggiornati che usano `torch.optim.AdamW` invece di `transformers.AdamW`

## 📋 Checklist Pre-Training

Prima di eseguire il training:

- [ ] File storie generati in `output/` (esegui `./run_xes_pipeline.sh`)
- [ ] `STORY_FORMAT` configurato in `train_llm.py`
- [ ] `model_name` scelto in `train_llm.py`
- [ ] Spazio disco sufficiente (~2-5 GB per modello)
- [ ] GPU configurata (opzionale ma raccomandato)
- [ ] Tempo disponibile (1-30 ore a seconda di hardware)

## 📋 Checklist Pre-Evaluation

Prima di eseguire l'evaluation:

- [ ] Training completato (esiste file `xes_{formato}_{modello}.pth`)
- [ ] `STORY_FORMAT` in `eval_model.py` UGUALE a `train_llm.py`
- [ ] `model_name` in `eval_model.py` UGUALE a `train_llm.py`
- [ ] Directory `prediction/` esiste (creata automaticamente)

## 🎯 Best Practices

1. **Nomenclatura Consistente**: Usa sempre stesso `STORY_FORMAT` per training e evaluation
2. **Documentazione**: Annota quale combinazione formato+modello funziona meglio
3. **Backup**: Salva modelli addestrati prima di training successivi
4. **Confronto**: Prova più combinazioni formato/modello e confronta metriche
5. **Monitoraggio**: Controlla loss durante training per evitare overfitting

## 📊 Workflow Completo

```
1. Generazione Storie
   └─> ./run_xes_pipeline.sh
       └─> output/*_train.pkl, output/*_test.pkl

2. Training
   └─> python train_llm.py (o ./launch_run_train_eval)
       └─> xes_{formato}_{modello}.pth

3. Evaluation
   └─> python eval_model.py
       └─> prediction/xes_{formato}_{modello}_*

4. Analisi
   └─> Leggi prediction/xes_{formato}_{modello}_report.txt
       └─> Confronta metriche di diverse configurazioni
```

## 🏆 Conclusione

✅ **Training**: Configurato e pronto
✅ **Evaluation**: Configurato e pronto
✅ **Script Helper**: Disponibile per uso facile
✅ **Documentazione**: Completa

**Inizia ora**:
```bash
./launch_run_train_eval
```

---

**Data**: Ottobre 2025
**Versione**: 1.0
**Status**: ✅ Produzione
