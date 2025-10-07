# Explainability Improvements - Clinical Actions Visualization

## Problemi Risolti

### 1. ❌ Problema: Visualizzazioni con Parole Singole
**Prima**: Heatmap e istogrammi mostravano parole singole (es: "urological", "catheter")
- ❌ Non chiaro a quale azione clinica si riferisce
- ❌ Difficile interpretare il contesto medico
- ❌ Token strani con alcuni modelli (clinical-modernbert)

**Dopo**: Visualizzazioni basate su **Azioni Cliniche Complete**
- ✅ Nomi completi delle azioni (es: "Central venous catheter placement")
- ✅ Direttamente interpretabili dal personale medico
- ✅ Aggrega scores da componenti multiple
- ✅ Usa le 621 azioni reali dal file `data/translation_cache.json`

### 2. ❌ Problema: File PKL Non Utilizzati
**Prima**: File `actions_*.pkl` venivano salvati ma non usati per visualizzazioni

**Dopo**: File PKL usati per generare visualizzazioni principali
- ✅ Heatmap azioni cliniche
- ✅ Histogram comparativo azioni cliniche

### 3. ❌ Problema: Token Strani con Clinical-ModernBERT
**Sintomo**: Token tipo `Ġ737`, `Ġ601` invece di parole leggibili
**Causa**: Tokenizer ModernBERT usa formato diverso da BERT standard (Ġ per spazi)
**Stato**: Identificato, soluzione parziale (aggregazione ad azioni risolve il problema per le visualizzazioni principali)

## Nuove Funzionalità

### 1. Visualizzazioni a Livello Azione Clinica

#### `plot_clinical_actions_heatmap()`
```python
plot_clinical_actions_heatmap(
    actions_class_0: Dict[str, Dict],  # {'action': {'mean_score': ..., 'count': ...}}
    actions_class_1: Dict[str, Dict],
    top_k: int = 25,
    save_path: str = None
)
```

**Output**:
- Heatmap 2xK (2 classi, K azioni)
- Azioni ordinate per importanza Classe 0
- Normalizzazione globale (max=1)
- Nomi azioni truncati a 50 caratteri

**File generato**: `heatmap_actions_{format}_{model}_{timestamp}.png`

#### `plot_clinical_actions_comparison()`
```python
plot_clinical_actions_comparison(
    actions_class_0: Dict[str, Dict],
    actions_class_1: Dict[str, Dict],
    top_k: int = 25,
    save_path: str = None
)
```

**Output**:
- Barre orizzontali affiancate
- Classe 0 (blue) vs Classe 1 (red)
- Normalizzazione separata per confronto
- Valori annotati sulle barre

**File generato**: `histogram_actions_{format}_{model}_{timestamp}.png`

### 2. Doppia Visualizzazione

Lo script `extract_explainability.py` ora genera **entrambe** le visualizzazioni:

1. **Word-level** (reference, per debugging):
   - `heatmap_words_*.png`
   - `histogram_words_*.png`

2. **Clinical Actions** (main, interpretabili):
   - `heatmap_actions_*.png` ← **PRINCIPALE**
   - `histogram_actions_*.png` ← **PRINCIPALE**

### 3. Caricamento Azioni da JSON

`ClinicalActionAggregator` ora legge le **621 azioni reali** da:
```
/home/simon/GitHub/LEGOLAS/data/translation_cache.json
```

**Struttura JSON**:
```json
{
    "VISITA PSICHIATRICA DI CONTROLLO": "Psychiatric follow-up visit",
    "ECOGRAFIA ADDOME INFERIORE": "Lower abdomen ultrasound",
    ...
}
```

**Modifiche**:
- Import `json` e `pathlib`
- Metodo `_default_clinical_actions()` legge dal file
- Fallback a set minimo in caso di errore
- Print informativo: `✅ Loaded 621 clinical actions`

## Output Files Aggiornati

```
output/explainability/
├── ig_results_{format}_{model}_{timestamp}.pkl          # Raw IG scores
├── actions_{format}_{model}_{timestamp}.pkl             # Aggregated actions
│
├── heatmap_actions_{format}_{model}_{timestamp}.png     # ✨ MAIN heatmap
├── histogram_actions_{format}_{model}_{timestamp}.png   # ✨ MAIN histogram
│
├── heatmap_words_{format}_{model}_{timestamp}.png       # Reference (words)
└── histogram_words_{format}_{model}_{timestamp}.png     # Reference (words)
```

## Esempio Output

### Console Output
```
🏥 Aggregating clinical actions...
✅ Loaded 621 clinical actions from translation file

   Top 25 clinical actions for Class 0:
      1. Endovascular treatment of intracranial aneurysms...
         Mean: 0.8257, Count: 1
      2. Central venous catheter placement...
         Mean: 0.5534, Count: 1
      3. Transesophageal cardiac color Doppler ultrasound...
         Mean: 0.5075, Count: 1
      ...

🎨 Creating visualizations...
   📝 Word-level visualizations...
   ✅ Heatmap salvata: heatmap_words_*.png
   ✅ Istogramma salvato: histogram_words_*.png
   
   🏥 Clinical actions visualizations...
   ✅ Clinical actions heatmap salvata: heatmap_actions_*.png
   ✅ Clinical actions comparison salvato: histogram_actions_*.png
```

### Heatmap Actions (Esempio)
```
+---------------------------+-------+-------+
| Clinical Action           | C0    | C1    |
+---------------------------+-------+-------+
| Central venous catheter   | 0.85  | 0.12  |
| Lower abdomen ultrasound  | 0.72  | 0.45  |
| Psychiatric follow-up     | 0.68  | 0.89  |
| ...                       | ...   | ...   |
+---------------------------+-------+-------+
```

**Colori**: YlOrRd (giallo → arancione → rosso per score crescente)

### Histogram Actions (Esempio)
```
Central venous catheter    ████████████████░░ 0.85
                           ██░░░░░░░░░░░░░░░░ 0.12

Lower abdomen ultrasound   ███████████░░░░░░░ 0.72
                           ███████░░░░░░░░░░░ 0.45

Psychiatric follow-up      ██████████░░░░░░░░ 0.68
                           ████████████████░░ 0.89
```

**Legenda**: Blue (Class 0), Red (Class 1)

## Testing

### Test con Bert-Base-Uncased ✅
```bash
python -m src.explainability.extract_explainability \
    --model bert-base-uncased \
    --format narrativo \
    --top_k 25 \
    --n_samples 50

# Output:
# ✅ 185 azioni trovate per Class 0
# ✅ 131 azioni trovate per Class 1
# ✅ Visualizzazioni generate
```

### Test con Clinical-ModernBERT ⚠️
```bash
python -m src.explainability.extract_explainability \
    --model clinical-modernbert \
    --format narrativo \
    --top_k 25 \
    --n_samples 50

# Output:
# ⚠️  Word-level: token strani (Ġ737, Ġ601)
# ✅ Clinical actions: funzionano correttamente!
```

**Conclusione**: Le visualizzazioni ad azioni cliniche funzionano anche con ModernBERT perché l'aggregazione bypassa il problema dei token strani.

## File Modificati

1. **`src/explainability/visualization.py`**
   - Aggiunte: `plot_clinical_actions_heatmap()`, `plot_clinical_actions_comparison()`
   - ~180 righe nuove

2. **`src/explainability/action_aggregator.py`**
   - Import `json`, `pathlib`
   - `_default_clinical_actions()` legge da JSON
   - Fallback a set minimo

3. **`src/explainability/extract_explainability.py`**
   - Import nuove funzioni
   - Generazione doppia visualizzazione (words + actions)
   - Output summary aggiornato

4. **`src/explainability/__init__.py`**
   - Export `plot_clinical_actions_heatmap`, `plot_clinical_actions_comparison`

## Prossimi Passi

### 1. Fix Token Strani ModernBERT (Optional)
Aggiornare `aggregate_subword_attributions()` per gestire:
- Token BERT: `##prefix`
- Token GPT2/ModernBERT: `Ġprefix`

### 2. Ottimizzazione Visualizzazioni
- Raggruppamento azioni per categoria (Visite, Esami, Procedure)
- Clustering gerarchico delle azioni simili
- Colormap differenziate per tipo azione

### 3. Statistical Significance
- Test chi-quadro per differenze classe 0 vs classe 1
- Confidence intervals sugli score medi
- P-values sulle differenze

### 4. Interactive Visualization
- Plotly per plot interattivi
- Hover per vedere testi originali
- Drill-down da azione → testi esempi

## Usage Examples

### Generate Explainability with Clinical Actions
```bash
# Full run
bash scripts/run_explainability.sh

# Or direct Python
python -m src.explainability.extract_explainability \
    --model bert-base-uncased \
    --format narrativo \
    --top_k 25 \
    --n_samples 150
```

### Load and Re-visualize
```python
import pickle
from src.explainability import plot_clinical_actions_heatmap

# Load actions
with open('output/explainability/actions_*.pkl', 'rb') as f:
    actions = pickle.load(f)

# Re-plot with different top-k
plot_clinical_actions_heatmap(
    actions['class_0'],
    actions['class_1'],
    top_k=30,
    save_path='custom_heatmap.png'
)
```

### Analyze Specific Action
```python
import pickle

with open('output/explainability/actions_*.pkl', 'rb') as f:
    actions = pickle.load(f)

# Check specific action
action = "Central venous catheter placement"
if action in actions['class_0']:
    stats = actions['class_0'][action]
    print(f"Action: {action}")
    print(f"  Mean score: {stats['mean_score']:.4f}")
    print(f"  Std: {stats['std_score']:.4f}")
    print(f"  Count: {stats['count']}")
```

## Benefits

1. **✅ Interpretabilità Medica**: Azioni cliniche complete invece di parole isolate
2. **✅ Consistenza**: 621 azioni reali dal file di traduzione
3. **✅ Flessibilità**: Doppia visualizzazione (words + actions)
4. **✅ Scalabilità**: Gestisce grandi numeri di azioni (top-k configurabile)
5. **✅ Robustezza**: Fallback per errori di caricamento JSON
6. **✅ Debugging**: Visualizzazioni word-level mantenute per reference
