# Explainability Module - Usage Guide

## Overview
Sistema di explainability per LEGOLAS basato su **Integrated Gradients** (Captum library).

Estrae attribution scores a tre livelli:
1. **Token level**: Scores per ogni sub-word BERT token
2. **Word level**: Aggregazione da sub-words a parole complete
3. **Action level**: Aggregazione da parole ad azioni cliniche (es: "Psychiatric visit")

## Quick Start

### Via Bash Script (Raccomandato)
```bash
bash scripts/run_explainability.sh
```

Menu interattivo per selezionare:
- Story format (narrativo, bullet, clinical)
- Model name (clinical-bert, pubmedbert, etc.)
- Advanced parameters (n_samples, top_k, batch_size)

### Via Python Diretto
```bash
# Example: clinical-bert + narrativo format
python -m src.explainability.extract_explainability \
    --model clinical-bert \
    --format narrativo \
    --top_k 25 \
    --batch_size 8 \
    --n_steps 50 \
    --device cuda

# Example: analyze only 100 samples
python -m src.explainability.extract_explainability \
    --model pubmedbert-base \
    --format bullet \
    --n_samples 100 \
    --top_k 20
```

## Parameters

### Required
- `--model`: Model name from config (clinical-bert, pubmedbert-base, etc.)
- `--format`: Story format (narrativo, bullet, clinical)

### Optional
- `--n_samples`: Number of test samples to analyze (default: all)
- `--top_k`: Number of top words to visualize (default: 25)
- `--batch_size`: Batch size for IG computation (default: 8)
- `--n_steps`: Integrated Gradients steps (default: 50)
- `--device`: Computation device (cuda/cpu, auto-detected)

## Output Files

Tutti i file vengono salvati in `output/explainability/`:

### 1. IG Results (Pickle)
`ig_results_{format}_{model}_{timestamp}.pkl`
- Raw attribution scores per ogni sample
- Formato: list of dicts con keys: 'text', 'words', 'token_attributions', 'word_attributions', 'label', 'predicted_class'

### 2. Clinical Actions (Pickle)
`actions_{format}_{model}_{timestamp}.pkl`
- Aggregazione per azioni cliniche
- Formato: dict con statistics per action (mean_score, std, count)

### 3. Heatmap Visualization (PNG)
`heatmap_{format}_{model}_{timestamp}.png`
- 2 righe: Class 0 (DISCHARGED), Class 1 (ADMITTED)
- Top-25 parole ordinate per importanza su classe 0
- Normalization: globale (max=1)
- Colormap: YlOrRd

### 4. Comparative Histogram (PNG)
`histogram_{format}_{model}_{timestamp}.png`
- Barre affiancate per classe 0 (blue) e classe 1 (red)
- Top-25 parole ordinate per importanza su classe 0
- Normalization: separata per classe (max in display)

## Implementation Details

### Integrated Gradients
- Baseline: PAD token sequence
- Steps: 50 (default, adjustable)
- Target: Logit della classe predetta
- Aggregation: Sum su dimensione embedding

### Sub-word to Word Aggregation
```python
# BERT tokenizer produces sub-words: ["psych", "##iatric", "visit"]
# Word aggregation sums scores: score("psychiatric") = score("psych") + score("##iatric")
```

### Word to Action Aggregation
```python
# Clinical actions dictionary (~40 actions)
clinical_actions = {
    'psychiatric_visit': ['psychiatric', 'visit', 'consultation'],
    'blood_test': ['blood', 'test', 'exam', 'analysis'],
    ...
}
# Action score = sum of component word scores
```

### Class Mapping
- **Class 0**: DISCHARGED (majority class in dataset)
- **Class 1**: ADMITTED (minority class)

Verificare con:
```python
from collections import Counter
Counter(true_labels)  # Should show Class 0 > Class 1
```

## Example Workflow

1. **Train model**:
   ```bash
   bash scripts/launch_run_train_eval
   ```

2. **Extract explainability**:
   ```bash
   bash scripts/run_explainability.sh
   # Select: narrativo + clinical-bert
   ```

3. **Check output**:
   ```bash
   ls -lh output/explainability/
   # Should see: ig_results_*.pkl, actions_*.pkl, heatmap_*.png, histogram_*.png
   ```

4. **Analyze results**:
   ```python
   import pickle
   with open('output/explainability/ig_results_narrativo_clinical-bert_*.pkl', 'rb') as f:
       results = pickle.load(f)
   
   # Top words per sample
   for r in results[:5]:
       print(f"Text: {r['text'][:100]}...")
       print(f"Top 5 words: {r['word_attributions'][:5]}")
   ```

## Troubleshooting

### Error: "Nessun modello addestrato trovato"
- Verificare che esista file in `output/models/xes_{format}_{model}*.pth`
- Trainare modello prima di estrarre explainability

### Error: "Class 0 has fewer samples than Class 1"
- Verificare mapping labels in `load_test_data()`
- Label mapping dovrebbe essere: DISCHARGED=0, ADMITTED=1

### Memory Error durante IG computation
- Ridurre `--batch_size` (es: 4 o 2)
- Ridurre `--n_steps` (es: 20)
- Usare `--n_samples` per analizzare subset

### CPU troppo lento
- Verificare CUDA disponibile: `nvidia-smi`
- Forzare GPU: `--device cuda`
- Ridurre `--n_samples` per test rapidi

## Integration with XAI.md

Questo modulo implementa la metodologia descritta in `docs/XAI.md`:

1. ✅ **Phase A**: Token-level attribution (Integrated Gradients)
2. ✅ **Phase B**: Word-level aggregation (sub-word → word)
3. ✅ **Phase C**: Action-level aggregation (word → clinical action)
4. ✅ **Visualization**: Heatmap + comparative histogram
5. ✅ **Class analysis**: Separation between Class 0 and Class 1

## Advanced Usage

### Custom Clinical Actions
```python
from src.explainability import ClinicalActionAggregator

custom_actions = {
    'neurological_exam': ['neurological', 'exam', 'examination', 'neurologic'],
    'cardiac_monitoring': ['cardiac', 'heart', 'monitoring', 'ecg', 'ekg']
}

aggregator = ClinicalActionAggregator(clinical_actions=custom_actions)
action_results = aggregator.aggregate_across_dataset(results, by_class=True)
```

### Load and Re-visualize
```python
import pickle
from src.explainability import plot_attention_heatmap, plot_class_comparison

# Load results
with open('output/explainability/ig_results_*.pkl', 'rb') as f:
    results = pickle.load(f)

# Extract top words
from src.explainability import IntegratedGradientsExplainer
explainer = IntegratedGradientsExplainer(model, tokenizer)
top_words = explainer.extract_top_words(results, top_k=30, by_class=True)

# Re-plot with different parameters
plot_attention_heatmap(
    top_words['class_0'], 
    top_words['class_1'], 
    top_k=30,
    save_path='custom_heatmap.png'
)
```

## Performance Notes

- **IG computation**: ~0.5-2 sec per sample (depends on batch_size, n_steps, device)
- **Full test set** (~1000 samples): ~15-30 min on GPU
- **Recommended**: Start with `--n_samples 100` for quick test

## References

- Integrated Gradients paper: https://arxiv.org/abs/1703.01365
- Captum library: https://captum.ai/
- LEGOLAS XAI methodology: `docs/XAI.md`
