# Explainability Module

Integrated Gradients explainability system for LEGOLAS.

## Quick Start

```bash
bash scripts/run_explainability.sh
```

## Module Structure

```
src/explainability/
├── __init__.py                    # Module exports
├── extract_explainability.py      # Main execution script
├── integrated_gradients.py        # IG attribution extraction
├── action_aggregator.py           # Clinical action aggregation
└── visualization.py               # Plotting functions
```

## Main Components

### 1. IntegratedGradientsExplainer
Extracts token-level attributions using Captum's Integrated Gradients.

**Key methods**:
- `explain_text(text, label)` - Single text analysis
- `explain_batch(texts, labels)` - Batch processing
- `aggregate_subword_attributions()` - BERT sub-word → word
- `extract_top_words()` - Top-K extraction per class

### 2. ClinicalActionAggregator
Aggregates word-level scores to clinical action level.

**Key methods**:
- `find_actions_in_text()` - Match actions in text
- `aggregate_across_dataset()` - Compute action statistics
- `get_top_actions()` - Top-K actions per class

### 3. Visualization Functions
Creates two required plots:

**Heatmap** (`plot_attention_heatmap`):
- 2 rows: Class 0, Class 1
- Top-25 words ordered by Class 0 importance
- Global normalization (max=1)
- YlOrRd colormap

**Histogram** (`plot_class_comparison`):
- Side-by-side bars (blue=Class 0, red=Class 1)
- Separate normalization per class
- Top-25 words ordered by Class 0 importance

## Usage Examples

### Via Bash Script
```bash
bash scripts/run_explainability.sh
# Interactive menu for format/model selection
```

### Via Python
```bash
python -m src.explainability.extract_explainability \
    --model clinical-bert \
    --format narrativo \
    --top_k 25 \
    --batch_size 8
```

### Programmatic
```python
from src.explainability import IntegratedGradientsExplainer

# Initialize
explainer = IntegratedGradientsExplainer(model, tokenizer, device='cuda')

# Analyze single text
result = explainer.explain_text(
    text="Patient admitted for psychiatric evaluation...",
    label=1,
    n_steps=50
)

# Top attributed words
print(result['word_attributions'][:10])
```

## Output Files

All files saved to `output/explainability/`:
- `ig_results_*.pkl` - Raw attribution scores
- `actions_*.pkl` - Clinical action statistics
- `heatmap_*.png` - Attribution heatmap
- `histogram_*.png` - Class comparison bars

## Documentation

See `docs/EXPLAINABILITY_GUIDE.md` for complete documentation.

## Requirements

```
torch
transformers
captum
numpy
matplotlib
seaborn
tqdm
```

Install via:
```bash
pip install captum matplotlib seaborn
```
(other dependencies already in LEGOLAS)
