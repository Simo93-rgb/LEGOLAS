# Test Rapido: Strategia Adattiva IG

## Quick Start

```bash
# Test con 10 samples (confronta fissa vs adattiva)
./test_adaptive_ig.sh
```

## Cosa Testa

1. **Strategia Fissa** (n_steps=1500):
   - Usa sempre 1500 steps per tutti i samples
   - Semplice e prevedibile
   - **Funziona per**: ensemble E single model
   
2. **Strategia Adattiva** (1000→2000):
   - Prova prima con 1000 steps
   - Se rel_error > 0.05 → ricalcola con 2000 steps
   - Risparmio tempo stimato: ~30%
   - **Funziona per**: ensemble E single model

## Output Atteso

```
📊 Adaptive strategy statistics:
   Started with 1000 steps: 10 samples
   Upgraded to 2000 steps: 3 samples (30.0%)
   Estimated time saved: ~70.0% vs fixed 2000 steps
```

## Comandi Manuali

### Test Strategia Fissa
```bash
uv run python src/explainability/extract_explainability.py \
    --model bert-base-uncased \
    --format narrativo \
    --use_ensemble \
    --n_samples 10 \
    --n_steps 1500 \
    --batch_size 200
```

### Test Strategia Adattiva
```bash
uv run python src/explainability/extract_explainability.py \
    --model bert-base-uncased \
    --format narrativo \
    --use_ensemble \
    --n_samples 10 \
    --adaptive_steps \
    --batch_size 200
```

## Per Dataset Completo (7393 samples)

**Raccomandato**: Strategia adattiva (risparmio ~30% tempo)

```bash
uv run python src/explainability/extract_explainability.py \
    --model bert-base-uncased \
    --format narrativo \
    --use_ensemble \
    --adaptive_steps \
    --batch_size 200
```

**Tempo stimato**: 20-30 minuti (dipende da quanti samples richiedono 2000 steps)

## File Output

- **IG Results**: `output/explainability/ig_results_narrativo_bert-base-uncased_ensemble_*.pkl`
- **Clinical Actions**: `output/explainability/actions_narrativo_bert-base-uncased_ensemble_*.pkl`
- **Visualizations**: 
  - `output/explainability/heatmap_actions_*.png`
  - `output/explainability/histogram_actions_*.png`

## Note

- Default n_steps cambiato: **50 → 1500** (basato su validazione)
- Soglia convergenza: **rel_error < 0.05** (5%)
- Strategia adattiva usa: **1000 → 2000** steps
- Baseline: **zero embeddings**

Vedi `docs/IG_STEPS_OPTIMIZATION.md` per dettagli completi.
