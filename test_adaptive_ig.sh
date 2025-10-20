#!/bin/bash
# Test rapido della strategia adattiva con 10 samples

echo "=========================================="
echo "Test 1: Strategia FISSA (n_steps=1500)"
echo "=========================================="
uv run python src/explainability/extract_explainability.py \
    --model bert-base-uncased \
    --format narrativo \
    --use_ensemble \
    --n_samples 50 \
    --n_steps 1500 \
    --batch_size 32

echo ""
echo "=========================================="
echo "Test 2: Strategia ADATTIVA Ensemble (1000→2000)"
echo "=========================================="
uv run python src/explainability/extract_explainability.py \
    --model bert-base-uncased \
    --format narrativo \
    --use_ensemble \
    --n_samples 50 \
    --adaptive_steps \
    --batch_size 32

echo ""
echo "=========================================="
echo "Test 3: Strategia FISSA Single Model (n_steps=1500)"
echo "=========================================="
uv run python src/explainability/extract_explainability.py \
    --model bert-base-uncased \
    --format narrativo \
    --n_samples 50 \
    --n_steps 1500 \
    --batch_size 1

echo ""
echo "=========================================="
echo "Test 4: Strategia ADATTIVA Single Model (1000→2000)"
echo "=========================================="
uv run python src/explainability/extract_explainability.py \
    --model bert-base-uncased \
    --format narrativo \
    --n_samples 50 \
    --adaptive_steps \
    --batch_size 1

echo ""
echo "✅ Test completati! Confronta i tempi di esecuzione tra:"
echo "   - Ensemble vs Single model"
echo "   - Strategia Fissa vs Adattiva"
