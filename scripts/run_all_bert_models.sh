#!/bin/bash

# Script per addestrare modelli LLM in sequenza con logging su file e terminale

MODELS=(
  "biobert-base"
  "bluebert-base"
  "clinical-bert"
  "clinicalbert-base"
  "pubmedbert-base"
  "bert-large-uncased"
  "scibert-base"
)

LOG_DIR="/home/simon/GitHub/LEGOLAS/output/logs"
mkdir -p "$LOG_DIR"  # crea la cartella se non esiste

for MODEL in "${MODELS[@]}"; do
  if [ "$MODEL" == "bert-large-uncased" ]; then
    BATCH_SIZE=24
  else
    BATCH_SIZE=32
  fi

  LOG_FILE="$LOG_DIR/kfold_training_${MODEL}.txt"

  echo "▶️ Avvio addestramento per il modello: $MODEL con batch_size=$BATCH_SIZE"
  echo "📁 Log: $LOG_FILE"
  echo "----------------------------------------"

  uv run python src/training/train_llm.py \
    --story_format narrativo \
    --model "$MODEL" \
    --epochs 50 \
    --patience 5 \
    --batch_size "$BATCH_SIZE" \
    --learning_rate 5e-6 \
    --use_kfold \
    --n_folds 5 \
    --use_focal_loss \
    --focal_alpha 0.25 0.75 \
    --focal_gamma 2.0 | tee "$LOG_FILE"

  echo "✅ Completato: $MODEL"
  echo "========================================"
done

echo "🎉 Tutti i modelli sono stati addestrati."