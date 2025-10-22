#!/bin/bash
#############################################
# LEGOLAS - Explainability Extraction Launcher
# Estrae attribution scores usando Integrated Gradients
#############################################

set -e

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   LEGOLAS - Explainability Extraction (Integrated Gradients)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 1. Select story format
echo -e "${YELLOW}📖 Story Format Selection:${NC}"
echo "   1) narrativo   - Natural narrative style (default)"
echo "   2) bullet      - Bullet point format"
echo "   3) clinical    - Clinical documentation style"
echo ""
read -p "Select format [1-3] (default: 1): " FORMAT_CHOICE
FORMAT_CHOICE=${FORMAT_CHOICE:-1}

case $FORMAT_CHOICE in
    1) STORY_FORMAT="narrativo" ;;
    2) STORY_FORMAT="bullet" ;;
    3) STORY_FORMAT="clinical" ;;
    *) 
        echo -e "${RED}Invalid choice, using narrativo${NC}"
        STORY_FORMAT="narrativo"
        ;;
esac

echo -e "${GREEN}✓ Selected format: ${STORY_FORMAT}${NC}"
echo ""

# 2. Select model
echo -e "${YELLOW}🤖 Model Selection:${NC}"
echo "   1) bert-base-uncased       - BERT Base (default)"
echo "   2) clinical-bert           - Clinical BERT (UMCUtrecht)"
echo "   3) clinical-modernbert     - Modern Clinical BERT"
echo "   4) pubmedbert-base         - PubMed BERT Base"
echo "   5) biobert-base            - BioBERT Base"
echo "   6) cambridgeltl-sapbert    - SapBERT"
echo "   7) medbert-base            - MedBERT Base"
echo "   8) scibert-base            - SciBERT Base"
echo "   9) clinical-longformer     - Clinical Longformer"
echo "   10) bert-large-uncased     - BERT Large"
echo ""
read -p "Select model [1-10] (default: 1): " MODEL_CHOICE
MODEL_CHOICE=${MODEL_CHOICE:-1}

case $MODEL_CHOICE in
    1) MODEL_NAME="bert-base-uncased" ;;
    2) MODEL_NAME="clinical-bert" ;;
    3) MODEL_NAME="clinical-modernbert" ;;
    4) MODEL_NAME="pubmedbert-base" ;;
    5) MODEL_NAME="biobert-base" ;;
    6) MODEL_NAME="cambridgeltl-sapbert" ;;
    7) MODEL_NAME="medbert-base" ;;
    8) MODEL_NAME="scibert-base" ;;
    9) MODEL_NAME="clinical-longformer" ;;
    10) MODEL_NAME="bert-large-uncased" ;;
    *) 
        echo -e "${RED}Invalid choice, using bert-base-uncased${NC}"
        MODEL_NAME="bert-base-uncased"
        ;;
esac

echo -e "${GREEN}✓ Selected model: ${MODEL_NAME}${NC}"
echo ""

# 3. Dataset selection
echo -e "${YELLOW}📊 Dataset Selection:${NC}"
echo "   1) test   - Test set only (default)"
echo "   2) train  - Training set only"
echo "   3) all    - Both train + test"
echo ""
read -p "Select dataset [1-3] (default: 1): " DATASET_CHOICE
DATASET_CHOICE=${DATASET_CHOICE:-1}

case $DATASET_CHOICE in
    1) DATASET="test" ;;
    2) DATASET="train" ;;
    3) DATASET="all" ;;
    *) 
        echo -e "${RED}Invalid choice, using test${NC}"
        DATASET="test"
        ;;
esac

echo -e "${GREEN}✓ Selected dataset: ${DATASET}${NC}"
echo ""

# 4. Advanced parameters
echo -e "${YELLOW}⚙️  Advanced Parameters:${NC}"
read -p "Number of samples to analyze (press Enter for all ${DATASET} set): " N_SAMPLES
read -p "Top-K words to visualize (default: 20): " TOP_K
TOP_K=${TOP_K:-20}
read -p "Internal batch size for IG interpolation (default: 32): " INTERNAL_BATCH_SIZE
INTERNAL_BATCH_SIZE=${INTERNAL_BATCH_SIZE:-32}
read -p "Integrated Gradients steps (default: 1500): " N_STEPS
N_STEPS=${N_STEPS:-1500}

echo ""
echo -e "${GREEN}✓ Configuration complete${NC}"
echo ""

# 5. Adaptive steps strategy
echo -e "${YELLOW}🔄 Adaptive Steps Strategy:${NC}"
echo "   Use adaptive n_steps (start 1500, increase to 5500 if needed)?"
read -p "Enable adaptive steps? [y/N] (default: N): " ADAPTIVE_CHOICE
ADAPTIVE_CHOICE=${ADAPTIVE_CHOICE:-N}

if [[ "$ADAPTIVE_CHOICE" =~ ^[Yy]$ ]]; then
    USE_ADAPTIVE="--adaptive_steps"
    echo -e "${GREEN}✓ Adaptive steps enabled (1500→5500, tolerance=0.05)${NC}"
else
    USE_ADAPTIVE=""
    echo -e "${GREEN}✓ Fixed steps: ${N_STEPS}${NC}"
fi

echo ""

# 6. Ensemble mode
echo -e "${YELLOW}🎯 Model Inference Mode:${NC}"
echo "   Use K-Fold ensemble (average attributions across folds)?"
read -p "Enable ensemble mode? [y/N] (default: N, uses best fold): " ENSEMBLE_CHOICE
ENSEMBLE_CHOICE=${ENSEMBLE_CHOICE:-N}

if [[ "$ENSEMBLE_CHOICE" =~ ^[Yy]$ ]]; then
    USE_ENSEMBLE="--use_ensemble"
    echo -e "${GREEN}✓ Ensemble mode enabled (K-Fold averaging)${NC}"
else
    USE_ENSEMBLE=""
    echo -e "${GREEN}✓ Best fold mode (single model)${NC}"
fi

echo ""

# 7. Build command
CMD="uv run python -m src.explainability.extract_explainability"
CMD="$CMD --model $MODEL_NAME"
CMD="$CMD --format $STORY_FORMAT"
CMD="$CMD --dataset $DATASET"
CMD="$CMD --top_k $TOP_K"
CMD="$CMD --internal_batch_size $INTERNAL_BATCH_SIZE"
CMD="$CMD --n_steps $N_STEPS"

if [ ! -z "$N_SAMPLES" ]; then
    CMD="$CMD --n_samples $N_SAMPLES"
fi

if [ ! -z "$USE_ADAPTIVE" ]; then
    CMD="$CMD $USE_ADAPTIVE"
fi

if [ ! -z "$USE_ENSEMBLE" ]; then
    CMD="$CMD $USE_ENSEMBLE"
fi

# Auto-detect GPU
if command -v nvidia-smi &> /dev/null; then
    CMD="$CMD --device cuda"
    echo -e "${GREEN}✓ GPU detected, using CUDA${NC}"
else
    CMD="$CMD --device cpu"
    echo -e "${YELLOW}⚠️  GPU not detected, using CPU (will be slower)${NC}"
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Starting Explainability Extraction${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Command:${NC}"
echo "$CMD"
echo ""

# Confirm execution
read -p "Press Enter to start, or Ctrl+C to cancel..."

# 8. Execute
echo ""
eval $CMD

# Exit status
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}   ✅ Explainability extraction completed successfully!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
else
    echo -e "${RED}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}   ❌ Explainability extraction failed with exit code $EXIT_CODE${NC}"
    echo -e "${RED}═══════════════════════════════════════════════════════════════════════${NC}"
    exit $EXIT_CODE
fi
