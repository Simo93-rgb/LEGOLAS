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
echo "   1) narrativo   - Natural narrative style"
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
echo "   1) clinical-bert           - Clinical BERT (UMCUtrecht)"
echo "   2) clinical-modernbert     - Modern Clinical BERT"
echo "   3) pubmedbert-base         - PubMed BERT Base"
echo "   4) biobert-base            - BioBERT Base"
echo "   5) cambridgeltl-sapbert    - SapBERT"
echo "   6) medbert-base            - MedBERT Base"
echo "   7) scibert-base            - SciBERT Base"
echo "   8) clinical-longformer     - Clinical Longformer"
echo "   9) bert-base-uncased       - BERT Base"
echo "   10) bert-large-uncased     - BERT Large"
echo ""
read -p "Select model [1-10] (default: 1): " MODEL_CHOICE
MODEL_CHOICE=${MODEL_CHOICE:-1}

case $MODEL_CHOICE in
    1) MODEL_NAME="clinical-bert" ;;
    2) MODEL_NAME="clinical-modernbert" ;;
    3) MODEL_NAME="pubmedbert-base" ;;
    4) MODEL_NAME="biobert-base" ;;
    5) MODEL_NAME="cambridgeltl-sapbert" ;;
    6) MODEL_NAME="medbert-base" ;;
    7) MODEL_NAME="scibert-base" ;;
    8) MODEL_NAME="clinical-longformer" ;;
    9) MODEL_NAME="bert-base-uncased" ;;
    10) MODEL_NAME="bert-large-uncased" ;;
    *) 
        echo -e "${RED}Invalid choice, using bert-base-uncased${NC}"
        MODEL_NAME="bert-base-uncased"
        ;;
esac

echo -e "${GREEN}✓ Selected model: ${MODEL_NAME}${NC}"
echo ""

# 3. Advanced parameters
echo -e "${YELLOW}⚙️  Advanced Parameters:${NC}"
read -p "Number of samples to analyze (press Enter for all test set): " N_SAMPLES
read -p "Top-K words to visualize (default: 25): " TOP_K
TOP_K=${TOP_K:-25}
read -p "Batch size (default: 8): " BATCH_SIZE
BATCH_SIZE=${BATCH_SIZE:-8}
read -p "Integrated Gradients steps (default: 50): " N_STEPS
N_STEPS=${N_STEPS:-50}

echo ""
echo -e "${GREEN}✓ Configuration complete${NC}"
echo ""

# 4. Build command
CMD="uv run python -m src.explainability.extract_explainability"
CMD="$CMD --model $MODEL_NAME"
CMD="$CMD --format $STORY_FORMAT"
CMD="$CMD --top_k $TOP_K"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --n_steps $N_STEPS"

if [ ! -z "$N_SAMPLES" ]; then
    CMD="$CMD --n_samples $N_SAMPLES"
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

# 5. Execute
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
