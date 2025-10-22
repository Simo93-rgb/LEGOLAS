#!/bin/bash
# Script di esempio per generare storie dal file XES del progetto

# Colori per output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   LEGOLAS - Pipeline XES Story Generation${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}\n"

# Default file XES
DEFAULT_XES="data/raw/ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025.xes"

# Chiedi all'utente il path del file XES
echo -e "${YELLOW}Inserisci il path del file XES da processare:${NC}"
echo -e "  (Premi INVIO per usare il default: ${BLUE}${DEFAULT_XES}${NC})"
echo ""
read -p "Path file XES: " user_xes_file

# Usa default se l'utente non inserisce nulla
if [ -z "$user_xes_file" ]; then
    XES_FILE="$DEFAULT_XES"
    echo -e "\n${GREEN}📁 Usando file default${NC}"
else
    XES_FILE="$user_xes_file"
    echo -e "\n${GREEN}📁 Usando file specificato${NC}"
fi

echo -e "   File: ${BLUE}${XES_FILE}${NC}\n"

# Verifica che il file esista
if [ ! -f "$XES_FILE" ]; then
    echo -e "${YELLOW}❌ Errore: File XES non trovato: $XES_FILE${NC}"
    echo "   Verifica il percorso e riprova."
    exit 1
fi

echo -e "${GREEN}✅ File XES trovato e verificato${NC}"
echo -e "   Size: $(du -h "$XES_FILE" | cut -f1)${NC}\n"

# Scenario 1: Formato narrativo (raccomandato per BERT)
echo -e "${BLUE}📝 Scenario 1: Generazione formato NARRATIVO${NC}"
echo "   (Raccomandato per modelli BERT-like)"
uv run python scripts/generate_stories.py \
    --pipeline xes \
    --input "$XES_FILE" \
    --output-prefix narrativo \
    --format narrative \
    --test-size 0.15 \
    --seed 42

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ Generazione narrativa completata!${NC}"
    echo -e "   File generati in: ${GREEN}output/narrativo_*.pkl${NC}\n"
else
    echo -e "\n${YELLOW}⚠️  Errore nella generazione narrativa${NC}\n"
fi

# # Scenario 2: Formato bullet points
# echo -e "${BLUE}📝 Scenario 2: Generazione formato BULLET POINTS${NC}"
# echo "   (Alternativa più compatta)"
# uv run python scripts/generate_stories.py \
#     --pipeline xes \
#     --input "$XES_FILE" \
#     --output-prefix bullet \
#     --format bullet_points \
#     --test-size 0.15 \
#     --seed 42

# if [ $? -eq 0 ]; then
#     echo -e "\n${GREEN}✅ Generazione bullet points completata!${NC}"
#     echo -e "   File generati in: ${GREEN}output/bullet_*.pkl${NC}\n"
# else
#     echo -e "\n${YELLOW}⚠️  Errore nella generazione bullet points${NC}\n"
# fi

# # Scenario 3: Con token clinici (SPERIMENTALE)
# echo -e "${BLUE}📝 Scenario 3: Generazione con TOKEN CLINICI${NC}"
# echo "   (SPERIMENTALE - per analisi XAI)"
# uv run python scripts/generate_stories.py \
#     --pipeline xes \
#     --input "$XES_FILE" \
#     --output-prefix clinical \
#     --format narrative \
#     --clinical-tokens \
#     --test-size 0.15 \
#     --seed 42

# if [ $? -eq 0 ]; then
#     echo -e "\n${GREEN}✅ Generazione con token clinici completata!${NC}"
#     echo -e "   File generati in: ${GREEN}output/clinical_*.pkl${NC}\n"
# else
#     echo -e "\n${YELLOW}⚠️  Errore nella generazione con token clinici${NC}\n"
# fi

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Pipeline completata!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}\n"

echo "📊 File generati:"
ls -lh output/*.pkl 2>/dev/null || echo "   Nessun file generato"

echo -e "\n📚 Prossimi passi:"
echo "   1. Modifica train_llm.py per usare i file generati"
echo "   2. Esegui: uv run python train_llm.py"
echo "   3. Valuta i risultati con: uv run python eval_model.py"
