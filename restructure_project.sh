#!/bin/bash
# Script per ristrutturare il progetto LEGOLAS in modo organizzato

set -e  # Exit on error

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   LEGOLAS - Ristrutturazione Progetto${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}\n"

# Conferma
echo -e "${YELLOW}⚠️  Questo script ristrutturerà il progetto.${NC}"
echo "   Verrà creato un backup prima di procedere."
echo ""
read -p "Vuoi continuare? [y/N]: " confirm

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Operazione annullata."
    exit 0
fi

echo ""
echo -e "${GREEN}📦 Creazione backup...${NC}"
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r . "$BACKUP_DIR/" 2>/dev/null || true
echo -e "   ✓ Backup creato in: $BACKUP_DIR"

echo ""
echo -e "${GREEN}📁 Creazione struttura directory...${NC}"

# Crea directory principali
mkdir -p docs
mkdir -p scripts
mkdir -p src/{models,data,generation,training}
mkdir -p data/raw
mkdir -p output/{stories,models}
mkdir -p legacy

echo "   ✓ Directory create"

echo ""
echo -e "${GREEN}📚 Spostamento documentazione...${NC}"

# Sposta documentazione
mv INTEGRATION_GUIDE.md docs/ 2>/dev/null || true
mv FLOW_DIAGRAM.md docs/ 2>/dev/null || true
mv TRAINING_EVALUATION_GUIDE.md docs/ 2>/dev/null || true
mv INTEGRATION_SUMMARY.md docs/ 2>/dev/null || true
mv NEXT_STEPS.md docs/ 2>/dev/null || true
mv INDEX.md docs/ 2>/dev/null || true
mv INTEGRATION_COMPLETE.txt docs/ 2>/dev/null || true
mv TRAINING_COMPLETE.txt docs/ 2>/dev/null || true
mv QUICK_START.txt docs/ 2>/dev/null || true
mv README_INTEGRATION.md docs/ 2>/dev/null || true
mv PROJECT_STRUCTURE.md docs/ 2>/dev/null || true

echo "   ✓ Documentazione spostata in docs/"

echo ""
echo -e "${GREEN}🚀 Spostamento script...${NC}"

# Sposta script
mv generate_stories.py scripts/ 2>/dev/null || true
mv train_xes_model.py scripts/ 2>/dev/null || true
mv test_integration.py scripts/ 2>/dev/null || true
mv run_xes_pipeline.sh scripts/ 2>/dev/null || true
mv launch_run_train_eval scripts/ 2>/dev/null || true

echo "   ✓ Script spostati in scripts/"

echo ""
echo -e "${GREEN}💾 Spostamento codice sorgente...${NC}"

# Sposta modelli
mv neural_network.py src/models/ 2>/dev/null || true
mv model_config.yaml src/models/ 2>/dev/null || true

# Sposta data processing
mv xes_parser.py src/data/ 2>/dev/null || true
mv history_dataset.py src/data/ 2>/dev/null || true
mv clinical_token_mapper.py src/data/ 2>/dev/null || true

# Sposta generazione
mv story_generator.py src/generation/ 2>/dev/null || true
mv skeleton.py src/generation/ 2>/dev/null || true

# Sposta training
mv train_llm.py src/training/ 2>/dev/null || true
mv eval_model.py src/training/ 2>/dev/null || true

echo "   ✓ Codice spostato in src/"

echo ""
echo -e "${GREEN}📊 Spostamento dati...${NC}"

# Sposta dati
mv translation_cache.json data/ 2>/dev/null || true
mv *.xes data/raw/ 2>/dev/null || true

echo "   ✓ Dati spostati in data/"

echo ""
echo -e "${GREEN}📤 Riorganizzazione output...${NC}"

# Sposta output storie
mv output/*.pkl output/stories/ 2>/dev/null || true

# Sposta modelli addestrati
mv *.pth output/models/ 2>/dev/null || true

echo "   ✓ Output riorganizzato"

echo ""
echo -e "${GREEN}📜 Spostamento codice legacy...${NC}"

# Sposta legacy
mv main.py legacy/ 2>/dev/null || true
mv launch_run_single legacy/ 2>/dev/null || true
mv launch_run_single_exe legacy/ 2>/dev/null || true
mv launch_run_eval legacy/ 2>/dev/null || true
mv launch_run_eval_exe legacy/ 2>/dev/null || true

echo "   ✓ Legacy spostato"

echo ""
echo -e "${GREEN}🔧 Creazione file __init__.py...${NC}"

# Crea __init__.py
touch src/__init__.py
touch src/models/__init__.py
touch src/data/__init__.py
touch src/generation/__init__.py
touch src/training/__init__.py

echo "   ✓ File __init__.py creati"

echo ""
echo -e "${GREEN}📝 Creazione .gitkeep per directory vuote...${NC}"

touch data/raw/.gitkeep

echo "   ✓ .gitkeep creati"

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Ristrutturazione completata!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}\n"

echo "📁 Nuova struttura:"
tree -L 2 -d 2>/dev/null || find . -type d -name ".venv" -prune -o -type d -print | grep -v "\.venv\|\.git\|__pycache__\|$BACKUP_DIR" | head -30

echo ""
echo -e "${YELLOW}⚠️  IMPORTANTE: Aggiorna gli import nei file!${NC}"
echo ""
echo "Esempi di modifiche necessarie:"
echo "  Prima:  from xes_parser import XESParser"
echo "  Dopo:   from src.data.xes_parser import XESParser"
echo ""
echo "  Prima:  from story_generator import StoryGenerator"
echo "  Dopo:   from src.generation.story_generator import StoryGenerator"
echo ""
echo "Consulta: docs/PROJECT_STRUCTURE.md per dettagli"
