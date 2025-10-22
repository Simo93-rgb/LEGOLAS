#!/bin/bash
# Script helper per training e evaluation con storie XES
#
# FUNZIONALITÀ:
# - Scelta formato storie (narrativo/bullet/clinical)
# - Scelta modello (da YAML o legacy)
# - K-Fold Cross Validation (opzionale)
# - Focal Loss (opzionale, per classi sbilanciate)
# - Configurazione hyperparameters (epochs, patience, batch_size, lr)
# - Training + Evaluation automatici
#
# AGGIORNATO: 9 Ottobre 2025 - Supporto K-Fold e Focal Loss (FASE 4)
#
# USO: ./scripts/launch_run_train_eval
#      Menu interattivo guida attraverso tutte le opzioni
#
# DOCUMENTAZIONE: docs/TRAIN_LLM_INTEGRATION.md

# ============================================================
# PATH CONFIGURATION
# ============================================================
# NOTA: I path sono gestiti centralmente in src/config/paths.py
# Questi path devono corrispondere a quelli definiti in paths.py
STORIES_DIR="output/stories"
MODELS_DIR="output/models"
EVALUATION_DIR="output/evaluation"
PREDICTION_DIR="prediction"

# Colori
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   LEGOLAS - Training & Evaluation Pipeline${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}\n"

# Verifica che i file di storie esistano
if [ ! -d "${STORIES_DIR}" ] || [ ! -f "${STORIES_DIR}/narrativo_train.pkl" ]; then
    echo -e "${RED}❌ Errore: File di storie non trovati!${NC}"
    echo "   Esegui prima: ./scripts/run_xes_pipeline.sh"
    exit 1
fi

echo -e "${GREEN}✅ File di storie trovati in ${STORIES_DIR}/${NC}\n"

# Assicura che la directory evaluation esista
mkdir -p "${EVALUATION_DIR}"

# ============================================================
# MENU AZIONE (PRIMA DI TUTTO)
# ============================================================

echo -e "${YELLOW}Cosa vuoi fare?${NC}"
echo "  1) Solo training"
echo "  2) Solo evaluation (modello già addestrato)"
echo "  3) Training + Evaluation (completo)"
echo ""
read -p "Scelta [1-3, default=3]: " action

ACTION=${action:-3}

echo ""

# Menu scelta formato
echo -e "${YELLOW}Scegli il formato delle storie:${NC}"
echo "  1) narrativo (raccomandato per BERT)"
echo "  2) bullet (formato compatto)"
echo "  3) clinical (con token clinici - sperimentale)"
echo ""
read -p "Scelta [1-3, default=1]: " choice

case $choice in
    2) FORMAT="bullet" ;;
    3) FORMAT="clinical" ;;
    *) FORMAT="narrativo" ;;
esac

echo -e "\n${GREEN}📝 Formato selezionato: ${FORMAT}${NC}\n"

# Menu scelta modello
echo -e "\n${YELLOW}Scegli il modello da usare:${NC}"
echo ""

# Usa script Python per caricare modelli da YAML
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_MENU=$("$SCRIPT_DIR/list_models.py" menu 2>/dev/null)

if [ $? -eq 0 ] && [ -n "$MODEL_MENU" ]; then
    # Menu dinamico da YAML
    echo -e "${GREEN}📋 Modelli disponibili da config/model_configs.yaml:${NC}"
    echo ""
    
    # Parse e display menu
    declare -a MODEL_IDS
    declare -a MODEL_DESCS
    i=0
    
    while IFS='|' read -r num model_id desc; do
        MODEL_IDS[$i]="$model_id"
        MODEL_DESCS[$i]="$desc"
        echo "  $num) $model_id"
        echo "      └─ $desc"
        i=$((i+1))
    done <<< "$MODEL_MENU"
    
    MODEL_COUNT=$i
    echo ""
    read -p "Scelta [1-${MODEL_COUNT}, default=1]: " model_choice
    
    # Valida scelta
    if [[ "$model_choice" =~ ^[0-9]+$ ]] && [ "$model_choice" -ge 1 ] && [ "$model_choice" -le "$MODEL_COUNT" ]; then
        idx=$((model_choice - 1))
        MODEL="${MODEL_IDS[$idx]}"
    else
        # Default al primo
        MODEL="${MODEL_IDS[0]}"
    fi
    
    echo -e "\n${GREEN}🤖 Modello selezionato: ${MODEL}${NC}"
    
    # Mostra info modello
    MODEL_INFO=$("$SCRIPT_DIR/list_models.py" info "$MODEL" 2>/dev/null)
    if [ $? -eq 0 ]; then
        eval "$MODEL_INFO"
        echo -e "   HuggingFace: ${BLUE}${HF_ID}${NC}"
        echo -e "   Tipo: ${TYPE}"
        echo -e "   Batch consigliato: ${BATCH}"
        echo -e "   Learning rate: ${LR}"
    fi
    
else
    # Fallback a menu legacy
    echo -e "${YELLOW}⚠️  Config YAML non trovato, uso modelli legacy${NC}"
    echo ""
    echo "  1) bertm (BERT Medium - default)"
    echo "  2) roberta (RoBERTa Base)"
    echo "  3) cbert (Clinical BERT)"
    echo "  4) gpt2 (GPT-2)"
    echo ""
    read -p "Scelta [1-4, default=1]: " model_choice

    case $model_choice in
        2) MODEL="roberta" ;;
        3) MODEL="cbert" ;;
        4) MODEL="gpt2" ;;
        *) MODEL="bertm" ;;
    esac
    
    echo -e "\n${GREEN}🤖 Modello selezionato: ${MODEL}${NC}\n"
fi

# ============================================================
# PARAMETRI TRAINING (solo se action = 1 o 3)
# ============================================================

USE_KFOLD=""
N_FOLDS=5
USE_FOCAL=""
FOCAL_ALPHA="0.25 0.75"
FOCAL_GAMMA="2.0"
EPOCHS=10
PATIENCE=5
BATCH_SIZE=16
LR=5e-6

if [ "$ACTION" = "1" ] || [ "$ACTION" = "3" ]; then
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}   CONFIGURAZIONE TRAINING${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}\n"
    
    # Menu scelta modalità training
    echo -e "${YELLOW}Quale modalità di training vuoi usare?${NC}"
    echo "  1) Training semplice (train/val split)"
    echo "  2) K-Fold Cross Validation (più robusto)"
    echo ""
    read -p "Scelta [1-2, default=1]: " training_mode
    
    case $training_mode in
        2)
            USE_KFOLD="--use_kfold"
            echo ""
            read -p "Numero di folds [default=5]: " n_folds_input
            if [[ "$n_folds_input" =~ ^[0-9]+$ ]] && [ "$n_folds_input" -ge 2 ]; then
                N_FOLDS=$n_folds_input
            fi
            echo -e "${GREEN}🔄 K-Fold Cross Validation: ATTIVO${NC}"
            echo -e "${GREEN}   Folds: ${N_FOLDS}${NC}"
            ;;
        *)
            echo -e "${GREEN}📊 Training Semplice: ATTIVO${NC}"
            ;;
    esac
    
    # Menu Loss Function
    echo ""
    echo -e "${YELLOW}Quale Loss Function vuoi usare?${NC}"
    echo "  1) Cross Entropy (default, con class weights automatici)"
    echo "  2) Focal Loss (per classi molto sbilanciate)"
    echo ""
    read -p "Scelta [1-2, default=1]: " loss_choice
    
    case $loss_choice in
        2)
            USE_FOCAL="--use_focal_loss"
            echo ""
            echo -e "${YELLOW}Configurazione Focal Loss:${NC}"
            read -p "Alpha per classe 0 [default=0.25]: " alpha0
            read -p "Alpha per classe 1 [default=0.75]: " alpha1
            read -p "Gamma (focusing) [default=2.0]: " gamma
            
            # Valida input
            if [[ "$alpha0" =~ ^[0-9.]+$ ]]; then
                FOCAL_ALPHA="$alpha0 ${alpha1:-0.75}"
            fi
            if [[ "$gamma" =~ ^[0-9.]+$ ]]; then
                FOCAL_GAMMA="$gamma"
            fi
            
            echo -e "\n${GREEN}🎯 Focal Loss: α=[${FOCAL_ALPHA}], γ=${FOCAL_GAMMA}${NC}"
            ;;
        *)
            echo -e "\n${GREEN}📊 Cross Entropy con class weights automatici${NC}"
            ;;
    esac
    
    # Menu Hyperparameters
    echo ""
    echo -e "${YELLOW}Hyperparameters Training:${NC}"
    read -p "Numero epoche [default=10]: " epochs
    read -p "Patience early stopping [default=5]: " patience
    read -p "Batch size [default=16]: " batch_size
    read -p "Learning rate [default=5e-6]: " lr
    
    # Defaults
    EPOCHS=${epochs:-10}
    PATIENCE=${patience:-5}
    BATCH_SIZE=${batch_size:-16}
    LR=${lr:-5e-6}
    
    echo ""
    echo -e "${GREEN}✅ Configurazione Training:${NC}"
    if [ -n "$USE_KFOLD" ]; then
        echo "   Modalità: K-Fold Cross Validation"
        echo "   Folds: ${N_FOLDS}"
    else
        echo "   Modalità: Training Semplice"
    fi
    echo "   Epochs: ${EPOCHS}"
    echo "   Patience: ${PATIENCE}"
    echo "   Batch size: ${BATCH_SIZE}"
    echo "   Learning rate: ${LR}"
    
    # Costruisci comando training
    TRAIN_CMD="uv run python src/training/train_llm.py \
        --story_format ${FORMAT} \
        --model_name ${MODEL} \
        --epochs ${EPOCHS} \
        --patience ${PATIENCE} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LR}"
    
    # Aggiungi K-Fold se selezionato
    if [ -n "$USE_KFOLD" ]; then
        TRAIN_CMD="${TRAIN_CMD} ${USE_KFOLD} --n_folds ${N_FOLDS}"
    fi
    
    # Aggiungi opzioni Focal Loss
    if [ -n "$USE_FOCAL" ]; then
        TRAIN_CMD="${TRAIN_CMD} ${USE_FOCAL} --focal_alpha ${FOCAL_ALPHA} --focal_gamma ${FOCAL_GAMMA}"
    fi
    
    echo ""
fi

# ============================================================
# PARAMETRI EVALUATION (solo se action = 2 o 3)
# ============================================================

USE_ENSEMBLE=""

if [ "$ACTION" = "2" ] || [ "$ACTION" = "3" ]; then
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}   CONFIGURAZIONE EVALUATION${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}\n"
    
    echo -e "${YELLOW}Vuoi usare l'ensemble K-Fold per l'evaluation?${NC}"
    echo "  1) No - Solo best fold (più veloce)"
    echo "  2) Sì - Ensemble di tutti i fold (più accurato)"
    echo ""
    read -p "Scelta [1-2, default=2]: " ensemble_choice
    
    case $ensemble_choice in
        1)
            echo -e "\n${GREEN}📊 Evaluation: Best fold only${NC}"
            ;;
        *)
            USE_ENSEMBLE="--use_ensemble"
            echo -e "\n${GREEN}🔄 Evaluation: K-Fold Ensemble${NC}"
            ;;
    esac
    
    echo ""
fi

# ============================================================
# ESECUZIONE
# ============================================================

case $ACTION in
    1)
        echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}   TRAINING${NC}"
        echo -e "${BLUE}════════════════════════════════════════════════════════${NC}\n"
        echo -e "${YELLOW}Comando: ${TRAIN_CMD}${NC}\n"
        eval ${TRAIN_CMD}
        ;;
    2)
        echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}   EVALUATION${NC}"
        echo -e "${BLUE}════════════════════════════════════════════════════════${NC}\n"
        
        # Genera timestamp per file unico
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        ENSEMBLE_SUFFIX=""
        if [ -n "$USE_ENSEMBLE" ]; then
            ENSEMBLE_SUFFIX="_ensemble"
        fi
        EVAL_LOG="${EVALUATION_DIR}/eval_${FORMAT}_${MODEL}${ENSEMBLE_SUFFIX}_${TIMESTAMP}.log"
        
        echo -e "${BLUE}📝 Output salvato in: ${EVAL_LOG}${NC}\n"
        
        # Costruisci comando evaluation
        EVAL_CMD="uv run python src/training/eval_model.py --story_format ${FORMAT} --model_name ${MODEL}"
        if [ -n "$USE_ENSEMBLE" ]; then
            EVAL_CMD="${EVAL_CMD} ${USE_ENSEMBLE}"
        fi
        
        echo -e "${YELLOW}Comando: ${EVAL_CMD}${NC}\n"
        
        # Esegui e salva output su file (con tee per mostrare anche a schermo)
        eval ${EVAL_CMD} 2>&1 | tee "${EVAL_LOG}"
        
        EVAL_EXIT_CODE=${PIPESTATUS[0]}
        
        if [ $EVAL_EXIT_CODE -eq 0 ]; then
            echo -e "\n${GREEN}✅ Log salvato in: ${EVAL_LOG}${NC}"
        else
            echo -e "\n${RED}❌ Errore durante evaluation (log: ${EVAL_LOG})${NC}"
        fi
        ;;
    *)
        echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}   TRAINING${NC}"
        echo -e "${BLUE}════════════════════════════════════════════════════════${NC}\n"
        echo -e "${YELLOW}Comando: ${TRAIN_CMD}${NC}\n"
        eval ${TRAIN_CMD}
        
        if [ $? -eq 0 ]; then
            echo -e "\n${GREEN}✅ Training completato!${NC}\n"
            echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
            echo -e "${BLUE}   EVALUATION${NC}"
            echo -e "${BLUE}════════════════════════════════════════════════════════${NC}\n"
            
            # Genera timestamp per file unico
            TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
            ENSEMBLE_SUFFIX=""
            if [ -n "$USE_ENSEMBLE" ]; then
                ENSEMBLE_SUFFIX="_ensemble"
            fi
            EVAL_LOG="${EVALUATION_DIR}/eval_${FORMAT}_${MODEL}${ENSEMBLE_SUFFIX}_${TIMESTAMP}.log"
            
            echo -e "${BLUE}📝 Output salvato in: ${EVAL_LOG}${NC}\n"
            
            # Costruisci comando evaluation
            EVAL_CMD="uv run python src/training/eval_model.py --story_format ${FORMAT} --model_name ${MODEL}"
            if [ -n "$USE_ENSEMBLE" ]; then
                EVAL_CMD="${EVAL_CMD} ${USE_ENSEMBLE}"
            fi
            
            echo -e "${YELLOW}Comando: ${EVAL_CMD}${NC}\n"
            
            # Esegui e salva output su file (con tee per mostrare anche a schermo)
            eval ${EVAL_CMD} 2>&1 | tee "${EVAL_LOG}"
            
            EVAL_EXIT_CODE=${PIPESTATUS[0]}
            
            if [ $EVAL_EXIT_CODE -eq 0 ]; then
                echo -e "\n${GREEN}✅ Evaluation completato!${NC}"
                echo -e "📄 Log salvato in: ${GREEN}${EVAL_LOG}${NC}"
                echo -e "📊 Risultati salvati in: ${GREEN}${PREDICTION_DIR}/xes_${FORMAT}_${MODEL}${ENSEMBLE_SUFFIX}_*${NC}"
            else
                echo -e "\n${RED}❌ Errore durante evaluation (log: ${EVAL_LOG})${NC}"
            fi
        else
            echo -e "\n${RED}❌ Errore durante il training${NC}"
            exit 1
        fi
        ;;
esac

echo -e "\n${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Operazione completata!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}\n"

# Mostra file generati
if [ -d "${PREDICTION_DIR}" ]; then
    echo "📁 File generati:"
    ls -lh ${PREDICTION_DIR}/xes_${FORMAT}_${MODEL}_* 2>/dev/null || echo "   (nessun file ancora)"
    echo ""
fi

echo -e "${YELLOW}💡 Suggerimenti:${NC}"

# Mostra path diversi a seconda della modalità
if [ -n "$USE_KFOLD" ]; then
    echo "   - Modelli salvati (${N_FOLDS} fold): ${MODELS_DIR}/best_model_${FORMAT}_${MODEL}_fold*.pth"
    echo "   - Risultati aggregati: output/reports/kfold_aggregated_${FORMAT}_${MODEL}_results.json"
    echo "   - Metriche per fold: output/reports/fold_*_${FORMAT}_${MODEL}_metrics.json"
else
    echo "   - Modello salvato: ${MODELS_DIR}/best_model_${FORMAT}_${MODEL}.pth"
    echo "   - Training history: ${MODELS_DIR}/training_history_${FORMAT}_${MODEL}.json"
fi
echo "   - Label mapping: output/reports/label_mapping.json"
echo "   - Report evaluation: ${PREDICTION_DIR}/xes_${FORMAT}_${MODEL}_report.txt"
echo "   - Log evaluation: ${EVALUATION_DIR}/eval_${FORMAT}_${MODEL}_*.log"
echo ""
echo -e "${YELLOW}📚 Documentazione:${NC}"
echo "   - Guida completa: docs/TRAIN_LLM_INTEGRATION.md"
echo "   - Path management: src/config/paths.py"
echo ""
echo -e "${YELLOW}🔄 Per rilanciare:${NC}"
echo "   - Stesso script: ./scripts/launch_run_train_eval"
echo "   - CLI diretta: uv run python src/training/train_llm.py --help"
