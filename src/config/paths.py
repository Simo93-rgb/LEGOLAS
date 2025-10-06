"""
Path Configuration per LEGOLAS
Centralizza tutti i path del progetto per facile manutenzione
"""

from pathlib import Path

# Root del progetto (assumendo che questo file sia in src/config/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Directory principali
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CONFIG_DIR = PROJECT_ROOT / "config"
DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Subdirectory output
STORIES_DIR = OUTPUT_DIR / "stories"
MODELS_DIR = OUTPUT_DIR / "models"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"
PREDICTION_DIR = PROJECT_ROOT / "prediction"

# File configurazione
MODEL_CONFIG_FILE = CONFIG_DIR / "model_configs.yaml"
TRANSLATION_CACHE_FILE = DATA_DIR / "translation_cache.json"

# XES data
XES_DATA_DIR = DATA_DIR / "raw"


def ensure_directories():
    """Crea tutte le directory necessarie se non esistono"""
    dirs = [
        DATA_DIR,
        OUTPUT_DIR,
        STORIES_DIR,
        MODELS_DIR,
        EVALUATION_DIR,
        PREDICTION_DIR,
        CONFIG_DIR,
        XES_DATA_DIR,
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_story_file_path(story_format: str, file_type: str) -> Path:
    """
    Genera path per file di storie
    
    Args:
        story_format: Formato storie ('narrativo', 'bullet', 'clinical')
        file_type: Tipo file ('train', 'test', 'label_train', 'label_test')
    
    Returns:
        Path completo al file
    """
    return STORIES_DIR / f"{story_format}_{file_type}.pkl"


def get_model_path(story_format: str, model_name: str, epoch: int = None) -> Path:
    """
    Genera path per file modello
    
    Args:
        story_format: Formato storie usato nel training
        model_name: Nome del modello
        epoch: Numero epoca (opzionale)
    
    Returns:
        Path completo al file modello
    """
    base_name = f"xes_{story_format}_{model_name}"
    if epoch is not None:
        return MODELS_DIR / f"{base_name}{epoch}.pth"
    return MODELS_DIR / f"{base_name}.pth"


def get_prediction_path(story_format: str, model_name: str, file_type: str) -> Path:
    """
    Genera path per file predizioni
    
    Args:
        story_format: Formato storie
        model_name: Nome modello
        file_type: Tipo ('prob', 'all_target', 'all_prediction', 'report')
    
    Returns:
        Path completo al file
    """
    prefix = f"xes_{story_format}_{model_name}"
    
    if file_type == 'report':
        return PREDICTION_DIR / f"{prefix}_report.txt"
    else:
        return PREDICTION_DIR / f"{prefix}_{file_type}.pkl"


# Utility per retrocompatibilità con path relativi
def to_str(path: Path) -> str:
    """Converte Path a stringa relativa al project root se possibile"""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    # Test configurazione path
    print("=" * 60)
    print("LEGOLAS - Path Configuration")
    print("=" * 60)
    
    print(f"\n📁 Directory principali:")
    print(f"   PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"   DATA_DIR: {to_str(DATA_DIR)}")
    print(f"   OUTPUT_DIR: {to_str(OUTPUT_DIR)}")
    print(f"   STORIES_DIR: {to_str(STORIES_DIR)}")
    print(f"   MODELS_DIR: {to_str(MODELS_DIR)}")
    print(f"   EVALUATION_DIR: {to_str(EVALUATION_DIR)}")
    print(f"   PREDICTION_DIR: {to_str(PREDICTION_DIR)}")
    
    print(f"\n📄 File configurazione:")
    print(f"   MODEL_CONFIG: {to_str(MODEL_CONFIG_FILE)}")
    print(f"   TRANSLATION_CACHE: {to_str(TRANSLATION_CACHE_FILE)}")
    
    print(f"\n🧪 Test path generation:")
    print(f"   Story train: {to_str(get_story_file_path('narrativo', 'train'))}")
    print(f"   Model: {to_str(get_model_path('narrativo', 'clinical-bert', 3))}")
    print(f"   Prediction: {to_str(get_prediction_path('narrativo', 'clinical-bert', 'prob'))}")
    
    print(f"\n✅ Configurazione path OK")
