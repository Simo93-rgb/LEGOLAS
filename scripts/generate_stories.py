#!/usr/bin/env python3
"""
Script unificato per generare storie narrative da diverse sorgenti di dati.
Supporta sia la pipeline originale (CSV con skeleton.py) che la nuova pipeline (XES con story_generator.py).

Utilizzo:
    python generate_stories.py --pipeline xes --input data.xes --output stories.pkl
    python generate_stories.py --pipeline csv --input data.csv --output stories.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Tuple
import pandas as pd

from src.data.xes_parser import XESParser
from src.generation.story_generator import StoryGenerator
from utils.types import PatientStory, PatientTrace


def generate_stories_from_xes(
    xes_file: str,
    format_style: str = "narrative",
    enable_clinical_tokens: bool = False
) -> Tuple[List[str], List[int]]:
    """
    Genera storie da un file XES usando la nuova pipeline.
    
    Args:
        xes_file: Path al file XES di input
        format_style: Stile di formattazione ("narrative" o "bullet_points")
        enable_clinical_tokens: Se abilitare i token clinici atomici
        
    Returns:
        Tupla (lista_storie, lista_labels)
    """
    print(f"📖 Caricamento file XES: {xes_file}")
    
    # Parse XES file
    parser = XESParser(xes_file)
    log, df = parser.load_xes_file()
    
    # Mostra statistiche dataset
    stats = parser.get_dataset_statistics()
    print(f"\n📊 Statistiche Dataset:")
    print(f"   - Casi totali: {stats['total_cases']}")
    print(f"   - Eventi totali: {stats['total_events']}")
    print(f"   - Attività uniche: {stats['unique_activities']}")
    print(f"   - Distribuzione classificazioni: {stats['classification_distribution']}")
    
    # Estrai tracce pazienti
    print(f"\n🔄 Estrazione tracce pazienti...")
    traces = parser.extract_patient_traces()
    print(f"✅ Estratte {len(traces)} tracce")
    
    # Genera storie
    print(f"\n📝 Generazione storie narrative (formato: {format_style})...")
    generator = StoryGenerator(
        format_style=format_style,
        enable_clinical_tokens=enable_clinical_tokens
    )
    stories = generator.generate_batch_stories(traces)
    
    # Estrai testi e labels
    story_texts = [story.story_text for story in stories]
    story_labels = [
        story.classification.value if story.classification else 0 
        for story in stories
    ]
    
    print(f"✅ Generate {len(stories)} storie narrative")
    
    return story_texts, story_labels


def generate_stories_from_csv(csv_file: str) -> Tuple[List[str], List[int]]:
    """
    Genera storie da un file CSV usando la pipeline originale (skeleton.py).
    
    Args:
        csv_file: Path al file CSV di input
        
    Returns:
        Tupla (lista_storie, lista_labels)
    """
    print(f"📖 Caricamento file CSV: {csv_file}")
    print("⚠️  Pipeline CSV originale non ancora completamente integrata.")
    print("   Usa la funzione history_conversion da main.py per ora.")
    
    raise NotImplementedError(
        "Pipeline CSV non ancora completamente integrata. "
        "Usa direttamente main.py per la pipeline originale."
    )


def split_train_test(
    stories: List[str],
    labels: List[int],
    test_size: float = 0.34,
    random_state: int = 42
) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Divide le storie in training e test set.
    
    Args:
        stories: Lista di storie narrative
        labels: Lista di label corrispondenti
        test_size: Proporzione del test set (default 0.34)
        random_state: Seed per riproducibilità
        
    Returns:
        Tupla (train_stories, test_stories, train_labels, test_labels)
    """
    from sklearn.model_selection import train_test_split
    
    train_stories, test_stories, train_labels, test_labels = train_test_split(
        stories,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    
    print(f"\n📊 Split dataset:")
    print(f"   - Training: {len(train_stories)} storie")
    print(f"   - Test: {len(test_stories)} storie")
    
    return train_stories, test_stories, train_labels, test_labels


def save_stories(
    train_stories: List[str],
    test_stories: List[str],
    train_labels: List[int],
    test_labels: List[int],
    output_prefix: str = "stories"
):
    """
    Salva le storie in file pickle separati per training e test.
    
    Args:
        train_stories: Storie di training
        test_stories: Storie di test
        train_labels: Label di training
        test_labels: Label di test
        output_prefix: Prefisso per i file di output
    """
    # Crea directory output se non esiste
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    files = {
        f"{output_prefix}_train.pkl": train_stories,
        f"{output_prefix}_test.pkl": test_stories,
        f"{output_prefix}_label_train.pkl": train_labels,
        f"{output_prefix}_label_test.pkl": test_labels,
    }
    
    print(f"\n💾 Salvataggio file pickle:")
    for filename, data in files.items():
        filepath = output_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"   - {filepath}")
    
    print(f"✅ Storie salvate con successo!")


def main():
    parser = argparse.ArgumentParser(
        description="Genera storie narrative da file XES o CSV"
    )
    
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["xes", "csv"],
        default="xes",
        help="Tipo di pipeline da usare (xes o csv)"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path al file di input (XES o CSV)"
    )
    
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="stories",
        help="Prefisso per i file di output (default: stories)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["narrative", "bullet_points"],
        default="narrative",
        help="Stile di formattazione delle storie (solo per pipeline XES)"
    )
    
    parser.add_argument(
        "--clinical-tokens",
        action="store_true",
        help="Abilita token clinici atomici (solo per pipeline XES, SPERIMENTALE)"
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.34,
        help="Proporzione del test set (default: 0.34)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed per riproducibilità (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Verifica che il file di input esista
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Errore: File di input non trovato: {args.input}")
        sys.exit(1)
    
    # Genera storie in base alla pipeline scelta
    try:
        if args.pipeline == "xes":
            stories, labels = generate_stories_from_xes(
                xes_file=args.input,
                format_style=args.format,
                enable_clinical_tokens=args.clinical_tokens
            )
        else:
            stories, labels = generate_stories_from_csv(csv_file=args.input)
        
        # Split train/test
        train_stories, test_stories, train_labels, test_labels = split_train_test(
            stories, labels, test_size=args.test_size, random_state=args.seed
        )
        
        # Salva risultati
        save_stories(
            train_stories, test_stories, train_labels, test_labels,
            output_prefix=args.output_prefix
        )
        
        print("\n✅ Pipeline completata con successo!")
        
    except Exception as e:
        print(f"\n❌ Errore durante l'esecuzione della pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
