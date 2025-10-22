#!/usr/bin/env python3
"""
Stampa la storia di test associata a un dato case_id.

Nota importante:
- generate_stories.py salva solo i testi in `{prefix}_test.pkl` e le label in `{prefix}_label_test.pkl`.
- I case_id non sono persistiti nei pickle; per mappare case_id→indice usiamo lo stesso XES
  e la stessa logica di split (seed/test_size) per ricostruire l'ordine del test set.

Utilizzo:
    python scripts/print_story_by_id.py --case-id 5923672 \
        [--prefix narrativo] [--xes PATH_TO_XES] [--test-size 0.15] [--seed 42]

Se `--xes` non è fornito, proverà percorsi di default.
"""

import argparse
import pickle
import sys
from pathlib import Path

# Garantisce import dei moduli del progetto (src/*)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.xes_parser import XESParser  # type: ignore


DEFAULT_XES_CANDIDATES = [
    # Percorso usato in run_xes_pipeline.sh
    Path("data/raw/ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025.xes"),
    # Percorso alternativo visto in test_integration.py
    Path("ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025.xes"),
]


def find_default_xes_file() -> Path | None:
    for p in DEFAULT_XES_CANDIDATES:
        if p.exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Stampa storia test per case_id")
    parser.add_argument("--case-id", type=str, required=True, help="Case ID della storia da stampare")
    parser.add_argument("--prefix", type=str, default="narrativo", help="Prefisso dei file di storie (default: narrativo)")
    parser.add_argument("--xes", type=str, default=None, help="Path al file XES usato per generare le storie")
    parser.add_argument("--test-size", type=float, default=0.15, help="Proporzione del test set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed per lo split")
    args = parser.parse_args()

    # Determina file XES
    xes_path = Path(args.xes) if args.xes else find_default_xes_file()
    if not xes_path or not xes_path.exists():
        print("❌ Errore: file XES non trovato.")
        print("   Specifica --xes PATH oppure posiziona il file in uno dei percorsi di default:")
        for p in DEFAULT_XES_CANDIDATES:
            print(f"   - {p}")
        sys.exit(1)

    # Carica testi di test e (opzionale) label
    stories_dir = Path("output") / "stories"
    test_texts_path = stories_dir / f"{args.prefix}_test.pkl"
    label_test_path = stories_dir / f"{args.prefix}_label_test.pkl"

    if not test_texts_path.exists():
        print(f"❌ File testi test non trovato: {test_texts_path}")
        print("   Assicurati di aver eseguito generate_stories.py con --output-prefix corrispondente.")
        sys.exit(1)

    with open(test_texts_path, "rb") as f:
        test_texts = pickle.load(f)

    # Carica XES e ricostruisci case_id test con stesso split
    parser = XESParser(str(xes_path))
    _log, _df = parser.load_xes_file()
    traces = parser.extract_patient_traces()

    case_ids = [str(t.case_id) for t in traces]
    labels = [t.classification.value if t.classification is not None else 0 for t in traces]

    # Stesso split di generate_stories.py
    from sklearn.model_selection import train_test_split
    case_ids_train, case_ids_test, labels_train, labels_test = train_test_split(
        case_ids,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    # Trova indice della storia tramite case_id
    target_id = str(args.case_id)
    try:
        idx = case_ids_test.index(target_id)
    except ValueError:
        print(f"❌ Case ID {target_id} non trovato nel test set.")
        print("   Possibili cause: ID è nel training, XES/seed/test_size diversi, o case_id inesistente.")
        sys.exit(1)

    # Stampa esclusivamente il testo della storia
    story_text = test_texts[idx]
    print(story_text)


if __name__ == "__main__":
    main()