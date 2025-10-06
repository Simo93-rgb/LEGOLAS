#!/usr/bin/env python3
"""
Test rapido dell'integrazione XES Pipeline in LEGOLAS.
Verifica che tutti i componenti siano correttamente integrati.
"""

import sys
from pathlib import Path
from datetime import datetime


def check_file_exists(filepath: str, description: str) -> bool:
    """Verifica esistenza file."""
    path = Path(filepath)
    if path.exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} NON TROVATO: {filepath}")
        return False


def check_imports() -> bool:
    """Verifica che tutti i moduli siano importabili."""
    print("\n🔍 Verifica Import Moduli:")
    print("-" * 50)
    
    modules = [
        ("xes_parser", "XESParser"),
        ("story_generator", "StoryGenerator"),
        ("utils.types", "PatientTrace, PatientStory"),
        ("train_llm", "get_weight_dir"),
        ("history_dataset", "TextDataset"),
        ("neural_network", "LongFormerMultiClassificationHeads"),
    ]
    
    all_ok = True
    for module_name, items in modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            all_ok = False
    
    return all_ok


def check_translation_cache() -> bool:
    """Verifica translation_cache.json."""
    print("\n🔍 Verifica Translation Cache:")
    print("-" * 50)
    
    # Cerca in posizioni possibili
    possible_paths = [
        Path("translation_cache.json"),
        Path("data/translation_cache.json"),
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✅ Translation cache trovato: {path}")
            
            # Verifica contenuto
            import json
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                print(f"   - Contiene {len(cache)} traduzioni IT→EN")
                
                # Mostra alcune traduzioni
                if cache:
                    print(f"   - Esempio: {list(cache.items())[0]}")
                return True
            except Exception as e:
                print(f"⚠️  Errore lettura cache: {e}")
                return False
    
    print("❌ Translation cache NON trovato")
    return False


def check_xes_file() -> bool:
    """Verifica presenza file XES."""
    print("\n🔍 Verifica File XES:")
    print("-" * 50)
    
    xes_file = "ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025.xes"
    
    if Path(xes_file).exists():
        size_mb = Path(xes_file).stat().st_size / (1024 * 1024)
        print(f"✅ File XES trovato: {xes_file}")
        print(f"   - Dimensione: {size_mb:.1f} MB")
        return True
    else:
        print(f"⚠️  File XES non trovato: {xes_file}")
        print("   (Questo è normale se stai testando con un file diverso)")
        return False


def test_basic_functionality() -> bool:
    """Test funzionalità base."""
    print("\n🧪 Test Funzionalità Base:")
    print("-" * 50)
    
    try:
        # Test 1: Import e creazione oggetti
        from story_generator import StoryGenerator
        from utils.types import Event, PatientTrace, ClassificationTarget
        from datetime import datetime, timedelta
        
        print("✅ Import moduli riuscito")
        
        # Test 2: Creazione StoryGenerator
        generator = StoryGenerator(format_style="narrative")
        print("✅ StoryGenerator creato")
        
        # Test 3: Creazione traccia di test
        test_events = [
            Event(
                activity="ACCETTAZIONE",
                timestamp=datetime.now(),
                case_id="TEST_001"
            ),
            Event(
                activity="VISITA MEDICA",
                timestamp=datetime.now() + timedelta(seconds=300),
                case_id="TEST_001"
            )
        ]
        
        test_trace = PatientTrace(
            case_id="TEST_001",
            events=test_events,
            classification=ClassificationTarget.ADMITTED,
            patient_age=55,
            patient_gender="M"
        )
        print("✅ PatientTrace di test creato")
        
        # Test 4: Generazione storia
        story = generator.generate_story(test_trace)
        print("✅ Storia generata con successo")
        print(f"\n📝 Storia di esempio:")
        print("-" * 50)
        print(story.story_text[:300] + "..." if len(story.story_text) > 300 else story.story_text)
        print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Errore durante test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_xes_parser() -> bool:
    """Test XESParser se file disponibile."""
    print("\n🧪 Test XESParser:")
    print("-" * 50)
    
    xes_file = "ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025.xes"
    
    if not Path(xes_file).exists():
        print("⚠️  File XES non disponibile, skip test")
        return True  # Non è un errore
    
    try:
        from xes_parser import XESParser
        
        print(f"📖 Caricamento {xes_file}...")
        parser = XESParser(xes_file)
        log, df = parser.load_xes_file()
        
        print(f"✅ File XES caricato")
        print(f"   - Righe DataFrame: {len(df)}")
        
        # Statistiche
        stats = parser.get_dataset_statistics()
        print(f"✅ Statistiche estratte:")
        print(f"   - Casi totali: {stats['total_cases']}")
        print(f"   - Eventi totali: {stats['total_events']}")
        
        # Estrai prima traccia come test
        print("📊 Estrazione tracce (solo prime 2 come test)...")
        traces = parser.extract_patient_traces()[:2]
        print(f"✅ Estratte {len(traces)} tracce di test")
        
        if traces:
            trace = traces[0]
            print(f"\n   Prima traccia:")
            print(f"   - Case ID: {trace.case_id}")
            print(f"   - Eventi: {len(trace.events)}")
            print(f"   - Classificazione: {trace.classification}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore XESParser: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_scripts() -> bool:
    """Verifica presenza script principali."""
    print("\n🔍 Verifica Script:")
    print("-" * 50)
    
    scripts = {
        "generate_stories.py": "Script unificato generazione",
        "train_xes_model.py": "Script training XES",
        "run_xes_pipeline.sh": "Bash automation",
    }
    
    all_ok = True
    for script, desc in scripts.items():
        if check_file_exists(script, desc):
            # Verifica se eseguibile (per .sh)
            if script.endswith('.sh'):
                path = Path(script)
                if path.stat().st_mode & 0o111:
                    print(f"   ✓ Eseguibile")
                else:
                    print(f"   ⚠️  Non eseguibile (esegui: chmod +x {script})")
        else:
            all_ok = False
    
    return all_ok


def check_documentation() -> bool:
    """Verifica presenza documentazione."""
    print("\n🔍 Verifica Documentazione:")
    print("-" * 50)
    
    docs = [
        "INTEGRATION_GUIDE.md",
        "FLOW_DIAGRAM.md",
        "INTEGRATION_SUMMARY.md",
        "README.md",
    ]
    
    found = 0
    for doc in docs:
        if Path(doc).exists():
            print(f"✅ {doc}")
            found += 1
        else:
            print(f"⚠️  {doc} (opzionale)")
    
    return found >= 2  # Almeno 2 doc presenti


def main():
    """Esegue tutti i test."""
    print("=" * 60)
    print("   LEGOLAS - Test Integrazione XES Pipeline")
    print("=" * 60)
    
    results = {}
    
    # Test 1: File principali
    results['files'] = check_scripts()
    
    # Test 2: Documentazione
    results['docs'] = check_documentation()
    
    # Test 3: Import moduli
    results['imports'] = check_imports()
    
    # Test 4: Translation cache
    results['cache'] = check_translation_cache()
    
    # Test 5: File XES (opzionale)
    results['xes_file'] = check_xes_file()
    
    # Test 6: Funzionalità base
    results['functionality'] = test_basic_functionality()
    
    # Test 7: XES Parser (se file disponibile)
    results['xes_parser'] = test_xes_parser()
    
    # Riepilogo
    print("\n" + "=" * 60)
    print("   RIEPILOGO TEST")
    print("=" * 60)
    
    for test_name, result in results.items():
        icon = "✅" if result else "❌"
        print(f"{icon} {test_name.replace('_', ' ').title()}")
    
    # Conteggio
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    print("\n" + "-" * 60)
    print(f"Risultato: {passed}/{total} test superati")
    
    if passed == total:
        print("\n🎉 TUTTO OK! L'integrazione è completa e funzionante.")
        print("\nProssimi passi:")
        print("  1. ./run_xes_pipeline.sh")
        print("  2. python train_xes_model.py")
        return 0
    elif passed >= total - 2:
        print("\n✅ Integrazione funzionante con avvertimenti minori.")
        print("   Verifica i warning sopra e correggi se necessario.")
        return 0
    else:
        print("\n⚠️  Alcuni test non sono passati.")
        print("   Controlla gli errori sopra e sistema i problemi.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
