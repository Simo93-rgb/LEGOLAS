"""
Extract Explainability Script
Estrae attribution scores usando Integrated Gradients e genera visualizzazioni

Usage:
    python extract_explainability.py --model clinical-bert --format narrativo
"""

import argparse
import pickle
import torch
import glob
import numpy as np
from pathlib import Path
from datetime import datetime

from transformers import AutoTokenizer, AutoModel
from src.models.neural_network import LongFormerMultiClassificationHeads
from src.explainability import (
    IntegratedGradientsExplainer,
    ClinicalActionAggregator,
    plot_attention_heatmap,
    plot_class_comparison,
    plot_clinical_actions_heatmap,
    plot_clinical_actions_comparison
)
from src.config.paths import (
    get_story_file_path,
    get_model_path,
    MODELS_DIR,
    OUTPUT_DIR,
    ensure_directories
)


def load_trained_model(model_name: str, story_format: str, device: str = 'cuda'):
    """
    Carica modello addestrato
    
    Args:
        model_name: Nome modello (es: 'clinical-bert')
        story_format: Formato storie usato nel training
        device: Device per computation
        
    Returns:
        model, tokenizer
    """
    # Carica configurazione modello
    from src.utils.model_config_loader import ModelConfigLoader
    
    config_loader = ModelConfigLoader()
    model_config = config_loader.get_model(model_name)
    
    if not model_config:
        raise ValueError(f"Modello '{model_name}' non trovato in config")
    
    print(f"\n🤖 Loading model: {model_name}")
    print(f"   HuggingFace ID: {model_config.hf_model_id}")
    
    # Carica tokenizer e base model
    tokenizer = AutoTokenizer.from_pretrained(model_config.hf_model_id)
    base_model = AutoModel.from_pretrained(model_config.hf_model_id)
    
    # Wrap con classification head
    model = LongFormerMultiClassificationHeads(base_model)
    
    # Cerca file modello addestrato
    model_pattern = str(MODELS_DIR / f'xes_{story_format}_{model_name}*.pth')
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        raise FileNotFoundError(
            f"Nessun modello addestrato trovato per {model_name} con formato {story_format}\n"
            f"Pattern cercato: {model_pattern}"
        )
    
    # Prendi ultimo file (epoch più alto)
    model_files.sort()
    model_file = model_files[-1]
    
    print(f"   📥 Loading weights: {model_file}")
    
    # Carica pesi
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"   ✅ Model loaded successfully\n")
    
    return model, tokenizer


def load_test_data(story_format: str):
    """
    Carica dati di test
    
    Returns:
        texts, true_labels, predicted_labels
    """
    print(f"📖 Loading test data (format: {story_format})...")
    
    # Carica test stories
    test_path = get_story_file_path(story_format, 'test')
    with open(test_path, 'rb') as f:
        texts = pickle.load(f)
    
    # Carica labels
    label_test_path = get_story_file_path(story_format, 'label_test')
    with open(label_test_path, 'rb') as f:
        labels_str = pickle.load(f)
    
    label_train_path = get_story_file_path(story_format, 'label_train')
    with open(label_train_path, 'rb') as f:
        labels_train_str = pickle.load(f)
    
    # Crea mapping label
    unique_labels = sorted(set(labels_train_str))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Converti labels
    true_labels = [label2id[label] for label in labels_str]
    
    print(f"   ✅ Loaded {len(texts)} test samples")
    print(f"   📊 Label mapping: {label2id}")
    
    # Identifica quale è classe 0 (più numerosa)
    from collections import Counter
    label_counts = Counter(true_labels)
    class_0_count = label_counts[0]
    class_1_count = label_counts[1]
    
    print(f"\n   📈 Class distribution:")
    print(f"      Class 0: {class_0_count} samples ({class_0_count/len(true_labels)*100:.1f}%)")
    print(f"      Class 1: {class_1_count} samples ({class_1_count/len(true_labels)*100:.1f}%)")
    
    if class_0_count < class_1_count:
        print(f"\n   ⚠️  WARNING: Class 0 has fewer samples than Class 1!")
        print(f"       Verify label mapping is correct")
    
    return texts, true_labels, label2id


def get_model_predictions(model, tokenizer, texts, device='cuda', batch_size=32):
    """
    Ottieni predizioni del modello su test set
    
    Returns:
        predicted_labels, predicted_probs
    """
    print(f"\n🔮 Generating predictions...")
    
    model.eval()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoding = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    
    print(f"   ✅ Predictions completed")
    
    return all_preds, all_probs


def main():
    parser = argparse.ArgumentParser(
        description='Extract explainability using Integrated Gradients'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='clinical-bert',
        help='Model name (es: clinical-bert, pubmedbert-base)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='narrativo',
        choices=['narrativo', 'bullet', 'clinical'],
        help='Story format'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=None,
        help='Number of samples to analyze (None = all test set)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=25,
        help='Number of top words to visualize'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for computation'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for IG computation'
    )
    parser.add_argument(
        '--n_steps',
        type=int,
        default=50,
        help='Number of steps for Integrated Gradients'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("  LEGOLAS - Explainability Extraction (Integrated Gradients)")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"   Model: {args.model}")
    print(f"   Format: {args.format}")
    print(f"   Device: {args.device}")
    print(f"   Top-K: {args.top_k}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   IG steps: {args.n_steps}")
    
    # Assicura directory output
    ensure_directories()
    explainability_dir = OUTPUT_DIR / "explainability"
    explainability_dir.mkdir(exist_ok=True)
    
    # Timestamp per file output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Load model
    model, tokenizer = load_trained_model(args.model, args.format, args.device)
    
    # 2. Load test data
    texts, true_labels, label2id = load_test_data(args.format)
    
    # Limita samples se richiesto
    if args.n_samples:
        texts = texts[:args.n_samples]
        true_labels = true_labels[:args.n_samples]
        print(f"\n   ⚠️  Analyzing only first {args.n_samples} samples")
    
    # 3. Get predictions
    predicted_labels, predicted_probs = get_model_predictions(
        model, tokenizer, texts, args.device
    )
    
    # Statistiche predizioni
    from collections import Counter
    pred_counts = Counter(predicted_labels)
    print(f"\n   📊 Predicted distribution:")
    print(f"      Class 0: {pred_counts[0]} samples")
    print(f"      Class 1: {pred_counts[1]} samples")
    
    # Accuracy
    correct = sum(t == p for t, p in zip(true_labels, predicted_labels))
    accuracy = correct / len(true_labels)
    print(f"\n   🎯 Accuracy: {accuracy:.2%} ({correct}/{len(true_labels)})")
    
    # 4. Extract Integrated Gradients
    print(f"\n{'='*80}")
    explainer = IntegratedGradientsExplainer(model, tokenizer, args.device)
    
    results = explainer.explain_batch(
        texts=texts,
        labels=true_labels,
        predicted_classes=predicted_labels,
        batch_size=args.batch_size,
        n_steps=args.n_steps
    )
    
    # Salva risultati raw
    results_file = explainability_dir / f"ig_results_{args.format}_{args.model}_{timestamp}.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n💾 Risultati salvati: {results_file}")
    
    # 5. Extract top words
    print(f"\n📊 Extracting top-{args.top_k} words...")
    top_words = explainer.extract_top_words(results, top_k=args.top_k, by_class=True)
    
    print(f"\n   Top {args.top_k} words for Class 0:")
    for i, (word, score) in enumerate(list(top_words['class_0'].items())[:10], 1):
        print(f"      {i:2d}. {word:20s} → {score:.4f}")
    
    print(f"\n   Top {args.top_k} words for Class 1:")
    for i, (word, score) in enumerate(list(top_words['class_1'].items())[:10], 1):
        print(f"      {i:2d}. {word:20s} → {score:.4f}")
    
    # 6. Clinical Actions Aggregation
    print(f"\n🏥 Aggregating clinical actions...")
    aggregator = ClinicalActionAggregator()
    action_results = aggregator.aggregate_across_dataset(results, by_class=True)
    
    top_actions = aggregator.get_top_actions(
        action_results,
        top_k=args.top_k,  # Usa stesso top_k delle parole
        sort_by='mean_score'
    )
    
    if top_actions.get('class_0'):
        print(f"\n   Top {args.top_k} clinical actions for Class 0:")
        for i, (action, stats) in enumerate(list(top_actions['class_0'].items())[:5], 1):
            print(f"      {i}. {action[:60]}...")
            print(f"         Mean: {stats['mean_score']:.4f}, Count: {stats['count']}")
    
    # Salva action results
    actions_file = explainability_dir / f"actions_{args.format}_{args.model}_{timestamp}.pkl"
    with open(actions_file, 'wb') as f:
        pickle.dump(action_results, f)
    print(f"\n💾 Action results salvati: {actions_file}")
    
    # 7. Visualizations
    print(f"\n🎨 Creating visualizations...")
    
    # WORD-LEVEL visualizations (originali, per riferimento)
    print(f"   📝 Word-level visualizations...")
    heatmap_words_path = explainability_dir / f"heatmap_words_{args.format}_{args.model}_{timestamp}.png"
    plot_attention_heatmap(
        top_words['class_0'],
        top_words['class_1'],
        top_k=args.top_k,
        save_path=heatmap_words_path
    )
    
    histogram_words_path = explainability_dir / f"histogram_words_{args.format}_{args.model}_{timestamp}.png"
    plot_class_comparison(
        top_words['class_0'],
        top_words['class_1'],
        top_k=args.top_k,
        save_path=histogram_words_path
    )
    
    # CLINICAL ACTIONS visualizations (principali, più interpretabili)
    print(f"   🏥 Clinical actions visualizations...")
    heatmap_actions_path = explainability_dir / f"heatmap_actions_{args.format}_{args.model}_{timestamp}.png"
    plot_clinical_actions_heatmap(
        action_results['class_0'],
        action_results['class_1'],
        top_k=args.top_k,
        save_path=heatmap_actions_path
    )
    
    histogram_actions_path = explainability_dir / f"histogram_actions_{args.format}_{args.model}_{timestamp}.png"
    plot_clinical_actions_comparison(
        action_results['class_0'],
        action_results['class_1'],
        top_k=args.top_k,
        save_path=histogram_actions_path
    )
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ Explainability extraction completed!")
    print(f"\n📁 Output files:")
    print(f"   • IG Results (raw): {results_file}")
    print(f"   • Clinical Actions (aggregated): {actions_file}")
    print(f"\n   📊 Clinical Actions Visualizations (MAIN - interpretable):")
    print(f"   • Heatmap (actions): {heatmap_actions_path}")
    print(f"   • Histogram (actions): {histogram_actions_path}")
    print(f"\n   📝 Word-level Visualizations (reference):")
    print(f"   • Heatmap (words): {heatmap_words_path}")
    print(f"   • Histogram (words): {histogram_words_path}")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
