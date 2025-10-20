"""
Extract Explainability Script
Estrae attribution scores usando Integrated Gradients e genera visualizzazioni

Usage:
    python extract_explainability.py --model clinical-bert --format narrativo
    python extract_explainability.py --model bert-base-uncased --format narrativo --use_ensemble
"""

import argparse
import pickle
import torch
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from src.models.neural_network import LongFormerMultiClassificationHeads
from src.models.ensemble import EnsembleModel
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


def load_trained_model(model_name: str, story_format: str, num_classes: int, 
                       device: str = 'cuda', use_ensemble: bool = False):
    """
    Carica modello addestrato (singolo o ensemble)
    
    Args:
        model_name: Nome modello (es: 'bert-base-uncased')
        story_format: Formato storie usato nel training
        num_classes: Numero di classi
        device: Device per computation
        use_ensemble: Se True, carica EnsembleModel; altrimenti best fold
    
    Returns:
        model, tokenizer, is_ensemble
    """
    from src.training.eval_model import load_model_for_eval
    
    print(f"\n🤖 Loading model: {model_name}")
    print(f"   Story format: {story_format}")
    print(f"   Mode: {'Ensemble (all folds)' if use_ensemble else 'Best fold only'}")
    
    # Usa helper function per caricare modello (singolo o ensemble)
    model, is_ensemble, hf_model_id = load_model_for_eval(
        story_format=story_format,
        model_name=model_name,
        num_classes=num_classes,
        device=device,
        use_ensemble=use_ensemble
    )
    
    print(f"   HuggingFace ID: {hf_model_id}")
    
    # Carica tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    
    print(f"   ✅ Model loaded successfully")
    if is_ensemble:
        print(f"   📊 Ensemble: {len(model.models)} fold models\n")
    else:
        print(f"   📊 Single model: best fold\n")
    
    return model, tokenizer, is_ensemble


def load_test_data(story_format: str, dataset: str = 'test'):
    """
    Carica dati dal dataset richiesto.
    
    Args:
        story_format: Formato storie ('narrativo', 'bullet', 'clinical')
        dataset: 'test' (default), 'train', oppure 'all' per unire train+test
    
    Returns:
        texts, true_labels, label2id
    """
    assert dataset in {'test', 'train', 'all'}, "dataset deve essere 'test', 'train' o 'all'"
    print(f"📖 Loading data (format: {story_format}, dataset: {dataset})...")

    def _load_split(split: str):
        # Carica stories
        stories_path = get_story_file_path(story_format, split)
        with open(stories_path, 'rb') as f:
            split_texts = pickle.load(f)
        # Carica labels stringhe per lo split
        label_path = get_story_file_path(story_format, f'label_{split}')
        with open(label_path, 'rb') as f:
            split_labels_str = pickle.load(f)
        return split_texts, split_labels_str

    if dataset == 'all':
        texts_test, labels_test_str = _load_split('test')
        texts_train, labels_train_str = _load_split('train')
        texts = list(texts_train) + list(texts_test)
        labels_str = list(labels_train_str) + list(labels_test_str)
        # Mapping dalle label TRAIN per coerenza con training
        unique_labels = sorted(set(labels_train_str))
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        true_labels = [label2id[label] for label in labels_str]
        print(f"   ✅ Loaded {len(texts)} samples (train+test)")
        print(f"   📊 Label mapping (from train): {label2id}")
    else:
        # Singolo split
        texts, labels_str = _load_split(dataset)
        # Servono le label train per definire mapping coerente
        _, labels_train_str = _load_split('train')
        unique_labels = sorted(set(labels_train_str))
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        true_labels = [label2id[label] for label in labels_str]
        print(f"   ✅ Loaded {len(texts)} {dataset} samples")
        print(f"   📊 Label mapping (from train): {label2id}")

    # Distribuzione classi
    from collections import Counter
    label_counts = Counter(true_labels)
    class_0_count = label_counts.get(0, 0)
    class_1_count = label_counts.get(1, 0)
    total = len(true_labels)
    print(f"\n   📈 Class distribution:")
    print(f"      Class 0: {class_0_count} samples ({(class_0_count/total*100 if total else 0):.1f}%)")
    print(f"      Class 1: {class_1_count} samples ({(class_1_count/total*100 if total else 0):.1f}%)")

    if class_0_count < class_1_count:
        print(f"\n   ⚠️  WARNING: Class 0 has fewer samples than Class 1!")
        print(f"       Verify label mapping is correct")

    return texts, true_labels, label2id


def get_model_predictions(model, tokenizer, texts, device='cuda', internal_batch_size=32, is_ensemble=False):
    """
    Ottieni predizioni del modello su test set
    
    Args:
        model: Model or EnsembleModel instance
        tokenizer: Tokenizer
        texts: List of text strings
        device: Device for computation
        internal_batch_size: Batch size
        is_ensemble: If True, model is EnsembleModel
        
    Returns:
        predicted_labels, predicted_probs
    """
    print(f"\n🔮 Generating predictions...")
    
    if is_ensemble:
        # Usa metodo predict dell'ensemble
        all_probs = []
        all_preds = []
        
        for i in range(0, len(texts), internal_batch_size):
            batch_texts = texts[i:i+internal_batch_size]
            
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
            
            # Predict con ensemble (ritorna batch_probs: [batch, num_classes])
            batch_probs = model.predict(input_ids, attention_mask)
            batch_preds = torch.argmax(batch_probs, dim=1)
            
            all_probs.extend(batch_probs.cpu().tolist())
            all_preds.extend(batch_preds.cpu().tolist())
            
    else:
        # Singolo modello (logica originale)
        model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for i in range(0, len(texts), internal_batch_size):
                batch_texts = texts[i:i+internal_batch_size]
                
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
        default='bert-base-uncased',
        help='Model name (es: clinical-bert, pubmedbert-base, bert-base-uncased)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='narrativo',
        choices=['narrativo', 'bullet', 'clinical'],
        help='Story format'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='test',
        choices=['test', 'train', 'all'],
        help="Dataset da usare: 'test', 'train' o 'all' (train+test)"
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=None,
        help='Number of samples to analyze (None = all selected dataset)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=20,
        help='Number of top words to visualize'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for computation'
    )
    parser.add_argument(
        '--internal_batch_size',
        type=int,
        default=32,
        help='Internal batch size for IG interpolation steps (higher=faster, but uses more GPU memory). Default: 32'
    )
    parser.add_argument(
        '--n_steps',
        type=int,
        default=1500,
        help='Number of steps for Integrated Gradients (default: 1500, validated for convergence)'
    )
    parser.add_argument(
        '--adaptive_steps',
        action='store_true',
        help='Use adaptive n_steps strategy: start with 1000, increase to 2000 if needed (saves ~30%% time)'
    )
    parser.add_argument(
        '--use_ensemble',
        action='store_true',
        help='Use K-Fold ensemble (average attributions across folds) instead of best fold'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("  LEGOLAS - Explainability Extraction (Integrated Gradients)")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"   Model: {args.model}")
    print(f"   Format: {args.format}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Device: {args.device}")
    print(f"   Mode: {'Ensemble (K-Fold)' if args.use_ensemble else 'Best fold only'}")
    print(f"   Top-K: {args.top_k}")
    print(f"   Internal batch size: {args.internal_batch_size} (for IG interpolation)")
    if args.adaptive_steps:
        print(f"   IG steps: Adaptive (1000→2000 if needed, tolerance=0.05)")
    else:
        print(f"   IG steps: {args.n_steps} (fixed)")
    
    # Assicura directory output
    ensure_directories()
    explainability_dir = OUTPUT_DIR / "explainability"
    explainability_dir.mkdir(exist_ok=True)
    
    # Timestamp per file output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Load data (per ottenere num_classes)
    texts, true_labels, label2id = load_test_data(args.format, dataset=args.dataset)
    num_classes = len(label2id)
    
    print(f"\n   📊 Dataset: {len(texts)} samples, {num_classes} classes")
    
    # 2. Load model (singolo o ensemble)
    model, tokenizer, is_ensemble = load_trained_model(
        model_name=args.model,
        story_format=args.format,
        num_classes=num_classes,
        device=args.device,
        use_ensemble=args.use_ensemble
    )
    
    # Limita samples se richiesto
    if args.n_samples:
        texts = texts[:args.n_samples]
        true_labels = true_labels[:args.n_samples]
        print(f"\n   ⚠️  Analyzing only first {args.n_samples} samples")
    
    # 3. Get predictions
    predicted_labels, predicted_probs = get_model_predictions(
        model, tokenizer, texts, args.device, 
        internal_batch_size=32,  # Prediction batch size (diverso da IG batch)
        is_ensemble=is_ensemble
    )
    
    # Statistiche predizioni
    from collections import Counter
    pred_counts = Counter(predicted_labels)
    print(f"\n   📊 Predicted distribution:")
    print(f"      Class 0: {pred_counts[0]} samples")
    print(f"      Class 1: {pred_counts[1]} samples")
    
    # Accuracy
    correct = sum(t == p for t, p in zip(true_labels, predicted_labels))
    accuracy = correct / len(true_labels) if len(true_labels) else 0.0
    print(f"\n   🎯 Accuracy: {accuracy:.2%} ({correct}/{len(true_labels)})")
    
    # 4. Extract Integrated Gradients
    print(f"\n{'='*80}")
    
    # Helper function: Strategia adattiva per IG (riutilizzabile per ensemble e single)
    def compute_attributions_adaptive(
        compute_fn,  # Funzione che calcola IG: (n_steps) -> (attributions, diagnostics?)
        use_adaptive: bool,
        fixed_n_steps: int,
        get_rel_error_fn=None  # Funzione per estrarre rel_error da diagnostics (se diverso)
    ):
        """
        Wrapper per strategia adattiva: prova 1000 steps, se non converge usa 2000.
        
        Args:
            compute_fn: Callable che accetta n_steps e ritorna (attributions, diagnostics?) 
            use_adaptive: Se True, usa strategia adattiva
            fixed_n_steps: n_steps da usare in modalità fissa
            get_rel_error_fn: Callable per estrarre rel_error da diagnostics
            
        Returns:
            attributions, adaptive_upgraded (bool indicating if upgraded to 2000)
        """
        if use_adaptive:
            n_steps_initial = 1000
            n_steps_max = 2000
            tolerance = 0.05
            
            # Tentativo con 1000 steps
            result = compute_fn(n_steps_initial)
            
            # Estrai rel_error (default: cerca 'avg_rel_error' o 'rel_error')
            if get_rel_error_fn:
                rel_error = get_rel_error_fn(result)
            else:
                # Default: result è tuple (attributions, diagnostics)
                diagnostics = result[1] if isinstance(result, tuple) else {}
                rel_error = diagnostics.get('avg_rel_error') or diagnostics.get('rel_error', float('inf'))
            
            # Se non converge, ricalcola con 2000
            if rel_error > tolerance:
                result = compute_fn(n_steps_max)
                upgraded = True
            else:
                upgraded = False
                
            return result, upgraded
        else:
            # Modalità fissa
            result = compute_fn(fixed_n_steps)
            return result, False
    
    
    if is_ensemble:
        # Ensemble mode: usa compute_ensemble_attributions per ogni sample
        print(f"🔍 Computing Ensemble Integrated Gradients...")
        print(f"   Strategy: Calculate IG for each fold model → Average attributions")
        print(f"   Models: {len(model.models)} folds")
        
        # Create temporary explainer with first fold model (just to use helper methods)
        temp_explainer = IntegratedGradientsExplainer(model.models[0], tokenizer, args.device)
        
        results = []
        
                # Stats per strategia adattiva
        adaptive_stats = {'started_1000': 0, 'upgraded_2000': 0}
        
        # Progress bar per ensemble IG
        for idx, text in enumerate(tqdm(texts, desc="   Computing Ensemble IG", unit="sample")):
            # Tokenize
            encoding = tokenizer(text, padding=True, truncation=True, 
                                max_length=512, return_tensors='pt')
            input_ids = encoding['input_ids'].to(args.device)
            attention_mask = encoding['attention_mask'].to(args.device)
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            
            # Define compute function for adaptive wrapper
            def compute_fn(n_steps):
                return model.compute_ensemble_attributions(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_class=predicted_labels[idx],
                    n_steps=n_steps,
                    return_diagnostics=True
                )
            
            # Track adaptive stats
            if args.adaptive_steps:
                adaptive_stats['started_1000'] += 1
            
            # Compute with adaptive/fixed strategy
            result, upgraded = compute_attributions_adaptive(
                compute_fn=compute_fn,
                use_adaptive=args.adaptive_steps,
                fixed_n_steps=args.n_steps
            )
            
            if upgraded:
                adaptive_stats['upgraded_2000'] += 1
            
            # Extract attributions (if return_diagnostics=True, result is tuple)
            if isinstance(result, tuple):
                ensemble_attributions, diagnostics = result
            else:
                ensemble_attributions = result
            
            # Use explainer's aggregate method to convert tokens to words
            word_attributions = temp_explainer.aggregate_subword_attributions(
                tokens, ensemble_attributions.detach().cpu().numpy()
            )
            
            results.append({
                'text': text,
                'tokens': tokens,
                'token_attributions': ensemble_attributions.detach().cpu().numpy(),
                'word_attributions': word_attributions,
                'true_label': true_labels[idx],
                'predicted_label': predicted_labels[idx],
                'predicted_prob': predicted_probs[idx]
            })
            
        print(f"   ✅ Ensemble IG completed for {len(results)} samples")
        
        # Report statistiche adattive
        if args.adaptive_steps:
            upgraded_pct = (adaptive_stats['upgraded_2000'] / adaptive_stats['started_1000'] * 100) if adaptive_stats['started_1000'] > 0 else 0
            print(f"\n   📊 Adaptive strategy statistics:")
            print(f"      Started with 1000 steps: {adaptive_stats['started_1000']} samples")
            print(f"      Upgraded to 2000 steps: {adaptive_stats['upgraded_2000']} samples ({upgraded_pct:.1f}%)")
            print(f"      Estimated time saved: ~{(1 - upgraded_pct/100) * 33:.1f}% vs fixed 2000 steps")
        
        # Use temp_explainer for subsequent processing
        explainer = temp_explainer
        
    else:
        # Single model mode con strategia adattiva
        print(f"🔍 Computing Integrated Gradients (single model)...")
        explainer = IntegratedGradientsExplainer(model, tokenizer, args.device)
        
        results = []
        adaptive_stats = {'started_1000': 0, 'upgraded_2000': 0}
        
        if args.adaptive_steps:
            # Strategia adattiva per single model
            from src.explainability.ig_completeness import compute_ig_with_completeness_check
            
            # Progress bar per single model IG
            for idx, text in enumerate(tqdm(texts, desc="   Computing Single Model IG", unit="sample")):
                # Tokenize
                encoding = tokenizer(text, padding=True, truncation=True,
                                    max_length=512, return_tensors='pt')
                input_ids = encoding['input_ids'].to(args.device)
                attention_mask = encoding['attention_mask'].to(args.device)
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                
                # Get embeddings
                embeddings = model.longformer.embeddings.word_embeddings(input_ids)
                baseline_embeds = torch.zeros_like(embeddings)
                
                # Define forward function for IG
                def forward_func(input_embeds, attention_mask_param=attention_mask):
                    outputs = model.longformer(
                        inputs_embeds=input_embeds,
                        attention_mask=attention_mask_param
                    )
                    logits = model.output_layer(outputs.pooler_output)
                    return logits
                
                # Define compute function for adaptive wrapper
                def compute_fn(n_steps):
                    return compute_ig_with_completeness_check(
                        forward_fn=forward_func,
                        input_embeds=embeddings,
                        baseline_embeds=baseline_embeds,
                        target_class=predicted_labels[idx],
                        n_steps=n_steps,
                        device=args.device
                    )
                
                # Track adaptive stats
                adaptive_stats['started_1000'] += 1
                
                # Compute with adaptive strategy
                (attributions, diagnostics), upgraded = compute_attributions_adaptive(
                    compute_fn=compute_fn,
                    use_adaptive=True,
                    fixed_n_steps=args.n_steps
                )
                
                if upgraded:
                    adaptive_stats['upgraded_2000'] += 1
                
                # Sum across embedding dimension e converti a numpy
                attributions_np = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
                
                # Aggregate subwords to words
                word_attributions = explainer.aggregate_subword_attributions(tokens, attributions_np)
                
                results.append({
                    'text': text,
                    'tokens': tokens,
                    'token_attributions': attributions_np.tolist(),
                    'word_attributions': word_attributions,
                    'true_label': true_labels[idx],
                    'predicted_label': predicted_labels[idx],
                    'predicted_prob': predicted_probs[idx]
                })
            
            print(f"   ✅ Single model IG completed for {len(results)} samples")
            
            # Report statistiche adattive
            upgraded_pct = (adaptive_stats['upgraded_2000'] / adaptive_stats['started_1000'] * 100) if adaptive_stats['started_1000'] > 0 else 0
            print(f"\n   📊 Adaptive strategy statistics:")
            print(f"      Started with 1000 steps: {adaptive_stats['started_1000']} samples")
            print(f"      Upgraded to 2000 steps: {adaptive_stats['upgraded_2000']} samples ({upgraded_pct:.1f}%)")
            print(f"      Estimated time saved: ~{(1 - upgraded_pct/100) * 33:.1f}% vs fixed 2000 steps")
            
        else:
            # Strategia fissa: usa explain_batch originale
            results = explainer.explain_batch(
                texts=texts,
                labels=true_labels,
                predicted_classes=predicted_labels,
                internal_batch_size=args.internal_batch_size,
                n_steps=args.n_steps
            )
    
    # Salva risultati raw
    mode_suffix = "ensemble" if is_ensemble else "single"
    results_file = explainability_dir / f"ig_results_{args.format}_{args.model}_{mode_suffix}_{timestamp}.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n💾 Risultati salvati: {results_file}")
    
    # 5. Extract top words (usa sempre explainer, funziona per entrambi i casi)
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
    actions_file = explainability_dir / f"actions_{args.format}_{args.model}_{mode_suffix}_{timestamp}.pkl"
    with open(actions_file, 'wb') as f:
        pickle.dump(action_results, f)
    print(f"\n💾 Action results salvati: {actions_file}")
    
    # 7. Visualizations
    print(f"\n🎨 Creating visualizations...")
    
    # WORD-LEVEL visualizations (originali, per riferimento)
    print(f"   📝 Word-level visualizations...")
    heatmap_words_path = explainability_dir / f"heatmap_words_{args.format}_{args.model}_{mode_suffix}_{timestamp}.png"
    plot_attention_heatmap(
        top_words['class_0'],
        top_words['class_1'],
        top_k=args.top_k,
        save_path=heatmap_words_path
    )
    
    histogram_words_path = explainability_dir / f"histogram_words_{args.format}_{args.model}_{mode_suffix}_{timestamp}.png"
    plot_class_comparison(
        top_words['class_0'],
        top_words['class_1'],
        top_k=args.top_k,
        save_path=histogram_words_path
    )
    
    # CLINICAL ACTIONS visualizations (principali, più interpretabili)
    print(f"   🏥 Clinical actions visualizations...")
    heatmap_actions_path = explainability_dir / f"heatmap_actions_{args.format}_{args.model}_{mode_suffix}_{timestamp}.png"
    plot_clinical_actions_heatmap(
        action_results['class_0'],
        action_results['class_1'],
        top_k=args.top_k,
        save_path=heatmap_actions_path
    )
    
    histogram_actions_path = explainability_dir / f"histogram_actions_{args.format}_{args.model}_{mode_suffix}_{timestamp}.png"
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
