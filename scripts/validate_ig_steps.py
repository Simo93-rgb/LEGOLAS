#!/usr/bin/env python3
"""
Script di validazione per trovare il valore ottimale di n_steps per Integrated Gradients

Testa diversi valori di n_steps (20, 50, 100, 200, 500) su N samples
e produce report CSV con rel_error per ogni combinazione (sample, fold, n_steps)

Usage:
    python scripts/validate_ig_steps.py --model bert-base-uncased --format narrativo --n_samples 10
    
Output:
    - output/explainability/ig_validation_{timestamp}.csv
    - Report console con raccomandazione n_steps ottimale
"""

import argparse
import pickle
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import numpy as np

from transformers import AutoTokenizer
from src.models.ensemble import EnsembleModel
from src.training.eval_model import load_model_for_eval
from src.config.paths import get_story_file_path, OUTPUT_DIR, ensure_directories


def load_test_samples(story_format: str, n_samples: int = 10):
    """Carica primi N campioni del test set"""
    test_path = get_story_file_path(story_format, 'test')
    with open(test_path, 'rb') as f:
        texts = pickle.load(f)
    
    label_test_path = get_story_file_path(story_format, 'label_test')
    with open(label_test_path, 'rb') as f:
        labels_str = pickle.load(f)
    
    label_train_path = get_story_file_path(story_format, 'label_train')
    with open(label_train_path, 'rb') as f:
        labels_train_str = pickle.load(f)
    
    unique_labels = sorted(set(labels_train_str))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    
    labels = [label2id[label] for label in labels_str]
    
    return texts[:n_samples], labels[:n_samples], label2id


def validate_n_steps_grid(
    ensemble: EnsembleModel,
    tokenizer,
    texts: List[str],
    labels: List[int],
    n_steps_values: List[int],
    device: str = 'cuda'
) -> pd.DataFrame:
    """
    Testa griglia di n_steps su samples e ritorna DataFrame con risultati
    
    Returns:
        DataFrame con colonne: sample_idx, fold_idx, n_steps, rel_error, abs_error, converged
    """
    results = []
    
    print(f"\n🔍 Testing {len(n_steps_values)} n_steps values on {len(texts)} samples...")
    print(f"   n_steps values: {n_steps_values}")
    print(f"   Folds: {len(ensemble.models)}\n")
    
    for sample_idx, (text, label) in enumerate(zip(texts, labels)):
        print(f"📄 Sample {sample_idx+1}/{len(texts)}")
        
        # Tokenize
        encoding = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get prediction per determinare target_class
        with torch.no_grad():
            probs = ensemble.predict(input_ids, attention_mask)
            predicted_class = torch.argmax(probs, dim=1).item()
        
        target_class = predicted_class
        
        # Test ogni valore di n_steps
        for n_steps in n_steps_values:
            print(f"   n_steps={n_steps:4d}: ", end='', flush=True)
            
            # Compute IG per ogni fold con questo n_steps
            fold_diagnostics = []
            
            for fold_idx, model in enumerate(ensemble.models):
                model.eval()
                
                # Define forward function
                def forward_func(input_embeds, attention_mask_param=attention_mask):
                    outputs = model.longformer(
                        inputs_embeds=input_embeds,
                        attention_mask=attention_mask_param
                    )
                    logits = model.output_layer(outputs.pooler_output)
                    return logits
                
                # Get embeddings
                embeddings = model.longformer.embeddings.word_embeddings(input_ids)
                baseline_embeds = torch.zeros_like(embeddings)
                
                # Compute IG with completeness check
                from src.explainability.ig_completeness import compute_ig_with_completeness_check
                
                _, diagnostics = compute_ig_with_completeness_check(
                    forward_fn=forward_func,
                    input_embeds=embeddings,
                    baseline_embeds=baseline_embeds,
                    target_class=target_class,
                    n_steps=n_steps,
                    device=device,
                    return_convergence_delta=True
                )
                
                fold_diagnostics.append(diagnostics)
                
                # Aggiungi al risultato
                results.append({
                    'sample_idx': sample_idx,
                    'fold_idx': fold_idx,
                    'n_steps': n_steps,
                    'rel_error': diagnostics['rel_error'],
                    'abs_error': diagnostics['abs_error'],
                    'converged': diagnostics['converged'],
                    'f_x': diagnostics['f_x'],
                    'f_baseline': diagnostics['f_baseline'],
                    'delta': diagnostics['delta']
                })
            
            # Report avg rel_error per questo n_steps
            avg_rel_error = np.mean([d['rel_error'] for d in fold_diagnostics])
            converged_folds = sum(d['converged'] for d in fold_diagnostics)
            status = "✅" if converged_folds == len(fold_diagnostics) else "⚠️"
            
            print(f"{status} avg_rel_error={avg_rel_error:.6f} ({converged_folds}/{len(fold_diagnostics)} converged)")
    
    df = pd.DataFrame(results)
    return df


def analyze_results(df: pd.DataFrame) -> Dict:
    """Analizza risultati e trova n_steps ottimale"""
    print(f"\n{'='*80}")
    print("  ANALYSIS RESULTS")
    print(f"{'='*80}\n")
    
    # Group by n_steps
    summary = df.groupby('n_steps').agg({
        'rel_error': ['mean', 'median', 'std', 'max'],
        'abs_error': ['mean', 'median'],
        'converged': ['sum', 'count']
    }).round(6)
    
    print("📊 Summary by n_steps:")
    print(summary)
    print()
    
    # Trova n_steps ottimale (miglior compromesso convergenza/costo)
    n_steps_stats = []
    for n_steps in sorted(df['n_steps'].unique()):
        subset = df[df['n_steps'] == n_steps]
        n_steps_stats.append({
            'n_steps': n_steps,
            'mean_rel_error': subset['rel_error'].mean(),
            'median_rel_error': subset['rel_error'].median(),
            'convergence_rate': subset['converged'].sum() / len(subset),
            'max_rel_error': subset['rel_error'].max()
        })
    
    stats_df = pd.DataFrame(n_steps_stats)
    
    # Raccomandazione: primo n_steps con convergence_rate >= 95% e mean_rel_error < 0.01
    recommended = None
    for _, row in stats_df.iterrows():
        if row['convergence_rate'] >= 0.95 and row['mean_rel_error'] <= 0.01:
            recommended = int(row['n_steps'])
            break
    
    # Se nessuno soddisfa criteria rigidi, usa primo con convergence >= 80%
    if recommended is None:
        for _, row in stats_df.iterrows():
            if row['convergence_rate'] >= 0.80:
                recommended = int(row['n_steps'])
                break
    
    # Fallback: usa quello con miglior mean_rel_error
    if recommended is None:
        recommended = int(stats_df.loc[stats_df['mean_rel_error'].idxmin(), 'n_steps'])
    
    print(f"🎯 RECOMMENDED n_steps: {recommended}")
    print(f"   Reason: Best balance between convergence and computational cost")
    print()
    
    rec_stats = stats_df[stats_df['n_steps'] == recommended].iloc[0]
    print(f"   Stats at n_steps={recommended}:")
    print(f"      Mean rel_error: {rec_stats['mean_rel_error']:.6f}")
    print(f"      Median rel_error: {rec_stats['median_rel_error']:.6f}")
    print(f"      Convergence rate: {rec_stats['convergence_rate']*100:.1f}%")
    print(f"      Max rel_error: {rec_stats['max_rel_error']:.6f}")
    print()
    
    return {
        'recommended_n_steps': recommended,
        'stats': stats_df.to_dict('records')
    }


def main():
    parser = argparse.ArgumentParser(
        description='Validate optimal n_steps for Integrated Gradients'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='bert-base-uncased',
        help='Model name'
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
        default=10,
        help='Number of test samples to validate (default: 10)'
    )
    parser.add_argument(
        '--n_steps_values',
        type=int,
        nargs='+',
        default=[20, 50, 100, 200, 500],
        help='List of n_steps values to test (default: 20 50 100 200 500)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for computation'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("  LEGOLAS - IG n_steps Validation")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"   Model: {args.model}")
    print(f"   Format: {args.format}")
    print(f"   Samples: {args.n_samples}")
    print(f"   n_steps values: {args.n_steps_values}")
    print(f"   Device: {args.device}")
    
    # Ensure directories
    ensure_directories()
    explainability_dir = OUTPUT_DIR / "explainability"
    explainability_dir.mkdir(exist_ok=True)
    
    # Load test data
    print(f"\n📖 Loading test data...")
    texts, labels, label2id = load_test_samples(args.format, args.n_samples)
    num_classes = len(label2id)
    print(f"   ✅ Loaded {len(texts)} samples")
    
    # Load ensemble model
    print(f"\n🤖 Loading ensemble model...")
    model, is_ensemble, hf_model_id = load_model_for_eval(
        story_format=args.format,
        model_name=args.model,
        num_classes=num_classes,
        device=args.device,
        use_ensemble=True
    )
    
    if not is_ensemble:
        print("❌ Error: This script requires ensemble model (K-Fold)")
        return
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    print(f"   ✅ Ensemble loaded: {len(model.models)} folds")
    
    # Run validation
    df_results = validate_n_steps_grid(
        ensemble=model,
        tokenizer=tokenizer,
        texts=texts,
        labels=labels,
        n_steps_values=args.n_steps_values,
        device=args.device
    )
    
    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = explainability_dir / f"ig_validation_{args.format}_{args.model}_{timestamp}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n💾 Raw results saved: {csv_path}")
    
    # Analyze and recommend
    analysis = analyze_results(df_results)
    
    # Save analysis report
    report_path = explainability_dir / f"ig_validation_report_{args.format}_{args.model}_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write("LEGOLAS - IG n_steps Validation Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Format: {args.format}\n")
        f.write(f"Samples: {args.n_samples}\n")
        f.write(f"n_steps tested: {args.n_steps_values}\n\n")
        f.write(f"RECOMMENDED n_steps: {analysis['recommended_n_steps']}\n\n")
        f.write("Stats by n_steps:\n")
        for stat in analysis['stats']:
            f.write(f"\n  n_steps={stat['n_steps']}:\n")
            f.write(f"    Mean rel_error: {stat['mean_rel_error']:.6f}\n")
            f.write(f"    Median rel_error: {stat['median_rel_error']:.6f}\n")
            f.write(f"    Convergence rate: {stat['convergence_rate']*100:.1f}%\n")
            f.write(f"    Max rel_error: {stat['max_rel_error']:.6f}\n")
    
    print(f"📄 Report saved: {report_path}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
