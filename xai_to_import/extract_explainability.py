"""
Extract Explainability Script
Estrae attribution scores usando Integrated Gradients e genera visualizzazioni

Usage:
    python extract_explainability.py --model clinical-bert 
    python extract_explainability.py --model bert-base-uncased  --use_ensemble
"""

import argparse
import pickle
import torch
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from transformers import AutoTokenizer
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
from src.explainability.ig_completeness import compute_ig_with_completeness_check
from src.utils.model_config_loader import ModelConfigLoader
from pathlib import Path as _Path
from typing import List

# Configurazione Adaptive Integrated Gradients (stessa del progetto di reference)
ADAPTIVE_IG_STEPS_INITIAL = 1500   # Step iniziali (~60% samples converge)
ADAPTIVE_IG_STEPS_MAX = 5500       # Step massimi per casi difficili
ADAPTIVE_IG_TOLERANCE = 0.05       # Soglia errore relativo (5%)


def load_trained_model(model_name: str, num_classes: int, 
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
        model, tokenizer, is_ensemble, max_len
    """
    from src.training.eval_model import load_model_for_eval
    
    print(f"\n🤖 Loading model: {model_name}")
    print(f"   Mode: {'Ensemble (all folds)' if use_ensemble else 'Best fold only'}")
    
    # Usa helper function per caricare modello (singolo o ensemble)
    # load_model_for_eval returns (model_or_ensemble, is_ensemble, hf_model_id, weights_ref)
    model, is_ensemble, hf_model_id, weights_ref = load_model_for_eval(
        model_name=model_name,
        num_classes=num_classes,
        device=device,
        use_ensemble=use_ensemble
    )
    
    print(f"   HuggingFace ID: {hf_model_id}")
    print(f"   Tokenizer weights ref: {weights_ref}")
    
    # Carica tokenizer (prefer local weights ref for consistency)
    try:
        tokenizer = AutoTokenizer.from_pretrained(weights_ref, truncation_side='left', local_files_only=True)
    except Exception as e:
        print(f"⚠️  Tokenizer not found locally for '{weights_ref}': {e}")
        print(f"   Fallback to 'bert-base-uncased' (local-only)")
        try:
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', truncation_side='left', local_files_only=True)
        except Exception as e2:
            print(f"⚠️  Local fallback failed: {e2}")
            print(f"   Attempting to load '{weights_ref}' without local-only constraint")
            tokenizer = AutoTokenizer.from_pretrained(weights_ref, truncation_side='left')
    
    # Resolve max length from YAML (fallback 512)
    try:
        cfg_loader = ModelConfigLoader()
        cfg = cfg_loader.get_model(model_name)
        max_len = cfg.max_length if cfg else 512
    except FileNotFoundError:
        max_len = 512
    
    print(f"   ✅ Model loaded successfully")
    if is_ensemble:
        print(f"   📊 Ensemble: {len(model.models)} fold models\n")
    else:
        print(f"   📊 Single model: best fold\n")
    
    return model, tokenizer, is_ensemble, max_len


def load_sequences(splits: List[str]):
    """
    Carica le sequenze temporali dai split selezionati e la mappatura delle label.

    Args:
        splits: Lista di split da includere, tra ['train','val','test','all'].
                Se contiene 'all', include ['train','val','test'].
    Returns:
        examples: lista combinata di dict con keys ['case_id','tokens','time_deltas','label','label_id']
        label2id: dict mapping label string -> id
    """
    print("📖 Loading time-sequence data...")
    processed_dir = _Path('data/processed')
    file_map = {
        'train': processed_dir / 'train_english_sequences.pkl',
        'val': processed_dir / 'val_english_sequences.pkl',
        'test': processed_dir / 'test_english_sequences.pkl',
    }
    label2id_path = processed_dir / 'label2id.pkl'

    if 'all' in splits:
        splits = ['train', 'val', 'test']

    # Validate files exist
    if not label2id_path.exists():
        raise FileNotFoundError(
            f"Processed label mapping not found in {processed_dir}. Expected label2id.pkl"
        )
    missing = [s for s in splits if s not in file_map or not file_map[s].exists()]
    if missing:
        raise FileNotFoundError(
            f"Processed split files not found in {processed_dir}: {', '.join(missing)}"
        )

    examples = []
    counts = {}
    for s in splits:
        with open(file_map[s], 'rb') as f:
            exs = pickle.load(f)
        examples.extend(exs)
        counts[s] = len(exs)

    with open(label2id_path, 'rb') as f:
        label2id = pickle.load(f)

    counts_str = ", ".join([f"{k}={v}" for k, v in counts.items()])
    print(f"   ✅ Loaded splits: {counts_str} | Total: {len(examples)} | classes: {len(label2id)}")
    return examples, label2id


def get_sequence_predictions(model, tokenizer, examples, device='cuda', batch_size=64, is_ensemble=False, max_length: int = 512):
    """
    Ottieni predizioni su sequenze temporali (actions + time_deltas).
    
    Returns:
        predicted_labels, predicted_probs
    """
    print(f"\n🔮 Generating predictions on sequences...")
    all_preds = []
    all_probs = []
    
    # Mini-batching manuale
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i+batch_size]
            # Tokenize each example into ids/mask and propagate time_deltas to sub-tokens
            input_ids_list = []
            attention_mask_list = []
            time_deltas_list = []
            
            cls_id = getattr(tokenizer, 'cls_token_id', None)
            sep_id = getattr(tokenizer, 'sep_token_id', None)
            
            for ex in batch:
                ids = []
                times = []
                if cls_id is not None:
                    ids.append(cls_id)
                    times.append(0.0)
                for act, d in zip(ex['tokens'], ex['time_deltas']):
                    tok_ids = tokenizer.encode(act, add_special_tokens=False)
                    ids.extend(tok_ids)
                    times.extend([float(d)] * len(tok_ids))
                    if sep_id is not None:
                        ids.append(sep_id)
                        times.append(0.0)
                # Truncate to model max length
                if len(ids) > max_length:
                    ids = ids[:max_length]
                    times = times[:max_length]
                input_ids_list.append(torch.tensor(ids, dtype=torch.long))
                attention_mask_list.append(torch.ones(len(ids), dtype=torch.long))
                time_deltas_list.append(torch.tensor(times, dtype=torch.float))
            
            # Pad to max length in batch
            max_len = max(t.size(0) for t in input_ids_list)
            pad_id = getattr(tokenizer, 'pad_token_id', 0)
            def pad_tensor(t, pad_value):
                if t.size(0) == max_len:
                    return t
                return torch.cat([t, torch.full((max_len - t.size(0),), pad_value, dtype=t.dtype)])
            input_ids = torch.stack([pad_tensor(t, pad_id) for t in input_ids_list]).to(device)
            attention_mask = torch.stack([pad_tensor(t, 0) for t in attention_mask_list]).to(device)
            time_deltas = torch.stack([pad_tensor(t, 0.0) for t in time_deltas_list]).to(device)
            
            if is_ensemble:
                batch_probs = model.predict(input_ids, attention_mask, time_deltas)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, time_deltas=time_deltas)
                batch_probs = torch.softmax(outputs, dim=1)
            batch_preds = torch.argmax(batch_probs, dim=1)
            
            all_preds.extend(batch_preds.cpu().tolist())
            all_probs.extend(batch_probs.cpu().tolist())
    
    print(f"   ✅ Predictions completed")
    return all_preds, all_probs


def _compute_action_attributions(actions, tokens, token_attributions):
    """Aggrega attribution per azione usando i token delimitati da [SEP]."""
    action_scores = {}
    idx = 0
    n = len(tokens)
    # Salta [CLS] iniziale se presente
    if idx < n and tokens[idx] == '[CLS]':
        idx += 1
    for action in actions:
        score = 0.0
        # Somma fino al prossimo [SEP] o [PAD]
        while idx < n and tokens[idx] not in ['[SEP]', '[PAD]']:
            score += float(token_attributions[idx])
            idx += 1
        # Salta [SEP] se presente
        if idx < n and tokens[idx] == '[SEP]':
            idx += 1
        action_scores[action] = score
    return action_scores


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
        '--splits',
        type=str,
        nargs='+',
        choices=['train','val','test','all'],
        default=['all'],
        help='Dataset splits to include (e.g., train test). Use "all" for train+val+test.'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        nargs='+',
        default=None,
        help='Alias for --splits (e.g., --dataset test). Overrides --splits if provided.'
    )
    parser.add_argument(
        '--format',
        type=str,
        default=None,
        help='(Ignored — kept for backward compatibility with to_import scripts)'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=None,
        help='Number of samples to analyze (None = all selected splits)'
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
        help='Internal batch size for IG interpolation steps (higher=faster, more VRAM). Default: 32'
    )
    parser.add_argument(
        '--n_steps',
        type=int,
        default=ADAPTIVE_IG_STEPS_INITIAL,
        help=f'Number of steps for Integrated Gradients (default: {ADAPTIVE_IG_STEPS_INITIAL}, fixed mode)'
    )
    parser.add_argument(
        '--adaptive_steps',
        action='store_true',
        help=(
            f'Adaptive n_steps: start with {ADAPTIVE_IG_STEPS_INITIAL}, '
            f'upgrade to {ADAPTIVE_IG_STEPS_MAX} if rel_error>{ADAPTIVE_IG_TOLERANCE}. '
            'Saves time on easy samples.'
        )
    )
    parser.add_argument(
        '--use_ensemble',
        action='store_true',
        help='Use K-Fold ensemble (average attributions across folds) instead of best fold'
    )
    parser.add_argument(
        '--both_classes',
        action='store_true',
        default=True,
        help=(
            'Compute IG for BOTH target classes (0 and 1) per sample. '
            'Required for proper cross-class comparison plots (no zero bars). '
            'Doubles IG computation time. Default: True'
        )
    )
    parser.add_argument(
        '--no_both_classes',
        action='store_false',
        dest='both_classes',
        help='Disable dual-class IG (only compute for predicted class, faster but plots may have zero bars)'
    )
    parser.add_argument(
        '--checkpoint_every',
        type=int,
        default=50,
        help='Save a recovery checkpoint every N samples (0 = disabled). Default: 50'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint if available (same model/splits/mode combination)'
    )

    args = parser.parse_args()

    # --dataset is an alias for --splits
    if args.dataset is not None:
        # Validate values
        valid = {'train', 'val', 'test', 'all'}
        bad = [v for v in args.dataset if v not in valid]
        if bad:
            parser.error(f'--dataset: invalid choices {bad}. Valid: {sorted(valid)}')
        args.splits = args.dataset
    
    print("=" * 80)
    print("  LEGOLAS - Explainability Extraction (Integrated Gradients)")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"   Model: {args.model}")
    print(f"   Splits: {args.splits}")
    print(f"   Device: {args.device}")
    print(f"   Mode: {'Ensemble (K-Fold)' if args.use_ensemble else 'Best fold only'}")
    print(f"   Top-K: {args.top_k}")
    print(f"   IG internal_batch_size: {args.internal_batch_size}")
    if args.adaptive_steps:
        print(f"   IG steps: Adaptive ({ADAPTIVE_IG_STEPS_INITIAL}→{ADAPTIVE_IG_STEPS_MAX} if needed, tolerance={ADAPTIVE_IG_TOLERANCE})")
    else:
        print(f"   IG steps: {args.n_steps} (fixed)")
    print(f"   Dual-class IG: {'Yes (both class 0 and 1 per sample)' if args.both_classes else 'No (predicted class only)'}")
    
    # Assicura directory output
    ensure_directories()
    explainability_dir = OUTPUT_DIR / "explainability"
    explainability_dir.mkdir(exist_ok=True)

    # Checkpoint path (deterministic: no timestamp, can be found on --resume)
    _mode_sfx = "ensemble" if args.use_ensemble else "single"
    _splits_sfx = "_".join(sorted(args.splits))
    checkpoint_file = explainability_dir / f"checkpoint__{args.model}_{_mode_sfx}_{_splits_sfx}.pkl"

    # Timestamp per file output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Load selected splits (time-aware)
    examples, label2id = load_sequences(args.splits)
    num_classes = len(label2id)
    print(f"\n   📊 Dataset: {len(examples)} sequences across splits {args.splits}, {num_classes} classes")
    
    # 2. Load model (singolo o ensemble)
    model, tokenizer, is_ensemble, max_len = load_trained_model(
        model_name=args.model,
        num_classes=num_classes,
        device=args.device,
        use_ensemble=args.use_ensemble
    )
    
    # Limita samples se richiesto
    if args.n_samples:
        examples = examples[:args.n_samples]
        print(f"\n   ⚠️  Analyzing only first {args.n_samples} sequences")
    
    # 3. Get predictions
    predicted_labels, predicted_probs = get_sequence_predictions(
        model, tokenizer, examples, args.device, 
        batch_size=128,  # Batch size for prediction
        is_ensemble=is_ensemble,
        max_length=max_len
    )
    
    # Statistiche predizioni
    from collections import Counter
    pred_counts = Counter(predicted_labels)
    print(f"\n   📊 Predicted distribution:")
    print(f"      Class 0: {pred_counts[0]} samples")
    print(f"      Class 1: {pred_counts[1]} samples")
    
    # Accuracy
    true_labels = [ex['label_id'] if 'label_id' in ex else label2id.get(ex['label'], 0) for ex in examples]
    correct = sum(t == p for t, p in zip(true_labels, predicted_labels))
    accuracy = correct / len(true_labels)
    print(f"\n   🎯 Accuracy: {accuracy:.2%} ({correct}/{len(true_labels)})")
    
    # 4. Extract Integrated Gradients
    print(f"\n{'='*80}")
    print(f"🔍 Computing Integrated Gradients over sequences...")
    explainer = IntegratedGradientsExplainer(
        model.models[0] if is_ensemble else model,
        tokenizer,
        args.device
    )
    results = []
    adaptive_stats = {'started_initial': 0, 'upgraded_max': 0}
    _resume_start = 0

    if args.resume and checkpoint_file.exists():
        with open(checkpoint_file, 'rb') as _f:
            _ckpt = pickle.load(_f)
        results = _ckpt['results']
        _resume_start = _ckpt['next_idx']
        adaptive_stats = _ckpt.get('adaptive_stats', adaptive_stats)
        timestamp = _ckpt.get('timestamp', timestamp)
        print(f"\n♻️  Resuming from checkpoint: {_resume_start}/{len(examples)} samples already done")
    elif args.resume:
        print(f"\n⚠️  --resume specified but no checkpoint found at {checkpoint_file}. Starting fresh.")

    def _save_checkpoint(next_idx):
        with open(checkpoint_file, 'wb') as _f:
            pickle.dump({
                'results': results,
                'next_idx': next_idx,
                'adaptive_stats': adaptive_stats,
                'timestamp': timestamp,
            }, _f)

    # --- Pre-tokenization phase (pure CPU) ---
    # Separate tokenization from GPU IG computation to avoid CPU/GPU ping-pong.
    # All tokenizer.encode() calls happen here; the IG loop below is GPU-only.
    print(f"\n🔄 Pre-tokenizing {len(examples)} sequences (CPU)...")
    _cls_id = getattr(tokenizer, 'cls_token_id', None)
    _sep_id = getattr(tokenizer, 'sep_token_id', None)
    _preprocessed = []   # list of (input_ids_t, attention_mask_t, time_deltas_t, tokens)
    for _ex in tqdm(examples, desc="   Tokenizing", unit="sample", leave=False):
        _ids: list = []
        _times: list = []
        if _cls_id is not None:
            _ids.append(_cls_id)
            _times.append(0.0)
        for _act, _d in zip(_ex['tokens'], _ex['time_deltas']):
            _tok_ids = tokenizer.encode(_act, add_special_tokens=False)
            _ids.extend(_tok_ids)
            _times.extend([float(_d)] * len(_tok_ids))
            if _sep_id is not None:
                _ids.append(_sep_id)
                _times.append(0.0)
        if len(_ids) > max_len:
            _ids = _ids[:max_len]
            _times = _times[:max_len]
        _preprocessed.append((
            torch.tensor(_ids,  dtype=torch.long,  device=args.device).unsqueeze(0),
            torch.ones(len(_ids), dtype=torch.long,  device=args.device).unsqueeze(0),
            torch.tensor(_times, dtype=torch.float, device=args.device).unsqueeze(0),
            tokenizer.convert_ids_to_tokens(_ids),
        ))
    print(f"   ✅ Pre-tokenization done")

    # base_model reference for embedding lookup (shared across all samples)
    _base_model_ref = (
        getattr(model.models[0] if is_ensemble else model, 'base', None)
        or getattr(model.models[0] if is_ensemble else model, 'longformer', None)
    )

    # Shared CUDA error flag (set by _fixed_attr/_adaptive_attr on CUDA crash)
    _cuda_err = [None]

    for idx, ex in enumerate(tqdm(examples, desc="   Computing IG", unit="sample")):
        if idx < _resume_start:
            continue

        actions = ex['tokens']
        deltas = ex['time_deltas']
        target_cls = predicted_labels[idx]

        # Unpack pre-tokenized data (no CPU tokenization here)
        input_ids_t, attention_mask_t, time_deltas_t, tokens = _preprocessed[idx]

        # Embedding lookup (fast GPU op, wrapped in no_grad)
        try:
            with torch.no_grad():
                embeddings = _base_model_ref.embeddings.word_embeddings(input_ids_t)
                baseline   = _base_model_ref.embeddings.word_embeddings(torch.zeros_like(input_ids_t))
        except RuntimeError as _e:
            if 'CUDA' in str(_e) or 'cuda' in str(_e) or 'illegal' in str(_e).lower():
                tqdm.write(f"\n💥 CUDA error at sample {idx}: {_e}")
                tqdm.write(f"   Saving checkpoint ({len(results)} results)...")
                _save_checkpoint(idx)
                tqdm.write(f"   ✅ Re-run with --resume to continue from sample {idx}.")
                return
            raise

        # Flag shared by _adaptive_attr / _fixed_attr / _ensemble_attr to signal a CUDA crash

        if is_ensemble:
            n_steps_ens = ADAPTIVE_IG_STEPS_INITIAL if args.adaptive_steps else args.n_steps

            def _ensemble_attr(cls):
                return model.compute_ensemble_attributions(
                    input_ids=input_ids_t,
                    attention_mask=attention_mask_t,
                    time_deltas=time_deltas_t,
                    target_class=cls,
                    n_steps=n_steps_ens,
                    internal_batch_size=args.internal_batch_size
                ).detach().cpu().numpy()

            if args.both_classes:
                ensemble_attr_0 = _ensemble_attr(0)
                ensemble_attr_1 = _ensemble_attr(1)
                word_attr_0 = explainer.aggregate_subword_attributions(tokens, ensemble_attr_0)
                word_attr_1 = explainer.aggregate_subword_attributions(tokens, ensemble_attr_1)
                action_attr_0 = _compute_action_attributions(actions, tokens, ensemble_attr_0)
                action_attr_1 = _compute_action_attributions(actions, tokens, ensemble_attr_1)
                ensemble_attr = ensemble_attr_1 if target_cls == 1 else ensemble_attr_0
            else:
                ensemble_attr = _ensemble_attr(target_cls)
                word_attr_0 = word_attr_1 = explainer.aggregate_subword_attributions(tokens, ensemble_attr)
                action_attr_0 = action_attr_1 = _compute_action_attributions(actions, tokens, ensemble_attr)

            word_attr = word_attr_1 if target_cls == 1 else word_attr_0
            action_attr = action_attr_1 if target_cls == 1 else action_attr_0
            result = {
                'text': ' '.join(actions),
                'tokens': tokens,
                'token_attributions': ensemble_attr,
                'word_attributions': word_attr,
                'action_attributions': action_attr,
                'word_attr_class_0': word_attr_0,
                'word_attr_class_1': word_attr_1,
                'action_attr_class_0': action_attr_0,
                'action_attr_class_1': action_attr_1,
                'true_label': ex['label_id'] if 'label_id' in ex else label2id.get(ex['label'], 0),
                'predicted_label': target_cls,
                'predicted_prob': predicted_probs[idx],
                'actions': actions,
                'time_deltas': deltas
            }
        else:
            if args.adaptive_steps:
                # --- Adaptive IG path usando ig_completeness ---
                # embeddings and baseline are already computed above (pre-tokenized)
                forward_fn = explainer.build_closed_forward_fn(attention_mask_t, time_deltas_t)

                def _adaptive_attr(cls):
                    """Returns (attributions, diagnostics, did_upgrade)."""
                    try:
                        a, d = compute_ig_with_completeness_check(
                            forward_fn=forward_fn,
                            input_embeds=embeddings,
                            baseline_embeds=baseline,
                            target_class=cls,
                            n_steps=ADAPTIVE_IG_STEPS_INITIAL,
                            internal_batch_size=args.internal_batch_size,
                            device=args.device
                        )
                        upgraded = False
                        if d['rel_error'] > ADAPTIVE_IG_TOLERANCE:
                            a, d = compute_ig_with_completeness_check(
                                forward_fn=forward_fn,
                                input_embeds=embeddings,
                                baseline_embeds=baseline,
                                target_class=cls,
                                n_steps=ADAPTIVE_IG_STEPS_MAX,
                                internal_batch_size=args.internal_batch_size,
                                device=args.device
                            )
                            upgraded = True
                        return a, d, upgraded
                    except RuntimeError as _e_adp:
                        if 'CUDA' in str(_e_adp) or 'cuda' in str(_e_adp) or 'illegal' in str(_e_adp).lower():
                            _cuda_err[0] = _e_adp
                            return None, None, False
                        raise

                adaptive_stats['started_initial'] += 1

                if args.both_classes:
                    attr_0, diag_0, up_0 = _adaptive_attr(0)
                    if _cuda_err[0]: break
                    attr_1, diag_1, up_1 = _adaptive_attr(1)
                    if _cuda_err[0]: break
                    if up_0 or up_1:
                        adaptive_stats['upgraded_max'] += 1
                    np_0 = attr_0.sum(dim=-1).squeeze(0).detach().cpu().numpy()
                    np_1 = attr_1.sum(dim=-1).squeeze(0).detach().cpu().numpy()
                    word_attr_0 = explainer.aggregate_subword_attributions(tokens, np_0)
                    word_attr_1 = explainer.aggregate_subword_attributions(tokens, np_1)
                    action_attr_0 = _compute_action_attributions(actions, tokens, np_0)
                    action_attr_1 = _compute_action_attributions(actions, tokens, np_1)
                    attributions_np = np_1 if target_cls == 1 else np_0
                    diag = diag_1 if target_cls == 1 else diag_0
                else:
                    attr, diag, upgraded = _adaptive_attr(target_cls)
                    if upgraded:
                        adaptive_stats['upgraded_max'] += 1
                    attributions_np = attr.sum(dim=-1).squeeze(0).detach().cpu().numpy()
                    word_attr_0 = word_attr_1 = explainer.aggregate_subword_attributions(tokens, attributions_np)
                    action_attr_0 = action_attr_1 = _compute_action_attributions(actions, tokens, attributions_np)

                word_attr = word_attr_1 if target_cls == 1 else word_attr_0
                action_attr = action_attr_1 if target_cls == 1 else action_attr_0
                result = {
                    'text': ' '.join(actions),
                    'tokens': tokens,
                    'token_attributions': attributions_np,
                    'word_attributions': word_attr,
                    'action_attributions': action_attr,
                    'word_attr_class_0': word_attr_0,
                    'word_attr_class_1': word_attr_1,
                    'action_attr_class_0': action_attr_0,
                    'action_attr_class_1': action_attr_1,
                    'true_label': ex['label_id'] if 'label_id' in ex else label2id.get(ex['label'], 0),
                    'predicted_label': target_cls,
                    'predicted_prob': predicted_probs[idx],
                    'actions': actions,
                    'time_deltas': deltas,
                    'ig_diagnostics': diag
                }
            else:
                # --- Fixed IG path ---
                # Use pre-computed embeddings/baseline from top of loop — no re-tokenization.
                def _fixed_attr(cls):
                    try:
                        attr = explainer.ig.attribute(
                            inputs=embeddings,
                            baselines=baseline,
                            method='riemann_trapezoid',
                            additional_forward_args=(attention_mask_t, time_deltas_t),
                            target=cls,
                            n_steps=args.n_steps,
                            internal_batch_size=args.internal_batch_size
                        )
                        return attr.sum(dim=-1).squeeze(0).detach().cpu().numpy()
                    except RuntimeError as _e_fix:
                        if 'CUDA' in str(_e_fix) or 'cuda' in str(_e_fix) or 'illegal' in str(_e_fix).lower():
                            _cuda_err[0] = _e_fix
                            return None
                        raise

                if args.both_classes:
                    tok_attr_0 = _fixed_attr(0)
                    if _cuda_err[0]: break
                    tok_attr_1 = _fixed_attr(1)
                    if _cuda_err[0]: break
                    word_attr_0 = explainer.aggregate_subword_attributions(tokens, tok_attr_0)
                    word_attr_1 = explainer.aggregate_subword_attributions(tokens, tok_attr_1)
                    action_attr_0 = _compute_action_attributions(actions, tokens, tok_attr_0)
                    action_attr_1 = _compute_action_attributions(actions, tokens, tok_attr_1)
                    tok_attr = tok_attr_1 if target_cls == 1 else tok_attr_0
                else:
                    tok_attr = _fixed_attr(target_cls)
                    if _cuda_err[0]: break
                    word_attr_0 = word_attr_1 = explainer.aggregate_subword_attributions(tokens, tok_attr)
                    action_attr_0 = action_attr_1 = _compute_action_attributions(actions, tokens, tok_attr)

                word_attr = word_attr_1 if target_cls == 1 else word_attr_0
                action_attr = action_attr_1 if target_cls == 1 else action_attr_0
                result = {
                    'text': ' '.join(actions),
                    'tokens': tokens,
                    'token_attributions': tok_attr,
                    'word_attributions': word_attr,
                    'action_attributions': action_attr,
                    'word_attr_class_0': word_attr_0,
                    'word_attr_class_1': word_attr_1,
                    'action_attr_class_0': action_attr_0,
                    'action_attr_class_1': action_attr_1,
                    'true_label': ex['label_id'] if 'label_id' in ex else label2id.get(ex['label'], 0),
                    'predicted_label': target_cls,
                    'predicted_prob': predicted_probs[idx],
                    'actions': actions,
                    'time_deltas': deltas
                }
        results.append(result)

        # Periodic checkpoint
        if args.checkpoint_every > 0 and (idx + 1) % args.checkpoint_every == 0:
            _save_checkpoint(idx + 1)
            tqdm.write(f"   💾 Checkpoint saved at sample {idx + 1}/{len(examples)}")

    # CUDA error during IG — checkpoint was saved by break, now exit gracefully
    if _cuda_err[0] is not None:
        tqdm.write(f"\n💥 CUDA error at sample {idx}: {_cuda_err[0]}")
        tqdm.write(f"   Saving checkpoint ({len(results)} results done)...")
        _save_checkpoint(idx)
        tqdm.write(f"   ✅ Re-run with --resume to continue from sample {idx}.")
        return

    print(f"   ✅ IG completed for {len(results)} sequences")

    # Report statistiche adaptive (solo single model)
    if args.adaptive_steps and not is_ensemble:
        total = adaptive_stats['started_initial']
        upgraded = adaptive_stats['upgraded_max']
        upgraded_pct = upgraded / total * 100 if total > 0 else 0.0
        saved_pct = (1 - upgraded_pct / 100) * (
            (ADAPTIVE_IG_STEPS_MAX - ADAPTIVE_IG_STEPS_INITIAL) / ADAPTIVE_IG_STEPS_MAX * 100
        )
        print(f"\n   📊 Adaptive IG statistics:")
        print(f"      Converged at {ADAPTIVE_IG_STEPS_INITIAL} steps: {total - upgraded} samples ({100 - upgraded_pct:.1f}%)")
        print(f"      Upgraded to {ADAPTIVE_IG_STEPS_MAX} steps: {upgraded} samples ({upgraded_pct:.1f}%)")
        print(f"      Estimated time saved: ~{saved_pct:.1f}% vs fixed {ADAPTIVE_IG_STEPS_MAX} steps")
    
    # Salva risultati raw
    mode_suffix = "ensemble" if is_ensemble else "single"
    results_file = explainability_dir / f"ig_results__{args.model}_{mode_suffix}_{timestamp}.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n💾 Risultati salvati: {results_file}")

    # Rimuovi checkpoint (esecuzione completata con successo)
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"   🗑️  Checkpoint rimosso (run completato)")
    # 5. Extract top words (usa sempre explainer, funziona per entrambi i casi)
    print(f"\n📊 Extracting top-{args.top_k} words...")
    top_words = explainer.extract_top_words(results, top_k=args.top_k, by_class=True)
    
    print(f"\n   Top {args.top_k} words for Class 1:")
    for i, (word, score) in enumerate(list(top_words['class_1'].items())[:10], 1):
        print(f"      {i:2d}. {word:20s} \u2192 {score:.4f}")

    print(f"\n   Top {args.top_k} words for Class 0:")
    for i, (word, score) in enumerate(list(top_words['class_0'].items())[:10], 1):
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
    
    if top_actions.get('class_1'):
        print(f"\n   Top {args.top_k} clinical actions for Class 1:")
        for i, (action, stats) in enumerate(list(top_actions['class_1'].items())[:5], 1):
            print(f"      {i}. {action[:60]}...")
            print(f"         Mean: {stats['mean_score']:.4f}, Count: {stats['count']}")
    if top_actions.get('class_0'):
        print(f"\n   Top {args.top_k} clinical actions for Class 0:")
        for i, (action, stats) in enumerate(list(top_actions['class_0'].items())[:5], 1):
            print(f"      {i}. {action[:60]}...")
            print(f"         Mean: {stats['mean_score']:.4f}, Count: {stats['count']}")
    
    # Salva action results
    actions_file = explainability_dir / f"actions_{args.model}_{mode_suffix}_{timestamp}.pkl"
    with open(actions_file, 'wb') as f:
        pickle.dump(action_results, f)
    print(f"\n💾 Action results salvati: {actions_file}")
    
    # 7. Visualizations
    print(f"\n🎨 Creating visualizations...")
    
    # WORD-LEVEL visualizations (originali, per riferimento)
    print(f"   📝 Word-level visualizations...")
    heatmap_words_path = explainability_dir / f"heatmap_words_{args.model}_{mode_suffix}_{timestamp}.png"
    plot_attention_heatmap(
        top_words['class_0'],
        top_words['class_1'],
        top_k=args.top_k,
        save_path=heatmap_words_path
    )
    
    histogram_words_path = explainability_dir / f"histogram_words_{args.model}_{mode_suffix}_{timestamp}.png"
    plot_class_comparison(
        top_words['class_0'],
        top_words['class_1'],
        top_k=args.top_k,
        save_path=histogram_words_path
    )
    
    # CLINICAL ACTIONS visualizations (principali, più interpretabili)
    print(f"   🏥 Clinical actions visualizations...")
    heatmap_actions_path = explainability_dir / f"heatmap_actions_{args.model}_{mode_suffix}_{timestamp}.png"
    plot_clinical_actions_heatmap(
        action_results['class_0'],
        action_results['class_1'],
        top_k=args.top_k,
        save_path=heatmap_actions_path
    )
    
    histogram_actions_path = explainability_dir / f"histogram_actions_{args.model}_{mode_suffix}_{timestamp}.png"
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
