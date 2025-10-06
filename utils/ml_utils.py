"""
ML Utilities per GALADRIEL Classification.
Adattato da BERTing utilities per storytelling medico.
"""

import torch
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    matthews_corrcoef, roc_auc_score, cohen_kappa_score, balanced_accuracy_score,
    classification_report
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from typing import cast
from src.classification.story_dataset_adapter import StoryDatasetAdapter


class FocalLoss(torch.nn.Module):
    """
    Focal Loss per gestire sbilanciamento classi.
    Privilegia la classe minoritaria (classe 1 - Admitted) penalizzando falsi negativi.
    
    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Peso per bilanciare classi [alpha_class0, alpha_class1]
        gamma: Focusing parameter (default 2.0)
        reduction: 'mean' o 'sum'
    """
    
    def __init__(self, alpha: Optional[List[float]] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            # Default: privilegia classe 1 con sbilanciamento 809/7393
            alpha = [0.25, 0.75]  # Classe 0: 25%, Classe 1: 75%
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits del modello [batch_size, num_classes]
            targets: Labels ground truth [batch_size]
        """
        # Sposta alpha su stesso device degli inputs
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)
        
        # Calcola Cross Entropy
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        
        # Calcola probabilità
        p_t = torch.exp(-ce_loss)
        
        # Seleziona alpha per ogni campione
        alpha_t = self.alpha[targets]
        
        # Calcola Focal Loss
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EarlyStopping:
    """
    Early stopping avanzato con loss ratio monitoring.
    Portato da BERTing con miglioramenti.
    """
    
    def __init__(self, 
                 patience: int = 5,
                 min_delta: float = 0.001,
                 use_loss_ratio: bool = False,
                 loss_ratio_threshold: float = 1.15,
                 loss_ratio_patience: int = 3,
                 restore_best_weights: bool = True,
                 mode: str = 'min'):
        """
        Inizializza early stopping.
        
        Args:
            patience: Numero di epoche senza miglioramento prima di fermarsi
            min_delta: Miglioramento minimo per considerare progresso
            use_loss_ratio: Se usare monitoring del rapporto train/val loss
            loss_ratio_threshold: Soglia del rapporto per considerare overfitting
            loss_ratio_patience: Patience specifica per loss ratio
            restore_best_weights: Se ripristinare i pesi migliori
            mode: 'min' per minimizzare, 'max' per massimizzare
        """
        self.patience = patience
        self.min_delta = min_delta
        self.use_loss_ratio = use_loss_ratio
        self.loss_ratio_threshold = loss_ratio_threshold
        self.loss_ratio_patience = loss_ratio_patience
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        
        # Stato interno
        self.best_score = None
        self.counter = 0
        self.loss_ratio_counter = 0
        self.best_weights = None
        self.early_stop = False
        self.stop_reason = None
        
        # Operatore di confronto
        if self.mode == 'min':
            self.monitor_op = lambda current, best: current < (best - self.min_delta)
        else:
            self.monitor_op = lambda current, best: current > (best + self.min_delta)
        
        print(f"⏱️ Early Stopping configurato:")
        print(f"  Patience: {self.patience}")
        print(f"  Min delta: {self.min_delta}")
        print(f"  Loss ratio: {self.use_loss_ratio}")
        if self.use_loss_ratio:
            print(f"  Loss ratio threshold: {self.loss_ratio_threshold}")
    
    def __call__(self, current_score: float, model=None, train_loss: Optional[float] = None) -> bool:
        """
        Verifica se fermare il training.
        
        Args:
            current_score: Score corrente (validation loss/metric)
            model: Modello di cui salvare i pesi
            train_loss: Training loss per calcolare loss ratio
            
        Returns:
            bool: True se fermare il training
        """
        # Check standard early stopping
        standard_stop = self._check_standard_early_stopping(current_score, model)
        
        # Check loss ratio se abilitato
        ratio_stop = False
        if self.use_loss_ratio and train_loss is not None:
            ratio_stop = self._check_loss_ratio_early_stopping(current_score, train_loss, model)
        
        return standard_stop or ratio_stop
    
    def _check_standard_early_stopping(self, current_score: float, model) -> bool:
        """Early stopping basato sulla metrica principale."""
        if self.best_score is None:
            self.best_score = current_score
            if model and self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            if model and self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.stop_reason = "no_improvement"
                if model and self.restore_best_weights and self.best_weights:
                    model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in self.best_weights.items()})
                    print(f"🔄 Ripristinati i pesi migliori (score: {self.best_score:.4f})")
                return True
        return False
    
    def _check_loss_ratio_early_stopping(self, val_loss: float, train_loss: float, model) -> bool:
        """Early stopping basato sul rapporto train/validation loss."""
        if train_loss <= 0 or val_loss <= 0:
            return False
        
        loss_ratio = val_loss / train_loss
        
        if loss_ratio > self.loss_ratio_threshold:
            self.loss_ratio_counter += 1
            print(f"⚠️ Loss ratio alto: {loss_ratio:.3f} (threshold: {self.loss_ratio_threshold})")
            
            if self.loss_ratio_counter >= self.loss_ratio_patience:
                self.early_stop = True
                self.stop_reason = "loss_ratio_overfitting"
                print(f"🛑 Early stop per loss ratio! Ratio: {loss_ratio:.3f}")
                return True
        else:
            self.loss_ratio_counter = 0
        
        return False


def format_metrics_for_display(metrics: Dict[str, Any]) -> str:
    """
    Formatta le metriche per display user-friendly.
    
    Args:
        metrics: Dizionario delle metriche
        
    Returns:
        str: Metriche formattate
    """
    formatted = []
    
    # Metriche principali
    formatted.append(f"📊 METRICHE PRINCIPALI:")
    formatted.append(f"  • Accuracy: {metrics.get('accuracy', 0):.4f}")
    formatted.append(f"  • Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
    formatted.append(f"  • F1-Score: {metrics.get('f1', 0):.4f}")
    formatted.append(f"  • Precision: {metrics.get('precision', 0):.4f}")
    formatted.append(f"  • Recall: {metrics.get('recall', 0):.4f}")
    
    # Metrica custom per classe 1
    if 'class1_focused_score' in metrics:
        formatted.append(f"  • Class1-Focused Score: {metrics['class1_focused_score']:.4f}")
        formatted.append(f"  • Recall Classe 1 (Admitted): {metrics.get('recall_class1', 0):.4f}")
        formatted.append(f"  • Recall Classe 0 (Discharged): {metrics.get('recall_class0', 0):.4f}")
    
    # Metriche avanzate
    if 'mcc' in metrics:
        formatted.append(f"  • Matthews Corr: {metrics['mcc']:.4f}")
    if 'cohen_kappa' in metrics:
        formatted.append(f"  • Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    if 'auc' in metrics:
        formatted.append(f"  • AUC-ROC: {metrics['auc']:.4f}")
    
    return '\n'.join(formatted)


def set_reproducibility_seed(seed: int = 42):
    """
    Imposta seed per riproducibilità.
    
    Args:
        seed: Seed per RNG
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def calculate_class_weights(labels: List[int]) -> torch.Tensor:
    """
    Calcola i pesi delle classi per il bilanciamento.
    
    Args:
        labels: lista delle etichette
        
    Returns:
        torch.Tensor: Pesi delle classi
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(
        'balanced', 
        classes=unique_labels, 
        y=labels
    )
    
    # Converte in tensor e assicura ordine corretto
    weights_dict = {label: weight for label, weight in zip(unique_labels, class_weights)}
    weights_tensor = torch.tensor([
        weights_dict.get(0, 1.0),  
        weights_dict.get(1, 1.0)   
    ], dtype=torch.float32)
    
    return weights_tensor


def calculate_comprehensive_metrics(y_true: List[int], y_pred: List[int], y_pred_proba: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Calcola metriche comprehensive per la classificazione.
    
    Args:
        y_true: Label vere
        y_pred: Predizioni
        y_pred_proba: Probabilità predette (opzionale)
        
    Returns:
        dict: Dizionario con tutte le metriche
    """
    metrics = {}
    
    # Metriche base
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # Metriche per classe
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # Converti in numpy array se necessario e poi in lista
    precision_per_class = np.array(precision_per_class) if not isinstance(precision_per_class, np.ndarray) else precision_per_class
    recall_per_class = np.array(recall_per_class) if not isinstance(recall_per_class, np.ndarray) else recall_per_class
    f1_per_class = np.array(f1_per_class) if not isinstance(f1_per_class, np.ndarray) else f1_per_class
    
    metrics['precision_per_class'] = precision_per_class.tolist()
    metrics['recall_per_class'] = recall_per_class.tolist()
    metrics['f1_per_class'] = f1_per_class.tolist()
    
    # Metrica custom per privilegiare classe 1 (Admitted)
    # Considera lo sbilanciamento 809/7393 ≈ 10.9% classe 1
    if len(recall_per_class) >= 2:
        recall_class1 = float(recall_per_class[1])  # Recall per classe 1 (Admitted)
        # Combina recall classe 1 (70%) + balanced accuracy (30%)
        metrics['class1_focused_score'] = recall_class1 * 0.7 + metrics['balanced_accuracy'] * 0.3
        metrics['recall_class1'] = recall_class1
        metrics['recall_class0'] = float(recall_per_class[0]) if len(recall_per_class) > 0 else 0.0
    else:
        metrics['class1_focused_score'] = 0.0
        metrics['recall_class1'] = 0.0
        metrics['recall_class0'] = 0.0
    
    # Matthews Correlation Coefficient
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # Cohen's Kappa
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # AUC-ROC se probabilità disponibili
    if y_pred_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            metrics['auc_roc'] = None
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification Report
    metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
    
    return metrics


def save_model_with_metadata(model, save_path: Path, metadata: Dict[str, Any]):
    """
    Salva modello con metadati.
    
    Args:
        model: Modello da salvare
        save_path: Path di salvataggio
        metadata: Metadati da includere
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Salva il modello
    torch.save(model.state_dict(), save_path)
    
    # Salva i metadati
    metadata_path = save_path.with_suffix('.json')
    metadata['saved_at'] = datetime.now().isoformat()
    metadata['model_path'] = str(save_path)
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Modello salvato: {save_path}")
    print(f"📄 Metadati salvati: {metadata_path}")


def load_model_with_metadata(model, load_path: Path) -> Tuple[Any, Dict[str, Any]]:
    """
    Carica modello con metadati.
    
    Args:
        model: Modello in cui caricare i pesi
        load_path: Path del modello
        
    Returns:
        tuple: (model, metadata)
    """
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"Modello non trovato: {load_path}")
    
    # Carica i pesi
    state_dict = torch.load(load_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    # Carica i metadati se disponibili
    metadata_path = load_path.with_suffix('.json')
    metadata = {}
    
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    print(f"📥 Modello caricato: {load_path}")
    if metadata:
        print(f"📄 Metadati caricati: {metadata_path}")
    
    return model, metadata


def print_metrics_report(metrics: Dict[str, Any], title: str = "Metriche di Valutazione"):
    """
    Stampa un report dettagliato delle metriche.
    
    Args:
        metrics: Dizionario delle metriche
        title: Titolo del report
    """
    print("\n" + "="*60)
    print(f"📊 {title}")
    print("="*60)
    
    # Metriche principali
    print(f"\n🎯 METRICHE PRINCIPALI:")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"  Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
    print(f"  F1 Score: {metrics.get('f1', 0):.4f}")
    print(f"  Precision: {metrics.get('precision', 0):.4f}")
    print(f"  Recall: {metrics.get('recall', 0):.4f}")
    
    # Metriche avanzate
    print(f"\n🔬 METRICHE AVANZATE:")
    print(f"  Matthews Correlation Coefficient: {metrics.get('mcc', 0):.4f}")
    print(f"  Cohen's Kappa: {metrics.get('cohen_kappa', 0):.4f}")
    
    if 'auc_roc' in metrics and metrics['auc_roc'] is not None:
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    
    # Confusion Matrix
    if 'confusion_matrix' in metrics:
        cm = np.array(metrics['confusion_matrix'])
        print(f"\n📋 CONFUSION MATRIX:")
        print(f"  TN: {cm[0,0]:4d} | FP: {cm[0,1]:4d}")
        print(f"  FN: {cm[1,0]:4d} | TP: {cm[1,1]:4d}")
    
    print("="*60)


# =============================================================================
# 🔄 K-FOLD CROSS-VALIDATION ORCHESTRATION
# =============================================================================

def run_kfold_training(
    dataset,
    config,
    model_name: Optional[str] = None,
    output_dir: Optional[Path] = None,
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Esegue K-Fold Cross-Validation come metodo di training esclusivo.

    Args:
        dataset: `StoryDatasetAdapter` già inizializzato con tokenizer.
        config: `ClassificationConfig` corrente.
        model_name: Opzionale, override del modello da usare.
        output_dir: Directory base per salvare i risultati dei fold.

    Returns:
        dict: Risultati per fold e metriche aggregate.
    """
    from src.classification.trainer import ModelTrainer
    from src.classification.model import MyModelWrapper

    if dataset is None:
        raise ValueError("Dataset non fornito per K-Fold training")
    if config is None:
        raise ValueError("Config non fornita per K-Fold training")

    labels = getattr(dataset, 'get_labels', None)
    if labels is None or not callable(labels):
        raise ValueError("Il dataset deve implementare get_labels() per la stratificazione")

    y = dataset.get_labels()
    n_samples = len(y)
    if config.K_FOLD_N_SPLITS < 2:
        raise ValueError("K_FOLD_N_SPLITS deve essere >= 2")
    if config.K_FOLD_N_SPLITS > n_samples:
        raise ValueError("K_FOLD_N_SPLITS non può superare il numero di campioni")

    # Setup K-Fold (stratificato per default)
    if config.K_FOLD_STRATIFIED:
        skf = StratifiedKFold(
            n_splits=config.K_FOLD_N_SPLITS,
            shuffle=config.K_FOLD_SHUFFLE,
            random_state=config.K_FOLD_RANDOM_STATE,
        )
        split_iter = skf.split(np.arange(n_samples), y)
    else:
        # Fallback: usa KFold standard se non stratificato
        from sklearn.model_selection import KFold
        kf = KFold(
            n_splits=config.K_FOLD_N_SPLITS,
            shuffle=config.K_FOLD_SHUFFLE,
            random_state=config.K_FOLD_RANDOM_STATE,
        )
        split_iter = kf.split(np.arange(n_samples))

    base_output_dir = Path(output_dir or (config.get_results_dir() / "kfold_training"))
    base_output_dir.mkdir(parents=True, exist_ok=True)

    fold_results: List[Dict[str, Any]] = []
    start_ts = datetime.now().isoformat()

    print("\n" + "=" * 60)
    print("🚀 AVVIO K-FOLD CROSS-VALIDATION")
    print("=" * 60)
    print(f"  • Modello: {model_name or config.MODEL_NAME}")
    print(f"  • Fold: {config.K_FOLD_N_SPLITS}")
    print(f"  • Stratified: {config.K_FOLD_STRATIFIED}")

    # Determina tokenizer da usare (se non passato, prova a recuperarlo dal dataset)
    tokenizer_to_use = tokenizer
    if tokenizer_to_use is None:
        try:
            tokenizer_to_use = getattr(dataset, 'tokenizer', None)
        except Exception:
            tokenizer_to_use = None

    # Salva tokenizer nella directory base per accesso globale
    if tokenizer_to_use is not None:
        try:
            tokenizer_base_path = base_output_dir / "tokenizer"
            tokenizer_base_path.mkdir(exist_ok=True)

            # Se è un CustomTokenizer, usa get_tokenizer()
            if hasattr(tokenizer_to_use, 'get_tokenizer'):
                tokenizer_to_use.get_tokenizer().save_pretrained(str(tokenizer_base_path))
            else:
                tokenizer_to_use.save_pretrained(str(tokenizer_base_path))

            print(f"✅ Tokenizer salvato in: {tokenizer_base_path}")
            print(f"   Questo tokenizer sarà accessibile a tutti i fold")
        except Exception as e:
            print(f"⚠️ Impossibile salvare tokenizer nella directory base: {e}")
            print(f"   Il tokenizer dovrà essere ricreato per ogni fold")

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
        print(f"\n🔄 FOLD {fold_idx + 1}/{config.K_FOLD_N_SPLITS}")
        print(f"  📄 Train size: {len(train_idx)} | Val size: {len(val_idx)}")

        # Stampa distribuzione classi per fold (stratificato)
        try:
            from collections import Counter
            train_counts = Counter([y[i] for i in train_idx])
            val_counts = Counter([y[i] for i in val_idx])
            def _fmt_counts(counts: Dict[int, int], total: int) -> str:
                parts = []
                for cls in sorted(counts.keys()):
                    pct = (counts[cls] / total * 100) if total > 0 else 0.0
                    parts.append(f"class {cls}: {counts[cls]} ({pct:.1f}%)")
                return ", ".join(parts) if parts else "n/a"
            print(f"  🎯 Distribuzione Train: {_fmt_counts(train_counts, len(train_idx))}")
            print(f"  🎯 Distribuzione Val: {_fmt_counts(val_counts, len(val_idx))}")
        except Exception as e:
            print(f"  ⚠️ Impossibile calcolare distribuzione classi per fold {fold_idx + 1}: {e}")

        # Crea subset per fold
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Inizializza nuovo modello per il fold
        wrapper = MyModelWrapper(
            num_labels=config.NUM_LABELS,
            output_attentions=True,
            model_name=model_name or config.MODEL_NAME,
        )

        # Allinea embeddings al tokenizer (evita token-id fuori range → assert CUDA)
        if tokenizer_to_use is not None:
            try:
                resize_info = wrapper.resize_embeddings_for_tokenizer(tokenizer_to_use)
                print(f"🔧 Embeddings ridimensionati per fold {fold_idx + 1}: +{resize_info.get('tokens_added', 0)} token")
            except Exception as e:
                print(f"⚠️ Impossibile ridimensionare embeddings per fold {fold_idx + 1}: {e}")

        # Directory per il fold
        fold_dir = base_output_dir / f"fold_{fold_idx + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Trainer per il fold
        trainer = ModelTrainer(
            model_wrapper=wrapper,
            train_dataset=cast(StoryDatasetAdapter, train_subset),
            val_dataset=cast(StoryDatasetAdapter, val_subset),
            config=config,
            output_dir=fold_dir,
        )

        # Esegui training del fold
        fold_start = time.time()
        training_summary = trainer.train()
        training_time = time.time() - fold_start

        # Valutazione finale su validation del fold
        final_metrics = trainer._evaluate_final_model()

        # Salva risultati fold
        fold_info = {
            'fold_idx': fold_idx,
            'train_size': int(len(train_idx)),
            'val_size': int(len(val_idx)),
            'best_mcc': training_summary.get('best_mcc', None),
            'best_balanced_acc': training_summary.get('best_balanced_acc', None),
            'best_class1_focus': training_summary.get('best_class1_focus', None),
            'final_metrics': final_metrics,
            'training_time': training_time,
            'epochs_completed': training_summary.get('epochs_completed', None),
            'early_stopped': training_summary.get('early_stopped', False),
            'stop_reason': training_summary.get('stop_reason', None),
        }
        fold_results.append(fold_info)

        # Persisti risultati fold in JSON
        _save_json(fold_info, fold_dir / "fold_results.json")

    # Aggrega metriche
    aggregated = aggregate_kfold_metrics(fold_results)
    aggregated['started_at'] = start_ts
    aggregated['completed_at'] = datetime.now().isoformat()
    aggregated['n_folds'] = config.K_FOLD_N_SPLITS
    aggregated['model_name'] = model_name or config.MODEL_NAME

    # Salva aggregati
    _save_json({'fold_results': fold_results, 'aggregated_metrics': aggregated}, base_output_dir / "kfold_summary.json")

    print("\n" + "=" * 60)
    print("✅ K-FOLD COMPLETATO")
    print(f"📄 Risultati: {base_output_dir}")
    print("=" * 60)

    return {
        'fold_results': fold_results,
        'aggregated_metrics': aggregated,
        'output_dir': str(base_output_dir),
    }


def aggregate_kfold_metrics(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggrega metriche chiave (mean/std) dai risultati dei fold."""
    import numpy as np

    # Colleziona metriche comuni
    keys_map = {
        'accuracy': 'accuracy',
        'balanced_accuracy': 'balanced_accuracy',
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall',
        'mcc': 'mcc',
    }

    agg: Dict[str, Any] = {}
    for out_key, metric_key in keys_map.items():
        values = []
        for fr in fold_results:
            fm = fr.get('final_metrics', {})
            # support both 'auc_roc' and 'auc'
            val = fm.get(metric_key, fm.get('auc' if metric_key == 'auc' else metric_key))
            if isinstance(val, (int, float)):
                values.append(float(val))
        if values:
            agg[f'{out_key}_mean'] = float(np.mean(values))
            agg[f'{out_key}_std'] = float(np.std(values))

    # Bests across folds
    best_mccs = [fr.get('best_mcc') for fr in fold_results]
    best_baccs = [fr.get('best_balanced_acc') for fr in fold_results]
    best_c1 = [fr.get('best_class1_focus') for fr in fold_results]
    if any(v is not None for v in best_mccs):
        agg['best_mcc_overall'] = float(np.max([float(v) for v in best_mccs if v is not None]))
    if any(v is not None for v in best_baccs):
        agg['best_balanced_acc_overall'] = float(np.max([float(v) for v in best_baccs if v is not None]))
    if any(v is not None for v in best_c1):
        agg['best_class1_focus_overall'] = float(np.max([float(v) for v in best_c1 if v is not None]))

    return agg


def _save_json(data: Dict[str, Any], path: Path):
    """Utility per salvare file JSON con encoding UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# =============================================================================
# 🧪 TESTING
# =============================================================================

if __name__ == "__main__":
    print("🧪 Test ML Utilities...")
    
    # Test metriche
    y_true = [0, 1, 0, 1, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 1, 0, 1, 1, 0]
    y_proba = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.9, 0.1]
    
    metrics = calculate_comprehensive_metrics(y_true, y_pred, y_proba)
    print_metrics_report(metrics, "Test Metrics")
    
    # Test early stopping
    early_stop = EarlyStopping(patience=3)
    print(f"\n⏱️ Early stopping creato: patience={early_stop.patience}")
    
    print("\n✅ Test completato!")
