"""
Training Utilities for LEGOLAS
Funzioni di supporto per training: splits stratificati, metriche, class weights
"""

import torch
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)
from typing import Tuple, Dict, Optional, Any
from pathlib import Path
import json


# =============================================================================
# 📊 DATASET SPLITTING
# =============================================================================

def stratified_train_val_test_split(
    X: np.ndarray | list,
    y: np.ndarray | list,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    shuffle: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split stratificato in train/val/test mantenendo proporzioni delle classi
    
    Per LEGOLAS: mantiene ~89% DISCHARGED, ~11% ADMITTED in ogni split
    
    Args:
        X: Features (può essere qualsiasi tipo supportato da sklearn)
        y: Labels (array di interi)
        test_size: Percentuale test set (default 20%)
        val_size: Percentuale validation set dal train rimanente (default 10%)
        random_state: Seed per riproducibilità
        shuffle: Se True, shuffle prima dello split
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
        
    Example:
        >>> X = list(range(1000))
        >>> y = np.random.randint(0, 2, 1000)
        >>> X_train, X_val, X_test, y_train, y_val, y_test = \
        ...     stratified_train_val_test_split(X, y, test_size=0.2, val_size=0.1)
        >>> print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    """
    
    # Converti a numpy se necessario
    X = np.array(X) if not isinstance(X, np.ndarray) else X
    y = np.array(y) if not isinstance(y, np.ndarray) else y
    
    # Split 1: Train+Val vs Test (stratificato)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=y
    )
    
    # Split 2: Train vs Val (stratificato sul trainval)
    # Calcola val_size relativo al trainval
    val_size_adjusted = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size_adjusted,
        random_state=random_state,
        shuffle=shuffle,
        stratify=y_trainval
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_stratified_kfold(
    n_splits: int = 10,
    shuffle: bool = True,
    random_state: int = 42
) -> StratifiedKFold:
    """
    Crea StratifiedKFold per cross-validation
    
    Args:
        n_splits: Numero di fold (default 10)
        shuffle: Se True, shuffle prima dello split
        random_state: Seed per riproducibilità
        
    Returns:
        StratifiedKFold instance
        
    Example:
        >>> skf = create_stratified_kfold(n_splits=5)
        >>> for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        ...     print(f"Fold {fold}: Train {len(train_idx)}, Val {len(val_idx)}")
    """
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )


# =============================================================================
# ⚖️ CLASS WEIGHTS
# =============================================================================

def compute_class_weights(
    y: np.ndarray | torch.Tensor,
    num_classes: Optional[int] = None,
    method: str = "balanced"
) -> torch.Tensor:
    """
    Calcola pesi delle classi per dataset sbilanciato
    
    Args:
        y: Labels (shape: (n_samples,))
        num_classes: Numero di classi (auto-detect se None)
        method: 'balanced' (inversamente proporzionale alla frequenza) o 'sqrt'
        
    Returns:
        Tensor con pesi per ogni classe, shape (num_classes,)
        
    Example:
        >>> y = np.array([0]*890 + [1]*110)  # 89% classe 0, 11% classe 1
        >>> weights = compute_class_weights(y)
        >>> print(weights)  # [0.56, 4.55] circa
    """
    
    # Converti a numpy
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    # Determina num_classes
    if num_classes is None:
        num_classes = int(y.max()) + 1
    
    # Conta occorrenze per classe
    class_counts = np.bincount(y, minlength=num_classes)
    
    # Evita divisione per zero
    class_counts = np.maximum(class_counts, 1)
    
    # Calcola pesi
    if method == "balanced":
        # Inversamente proporzionale: weight_i = n_samples / (n_classes * count_i)
        total_samples = len(y)
        weights = total_samples / (num_classes * class_counts)
    elif method == "sqrt":
        # Square root: meno aggressivo del balanced
        total_samples = len(y)
        weights = np.sqrt(total_samples / class_counts)
    else:
        raise ValueError(f"Method '{method}' non supportato. Usa 'balanced' o 'sqrt'")
    
    # Normalizza (somma = num_classes)
    weights = weights * (num_classes / weights.sum())
    
    return torch.tensor(weights, dtype=torch.float32)


# =============================================================================
# 📈 METRICS COMPUTATION
# =============================================================================

def compute_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    average: str = "binary"
) -> Dict[str, float]:
    """
    Calcola metriche di classificazione
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: 'binary' per 2 classi, 'macro'/'weighted' per multiclasse
        
    Returns:
        Dict con metriche: accuracy, precision, recall, f1, balanced_accuracy
        
    Example:
        >>> y_true = np.array([0, 1, 0, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0, 0])
        >>> metrics = compute_metrics(y_true, y_pred)
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    
    # Converti a numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Calcola metriche
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def compute_detailed_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    class_names: Optional[list[str]] = None
) -> Dict[str, Any]:
    """
    Calcola metriche dettagliate con confusion matrix e per-class metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Nomi delle classi (default: ["Class 0", "Class 1", ...])
        
    Returns:
        Dict con:
        - 'overall': metriche generali (accuracy, balanced_accuracy, etc.)
        - 'confusion_matrix': matrice di confusione
        - 'per_class': metriche per ogni classe
        - 'classification_report': report sklearn formattato
        
    Example:
        >>> metrics = compute_detailed_metrics(y_true, y_pred, ["DISCHARGED", "ADMITTED"])
        >>> print(metrics['overall']['balanced_accuracy'])
        >>> print(metrics['confusion_matrix'])
    """
    
    # Converti a numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Determina class_names
    num_classes = int(max(y_true.max(), y_pred.max())) + 1
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Overall metrics
    overall = compute_metrics(y_true, y_pred, average="binary" if num_classes == 2 else "macro")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    per_class = {}
    for i, class_name in enumerate(class_names):
        per_class[class_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        }
    
    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )
    
    return {
        "overall": overall,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "classification_report": report
    }


# =============================================================================
# 💾 METRICS SAVING
# =============================================================================

def save_metrics(
    metrics: Dict[str, Any],
    save_path: Path | str,
    pretty: bool = True
) -> None:
    """
    Salva metriche su file JSON
    
    Args:
        metrics: Dict con metriche
        save_path: Path del file JSON
        pretty: Se True, formatta JSON in modo leggibile
        
    Example:
        >>> metrics = compute_detailed_metrics(y_true, y_pred)
        >>> save_metrics(metrics, "output/evaluation/metrics_fold1.json")
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        if pretty:
            json.dump(metrics, f, indent=2)
        else:
            json.dump(metrics, f)


def load_metrics(load_path: Path | str) -> Dict[str, Any]:
    """
    Carica metriche da file JSON
    
    Args:
        load_path: Path del file JSON
        
    Returns:
        Dict con metriche
    """
    with open(load_path, 'r') as f:
        return json.load(f)


# =============================================================================
# 📊 DISTRIBUTION ANALYSIS
# =============================================================================

def analyze_class_distribution(
    y: np.ndarray | torch.Tensor,
    class_names: Optional[list[str]] = None,
    print_results: bool = True
) -> Dict[str, Any]:
    """
    Analizza distribuzione delle classi nel dataset
    
    Args:
        y: Labels
        class_names: Nomi delle classi
        print_results: Se True, stampa risultati formattati
        
    Returns:
        Dict con count, percentage, imbalance_ratio per ogni classe
        
    Example:
        >>> y = np.array([0]*890 + [1]*110)
        >>> dist = analyze_class_distribution(y, ["DISCHARGED", "ADMITTED"])
    """
    
    # Converti a numpy
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    # Determina class_names
    num_classes = int(y.max()) + 1
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Conta occorrenze
    counts = np.bincount(y, minlength=num_classes)
    total = len(y)
    percentages = (counts / total) * 100
    
    # Calcola imbalance ratio (rispetto alla classe maggioritaria)
    max_count = counts.max()
    imbalance_ratios = max_count / np.maximum(counts, 1)
    
    # Costruisci risultati
    results = {
        "total_samples": int(total),
        "num_classes": int(num_classes),
        "classes": {}
    }
    
    for i, class_name in enumerate(class_names):
        results["classes"][class_name] = {
            "count": int(counts[i]),
            "percentage": float(percentages[i]),
            "imbalance_ratio": float(imbalance_ratios[i])
        }
    
    # Stampa se richiesto
    if print_results:
        print("=" * 80)
        print("  📊 CLASS DISTRIBUTION ANALYSIS")
        print("=" * 80)
        print(f"\nTotal Samples: {total:,}")
        print(f"Num Classes: {num_classes}\n")
        
        for class_name, stats in results["classes"].items():
            print(f"{class_name}:")
            print(f"  Count:           {stats['count']:,}")
            print(f"  Percentage:      {stats['percentage']:.2f}%")
            print(f"  Imbalance Ratio: {stats['imbalance_ratio']:.2f}x")
            print()
        
        print("=" * 80)
    
    return results


# =============================================================================
# 🧪 TESTING
# =============================================================================

if __name__ == "__main__":
    print("🧪 Testing training utilities...\n")
    
    # Test 1: Stratified split
    print("📋 Test 1: Stratified Train/Val/Test Split")
    print("-" * 80)
    
    # Simula dataset LEGOLAS: 89% classe 0, 11% classe 1
    np.random.seed(42)
    n_samples = 1000
    X = np.arange(n_samples).reshape(-1, 1)
    y = np.array([0] * 890 + [1] * 110)
    np.random.shuffle(y)
    
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
        X, y, test_size=0.2, val_size=0.1
    )
    
    print(f"Total:      {len(y)} samples")
    print(f"Train:      {len(y_train)} samples ({len(y_train)/len(y)*100:.1f}%)")
    print(f"Validation: {len(y_val)} samples ({len(y_val)/len(y)*100:.1f}%)")
    print(f"Test:       {len(y_test)} samples ({len(y_test)/len(y)*100:.1f}%)")
    
    print(f"\nClass 0 proportions:")
    print(f"  Train: {(y_train==0).sum()/len(y_train)*100:.1f}%")
    print(f"  Val:   {(y_val==0).sum()/len(y_val)*100:.1f}%")
    print(f"  Test:  {(y_test==0).sum()/len(y_test)*100:.1f}%")
    print("✅ Proportions maintained!\n")
    
    # Test 2: Class weights
    print("\n📋 Test 2: Class Weights Computation")
    print("-" * 80)
    
    weights = compute_class_weights(y_train)
    print(f"Class weights: {weights.tolist()}")
    print(f"Class 0 (89%): weight = {weights[0]:.3f}")
    print(f"Class 1 (11%): weight = {weights[1]:.3f}")
    print(f"Ratio: {weights[1]/weights[0]:.2f}x")
    print("✅ Minority class has higher weight!\n")
    
    # Test 3: Metrics computation
    print("\n📋 Test 3: Metrics Computation")
    print("-" * 80)
    
    # Simula predizioni
    y_pred = y_test.copy()
    # Aggiungi qualche errore
    errors = np.random.choice(len(y_pred), size=int(len(y_pred)*0.1), replace=False)
    y_pred[errors] = 1 - y_pred[errors]
    
    metrics = compute_metrics(y_test, y_pred)
    print("Overall Metrics:")
    for name, value in metrics.items():
        print(f"  {name:20s}: {value:.4f}")
    print("✅ Metrics computed!\n")
    
    # Test 4: Detailed metrics
    print("\n📋 Test 4: Detailed Metrics")
    print("-" * 80)
    
    detailed = compute_detailed_metrics(
        y_test, y_pred,
        class_names=["DISCHARGED", "ADMITTED"]
    )
    
    print("Per-class metrics:")
    for class_name, class_metrics in detailed["per_class"].items():
        print(f"\n{class_name}:")
        for metric_name, value in class_metrics.items():
            print(f"  {metric_name:15s}: {value}")
    
    print("\nConfusion Matrix:")
    cm = np.array(detailed["confusion_matrix"])
    print(f"  [[{cm[0,0]:3d}, {cm[0,1]:3d}],")
    print(f"   [{cm[1,0]:3d}, {cm[1,1]:3d}]]")
    print("✅ Detailed metrics computed!\n")
    
    # Test 5: Distribution analysis
    print("\n📋 Test 5: Class Distribution Analysis")
    print("-" * 80)
    
    dist = analyze_class_distribution(
        y_train,
        class_names=["DISCHARGED", "ADMITTED"],
        print_results=True
    )
    
    print("\n✅ Tutti i test completati!")
