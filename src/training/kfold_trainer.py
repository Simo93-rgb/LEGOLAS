"""
K-Fold Cross Validation Trainer for LEGOLAS
Wrapper per training con K-Fold CV stratificato
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple
import json
from datetime import datetime
from torch.utils.data import TensorDataset
import torch.optim as optim


from src.training.config import TrainingConfig
from src.training.utils import create_stratified_kfold, compute_metrics, save_metrics
from src.training.checkpoint import ModelCheckpoint
from src.training.early_stopping import EarlyStopping


class KFoldTrainer:
    """
    K-Fold Cross Validation Trainer
    
    Gestisce training con K-Fold stratificato:
    1. Split train+val in K folds stratificati
    2. Per ogni fold:
       - Training con ModelCheckpoint e EarlyStopping
       - Salva best model: best_model_{format}_{model}_fold{k}.pth
       - Calcola metriche validation
    3. Aggrega risultati: mean ± std per ogni metrica
    
    Args:
        config: TrainingConfig instance
        train_func: Funzione di training per singolo fold
                   Signature: train_func(model, train_dataset, val_dataset, fold) -> metrics
        model_factory: Factory per creare nuovo modello per ogni fold
        dataset: Dataset completo (train+val, test escluso)
        labels: Array di label per stratificazione
        verbose: Se True, stampa progresso
        
    Example:
        >>> config = TrainingConfig(use_kfold=True, n_folds=5)
        >>> kfold_trainer = KFoldTrainer(
        ...     config=config,
        ...     train_func=my_train_function,
        ...     model_factory=lambda: create_model(),
        ...     dataset=train_val_dataset,
        ...     labels=train_val_labels
        ... )
        >>> results = kfold_trainer.run()
        >>> print(f"Mean accuracy: {results['mean']['accuracy']:.4f}")
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        train_func: Callable,
        model_factory: Callable[[], nn.Module],
        dataset: Dataset,
        labels: np.ndarray,
        verbose: bool = True
    ):
        self.config = config
        self.train_func = train_func
        self.model_factory = model_factory
        self.dataset = dataset
        self.labels = labels
        self.verbose = verbose
        
        # Validate
        if not config.use_kfold:
            raise ValueError("Config deve avere use_kfold=True")
        
        if len(labels) != len(dataset):
            raise ValueError(f"Mismatch: labels {len(labels)} vs dataset {len(dataset)}")
        
        # Setup K-Fold
        self.kfold = create_stratified_kfold(
            n_splits=config.n_folds,
            shuffle=config.kfold_shuffle,
            random_state=config.seed
        )
        
        # Results storage
        self.fold_results: List[Dict[str, Any]] = []
        self.fold_models: List[Path] = []
    
    def run(self) -> Dict[str, Any]:
        """
        Esegue K-Fold training
        
        Returns:
            Dict con risultati aggregati:
            - 'fold_results': Lista metriche per fold
            - 'mean': Metriche medie
            - 'std': Std metriche
            - 'fold_models': Path modelli salvati
        """
        
        if self.verbose:
            print("\n" + "=" * 80)
            print(f"  🔄 K-FOLD CROSS VALIDATION: {self.config.n_folds} FOLDS")
            print("=" * 80)
        
        # Loop sui fold
        for fold_idx, (train_indices, val_indices) in enumerate(self.kfold.split(
            np.arange(len(self.dataset)),
            self.labels
        )):
            if self.verbose:
                print(f"\n{'=' * 80}")
                print(f"  📊 FOLD {fold_idx + 1}/{self.config.n_folds}")
                print(f"{'=' * 80}")
                print(f"   Train samples: {len(train_indices)}")
                print(f"   Val samples:   {len(val_indices)}")
                
                # Class distribution
                train_labels = self.labels[train_indices]
                val_labels = self.labels[val_indices]
                train_dist = np.bincount(train_labels) / len(train_labels) * 100
                val_dist = np.bincount(val_labels) / len(val_labels) * 100
                
                print(f"   Train class 0: {train_dist[0]:.1f}%, class 1: {train_dist[1]:.1f}%")
                print(f"   Val   class 0: {val_dist[0]:.1f}%, class 1: {val_dist[1]:.1f}%")
            
            # Crea subset
            train_dataset = Subset(self.dataset, train_indices)
            val_dataset = Subset(self.dataset, val_indices)
            
            # Crea nuovo modello per questo fold
            model = self.model_factory()
            
            # Setup checkpoint e early stopping
            checkpoint = ModelCheckpoint(
                save_dir=self.config.models_dir,
                metric=self.config.checkpoint_metric,
                mode=self.config.checkpoint_mode,
                fold=fold_idx,
                model_name=self.config.model_name,
                story_format=self.config.story_format,
                verbose=self.verbose
            )
            
            early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta,
                use_loss_ratio=self.config.use_loss_ratio_monitoring,
                loss_ratio_threshold=self.config.get_loss_ratio_threshold(),
                loss_ratio_patience=self.config.loss_ratio_patience,
                restore_best_weights=True,
                verbose=self.verbose
            )
            
            # Training per questo fold
            fold_metrics = self.train_func(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                fold=fold_idx,
                checkpoint=checkpoint,
                early_stopping=early_stopping,
                config=self.config
            )
            
            # Salva risultati
            self.fold_results.append({
                "fold": fold_idx,
                "train_size": len(train_indices),
                "val_size": len(val_indices),
                **fold_metrics
            })
            
            # Salva model path
            if checkpoint.best_path is not None:
                self.fold_models.append(checkpoint.best_path)
            
            # Salva metriche fold
            fold_metrics_path = self.config.reports_dir / f"fold_{fold_idx}_{self.config.story_format}_{self.config.model_name}_metrics.json"
            save_metrics(fold_metrics, fold_metrics_path)
            
            if self.verbose:
                print(f"\n   📊 Fold {fold_idx} Results:")
                for key, value in fold_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"      {key:20s}: {value:.4f}")
        
        # Aggrega risultati
        aggregated = self._aggregate_results()
        
        # Salva risultati aggregati
        aggregated_path = self.config.reports_dir / f"kfold_aggregated_{self.config.story_format}_{self.config.model_name}_results.json"
        save_metrics(aggregated, aggregated_path)
        
        if self.verbose:
            self._print_aggregated_results(aggregated)
        
        return aggregated
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggrega metriche da tutti i fold"""
        
        # Estrai metriche numeriche
        metric_names = [k for k in self.fold_results[0].keys() 
                       if isinstance(self.fold_results[0][k], (int, float)) and k != "fold"]
        
        aggregated = {
            "n_folds": self.config.n_folds,
            "fold_results": self.fold_results,
            "fold_models": [str(p) for p in self.fold_models],
            "mean": {},
            "std": {},
            "min": {},
            "max": {}
        }
        
        # Calcola statistiche
        for metric in metric_names:
            values = [fold[metric] for fold in self.fold_results]
            aggregated["mean"][metric] = float(np.mean(values))
            aggregated["std"][metric] = float(np.std(values))
            aggregated["min"][metric] = float(np.min(values))
            aggregated["max"][metric] = float(np.max(values))
        
        return aggregated
    
    def _print_aggregated_results(self, aggregated: Dict[str, Any]):
        """Stampa risultati aggregati in formato leggibile"""
        
        print("\n" + "=" * 80)
        print(f"  📊 K-FOLD AGGREGATED RESULTS ({aggregated['n_folds']} folds)")
        print("=" * 80)
        
        print(f"\n{'Metric':<25} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 80)
        
        for metric in sorted(aggregated["mean"].keys()):
            mean = aggregated["mean"][metric]
            std = aggregated["std"][metric]
            min_val = aggregated["min"][metric]
            max_val = aggregated["max"][metric]
            
            print(f"{metric:<25} {mean:>10.4f}   {std:>10.4f}   {min_val:>10.4f}   {max_val:>10.4f}")
        
        print("\n" + "=" * 80)
        print(f"  💾 Modelli salvati: {len(self.fold_models)}")
        for i, model_path in enumerate(self.fold_models):
            print(f"     Fold {i}: {model_path.name}")
        print("=" * 80)
    
    def load_fold_model(self, fold: int, device: Optional[torch.device] = None) -> nn.Module:
        """
        Carica modello da fold specifico
        
        Args:
            fold: Numero fold (0-indexed)
            device: Device su cui caricare
            
        Returns:
            Modello caricato
        """
        if fold >= len(self.fold_models):
            raise ValueError(f"Fold {fold} non disponibile. Folds: 0-{len(self.fold_models)-1}")
        
        model_path = self.fold_models[fold]
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model non trovato: {model_path}")
        
        # Crea nuovo modello
        model = self.model_factory()
        
        # Carica checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if self.verbose:
            print(f"✅ Modello fold {fold} caricato: {model_path.name}")
        
        return model
    
    def get_best_fold(self, metric: str = "balanced_accuracy") -> int:
        """
        Identifica fold con metrica migliore
        
        Args:
            metric: Metrica da usare
            
        Returns:
            Indice del fold migliore
        """
        values = [fold[metric] for fold in self.fold_results]
        best_idx = int(np.argmax(values))
        
        if self.verbose:
            print(f"Best fold: {best_idx}, {metric}: {values[best_idx]:.4f}")
        
        return best_idx


# =============================================================================
# 🏭 HELPER FUNCTIONS
# =============================================================================

def save_kfold_summary(
    kfold_results: Dict[str, Any],
    save_path: Path,
    config: TrainingConfig
):
    """
    Salva summary del K-Fold training
    
    Args:
        kfold_results: Risultati da KFoldTrainer.run()
        save_path: Path file di output
        config: TrainingConfig usata
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_name": config.model_name,
            "story_format": config.story_format,
            "n_folds": config.n_folds,
            "loss_function": config.loss_function,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs
        },
        "results": kfold_results
    }
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)


# =============================================================================
# 🧪 TESTING
# =============================================================================

if __name__ == "__main__":

    
    print("🧪 Testing KFoldTrainer...\n")
    
    # Setup dummy data
    n_samples = 100
    n_features = 10
    n_classes = 2
    
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    dataset = TensorDataset(X, y)
    labels = y.numpy()
    
    # Dummy model factory
    def create_model():
        return nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )
    
    # Dummy train function
    def dummy_train(model, train_dataset, val_dataset, fold, checkpoint, early_stopping, config):
        print(f"   🏋️  Training fold {fold}...")
        
        # Simula training rapido
        device = config.get_device()
        model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8)
        
        # 3 epoche solo per test
        for epoch in range(3):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    outputs = model(batch_X)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(batch_y.numpy())
            
            # Metriche
            metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
            
            # Update checkpoint
            checkpoint.update(model, optimizer, epoch, metrics)
        
        return metrics
    
    # Test KFoldTrainer
    config = TrainingConfig(use_kfold=True, n_folds=3, num_epochs=3)
    
    kfold_trainer = KFoldTrainer(
        config=config,
        train_func=dummy_train,
        model_factory=create_model,
        dataset=dataset,
        labels=labels,
        verbose=True
    )
    
    results = kfold_trainer.run()
    
    print("\n✅ Test completato!")
