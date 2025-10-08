"""
Model Checkpoint Manager for LEGOLAS
Salva solo il best model basato su balanced_accuracy
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Literal, Dict, Any
import json
from datetime import datetime


class ModelCheckpoint:
    """
    Gestisce il salvataggio del miglior modello durante il training
    
    Traccia la metrica specificata (default: balanced_accuracy) e salva
    il modello solo quando migliora rispetto al best precedente.
    
    Args:
        save_dir: Directory dove salvare i modelli
        metric: Metrica da monitorare ('balanced_accuracy', 'f1', 'accuracy', etc.)
        mode: 'max' se metrica più alta è migliore, 'min' per loss
        fold: Numero fold (None per training semplice)
        model_name: Nome modello (es: 'clinical-bert')
        story_format: Formato storie (es: 'narrativo')
        verbose: Se True, stampa informazioni
        
    Example:
        >>> checkpoint = ModelCheckpoint(
        ...     save_dir=Path("output/models"),
        ...     metric="balanced_accuracy",
        ...     mode="max",
        ...     fold=1
        ... )
        >>> for epoch in range(num_epochs):
        ...     # Training...
        ...     metrics = {"balanced_accuracy": 0.85, "f1": 0.82}
        ...     checkpoint.update(model, optimizer, epoch, metrics)
    """
    
    def __init__(
        self,
        save_dir: Path | str,
        metric: str = "balanced_accuracy",
        mode: Literal["max", "min"] = "max",
        fold: Optional[int] = None,
        model_name: str = "clinical-bert",
        story_format: str = "narrativo",
        verbose: bool = True
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.metric = metric
        self.mode = mode
        self.fold = fold
        self.model_name = model_name
        self.story_format = story_format
        self.verbose = verbose
        
        # Best tracking
        self.best_value = -np.inf if mode == "max" else np.inf
        self.best_epoch = -1
        self.best_path: Optional[Path] = None
        
        # History
        self.history: list[Dict[str, Any]] = []
        
    def _is_better(self, current: float) -> bool:
        """Verifica se current è migliore di best"""
        if self.mode == "max":
            return current > self.best_value
        else:
            return current < self.best_value
    
    def _get_model_path(self, epoch: Optional[int] = None) -> Path:
        """Genera path per salvare il modello"""
        fold_str = f"_fold{self.fold}" if self.fold is not None else ""
        epoch_str = f"_epoch{epoch}" if epoch is not None else ""
        filename = f"best_model_{self.story_format}_{self.model_name}{fold_str}{epoch_str}.pth"
        return self.save_dir / filename
    
    def update(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        extra_state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Aggiorna checkpoint se metrica migliora
        
        Args:
            model: Modello PyTorch
            optimizer: Optimizer
            epoch: Numero epoca corrente
            metrics: Dict con metriche (deve contenere self.metric)
            extra_state: Stato aggiuntivo da salvare (es: scheduler, scaler)
            
        Returns:
            True se modello è stato salvato, False altrimenti
        """
        
        if self.metric not in metrics:
            raise KeyError(f"Metrica '{self.metric}' non trovata in metrics. Disponibili: {list(metrics.keys())}")
        
        current_value = metrics[self.metric]
        
        # Aggiungi a history
        self.history.append({
            "epoch": epoch,
            "metric": self.metric,
            "value": current_value,
            **metrics
        })
        
        # Verifica se migliora
        if self._is_better(current_value):
            # Rimuovi vecchio best se esiste
            if self.best_path is not None and self.best_path.exists():
                self.best_path.unlink()
                if self.verbose:
                    print(f"   🗑️  Rimosso vecchio best: {self.best_path.name}")
            
            # Aggiorna best
            old_best = self.best_value
            self.best_value = current_value
            self.best_epoch = epoch
            self.best_path = self._get_model_path()
            
            # Prepara state da salvare
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "best_metric": self.metric,
                "best_value": self.best_value,
                "fold": self.fold,
                "model_name": self.model_name,
                "story_format": self.story_format,
                "timestamp": datetime.now().isoformat()
            }
            
            # Aggiungi extra_state se fornito
            if extra_state is not None:
                checkpoint_state.update(extra_state)
            
            # Salva
            torch.save(checkpoint_state, self.best_path)
            
            if self.verbose:
                improvement = current_value - old_best if self.mode == "max" else old_best - current_value
                print(f"   💾 Best model salvato: {self.best_path.name}")
                print(f"      {self.metric}: {old_best:.4f} → {current_value:.4f} (Δ {improvement:+.4f})")
            
            return True
        
        return False
    
    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Carica il best model salvato
        
        Args:
            model: Modello PyTorch (verrà modificato in-place)
            optimizer: Optimizer (opzionale, verrà modificato in-place se fornito)
            device: Device su cui caricare il modello
            
        Returns:
            Dict con checkpoint state (metrics, epoch, etc.)
        """
        
        if self.best_path is None or not self.best_path.exists():
            raise FileNotFoundError(f"Nessun best model trovato. Path: {self.best_path}")
        
        # Carica checkpoint
        checkpoint = torch.load(self.best_path, map_location=device)
        
        # Carica model state
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Carica optimizer state se richiesto
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.verbose:
            print(f"   ✅ Best model caricato: {self.best_path.name}")
            print(f"      Epoch: {checkpoint['epoch']}, {self.metric}: {checkpoint['best_value']:.4f}")
        
        return checkpoint
    
    def save_history(self, filename: str = "checkpoint_history.json") -> Path:
        """
        Salva history delle metriche su file JSON
        
        Args:
            filename: Nome file (verrà salvato in save_dir)
            
        Returns:
            Path del file salvato
        """
        fold_str = f"_fold{self.fold}" if self.fold is not None else ""
        filename_with_fold = filename.replace(".json", f"{fold_str}.json")
        save_path = self.save_dir / filename_with_fold
        
        history_data = {
            "metric": self.metric,
            "mode": self.mode,
            "best_value": float(self.best_value),
            "best_epoch": int(self.best_epoch),
            "fold": self.fold,
            "history": self.history
        }
        
        with open(save_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        if self.verbose:
            print(f"   💾 History salvata: {save_path.name}")
        
        return save_path
    
    def get_best_info(self) -> Dict[str, Any]:
        """Restituisce info sul best model"""
        return {
            "metric": self.metric,
            "best_value": float(self.best_value),
            "best_epoch": int(self.best_epoch),
            "best_path": str(self.best_path) if self.best_path else None,
            "fold": self.fold,
            "num_updates": len(self.history)
        }
    
    def __repr__(self) -> str:
        return (
            f"ModelCheckpoint(metric={self.metric}, mode={self.mode}, "
            f"best={self.best_value:.4f}, epoch={self.best_epoch}, fold={self.fold})"
        )


# =============================================================================
# 🏭 FACTORY FUNCTION
# =============================================================================

def create_checkpoint_from_config(
    config,
    fold: Optional[int] = None
) -> ModelCheckpoint:
    """
    Crea ModelCheckpoint da TrainingConfig
    
    Args:
        config: TrainingConfig instance
        fold: Numero fold (None per training semplice)
        
    Returns:
        ModelCheckpoint configurato
    """
    return ModelCheckpoint(
        save_dir=config.models_dir,
        metric=config.checkpoint_metric,
        mode=config.checkpoint_mode,
        fold=fold,
        model_name=config.model_name,
        story_format=config.story_format,
        verbose=True
    )


# =============================================================================
# 🧪 TESTING
# =============================================================================

if __name__ == "__main__":
    import torch.nn as nn
    import torch.optim as optim
    
    print("🧪 Testing ModelCheckpoint...\n")
    
    # Setup test
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
    
    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Test checkpoint
    checkpoint = ModelCheckpoint(
        save_dir=Path("output/models"),
        metric="balanced_accuracy",
        mode="max",
        fold=1,
        verbose=True
    )
    
    print("📋 Simulazione training con miglioramenti...")
    print("=" * 80)
    
    # Simula epoche con metriche in miglioramento
    metrics_sequence = [
        {"balanced_accuracy": 0.75, "f1": 0.70, "accuracy": 0.80},
        {"balanced_accuracy": 0.78, "f1": 0.73, "accuracy": 0.82},  # Migliora
        {"balanced_accuracy": 0.77, "f1": 0.72, "accuracy": 0.81},  # Peggiora
        {"balanced_accuracy": 0.82, "f1": 0.78, "accuracy": 0.85},  # Migliora
        {"balanced_accuracy": 0.80, "f1": 0.76, "accuracy": 0.83},  # Peggiora
    ]
    
    for epoch, metrics in enumerate(metrics_sequence):
        print(f"\nEpoch {epoch}: balanced_accuracy = {metrics['balanced_accuracy']:.4f}")
        saved = checkpoint.update(model, optimizer, epoch, metrics)
        if not saved:
            print(f"   ⏭️  Nessun miglioramento")
    
    print("\n" + "=" * 80)
    print("📊 Best Model Info:")
    print("=" * 80)
    for key, value in checkpoint.get_best_info().items():
        print(f"   {key:15s}: {value}")
    
    # Salva history
    print("\n" + "=" * 80)
    checkpoint.save_history()
    
    # Test load
    print("\n" + "=" * 80)
    print("📋 Test caricamento best model...")
    print("=" * 80)
    
    new_model = DummyModel()
    new_optimizer = optim.Adam(new_model.parameters(), lr=1e-3)
    checkpoint.load_best(new_model, new_optimizer)
    
    print("\n✅ Test completato!")
