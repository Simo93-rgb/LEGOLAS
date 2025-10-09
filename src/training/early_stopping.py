"""
Early Stopping with Model State Buffer for LEGOLAS
Implementa early stopping avanzato con ripristino pesi epoca trigger - patience
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Deque
from collections import deque
import copy


class EarlyStopping:
    """
    Early Stopping con buffer di stati modello
    
    Features:
    - Monitoraggio validation loss con min_delta
    - Ratio monitoring (train_loss / val_loss) per rilevare overfitting
    - Buffer degli ultimi N stati modello (N = patience)
    - Ripristino pesi a epoca trigger_epoch - patience
    
    Args:
        patience: Epoche da aspettare prima di fermare
        min_delta: Miglioramento minimo richiesto su val_loss
        use_loss_ratio: Se True, monitora anche train/val ratio
        loss_ratio_threshold: Soglia ratio oltre cui considerare overfitting
        loss_ratio_patience: Epoche consecutive sopra threshold
        restore_best_weights: Se True, ripristina pesi migliori al trigger
        verbose: Se True, stampa informazioni
        
    Example:
        >>> early_stop = EarlyStopping(patience=5, min_delta=0.001)
        >>> for epoch in range(num_epochs):
        ...     train_loss, val_loss = train_epoch(...)
        ...     early_stop.update(val_loss, train_loss, model)
        ...     if early_stop.should_stop():
        ...         print(f"Early stopping at epoch {epoch}")
        ...         early_stop.restore_weights(model)
        ...         break
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        use_loss_ratio: bool = True,
        loss_ratio_threshold: float = 1.15,
        loss_ratio_patience: int = 3,
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.use_loss_ratio = use_loss_ratio
        self.loss_ratio_threshold = loss_ratio_threshold
        self.loss_ratio_patience = loss_ratio_patience
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        # State tracking
        self.best_val_loss = np.inf
        self.best_epoch = -1
        self.wait_count = 0
        self.ratio_violations = 0
        self.trigger_epoch = -1
        self.stopped = False
        
        # Model states buffer (ultimi N stati dove N = patience)
        self.model_states_buffer: Deque[Dict[str, Any]] = deque(maxlen=patience)
        self.best_model_state: Optional[Dict[str, Any]] = None
        
        # History
        self.val_loss_history: list[float] = []
        self.train_loss_history: list[float] = []
        self.ratio_history: list[float] = []
    
    def _save_model_state(self, model: torch.nn.Module, epoch: int) -> Dict[str, Any]:
        """
        Saves a deep copy of the model's state dictionary along with the current epoch.

        Args:
            model (torch.nn.Module): The model whose state is to be saved.
            epoch (int): The current epoch number.

        Returns:
            Dict[str, Any]: A dictionary containing the epoch and a deep copy of the model's state dictionary.
        """
        """Salva lo stato del modello con deep copy"""
        return {
            "epoch": epoch,
            "state_dict": copy.deepcopy(model.state_dict())
        }
    
    def update(
        self,
        val_loss: float,
        train_loss: Optional[float] = None,
        model: Optional[torch.nn.Module] = None
    ) -> bool:
        """
        Aggiorna early stopping state
        
        Args:
            val_loss: Validation loss corrente
            train_loss: Training loss corrente (opzionale, per ratio monitoring)
            model: Modello (opzionale, per buffer states)
            
        Returns:
            True se val_loss migliora, False altrimenti
        """
        
        epoch = len(self.val_loss_history)
        
        # Aggiungi a history
        self.val_loss_history.append(val_loss)
        if train_loss is not None:
            self.train_loss_history.append(train_loss)
            ratio = train_loss / val_loss if val_loss > 0 else 0.0
            self.ratio_history.append(ratio)
        
        # Salva model state nel buffer se fornito
        if model is not None:
            state = self._save_model_state(model, epoch)
            self.model_states_buffer.append(state)
        
        # Check se val_loss migliora
        improved = False
        if val_loss < (self.best_val_loss - self.min_delta):
            improved = True
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.wait_count = 0
            self.ratio_violations = 0
            
            # Salva best model state
            if model is not None:
                self.best_model_state = self._save_model_state(model, epoch)
            
            if self.verbose:
                print(f"   ✅ Validation loss migliorata: {val_loss:.6f}")
        else:
            self.wait_count += 1
            if self.verbose and self.wait_count > 0:
                print(f"   ⏳ No improvement: {self.wait_count}/{self.patience}")
        
        # Check ratio monitoring
        if self.use_loss_ratio and train_loss is not None:
            ratio = train_loss / val_loss if val_loss > 0 else 0.0
            
            if ratio < self.loss_ratio_threshold:
                # Ratio OK, reset violations
                if self.ratio_violations > 0:
                    if self.verbose:
                        print(f"   ✅ Loss ratio OK: {ratio:.3f} < {self.loss_ratio_threshold:.3f}")
                self.ratio_violations = 0
            else:
                # Ratio violation
                self.ratio_violations += 1
                if self.verbose:
                    print(f"   ⚠️  Loss ratio violation: {ratio:.3f} > {self.loss_ratio_threshold:.3f} ({self.ratio_violations}/{self.loss_ratio_patience})")
        
        return improved
    
    def should_stop(self) -> bool:
        """
        Determina se training deve fermarsi
        
        Returns:
            True se condizioni early stopping soddisfatte
        """
        
        if self.stopped:
            return True
        
        # Condizione 1: Patience esaurita (no improvement)
        if self.wait_count >= self.patience:
            self.trigger_epoch = len(self.val_loss_history) - 1
            self.stopped = True
            if self.verbose:
                print(f"\n   🛑 EARLY STOPPING: Patience esaurita ({self.patience} epoche senza miglioramento)")
                print(f"      Best epoch: {self.best_epoch}, Val loss: {self.best_val_loss:.6f}")
                print(f"      Trigger epoch: {self.trigger_epoch}")
            return True
        
        # Condizione 2: Loss ratio violations (overfitting detection)
        if self.use_loss_ratio and self.ratio_violations >= self.loss_ratio_patience:
            self.trigger_epoch = len(self.val_loss_history) - 1
            self.stopped = True
            if self.verbose:
                print(f"\n   🛑 EARLY STOPPING: Loss ratio troppo alto ({self.loss_ratio_patience} epoche consecutive)")
                print(f"      Possibile overfitting rilevato")
                print(f"      Best epoch: {self.best_epoch}, Val loss: {self.best_val_loss:.6f}")
                print(f"      Trigger epoch: {self.trigger_epoch}")
            return True
        
        return False
    
    def restore_weights(self, model: torch.nn.Module) -> int:
        """
        Ripristina pesi modello a epoca trigger - patience
        
        Se restore_best_weights=True e best_model_state disponibile,
        ripristina al best epoch. Altrimenti usa buffer.
        
        Args:
            model: Modello da ripristinare
            
        Returns:
            Epoca a cui sono stati ripristinati i pesi
        """
        
        if not self.stopped:
            raise RuntimeError("Cannot restore weights: early stopping non triggered")
        
        # Opzione 1: Ripristina best state se disponibile
        if self.restore_best_weights and self.best_model_state is not None:
            model.load_state_dict(self.best_model_state["state_dict"])
            restored_epoch = self.best_model_state["epoch"]
            if self.verbose:
                print(f"   🔄 Pesi ripristinati al BEST epoch: {restored_epoch}")
            return restored_epoch
        
        # Opzione 2: Usa buffer (trigger - patience)
        if len(self.model_states_buffer) == 0:
            if self.verbose:
                print(f"   ⚠️  Buffer vuoto, impossibile ripristinare pesi")
            return self.trigger_epoch
        
        # Calcola target epoch
        target_epoch = max(0, self.trigger_epoch - self.patience)
        
        # Trova stato più vicino nel buffer
        best_match = None
        min_diff = float('inf')
        
        for state in self.model_states_buffer:
            diff = abs(state["epoch"] - target_epoch)
            if diff < min_diff:
                min_diff = diff
                best_match = state
        
        if best_match is not None:
            model.load_state_dict(best_match["state_dict"])
            restored_epoch = best_match["epoch"]
            if self.verbose:
                print(f"   🔄 Pesi ripristinati a epoca {restored_epoch} (target: {target_epoch})")
            return restored_epoch
        
        if self.verbose:
            print(f"   ⚠️  Impossibile trovare stato nel buffer")
        return self.trigger_epoch
    
    def get_state(self) -> Dict[str, Any]:
        """Restituisce stato corrente di early stopping"""
        return {
            "best_val_loss": float(self.best_val_loss),
            "best_epoch": int(self.best_epoch),
            "wait_count": int(self.wait_count),
            "ratio_violations": int(self.ratio_violations),
            "trigger_epoch": int(self.trigger_epoch),
            "stopped": bool(self.stopped),
            "buffer_size": len(self.model_states_buffer),
            "has_best_state": self.best_model_state is not None
        }
    
    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, min_delta={self.min_delta}, "
            f"best_loss={self.best_val_loss:.4f}, wait={self.wait_count}/{self.patience}, "
            f"stopped={self.stopped})"
        )


# =============================================================================
# 🏭 FACTORY FUNCTION
# =============================================================================

def create_early_stopping_from_config(config) -> EarlyStopping:
    """
    Crea EarlyStopping da TrainingConfig
    
    Args:
        config: TrainingConfig instance
        
    Returns:
        EarlyStopping configurato
    """
    return EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        use_loss_ratio=config.use_loss_ratio_monitoring,
        loss_ratio_threshold=config.get_loss_ratio_threshold(),
        loss_ratio_patience=config.loss_ratio_patience,
        restore_best_weights=True,
        verbose=True
    )


# =============================================================================
# 🧪 TESTING
# =============================================================================

if __name__ == "__main__":
    import torch.nn as nn
    
    print("🧪 Testing EarlyStopping...\n")
    
    # Setup test
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
    
    model = DummyModel()
    
    # Test 1: Patience esaurita
    print("=" * 80)
    print("📋 Test 1: Early stopping per patience esaurita")
    print("=" * 80)
    
    early_stop = EarlyStopping(patience=3, min_delta=0.01, use_loss_ratio=False)
    
    # Simula val_loss che non migliora
    val_losses = [0.5, 0.49, 0.50, 0.51, 0.52, 0.53]
    
    for epoch, val_loss in enumerate(val_losses):
        print(f"\nEpoch {epoch}: val_loss = {val_loss:.4f}")
        early_stop.update(val_loss, model=model)
        
        if early_stop.should_stop():
            print("\n" + "=" * 80)
            restored_epoch = early_stop.restore_weights(model)
            print(f"   Pesi ripristinati a epoca: {restored_epoch}")
            print("=" * 80)
            break
    
    # Test 2: Ratio violations (overfitting)
    print("\n\n" + "=" * 80)
    print("📋 Test 2: Early stopping per loss ratio violations")
    print("=" * 80)
    
    early_stop2 = EarlyStopping(
        patience=5,
        min_delta=0.01,
        use_loss_ratio=True,
        loss_ratio_threshold=1.5,
        loss_ratio_patience=3
    )
    
    model2 = DummyModel()
    
    # Simula overfitting: train_loss scende, val_loss sale
    train_losses = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1]
    val_losses2 = [0.5, 0.48, 0.50, 0.52, 0.55, 0.60]
    
    for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses2)):
        ratio = train_loss / val_loss
        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, ratio={ratio:.3f}")
        early_stop2.update(val_loss, train_loss, model2)
        
        if early_stop2.should_stop():
            print("\n" + "=" * 80)
            restored_epoch = early_stop2.restore_weights(model2)
            print(f"   Pesi ripristinati a epoca: {restored_epoch}")
            print("=" * 80)
            break
    
    # Test 3: No early stopping (training normale)
    print("\n\n" + "=" * 80)
    print("📋 Test 3: Training normale (no early stopping)")
    print("=" * 80)
    
    early_stop3 = EarlyStopping(patience=3, min_delta=0.01)
    model3 = DummyModel()
    
    # Simula val_loss che migliora continuamente
    val_losses3 = [0.5, 0.4, 0.3, 0.2, 0.1]
    
    for epoch, val_loss in enumerate(val_losses3):
        print(f"\nEpoch {epoch}: val_loss = {val_loss:.4f}")
        early_stop3.update(val_loss, model=model3)
        
        if early_stop3.should_stop():
            print("Early stopping triggered!")
            break
    else:
        print("\n✅ Training completato senza early stopping")
    
    print("\n✅ Test completato!")
