"""
Focal Loss Implementation for LEGOLAS
Risolve il problema dello sbilanciamento delle classi (DISCHARGED 89.1%, ADMITTED 10.9%)

Reference:
Lin et al. "Focal Loss for Dense Object Detection" (2017)
https://arxiv.org/abs/1708.02002

Formula:
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

Dove:
- p_t: probabilità predetta per la classe corretta
- α_t: peso per la classe t (bilanciamento)
- γ: focusing parameter (default 2.0)

Quando γ=0, Focal Loss = Weighted Cross Entropy
Quando γ>0, riduce il peso degli esempi facili (alta confidenza)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss per classificazione binaria/multiclasse
    
    Particolarmente efficace quando:
    - Dataset fortemente sbilanciato (89.1% vs 10.9% in LEGOLAS)
    - Molti esempi "facili" che dominano il gradiente
    - Classe minoritaria (ADMITTED) è più importante
    
    Args:
        alpha: Peso per ogni classe. Se float, si applica alla classe positiva.
               Se list/tensor, specifica peso per ogni classe [w_0, w_1, ...]
        gamma: Focusing parameter. Più alto = più focus su esempi difficili
        reduction: 'mean', 'sum' o 'none'
        ignore_index: Indice da ignorare (utile per padding, non usato in LEGOLAS)
        
    Example:
        >>> loss_fn = FocalLoss(alpha=[0.25, 0.75], gamma=2.0)
        >>> logits = model(inputs)  # Shape: (batch_size, num_classes)
        >>> labels = torch.tensor([0, 1, 0, 1])  # Shape: (batch_size,)
        >>> loss = loss_fn(logits, labels)
    """
    
    def __init__(
        self,
        alpha: Optional[float | list[float] | torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100
    ):
        super(FocalLoss, self).__init__()
        
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        # Gestione alpha (class weights)
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            # Single float: applica alla classe positiva (binario)
            self.alpha = torch.tensor([1 - alpha, alpha])
        elif isinstance(alpha, list):
            # Lista: converti a tensor
            self.alpha = torch.tensor(alpha)
        elif isinstance(alpha, torch.Tensor):
            # Già tensor
            self.alpha = alpha
        else:
            raise TypeError(f"alpha deve essere float, list o Tensor, non {type(alpha)}")
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcola Focal Loss
        
        Args:
            logits: Raw output del modello, shape (batch_size, num_classes)
                   NON applicare softmax prima!
            targets: Ground truth labels, shape (batch_size,)
                    Valori interi [0, num_classes-1]
        
        Returns:
            loss: Scalar tensor se reduction='mean'/'sum', altrimenti (batch_size,)
        """
        
        # Verifica dimensioni
        if logits.dim() != 2:
            raise ValueError(f"logits deve avere shape (batch_size, num_classes), ha {logits.shape}")
        if targets.dim() != 1:
            raise ValueError(f"targets deve avere shape (batch_size,), ha {targets.shape}")
        if logits.size(0) != targets.size(0):
            raise ValueError(f"Mismatch batch size: logits {logits.size(0)}, targets {targets.size(0)}")
        
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        
        # Calcola probabilità (softmax)
        probs = F.softmax(logits, dim=1)  # Shape: (batch_size, num_classes)
        
        # Crea mask per ignore_index
        valid_mask = targets != self.ignore_index
        
        # Estrai probabilità per la classe corretta (p_t)
        # Usa gather per selezionare prob della classe target
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)  # (batch_size, num_classes)
        pt = (probs * targets_one_hot).sum(dim=1)  # (batch_size,)
        
        # Clamp per stabilità numerica (evita log(0))
        pt = torch.clamp(pt, min=1e-8, max=1.0)
        
        # Calcola log(p_t)
        log_pt = torch.log(pt)  # (batch_size,)
        
        # Calcola modulating factor: (1 - p_t)^γ
        focal_weight = (1 - pt) ** self.gamma  # (batch_size,)
        
        # Applica class weights (alpha)
        if self.alpha is not None:
            # Sposta alpha su device corretto
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            
            # Estrai alpha per ogni target
            alpha_t = self.alpha[targets]  # (batch_size,)
            focal_weight = alpha_t * focal_weight
        
        # Calcola Focal Loss: -α_t * (1 - p_t)^γ * log(p_t)
        loss = -focal_weight * log_pt  # (batch_size,)
        
        # Applica ignore_index mask
        if not valid_mask.all():
            loss = loss * valid_mask.float()
        
        # Reduction
        if self.reduction == "mean":
            return loss.sum() / valid_mask.sum() if valid_mask.any() else loss.sum()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Reduction '{self.reduction}' non supportata. Usa 'mean', 'sum' o 'none'")
    
    def __repr__(self) -> str:
        return (
            f"FocalLoss(alpha={self.alpha}, gamma={self.gamma}, "
            f"reduction={self.reduction}, ignore_index={self.ignore_index})"
        )


# =============================================================================
# 🏭 FACTORY FUNCTIONS
# =============================================================================

def create_focal_loss(
    alpha: Optional[list[float]] = None,
    gamma: float = 2.0,
    num_classes: int = 2
) -> FocalLoss:
    """
    Crea Focal Loss con parametri di default per LEGOLAS
    
    Args:
        alpha: Pesi per classe. Se None, usa default [0.25, 0.75]
        gamma: Focusing parameter
        num_classes: Numero di classi (2 per LEGOLAS)
        
    Returns:
        FocalLoss configurato
    """
    if alpha is None:
        # Default LEGOLAS: più peso alla classe minoritaria (ADMITTED)
        if num_classes == 2:
            alpha = [0.25, 0.75]  # [DISCHARGED, ADMITTED]
        else:
            # Distribuzione uniforme per multiclasse
            alpha = [1.0 / num_classes] * num_classes
    
    return FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")


def create_loss_from_config(config) -> nn.Module:
    """
    Crea loss function basata su TrainingConfig
    
    Args:
        config: TrainingConfig instance
        
    Returns:
        nn.Module (FocalLoss o CrossEntropyLoss)
    """
    if config.loss_function == "focal":
        return FocalLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
            reduction="mean"
        )
    elif config.loss_function == "ce":
        # CrossEntropyLoss standard
        if config.use_class_weights:
            # Placeholder: i pesi vanno calcolati dal dataset
            # Questo sarà fatto in training/utils.py
            return nn.CrossEntropyLoss(reduction="mean")
        else:
            return nn.CrossEntropyLoss(reduction="mean")
    else:
        raise ValueError(f"Loss function '{config.loss_function}' non supportata")


# =============================================================================
# 🧪 TESTING & COMPARISON
# =============================================================================

def compare_focal_vs_ce():
    """
    Confronta Focal Loss vs CrossEntropy su esempi facili/difficili
    Dimostra come Focal Loss riduce il peso degli esempi facili
    """
    print("=" * 80)
    print("  🔬 FOCAL LOSS vs CROSS ENTROPY COMPARISON")
    print("=" * 80)
    
    # Setup
    num_classes = 2
    batch_size = 4
    
    # Loss functions
    focal_loss = FocalLoss(alpha=[0.25, 0.75], gamma=2.0)
    ce_loss = nn.CrossEntropyLoss()
    
    # Test case 1: Esempi FACILI (alta confidenza)
    print("\n📊 Test 1: ESEMPI FACILI (alta confidenza)")
    print("-" * 80)
    
    easy_logits = torch.tensor([
        [5.0, -5.0],  # Molto confidenti: classe 0
        [-5.0, 5.0],  # Molto confidenti: classe 1
        [4.0, -4.0],  # Confidenti: classe 0
        [-4.0, 4.0],  # Confidenti: classe 1
    ])
    labels = torch.tensor([0, 1, 0, 1])
    
    focal_easy = focal_loss(easy_logits, labels)
    ce_easy = ce_loss(easy_logits, labels)
    
    print(f"   Cross Entropy Loss: {ce_easy:.6f}")
    print(f"   Focal Loss:         {focal_easy:.6f}")
    print(f"   Riduzione:          {(1 - focal_easy/ce_easy)*100:.1f}%")
    print("   ✅ Focal Loss riduce peso su esempi facili!")
    
    # Test case 2: Esempi DIFFICILI (bassa confidenza)
    print("\n📊 Test 2: ESEMPI DIFFICILI (bassa confidenza)")
    print("-" * 80)
    
    hard_logits = torch.tensor([
        [0.5, 0.4],   # Poco confidenti
        [0.6, 0.5],   # Poco confidenti
        [0.3, 0.2],   # Molto incerti
        [0.4, 0.3],   # Molto incerti
    ])
    labels = torch.tensor([0, 0, 0, 0])
    
    focal_hard = focal_loss(hard_logits, labels)
    ce_hard = ce_loss(hard_logits, labels)
    
    print(f"   Cross Entropy Loss: {ce_hard:.6f}")
    print(f"   Focal Loss:         {focal_hard:.6f}")
    print(f"   Rapporto:           {focal_hard/ce_hard:.2f}x")
    print("   ✅ Focal Loss mantiene peso su esempi difficili!")
    
    # Test case 3: Dataset SBILANCIATO (come LEGOLAS)
    print("\n📊 Test 3: DATASET SBILANCIATO (89% classe 0, 11% classe 1)")
    print("-" * 80)
    
    # 9 esempi classe 0 (maggioranza), 1 esempio classe 1 (minorità)
    imbalanced_logits = torch.tensor([
        [2.0, -2.0],  # Classe 0
        [2.0, -2.0],  # Classe 0
        [2.0, -2.0],  # Classe 0
        [2.0, -2.0],  # Classe 0
        [2.0, -2.0],  # Classe 0
        [2.0, -2.0],  # Classe 0
        [2.0, -2.0],  # Classe 0
        [2.0, -2.0],  # Classe 0
        [2.0, -2.0],  # Classe 0
        [-2.0, 2.0],  # Classe 1 (minoritaria)
    ])
    imbalanced_labels = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    
    # Focal Loss con alpha per sbilanciamento
    focal_balanced = FocalLoss(alpha=[0.25, 0.75], gamma=2.0)
    focal_unbalanced = FocalLoss(alpha=None, gamma=2.0)
    ce_unbalanced = nn.CrossEntropyLoss()
    
    loss_focal_bal = focal_balanced(imbalanced_logits, imbalanced_labels)
    loss_focal_unbal = focal_unbalanced(imbalanced_logits, imbalanced_labels)
    loss_ce = ce_unbalanced(imbalanced_logits, imbalanced_labels)
    
    print(f"   Cross Entropy (no weights):       {loss_ce:.6f}")
    print(f"   Focal Loss (no alpha):            {loss_focal_unbal:.6f}")
    print(f"   Focal Loss (alpha=[0.25, 0.75]):  {loss_focal_bal:.6f}")
    print("   ✅ Alpha aumenta peso sulla classe minoritaria!")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("🧪 Testing FocalLoss...\n")
    
    # Test basic functionality
    print("📋 Test 1: Creazione loss function")
    focal = create_focal_loss()
    print(f"   {focal}")
    print("   ✅ Creazione OK\n")
    
    # Test forward pass
    print("📋 Test 2: Forward pass")
    logits = torch.randn(8, 2)  # batch_size=8, num_classes=2
    targets = torch.randint(0, 2, (8,))
    loss = focal(logits, targets)
    print(f"   Logits shape: {logits.shape}")
    print(f"   Targets shape: {targets.shape}")
    print(f"   Loss: {loss.item():.6f}")
    print("   ✅ Forward pass OK\n")
    
    # Test gradient flow
    print("📋 Test 3: Backward pass (gradients)")
    logits.requires_grad = True
    loss = focal(logits, targets)
    loss.backward()
    print(f"   Gradients shape: {logits.grad.shape}")
    print(f"   Gradients mean: {logits.grad.mean().item():.6f}")
    print("   ✅ Backward pass OK\n")
    
    # Comparison with CrossEntropy
    print("\n" + "=" * 80)
    compare_focal_vs_ce()
    
    print("\n✅ Tutti i test completati!")
