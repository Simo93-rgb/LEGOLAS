"""
Training Configuration for LEGOLAS
Gestisce tutti i parametri di training, K-Fold CV, Early Stopping, e Focal Loss
"""

import torch
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """
    Configurazione completa per il training dei modelli LEGOLAS
    
    Supporta:
    - Training semplice o K-Fold Cross Validation
    - Focal Loss o CrossEntropy
    - Early Stopping avanzato con loss ratio
    - Best model tracking con balanced accuracy
    """
    
    # =============================================================================
    # 🔧 DEVICE E SETUP BASE
    # =============================================================================
    
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42
    
    # =============================================================================
    # 🤖 MODELLO
    # =============================================================================
    
    model_name: str = "clinical-bert"  # Da model_configs.yaml
    story_format: str = "narrativo"  # narrativo, bullet, clinical
    num_classes: int = 2  # CLS_0 (0), CLS_1 (1)
    max_length: int = 512  # Max sequence length per tokenizer
    
    # =============================================================================
    # 🎯 TRAINING PARAMETERS
    # =============================================================================
    
    # Basic training
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 20
    gradient_accumulation_steps: int = 1
    
    # Optimizer
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # =============================================================================
    # 📊 DATASET SPLITS
    # =============================================================================
    
    # Train/Val/Test splits
    test_size: float = 0.2  # 20% test set (stratificato)
    val_size: float = 0.1  # 10% validation (stratificato, dal train rimanente)
    stratify: bool = True  # Mantiene proporzioni classi
    
    # =============================================================================
    # 🔄 K-FOLD CROSS VALIDATION
    # =============================================================================
    
    use_kfold: bool = False  # False = training semplice, True = K-Fold CV
    n_folds: int = 10  # Numero di fold
    kfold_shuffle: bool = True
    kfold_stratified: bool = True  # Mantiene proporzioni classi nei fold
    
    # =============================================================================
    # 🎲 LOSS FUNCTION
    # =============================================================================
    
    loss_function: Literal["focal", "ce"] = "focal"  # focal = Focal Loss, ce = CrossEntropy
    
    # Focal Loss parameters (solo se loss_function="focal")
    focal_alpha: list[float] = field(default_factory=lambda: [0.25, 0.75])  # [classe_0, classe_1]
    focal_gamma: float = 2.0  # Focusing parameter
    
    # Class weights (solo se loss_function="ce")
    use_class_weights: bool = True  # Auto-calcola pesi da distribuzione classi
    
    # =============================================================================
    # ⏱️ EARLY STOPPING
    # =============================================================================
    
    use_early_stopping: bool = True
    early_stopping_patience: int = 5  # Epoche da aspettare prima di fermare
    early_stopping_min_delta: float = 0.001  # Miglioramento minimo richiesto
    
    # Advanced: Loss ratio monitoring (train_loss / val_loss)
    use_loss_ratio_monitoring: bool = True
    loss_ratio_threshold: float = 1.15  # Se train/val > threshold → possibile overfitting
    loss_ratio_patience: int = 3  # Epoche consecutive sopra threshold prima di fermare
    
    # Focal Loss ha ratio più alti, quindi soglia diversa
    focal_loss_ratio_threshold: float = 20.0
    
    # =============================================================================
    # 💾 MODEL CHECKPOINTING
    # =============================================================================
    
    save_best_only: bool = True  # Salva solo best model (non ogni epoca)
    checkpoint_metric: str = "balanced_accuracy"  # Metrica per best model
    checkpoint_mode: str = "max"  # max = metrica più alta è migliore
    
    # =============================================================================
    # 📁 PATHS
    # =============================================================================
    
    @property
    def project_root(self) -> Path:
        """Root directory del progetto LEGOLAS"""
        # Risale da src/training/ alla root
        return Path(__file__).resolve().parent.parent.parent
    
    @property
    def output_dir(self) -> Path:
        """Directory output principale"""
        return self.project_root / "output"
    
    @property
    def models_dir(self) -> Path:
        """Directory per modelli salvati"""
        path = self.output_dir / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def logs_dir(self) -> Path:
        """Directory per log di training"""
        path = self.output_dir / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def reports_dir(self) -> Path:
        """Directory per report e metriche"""
        path = self.output_dir / "evaluation"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    # =============================================================================
    # 🛠️ HELPER METHODS
    # =============================================================================
    
    def get_model_filename(self, fold: Optional[int] = None, suffix: str = "") -> str:
        """
        Genera nome file per modello salvato
        
        Args:
            fold: Numero fold (None per training semplice)
            suffix: Suffisso aggiuntivo (es: 'epoch_10')
            
        Returns:
            Nome file (es: 'best_model_narrativo_clinical-bert_fold3.pth')
        """
        fold_str = f"_fold{fold}" if fold is not None else ""
        suffix_str = f"_{suffix}" if suffix else ""
        return f"best_model_{self.story_format}_{self.model_name}{fold_str}{suffix_str}.pth"
    
    def get_model_path(self, fold: Optional[int] = None, suffix: str = "") -> Path:
        """Restituisce path completo per salvare il modello"""
        return self.models_dir / self.get_model_filename(fold, suffix)
    
    def get_loss_ratio_threshold(self) -> float:
        """Restituisce threshold corretto in base alla loss function"""
        if self.loss_function == "focal":
            return self.focal_loss_ratio_threshold
        return self.loss_ratio_threshold
    
    def get_device(self) -> torch.device:
        """Restituisce torch device"""
        return torch.device(self.device)
    
    # =============================================================================
    # 🖨️ PRINT & VALIDATION
    # =============================================================================
    
    def print_config(self):
        """Stampa configurazione corrente in modo leggibile"""
        print("=" * 80)
        print("  🔧 LEGOLAS TRAINING CONFIGURATION")
        print("=" * 80)
        
        print(f"\n🤖 MODEL:")
        print(f"   Name: {self.model_name}")
        print(f"   Story Format: {self.story_format}")
        print(f"   Device: {self.device}")
        print(f"   Max Length: {self.max_length}")
        print(f"   Num Classes: {self.num_classes}")
        
        print(f"\n🎯 TRAINING:")
        print(f"   Batch Size: {self.batch_size}")
        print(f"   Learning Rate: {self.learning_rate}")
        print(f"   Num Epochs: {self.num_epochs}")
        print(f"   Gradient Accumulation: {self.gradient_accumulation_steps}")
        
        print(f"\n📊 DATASET:")
        print(f"   Test Size: {self.test_size * 100:.0f}%")
        print(f"   Val Size: {self.val_size * 100:.0f}%")
        print(f"   Stratified: {self.stratify}")
        
        if self.use_kfold:
            print(f"\n🔄 K-FOLD CROSS VALIDATION:")
            print(f"   Enabled: {self.use_kfold}")
            print(f"   N Folds: {self.n_folds}")
            print(f"   Stratified: {self.kfold_stratified}")
            print(f"   Shuffle: {self.kfold_shuffle}")
        else:
            print(f"\n🔄 K-FOLD: Disabled (simple train/val split)")
        
        print(f"\n🎲 LOSS FUNCTION:")
        print(f"   Type: {self.loss_function.upper()}")
        if self.loss_function == "focal":
            print(f"   Focal Alpha: {self.focal_alpha}")
            print(f"   Focal Gamma: {self.focal_gamma}")
        else:
            print(f"   Use Class Weights: {self.use_class_weights}")
        
        if self.use_early_stopping:
            print(f"\n⏱️  EARLY STOPPING:")
            print(f"   Enabled: {self.use_early_stopping}")
            print(f"   Patience: {self.early_stopping_patience}")
            print(f"   Min Delta: {self.early_stopping_min_delta}")
            if self.use_loss_ratio_monitoring:
                print(f"   Loss Ratio Monitoring: {self.use_loss_ratio_monitoring}")
                print(f"   Loss Ratio Threshold: {self.get_loss_ratio_threshold()}")
                print(f"   Loss Ratio Patience: {self.loss_ratio_patience}")
        
        print(f"\n💾 CHECKPOINTING:")
        print(f"   Save Best Only: {self.save_best_only}")
        print(f"   Metric: {self.checkpoint_metric}")
        print(f"   Mode: {self.checkpoint_mode}")
        
        print(f"\n📁 PATHS:")
        print(f"   Project Root: {self.project_root}")
        print(f"   Models Dir: {self.models_dir}")
        print(f"   Logs Dir: {self.logs_dir}")
        print(f"   Reports Dir: {self.reports_dir}")
        
        print("=" * 80)
    
    def validate(self) -> bool:
        """
        Valida la configurazione
        
        Returns:
            True se configurazione valida, False altrimenti
        """
        issues = []
        
        # Verifica device
        if self.device == "cuda" and not torch.cuda.is_available():
            issues.append("⚠️  CUDA richiesto ma non disponibile")
        
        # Verifica parametri
        if self.batch_size <= 0:
            issues.append("⚠️  Batch size deve essere > 0")
        
        if self.learning_rate <= 0:
            issues.append("⚠️  Learning rate deve essere > 0")
        
        if self.num_epochs <= 0:
            issues.append("⚠️  Num epochs deve essere > 0")
        
        # Verifica splits
        if not 0 < self.test_size < 1:
            issues.append("⚠️  Test size deve essere tra 0 e 1")
        
        if not 0 < self.val_size < 1:
            issues.append("⚠️  Val size deve essere tra 0 e 1")
        
        if self.test_size + self.val_size >= 1:
            issues.append("⚠️  Test + Val size deve essere < 1")
        
        # Verifica K-Fold
        if self.use_kfold and self.n_folds < 2:
            issues.append("⚠️  N folds deve essere >= 2")
        
        # Verifica Loss function
        if self.loss_function not in ["focal", "ce"]:
            issues.append(f"⚠️  Loss function '{self.loss_function}' non supportata")
        
        # Verifica Focal Loss params
        if self.loss_function == "focal":
            if len(self.focal_alpha) != self.num_classes:
                issues.append(f"⚠️  Focal alpha deve avere {self.num_classes} elementi")
            if self.focal_gamma < 0:
                issues.append("⚠️  Focal gamma deve essere >= 0")
        
        # Verifica Early Stopping
        if self.use_early_stopping:
            if self.early_stopping_patience <= 0:
                issues.append("⚠️  Early stopping patience deve essere > 0")
        
        # Verifica directory
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            self.reports_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"⚠️  Errore creazione directory: {e}")
        
        # Stampa risultati
        if issues:
            print("\n❌ PROBLEMI DI CONFIGURAZIONE:")
            for issue in issues:
                print(f"   {issue}")
            return False
        else:
            print("\n✅ Configurazione validata con successo!")
            return True


# =============================================================================
# 🏭 FACTORY FUNCTIONS
# =============================================================================

def create_default_config(
    model_name: str = "clinical-bert",
    story_format: str = "narrativo",
    use_kfold: bool = False,
    loss_function: str = "focal"
) -> TrainingConfig:
    """
    Crea configurazione di default per training
    
    Args:
        model_name: Nome modello da model_configs.yaml
        story_format: Formato storie (narrativo, bullet, clinical)
        use_kfold: Abilita K-Fold CV
        loss_function: Tipo di loss ('focal' o 'ce')
        
    Returns:
        TrainingConfig con parametri di default
    """
    return TrainingConfig(
        model_name=model_name,
        story_format=story_format,
        use_kfold=use_kfold,
        loss_function=loss_function
    )


def create_kfold_config(
    model_name: str = "clinical-bert",
    story_format: str = "narrativo",
    n_folds: int = 10
) -> TrainingConfig:
    """
    Crea configurazione per K-Fold Cross Validation
    
    Args:
        model_name: Nome modello
        story_format: Formato storie
        n_folds: Numero di fold
        
    Returns:
        TrainingConfig con K-Fold abilitato
    """
    return TrainingConfig(
        model_name=model_name,
        story_format=story_format,
        use_kfold=True,
        n_folds=n_folds,
        loss_function="focal"
    )


# =============================================================================
# 🧪 TESTING
# =============================================================================

if __name__ == "__main__":
    print("🧪 Testing TrainingConfig...\n")
    
    # Test default config
    config = create_default_config()
    config.print_config()
    config.validate()
    
    print("\n" + "="*80 + "\n")
    
    # Test K-Fold config
    kfold_config = create_kfold_config(n_folds=5)
    kfold_config.print_config()
    kfold_config.validate()
    
    print("\n✅ Test completato!")
