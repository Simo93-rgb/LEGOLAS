"""
Ensemble Model for K-Fold Cross Validation
Gestisce predizione ensemble da modelli K-Fold

Features:
- Auto-discovery di modelli fold da directory
- Caricamento automatico metriche per identificare best fold
- Predizione ensemble: media delle probabilità softmax
- XAI ensemble: media degli attribution scores per ogni modello
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
import json
import numpy as np
from tqdm import tqdm


class EnsembleModel:
    """
    Ensemble di modelli K-Fold per predizione e explainability
    
    Architecture:
    - Auto-discovery: Scansiona output/models/ per trovare fold models
    - Best fold selection: Usa balanced_accuracy da metrics JSON
    - Prediction: Media probabilità softmax da tutti i fold
    - XAI: Calcola IG per ogni modello, poi media attribution scores
    
    Args:
        story_format: Formato storie ('narrativo', 'tabellare', etc.)
        model_name: Nome modello ('bert-base-uncased', etc.)
        model_factory: Callable che ritorna nuovo modello instance
        models_dir: Directory dove cercare modelli (default: output/models)
        metrics_dir: Directory dove cercare metriche (default: output/metrics)
        device: Device per inference ('cuda' o 'cpu')
        
    Example:
        >>> def create_model():
        ...     return MyModel(num_classes=5)
        >>> 
        >>> ensemble = EnsembleModel(
        ...     story_format='narrativo',
        ...     model_name='bert-base-uncased',
        ...     model_factory=create_model
        ... )
        >>> probs = ensemble.predict(input_ids, attention_mask)
        >>> print(f"Ensemble predictions shape: {probs.shape}")
    """
    
    def __init__(
        self,
        story_format: str,
        model_name: str,
        model_factory: Callable[[], nn.Module],
        models_dir: Path = None,
        metrics_dir: Path = None,
        device: str = 'cuda'
    ):
        self.story_format = story_format
        self.model_name = model_name
        self.model_factory = model_factory
        self.device = device
        
        # Default directories
        if models_dir is None:
            models_dir = Path('output/models')
        if metrics_dir is None:
            metrics_dir = Path('output/metrics')
            
        self.models_dir = Path(models_dir)
        self.metrics_dir = Path(metrics_dir)
        
        # Find and load fold models
        self.model_paths = EnsembleModel.find_kfold_models(
            story_format=self.story_format,
            model_name=self.model_name,
            models_dir=self.models_dir
        )
        
        # Try to load metrics for all folds (optional for ensemble)
        try:
            self.fold_metrics = EnsembleModel.load_fold_metrics(
                story_format=self.story_format,
                model_name=self.model_name,
                n_folds=len(self.model_paths),
                metrics_dir=self.metrics_dir
            )
        except FileNotFoundError as e:
            print(f"⚠️  Metrics JSON not found: {e}")
            print(f"   Creating placeholder metrics (ensemble uses all folds anyway)")
            # Create placeholder metrics with unknown accuracy
            self.fold_metrics = []
            for i in range(len(self.model_paths)):
                self.fold_metrics.append({
                    'metric': 'balanced_accuracy',
                    'mode': 'max',
                    'best_value': 0.0,  # Unknown
                    'best_epoch': 0,
                    'fold': i
                })
        
        # Load all models
        self.models: List[nn.Module] = []
        self._load_models()
        
        # Cache per predictions (evita ricalcolo)
        self.probs: Optional[torch.Tensor] = None
        
    @staticmethod
    def find_kfold_models(
        story_format: str,
        model_name: str,
        models_dir: Path = None
    ) -> List[Path]:
        """
        Auto-discovery di modelli fold da directory
        
        Pattern: best_model_{format}_{model}_fold{k}.pth
        
        Args:
            story_format: Formato storie
            model_name: Nome modello
            models_dir: Directory dove cercare (default: output/models)
            
        Returns:
            List di Path ordinati per fold index
            
        Raises:
            FileNotFoundError: Se nessun fold model trovato
            ValueError: Se sequenza fold non è continua (es: fold0,1,3 senza 2)
            
        Example:
            >>> paths = EnsembleModel.find_kfold_models('narrativo', 'bert-base-uncased')
            >>> print(f"Found {len(paths)} folds")
        """
        if models_dir is None:
            models_dir = Path('output/models')
        else:
            models_dir = Path(models_dir)
            
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        # Pattern matching
        pattern = f"best_model_{story_format}_{model_name}_fold*.pth"
        fold_files = list(models_dir.glob(pattern))
        
        if not fold_files:
            raise FileNotFoundError(
                f"No fold models found in {models_dir}\n"
                f"Pattern: {pattern}\n"
                f"Make sure you have trained with K-Fold CV."
            )
        
        # Estrai indici fold e ordina
        fold_indices = []
        fold_map = {}
        for path in fold_files:
            # Estrai numero fold dal filename
            # best_model_narrativo_bert-base-uncased_fold0.pth -> 0
            try:
                fold_idx = int(path.stem.split('_fold')[-1])
                fold_indices.append(fold_idx)
                fold_map[fold_idx] = path
            except ValueError:
                continue
                
        if not fold_indices:
            raise ValueError(f"Could not parse fold indices from: {[f.name for f in fold_files]}")
        
        fold_indices.sort()
        
        # Verifica sequenza continua (0,1,2,... senza gap)
        expected = list(range(len(fold_indices)))
        if fold_indices != expected:
            raise ValueError(
                f"Fold sequence has gaps!\n"
                f"Found: {fold_indices}\n"
                f"Expected: {expected}\n"
                f"All folds from 0 to {len(fold_indices)-1} must exist."
            )
        
        # Ritorna paths ordinati
        return [fold_map[i] for i in fold_indices]
    
    @staticmethod
    def load_fold_metrics(
        story_format: str,
        model_name: str,
        n_folds: int,
        metrics_dir: Path = None
    ) -> List[Dict]:
        """
        Carica metriche JSON per ogni fold
        
        Pattern: fold_{k}_{format}_{model}_metrics.json
        
        Args:
            story_format: Formato storie
            model_name: Nome modello
            n_folds: Numero di fold da caricare
            metrics_dir: Directory metriche (default: output/metrics)
            
        Returns:
            List di dict con metriche per ogni fold
            
        Raises:
            FileNotFoundError: Se un metrics file non trovato
            
        Example:
            >>> metrics = EnsembleModel.load_fold_metrics('narrativo', 'bert-base-uncased', 5)
            >>> print(f"Fold 0 balanced_accuracy: {metrics[0]['best_value']:.4f}")
        """
        if metrics_dir is None:
            metrics_dir = Path('output/metrics')
        else:
            metrics_dir = Path(metrics_dir)
            
        fold_metrics = []
        for fold_idx in range(n_folds):
            metrics_path = metrics_dir / f"fold_{fold_idx}_{story_format}_{model_name}_metrics.json"
            
            if not metrics_path.exists():
                raise FileNotFoundError(
                    f"Metrics file not found: {metrics_path}\n"
                    f"Expected for fold {fold_idx}"
                )
            
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            fold_metrics.append(metrics)
            
        return fold_metrics
    
    @staticmethod
    def get_best_fold(fold_metrics: List[Dict]) -> int:
        """
        Identifica best fold da metriche
        
        Usa balanced_accuracy come criterio (mode='max')
        
        Args:
            fold_metrics: List di dict con metriche per ogni fold
            
        Returns:
            Index del best fold (0-based)
            
        Example:
            >>> metrics = EnsembleModel.load_fold_metrics('narrativo', 'bert-base-uncased', 5)
            >>> best = EnsembleModel.get_best_fold(metrics)
            >>> print(f"Best fold: {best} with accuracy: {metrics[best]['best_value']:.4f}")
        """
        best_fold_idx = 0
        best_value = -float('inf')
        
        for fold_idx, metrics in enumerate(fold_metrics):
            # Metrics JSON contiene: metric, mode, best_value, best_epoch, fold
            value = metrics['best_value']
            if value > best_value:
                best_value = value
                best_fold_idx = fold_idx
                
        return best_fold_idx
    
    def _load_models(self):
        """
        Carica tutti i fold models in memoria
        
        Per ogni fold:
        1. Crea nuovo modello da factory
        2. Carica state_dict
        3. Sposta su device
        4. Imposta eval mode
        """
        self.models = []
        
        print(f"🔄 Loading {len(self.model_paths)} fold models...")
        for fold_idx, model_path in enumerate(self.model_paths):
            # Crea nuovo modello
            model = self.model_factory()
            
            # Carica checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Estrai state_dict (gestisce sia checkpoint dict che state_dict diretto)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            model.load_state_dict(state_dict)
            
            # Setup per inference
            model.to(self.device)
            model.eval()
            
            self.models.append(model)
            print(f"   ✅ Fold {fold_idx}: {model_path.name}")
            
        print(f"✅ All {len(self.models)} models loaded successfully\n")
    
    def predict_single(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Predizione con singolo modello fold
        
        Args:
            model: Modello fold
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Probabilità softmax (batch_size, num_classes)
            
        Note:
            Opera in modalità no_grad per efficienza
            Output già su device corretto
        """
        with torch.no_grad():
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Estrai logits (gestisce vari output format)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Softmax per ottenere probabilità
            probs = torch.softmax(logits, dim=-1)
            
        return probs
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_individual: bool = False
    ) -> torch.Tensor:
        """
        Predizione ensemble: media probabilità softmax da tutti i fold
        
        Strategy:
        1. Per ogni fold: compute softmax probabilities
        2. Stack probabilities: (k_folds, batch_size, num_classes)
        3. Average: mean(probs, dim=0) -> (batch_size, num_classes)
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            return_individual: Se True, ritorna anche probs individuali per ogni fold
            
        Returns:
            Se return_individual=False:
                Probabilità ensemble (batch_size, num_classes)
            Se return_individual=True:
                Tuple (ensemble_probs, individual_probs)
                - ensemble_probs: (batch_size, num_classes)
                - individual_probs: (k_folds, batch_size, num_classes)
                
        Example:
            >>> ensemble = EnsembleModel(...)
            >>> probs = ensemble.predict(input_ids, attention_mask)
            >>> predictions = probs.argmax(dim=-1)
            >>> print(f"Predictions: {predictions}")
        """
        # Sposta input su device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Collect predictions da tutti i fold
        all_probs = []
        
        for fold_idx, model in enumerate(self.models):
            # Predizione singolo fold
            probs = self.predict_single(model, input_ids, attention_mask)
            all_probs.append(probs)
        
        # Stack: (k_folds, batch_size, num_classes)
        all_probs = torch.stack(all_probs, dim=0)
        
        # Average ensemble: (batch_size, num_classes)
        ensemble_probs = torch.mean(all_probs, dim=0)
        
        # Cache risultato (utile per compute_ensemble_attributions)
        self.probs = ensemble_probs
        
        if return_individual:
            return ensemble_probs, all_probs
        else:
            return ensemble_probs
    
    def compute_ensemble_attributions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: int,
        n_steps: int = 50,
        internal_batch_size: int = 32
    ) -> torch.Tensor:
        """
        Compute Ensemble Integrated Gradients: calcola IG per ogni fold, poi media
        
        Strategy (from docs/XAI.md):
        1. Per ogni fold model: compute Integrated Gradients
        2. Average attributions across folds: mean(attributions, axis=0)
        
        Args:
            input_ids: Input token IDs (1, seq_len) - SINGLE SAMPLE
            attention_mask: Attention mask (1, seq_len)
            target_class: Target class index for attribution
            n_steps: Number of IG interpolation steps (default: 50)
            internal_batch_size: Batch size for IG steps (default: 32)
            
        Returns:
            Averaged attributions: (seq_len,) - IG scores per token
            
        Example:
            >>> ensemble = EnsembleModel(...)
            >>> input_ids = tokenizer(text, return_tensors='pt')['input_ids']
            >>> attention_mask = tokenizer(text, return_tensors='pt')['attention_mask']
            >>> attributions = ensemble.compute_ensemble_attributions(
            ...     input_ids, attention_mask, target_class=1, n_steps=50
            ... )
            >>> print(f"Attribution shape: {attributions.shape}")  # (seq_len,)
        """
        from captum.attr import IntegratedGradients
        
        # Ensure single sample
        assert input_ids.size(0) == 1, "compute_ensemble_attributions supports SINGLE sample only"
        
        # Sposta input su device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Collect attributions da tutti i fold
        all_attributions = []
        
        for fold_idx, model in enumerate(self.models):
            # Set model to eval mode
            model.eval()
            
            # Define forward function for IG
            def forward_func(input_embeds, attention_mask_param=attention_mask):
                # Forward pass con embeddings
                outputs = model.longformer(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask_param
                )
                # Get logits usando pooler_output
                logits = model.output_layer(outputs.pooler_output)
                return logits
            
            # Get input embeddings
            embeddings = model.longformer.embeddings.word_embeddings(input_ids)
            
            # Initialize IntegratedGradients
            ig = IntegratedGradients(forward_func)
            
            # Compute attributions per questo fold
            attributions = ig.attribute(
                embeddings,
                target=target_class,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size
            )
            
            # Sum across embedding dimension: (1, seq_len, hidden_dim) -> (1, seq_len)
            attributions_summed = attributions.sum(dim=-1)
            
            # Append: (seq_len,)
            all_attributions.append(attributions_summed.squeeze(0))
        
        # Stack: (k_folds, seq_len)
        all_attributions = torch.stack(all_attributions, dim=0)
        
        # Average ensemble attributions: (seq_len,)
        ensemble_attributions = torch.mean(all_attributions, dim=0)
        
        return ensemble_attributions
