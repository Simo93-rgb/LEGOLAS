"""
Explainability module per LEGOLAS
Estrae attribution scores e visualizza l'importanza delle features

Usage:
    # Via script bash (raccomandato)
    bash scripts/run_explainability.sh
    
    # Via Python diretto
    python -m src.explainability.extract_explainability --model clinical-bert --format narrativo
"""

from .integrated_gradients import IntegratedGradientsExplainer
from .action_aggregator import ClinicalActionAggregator
from .ig_completeness import compute_ig_with_completeness_check, find_optimal_n_steps
from .visualization import (
    plot_attention_heatmap,
    plot_class_comparison,
    plot_action_importance,
    plot_clinical_actions_heatmap,
    plot_clinical_actions_comparison
)

__all__ = [
    'IntegratedGradientsExplainer',
    'ClinicalActionAggregator',
    'compute_ig_with_completeness_check',
    'find_optimal_n_steps',
    'plot_attention_heatmap',
    'plot_class_comparison',
    'plot_action_importance',
    'plot_clinical_actions_heatmap',
    'plot_clinical_actions_comparison',
]
