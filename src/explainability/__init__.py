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
from .visualization import (
    plot_attention_heatmap,
    plot_class_comparison,
    plot_action_importance
)

__all__ = [
    'IntegratedGradientsExplainer',
    'ClinicalActionAggregator',
    'plot_attention_heatmap',
    'plot_class_comparison',
    'plot_action_importance'
]

from .integrated_gradients import IntegratedGradientsExplainer
from .action_aggregator import ClinicalActionAggregator
from .visualization import plot_attention_heatmap, plot_class_comparison

__all__ = [
    'IntegratedGradientsExplainer',
    'ClinicalActionAggregator',
    'plot_attention_heatmap',
    'plot_class_comparison',
]
