"""
Visualization Module per Explainability
Crea heatmap e istogrammi comparativi per l'analisi delle attribution
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


# Configurazione stile
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def plot_attention_heatmap(
    top_words_class_0: Dict[str, float],
    top_words_class_1: Dict[str, float],
    top_k: int = 25,
    save_path: str = None,
    normalize_global: bool = True
):
    """
    Crea heatmap che mostra top-K parole per Classe 0 e Classe 1
    
    Normalizzazione: tutto normalizzato a 1, dove 1 = valore massimo globale
    
    Args:
        top_words_class_0: Dict parola -> score per classe 0
        top_words_class_1: Dict parola -> score per classe 1
        top_k: Numero di parole da visualizzare
        save_path: Path dove salvare il plot
        normalize_global: Se True, normalizza al max globale (cross-class)
    """
    # Prendi top-K parole per classe 0 (ordinate per score decrescente)
    words_c0 = list(top_words_class_0.keys())[:top_k]
    scores_c0 = [top_words_class_0[w] for w in words_c0]
    
    # Per classe 1, prendi score delle stesse parole (se presenti) o 0
    scores_c1 = []
    for word in words_c0:
        scores_c1.append(top_words_class_1.get(word, 0.0))
    
    # Normalizzazione globale
    if normalize_global:
        max_score = max(max(scores_c0), max(scores_c1)) if scores_c1 else max(scores_c0)
        if max_score > 0:
            scores_c0_norm = [s / max_score for s in scores_c0]
            scores_c1_norm = [s / max_score for s in scores_c1]
        else:
            scores_c0_norm = scores_c0
            scores_c1_norm = scores_c1
    else:
        scores_c0_norm = scores_c0
        scores_c1_norm = scores_c1
    
    # Crea matrice per heatmap (2 righe: classe 0 e classe 1)
    data = np.array([scores_c0_norm, scores_c1_norm])
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))
    
    sns.heatmap(
        data,
        xticklabels=words_c0,
        yticklabels=['Class 0\n(DISCHARGED)', 'Class 1\n(ADMITTED)'],
        cmap='YlOrRd',
        annot=False,
        fmt='.3f',
        cbar_kws={'label': 'Normalized Attribution Score'},
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax
    )
    
    # Rotazione labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Titolo e labels
    plt.title(
        f'Attribution Heatmap: Top {top_k} Words (ordered by Class 0 importance)\n'
        f'Normalized to max global value = 1.0',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    plt.xlabel('Words/Tokens', fontsize=12, labelpad=10)
    plt.ylabel('Class', fontsize=12, labelpad=10)
    
    plt.tight_layout()
    
    # Salva
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Heatmap salvata: {save_path}")
    
    plt.close()
    
    return fig


def plot_class_comparison(
    top_words_class_0: Dict[str, float],
    top_words_class_1: Dict[str, float],
    top_k: int = 25,
    save_path: str = None
):
    """
    Crea istogramma comparativo per Classe 0 e Classe 1
    
    Normalizzazione SEPARATA per classe:
    - Max valore per classe 0 = max tra i top_k di classe 0
    - Max valore per classe 1 = max tra i top_k di classe 1
    
    Args:
        top_words_class_0: Dict parola -> score per classe 0
        top_words_class_1: Dict parola -> score per classe 1
        top_k: Numero di parole da visualizzare
        save_path: Path dove salvare il plot
    """
    # Top-K parole per classe 0 (ordinate decrescente)
    words_c0 = list(top_words_class_0.keys())[:top_k]
    scores_c0 = [top_words_class_0[w] for w in words_c0]
    
    # Score classe 1 per le stesse parole
    scores_c1 = [top_words_class_1.get(w, 0.0) for w in words_c0]
    
    # Normalizzazione SEPARATA per classe
    max_c0 = max(scores_c0) if scores_c0 else 1.0
    max_c1 = max(scores_c1) if scores_c1 else 1.0
    
    # Normalizza al max della PROPRIA classe mostrata nel plot
    # (non al max globale tra tutte le parole)
    max_display = max(max_c0, max_c1)
    
    scores_c0_norm = [s / max_display for s in scores_c0]
    scores_c1_norm = [s / max_display for s in scores_c1]
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(words_c0))
    width = 0.35
    
    # Bars
    bars1 = ax.bar(
        x - width/2,
        scores_c0_norm,
        width,
        label='Class 0 (DISCHARGED)',
        color='#3498db',
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    bars2 = ax.bar(
        x + width/2,
        scores_c1_norm,
        width,
        label='Class 1 (ADMITTED)',
        color='#e74c3c',
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Aggiungi valori sopra le barre
    def autolabel(bars, scores_original):
        """Attach a text label above each bar displaying its height"""
        for bar, score_orig in zip(bars, scores_original):
            height = bar.get_height()
            if height > 0.05:  # Mostra solo se barra visibile
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{score_orig:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=7,
                    rotation=0
                )
    
    # Valori originali (non normalizzati) per le labels
    autolabel(bars1, scores_c0)
    autolabel(bars2, scores_c1)
    
    # Configurazione assi
    ax.set_xlabel('Words/Tokens', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Normalized Attribution Score', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(
        f'Class Comparison: Top {top_k} Words Attribution (ordered by Class 0)\n'
        f'Normalized to max value in display = 1.0',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels(words_c0, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.set_ylim(0, 1.1)  # Lascia spazio per labels
    
    # Griglia
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Salva
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Istogramma salvato: {save_path}")
    
    plt.close()
    
    return fig


def plot_action_importance(
    action_stats_class_0: Dict,
    action_stats_class_1: Dict,
    top_k: int = 15,
    save_path: str = None,
    metric: str = 'mean_score'
):
    """
    Plot importanza delle azioni cliniche aggregate
    
    Args:
        action_stats_class_0: Stats azioni per classe 0
        action_stats_class_1: Stats azioni per classe 1
        top_k: Numero di azioni da mostrare
        save_path: Path salvataggio
        metric: Metrica da plottare ('mean_score', 'total_score', 'count')
    """
    # Estrai top-K azioni per classe 0
    actions_c0 = sorted(
        action_stats_class_0.items(),
        key=lambda x: x[1][metric],
        reverse=True
    )[:top_k]
    
    action_names = [a[0] for a in actions_c0]
    scores_c0 = [a[1][metric] for a in actions_c0]
    
    # Score corrispondenti per classe 1
    scores_c1 = [
        action_stats_class_1.get(action, {}).get(metric, 0.0)
        for action in action_names
    ]
    
    # Normalizza
    max_score = max(max(scores_c0), max(scores_c1)) if scores_c1 else max(scores_c0)
    if max_score > 0:
        scores_c0_norm = [s / max_score for s in scores_c0]
        scores_c1_norm = [s / max_score for s in scores_c1]
    else:
        scores_c0_norm = scores_c0
        scores_c1_norm = scores_c1
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    y = np.arange(len(action_names))
    height = 0.35
    
    bars1 = ax.barh(
        y - height/2,
        scores_c0_norm,
        height,
        label='Class 0 (DISCHARGED)',
        color='#3498db',
        alpha=0.8
    )
    
    bars2 = ax.barh(
        y + height/2,
        scores_c1_norm,
        height,
        label='Class 1 (ADMITTED)',
        color='#e74c3c',
        alpha=0.8
    )
    
    ax.set_yticks(y)
    ax.set_yticklabels(action_names, fontsize=9)
    ax.set_xlabel(f'Normalized {metric.replace("_", " ").title()}', fontsize=12)
    ax.set_title(
        f'Clinical Action Importance: Top {top_k} Actions\n'
        f'Metric: {metric.replace("_", " ").title()}',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.legend(loc='lower right', fontsize=10)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Action importance plot salvato: {save_path}")
    
    plt.close()
    
    return fig


if __name__ == "__main__":
    print("Visualization module for explainability")
    
    # Test esempio
    test_words_c0 = {f"word{i}": np.random.rand() for i in range(30)}
    test_words_c1 = {f"word{i}": np.random.rand() * 0.7 for i in range(30)}
    
    print("\n🎨 Test visualization functions...")
    plot_attention_heatmap(test_words_c0, test_words_c1, top_k=10)
    print("✅ Heatmap test OK")
    
    plot_class_comparison(test_words_c0, test_words_c1, top_k=10)
    print("✅ Comparison histogram test OK")
