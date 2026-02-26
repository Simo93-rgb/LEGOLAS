"""
Visualization Module per Explainability
Crea heatmap e istogrammi comparativi per l'analisi delle attribution
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
from pathlib import Path


# Configurazione stile
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def _load_italian_translations() -> Dict[str, str]:
    """
    Carica traduzioni dal file JSON e crea mapping inglese -> italiano
    
    Returns:
        Dict[str, str]: Mapping da azione inglese a azione italiana
    """
    translation_file = Path(__file__).parent.parent.parent / "data" / "translation_cache.json"
    
    if not translation_file.exists():
        print(f"⚠️  Translation file not found: {translation_file}")
        return {}
    
    try:
        with open(translation_file, 'r', encoding='utf-8') as f:
            translations = json.load(f)
        
        # Inverti mapping: inglese -> italiano
        english_to_italian = {v: k for k, v in translations.items()}
        
        return english_to_italian
        
    except Exception as e:
        print(f"⚠️  Error loading translations: {e}")
        return {}


def _translate_to_italian(action_english: str, translations: Dict[str, str]) -> str:
    """
    Traduce azione da inglese a italiano usando il dizionario
    
    Args:
        action_english: Nome azione in inglese
        translations: Dizionario inglese -> italiano
        
    Returns:
        Nome azione in italiano (o inglese se traduzione non trovata)
    """
    return translations.get(action_english, action_english)


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
    # FOCUS: Prendi top-K parole per CLASSE 1 (ordinate per score decrescente)
    words_c1 = list(top_words_class_1.keys())[:top_k]
    scores_c1 = [top_words_class_1[w] for w in words_c1]

    # Per classe 0, prendi score delle stesse parole (se presenti) o 0
    scores_c0 = [top_words_class_0.get(word, 0.0) for word in words_c1]

    # Normalizzazione globale (safe: gestisce liste vuote)
    if normalize_global:
        candidates: List[float] = []
        if scores_c1:
            candidates.append(max(scores_c1))
        if scores_c0:
            candidates.append(max(scores_c0))
        max_score = max(candidates) if candidates else 0.0
        if max_score > 0:
            scores_c0_norm = [s / max_score for s in scores_c0]
            scores_c1_norm = [s / max_score for s in scores_c1]
        else:
            scores_c0_norm = [0.0 for _ in scores_c0]
            scores_c1_norm = [0.0 for _ in scores_c1]
    else:
        scores_c0_norm = scores_c0
        scores_c1_norm = scores_c1

    # Crea matrice per heatmap — Classe 1 in ALTO (focus principale)
    data = np.array([scores_c1_norm, scores_c0_norm])

    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))

    sns.heatmap(
        data,
        xticklabels=words_c1,
        yticklabels=['Class 1 (target)', 'Class 0 (reference)'],
        cmap='YlOrRd',
        annot=True,
        fmt='.3f',
        cbar_kws={'label': 'Normalized Attribution Score'},
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        annot_kws={'fontsize': 8}
    )

    # Rotazione labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0, ha='left')

    # Titolo e labels (FOCUS su Classe 1)
    plt.title(
        f'Attribution Heatmap: Top {top_k} Words (ordered by Class 1 importance)\n'
        f'Normalized to max global value = 1.0',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    plt.xlabel('Words/Tokens', fontsize=12, labelpad=10)
    plt.ylabel('Class', fontsize=12, labelpad=10)
    fig.subplots_adjust(left=0.22)
    ax.set_yticklabels(['Class 1 (target)', 'Class 0 (reference)'], rotation=0, ha='right', fontsize=10)

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
    Crea istogramma comparativo per Classe 0 e Classe 1 (ORIZZONTALE)
    
    Normalizzazione SEPARATA per classe:
    - Max valore per classe 0 = max tra i top_k di classe 0
    - Max valore per classe 1 = max tra i top_k di classe 1
    
    Args:
        top_words_class_0: Dict parola -> score per classe 0
        top_words_class_1: Dict parola -> score per classe 1
        top_k: Numero di parole da visualizzare
        save_path: Path dove salvare il plot
    """
    # FOCUS: Top-K parole per CLASSE 1 (ordinate decrescente)
    words_c1 = list(top_words_class_1.keys())[:top_k]
    scores_c1 = [top_words_class_1[w] for w in words_c1]

    # Score classe 0 per le stesse parole
    scores_c0 = [top_words_class_0.get(w, 0.0) for w in words_c1]

    # Normalizzazione per-class: ogni classe al proprio max
    max_c0 = max(scores_c0) if scores_c0 else 0.0
    max_c1 = max(scores_c1) if scores_c1 else 0.0

    scores_c0_norm = [s / max_c0 for s in scores_c0] if max_c0 > 0 else [0.0] * len(scores_c0)
    scores_c1_norm = [s / max_c1 for s in scores_c1] if max_c1 > 0 else [0.0] * len(scores_c1)

    # Setup plot ORIZZONTALE
    fig, ax = plt.subplots(figsize=(14, 10))

    x = np.arange(len(words_c1))
    width = 0.35

    # Classe 1 in evidenza (barra superiore)
    bars1 = ax.barh(
        x - width/2,
        scores_c1_norm,
        width,
        label='Class 1 (target)',
        color='steelblue',
        alpha=0.8
    )

    bars2 = ax.barh(
        x + width/2,
        scores_c0_norm,
        width,
        label='Class 0',
        color='coral',
        alpha=0.8
    )

    # Aggiungi valori sulle barre
    for bars, scores_original in [(bars1, scores_c1), (bars2, scores_c0)]:
        for bar, score_orig in zip(bars, scores_original):
            width_val = bar.get_width()
            if width_val > 0.05:  # Mostra solo se barra visibile
                ax.text(
                    width_val + 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'{score_orig:.3f}',
                    ha='left',
                    va='center',
                    fontsize=7
                )

    # Configurazione assi
    ax.set_yticks(x)
    ax.set_yticklabels(words_c1, fontsize=9)
    ax.set_xlabel('Normalized Attribution Score (per-class)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(
        f'Class Comparison: Top {top_k} Words Attribution (ordered by Class 1)\n'
        f'Each class normalized to its own max = 1.0',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()  # Top word in alto
    
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
        label='Class 0 ',
        color='#3498db',
        alpha=0.8
    )
    
    bars2 = ax.barh(
        y + height/2,
        scores_c1_norm,
        height,
        label='Class 1 ',
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


def plot_clinical_actions_heatmap(
    actions_class_0: Dict[str, Dict],
    actions_class_1: Dict[str, Dict],
    top_k: int = 25,
    save_path: str = None,
    normalize_global: bool = True
):
    """
    Crea heatmap per AZIONI CLINICHE invece di parole singole
    
    Args:
        actions_class_0: Dict azione -> {'mean_score', 'count', ...} per classe 0
        actions_class_1: Dict azione -> {'mean_score', 'count', ...} per classe 1
        top_k: Numero di azioni da visualizzare
        save_path: Path dove salvare il plot
        normalize_global: Se True, normalizza al max globale
    """
    # FOCUS: Ordina azioni per mean_score decrescente della CLASSE 1
    sorted_actions_c1 = sorted(
        actions_class_1.items(),
        key=lambda x: x[1]['mean_score'],
        reverse=True
    )[:top_k]

    # Carica traduzioni inglese -> italiano
    translations = _load_italian_translations()

    # Estrai nomi azioni e scores per Classe 1
    action_names_english = [action for action, _ in sorted_actions_c1]
    scores_c1 = [stats['mean_score'] for _, stats in sorted_actions_c1]

    # Traduci azioni in italiano
    action_names_italian = [_translate_to_italian(action, translations)
                            for action in action_names_english]

    # Per classe 0, prendi score delle stesse azioni (se presenti) o 0
    scores_c0 = [
        actions_class_0[action]['mean_score'] if action in actions_class_0 else 0.0
        for action in action_names_english
    ]

    # Normalizzazione globale (safe: evita max() su liste vuote)
    if normalize_global:
        candidates: List[float] = []
        if scores_c1:
            candidates.append(max(scores_c1))
        if scores_c0:
            candidates.append(max(scores_c0))
        max_score = max(candidates) if candidates else 0.0
        if max_score > 0:
            scores_c0 = [s / max_score for s in scores_c0]
            scores_c1 = [s / max_score for s in scores_c1]

    # Crea matrice 2xN — Classe 1 in ALTO (focus principale)
    data = np.array([scores_c1, scores_c0])

    # Trunca nomi azioni troppo lunghi (usa italiano)
    action_labels = [action[:50] + '...' if len(action) > 50 else action
                     for action in action_names_italian]

    # Plot
    fig, ax = plt.subplots(figsize=(16, 4))

    sns.heatmap(
        data,
        xticklabels=action_labels,
        yticklabels=['Class 1 (target)', 'Class 0 (reference)'],
        cmap='YlOrRd',
        annot=True,
        fmt='.3f',
        cbar=True,
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={'label': 'Attribution Score (normalized)'},
        annot_kws={'fontsize': 8}
    )

    plt.title(f'Top-{top_k} Clinical Actions Attribution Heatmap\n(Ordered by Class 1 importance)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Clinical Actions', fontsize=12)
    plt.ylabel('Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    fig.subplots_adjust(left=0.22)
    ax.set_yticklabels(['Class 1 (target)', 'Class 0 (reference)'], rotation=0, ha='right', fontsize=10)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Clinical actions heatmap salvata: {save_path}")
    
    plt.close()
    
    return fig


def plot_clinical_actions_comparison(
    actions_class_0: Dict[str, Dict],
    actions_class_1: Dict[str, Dict],
    top_k: int = 25,
    save_path: str = None
):
    """
    Crea istogramma comparativo per AZIONI CLINICHE invece di parole singole
    
    Args:
        actions_class_0: Dict azione -> {'mean_score', 'count', ...} per classe 0
        actions_class_1: Dict azione -> {'mean_score', 'count', ...} per classe 1
        top_k: Numero di azioni da visualizzare
        save_path: Path dove salvare il plot
    """
    # FOCUS: Ordina azioni per mean_score decrescente della CLASSE 1
    sorted_actions_c1 = sorted(
        actions_class_1.items(),
        key=lambda x: x[1]['mean_score'],
        reverse=True
    )[:top_k]

    # Carica traduzioni inglese -> italiano
    translations = _load_italian_translations()

    # Estrai nomi azioni e scores per Classe 1
    action_names_english = [action for action, _ in sorted_actions_c1]
    scores_c1 = [stats['mean_score'] for _, stats in sorted_actions_c1]

    # Traduci azioni in italiano
    action_names_italian = [_translate_to_italian(action, translations)
                            for action in action_names_english]

    # Per classe 0, prendi score delle stesse azioni
    scores_c0 = [
        actions_class_0[action]['mean_score'] if action in actions_class_0 else 0.0
        for action in action_names_english
    ]

    # Normalizzazione per-class: ogni classe al proprio max
    max_c0 = max(scores_c0) if scores_c0 else 0.0
    max_c1 = max(scores_c1) if scores_c1 else 0.0

    scores_c0_norm = [s / max_c0 for s in scores_c0] if max_c0 > 0 else [0.0] * len(scores_c0)
    scores_c1_norm = [s / max_c1 for s in scores_c1] if max_c1 > 0 else [0.0] * len(scores_c1)

    # Trunca nomi azioni (usa italiano)
    action_labels = [action[:50] + '...' if len(action) > 50 else action
                     for action in action_names_italian]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))

    x = np.arange(len(action_labels))
    width = 0.35

    # Classe 1 in evidenza (barra superiore)
    bars1 = ax.barh(x - width/2, scores_c1_norm, width,
                    label='Class 1 (target)', color='steelblue', alpha=0.8)
    bars2 = ax.barh(x + width/2, scores_c0_norm, width,
                    label='Class 0', color='coral', alpha=0.8)

    # Labels e styling
    ax.set_yticks(x)
    ax.set_yticklabels(action_labels, fontsize=9)
    ax.set_xlabel('Attribution Score (each class normalized to own max = 1.0)', fontsize=12)
    ax.set_title(f'Top-{top_k} Clinical Actions Comparison\n(Ordered by Class 1, per-class normalization)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()  # Top action in alto

    # Aggiungi valori sulle barre (solo se > 0.05)
    for bars in [bars1, bars2]:
        for bar in bars:
            width_val = bar.get_width()
            if width_val > 0.05:
                ax.text(width_val + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{width_val:.3f}',
                        ha='left', va='center', fontsize=7)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Clinical actions comparison salvato: {save_path}")
    
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
