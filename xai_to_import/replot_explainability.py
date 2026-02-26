"""
Replot Explainability Script
Rigenera i plot a partire dai file pkl già calcolati, senza ricalcolare gli IG.

Uso tipico:
    # Da ig_results raw (ri-aggrega parole + azioni, poi plotta)
    uv run python -m src.explainability.replot_explainability \\
        --ig_results output/explainability/ig_results__bert-base-uncased_single_20260224_171659.pkl

    # Da actions pkl già aggregato (solo plot)
    uv run python -m src.explainability.replot_explainability \\
        --actions output/explainability/actions_bert-base-uncased_single_20260224_171659.pkl

    # Entrambi (usa actions per le azioni, ig_results per le parole)
    uv run python -m src.explainability.replot_explainability \\
        --ig_results output/explainability/ig_results__bert-base-uncased_single_20260224_171659.pkl \\
        --actions output/explainability/actions_bert-base-uncased_single_20260224_171659.pkl

    # Con top_k personalizzato e output dir esplicita
    uv run python -m src.explainability.replot_explainability \\
        --ig_results output/explainability/ig_results__bert-base-uncased_single_20260224_171659.pkl \\
        --top_k 15 --output_dir output/explainability/replot
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path


def _infer_stem(pkl_path: Path) -> str:
    """Estrae stem significativo dal nome del file pkl per i nomi dei plot."""
    name = pkl_path.stem
    # Rimuovi prefissi noti
    for prefix in ('ig_results__', 'ig_results_', 'actions_'):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name


def main():
    parser = argparse.ArgumentParser(
        description='Rigenera plot explainability da pkl esistenti (senza ricalcolare gli IG)'
    )
    parser.add_argument(
        '--ig_results',
        type=str,
        default=None,
        help='Path al file ig_results__.pkl (output raw di extract_explainability). '
             'Se fornito, ricalcola top_words e action aggregation, poi plotta.'
    )
    parser.add_argument(
        '--actions',
        type=str,
        default=None,
        help='Path al file actions_*.pkl (output aggregato di extract_explainability). '
             'Se fornito senza --ig_results, genera solo i plot delle azioni cliniche.'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=20,
        help='Numero di parole/azioni da visualizzare (default: 20)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory output per i plot. Default: stessa directory del pkl di input.'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default=None,
        help='Suffisso aggiuntivo per i nomi dei file di output (default: timestamp)'
    )
    args = parser.parse_args()

    if args.ig_results is None and args.actions is None:
        parser.error('Fornire almeno uno tra --ig_results e --actions')

    timestamp = args.suffix or datetime.now().strftime('%Y%m%d_%H%M%S')

    # Import lazy (evita overhead inutile se non disponibili)
    from src.explainability import (
        ClinicalActionAggregator,
        IntegratedGradientsExplainer,
        plot_attention_heatmap,
        plot_class_comparison,
        plot_clinical_actions_heatmap,
        plot_clinical_actions_comparison,
    )

    # -------------------------------------------------------------------------
    # Percorso A: ig_results raw → word aggregation + action aggregation + plot
    # -------------------------------------------------------------------------
    if args.ig_results:
        ig_results_path = Path(args.ig_results)
        if not ig_results_path.exists():
            raise FileNotFoundError(f'ig_results file not found: {ig_results_path}')

        print(f'📂 Loading ig_results from: {ig_results_path}')
        with open(ig_results_path, 'rb') as f:
            results = pickle.load(f)
        print(f'   ✅ Loaded {len(results)} samples')

        stem = _infer_stem(ig_results_path)
        out_dir = Path(args.output_dir) if args.output_dir else ig_results_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # Serve una istanza di IntegratedGradientsExplainer solo per extract_top_words
        # (non carica alcun modello, usa solo metodi statici)
        class _DummyExplainer:
            """Proxy leggero per extract_top_words (non richiede modello)."""
            def extract_top_words(self, results, top_k, by_class):
                import numpy as np
                if by_class:
                    has_dual = results and 'word_attr_class_0' in results[0] and 'word_attr_class_1' in results[0]
                    class_0_words: dict = {}
                    class_1_words: dict = {}
                    for r in results:
                        if has_dual:
                            for word, score in r.get('word_attr_class_0', {}).items():
                                class_0_words.setdefault(word, []).append(abs(score))
                            for word, score in r.get('word_attr_class_1', {}).items():
                                class_1_words.setdefault(word, []).append(abs(score))
                        else:
                            target = class_0_words if r.get('predicted_label', 0) == 0 else class_1_words
                            wa = r.get('word_attributions', {})
                            if isinstance(wa, dict):
                                for word, score in wa.items():
                                    target.setdefault(word, []).append(abs(score))
                    class_0_avg = {w: float(np.mean(s)) for w, s in class_0_words.items()}
                    class_1_avg = {w: float(np.mean(s)) for w, s in class_1_words.items()}
                    top_c0 = dict(sorted(class_0_avg.items(), key=lambda x: x[1], reverse=True)[:top_k])
                    top_c1 = dict(sorted(class_1_avg.items(), key=lambda x: x[1], reverse=True)[:top_k])
                    if has_dual:
                        print(f'   ℹ️  Dual-class mode: aggregated over all {len(results)} samples per class')
                    else:
                        print(f'   ⚠️  Legacy mode: split by predicted_label (some zeros expected in cross-class bars)')
                    return {'class_0': top_c0, 'class_1': top_c1}
                else:
                    all_words: dict = {}
                    for r in results:
                        for word, score in r.get('word_attributions', {}).items():
                            all_words.setdefault(word, []).append(abs(score))
                    avg = {w: float(np.mean(s)) for w, s in all_words.items()}
                    return {'all': dict(sorted(avg.items(), key=lambda x: x[1], reverse=True)[:top_k])}

        explainer = _DummyExplainer()

        # --- Word-level plots ---
        print(f'\n📊 Extracting top-{args.top_k} words...')
        top_words = explainer.extract_top_words(results, top_k=args.top_k, by_class=True)

        print(f'\n   Top {args.top_k} words for Class 1:')
        for i, (word, score) in enumerate(list(top_words['class_1'].items())[:10], 1):
            print(f'      {i:2d}. {word:20s} → {score:.4f}')
        print(f'\n   Top {args.top_k} words for Class 0:')
        for i, (word, score) in enumerate(list(top_words['class_0'].items())[:10], 1):
            print(f'      {i:2d}. {word:20s} → {score:.4f}')

        print('\n🎨 Creating word-level plots...')
        heatmap_words_path = out_dir / f'heatmap_words_{stem}_{timestamp}.png'
        plot_attention_heatmap(
            top_words['class_0'], top_words['class_1'],
            top_k=args.top_k, save_path=heatmap_words_path
        )
        histogram_words_path = out_dir / f'histogram_words_{stem}_{timestamp}.png'
        plot_class_comparison(
            top_words['class_0'], top_words['class_1'],
            top_k=args.top_k, save_path=histogram_words_path
        )

        # --- Action aggregation ---
        print('\n🏥 Aggregating clinical actions...')
        aggregator = ClinicalActionAggregator()
        action_results = aggregator.aggregate_across_dataset(results, by_class=True)

        top_actions = aggregator.get_top_actions(
            action_results, top_k=args.top_k, sort_by='mean_score'
        )
        if top_actions.get('class_1'):
            print(f'\n   Top {args.top_k} clinical actions for Class 1:')
            for i, (action, stats) in enumerate(list(top_actions['class_1'].items())[:5], 1):
                print(f'      {i}. {action[:60]}...')
                print(f'         Mean: {stats["mean_score"]:.4f}, Count: {stats["count"]}')
        if top_actions.get('class_0'):
            print(f'\n   Top {args.top_k} clinical actions for Class 0:')
            for i, (action, stats) in enumerate(list(top_actions['class_0'].items())[:5], 1):
                print(f'      {i}. {action[:60]}...')
                print(f'         Mean: {stats["mean_score"]:.4f}, Count: {stats["count"]}')

        # Save aggregated actions
        actions_out = out_dir / f'actions_{stem}_{timestamp}.pkl'
        with open(actions_out, 'wb') as f:
            pickle.dump(action_results, f)
        print(f'\n💾 Action results salvati: {actions_out}')

        print('\n🎨 Creating clinical action plots...')
        heatmap_actions_path = out_dir / f'heatmap_actions_{stem}_{timestamp}.png'
        plot_clinical_actions_heatmap(
            action_results['class_0'], action_results['class_1'],
            top_k=args.top_k, save_path=heatmap_actions_path
        )
        histogram_actions_path = out_dir / f'histogram_actions_{stem}_{timestamp}.png'
        plot_clinical_actions_comparison(
            action_results['class_0'], action_results['class_1'],
            top_k=args.top_k, save_path=histogram_actions_path
        )

        print(f'\n{"="*70}')
        print('✅ Replot completed!')
        print('\n📁 Output files:')
        print(f'   • Word Heatmap:        {heatmap_words_path}')
        print(f'   • Word Histogram:      {histogram_words_path}')
        print(f'   • Actions pkl:         {actions_out}')
        print(f'   • Actions Heatmap:     {heatmap_actions_path}')
        print(f'   • Actions Histogram:   {histogram_actions_path}')
        print(f'{"="*70}')
        return

    # -------------------------------------------------------------------------
    # Percorso B: actions pkl già aggregato → solo plot azioni
    # -------------------------------------------------------------------------
    if args.actions:
        actions_path = Path(args.actions)
        if not actions_path.exists():
            raise FileNotFoundError(f'actions file not found: {actions_path}')

        print(f'📂 Loading actions from: {actions_path}')
        with open(actions_path, 'rb') as f:
            action_results = pickle.load(f)
        print(f'   ✅ Loaded action results')

        stem = _infer_stem(actions_path)
        out_dir = Path(args.output_dir) if args.output_dir else actions_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        aggregator = ClinicalActionAggregator()
        top_actions = aggregator.get_top_actions(
            action_results, top_k=args.top_k, sort_by='mean_score'
        )
        if top_actions.get('class_1'):
            print(f'\n   Top {args.top_k} clinical actions for Class 1:')
            for i, (action, stats) in enumerate(list(top_actions['class_1'].items())[:5], 1):
                print(f'      {i}. {action[:60]}...')
                print(f'         Mean: {stats["mean_score"]:.4f}, Count: {stats["count"]}')
        if top_actions.get('class_0'):
            print(f'\n   Top {args.top_k} clinical actions for Class 0:')
            for i, (action, stats) in enumerate(list(top_actions['class_0'].items())[:5], 1):
                print(f'      {i}. {action[:60]}...')
                print(f'         Mean: {stats["mean_score"]:.4f}, Count: {stats["count"]}')

        print('\n🎨 Creating clinical action plots...')
        heatmap_actions_path = out_dir / f'heatmap_actions_{stem}_{timestamp}.png'
        plot_clinical_actions_heatmap(
            action_results['class_0'], action_results['class_1'],
            top_k=args.top_k, save_path=heatmap_actions_path
        )
        histogram_actions_path = out_dir / f'histogram_actions_{stem}_{timestamp}.png'
        plot_clinical_actions_comparison(
            action_results['class_0'], action_results['class_1'],
            top_k=args.top_k, save_path=histogram_actions_path
        )

        print(f'\n{"="*70}')
        print('✅ Replot completed!')
        print('\n📁 Output files:')
        print(f'   • Actions Heatmap:   {heatmap_actions_path}')
        print(f'   • Actions Histogram: {histogram_actions_path}')
        print(f'{"="*70}')


if __name__ == '__main__':
    main()
