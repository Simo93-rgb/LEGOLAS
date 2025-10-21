"""
Diagnose Numerical Instability in IG Results
Analizza i risultati IG per identificare problemi numerici vs algoritmici

Usage:
    python scripts/diagnose_ig_numerical.py output/explainability/ig_results_*.pkl
"""

import pickle
import sys
from pathlib import Path


def diagnose_results(results_file):
    """Analizza risultati IG per problemi numerici"""
    
    print(f"\n{'='*80}")
    print(f"Analyzing: {results_file.name}")
    print(f"{'='*80}\n")
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Total samples: {len(results)}\n")
    
    # Categorie
    numerical_issues = []
    algorithmic_issues = []
    converged = []
    
    for idx, res in enumerate(results):
        if not res.get('diagnostics'):
            continue
            
        diag = res['diagnostics']
        
        if 'fold_diagnostics' not in diag:
            continue
        
        # Analizza per-fold
        fold_stats = {
            'sample_idx': idx,
            'avg_rel_error': diag.get('avg_rel_error', float('inf')),
            'converged_folds': diag.get('converged_folds', 0),
            'total_folds': diag.get('total_folds', 0),
            'folds': []
        }
        
        for fold_idx, fd in enumerate(diag['fold_diagnostics']):
            fold_info = {
                'fold_idx': fold_idx,
                'rel_error': fd['rel_error'],
                'f_x': fd.get('f_x', 0),
                'f_baseline': fd.get('f_baseline', 0),
                'denominator': fd.get('denominator', 0),
                'numerical_instability': fd.get('numerical_instability', False)
            }
            fold_stats['folds'].append(fold_info)
        
        # Categorizza
        has_numerical = any(f['numerical_instability'] for f in fold_stats['folds'])
        max_error = max(f['rel_error'] for f in fold_stats['folds'] if f['rel_error'] != float('inf'))
        
        if has_numerical:
            numerical_issues.append(fold_stats)
        elif max_error > 1.0:
            algorithmic_issues.append(fold_stats)
        elif diag['converged_folds'] == diag['total_folds']:
            converged.append(fold_stats)
    
    # Report
    print(f"📊 Summary:")
    print(f"   ✅ Fully converged: {len(converged)} samples")
    print(f"   💤 Numerical instability (f(x)≈f(baseline)): {len(numerical_issues)} samples")
    print(f"   🔥 Algorithmic issues (>100% error): {len(algorithmic_issues)} samples")
    
    # Details - Numerical issues
    if numerical_issues:
        print(f"\n💤 Numerical Instability Details (top 5):")
        for i, sample in enumerate(numerical_issues[:5], 1):
            print(f"\n   {i}. Sample {sample['sample_idx']}:")
            print(f"      Avg rel_error: {sample['avg_rel_error']:.4f}")
            print(f"      Converged folds: {sample['converged_folds']}/{sample['total_folds']}")
            
            for fold in sample['folds']:
                if fold['numerical_instability']:
                    print(f"         Fold {fold['fold_idx']}: "
                          f"f(x)={fold['f_x']:.6f}, f(b)={fold['f_baseline']:.6f}, "
                          f"|f(x)-f(b)|={fold['denominator']:.6f} (< 1e-4)")
    
    # Details - Algorithmic issues
    if algorithmic_issues:
        print(f"\n🔥 Algorithmic Issues Details (top 5):")
        for i, sample in enumerate(algorithmic_issues[:5], 1):
            print(f"\n   {i}. Sample {sample['sample_idx']}:")
            print(f"      Avg rel_error: {sample['avg_rel_error']:.4f}")
            print(f"      Converged folds: {sample['converged_folds']}/{sample['total_folds']}")
            
            for fold in sample['folds']:
                if fold['rel_error'] > 1.0 and not fold['numerical_instability']:
                    print(f"         Fold {fold['fold_idx']}: "
                          f"rel_error={fold['rel_error']:.4f}, "
                          f"f(x)={fold['f_x']:.4f}, f(b)={fold['f_baseline']:.4f}, "
                          f"|f(x)-f(b)|={fold['denominator']:.4f}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_ig_numerical.py <results_file.pkl>")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    
    if not results_path.exists():
        print(f"Error: File not found: {results_path}")
        sys.exit(1)
    
    diagnose_results(results_path)
