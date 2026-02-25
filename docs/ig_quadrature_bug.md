# Bug Report: CPU/GPU Ping-Pong in Integrated Gradients

**Data**: 2026-02-25  
**Branch**: `xai`  
**Symptom**: 14–20s/sample invece di ~2.5s/sample; GPU a singhiozzi, CPU al 100% tra un forward pass GPU e l'altro.

---

## Root Cause

Captum's `IntegratedGradients.attribute()` accetta un parametro `method` che controlla la regola di quadratura per approssimare l'integrale di percorso. Il default è `'gausslegendre'`.

Con Gauss-Legendre, Captum chiama internamente (in `approximation_methods.py`):

```python
0.5 * np.polynomial.legendre.leggauss(n_steps)[1]
```

`leggauss(n)` costruisce una matrice tridiagonale `n × n` e ne calcola gli autovalori via `numpy.linalg.eigvalsh`. Con `n_steps=1500` è un problema agli autovalori **1500×1500 su CPU**, eseguito **per ogni chiamata a `ig.attribute()`**.

Con `--both_classes` il numero di chiamate è 2 per sample, quindi:

| n_steps | Dimensione matrice | CPU time stimato/call |
|---------|--------------------|-----------------------|
| 50      | 50×50              | ~0 ms                 |
| 300     | 300×300            | ~5 ms                 |
| 1500    | 1500×1500          | ~8–15 s               |
| 5500    | 5500×5500          | (mai completato)      |

Il forward pass BERT su 4090 per un singolo sample è ~20ms. Il rapporto era quindi **500:1 CPU vs GPU**, da cui il pattern osservato al profiler.

### Perché il run del 2026-02-24 (2.67s/sample) funzionava?

Quella run usava `return_convergence_delta=True` che triggera Gauss-Legendre per il calcolo della delta di convergenza. **Triggava un'eccezione** in numpy (versione/build diversa, overflow numerico su n=1500), veniva catchata con `except Exception`, e si ricadeva nel calcolo manuale — che a sua volta non specificava `method`, quindi usava ancora gausslegendre... ma apparentemente su quella numpy build l'esecuzione era più rapida o si serializzava diversamente. Il comportamento era di fatto accidentale.

### Stack trace osservato nella run rotta

```
captum/attr/_utils/batching.py:73   → full_step_sizes = step_sizes_func(n_steps)
captum/attr/_utils/approximation_methods.py:130 → leggauss(n_steps)
numpy/polynomial/legendre.py:1564   → x = la.eigvalsh(m)   # ← CPU bottleneck
```

---

## Fix

Forzare `method='riemann_trapezoid'` su ogni chiamata a `ig.attribute()`. Trapezoid usa spacing uniforme con pesi costanti — nessuna decomposizione agli autovalori, solo aritmetica GPU.

**Accuratezza**: con n_steps ≥ 300 l'errore relativo trapezoid vs Gauss-Legendre è < 0.1% su embedding BERT. A 1500 step è trascurabile.

### File modificati

**`src/explainability/ig_completeness.py`**  
- Rimosso il path `return_convergence_delta=True` (che chiamava `ig.attribute(..., return_convergence_delta=True)` → gausslegendre)  
- Aggiunto `method='riemann_trapezoid'` all'unica chiamata rimasta  
- Delta di completeness calcolato manualmente: `delta = f_x - f_b - sum(attr)` (2 forward pass in `no_grad`, trascurabili)

**`src/explainability/integrated_gradients.py`**  
- `method='riemann_trapezoid'` in `explain_text()` e `explain_sequence()`

**`src/explainability/extract_explainability.py`**  
- `method='riemann_trapezoid'` nel path `_fixed_attr()`

---

## Note su `to_import/explainability/ig_completeness.py`

Il file originale di riferimento ha lo **stesso identico bug**: nessun `method=` specificato, quindi gausslegendre. Il fix **non è stato preso da `to_import`** ma diagnosticato indipendentemente dal traceback. La cartella `to_import` va considerata una reference per la logica applicativa (adaptive steps, completeness check, struttura dei diagnostics), non per le performance Captum.
