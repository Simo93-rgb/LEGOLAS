"""
Integrated Gradients Completeness Check
Verifica la convergenza di IG secondo l'assioma di Completeness (Sundararajan et al., 2017)

Completeness: sum(attributions) ≈ f(x) - f(baseline)

Uso:
    attributions, diag = compute_ig_with_completeness_check(...)
    if diag['rel_error'] > 0.05:
        print(f"Warning: IG non converso (rel_error={diag['rel_error']:.3f})")
"""

import torch
from captum.attr import IntegratedGradients
from typing import Callable, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def compute_ig_with_completeness_check(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    input_embeds: torch.Tensor,
    baseline_embeds: torch.Tensor,
    target_class: int,
    n_steps: int = 50,
    internal_batch_size: int = 32,
    device: str = 'cuda',
    eps: float = 1e-8,
    return_convergence_delta: bool = False  # kept for API compat, always ignored
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute Integrated Gradients con verifica di completeness.

    IMPORTANT: always uses method='riemann_trapezoid' to avoid the
    Gauss-Legendre CPU eigendecomposition (O(n_steps^2)) that fires when
    Captum's default 'gausslegendre' method is used.  With 1500 steps,
    riemann_trapezoid is accurate enough and 10-50x faster.

    Completeness axiom: sum(attributions) ≈ f(x) - f(baseline)
    Delta is computed manually: delta = f_x - f_baseline - sum(attr)
    """
    # Ensure tensors on correct device
    input_embeds = input_embeds.to(device)
    baseline_embeds = baseline_embeds.to(device)

    # Clone and require grad for IG
    input_embeds_grad = input_embeds.clone().detach().requires_grad_(True)
    baseline_embeds_clean = baseline_embeds.clone().detach()

    # Initialize IntegratedGradients with riemann_trapezoid to avoid the
    # Gauss-Legendre CPU eigendecomposition (leggauss(n_steps)) that fires
    # with Captum's default 'gausslegendre' method and is O(n_steps^2) on CPU.
    ig = IntegratedGradients(forward_fn)

    attributions = ig.attribute(
        inputs=input_embeds_grad,
        baselines=baseline_embeds_clean,
        method='riemann_trapezoid',
        target=target_class,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
    )

    # Compute completeness delta manually (f_x - f_baseline - sum_attr)
    with torch.no_grad():
        f_x = forward_fn(input_embeds).squeeze(0)[target_class].item()
        f_b = forward_fn(baseline_embeds).squeeze(0)[target_class].item()

    sum_attr = attributions.sum().item()
    delta = f_x - f_b - sum_attr
    abs_error = abs(delta)

    # Relative error (safe division con soglia per stabilità numerica)
    denominator = abs(f_x - f_b)

    # Soglia: se differenza < 0.0001, consideriamo f(x) ≈ f(baseline)
    NUMERICAL_THRESHOLD = 1e-4

    if denominator < NUMERICAL_THRESHOLD:
        # f(x) ≈ f(baseline) → attributions dovrebbero essere ~0
        if abs_error < NUMERICAL_THRESHOLD:
            rel_error = 0.0  # Convergenza perfetta su signal nullo
        else:
            # Errore numerico: signal troppo debole per IG affidabile
            rel_error = float('inf')
    else:
        rel_error = abs_error / denominator

    # Convergence criterion: rel_error < 5% E signal sufficiente
    converged = rel_error < 0.05 and denominator >= NUMERICAL_THRESHOLD

    diagnostics = {
        'f_x': float(f_x),
        'f_baseline': float(f_b),
        'sum_attributions': float(sum_attr),
        'delta': float(delta),
        'abs_error': float(abs_error),
        'rel_error': float(rel_error),
        'denominator': float(denominator),
        'n_steps': int(n_steps),
        'converged': bool(converged),
        'numerical_instability': bool(denominator < NUMERICAL_THRESHOLD)
    }

    return attributions.detach(), diagnostics


def find_optimal_n_steps(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    input_embeds: torch.Tensor,
    baseline_embeds: torch.Tensor,
    target_class: int,
    initial_steps: int = 50,
    max_steps: int = 1000,
    max_attempts: int = 5,
    target_rel_error: float = 0.01,
    internal_batch_size: int = 32,
    device: str = 'cuda',
    verbose: bool = True
) -> Tuple[torch.Tensor, Dict, int]:
    """
    Trova automaticamente il numero ottimale di steps aumentando n_steps fino a convergenza.

    Strategy:
    1. Start con initial_steps
    2. Compute IG e controlla rel_error
    3. Se rel_error > target → raddoppia steps e riprova
    4. Max max_attempts tentativi

    Args:
        forward_fn: Forward function per IG
        input_embeds: Input embeddings
        baseline_embeds: Baseline embeddings
        target_class: Target class
        initial_steps: Starting number of steps (default: 50)
        max_steps: Maximum steps to try (default: 1000)
        max_attempts: Max doubling attempts (default: 5)
        target_rel_error: Target relative error threshold (default: 0.01 = 1%)
        internal_batch_size: Batch size for IG steps
        device: Device
        verbose: Print progress

    Returns:
        best_attributions: IG attributions con miglior convergenza
        best_diagnostics: Diagnostics del best run
        optimal_steps: Numero ottimale di steps trovato

    Example:
        >>> attr, diag, optimal_m = find_optimal_n_steps(
        ...     forward_fn, embeds, baseline, target_class=1,
        ...     initial_steps=50, target_rel_error=0.01
        ... )
        >>> print(f"Optimal n_steps: {optimal_m}, rel_error: {diag['rel_error']:.4f}")
    """
    n_steps = initial_steps
    best_attributions = None
    best_diagnostics = None

    for attempt in range(max_attempts):
        if verbose:
            print(f"   🔄 Attempt {attempt+1}/{max_attempts}: n_steps={n_steps}")

        attributions, diagnostics = compute_ig_with_completeness_check(
            forward_fn=forward_fn,
            input_embeds=input_embeds,
            baseline_embeds=baseline_embeds,
            target_class=target_class,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            device=device
        )

        best_attributions = attributions
        best_diagnostics = diagnostics

        if verbose:
            print(f"      rel_error={diagnostics['rel_error']:.6f}, "
                  f"abs_error={diagnostics['abs_error']:.6f}")

        # Check convergenza
        if diagnostics['rel_error'] <= target_rel_error:
            if verbose:
                print(f"   ✅ Convergenza raggiunta con n_steps={n_steps}")
            return best_attributions, best_diagnostics, n_steps

        # Raddoppia steps (con limite)
        n_steps = min(n_steps * 2, max_steps)

        # Se raggiunto max_steps, stop
        if n_steps >= max_steps:
            if verbose:
                print(f"   ⚠️  Raggiunto max_steps={max_steps} senza convergenza completa")
            break

    if verbose:
        print(f"   ⚠️  Convergenza non raggiunta dopo {max_attempts} tentativi")
        if best_diagnostics:
            print(f"      Best rel_error={best_diagnostics['rel_error']:.6f} "
                  f"con n_steps={best_diagnostics['n_steps']}")

    return best_attributions, best_diagnostics, best_diagnostics['n_steps'] if best_diagnostics else initial_steps
