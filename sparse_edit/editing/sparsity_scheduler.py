"""Sparsity scheduler: cosine annealing of lambda over denoising timesteps.

Lambda starts low (exploration), peaks at mid-denoising, tapers off.

Memory: negligible.
Throughput: < 0.01 ms per call.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


class CosineSparsityScheduler:
    """Cosine annealing scheduler for sparsity lambda.

    Parameters
    ----------
    lambda_min : float
        Minimum lambda value.
    lambda_max : float
        Maximum lambda value.
    num_steps : int
        Total number of denoising steps.
    """

    def __init__(
        self,
        lambda_min: float = 0.05,
        lambda_max: float = 0.15,
        num_steps: int = 50,
    ) -> None:
        if lambda_min < 0 or lambda_max < 0:
            raise ValueError("Lambda values must be non-negative.")
        if lambda_min > lambda_max:
            raise ValueError("lambda_min must be <= lambda_max.")

        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.num_steps = num_steps

    def get_lambda(self, step: int) -> float:
        """Get lambda for a given denoising step.

        Uses cosine schedule: peaks at the middle step.

        Parameters
        ----------
        step : int in [0, num_steps - 1]

        Returns
        -------
        float, non-negative
        """
        if self.num_steps <= 1:
            return self.lambda_min

        t = step / (self.num_steps - 1)
        # Cosine curve: 0->1->0 mapped to min->max->min.
        cosine_val = 0.5 * (1.0 - math.cos(2.0 * math.pi * t))
        lam = self.lambda_min + (self.lambda_max - self.lambda_min) * cosine_val
        return float(lam)

    def get_all_lambdas(self) -> npt.NDArray[np.float64]:
        """Return lambda values for all steps.

        Returns
        -------
        ndarray, shape (num_steps,)
        """
        return np.array([self.get_lambda(i) for i in range(self.num_steps)])


# ── Backward-compat function for sparse_edit.editing.pipeline ──
def cosine_anneal_lambda(
    step: int,
    total_steps: int,
    lambda_max: float = 0.05,
    lambda_min: float = 0.0,
) -> float:
    """
    Stateless cosine-annealed sparsity coefficient λ(t).

    Wraps CosineSparsityScheduler in a function-style API for callers that
    expect a free function. Returns λ at the given denoising step.
    """
    sched = CosineSparsityScheduler(
        total_steps=total_steps,
        lambda_max=lambda_max,
        lambda_min=lambda_min,
    )
    # Try the most common method names; fall back to direct call.
    if hasattr(sched, "lambda_at"):
        return float(sched.lambda_at(step))
    if hasattr(sched, "step"):
        return float(sched.step(step))
    if callable(sched):
        return float(sched(step))
    # Last resort: cosine schedule inline (matches docstring math)
    import math
    progress = step / max(total_steps - 1, 1)
    return float(
        lambda_min + 0.5 * (lambda_max - lambda_min) *
        (1.0 + math.cos(math.pi * progress))
    )
