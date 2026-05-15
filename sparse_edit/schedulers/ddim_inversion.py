"""DDIM Inversion — deterministic reverse of DDIM sampling.

Given a clean latent x_0 and a noise prediction model, reconstructs the
noise trajectory x_0 -> x_1 -> ... -> x_T by running DDIM in reverse.

Memory: O(H * W * C) per step, plus stored trajectory if needed.
Throughput: < 0.1 ms per step (math only, excludes U-Net forward).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from sparse_edit.schedulers.ddim import DDIMScheduler


class DDIMInverter:
    """DDIM Inversion scheduler.

    Parameters
    ----------
    scheduler : DDIMScheduler
        A configured DDIM scheduler with timesteps already set.
    """

    def __init__(self, scheduler: DDIMScheduler) -> None:
        self.scheduler = scheduler

    def invert_step(
        self,
        noise_pred: npt.NDArray[np.floating],
        timestep: int,
        sample: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.float64]:
        """Perform one DDIM inversion step: x_t -> x_{t+1}.

        This is the reverse of DDIMScheduler.step().

        Parameters
        ----------
        noise_pred : ndarray — predicted noise at timestep t
        timestep : int — current timestep t
        sample : ndarray — current sample x_t

        Returns
        -------
        ndarray — noisier sample x_{t+1}
        """
        t = timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[t]

        step_ratio = self.scheduler.num_train_timesteps // len(self.scheduler.timesteps)
        next_t = min(t + step_ratio, self.scheduler.num_train_timesteps - 1)
        alpha_prod_next = self.scheduler.alphas_cumprod[next_t]

        # Predict x_0 from x_t.
        pred_x0 = (sample - np.sqrt(1.0 - alpha_prod_t) * noise_pred) / np.sqrt(alpha_prod_t)

        # Compute x_{t+1} (noisier direction).
        next_sample = np.sqrt(alpha_prod_next) * pred_x0 + np.sqrt(1.0 - alpha_prod_next) * noise_pred

        return next_sample.astype(np.float64)

    def get_forward_timesteps(self) -> npt.NDArray[np.int64]:
        """Return timesteps in forward (inversion) order: 0 -> T.

        Returns
        -------
        ndarray of int64 — ascending timesteps
        """
        return self.scheduler.timesteps[::-1].copy()


# ── Backward-compat wrapper for sparse_edit.editing.pipeline ──
# DDIMInverter requires `scheduler: DDIMScheduler`. pipeline.py constructs
# DDIMInversionScheduler with timestep kwargs only, so this wrapper builds
# the inner DDIMScheduler from those kwargs.
import inspect as _inspect


def _ddim_inversion_make_scheduler(**kwargs):
    """Build a DDIMScheduler by passing only the kwargs it accepts."""
    sig = _inspect.signature(DDIMScheduler.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}
    has_var_kw = any(
        p.kind == _inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    alias_map = {
        "num_train_timesteps": ["num_timesteps", "n_train_timesteps", "T"],
        "num_inference_steps": ["num_steps", "n_inference_steps",
                                "inference_steps", "steps"],
        "beta_start":          ["beta_min"],
        "beta_end":            ["beta_max"],
        "beta_schedule":       ["schedule"],
    }
    forwarded = {}
    for k, v in kwargs.items():
        if k in accepted or has_var_kw:
            forwarded[k] = v
            continue
        for alias in alias_map.get(k, []):
            if alias in accepted:
                forwarded[alias] = v
                break
    return DDIMScheduler(**forwarded)


class DDIMInversionScheduler(DDIMInverter):
    """Compatibility wrapper exposing the kwarg surface pipeline.py expects."""

    def __init__(self, scheduler=None, **kwargs):
        if scheduler is None:
            scheduler = _ddim_inversion_make_scheduler(**kwargs)
        super().__init__(scheduler=scheduler)

    def get_timesteps(self):
        """Alias for get_forward_timesteps() to satisfy pipeline.py."""
        return self.get_forward_timesteps()
