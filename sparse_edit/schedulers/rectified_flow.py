"""Rectified Flow Scheduler.

Linear interpolation between data (t=0) and noise (t=1).
Velocity prediction model: v = noise - data.
ODE step: x_{t+dt} = x_t + v * dt.

Memory: negligible.
Throughput: < 0.1 ms per step.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class RectifiedFlowScheduler:
    """Rectified Flow ODE scheduler.

    Parameters
    ----------
    num_inference_steps : int
    """

    def __init__(self, num_inference_steps: int = 50) -> None:
        self.num_inference_steps = num_inference_steps
        self.timesteps = np.linspace(1.0, 0.0, num_inference_steps + 1, dtype=np.float64)

    def step(
        self,
        velocity: npt.NDArray[np.floating],
        t_idx: int,
        sample: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.float64]:
        """One Euler step of the rectified flow ODE.

        Parameters
        ----------
        velocity : ndarray — predicted velocity v(x_t, t)
        t_idx : int — index into self.timesteps
        sample : ndarray — current sample x_t

        Returns
        -------
        ndarray — next sample x_{t-dt}
        """
        t_cur = self.timesteps[t_idx]
        t_next = self.timesteps[t_idx + 1]
        dt = t_next - t_cur  # Negative (moving from noise to data).

        next_sample = sample + velocity * dt
        return next_sample.astype(np.float64)

    @staticmethod
    def interpolate(
        x0: npt.NDArray[np.floating],
        x1: npt.NDArray[np.floating],
        t: float,
    ) -> npt.NDArray[np.float64]:
        """Linear interpolation: x_t = (1-t)*x0 + t*x1.

        Parameters
        ----------
        x0 : ndarray — data
        x1 : ndarray — noise
        t : float in [0, 1]

        Returns
        -------
        ndarray
        """
        return ((1.0 - t) * x0 + t * x1).astype(np.float64)
