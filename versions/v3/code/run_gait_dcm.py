"""Compatibility adapter for the shared external walking reference.

New experiments should import :class:`DCMReferenceProvider` directly.  The
legacy torque-realizer smoke runner still expects the historical ``DCMWalk``
method names, so this adapter keeps that gate runnable.
"""

from __future__ import annotations

from reference_provider import DCMReferenceProvider


class DCMWalk(DCMReferenceProvider):
    def __init__(self, left0, right0, step_len, n_steps, z_c, t_step, t_ds,
                 t_settle, zmp_y_scale=1.0):
        super().__init__(
            left0, right0, step_length=step_len, n_steps=n_steps,
            com_height=z_c, step_time=t_step, double_support_time=t_ds,
            settle_time=t_settle, lateral_zmp_scale=zmp_y_scale,
        )
        self.w = self.omega
        self.total = self.total_time

    def xi_and_zmp(self, t):
        sample = self.sample(t)
        return sample.dcm_xy, sample.zmp_xy

    def schedule(self, t):
        sample = self.sample(t)
        return (
            sample.stance,
            sample.swing,
            sample.swing_progress,
            sample.swing_start_xy,
            sample.swing_target_xy,
        )
