import torch
import torch.nn as nn
from typing import Optional, Unpack

from ..base import (
    SpikingModule,
    SpikingModuleKwargs,
    StateSpec,
    Tensor,
    TensorConstraint,
    identity,
)
from ..surrogate import SpikeGrad, default_spike_grad


def _hh_rate(x: Tensor, a: float, c: float) -> Tensor:
    """Stable HH forward rate ``a * x / (1 - exp(-x / c))``.

    Uses the analytical limit ``a * c`` when ``x`` approaches zero. The
    denominator is replaced by 1 inside the mask so *both* ``torch.where``
    branches stay numerically safe in the backward pass (an unsafe
    unselected branch can leak ``0 * inf = NaN`` into gradients).
    """
    mask = x.abs() < 1e-4
    d = torch.where(mask, torch.ones_like(x), 1.0 - torch.exp(-x / c))
    return torch.where(mask, torch.full_like(x, a * c), a * x / d)


class HH(SpikingModule):
    """Hodgkin-Huxley neuron.

    Voltages are absolute millivolts (``ENa=+50``, ``EK=-77``, leak
    reversal ``EL=-54.4``); with the default conductances the resting
    potential is ``REST=-65`` mV. State is ``(mem, m, h, n)`` with the
    gating variables in ``[0, 1]``. ``dt`` is the integration step;
    ``substeps`` subdivides it for stability in long rollouts without
    changing the input scaling (each substep integrates ``dt/substeps``).
    ``initial_state`` already starts at rest with gates at their
    steady-state values; call ``steady_state(v=...)`` only if you change
    the conductances or want a different starting voltage.

    Note:
        Spike emission is currently a direct thresholded readout of the
        membrane voltage. Because HH is a continuous dynamical system, a
        single action potential can remain above threshold for multiple
        timesteps. If you need one discrete spike per action potential, use
        a higher threshold, edge detection, or a refractory/peak-detection
        postprocessor.
    """

    neuron_name = "HH"

    #: Resting potential of the classic parameter set (mV).
    REST: float = -65.0

    mem: Tensor
    m: Tensor
    h: Tensor
    n: Tensor

    def __init__(
        self,
        gNa: float = 120.0,
        gK: float = 36.0,
        gL: float = 0.3,
        ENa: float = 50.0,
        EK: float = -77.0,
        EL: float = -54.4,
        C: float = 1.0,
        threshold: float = 0.0,
        spike_grad: SpikeGrad = default_spike_grad,
        learnable_threshold: bool = False,
        threshold_constraint: TensorConstraint = identity,
        dt: float = 0.01,
        substeps: int = 1,
        **kwargs: Unpack[SpikingModuleKwargs],
    ):
        super().__init__(**kwargs)
        if not isinstance(substeps, int) or substeps < 1:
            raise ValueError(f"HH substeps must be a positive int, got {substeps!r}")
        self.gNa = gNa
        self.gK = gK
        self.gL = gL
        self.ENa = ENa
        self.EK = EK
        self.EL = EL
        self.C = C
        self.dt = dt
        self.substeps = substeps
        self.threshold = nn.Parameter(
            torch.tensor(threshold), requires_grad=learnable_threshold
        )
        self.threshold_constraint = threshold_constraint
        self.spike_grad = spike_grad

    def _get_state_specs(self) -> tuple[StateSpec, ...]:
        # Steady-state gates at REST with the classic parameters:
        # m_inf = 0.0529, h_inf = 0.5961, n_inf = 0.3177.
        return (
            StateSpec("spk", 0.0, differentiable=False),
            StateSpec("mem", self.REST),
            StateSpec("m", 0.0529),
            StateSpec("h", 0.5961),
            StateSpec("n", 0.3177),
        )

    def _steady_state_gates(self, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Steady-state gating ``(m, h, n)`` at membrane voltage ``v``."""
        am = _hh_rate(v + 40, 0.1, 10.0)
        bm = 4.0 * torch.exp(-(v + 65) / 18)
        ah = 0.07 * torch.exp(-(v + 65) / 20)
        bh = 1.0 / (1 + torch.exp(-(v + 35) / 10))
        an = _hh_rate(v + 55, 0.01, 10.0)
        bn = 0.125 * torch.exp(-(v + 65) / 80)
        return am / (am + bm), ah / (ah + bh), an / (an + bn)

    def steady_state(
        self,
        batch_shape: tuple[int, ...],
        v: float = -65.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[Tensor, ...]:
        """Explicit-state tuple at voltage ``v`` with gates at steady state.

        ``v`` defaults to ``-65`` mV, the resting potential of the classic
        parameter set; pass another voltage if you changed conductances.
        """
        volt = torch.full(batch_shape, v, device=device, dtype=dtype)
        m, h, n = self._steady_state_gates(volt)
        return (volt, m, h, n)

    def _step(
        self, x: Tensor, mem: Tensor, m: Tensor, h: Tensor, n: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        threshold = self.threshold_constraint(self.threshold)
        dt = self.dt / self.substeps
        for _ in range(self.substeps):
            gNa = self.gNa * (m ** 3) * h
            gK = self.gK * (n ** 4)
            INa = gNa * (mem - self.ENa)
            IK = gK * (mem - self.EK)
            IL = self.gL * (mem - self.EL)
            mem = mem + (x - INa - IK - IL) / self.C * dt

            am = _hh_rate(mem + 40, 0.1, 10.0)
            bm = 4.0 * torch.exp(-(mem + 65) / 18)
            ah = 0.07 * torch.exp(-(mem + 65) / 20)
            bh = 1.0 / (1 + torch.exp(-(mem + 35) / 10))
            an = _hh_rate(mem + 55, 0.01, 10.0)
            bn = 0.125 * torch.exp(-(mem + 65) / 80)

            m = torch.clamp(m + (am * (1 - m) - bm * m) * dt, 0.0, 1.0)
            h = torch.clamp(h + (ah * (1 - h) - bh * h) * dt, 0.0, 1.0)
            n = torch.clamp(n + (an * (1 - n) - bn * n) * dt, 0.0, 1.0)

        spk = self.spike_grad(mem - threshold)
        return spk, mem, m, h, n
