import torch
import torch.nn as nn
from typing import Unpack

from ..base import (
    SpikingModule,
    SpikingModuleKwargs,
    StateSpec,
    Tensor,
    TensorConstraint,
    clamp_positive,
    identity,
)
from ..surrogate import SpikeGrad, default_spike_grad


class AdEx(SpikingModule):
    """Adaptive exponential integrate-and-fire neuron.

    ``x`` is the input current pre-scaled into voltage units (R·I folded
    in), added to the membrane equation every step. ``beta`` is the Euler
    gain ``dt`` (with tau_m = 1); the adaptation current ``w`` shares the
    same step size via ``beta / tau_w``. Fires when ``mem`` crosses
    ``threshold``, hard-resets ``mem`` to ``v_reset`` and kicks
    ``w`` by ``b``.
    """

    neuron_name = "AdEx"

    mem: Tensor
    w: Tensor

    _constrained_param_specs = {
        "beta": "beta_constraint",
        "threshold": "threshold_constraint",
    }

    def __init__(
        self,
        beta: float,
        a: float = 4.0,
        b: float = 0.0805,
        v_rest: float = -70.0,
        v_thresh: float = -50.0,
        v_reset: float = -70.0,
        delta_T: float = 2.0,
        tau_w: float = 144.0,
        threshold: float = -50.0,
        spike_grad: SpikeGrad = default_spike_grad,
        learnable_beta: bool = False,
        learnable_threshold: bool = False,
        beta_constraint: TensorConstraint = clamp_positive,
        threshold_constraint: TensorConstraint = identity,
        **kwargs: Unpack[SpikingModuleKwargs],
    ):
        super().__init__(**kwargs)
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=learnable_beta)
        self.a = a
        self.b = b
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.delta_T = delta_T
        self.tau_w = tau_w
        self.threshold = nn.Parameter(
            torch.tensor(threshold), requires_grad=learnable_threshold
        )
        self.beta_constraint = beta_constraint
        self.threshold_constraint = threshold_constraint
        self.spike_grad = spike_grad

    def _get_state_specs(self) -> tuple[StateSpec, ...]:
        return (
            StateSpec("spk", 0.0, differentiable=False),
            StateSpec("mem", self.v_rest),
            StateSpec("w", 0.0),
        )

    def _step(self, x: Tensor, mem: Tensor, w: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        beta, threshold = self._constrained_params()
        exponent = torch.clamp((mem - self.v_thresh) / self.delta_T, max=20.0)
        exp_term = self.delta_T * torch.exp(exponent)
        mem = mem + beta * (-(mem - self.v_rest) + exp_term - w + x)
        w = w + beta * (self.a * (mem - self.v_rest) - w) / self.tau_w
        spk = self.spike_grad(mem - threshold)
        mem = mem.masked_fill(spk > 0, self.v_reset)
        w = w + self.b * spk
        return spk, mem, w
