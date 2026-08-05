import torch
import torch.nn as nn
from typing import Unpack

from ..base import (
    SpikingModule,
    SpikingModuleKwargs,
    ResetFn,
    StateSpec,
    Tensor,
    TensorConstraint,
    clamp_unit_interval,
    clamp_positive,
    subtract_reset,
)
from ..surrogate import SpikeGrad, default_spike_grad

class LIF(SpikingModule):
    """Leaky integrate-and-fire neuron.

    ``mem[t] = beta * mem[t-1] + x[t]``; a spike is emitted (via the
    surrogate gradient) where ``mem`` crosses ``threshold``, then
    ``reset_mechanism`` is applied (default: subtract the threshold).
    ``beta`` is the per-step leak gain, constrained to ``[0, 1]`` by
    default for stability.
    """

    neuron_name = "LIF"
    mem: Tensor

    _constrained_param_specs = {
        "beta": "beta_constraint",
        "threshold": "threshold_constraint",
    }

    def __init__(
        self,
        beta: float,
        threshold: float = 1.0,
        learnable_beta: bool = False,
        learnable_threshold: bool = False,
        spike_grad: SpikeGrad = default_spike_grad,
        beta_constraint: TensorConstraint = clamp_unit_interval,
        threshold_constraint: TensorConstraint = clamp_positive,
        reset_mechanism: ResetFn = subtract_reset,
        **kwargs: Unpack[SpikingModuleKwargs],
    ):
        super().__init__(**kwargs)
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=learnable_beta)
        self.threshold = nn.Parameter(
            torch.tensor(threshold), requires_grad=learnable_threshold
        )
        self.beta_constraint = beta_constraint
        self.threshold_constraint = threshold_constraint
        self.spike_grad = spike_grad
        self.reset_mechanism = reset_mechanism

    def _get_state_specs(self) -> tuple[StateSpec, ...]:
        return (
            StateSpec("spk", 0.0, differentiable=False),
            StateSpec("mem", 0.0),
        )

    def _step(self, x: Tensor, mem: Tensor) -> tuple[Tensor, Tensor]:
        beta, threshold = self._constrained_params()
        mem = beta * mem + x
        spk = self.spike_grad(mem - threshold)
        mem = self.reset_mechanism(mem, spk, threshold)
        return spk, mem
