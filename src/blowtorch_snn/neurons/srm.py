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
    _constrain_if_learnable,
    clamp_unit_interval,
    clamp_positive,
    subtract_reset,
)
from ..surrogate import SpikeGrad, default_spike_grad


class SRM(SpikingModule):
    """Spike Response Model.

    Like LIF, the membrane ``mem`` accumulates input and decays with ``beta``,
    but firing imposes a refractory period while the membrane keeps
    integrating. Counting the spike itself, the next spike can occur at
    earliest ``tau_ref`` steps later (``tau_ref - 1`` enforced silent steps
    follow each spike).
    """

    neuron_name = "SRM"

    mem: Tensor
    ref: Tensor

    def __init__(
        self,
        beta: float,
        threshold: float = 1.0,
        tau_ref: float = 4.0,
        spike_grad: SpikeGrad = default_spike_grad,
        learnable_beta: bool = False,
        learnable_threshold: bool = False,
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
        self.tau_ref = tau_ref
        self.spike_grad = spike_grad
        self.reset_mechanism = reset_mechanism

    def _get_state_specs(self) -> tuple[StateSpec, ...]:
        return (
            StateSpec("spk", 0.0, differentiable=False),
            StateSpec("mem", 0.0),
            StateSpec("ref", 0.0),
        )

    def _step(self, x: Tensor, mem: Tensor, ref: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        beta = _constrain_if_learnable(self.beta, self.beta_constraint)
        threshold = _constrain_if_learnable(
            self.threshold, self.threshold_constraint
        )
        ref = torch.clamp(ref - 1.0, min=0.0)
        mem = beta * mem + x
        spk = self.spike_grad(mem - threshold) * (ref <= 1e-6).to(mem.dtype)
        ref = ref + self.tau_ref * spk
        mem = self.reset_mechanism(mem, spk, threshold)
        return spk, mem, ref
