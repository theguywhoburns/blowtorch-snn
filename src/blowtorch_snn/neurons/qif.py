import torch
import torch.nn as nn
from typing import Optional, Unpack

from ..base import (
    SpikingModule,
    SpikingModuleKwargs,
    ResetFn,
    StateSpec,
    Tensor,
    TensorConstraint,
    _constrain_if_learnable,
    clamp_positive,
    identity,
    subtract_reset,
)
from ..surrogate import SpikeGrad, default_spike_grad


class QIF(SpikingModule):
    """Quadratic integrate-and-fire neuron.

    ``mem += beta * ((mem - v_rest) * (mem - v_thresh) + R * x)`` with
    optional hard clamps ``v_min``/``v_max``. Note ``v_thresh`` (the
    quadratic's saddle point) and ``threshold`` (spike detection) are
    distinct quantities that merely share a default value.
    """

    neuron_name = "QIF"

    mem: Tensor

    def __init__(
        self,
        beta: float,
        v_rest: float = 0.0,
        v_thresh: float = 1.0,
        membrane_resistance: float = 1.0,
        threshold: float = 1.0,
        spike_grad: SpikeGrad = default_spike_grad,
        learnable_beta: bool = False,
        learnable_threshold: bool = False,
        learnable_v_rest: bool = False,
        learnable_v_thresh: bool = False,
        learnable_membrane_resistance: bool = False,
        beta_constraint: TensorConstraint = clamp_positive,
        threshold_constraint: TensorConstraint = identity,
        reset_mechanism: ResetFn = subtract_reset,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        **kwargs: Unpack[SpikingModuleKwargs],
    ):
        super().__init__(**kwargs)
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=learnable_beta)
        self.v_rest = nn.Parameter(
            torch.tensor(v_rest), requires_grad=learnable_v_rest
        )
        self.v_thresh = nn.Parameter(
            torch.tensor(v_thresh), requires_grad=learnable_v_thresh
        )
        self.membrane_resistance = nn.Parameter(
            torch.tensor(membrane_resistance),
            requires_grad=learnable_membrane_resistance,
        )
        self.threshold = nn.Parameter(
            torch.tensor(threshold), requires_grad=learnable_threshold
        )
        self.beta_constraint = beta_constraint
        self.threshold_constraint = threshold_constraint
        self.spike_grad = spike_grad
        self.reset_mechanism = reset_mechanism
        self.v_min = v_min
        self.v_max = v_max

    def _get_state_specs(self) -> tuple[StateSpec, ...]:
        return (
            StateSpec("spk", 0.0, differentiable=False),
            StateSpec("mem", float(self.v_rest.detach())),
        )

    def _current_reset_values(self) -> tuple[float, ...]:
        values = list(super()._current_reset_values())
        # State order is: spk, mem. Keep the membrane reset/init in sync with
        # a learned v_rest instead of the construction-time snapshot.
        values[1] = float(self.v_rest.detach())
        return tuple(values)

    def _step(self, x: Tensor, mem: Tensor) -> tuple[Tensor, Tensor]:
        beta = _constrain_if_learnable(self.beta, self.beta_constraint)
        threshold = _constrain_if_learnable(
            self.threshold, self.threshold_constraint
        )
        mem = mem + beta * ((mem - self.v_rest) * (mem - self.v_thresh) + self.membrane_resistance * x)
        if self.v_min is not None:
            mem = torch.clamp(mem, min=self.v_min)
        if self.v_max is not None:
            mem = torch.clamp(mem, max=self.v_max)
        spk = self.spike_grad(mem - threshold)
        mem = self.reset_mechanism(mem, spk, threshold)
        return spk, mem
