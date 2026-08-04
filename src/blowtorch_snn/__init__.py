import torch.nn as nn
from .base import (
    SpikingModule,
    SpikingModuleKwargs,
    StateSpec,
    StepFn,
    Forward,
    FusedSequenceFn,
    TensorConstraint,
    ResetFn,
    identity,
    clamp_unit_interval,
    clamp_positive,
    set_validation,
    get_validation,
    subtract_reset,
    zero_reset,
    hard_zero_reset,
    no_reset,
    no_validation,
    SequenceOutput,
)
from .neurons.lif import LIF
from .neurons.qif import QIF
from .neurons.izhikevich import Izhikevich, IzhPreset, IzhParams
from .neurons.adex import AdEx
from .neurons.srm import SRM
from .neurons.hh import HH

__all__ = [
    "SpikingModule",
    "SpikingModuleKwargs",
    "StateSpec",
    "StepFn",
    "Forward",
    "FusedSequenceFn",
    "TensorConstraint",
    "ResetFn",
    "identity",
    "clamp_unit_interval",
    "clamp_positive",
    "set_validation",
    "get_validation",
    "subtract_reset",
    "zero_reset",
    "hard_zero_reset",
    "no_reset",
    "no_validation",
    "SequenceOutput",
    "LIF",
    "QIF",
    "Izhikevich",
    "IzhPreset",
    "IzhParams",
    "AdEx",
    "SRM",
    "HH",
    "reset",
    "detach",
]


def reset(module: nn.Module) -> None:
    """Reset hidden state on every ``SpikingModule`` under ``module``."""
    for m in module.modules():
        if isinstance(m, SpikingModule):
            m.reset()


def detach(module: nn.Module) -> None:
    """Detach hidden state on every ``SpikingModule`` under ``module``."""
    for m in module.modules():
        if isinstance(m, SpikingModule):
            m.detach()
