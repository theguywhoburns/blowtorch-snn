import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Unpack

from ..base import (
    SpikingModule,
    SpikingModuleKwargs,
    StateSpec,
    Tensor,
    TensorConstraint,
    identity,
)
from ..surrogate import SpikeGrad, default_spike_grad

STANDARD_MEMBRANE: tuple[float, float, float] = (0.04, 5.0, 140.0)
CLASS1_MEMBRANE: tuple[float, float, float] = (0.04, 4.1, 108.0)


@dataclass(frozen=True)
class IzhParams:
    """Parameter set for the Izhikevich model.

    ``a``, ``b``, ``c``, ``d`` follow Izhikevich 2003. ``membrane`` is the
    ``(v^2, v, const)`` polynomial of the voltage equation, which differs for
    the class-1 excitable / integrator patterns. ``u_mode`` selects the
    recovery dynamics: ``"standard"`` (``du/dt = a(bv - u)``) or
    ``"accommodation"`` (``du/dt = a*b*(v + 65)``, pattern R).
    """

    a: float
    b: float
    c: float
    d: float
    membrane: tuple[float, float, float] = STANDARD_MEMBRANE
    u_mode: Literal["standard", "accommodation"] = "standard"


class IzhPreset(Enum):
    """Typed Izhikevich presets.

    The 2003 "canonical" types (RS, IB, CH, FS, LTS, TC, RZ) plus the 2004
    behavioral patterns (A-T) from Izhikevich's paper and ``figure1.m``.
    Access parameters via ``IzhPreset.RS.params`` and a human-readable name
    via ``IzhPreset.RS.description``.
    """

    RS = ("Regular spiking", IzhParams(0.02, 0.2, -65.0, 8.0))
    IB = ("Intrinsically bursting", IzhParams(0.02, 0.2, -55.0, 4.0))
    CH = ("Chattering", IzhParams(0.02, 0.2, -50.0, 2.0))
    FS = ("Fast spiking", IzhParams(0.1, 0.2, -65.0, 2.0))
    LTS = ("Low-threshold spiking", IzhParams(0.02, 0.25, -65.0, 2.0))
    TC = ("Thalamo-cortical", IzhParams(0.02, 0.25, -65.0, 0.05))
    RZ = ("Resonator (2003)", IzhParams(0.1, 0.26, -65.0, 2.0))

    TONIC_SPIKING = (
        "Tonic spiking",
        IzhParams(0.02, 0.2, -65.0, 6.0),
    )
    PHASIC_SPIKING = (
        "Phasic spiking",
        IzhParams(0.02, 0.25, -65.0, 6.0),
    )
    TONIC_BURSTING = (
        "Tonic bursting",
        IzhParams(0.02, 0.2, -50.0, 2.0),
    )
    PHASIC_BURSTING = (
        "Phasic bursting",
        IzhParams(0.02, 0.25, -55.0, 0.05),
    )
    MIXED_MODE = ("Mixed mode", IzhParams(0.02, 0.2, -55.0, 4.0))
    SPIKE_FREQUENCY_ADAPTATION = (
        "Spike-frequency adaptation",
        IzhParams(0.01, 0.2, -65.0, 8.0),
    )
    CLASS_1_EXCITABLE = (
        "Class 1 excitable",
        IzhParams(0.02, -0.1, -55.0, 6.0, membrane=CLASS1_MEMBRANE),
    )
    CLASS_2_EXCITABLE = (
        "Class 2 excitable",
        IzhParams(0.2, 0.26, -65.0, 0.0),
    )
    SPIKE_LATENCY = ("Spike latency", IzhParams(0.02, 0.2, -65.0, 6.0))
    SUBTHRESHOLD_OSCILLATION = (
        "Subthreshold oscillation",
        IzhParams(0.05, 0.26, -60.0, 0.0),
    )
    RESONATOR = ("Resonator (2004)", IzhParams(0.1, 0.26, -60.0, -1.0))
    INTEGRATOR = (
        "Integrator",
        IzhParams(0.02, -0.1, -55.0, 6.0, membrane=CLASS1_MEMBRANE),
    )
    REBOUND_SPIKE = ("Rebound spike", IzhParams(0.03, 0.25, -60.0, 4.0))
    REBOUND_BURST = ("Rebound burst", IzhParams(0.03, 0.25, -52.0, 0.0))
    THRESHOLD_VARIABILITY = (
        "Threshold variability",
        IzhParams(0.03, 0.25, -60.0, 4.0),
    )
    BISTABILITY = ("Bistability", IzhParams(0.1, 0.26, -60.0, 0.0))
    DAP = ("Depolarizing after-potential", IzhParams(1.0, 0.2, -60.0, -21.0))
    ACCOMMODATION = (
        "Accommodation",
        IzhParams(0.02, 1.0, -55.0, 4.0, u_mode="accommodation"),
    )
    INHIBITION_INDUCED_SPIKING = (
        "Inhibition-induced spiking",
        IzhParams(-0.02, -1.0, -60.0, 8.0),
    )
    INHIBITION_INDUCED_BURSTING = (
        "Inhibition-induced bursting",
        IzhParams(-0.026, -1.0, -45.0, -2.0),
    )

    @property
    def description(self) -> str:
        return self.value[0]

    @property
    def params(self) -> IzhParams:
        return self.value[1]


class Izhikevich(SpikingModule):
    """Izhikevich neuron.

    Follows the detect-first ordering loop: the spike is
    detected from the membrane at the start of the step, fired neurons reset
    to ``c``, ``u`` receives the after-spike kick ``d * spk``, then the
    membrane integrates and ``u`` is updated from the freshly integrated
    membrane.

    The forward pass reproduces the paper's hard reset exactly. The backward
    pass flows through the surrogate gradient for spike detection; the reset
    is straight-through (the masked-fill mask is a constant), so training
    never sees a gradient through the reset value.

    Integration uses ``substeps`` explicit-Euler sub-steps of size
    ``dt / substeps`` (default two ``0.5`` half-steps) for numerical
    stability.

    Note:
        This ordering can produce spike times shifted by one step relative to
        implementations that integrate first and then detect threshold
        crossing. This is intentional for this implementation and should be
        noted when comparing against other Izhikevich implementations.
    """

    neuron_name = "Izhikevich"

    mem: Tensor
    u: Tensor

    _constrained_param_specs = {"threshold": "threshold_constraint"}

    def __init__(
        self,
        a: Optional[float] = None,
        b: Optional[float] = None,
        c: Optional[float] = None,
        d: Optional[float] = None,
        threshold: float = 30.0,
        dt: float = 1.0,
        substeps: int = 2,
        spike_detection: Literal["pre", "post"] = "pre",
        u_init: Optional[float] = None,
        spike_grad: SpikeGrad = default_spike_grad,
        learnable_threshold: bool = False,
        threshold_constraint: TensorConstraint = identity,
        preset: Optional[IzhPreset] = None,
        v_reset: Optional[float] = None,
        **kwargs: Unpack[SpikingModuleKwargs],
    ):
        if not isinstance(substeps, int) or substeps < 1:
            raise ValueError(
                f"Izhikevich substeps must be a positive int, got {substeps!r}"
            )
        if spike_detection not in ("pre", "post"):
            raise ValueError(
                "Izhikevich spike_detection must be 'pre' or 'post', "
                f"got {spike_detection!r}"
            )
        defaults = IzhParams(0.02, 0.2, -65.0, 8.0)
        given = tuple(
            name
            for name, val in (("a", a), ("b", b), ("c", c), ("d", d))
            if val is not None
        )
        if preset is not None:
            if not isinstance(preset, IzhPreset):
                raise ValueError(
                    f"preset must be an IzhPreset, got {preset!r}; "
                    f"use IzhPreset.RS, IzhPreset.CH, ..."
                )
            if given:
                raise ValueError(
                    f"preset={preset.name} already sets {', '.join(given)}; "
                    f"pass a preset or explicit parameters, not both "
                    f"(v_reset/u_init still override c and the u init)"
                )
            params = preset.params
            a, b, c, d = params.a, params.b, params.c, params.d
            self.membrane = params.membrane
            self.u_mode = params.u_mode
        else:
            self.membrane = STANDARD_MEMBRANE
            self.u_mode = "standard"
            a = defaults.a if a is None else a
            b = defaults.b if b is None else b
            c = defaults.c if c is None else c
            d = defaults.d if d is None else d
        if v_reset is not None:
            c = v_reset
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v_reset = c
        self.dt = dt
        self.substeps = substeps
        self.spike_detection = spike_detection
        self.u_init = u_init if u_init is not None else self.b * self.c
        self.threshold = nn.Parameter(
            torch.tensor(threshold), requires_grad=learnable_threshold
        )
        self.threshold_constraint = threshold_constraint
        self.spike_grad = spike_grad

    def _get_state_specs(self) -> tuple[StateSpec, ...]:
        return (
            StateSpec("spk", 0.0, differentiable=False),
            StateSpec("mem", self.c),
            StateSpec("u", self.u_init),
        )

    def _integrate(self, x: Tensor, mem: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
        """Advance membrane and recovery variables without spike reset."""
        v2, v1, bias = self.membrane
        h = self.dt / self.substeps

        if self.u_mode == "accommodation":
            for _ in range(self.substeps):
                mem = mem + h * (v2 * mem * mem + v1 * mem + bias - u + x)
                u = u + h * self.a * self.b * (mem + 65.0)
        else:
            for _ in range(self.substeps):
                mem = mem + h * (v2 * mem * mem + v1 * mem + bias - u + x)
                u = u + h * self.a * (self.b * mem - u)

        return mem, u

    def _step(self, x: Tensor, mem: Tensor, u: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        (threshold,) = self._constrained_params()

        if self.spike_detection == "post":
            mem, u = self._integrate(x, mem, u)
            spk = self.spike_grad(mem - threshold)
            mem = mem.masked_fill(spk > 0, self.c)
            u = u + self.d * spk
            return spk, mem, u

        # Existing detect-first behavior.
        spk = self.spike_grad(mem - threshold)
        mem = mem.masked_fill(spk > 0, self.c)
        u = u + self.d * spk
        mem, u = self._integrate(x, mem, u)

        return spk, mem, u
