import torch
import torch.nn as nn
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, Optional, TypedDict, Unpack

from .pack import pack_spikes

Tensor = torch.Tensor

NeuronOutput = Tensor | tuple[Tensor, ...]

ResetFn = Callable[[Tensor, Tensor, Tensor], Tensor]

TensorConstraint = Callable[[Tensor], Tensor]

#: Declared min/max bound for a state. ``None`` (or ``(None, None)``) means no
#: range; ``(lo, None)`` keeps values at least ``lo``; ``(None, hi)`` keeps them
#: at most ``hi``; ``(lo, hi)`` keeps them inside ``[lo, hi]``. Bounds are
#: inclusive.
ValueRange = tuple[Optional[float], Optional[float]]

Forward = Callable[..., NeuronOutput]

StepFn = Callable[..., tuple[Tensor, ...]]
"""See ``SpikingModule``: ``_step(x, *state) -> (spk, *next_state)``."""

#: Eager-mode sequence scan batch size: each chunk writes ``K`` timesteps with
#: one ``index_copy_`` instead of one per timestep. The eager scan then runs at
#: ``torch.stack`` speed (the per-step loop pays T tiny scatter launches; a
#: chunked write pays ~T/K) while peak memory stays at input + output + one
#: K-step transient instead of stack's full (T, B, F) list plus its copy.
_SEQUENCE_SCAN_CHUNK = 8

_GLOBAL_VALIDATE = True


def set_validation(enabled: bool) -> None:
    """Set the global default for ``SpikingModule(validate=...)``."""
    global _GLOBAL_VALIDATE
    _GLOBAL_VALIDATE = bool(enabled)


def get_validation() -> bool:
    """Return the current global validation default."""
    return _GLOBAL_VALIDATE


@contextmanager
def no_validation():
    """Context manager that disables global validation temporarily.

    Modules constructed with ``validate=None`` follow the global toggle, so
    wrapping a hot loop in ``no_validation()`` skips their per-forward checks.
    """
    prev = get_validation()
    set_validation(False)
    try:
        yield
    finally:
        set_validation(prev)


FusedSequenceFn = Callable[
    [Tensor, Optional[tuple[Tensor, ...]]], Tensor | tuple[Tensor, ...]
]
"""Signature for a ``_fused_forward_sequence`` implementation.

A kernel package assigns a fused kernel to ``module._fused_forward_sequence``;
until then, the shipped reference (``_reference_sequence_scan``) is wired in
when ``use_fused_sequence=True`` so the hook always has a tested consumer.
"""


class SpikingModuleKwargs(TypedDict, total=False):
    """Shared construction keywords forwarded from every neuron to
    ``SpikingModule``.

    ``size`` declares the feature dimension (feature-last), ``init_hidden``
    picks the state-tracking (hidden) vs. explicit path, ``validate`` overrides
    the global validation
    toggle per instance, and ``use_fused_sequence`` routes
    ``forward_sequence`` through ``_fused_forward_sequence`` (falling back to
    the reference per-step scan). All are optional and keyword-only on the
    neuron constructors. See the ``SpikingModule`` docstring for full
    semantics.
    """

    size: Optional[int]
    init_hidden: bool
    validate: Optional[bool]
    use_fused_sequence: bool

    #: Bit-pack returned spikes into ``int32`` (32 per word). WARN: preferred
    #: in compiled mode (``fast_sequence_``/``compile_sequence_scan``) where the
    #: pack fuses into the scan and peak memory drops; in plain eager mode the
    #: float ``(T, B, F)`` stack stays live while it is packed, so peak memory
    #: is not reduced.
    pack_output: bool


@dataclass(frozen=True)
class RangeSpec:
    """Declarative bound for a state tensor.

    ``clamp`` is the execution-time behavior. ``warn`` and ``error`` are
    diagnostic behaviors used only when validation/debug checks are enabled.
    """

    low: Optional[float] = None
    high: Optional[float] = None

    #: Execution-time enforcement. If True, values are clamped into range.
    clamp: bool = True

    #: Debug/validation only: warn when a violation is observed.
    warn: bool = False

    #: Debug/validation only: raise when a violation is observed.
    #: If both ``warn`` and ``error`` are true, ``error`` wins.
    error: bool = False

    #: Diagnostic tolerance for warn/error checks.
    tol: float = 1e-6

    def __post_init__(self) -> None:
        if self.low is None and self.high is None:
            raise ValueError("RangeSpec requires at least one bound")

        if (
            self.low is not None
            and self.high is not None
            and self.low > self.high
        ):
            raise ValueError(
                f"RangeSpec low must be <= high, got {self.low} > {self.high}"
            )

    def violates(self, t: Tensor) -> bool:
        """Debug-only violation check. Expensive; not for hot paths."""
        if self.low is not None:
            if bool((t < self.low - self.tol).any()):
                return True

        if self.high is not None:
            if bool((t > self.high + self.tol).any()):
                return True

        return False

    def describe(self) -> str:
        low = "-inf" if self.low is None else str(self.low)
        high = "inf" if self.high is None else str(self.high)
        return f"[{low}, {high}]"


@dataclass(frozen=True)
class StateSpec:
    """Formal description of one state tensor in a neuron's step.

    ``name`` must match the state attribute and appear first in the ``spk``
    ordering used by ``_step``/``_step_forward``. ``reset_value`` is the value
    used to (re)initialize the state. ``dtype`` defaults to following the
    input dtype. ``shape`` defaults to ``"input"``, meaning the state mirrors
    the input tensor's shape. A concrete tuple pins that exact geometry
    (validated in explicit mode, used for hidden allocation); ``None``
    disables shape validation (hidden allocation then mirrors the input).
    ``differentiable`` marks whether the state participates in autograd
    (spike tensors are typically not).

    ``value_range`` declares an inclusive ``(lo, hi)`` bound for the state
    (see ``ValueRange``). ``soft_range=True`` marks the bound as soft: values
    may exceed it and the module never acts on them. Prefer the ``range``
    field (a ``RangeSpec``) for full control over clamping and diagnostics.
    """

    name: str
    reset_value: float
    dtype: Optional[torch.dtype] = None
    shape: Literal["input"] | tuple[int, ...] | None = "input"
    differentiable: bool = True
    value_range: Optional[ValueRange] = None
    soft_range: bool = False

    #: Preferred range declaration. If set, this wins over the legacy
    #: ``value_range`` / ``soft_range`` fields.
    range: Optional[RangeSpec] = None

    #: Optional sugar for constructing a RangeSpec from legacy tuple ranges.
    #: These are ignored when ``range`` is provided.
    range_clamp: Optional[bool] = None
    range_warn: Optional[bool] = None
    range_error: Optional[bool] = None
    range_tol: Optional[float] = None


@dataclass(frozen=True)
class SequenceOutput:
    """Structured output for sequence-oriented forward calls.

    ``spikes`` is always present. ``final_state`` and ``states`` are only
    populated in explicit mode (and ``states`` only when intermediate states
    are requested).
    """

    spikes: Tensor
    final_state: Optional[tuple[Tensor, ...]] = None
    states: Optional[list[tuple[Tensor, ...]]] = None


def subtract_reset(mem: Tensor, spk: Tensor, threshold: Tensor) -> Tensor:
    """Reset by subtracting the threshold from fired neurons."""
    return torch.addcmul(mem, spk, threshold, value=-1.0)


def zero_reset(mem: Tensor, spk: Tensor, threshold: Tensor) -> Tensor:
    """Reset fired neurons to zero."""
    return mem * (1 - spk)


def hard_zero_reset(mem: Tensor, spk: Tensor, threshold: Tensor) -> Tensor:
    """Hard-reset fired neurons to zero using a mask."""
    return mem.masked_fill(spk > 0, 0.0)


def no_reset(mem: Tensor, spk: Tensor, threshold: Tensor) -> Tensor:
    """Keep the membrane potential untouched."""
    return mem


def identity(t: Tensor) -> Tensor:
    """Return the tensor unchanged (no constraint)."""
    return t


def clamp_unit_interval(t: Tensor) -> Tensor:
    """Clamp to the stable ``[0, 1]`` leak range (e.g. LIF/SRM beta)."""
    return torch.clamp(t, 0.0, 1.0)


def clamp_positive(t: Tensor) -> Tensor:
    """Clamp to stay strictly positive (e.g. thresholds, QIF/AdEx beta)."""
    return torch.clamp(t, min=1e-6)


def _clamp_low_high(t: Tensor, low: float, high: float) -> Tensor:
    return torch.clamp(t, low, high)


def _clamp_low(t: Tensor, low: float) -> Tensor:
    return torch.clamp_min(t, low)


def _clamp_high(t: Tensor, high: float) -> Tensor:
    return torch.clamp_max(t, high)


def _make_range_clamp(rng: Optional[RangeSpec]) -> TensorConstraint:
    """Build a picklable clamp constraint from a RangeSpec.

    Clamps are built from module-level helpers and ``functools.partial`` so
    they survive serialization (no closures stored on the module).
    """
    if rng is None or not rng.clamp:
        return identity

    low = rng.low
    high = rng.high

    if low is None and high is None:
        return identity

    if low is None:
        assert high is not None
        return partial(_clamp_high, high=high)

    if high is None:
        assert low is not None
        return partial(_clamp_low, low=low)

    return partial(_clamp_low_high, low=low, high=high)


def _floating_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return a floating dtype.

    State tensors are generally continuous dynamical variables. If the input is
    an integer tensor, default to the global default float dtype instead of
    creating integer state buffers.
    """
    return dtype if dtype.is_floating_point else torch.get_default_dtype()


class SpikingModule(nn.Module):
    """Base class for every spiking neuron layer.

    Subclasses ``nn.Module`` so a single global change here propagates
    to every neuron type. Picks the state-tracking (hidden) or stateless
    (explicit) forward path once at construction, so ``forward`` never
    branches at runtime.

    Subclasses declare their state via ``_get_state_specs()``: a tuple of
    ``StateSpec`` (spk first, matching the order ``_step`` returns). Hidden
    state lives in non-persistent buffers so it
    follows ``.to()``/``.float()``/``.half()`` like normal PyTorch state, and
    is regenerated with a fresh buffer tensor each step so gradient flow
    through time is preserved. State is never
    silently reallocated to a new geometry -- a shape, device, or dtype change
    raises.

    Hidden buffers are non-persistent, so ``state_dict()`` omits them by
    default; ``get_extra_state``/``set_extra_state`` are overridden to carry
    the buffers through ``torch.save``/``load`` so a stateful model round-trips
    mid-sequence.

    ``_step(x, *state) -> (spk, *next_state)`` (typed ``StepFn``) is a pure
    function of its arguments: no reads or writes of module buffers or
    parameters beyond the neuron's own hyperparameters/constraints, no
    in-place mutation of ``*state``. ``spk`` must be returned first, and the
    remaining outputs follow the ``StateSpec`` order. Each neuron picks its
    own arity; Pyright checks that its ``_step`` returns a tuple of tensors.
    Hidden-state side effects belong exclusively in ``_step_forward`` / the
    hidden forward path.

    ``spk`` is stored as a hidden buffer too (the first ``StateSpec``), purely
    for uniformity with the ``_step`` output tuple: ``self.spk`` holds the
    layer's last spike output, but it is never fed back into ``_step`` (only
    the recurrent states after ``spk`` are). All subclasses must include
    ``spk`` as the first ``StateSpec`` and return it first from ``_step``.

    Inputs use a feature-last layout: the last dimension is the feature axis,
    validated against ``size`` when set. For convolutional or other non
    feature-last tensors, reshape before passing or leave ``size=None``.

    ``validate`` (default: the global ``set_validation`` toggle, on) controls
    runtime validation of step outputs and state shapes on every forward pass;
    pass ``validate=False`` to skip the checks in hot inference loops.

    Every neuron constructor forwards the shared ``size``, ``init_hidden``,
    ``validate``, and ``use_fused_sequence`` options here as
    keyword-only arguments (see ``SpikingModuleKwargs``); they take their
    semantics from this docstring rather than being re-declared on each
    neuron.

    For BPTT, benchmarking, and kernel lowering the explicit path is the
    recommended training mode (state passed in/out each step); hidden mode is
    best for quick research code, inference, and stateful rollout.
    """

    neuron_name = "SpikingModule"

    # NOTE: private contract (subclass contract methods, never public API)
    _step: StepFn

    _cached_state_specs: Optional[tuple[StateSpec, ...]] = None
    _cached_state_specs_no_spk: Optional[tuple[StateSpec, ...]] = None
    _cached_state_names: Optional[tuple[str, ...]] = None
    _cached_reset_values: Optional[tuple[float, ...]] = None
    _cached_n_state: Optional[int] = None
    _cached_n_explicit_state: Optional[int] = None
    _cached_state_clamps: Optional[tuple[TensorConstraint, ...]] = None
    _cached_state_ranges: Optional[tuple[Optional[RangeSpec], ...]] = None

    #: Selected range enforcer. Usually a bound method selected at metadata
    #: time. Kept as a callable so the hot path can call it directly.
    _range_enforcer: Optional[
        Callable[[tuple[Tensor, ...]], tuple[Tensor, ...]]
    ] = None

    #: Debug-only entries used when validate=True.
    _range_debug_entries: tuple[tuple[int, str, RangeSpec], ...] = ()

    #: Declares which learnable parameters carry a runtime constraint,
    #: mapping the parameter attribute name to the attribute name holding the
    #: user-supplied constraint callable. Subclasses override this; the base
    #: resolves it once (cached) into per-parameter effective constraints.
    _constrained_param_specs: dict[str, str] = {}

    #
    # Construction / public forward interface
    #

    def __init__(
        self,
        size: Optional[int] = None,
        init_hidden: bool = False,
        validate: Optional[bool] = None,
        use_fused_sequence: bool = False,
        pack_output: bool = False,
    ):
        """Construct a spiking layer.

        Args:
            size: number of features. ``None`` infers from the input.
            init_hidden: ``True`` for hidden-mode layers (state held as module
                buffers, sequence scans reject an explicit initial state).
            validate: override the module-level validation default (input and
                state range/shape checks).
            use_fused_sequence: route ``forward_sequence`` through
                ``_fused_forward_sequence`` when assigned.
            pack_output: bit-pack returned spikes into ``int32`` (32 per word).

                WARN: preferred in compiled mode (``fast_sequence_`` /
                ``compile_sequence_scan``): the pack then fuses into the scan
                and peak memory drops (spikes stored 32x smaller). In plain
                eager mode the ``(T, B, F)`` float stack stays live while it is
                packed, so peak memory is not reduced.
        """
        super().__init__()
        self.size = size
        self.init_hidden = init_hidden
        self._validate_override = validate
        self.use_fused_sequence = use_fused_sequence
        self.pack_output = pack_output
        self._fused_forward_sequence: Optional[FusedSequenceFn] = None
        self._range_warn_counts: dict[str, int] = {}
        if use_fused_sequence:
            # Reference implementation: the per-step scan is the shipped
            # default consumer of the hook until a kernel package overrides
            # ``_fused_forward_sequence`` with a fused kernel.
            self._fused_forward_sequence = self._reference_sequence_scan
        if size is not None:
            if not isinstance(size, int) or size < 1:
                raise ValueError(
                    f"{self.neuron_name} size must be a positive int, got {size!r}"
                )

    def forward(self, x: Tensor, *state: Tensor) -> NeuronOutput:
        """Dispatch to the hidden or explicit step path for this layer.

        The branch is a couple of constant booleans, so ``torch.compile``
        specializes on it; keeping ``forward`` a real method (rather than a
        dynamically assigned attribute) keeps the module conventional for
        subclassing, export, and tooling.
        """
        if self.init_hidden:
            return self._forward_hidden(x, *state)
        return self._forward_explicit(x, *state)

    @property
    def validate(self) -> bool:
        """Runtime validation flag.

        If the module was constructed with ``validate=None``, this follows the
        global ``set_validation`` / ``no_validation`` toggle. If an explicit
        bool was passed at construction, that value wins.
        """
        return get_validation() if self._validate_override is None else self._validate_override

    @validate.setter
    def validate(self, value: bool) -> None:
        self._validate_override = bool(value)

    def extra_repr(self) -> str:
        parts: list[str] = []
        if self.size is not None:
            parts.append(f"size={self.size}")
        parts.append(f"init_hidden={self.init_hidden}")
        if self.use_fused_sequence:
            parts.append("use_fused_sequence=True")
        return ", ".join(parts)

    #
    # Subclass contract
    #

    def _get_state_specs(self) -> tuple[StateSpec, ...]:
        """Return the formal state spec, ``spk`` first."""
        raise NotImplementedError

    def _check_state_specs(self, specs: tuple[StateSpec, ...]) -> None:
        """Validate a state spec: non-empty, unique names, ``spk`` first."""
        if not specs:
            raise ValueError(f"{self.neuron_name} declares an empty state spec")
        names = [spec.name for spec in specs]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.neuron_name} state names must be unique")
        if specs[0].name != "spk":
            raise ValueError(
                f"{self.neuron_name} state spec must list 'spk' first"
            )

    def _get_values_to_reset(self) -> dict[str, float]:
        """Map state name to its reset value (derived from the state spec)."""
        return dict(zip(self._state_names, self._current_reset_values()))

    def _current_reset_values(self) -> tuple[float, ...]:
        """Reset values at call time.

        Defaults to the static values from the state spec. Override this if a
        neuron has learnable reset-related parameters (e.g. a learnable
        ``v_rest``), so resets/initialization always track the current value.
        """
        self._ensure_state_metadata()
        assert self._cached_reset_values is not None
        return self._cached_reset_values

    def _resolved_constraints(self) -> dict[str, TensorConstraint]:
        """Resolve, once and cached, each constrained parameter's constraint.

        For every parameter declared in ``_constrained_param_specs``, pick the
        effective runtime constraint: the user's ``clamp``-style callable when
        the parameter ``requires_grad`` (so training stays in range) or
        ``identity`` when the parameter is fixed (the constraint would be dead
        work and could silently override an intentionally set value).

        ``requires_grad`` is fixed at parameter creation and never changes, so
        this decision is safe to cache forever. The cache is stored in
        ``self.__dict__`` directly (under a key distinct from the method name)
        because ``nn.Module.__setattr__`` would otherwise register the stored
        callables (they are not parameters).
        """
        resolved = self.__dict__.get("_resolved_constraint_map")
        if resolved is None:
            resolved = {}
            for param_name, constraint_attr in type(self)._constrained_param_specs.items():
                value = getattr(self, param_name)
                constraint = getattr(self, constraint_attr)
                resolved[param_name] = constraint if value.requires_grad else identity
            self.__dict__["_resolved_constraint_map"] = resolved
        return resolved

    def _constrained_params(self, *names: str) -> tuple[Tensor, ...]:
        """Apply each named parameter's effective constraint.

        ``names`` defaults to the neuron's declared constrained parameters in
        declaration order; pass explicit names to fetch a subset (QIF/AdEx/SRM
        constrain both ``beta`` and ``threshold``, Izhikevich/HH only
        ``threshold``). Fixed parameters are returned unchanged (``identity``),
        learnable ones are clamped. The learnable-vs-fixed selection is cached
        by ``_resolved_constraints``, so this runs no per-parameter branching.
        """
        resolved = self._resolved_constraints()
        if names:
            return tuple(resolved[name](getattr(self, name)) for name in names)
        decl = type(self)._constrained_param_specs
        return tuple(resolved[name](getattr(self, name)) for name in decl)

    def _resolve_range_spec(self, spec: StateSpec) -> Optional[RangeSpec]:
        """Resolve a StateSpec into an effective RangeSpec.

        Precedence:
            1. explicit ``spec.range``
            2. legacy ``value_range`` plus optional range_* sugar fields
            3. no range
        """
        if spec.range is not None:
            return spec.range

        if spec.value_range is None:
            return None

        low, high = spec.value_range
        if low is None and high is None:
            return None

        if spec.soft_range:
            return RangeSpec(
                low=low,
                high=high,
                clamp=False,
                warn=False,
                error=False,
            )

        return RangeSpec(
            low=low,
            high=high,
            clamp=True if spec.range_clamp is None else spec.range_clamp,
            warn=False if spec.range_warn is None else spec.range_warn,
            error=False if spec.range_error is None else spec.range_error,
            tol=1e-6 if spec.range_tol is None else spec.range_tol,
        )

    def _build_range_debug_entries(
        self,
        specs: tuple[StateSpec, ...],
        ranges: tuple[Optional[RangeSpec], ...],
    ) -> tuple[tuple[int, str, RangeSpec], ...]:
        active: list[tuple[int, str, RangeSpec]] = []

        for i, (spec, rng) in enumerate(zip(specs, ranges)):
            if rng is None:
                continue

            if rng.warn or rng.error:
                active.append((i, spec.name, rng))

        return tuple(active)

    def _check_reset_values_against_ranges(
        self,
        specs: tuple[StateSpec, ...],
        ranges: tuple[Optional[RangeSpec], ...],
    ) -> None:
        for spec, rng in zip(specs, ranges):
            if rng is None:
                continue

            value = spec.reset_value

            if rng.low is not None and value < rng.low - rng.tol:
                msg = (
                    f"{self.neuron_name} state '{spec.name}' reset_value "
                    f"{value} is below range lower bound {rng.low}"
                )

                if rng.error:
                    raise ValueError(msg)

                if rng.warn:
                    warnings.warn(msg, RuntimeWarning, stacklevel=3)

            if rng.high is not None and value > rng.high + rng.tol:
                msg = (
                    f"{self.neuron_name} state '{spec.name}' reset_value "
                    f"{value} is above range upper bound {rng.high}"
                )

                if rng.error:
                    raise ValueError(msg)

                if rng.warn:
                    warnings.warn(msg, RuntimeWarning, stacklevel=3)

    def _run_range_debug(self, out: tuple[Tensor, ...]) -> None:
        """Debug-only range diagnostics.

        This must never run in benchmark/compiled fast paths.
        """
        for i, name, rng in self._range_debug_entries:
            t = out[i]

            if not rng.violates(t):
                continue

            msg = (
                f"{self.neuron_name} state '{name}' violated range "
                f"{rng.describe()}"
            )

            if rng.error:
                raise ValueError(msg)

            if rng.warn:
                self._warn_range_violation(name, rng)

    def _warn_range_violation(self, name: str, rng: RangeSpec) -> None:
        count = self._range_warn_counts.get(name, 0)

        if count == 0:
            warnings.warn(
                f"{self.neuron_name} state '{name}' violated range "
                f"{rng.describe()}; clamping into range.",
                RuntimeWarning,
                stacklevel=4,
            )
        elif count == 10:
            warnings.warn(
                f"{self.neuron_name} state '{name}' has repeatedly violated "
                f"range {rng.describe()}; suppressing further warnings.",
                RuntimeWarning,
                stacklevel=4,
            )

        self._range_warn_counts[name] = count + 1

    #
    # State metadata
    #

    def _ensure_state_metadata(self) -> None:
        """Populate the cached state metadata on first use.

        The structural spec check (non-empty, unique names, ``spk`` first)
        runs exactly once and is *not* gated on ``validate``: downstream
        code assumes those invariants unconditionally. Per-forward checks
        remain gated on ``validate``.
        """
        if self._cached_state_specs is not None:
            return
        specs = self._get_state_specs()
        self._check_state_specs(specs)
        self._cached_state_specs = specs
        self._cached_state_specs_no_spk = specs[1:]
        self._cached_state_names = tuple(spec.name for spec in specs)
        self._cached_reset_values = tuple(spec.reset_value for spec in specs)
        self._cached_n_state = len(specs)
        self._cached_n_explicit_state = len(specs) - 1

        ranges = tuple(self._resolve_range_spec(spec) for spec in specs)
        self._cached_state_ranges = ranges

        self._cached_state_clamps = tuple(
            _make_range_clamp(rng) for rng in ranges
        )

        self._range_enforcer = self._select_range_enforcer(
            self._cached_state_clamps
        )
        self._range_debug_entries = self._build_range_debug_entries(specs, ranges)
        self._check_reset_values_against_ranges(specs, ranges)

    @property
    def _state_specs(self) -> tuple[StateSpec, ...]:
        """The full cached state spec, ``spk`` first."""
        self._ensure_state_metadata()
        assert self._cached_state_specs is not None
        return self._cached_state_specs

    @property
    def _state_names(self) -> tuple[str, ...]:
        """Cached state attribute names, ``spk`` first."""
        self._ensure_state_metadata()
        assert self._cached_state_names is not None
        return self._cached_state_names

    @property
    def _reset_values(self) -> tuple[float, ...]:
        """Cached reset values, aligned with ``_state_names``."""
        self._ensure_state_metadata()
        assert self._cached_reset_values is not None
        return self._cached_reset_values

    @property
    def _state_clamps(self) -> tuple[TensorConstraint, ...]:
        """Per-state clamps resolved once at metadata construction."""
        self._ensure_state_metadata()
        assert self._cached_state_clamps is not None
        return self._cached_state_clamps

    @property
    def _state_ranges(self) -> tuple[Optional[RangeSpec], ...]:
        """Per-state resolved ranges, ``spk`` first."""
        self._ensure_state_metadata()
        assert self._cached_state_ranges is not None
        return self._cached_state_ranges

    @property
    def _n_state(self) -> int:
        """Number of state tensors ``_step`` returns, including ``spk``."""
        self._ensure_state_metadata()
        assert self._cached_n_state is not None
        return self._cached_n_state

    def _check_step_output(self, out: tuple[Tensor, ...], expected: int) -> None:
        """Validate that a step produced exactly ``expected`` tensors."""
        if not isinstance(out, tuple):
            raise TypeError(
                f"{self.neuron_name} step must return a tuple of tensors, "
                f"got {type(out).__name__}"
            )
        if len(out) != expected:
            raise ValueError(
                f"{self.neuron_name} step returned {len(out)} tensor(s), "
                f"expected {expected} (spk first, then state in spec order)"
            )
        for i, t in enumerate(out):
            if not isinstance(t, torch.Tensor):
                raise TypeError(
                    f"{self.neuron_name} step output {i} is "
                    f"{type(t).__name__}, expected Tensor"
                )

    def _select_range_enforcer(
        self,
        clamps: tuple[TensorConstraint, ...],
    ) -> Optional[Callable[[tuple[Tensor, ...]], tuple[Tensor, ...]]]:
        """Select a specialized range enforcer at metadata time.

        The returned object is a bound method, not a lambda, so modules stay
        easier to serialize than if we stored closures directly.
        """
        if all(clamp is identity for clamp in clamps):
            return None

        n = len(clamps)

        if n == 2:
            c0_identity = clamps[0] is identity
            c1_identity = clamps[1] is identity

            if c0_identity and not c1_identity:
                return self._apply_ranges_2_keep0_clamp1

            if not c0_identity and c1_identity:
                return self._apply_ranges_2_clamp0_keep1

            if not c0_identity and not c1_identity:
                return self._apply_ranges_2_clamp0_clamp1

        return self._apply_ranges_generic

    def _apply_ranges_2_keep0_clamp1(
        self,
        out: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        assert self._cached_state_clamps is not None
        return (out[0], self._cached_state_clamps[1](out[1]))

    def _apply_ranges_2_clamp0_keep1(
        self,
        out: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        assert self._cached_state_clamps is not None
        return (self._cached_state_clamps[0](out[0]), out[1])

    def _apply_ranges_2_clamp0_clamp1(
        self,
        out: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        assert self._cached_state_clamps is not None
        return (
            self._cached_state_clamps[0](out[0]),
            self._cached_state_clamps[1](out[1]),
        )

    def _apply_ranges_generic(
        self,
        out: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        assert self._cached_state_clamps is not None
        return tuple(
            clamp(t) for clamp, t in zip(self._cached_state_clamps, out)
        )

    def _step_forward(self, x: Tensor) -> tuple[Tensor, ...]:
        """Feed the current buffers into the pure ``_step`` and return its output."""
        # Read directly from _buffers: nn.Module.__getattr__ would add a Python
        # dispatch hop per state per step.
        buffers = self._buffers
        state = tuple(buffers[spec.name] for spec in self._state_specs_no_spk)
        return self._step(x, *state)

    @property
    def _n_explicit_state(self) -> int:
        """Number of state tensors the explicit forward must receive."""
        self._ensure_state_metadata()
        assert self._cached_n_explicit_state is not None
        return self._cached_n_explicit_state

    #
    # Validation helpers
    #

    def _check_input(self, x: Tensor) -> None:
        """Validate an input tensor against this layer's ``size``.

        ``size`` assumes a feature-last layout: the last dimension is the
        feature axis. A 0-dim tensor has no feature axis and is rejected when
        ``size`` is set. Convolutional tensors may need reshaping, or use
        ``size=None`` to skip the check.
        """
        if x.dim() == 0:
            if self.size is not None:
                raise ValueError(
                    f"{self.neuron_name} got a 0-dim input but size={self.size}; "
                    f"inputs must have a feature-last layout"
                )
            return
        if self.size is not None and x.shape[-1] != self.size:
            raise ValueError(
                f"{self.neuron_name} got {x.shape[-1]} features but size={self.size}. "
                "Inputs are assumed to be feature-last; reshape the input or use size=None."
            )

    def _check_explicit(self, x: Tensor, *state: Tensor) -> None:
        self._check_input(x)
        if len(state) != self._n_explicit_state:
            raise ValueError(
                f"{self.neuron_name} explicit forward expects "
                f"{self._n_explicit_state} state tensor(s), got {len(state)}"
            )

    def _check_sequence_input(self, x_seq: Tensor, caller: str) -> None:
        if x_seq.dim() < 3:
            raise ValueError(
                f"{self.neuron_name} {caller} expects a tensor with "
                f"at least 3 dims (time, batch, features), got {x_seq.dim()}"
            )
        if x_seq.shape[0] == 0:
            raise ValueError(
                f"{self.neuron_name} {caller} expects at least one "
                f"timestep, got {x_seq.shape[0]}"
            )
        if self.validate:
            self._check_input(x_seq)

    #
    # Hidden-mode helpers
    #

    def _prepare_hidden(self, x: Tensor) -> None:
        if self.validate:
            self._check_input(x)

        if self._needs_alloc(x):
            self._alloc_state(x)

    def _store_hidden_outputs(self, out: tuple[Tensor, ...]) -> None:
        buffers = self._buffers
        for spec, t in zip(self._state_specs, out):
            if not spec.differentiable:
                t = t.detach()
            # Buffers were registered by _alloc_state before the step, so the
            # names are already in self._buffers. Write the entry directly:
            # nn.Module.__setattr__ would route the assignment through
            # register_buffer() every step (module.py), which is pure per-step
            # Python overhead and re-runs the registration hooks. Attribute
            # reads already resolve through nn.Module.__getattr__ -> self._buffers,
            # so a plain dict write is fully equivalent.
            buffers[spec.name] = t

    def _hidden_step(self, x: Tensor) -> Tensor:
        """Run one hidden step assuming buffers are already allocated.

        Shared by ``_forward_hidden`` and the hidden sequence scan, which
        hoists ``_prepare_hidden`` out of the per-step loop.
        """
        out = self._step_forward(x)

        if self.validate:
            self._check_step_output(out, self._n_state)

            if self._range_debug_entries:
                self._run_range_debug(out)

        if self._range_enforcer is not None:
            out = self._range_enforcer(out)

        self._store_hidden_outputs(out)
        return out[0]

    def _forward_hidden(self, x: Tensor, *state: Tensor) -> Tensor:
        if state:
            raise ValueError(
                f"{self.neuron_name} hidden forward takes no state, got {len(state)}"
            )
        self._prepare_hidden(x)
        return self._hidden_step(x)

    #
    # Explicit-mode helpers
    #

    def _prepare_explicit_sequence_state(
        self,
        x_seq: Tensor,
        state: Optional[tuple[Tensor, ...]],
    ) -> tuple[Tensor, ...]:
        if state is None:
            state = self.initial_state(
                tuple(x_seq.shape[1:]),
                device=x_seq.device,
                dtype=x_seq.dtype,
            )

        if self.validate:
            self._check_explicit(x_seq[0], *state)
            self._check_state_shapes(x_seq[0], *state)

        return state

    def _forward_explicit(self, x: Tensor, *state: Tensor) -> tuple[Tensor, ...]:
        self._ensure_state_metadata()

        if self.validate:
            self._check_explicit(x, *state)
            self._check_state_shapes(x, *state)
        out = self._step(x, *state)

        if self.validate:
            self._check_step_output(out, self._n_state)

            if self._range_debug_entries:
                self._run_range_debug(out)

        if self._range_enforcer is not None:
            out = self._range_enforcer(out)

        return out

    #
    # Sequence scans
    #

    def forward_sequence(
        self,
        x_seq: Tensor,
        state: Optional[tuple[Tensor, ...]] = None,
    ) -> Tensor | tuple[Tensor, ...]:
        """Evolve a sequence with shape ``(time, batch, features)``.

        In hidden mode returns spikes shaped ``(time, batch, features)`` and
        rejects ``state``. In explicit mode ``state`` is the initial state
        tuple (``None`` means a fresh ``initial_state``) and
        ``(spikes, *final_state)`` is returned.

        If the module was constructed with ``pack_output=True``, the returned
        spikes are bit-packed ``int32`` (32 per word, see :func:`pack_spikes`);
        the module state stays float.

        WARN: ``pack_output`` is preferred in compiled mode (``fast_sequence_``
        / ``compile_sequence_scan``). The scan packs the stacked ``(T, B, F)``
        output once; under ``torch.compile`` that pack fuses into the scan and
        the float stack never materializes, giving the memory win. In plain
        eager mode the float stack is held while it is packed, so peak memory
        is not reduced and may exceed the unpacked scan.

        If ``use_fused_sequence`` is set and ``_fused_forward_sequence`` has
        been assigned (e.g. by a kernel package, or by a neuron shipping its
        own reference implementation), that implementation is used instead of
        the default per-step scan (``_reference_sequence_scan``).
        """
        if self.use_fused_sequence and self._fused_forward_sequence is not None:
            return self._fused_forward_sequence(x_seq, state)
        return self._reference_sequence_scan(x_seq, state)

    def _reference_sequence_scan(
        self,
        x_seq: Tensor,
        state: Optional[tuple[Tensor, ...]] = None,
    ) -> Tensor | tuple[Tensor, ...]:
        """Per-step sequence scan: the reference ``FusedSequenceFn``.

        Same contract as ``forward_sequence`` (handles hidden vs. explicit
        mode, ``state`` rejection/initialization, and validation). This is the
        default loop a fused implementation must reproduce numerically; it is
        also what ``forward_sequence`` falls back to when no fused
        implementation is wired up.

        With ``pack_output`` set at construction, spikes are bit-packed into
        ``int32`` (32 per word) once on the stacked output in both modes. The
        module state stays float and the returned tensor carries packed spikes.
        See :func:`pack_spikes`.
        """
        self._check_sequence_input(x_seq, "forward_sequence")

        if self.init_hidden:
            if state is not None:
                raise ValueError(
                    f"{self.neuron_name} hidden forward_sequence takes no "
                    f"initial state; it evolves the module buffers"
                )
            return self._hidden_sequence_scan(x_seq)

        return self._reference_explicit_sequence_scan(x_seq, state)

    def _hidden_sequence_scan(self, x_seq: Tensor) -> Tensor:
        """Hidden-mode sequence scan, optionally bit-packing the spike output."""
        if torch.is_grad_enabled() and self.training:
            if not getattr(self, "_warned_hidden_sequence_train", False):
                warnings.warn(
                    f"{self.neuron_name} hidden-mode forward_sequence with gradients is "
                    "usually slower and harder to train; use init_hidden=False and "
                    "step_state()/forward_sequence_output() for training.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_hidden_sequence_train = True
        # Allocate/validate once up front instead of re-running _needs_alloc
        # every step: every row of x_seq shares one shape, so the per-step
        # check is constant work.
        self._prepare_hidden(x_seq[0])

        # Write each step's spikes into a preallocated (T, B, F) output with a
        # constant-index index_copy_ instead of collecting a list and calling
        # torch.stack. The list holds every per-step (B, F) tensor alive while
        # stack copies it, adding a full (T, B, F) of transient peak memory
        # (~12.5 MiB here); preallocating keeps peak at input + output. Under
        # torch.compile, constant-index index_copy_ lowers to a plain
        # contiguous store fused into that step's kernel, so the compiled path
        # keeps the same fusion/runtime as the old torch.stack version. In
        # plain eager the per-step scatter launches are the slow spot, so eager
        # batches K steps into a single index_copy_ (see _SEQUENCE_SCAN_CHUNK):
        # ~T/K launches instead of T, matching torch.stack's eager speed while
        # keeping peak at input + output plus only a K-step transient.
        spikes = torch.empty_like(x_seq)
        if torch.compiler.is_compiling():
            for t, x_t in enumerate(x_seq):
                spikes.index_copy_(
                    0,
                    torch.tensor([t], device=x_seq.device),
                    self._hidden_step(x_t).unsqueeze(0),
                )
        else:
            K = _SEQUENCE_SCAN_CHUNK
            idx = torch.arange(x_seq.shape[0], device=x_seq.device)
            for lo in range(0, x_seq.shape[0], K):
                hi = min(lo + K, x_seq.shape[0])
                chunk = [self._hidden_step(x_t) for x_t in x_seq[lo:hi]]
                spikes.index_copy_(0, idx[lo:hi], torch.stack(chunk))

        if self.pack_output:
            # Pack once after the scan instead of per step. Per-step packing
            # injects ~T reduction ops into the fused graph (one per timestep),
            # which makes torch.compile's O(N^2) fusion scheduler / codegen
            # explode on long sequences. The float scan fuses fast; pack once on
            # the stacked (T, B, F) output for the same compressed result.
            # WARN: this boundary pack only delivers its memory win under
            # torch.compile (the pack fuses and the float stack is not held);
            # in eager the float stack stays live while it is packed, so peak
            # memory is not reduced.
            return pack_spikes(spikes)

        return spikes

    def _reference_explicit_sequence_scan(
        self,
        x_seq: Tensor,
        state: Optional[tuple[Tensor, ...]],
    ) -> tuple[Tensor, ...]:
        state = self._prepare_explicit_sequence_state(x_seq, state)

        # Same preallocated-output collection as _hidden_sequence_scan: write
        # each step into a (T, B, F) buffer via constant-index index_copy_ so
        # eager peak stays at input + output (no (B, F) list held for stack)
        # while torch.compile fuses the store into each step's kernel. The
        # buffer dtype follows the step output (state may carry a different
        # dtype than x_seq, and torch.stack preserved that). In eager, K steps
        # are batched into one index_copy_ (see _SEQUENCE_SCAN_CHUNK) to match
        # torch.stack's speed without its (T, B, F) list + copy peak.
        first = self.forward(x_seq[0], *state)
        assert isinstance(first, tuple)
        spikes = torch.empty(
            (x_seq.shape[0], *first[0].shape),
            dtype=first[0].dtype,
            device=first[0].device,
        )
        spikes.index_copy_(
            0, torch.tensor([0], device=first[0].device), first[0].unsqueeze(0)
        )
        cur = first[1:]
        if torch.compiler.is_compiling():
            for t in range(1, x_seq.shape[0]):
                out = self.forward(x_seq[t], *cur)
                assert isinstance(out, tuple)
                spikes.index_copy_(
                    0, torch.tensor([t], device=out[0].device), out[0].unsqueeze(0)
                )
                cur = out[1:]
        else:
            K = _SEQUENCE_SCAN_CHUNK
            idx = torch.arange(x_seq.shape[0], device=spikes.device)
            for lo in range(1, x_seq.shape[0], K):
                hi = min(lo + K, x_seq.shape[0])
                chunk = []
                for t in range(lo, hi):
                    out = self.forward(x_seq[t], *cur)
                    assert isinstance(out, tuple)
                    chunk.append(out[0])
                    cur = out[1:]
                spikes.index_copy_(0, idx[lo:hi], torch.stack(chunk))

        if self.pack_output:
            # Pack once after the scan instead of per step, mirroring
            # _hidden_sequence_scan: per-step packing injects ~T reduction ops
            # into the fused graph (one per timestep), which makes
            # torch.compile's O(N^2) fusion scheduler / codegen explode on long
            # sequences. The float scan fuses fast; pack once on the stacked
            # (T, B, F) output for the same compressed result.
            # WARN: see the note at _hidden_sequence_scan -- the memory win
            # shows up under torch.compile, not in plain eager mode.
            spikes = pack_spikes(spikes)

        return (spikes, *cur)

    def compile_sequence_scan(self, **kwargs) -> None:
        """Compile the reference sequence scan and route through the fused hook.

        This is a convenience wrapper for research prototyping: ``torch.compile``
        on the existing per-step scan. It does not replace custom fused kernels;
        it simply makes ``torch.compile`` easy to try on the default path.

        Output tensors are cloned before returning so ``mode="reduce-overhead"``
        (CUDA graphs) is safe across repeated calls: without the clone, a
        subsequent graph run overwrites the previously returned tensor.
        """
        self._ensure_state_metadata()

        compiled = torch.compile(self._reference_sequence_scan, **kwargs)

        def _fused(x_seq: Tensor, state: Optional[tuple[Tensor, ...]] = None):
            if self.init_hidden and x_seq.shape[0] > 0:
                # Allocate the hidden buffers *before* the first compiled call.
                # If the initial trace runs _alloc_state, the buffer metadata
                # reads (e.g. QIF's float(self.v_rest.detach())) force a
                # ``Tensor.item()`` graph break and the scan silently falls
                # back to per-step eager execution. Pre-allocating keeps the
                # traced ``_needs_alloc`` path constant-False so the scan
                # fuses into a single graph.
                self._prepare_hidden(x_seq[0])
            out = compiled(x_seq, state)
            if isinstance(out, Tensor):
                return out.clone()
            return tuple(t.clone() for t in out)

        self._fused_forward_sequence = _fused
        self.use_fused_sequence = True

    def fast_sequence_(self, compile_scan: bool = True, **compile_kwargs) -> "SpikingModule":
        """Enable a fast research path: validation off + optional compiled scan.

        This is intended for prototyping and benchmarking, not as a strict
        production guarantee. It keeps explicit-mode semantics while reducing
        Python-level overhead where possible.

        ``mode="default"`` is always used. ``reduce-overhead`` (CUDA graphs)
        is avoided: it is significantly slower to (re)compile and gains little
        at the short sequence lengths typical of prototyping, and it is
        incompatible with hidden-mode state registration anyway.
        """
        self.validate = False
        if compile_scan:
            compile_kwargs.setdefault("mode", "default")
            self.compile_sequence_scan(**compile_kwargs)
        return self

    def forward_sequence_with_states(
        self,
        x_seq: Tensor,
        state: Optional[tuple[Tensor, ...]] = None,
        *,
        store_states: bool = True,
        state_callback: Optional[Callable[[int, tuple[Tensor, ...]], None]] = None,
    ) -> tuple[Tensor, tuple[Tensor, ...], Optional[list[tuple[Tensor, ...]]]]:
        """Explicit-mode sequence scan returning all intermediate states.

        Returns:
            spikes: shape ``(time, batch, features)``.
            final_state: the state after the last timestep.
            all_states: length-``time`` list when ``store_states=True``. If
                ``store_states=False``, returns ``None``. Use ``state_callback``
                to consume states without storing them (avoids memory blowup on
                long sequences).

        This is primarily intended for training methods that need per-step
        membrane potentials or other neuron states (e-prop, eligibility
        traces, membrane regularization, oline RL losses). Hidden mode is not
        supported: pass state explicitly for training.
        """
        if self.init_hidden:
            raise ValueError(
                f"{self.neuron_name} forward_sequence_with_states requires "
                f"init_hidden=False; use explicit state passing for training"
            )
        self._check_sequence_input(x_seq, "forward_sequence_with_states")
        state = self._prepare_explicit_sequence_state(x_seq, state)

        spike_list: list[Tensor] = []
        state_list: Optional[list[tuple[Tensor, ...]]] = [] if store_states else None
        cur = state
        for t in range(x_seq.shape[0]):
            out = self.forward(x_seq[t], *cur)
            assert isinstance(out, tuple)
            spike_list.append(out[0])
            cur = out[1:]

            if state_callback is not None:
                state_callback(t, cur)

            if state_list is not None:
                state_list.append(cur)

        return torch.stack(spike_list), cur, state_list

    def step_state(
        self,
        x: Tensor,
        state: tuple[Tensor, ...],
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Explicit step with state passed as a tuple.

        Returns ``(spk, next_state)``. This is a convenience wrapper around
        the varargs explicit forward path, useful for generic trainers,
        truncated BPTT, replay buffers, and state-dict-style loops.
        """
        out = self.forward(x, *state)
        assert isinstance(out, tuple)
        return out[0], out[1:]

    def step(
        self,
        x: Tensor,
        state: tuple[Tensor, ...],
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Alias of ``step_state`` for trainer ergonomics."""
        return self.step_state(x, state)

    def forward_sequence_output(
        self,
        x_seq: Tensor,
        state: Optional[tuple[Tensor, ...]] = None,
        *,
        return_states: bool = False,
    ) -> SequenceOutput:
        """Structured sequence forward.

        Hidden mode returns only spikes. Explicit mode returns spikes plus
        final state. With ``return_states=True``, explicit mode returns all
        intermediate states.
        """
        if self.init_hidden:
            spikes = self.forward_sequence(x_seq)
            assert isinstance(spikes, Tensor)
            return SequenceOutput(spikes=spikes)

        if return_states:
            spikes, final_state, states = self.forward_sequence_with_states(
                x_seq,
                state,
                store_states=True,
            )
            return SequenceOutput(spikes=spikes, final_state=final_state, states=states)

        out = self.forward_sequence(x_seq, state)
        assert isinstance(out, tuple)
        spikes = out[0]
        final_state = out[1:]
        return SequenceOutput(spikes=spikes, final_state=final_state)

    #
    # State shape / dtype helpers
    #

    def _check_state_shapes(self, x: Tensor, *state: Tensor) -> None:
        """Fail loudly if any state tensor does not match its spec shape.

        ``shape`` defaults to ``"input"`` (mirror the input's shape); a
        concrete tuple validates against that exact geometry, and ``None``
        disables the check for that state.
        """
        for spec, t in zip(self._state_specs_no_spk, state):
            if spec.shape is not None:
                expected = x.shape if spec.shape == "input" else spec.shape
                if t.shape != expected:
                    raise ValueError(
                        f"{self.neuron_name} state '{spec.name}' has shape "
                        f"{tuple(t.shape)} but expected {tuple(expected)}"
                    )

            expected_dtype = self._state_dtype(spec, x)
            if t.device != x.device or t.dtype != expected_dtype:
                raise ValueError(
                    f"{self.neuron_name} state '{spec.name}' has "
                    f"device/dtype {t.device}/{t.dtype} but expected "
                    f"{x.device}/{expected_dtype}"
                )

    def _state_dtype(self, spec: StateSpec, x: Tensor) -> torch.dtype:
        """Dtype for a hidden state buffer associated with input ``x``."""
        return _floating_dtype(spec.dtype or x.dtype)

    def _explicit_state_dtype(
        self,
        spec: StateSpec,
        dtype: Optional[torch.dtype],
    ) -> Optional[torch.dtype]:
        """Dtype for explicit-state creation.

        If the resolved dtype is non-floating, promote to the default float
        dtype. ``None`` is preserved so PyTorch can use its default dtype.
        """
        chosen = spec.dtype or dtype
        if chosen is None:
            return None
        return _floating_dtype(chosen)

    def _spec_shape(self, spec: StateSpec, x: Tensor):
        """Hidden-buffer shape for ``spec`` given input ``x``."""
        if spec.shape is None or spec.shape == "input":
            return x.shape
        return spec.shape

    #
    # Hidden buffer allocation / reset / checkpointing
    #

    def _needs_alloc(self, x: Tensor) -> bool:
        """True when hidden state must be allocated for the first time.

        Raises instead of silently reallocating when the input geometry,
        device, or dtype changes after the state already exists.
        """
        for spec in self._state_specs:
            t = getattr(self, spec.name, None)
            if t is None:
                return True
            expected_dtype = self._state_dtype(spec, x)
            expected_shape = self._spec_shape(spec, x)
            if (t.shape, t.device, t.dtype) != (
                expected_shape,
                x.device,
                expected_dtype,
            ):
                raise ValueError(
                    f"{self.neuron_name} hidden {spec.name} has "
                    f"{tuple(t.shape)}/{t.device}/{t.dtype} but input is "
                    f"{tuple(x.shape)}/{x.device}/{x.dtype}; feed inputs with a "
                    f"consistent shape or pass state explicitly with "
                    f"init_hidden=False"
                )
        return False

    def _alloc_state(self, x: Tensor) -> None:
        reset_values = dict(zip(self._state_names, self._current_reset_values()))
        for spec in self._state_specs:
            self.register_buffer(
                spec.name,
                torch.full(
                    self._spec_shape(spec, x),
                    reset_values[spec.name],
                    device=x.device,
                    dtype=self._state_dtype(spec, x),
                ),
                persistent=False,
            )

    def reset(self) -> None:
        if not self.init_hidden:
            return
        for name, reset_value in zip(self._state_names, self._current_reset_values()):
            t = getattr(self, name, None)
            if t is not None:
                setattr(self, name, torch.full_like(t, reset_value))

    def detach(self) -> None:
        if not self.init_hidden:
            return
        for name in self._state_names:
            t = getattr(self, name, None)
            if t is not None:
                setattr(self, name, t.detach())

    def get_extra_state(self):
        """Snapshot hidden buffers for ``state_dict()``/``torch.save``.

        Hidden buffers are registered ``persistent=False`` so they are
        normally absent from ``state_dict()``; this captures them so a
        mid-sequence stateful model round-trips through save/load. Returns
        ``None`` in explicit mode (nothing to checkpoint).
        """
        if not self.init_hidden:
            return None

        return {
            name: getattr(self, name).detach()
            for name in self._state_names
            if getattr(self, name, None) is not None
        }

    def set_extra_state(self, state) -> None:
        """Restore hidden buffers captured by ``get_extra_state``."""
        if not self.init_hidden or state is None:
            return
        for name, t in state.items():
            if getattr(self, name, None) is None:
                self.register_buffer(name, t, persistent=False)
            else:
                setattr(self, name, t)

    #
    # State factories / state utilities
    #

    @property
    def _state_specs_no_spk(self) -> tuple[StateSpec, ...]:
        """State specs for the explicit-state tuple (everything but ``spk``)."""
        self._ensure_state_metadata()
        assert self._cached_state_specs_no_spk is not None
        return self._cached_state_specs_no_spk

    def zero_state(
        self,
        batch_shape: tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[Tensor, ...]:
        """Create a zeroed explicit-state tuple (excluding ``spk``).

        ``batch_shape`` is the input shape the state mirrors; each returned
        tensor has that shape. ``dtype`` defaults to each spec's dtype, then
        to ``None`` (PyTorch's default float32) -- it does NOT follow the
        module's dtype. If the module uses a non-default dtype (e.g.
        ``.half()`` or ``init_hidden=True`` buffers at half precision), pass
        ``dtype`` explicitly or mirror an existing buffer. This is a
        literal-zero initializer; for the neuron's canonical initial state
        (its reset values) use ``initial_state``.
        """
        self._ensure_state_metadata()
        return tuple(
            torch.zeros(
                batch_shape,
                device=device,
                dtype=self._explicit_state_dtype(spec, dtype),
            )
            for spec in self._state_specs_no_spk
        )

    @property
    def module_dtype(self) -> torch.dtype:
        """Best-effort dtype of the module's parameters."""
        for p in self.parameters():
            return p.dtype
        return torch.get_default_dtype()

    def initial_state_for_module(
        self,
        batch_shape: tuple[int, ...],
        device: Optional[torch.device] = None,
    ) -> tuple[Tensor, ...]:
        """Create canonical initial state using the module's parameter dtype."""
        return self.initial_state(batch_shape, device=device, dtype=self.module_dtype)

    def initial_state(
        self,
        batch_shape: tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[Tensor, ...]:
        """Create the canonical fresh explicit-state tuple.

        Each tensor holds its spec's ``reset_value`` (e.g. Izhikevich ``mem``
        at ``c``, ``u`` at ``u_init``), matching what the hidden forward path
        and ``reset()`` produce. ``dtype`` follows the same rules as
        ``zero_state``: it defaults to each spec's dtype, then to float32 --
        NOT the module's dtype -- so pass it explicitly for non-float32
        modules. Use ``zero_state`` for literal zeros.
        """
        self._ensure_state_metadata()
        reset_values = self._current_reset_values()
        return tuple(
            torch.full(
                batch_shape,
                reset_value,
                device=device,
                dtype=self._explicit_state_dtype(spec, dtype),
            )
            for spec, reset_value in zip(self._state_specs_no_spk, reset_values[1:])
        )

    def initial_state_like(
        self,
        x: Tensor,
        batch_shape: Optional[tuple[int, ...]] = None,
    ) -> tuple[Tensor, ...]:
        """Create canonical initial state matching ``x``'s device/dtype.

        If ``batch_shape`` is omitted, ``x.shape`` is used. For a time-major
        sequence, pass ``batch_shape=x_seq.shape[1:]`` or use
        ``initial_state_for_sequence``.
        """
        shape = tuple(x.shape) if batch_shape is None else tuple(batch_shape)
        return self.initial_state(shape, device=x.device, dtype=x.dtype)

    def initial_state_for_sequence(self, x_seq: Tensor) -> tuple[Tensor, ...]:
        """Create initial state for a time-major ``(time, batch, features)``."""
        if x_seq.dim() < 3:
            raise ValueError(
                f"{self.neuron_name} initial_state_for_sequence expects a tensor "
                f"with at least 3 dims (time, batch, features), got {x_seq.dim()}"
            )
        return self.initial_state(
            tuple(x_seq.shape[1:]),
            device=x_seq.device,
            dtype=x_seq.dtype,
        )

    def zero_state_like(
        self,
        x: Tensor,
        batch_shape: Optional[tuple[int, ...]] = None,
    ) -> tuple[Tensor, ...]:
        """Create zeroed explicit state matching ``x``'s device/dtype."""
        shape = tuple(x.shape) if batch_shape is None else tuple(batch_shape)
        return self.zero_state(shape, device=x.device, dtype=x.dtype)

    def reset_state(self, state: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """Return a fresh explicit-state tuple at each spec's reset value."""
        self._ensure_state_metadata()
        specs = self._state_specs_no_spk
        if len(state) != len(specs):
            raise ValueError(
                f"{self.neuron_name} reset_state expects {len(specs)} state "
                f"tensor(s), got {len(state)}"
            )
        reset_values = self._current_reset_values()
        return tuple(
            torch.full_like(t, reset_value)
            for t, spec, reset_value in zip(state, specs, reset_values[1:])
        )

    def detach_state(self, state: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """Detach every tensor in an explicit-state tuple from autograd."""
        return tuple(t.detach() for t in state)

    def state_to_device(
        self,
        state: tuple[Tensor, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[Tensor, ...]:
        """Move/cast every tensor in an explicit-state tuple."""
        return tuple(t.to(device=device, dtype=dtype) for t in state)