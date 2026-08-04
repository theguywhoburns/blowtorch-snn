import torch
import torch.nn as nn
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Callable, Literal, Optional, TypedDict, Unpack

Tensor = torch.Tensor

NeuronOutput = Tensor | tuple[Tensor, ...]

ResetFn = Callable[[Tensor, Tensor, Tensor], Tensor]

TensorConstraint = Callable[[Tensor], Tensor]

Forward = Callable[..., NeuronOutput]

StepFn = Callable[..., tuple[Tensor, ...]]
"""See ``SpikingModule``: ``_step(x, *state) -> (spk, *next_state)``."""

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
    picks the state-tracking (hidden) vs. explicit path, ``preallocated``
    enables the inference-only in-place buffer path (requires
    ``init_hidden=True``), ``validate`` overrides the global validation
    toggle per instance, and ``use_fused_sequence`` routes
    ``forward_sequence`` through ``_fused_forward_sequence`` (falling back to
    the reference per-step scan). All are optional and keyword-only on the
    neuron constructors. See the ``SpikingModule`` docstring for full
    semantics.
    """

    size: Optional[int]
    init_hidden: bool
    preallocated: bool
    validate: Optional[bool]
    use_fused_sequence: bool


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
    """

    name: str
    reset_value: float
    dtype: Optional[torch.dtype] = None
    shape: Literal["input"] | tuple[int, ...] | None = "input"
    differentiable: bool = True


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
    return mem - spk * threshold


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
    through time is preserved. Set ``preallocated=True`` to reuse buffers in
    place instead (inference only, no gradient through time). State is never
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
    ``preallocated``, ``validate``, and ``use_fused_sequence`` options here as
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
    _cached_state_names: Optional[tuple[str, ...]] = None
    _cached_reset_values: Optional[tuple[float, ...]] = None
    _cached_n_state: Optional[int] = None
    _cached_n_explicit_state: Optional[int] = None

    #
    # Construction / public forward interface
    #

    def __init__(
        self,
        size: Optional[int] = None,
        init_hidden: bool = False,
        preallocated: bool = False,
        validate: Optional[bool] = None,
        use_fused_sequence: bool = False,
    ):
        super().__init__()
        if preallocated and not init_hidden:
            raise ValueError(
                f"{self.neuron_name} preallocated mode requires init_hidden=True"
            )
        self.size = size
        self.init_hidden = init_hidden
        self.preallocated = preallocated
        self._validate_override = validate
        self.use_fused_sequence = use_fused_sequence
        self._fused_forward_sequence: Optional[FusedSequenceFn] = None
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
            if self.preallocated:
                return self._forward_hidden_prealloc(x, *state)
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
        if self.init_hidden:
            parts.append(f"preallocated={self.preallocated}")
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
        self._cached_state_names = tuple(spec.name for spec in specs)
        self._cached_reset_values = tuple(spec.reset_value for spec in specs)
        self._cached_n_state = len(specs)
        self._cached_n_explicit_state = len(specs) - 1

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

    def _step_forward(self, x: Tensor) -> tuple[Tensor, ...]:
        """Feed the current buffers into the pure ``_step`` and return its output."""
        state = tuple(getattr(self, spec.name) for spec in self._state_specs_no_spk)
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
        for spec, t in zip(self._state_specs, out):
            if not spec.differentiable:
                t = t.detach()
            # Buffers were registered by _alloc_state before the step; a
            # plain attribute assignment replaces the _buffers entry without
            # register_buffer's per-step bookkeeping.
            setattr(self, spec.name, t)

    def _copy_hidden_outputs(self, out: tuple[Tensor, ...]) -> None:
        for name, t in zip(self._state_names, out):
            getattr(self, name).copy_(t.detach())

    def _forward_hidden(self, x: Tensor, *state: Tensor) -> Tensor:
        if state:
            raise ValueError(
                f"{self.neuron_name} hidden forward takes no state, got {len(state)}"
            )
        self._prepare_hidden(x)
        out = self._step_forward(x)
        if self.validate:
            self._check_step_output(out, self._n_state)
        self._store_hidden_outputs(out)
        return out[0]

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
        if self.validate:
            self._ensure_state_metadata()
            self._check_explicit(x, *state)
            self._check_state_shapes(x, *state)
        out = self._step(x, *state)
        if self.validate:
            self._check_step_output(out, self._n_state)
        return out

    def _forward_hidden_prealloc(self, x: Tensor, *state: Tensor) -> Tensor:
        """Inference path: allocate buffers once, then update them in place.

        Unlike ``_forward_hidden`` (which re-registers a fresh buffer tensor
        each step to preserve gradient flow through time), this writes each
        detached step output into the existing buffer with ``copy_``, so there
        is no per-step ``register_buffer`` bookkeeping and no autograd graph
        kept across timesteps. This trades away differentiation through time
        for inference throughput.
        """
        if self.training and torch.is_grad_enabled():
            raise RuntimeError(
                f"{self.neuron_name} preallocated mode is inference-only; "
                "disable gradients or use init_hidden=False for training"
            )
        if state:
            raise ValueError(
                f"{self.neuron_name} preallocated forward takes no state, "
                f"got {len(state)}"
            )
        self._prepare_hidden(x)
        out = self._step_forward(x)
        if self.validate:
            self._check_step_output(out, self._n_state)
        self._copy_hidden_outputs(out)
        return out[0]

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
        """
        self._check_sequence_input(x_seq, "forward_sequence")

        if self.init_hidden:
            if state is not None:
                raise ValueError(
                    f"{self.neuron_name} hidden forward_sequence takes no "
                    f"initial state; it evolves the module buffers"
                )
            return self._reference_hidden_sequence_scan(x_seq)

        return self._reference_explicit_sequence_scan(x_seq, state)

    def _reference_hidden_sequence_scan(self, x_seq: Tensor) -> Tensor:
        if torch.is_grad_enabled() and self.training and not self.preallocated:
            if not getattr(self, "_warned_hidden_sequence_train", False):
                warnings.warn(
                    f"{self.neuron_name} hidden-mode forward_sequence with gradients is "
                    "usually slower and harder to train; use init_hidden=False and "
                    "step_state()/forward_sequence_output() for training.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_hidden_sequence_train = True
        hidden_fwd = (
            self._forward_hidden_prealloc if self.preallocated else self._forward_hidden
        )
        return torch.stack([hidden_fwd(x_t) for x_t in x_seq])

    def _reference_explicit_sequence_scan(
        self,
        x_seq: Tensor,
        state: Optional[tuple[Tensor, ...]],
    ) -> tuple[Tensor, ...]:
        state = self._prepare_explicit_sequence_state(x_seq, state)

        spike_list: list[Tensor] = []
        cur = state
        for t in range(x_seq.shape[0]):
            out = self.forward(x_seq[t], *cur)
            assert isinstance(out, tuple)
            spike_list.append(out[0])
            cur = out[1:]
        return (torch.stack(spike_list), *cur)

    def compile_sequence_scan(self, **kwargs) -> None:
        """Compile the reference sequence scan and route through the fused hook.

        This is a convenience wrapper for research prototyping: ``torch.compile``
        on the existing per-step scan. It does not replace custom fused kernels;
        it simply makes ``torch.compile`` easy to try on the default path.

        Output tensors are cloned before returning so ``mode="reduce-overhead"``
        (CUDA graphs) is safe across repeated calls: without the clone, a
        subsequent graph run overwrites the previously returned tensor.
        """
        compiled = torch.compile(self._reference_sequence_scan, **kwargs)

        def _fused(x_seq: Tensor, state: Optional[tuple[Tensor, ...]] = None):
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

        ``mode="reduce-overhead"`` (CUDA graphs) is the default for explicit
        modules, which are graph-friendly. Hidden-mode scans re-register their
        state buffers every step, which is incompatible with CUDA graph
        capture, so they fall back to ``mode="default"``.
        """
        self.validate = False
        if compile_scan:
            if self.init_hidden:
                compile_kwargs.setdefault("mode", "default")
            else:
                compile_kwargs.setdefault("mode", "reduce-overhead")
            self.compile_sequence_scan(**compile_kwargs)
        return self

    def infer_sequence(self, x_seq: Tensor, *, validate: Optional[bool] = False) -> Tensor:
        """Evolve a sequence under ``torch.inference_mode()``.

        ``validate=False`` is the default for speed; pass ``validate=True`` to
        keep per-step checks if debugging. Requires ``init_hidden=True`` and
        ``preallocated=True`` (the inference-only rollout path); returns
        spikes shaped ``(time, batch, features)`` without building an
        autograd graph.
        """
        if not (self.init_hidden and self.preallocated):
            raise ValueError(
                f"{self.neuron_name} infer_sequence requires "
                f"init_hidden=True and preallocated=True"
            )
        validation_ctx = no_validation() if validate is False else nullcontext()
        with torch.inference_mode(), validation_ctx:
            if self.use_fused_sequence and self._fused_forward_sequence is not None:
                spikes = self.forward_sequence(x_seq)
                assert isinstance(spikes, Tensor)
                return spikes

            self._check_sequence_input(x_seq, "infer_sequence")

            hidden_fwd = self._forward_hidden_prealloc
            return torch.stack([hidden_fwd(x_seq[t]) for t in range(x_seq.shape[0])])

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
        return self._state_specs[1:]

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