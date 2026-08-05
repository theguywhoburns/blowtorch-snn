import torch
import pytest
from typing import cast, Optional

from blowtorch_snn import LIF, QIF, Izhikevich, AdEx, SRM, HH, SpikingModule, StateSpec

B, F = 4, 8
X = torch.randn(B, F)

NEURONS = [
    (LIF, {"beta": 0.9}),
    (QIF, {"beta": 0.01}),
    (Izhikevich, {}),
    (AdEx, {"beta": 0.9}),
    (SRM, {"beta": 0.9}),
    (HH, {}),
]


class _WrongArityHidden(SpikingModule):
    neuron_name = "_WrongArityHidden"

    def __init__(self, validate=None):
        super().__init__(size=None, init_hidden=True, validate=validate)

    def _get_state_specs(self) -> tuple[StateSpec, ...]:
        return (StateSpec("spk", 0.0), StateSpec("mem", 0.0))

    def _step(self, x, mem):
        return torch.zeros_like(x), torch.zeros_like(x)

    def _step_forward(self, x):
        return (self._step(x, self.mem)[0],)


class _WrongArityExplicit(SpikingModule):
    neuron_name = "_WrongArityExplicit"

    def __init__(self, validate=None):
        super().__init__(size=None, init_hidden=False, validate=validate)

    def _get_state_specs(self) -> tuple[StateSpec, ...]:
        return (
            StateSpec("spk", 0.0),
            StateSpec("mem", 0.0),
            StateSpec("w", 0.0),
        )

    def _step(self, x, mem, w):
        return torch.zeros_like(x), torch.zeros_like(x)



@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_step_forward_arity_matches_spec(cls, kwargs):
    neuron = cls(init_hidden=True, **kwargs)
    neuron(X)
    out = neuron._step_forward(X)
    assert len(out) == len(neuron._get_values_to_reset())
    assert isinstance(out, tuple)


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_step_is_pure(cls, kwargs):
    neuron = cls(init_hidden=False, **kwargs)
    mem0 = getattr(neuron, "mem", None)
    x = X
    state = [torch.full_like(X, 0.0) for _ in range(neuron._n_explicit_state)]
    spec_names = list(neuron._get_values_to_reset())
    for i, name in enumerate(spec_names[1:]):
        getattr(neuron, name, None)
    _ = neuron._step(x, *state)
    if mem0 is not None:
        assert mem0 is getattr(neuron, "mem", None)


def test_wrong_arity_hidden_raises():
    n = _WrongArityHidden()
    with pytest.raises(ValueError, match="expected 2"):
        n(X)


def test_wrong_arity_explicit_raises():
    n = _WrongArityExplicit()
    mem = torch.zeros(B, F)
    w = torch.zeros(B, F)
    with pytest.raises(ValueError, match="expected 3"):
        n(X, mem, w)


def test_step_pure_contract_documented():
    assert SpikingModule.__doc__ is not None
    assert "pure" in SpikingModule.__doc__


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_state_specs_spk_first_unique(cls, kwargs):
    neuron = cls(**kwargs)
    specs = neuron._get_state_specs()
    assert specs[0].name == "spk"
    names = [s.name for s in specs]
    assert len(names) == len(set(names))
    assert all(isinstance(s, StateSpec) for s in specs)
    assert all(s.shape == "input" for s in specs)
    assert specs[0].differentiable is False


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_state_specs_dtype_follows_input(cls, kwargs):
    neuron = cls(init_hidden=True, **kwargs)
    neuron(X.double())
    assert neuron.mem.dtype == torch.float64


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_derived_reset_values_match_spec(cls, kwargs):
    neuron = cls(**kwargs)
    assert neuron._get_values_to_reset() == {
        s.name: s.reset_value for s in neuron._get_state_specs()
    }


class _BadSpecFirst(SpikingModule):
    neuron_name = "_BadSpecFirst"

    def __init__(self):
        super().__init__(size=None, init_hidden=False)

    def _get_state_specs(self) -> tuple[StateSpec, ...]:
        return (StateSpec("mem", 0.0),)

    def _step(self, x, mem):
        return torch.zeros_like(x), torch.zeros_like(x)


class _BadSpecDuplicate(SpikingModule):
    neuron_name = "_BadSpecDuplicate"

    def __init__(self, validate: Optional[bool] = None):
        super().__init__(size=None, init_hidden=False, validate=validate)

    def _get_state_specs(self) -> tuple[StateSpec, ...]:
        return (StateSpec("spk", 0.0), StateSpec("mem", 0.0), StateSpec("mem", 0.0))

    def _step(self, x, mem):
        return torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)


class _BadSpecEmpty(SpikingModule):
    neuron_name = "_BadSpecEmpty"

    def __init__(self):
        super().__init__(size=None, init_hidden=False)

    def _get_state_specs(self) -> tuple[StateSpec, ...]:
        return ()

    def _step(self, x):
        return (torch.zeros_like(x),)


def test_bad_spec_spk_not_first_raises():
    n = _BadSpecFirst()
    mem = torch.zeros(B, F)
    with pytest.raises(ValueError, match="'spk' first"):
        n(X, mem)


def test_bad_spec_duplicate_raises():
    n = _BadSpecDuplicate()
    mem = torch.zeros(B, F)
    with pytest.raises(ValueError, match="unique"):
        n(X, mem, mem)


def test_bad_spec_empty_raises():
    n = _BadSpecEmpty()
    with pytest.raises(ValueError, match="empty"):
        n(X)


def test_state_spec_shape_input_means_mirror_input():
    n = _FixedShapeExplicit()
    mem = torch.zeros(B, 2 * F)
    spk, _ = n(X, mem)
    assert spk.shape == X.shape


class _FixedShapeExplicit(SpikingModule):
    neuron_name = "_FixedShapeExplicit"

    def __init__(self):
        super().__init__(size=None, init_hidden=False, validate=True)

    def _get_state_specs(self) -> tuple[StateSpec, ...]:
        return (
            StateSpec("spk", 0.0),
            StateSpec("mem", 0.0, shape=(B, 2 * F)),
        )

    def _step(self, x, mem):
        spk = torch.zeros_like(x)
        return spk, mem


def test_state_spec_shape_tuple_is_enforced():
    n = _FixedShapeExplicit()
    mem = torch.zeros(B, 2 * F)
    spk, _ = n(X, mem)
    assert spk.shape == X.shape
    with pytest.raises(ValueError, match="expected"):
        n(X, torch.zeros(B, F))


class _NoShapeCheckExplicit(SpikingModule):
    neuron_name = "_NoShapeCheckExplicit"

    def __init__(self):
        super().__init__(size=None, init_hidden=False, validate=True)

    def _get_state_specs(self) -> tuple[StateSpec, ...]:
        return (
            StateSpec("spk", 0.0),
            StateSpec("mem", 0.0, shape=None),
        )

    def _step(self, x, mem):
        spk = torch.zeros_like(x)
        return spk, mem


def test_state_spec_shape_none_disables_check():
    n = _NoShapeCheckExplicit()
    spk, _ = n(X, torch.zeros(B, F + 1))
    assert spk.shape == X.shape


def test_state_spec_non_differentiable_stored_detached():
    n = LIF(beta=0.9, init_hidden=True)
    n(X)
    assert getattr(n, "spk").requires_grad is False
    assert n.mem.requires_grad is False
    n.reset()
    assert torch.equal(getattr(n, "spk"), torch.zeros_like(n.mem))



@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_zero_state_shape_and_count(cls, kwargs):
    n = cls(**kwargs)
    state = n.zero_state((B, F))
    assert len(state) == n._n_explicit_state
    for t, spec in zip(state, n._state_specs_no_spk):
        assert t.shape == (B, F)
        assert torch.all(t == 0)


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_zero_state_dtype_follows_spec_or_arg(cls, kwargs):
    n = cls(**kwargs)
    state = n.zero_state((B, F), dtype=torch.float64)
    assert all(t.dtype == torch.float64 for t in state)


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_reset_state_returns_reset_values(cls, kwargs):
    n = cls(**kwargs)
    state = n.zero_state((B, F))
    reset = n.reset_state(state)
    for t, spec in zip(reset, n._state_specs_no_spk):
        assert torch.all(t == spec.reset_value)


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_initial_state_returns_reset_values(cls, kwargs):
    n = cls(**kwargs)
    state = n.initial_state((B, F))
    for t, spec in zip(state, n._state_specs_no_spk):
        assert torch.all(t == spec.reset_value)


def test_reset_state_arity_mismatch_raises():
    n = LIF(beta=0.9)
    with pytest.raises(ValueError, match="expects 1"):
        n.reset_state((torch.zeros(B, F), torch.zeros(B, F)))


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_detach_state_breaks_graph(cls, kwargs):
    n = cls(**kwargs)
    state = tuple(torch.randn(B, F, requires_grad=True) for _ in range(n._n_explicit_state))
    detached = n.detach_state(state)
    assert all(not t.requires_grad for t in detached)


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_state_to_device_dtype_roundtrip(cls, kwargs):
    n = cls(**kwargs)
    state = n.zero_state((B, F))
    moved = n.state_to_device(state, dtype=torch.float64)
    assert all(t.dtype == torch.float64 for t in moved)
    assert all(t.shape == (B, F) for t in moved)


T = 5
X_SEQ = torch.randn(T, B, F)


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_forward_sequence_hidden_shape(cls, kwargs):
    n = cls(init_hidden=True, **kwargs)
    spk = n.forward_sequence(X_SEQ)
    assert isinstance(spk, torch.Tensor)
    assert spk.shape == (T, B, F)


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_forward_sequence_explicit_shape(cls, kwargs):
    n = cls(**kwargs)
    state = n.initial_state((B, F), device=X_SEQ.device, dtype=X_SEQ.dtype)
    out = n.forward_sequence(X_SEQ, state)
    spk, *final = out
    assert spk.shape == (T, B, F)
    assert len(final) == n._n_explicit_state


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_forward_sequence_explicit_default_state(cls, kwargs):
    n = cls(**kwargs)
    out = n.forward_sequence(X_SEQ)
    spk, *final = out
    assert spk.shape == (T, B, F)
    assert len(final) == n._n_explicit_state


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_forward_sequence_hidden_explicit_match(cls, kwargs):
    hidden = cls(init_hidden=True, **kwargs)
    explicit = cls(**kwargs)
    state = explicit.initial_state((B, F), device=X_SEQ.device, dtype=X_SEQ.dtype)
    spk_h = hidden.forward_sequence(X_SEQ)
    spk_e, *_ = explicit.forward_sequence(X_SEQ, state)
    assert torch.equal(spk_h, spk_e)


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_forward_sequence_returns_same_as_step_loop(cls, kwargs):
    n = cls(**kwargs)
    state = n.zero_state((B, F), device=X_SEQ.device, dtype=X_SEQ.dtype)
    out = n.forward_sequence(X_SEQ, state)
    spk_seq, *final = out
    spk_loop, state_loop = [], state
    for t in range(T):
        o = n.forward(X_SEQ[t], *state_loop)
        spk_loop.append(o[0])
        state_loop = o[1:]
    assert torch.equal(spk_seq, torch.stack(spk_loop))
    for a, b in zip(final, state_loop):
        assert torch.equal(a, b)


def test_forward_sequence_hidden_rejects_state():
    n = LIF(beta=0.9, init_hidden=True)
    with pytest.raises(ValueError, match="takes no initial state"):
        n.forward_sequence(X_SEQ, (torch.zeros(B, F),))


def test_forward_sequence_too_few_dims_raises():
    n = LIF(beta=0.9)
    with pytest.raises(ValueError, match="at least 3 dims"):
        n.forward_sequence(torch.randn(B, F))


def test_forward_sequence_state_shape_mismatch_raises():
    n = LIF(beta=0.9)
    bad = (torch.zeros(B, 2 * F),)
    with pytest.raises(ValueError, match="state 'mem'"):
        n.forward_sequence(X_SEQ, bad)


def test_check_input_0dim_with_size_raises():
    n = LIF(beta=0.9, size=F)
    with pytest.raises(ValueError, match="0-dim"):
        n(torch.tensor(1.0))


def test_check_input_0dim_without_size_ok():
    n = LIF(beta=0.9, size=None, init_hidden=True)
    spk = n(torch.tensor(1.0))
    assert spk.shape == ()


def test_check_input_feature_mismatch_raises():
    n = LIF(beta=0.9, size=F)
    with pytest.raises(ValueError, match="got 2 features but size"):
        n(torch.randn(1, 2))


X_SEQ3 = torch.randn(3, B, F)


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_compile_hidden_matches_eager(cls, kwargs):
    eager = cls(init_hidden=True, **kwargs)
    compiled = torch.compile(cls(init_hidden=True, **kwargs))
    eager_out = eager(X_SEQ3[0])
    compiled_out = compiled(X_SEQ3[0])
    assert torch.allclose(eager_out, compiled_out, atol=1e-5)


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_compile_explicit_matches_eager(cls, kwargs):
    eager = cls(**kwargs)
    compiled = cast(SpikingModule, torch.compile(cls(**kwargs)))
    state_e = eager.initial_state((B, F))
    state_c = compiled.initial_state((B, F))
    e = eager(X_SEQ3[0], *state_e)
    c = compiled(X_SEQ3[0], *state_c)
    assert len(e) == len(c)
    for a, b in zip(e, c):
        assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_compile_sequence_hidden_matches_eager(cls, kwargs):
    eager = cls(init_hidden=True, **kwargs)
    compiled = cast(SpikingModule, torch.compile(cls(init_hidden=True, **kwargs)))
    e = eager.forward_sequence(X_SEQ3)
    c = cast(torch.Tensor, compiled.forward_sequence(X_SEQ3))
    assert torch.allclose(e, c, atol=1e-5)


def test_state_metadata_cached_once():
    n = LIF(beta=0.9)
    assert n._cached_state_specs is None
    assert n._state_names == ("spk", "mem")
    assert n._reset_values == (0.0, 0.0)
    assert n._n_state == 2
    assert n._n_explicit_state == 1
    assert n._cached_state_specs is not None
    assert n._get_values_to_reset() == {"spk": 0.0, "mem": 0.0}


def test_validate_true_raises_on_bad_step_arity():
    n = _WrongArityExplicit(validate=True)
    mem = torch.zeros(B, F)
    w = torch.zeros(B, F)
    with pytest.raises(ValueError, match="returned 2 tensor"):
        n(X, mem, w)


def test_validate_false_skips_step_arity_check():
    n = _WrongArityExplicit(validate=False)
    mem = torch.zeros(B, F)
    w = torch.zeros(B, F)
    out = n(X, mem, w)
    assert len(out) == 2


def test_validate_true_raises_on_state_shape_mismatch():
    n = _WrongArityExplicit(validate=True)
    mem = torch.zeros(B + 1, F)
    w = torch.zeros(B + 1, F)
    with pytest.raises(ValueError, match="state 'mem'"):
        n(X, mem, w)


def test_validate_false_skips_state_shape_check():
    n = _WrongArityExplicit(validate=False)
    mem = torch.zeros(B + 1, F)
    w = torch.zeros(B + 1, F)
    out = n(X, mem, w)
    assert len(out) == 2


def test_set_validation_global_toggle(monkeypatch):
    from blowtorch_snn import set_validation, get_validation

    original = get_validation()
    try:
        set_validation(False)
        assert get_validation() is False
        n = LIF(beta=0.9)
        assert n.validate is False
        n = LIF(beta=0.9, validate=True)
        assert n.validate is True
        set_validation(True)
        n = LIF(beta=0.9)
        assert n.validate is True
    finally:
        set_validation(original)


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_fused_sequence_hook_dispatch(cls, kwargs):
    n = cls(init_hidden=True, validate=False, **kwargs)

    def fused(x_seq, *state):
        return torch.full(x_seq.shape, -1.0)

    n.use_fused_sequence = True
    n._fused_forward_sequence = fused
    assert torch.all(n.forward_sequence(X_SEQ) == -1.0)
    n.use_fused_sequence = False
    assert n.forward_sequence(X_SEQ).shape == X_SEQ.shape


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_fused_sequence_default_matches_reference_scan(cls, kwargs):
    ref = cls(init_hidden=True, validate=False, **kwargs)
    fused = cls(init_hidden=True, validate=False, use_fused_sequence=True, **kwargs)
    expected = ref.forward_sequence(X_SEQ)
    got = fused.forward_sequence(X_SEQ)
    assert torch.equal(got, expected)
    assert got.shape == X_SEQ.shape


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_hidden_state_roundtrips_through_state_dict(cls, kwargs):
    n = cls(init_hidden=True, validate=False, **kwargs)
    n(X[0])
    snapshot = n.state_dict()
    assert "_extra_state" in snapshot
    assert snapshot["_extra_state"] is not None
    n.reset()
    n.load_state_dict(snapshot)
    for name in n._state_names:
        assert torch.equal(getattr(n, name), snapshot["_extra_state"][name])


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_explicit_mode_extra_state_is_none(cls, kwargs):
    n = cls(init_hidden=False, validate=False, **kwargs)
    assert n.get_extra_state() is None


def test_forward_sequence_empty_raises():
    n = LIF(beta=0.9)
    with pytest.raises(ValueError, match="at least one timestep"):
        n.forward_sequence(torch.empty(0, B, F))


def test_hidden_state_spec_check_unconditional_when_validate_off():
    n = _BadSpecDuplicate(validate=False)
    with pytest.raises(ValueError, match="unique"):
        n._state_names


class _FixedShapeHidden(SpikingModule):
    neuron_name = "_FixedShapeHidden"

    def __init__(self):
        super().__init__(size=None, init_hidden=True, validate=True)

    def _get_state_specs(self) -> tuple[StateSpec, ...]:
        return (
            StateSpec("spk", 0.0),
            StateSpec("mem", 0.0, shape=(B, 2 * F)),
        )

    def _step(self, x, mem):
        spk = torch.zeros_like(x)
        return spk, mem


def test_hidden_alloc_honors_concrete_shape():
    n = _FixedShapeHidden()
    n(X)
    assert n.mem.shape == (B, 2 * F)
    assert n.spk.shape == X.shape
    with pytest.raises(ValueError, match="has"):
        n(torch.randn(B, F + 1))


def test_hidden_buffer_replaced_by_setattr_still_registered():
    n = LIF(beta=0.9, init_hidden=True)
    n(X)
    assert "mem" in n._buffers
    assert n.mem.shape == X.shape
    n(X)
    assert "mem" in n._buffers


def test_extra_repr_includes_shared_options():
    n = LIF(beta=0.9, size=16, init_hidden=True)
    r = repr(n)
    assert "size=16" in r
    assert "init_hidden=True" in r
    n2 = LIF(beta=0.9)
    assert "init_hidden=False" in repr(n2)


def test_top_level_reset_restricted_to_spiking_modules():
    import torch.nn as nn

    class Foreign(nn.Module):
        def reset(self):
            raise AssertionError("foreign reset must not be called")

    container = nn.Sequential(LIF(beta=0.9, init_hidden=True), Foreign())
    lif = cast(LIF, container[0])
    lif(X)
    from blowtorch_snn import reset as bsnn_reset

    bsnn_reset(container)
    assert torch.equal(lif.mem, torch.zeros_like(lif.mem))


def test_top_level_detach_restricted_to_spiking_modules():
    import torch.nn as nn

    class Foreign(nn.Module):
        def detach(self):
            raise AssertionError("foreign detach must not be called")

    container = nn.Sequential(LIF(beta=0.9, init_hidden=True), Foreign())
    container[0](X)
    from blowtorch_snn import detach as bsnn_detach

    bsnn_detach(container)
    assert container[0].mem.requires_grad is False


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_forward_sequence_with_states_matches_forward_sequence(cls, kwargs):
    n = cls(validate=False, **kwargs)
    state = n.initial_state((B, F), device=X_SEQ.device, dtype=X_SEQ.dtype)
    spikes, final, all_states = n.forward_sequence_with_states(X_SEQ, state)
    ref_spikes, *ref_final = n.forward_sequence(X_SEQ, state)
    assert torch.equal(spikes, ref_spikes)
    for a, b in zip(final, ref_final):
        assert torch.equal(a, b)
    assert len(all_states) == T
    assert torch.equal(all_states[-1][0], ref_final[0])
    # each intermediate state matches the state after running t+1 steps
    cur = state
    for t in range(T):
        cur = n.forward(X_SEQ[t], *cur)[1:]
        for a, b in zip(all_states[t], cur):
            assert torch.equal(a, b)


def test_forward_sequence_with_states_rejects_hidden():
    n = LIF(beta=0.9, init_hidden=True)
    with pytest.raises(ValueError, match="init_hidden=False"):
        n.forward_sequence_with_states(X_SEQ)


def test_forward_sequence_with_states_empty_raises():
    n = LIF(beta=0.9)
    with pytest.raises(ValueError, match="at least one timestep"):
        n.forward_sequence_with_states(torch.empty(0, B, F))


@pytest.mark.parametrize("cls,kwargs", NEURONS)
def test_initial_state_like_matches_device_dtype(cls, kwargs):
    n = cls(validate=False, **kwargs)
    x = X.to(dtype=torch.float64)
    state = n.initial_state_like(x)
    for t in state:
        assert t.shape == x.shape
        assert t.dtype == torch.float64
    state_seq = n.initial_state_for_sequence(X_SEQ.to(dtype=torch.float64))
    for t in state_seq:
        assert t.shape == (B, F)
        assert t.dtype == torch.float64
    zero = n.zero_state_like(x)
    for t in zero:
        assert torch.all(t == 0)
        assert t.dtype == torch.float64


def test_initial_state_for_sequence_requires_3dims():
    n = LIF(beta=0.9)
    with pytest.raises(ValueError, match="at least 3 dims"):
        n.initial_state_for_sequence(torch.randn(B, F))


def test_qif_learnable_reset_tracks_v_rest():
    n = QIF(beta=0.01, v_rest=0.0, learnable_v_rest=True, init_hidden=True)
    n(X)
    n.v_rest.data.fill_(5.0)
    n.reset()
    assert torch.all(n.mem == 5.0)
    state = n.initial_state((B, F))
    assert torch.all(state[0] == 5.0)


def test_validate_false_skips_sequence_input_check(monkeypatch):
    n = LIF(beta=0.9, validate=False, init_hidden=True)
    n.forward_sequence(X_SEQ)


def test_compile_sequence_scan_routes_through_fused_hook():
    n = LIF(beta=0.9, init_hidden=True)
    n.compile_sequence_scan()
    assert n.use_fused_sequence is True
    assert n._fused_forward_sequence is not None
    spk = cast(torch.Tensor, n.forward_sequence(X_SEQ))
    assert spk.shape == (T, B, F)


def test_hard_zero_reset_matches_zero_reset_on_spikes():
    from blowtorch_snn import hard_zero_reset, zero_reset
    mem = torch.tensor([0.5, 2.0, -1.0])
    spk = torch.tensor([0.0, 1.0, 0.0])
    thresh = torch.tensor(1.0)
    assert torch.equal(hard_zero_reset(mem, spk, thresh), zero_reset(mem, spk, thresh))


def test_step_state_returns_spk_and_next():
    n = LIF(beta=0.9, validate=False)
    state = n.initial_state((B, F))
    spk, next_state = n.step_state(X, state)
    ref = n.forward(X, *state)
    assert torch.equal(spk, ref[0])
    for a, b in zip(next_state, ref[1:]):
        assert torch.equal(a, b)


def test_step_alias_matches_step_state():
    n = LIF(beta=0.9, validate=False)
    state = n.initial_state((B, F))
    assert torch.equal(n.step(X, state)[0], n.step_state(X, state)[0])
    assert all(
        torch.equal(a, b) for a, b in zip(n.step(X, state)[1], n.step_state(X, state)[1])
    )


def test_no_validation_context_disables_and_restores():
    from blowtorch_snn import no_validation

    n = LIF(beta=0.9, init_hidden=True)
    assert n.validate is True
    with no_validation():
        assert n.validate is False
        with no_validation():
            assert n.validate is False
        assert n.validate is False
    assert n.validate is True


def test_validate_property_follows_global_toggle():
    from blowtorch_snn import no_validation, set_validation

    n = LIF(beta=0.9, init_hidden=True, validate=None)
    assert n.validate is True
    set_validation(False)
    try:
        assert n.validate is False
    finally:
        set_validation(True)
    assert n.validate is True


def test_validate_constructor_override_wins_globally():
    from blowtorch_snn import no_validation

    n = LIF(beta=0.9, init_hidden=True, validate=False)
    with no_validation():
        assert n.validate is False


def test_state_device_mismatch_raises():
    n = LIF(beta=0.9, validate=True)
    state = n.initial_state((B, F), device=torch.device("meta"))
    with pytest.raises(ValueError, match="device"):
        n.forward_sequence(X_SEQ, state)


def test_state_dtype_mismatch_raises():
    n = LIF(beta=0.9, validate=True)
    state = n.initial_state((B, F), dtype=torch.float64)
    with pytest.raises(ValueError, match="dtype"):
        n.forward_sequence(X_SEQ, state)


def test_state_dtype_mismatch_skipped_when_validate_off():
    n = LIF(beta=0.9, validate=False)
    state = n.initial_state((B, F), dtype=torch.float64)
    out = n.forward_sequence(X_SEQ, state)
    assert isinstance(out, tuple)
    assert out[0].shape == (T, B, F)


def test_forward_sequence_output_explicit():
    from blowtorch_snn import SequenceOutput

    n = LIF(beta=0.9, validate=False)
    out = n.forward_sequence_output(X_SEQ)
    assert isinstance(out, SequenceOutput)
    assert out.final_state is not None
    assert out.states is None
    spikes, *final = n.forward_sequence(X_SEQ)
    assert torch.equal(out.spikes, spikes)
    for a, b in zip(out.final_state, final):
        assert torch.equal(a, b)


def test_forward_sequence_output_hidden():
    from blowtorch_snn import SequenceOutput

    n = LIF(beta=0.9, init_hidden=True)
    out = n.forward_sequence_output(X_SEQ)
    assert isinstance(out, SequenceOutput)
    assert out.final_state is None
    assert out.states is None


def test_forward_sequence_output_return_states():
    from blowtorch_snn import SequenceOutput

    n = LIF(beta=0.9, validate=False)
    out = n.forward_sequence_output(X_SEQ, return_states=True)
    assert isinstance(out, SequenceOutput)
    assert out.states is not None
    assert len(out.states) == T
    assert out.final_state is not None
    assert len(out.states[0]) == len(out.final_state)


def test_forward_sequence_with_states_store_states_false():
    n = LIF(beta=0.9, validate=False)
    spikes, final, states = n.forward_sequence_with_states(X_SEQ, store_states=False)
    assert states is None
    assert spikes.shape == (T, B, F)
    assert final is not None


def test_forward_sequence_with_states_callback():
    n = LIF(beta=0.9, validate=False)
    seen = []
    n.forward_sequence_with_states(
        X_SEQ, store_states=False, state_callback=lambda t, st: seen.append((t, st))
    )
    assert len(seen) == T
    assert [t for t, _ in seen] == list(range(T))


def test_extra_state_detached():
    n = LIF(beta=0.9, init_hidden=True)
    n(X)
    extra = n.get_extra_state()
    assert extra is not None
    assert all(not t.requires_grad for t in extra.values())
    assert all(t.grad_fn is None for t in extra.values())
