# NOTE: B = batch size, F = number of features (neurons), X = input tensor of shape (B, F).
import torch
import pytest

from blowtorch_snn import (
    LIF,
    QIF,
    Izhikevich,
    IzhPreset,
    AdEx,
    SRM,
    HH,
    SpikingModule,
    subtract_reset,
    zero_reset,
    no_reset,
    reset,
    detach,
)

NEURONS = [
    ("LIF", LIF(beta=0.9, init_hidden=True)),
    ("QIF", QIF(beta=0.01, init_hidden=True)),
    ("Izhikevich", Izhikevich(init_hidden=True)),
    ("AdEx", AdEx(beta=0.9, init_hidden=True)),
    ("SRM", SRM(beta=0.9, init_hidden=True)),
    ("HH", HH(init_hidden=True)),
]

B, F = 4, 64
X = torch.randn(B, F)


@pytest.mark.parametrize("name,neuron", NEURONS, ids=lambda n: n if isinstance(n, str) else "")
def test_all_are_spiking_modules(name, neuron):
    assert isinstance(neuron, SpikingModule)


@pytest.mark.parametrize("name,neuron", NEURONS)
def test_hidden_forward_shape(name, neuron):
    spk = neuron(X)
    assert spk.shape == X.shape
    assert spk.dtype == X.dtype
    assert set(torch.unique(spk).tolist()).issubset({0.0, 1.0})
    neuron.reset()


RESET_VALUES = {
    "LIF": {"mem": 0.0},
    "QIF": {"mem": 0.0},
    "Izhikevich": {"mem": -65.0, "u": -13.0},
    "AdEx": {"mem": -70.0, "w": 0.0},
    "SRM": {"mem": 0.0},
    "HH": {"mem": -65.0, "m": 0.0529, "h": 0.5961, "n": 0.3177},
}


@pytest.mark.parametrize("name,neuron", NEURONS)
def test_reset_clears_state(name, neuron):
    neuron(X)
    neuron.reset()
    expected = RESET_VALUES[name]
    for buf, val in expected.items():
        assert torch.allclose(
            getattr(neuron, buf), torch.full_like(getattr(neuron, buf), val)
        )


@pytest.mark.parametrize("name,neuron", NEURONS)
def test_detach_detaches_state(name, neuron):
    neuron(X)
    neuron.detach()
    for buf in ("mem", "u", "w", "m", "h", "n"):
        if hasattr(neuron, buf):
            assert getattr(neuron, buf).requires_grad is False


def test_reset_helper_walks_modules():
    lif = LIF(beta=0.9, init_hidden=True)
    lif(X)
    reset(lif)
    assert torch.allclose(lif.mem, torch.zeros_like(lif.mem))


def test_detach_helper_walks_modules():
    lif = LIF(beta=0.9, init_hidden=True)
    lif(X)
    detach(lif)
    assert lif.mem.requires_grad is False


def test_explicit_state_passing_lif():
    lif = LIF(beta=0.9, init_hidden=False)
    mem = torch.zeros(B, F)
    spk, mem = lif(X, mem)
    assert spk.shape == X.shape
    assert mem.shape == X.shape
    assert spk.dtype == X.dtype


def test_explicit_state_passing_izhikevich():
    izh = Izhikevich(init_hidden=False)
    mem = torch.full((B, F), -65.0)
    u = torch.zeros(B, F)
    spk, mem, u = izh(X, mem, u)
    assert spk.shape == X.shape and mem.shape == X.shape and u.shape == X.shape


def test_explicit_state_passing_hh():
    hh = HH(init_hidden=False)
    mem = torch.zeros(B, F)
    m = torch.zeros(B, F)
    h = torch.ones(B, F)
    n = torch.zeros(B, F)
    spk, mem, m, h, n = hh(X, mem, m, h, n)
    assert spk.shape == X.shape
    assert mem.shape == X.shape


def test_state_accessible_off_module():
    lif = LIF(beta=0.9, init_hidden=True)
    spk = lif(X)
    assert spk.shape == X.shape
    assert lif.mem.shape == X.shape


def test_hidden_state_shape_mismatch_raises():
    lif = LIF(beta=0.9, init_hidden=True)
    lif(X)
    with pytest.raises(ValueError):
        lif(torch.randn(B + 1, F))


def test_hidden_state_dtype_mismatch_raises():
    lif = LIF(beta=0.9, init_hidden=True)
    lif(X)
    with pytest.raises(ValueError):
        lif(X.double())


@pytest.mark.parametrize("mechanism", [subtract_reset, zero_reset, no_reset])
def test_reset_mechanisms(mechanism):
    lif = LIF(beta=0.9, init_hidden=True, reset_mechanism=mechanism)
    spk = lif(X)
    assert spk.shape == X.shape


def test_learnable_params_gradient_flows():
    lif = LIF(beta=0.9, init_hidden=True, learnable_beta=True, learnable_threshold=True)
    spk = lif(X)
    loss = spk.mean()
    loss.backward()
    assert lif.beta.grad is not None
    assert lif.threshold.grad is not None
    assert torch.isfinite(lif.beta.grad)


def test_input_gradient_flows():
    x = X.clone().requires_grad_(True)
    lif = LIF(beta=0.9, init_hidden=True)
    spk = lif(x)
    spk.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("cls,kwargs", [
    (LIF, {"beta": 0.9}),
    (QIF, {"beta": 0.01}),
    (Izhikevich, {}),
    (AdEx, {"beta": 0.9}),
    (SRM, {"beta": 0.9}),
    (HH, {}),
])
def test_size_validation_at_construction(cls, kwargs):
    with pytest.raises(ValueError):
        cls(size=0, **kwargs)
    with pytest.raises(ValueError):
        cls(size=-5, **kwargs)
    with pytest.raises(ValueError):
        cls(size=3.5, **kwargs)


@pytest.mark.parametrize("cls,kwargs", [
    (LIF, {"beta": 0.9}),
    (QIF, {"beta": 0.01}),
    (Izhikevich, {}),
    (AdEx, {"beta": 0.9}),
    (SRM, {"beta": 0.9}),
    (HH, {}),
])
def test_size_none_is_valid(cls, kwargs):
    neuron = cls(size=None, init_hidden=True, **kwargs)
    spk = neuron(X)
    assert spk.shape == X.shape
    neuron.reset()


def test_declared_size_accepts_matching_input():
    lif = LIF(beta=0.9, size=F, init_hidden=True)
    spk = lif(X)
    assert spk.shape == X.shape


def test_state_persists_across_steps():
    lif = LIF(beta=0.9, init_hidden=True)
    first = lif(X)
    second = lif(X)
    assert not torch.equal(first, second)


def test_hidden_state_dtype_move_float_half():
    lif = LIF(beta=0.9, init_hidden=True)
    lif(X)
    lif = lif.float()
    assert lif.mem.dtype == torch.float32
    assert lif.beta.dtype == torch.float32
    lif = lif.half()
    assert lif.mem.dtype == torch.float16
    assert lif.beta.dtype == torch.float16
    spk = lif(X.half())
    assert spk.dtype == torch.float16


def test_hidden_state_device_move():
    lif = LIF(beta=0.9, init_hidden=True)
    lif(X)
    lif = lif.to("cuda")
    assert lif.mem.device.type == "cuda"
    spk = lif(X.to("cuda"))
    assert spk.device.type == "cuda"


def test_constraints_apply_per_neuron():
    lif = LIF(beta=0.9, learnable_beta=True, beta_constraint=torch.sigmoid, init_hidden=True)
    spk = lif(X)
    assert torch.allclose(lif.beta_constraint(lif.beta), torch.sigmoid(lif.beta))
    assert spk.shape == X.shape


def test_default_beta_constraint_clamps_to_unit_range():
    lif = LIF(beta=2.0, learnable_beta=True, init_hidden=True)
    assert lif.beta_constraint(torch.tensor(2.0)).item() == 1.0
    assert lif.beta_constraint(torch.tensor(-0.5)).item() == 0.0
    assert lif.beta_constraint(torch.tensor(0.9)).item() == pytest.approx(0.9)


def test_default_threshold_constraint_clamps_positive():
    lif = LIF(beta=0.9, learnable_threshold=True, init_hidden=True)
    assert lif.threshold_constraint(torch.tensor(0.0)).item() == pytest.approx(1e-6)
    assert lif.threshold_constraint(torch.tensor(1.0)).item() == 1.0


def test_threshold_constraint_negative_allowed():
    adex = AdEx(beta=0.9, learnable_threshold=True, init_hidden=True)
    adex(X)
    assert torch.allclose(adex.threshold_constraint(adex.threshold), adex.threshold)


def test_constrained_beta_gradient_flows():
    lif = LIF(beta=0.9, learnable_beta=True, beta_constraint=torch.sigmoid, init_hidden=True)
    spk = lif(X)
    spk.mean().backward()
    assert lif.beta.grad is not None
    assert torch.isfinite(lif.beta.grad)
    assert (lif.beta > 0.0).all() and (lif.beta < 1.0).all()


def test_qif_v_min_v_max_clamp():
    n = QIF(beta=0.01, init_hidden=False, v_min=-1.0, v_max=1.0)
    x = torch.zeros(B, F)
    spk, mem = n(x, torch.full((B, F), 5.0))
    assert torch.all(mem <= 1.0)
    spk, mem = n(x, torch.full((B, F), -5.0))
    assert torch.all(mem >= -1.0)


def test_qif_no_clamps_by_default():
    n = QIF(beta=0.01, init_hidden=False)
    assert n.v_min is None and n.v_max is None
    x = torch.zeros(B, F)
    spk, mem = n(x, torch.full((B, F), 5.0))
    assert torch.all(mem > 1.0)


def test_hh_substeps_must_be_positive():
    with pytest.raises(ValueError, match="positive int"):
        HH(substeps=0)


def test_hh_substeps_preserves_input_scaling():
    B, F, dt = 4, 64, 0.01
    x = torch.full((B, F), 2.0)
    expected = 2.0 * dt  # x*dt/C with C=1 => 0.02
    for sub in (1, 4):
        n = HH(
            init_hidden=True,
            substeps=sub,
            dt=dt,
            validate=False,
            gNa=0.0,
            gK=0.0,
            gL=0.0,
        )
        n(torch.zeros(B, F))
        mem0 = n.mem.clone()
        n(x)
        dmem = n.mem - mem0
        # mem starts at REST (-65 mV); float32 magnitude-cancellation near
        # -65 limits the measurable delta to ~1e-5, hence atol=1e-4.
        assert torch.allclose(dmem, torch.full_like(dmem, expected), atol=1e-4)


def test_hh_steady_state_gates_at_rest():
    n = HH()
    mem, m, h, g = n.steady_state((B, F))
    assert torch.all(mem == n.REST)
    for gate in (m, h, g):
        assert gate.shape == (B, F)
        assert torch.all(gate >= 0.0) and torch.all(gate <= 1.0)
    assert torch.all(h > m) and torch.all(g > m)


def test_hh_steady_state_accepts_voltage():
    n = HH()
    mem, m, h, g = n.steady_state((B, F), v=-50.0)
    assert torch.all(mem == -50.0)
    for gate in (m, h, g):
        assert torch.all(gate >= 0.0) and torch.all(gate <= 1.0)


def test_hh_steady_state_rollout_stays_finite():
    n = HH(init_hidden=True, validate=False)
    n(torch.zeros(B, F))
    m, h, g = n._steady_state_gates(torch.full((B, F), n.EL))
    n.mem.copy_(torch.full((B, F), n.EL))
    n.m.copy_(m)
    n.h.copy_(h)
    n.n.copy_(g)
    for _ in range(5):
        spk = n(torch.randn(B, F))
        assert torch.isfinite(n.mem).all()


def test_explicit_hidden_equivalence_lif():
    hidden = LIF(beta=0.9, init_hidden=True)
    explicit = LIF(beta=0.9, init_hidden=False)
    T = 5
    x = torch.randn(T, B, F)
    mem = torch.zeros(B, F)
    for t in range(T):
        spk_h = hidden(x[t])
        spk_e, mem = explicit(x[t], mem)
        assert torch.allclose(spk_h, spk_e)
        assert torch.allclose(hidden.mem, mem)


def test_explicit_hidden_equivalence_izhikevich():
    hidden = Izhikevich(init_hidden=True)
    explicit = Izhikevich(init_hidden=False)
    T = 5
    x = torch.randn(T, B, F)
    mem = torch.full((B, F), -65.0)
    u = torch.full((B, F), hidden.u_init)
    for t in range(T):
        spk_h = hidden(x[t])
        spk_e, mem, u = explicit(x[t], mem, u)
        assert torch.allclose(spk_h, spk_e)
        assert torch.allclose(hidden.mem, mem)
        assert torch.allclose(hidden.u, u)


def test_adex_responds_to_input():
    # Push v_thresh/threshold far away so no spike/reset interferes with the
    # comparison; mem must rise with the input current x.
    n = AdEx(beta=0.9, init_hidden=False, v_thresh=1000.0, threshold=1000.0)
    _, mem_zero, _ = n(torch.zeros(B, F), torch.zeros(B, F), torch.zeros(B, F))
    _, mem_one, _ = n(torch.ones(B, F), torch.zeros(B, F), torch.zeros(B, F))
    assert (mem_one > mem_zero).all()


def test_adex_w_shares_beta_gain():
    n = AdEx(beta=0.5, init_hidden=False, a=0.0, b=0.0)
    tau_w = n.tau_w
    mem = torch.zeros(B, F)
    w = torch.ones(B, F)
    _, mem, w = n(torch.zeros(B, F), mem, w)
    expected_w = 1.0 * (1.0 - float(n.beta) / tau_w)
    assert torch.allclose(w, torch.full_like(w, expected_w), atol=1e-6)


def test_hh_initial_state_no_spike_at_rest():
    n = HH(init_hidden=True, validate=False)
    for _ in range(100):
        spk = n(torch.zeros(B, F))
        assert spk.sum() == 0.0


def test_hh_rate_gradient_nan_free():
    from blowtorch_snn.neurons.hh import _hh_rate
    xs = torch.linspace(-1.0, 1.0, 101).requires_grad_(True)
    for a, c in ((0.1, 10.0), (0.01, 10.0)):
        y = _hh_rate(xs, a, c).sum()
        y.backward()
        assert xs.grad is not None
        assert torch.isfinite(xs.grad).all()
        xs.grad = None


def test_izhikevich_preset_conflict_raises():
    with pytest.raises(ValueError, match="not both"):
        Izhikevich(preset=IzhPreset.RS, a=0.02)


def test_izhikevich_substeps_must_be_positive():
    from typing import cast, Any
    with pytest.raises(ValueError, match="positive int"):
        Izhikevich(substeps=0)
    with pytest.raises(ValueError, match="positive int"):
        Izhikevich(substeps=cast(Any, 2.5))
    with pytest.raises(ValueError, match="positive int"):
        Izhikevich(substeps=-1)


def test_izhikevich_preset_overrides_explicit_raises():
    with pytest.raises(ValueError, match="preset"):
        Izhikevich(preset=IzhPreset.RS, b=0.5)


def test_srm_refractory_window():
    n = SRM(beta=0.9, tau_ref=3.0, init_hidden=False)
    x = torch.ones(B, F) * 5.0
    mem = torch.zeros(B, F)
    ref = torch.zeros(B, F)
    spks = []
    for _ in range(4):
        spk, mem, ref = n(x, mem, ref)
        spks.append(spk)
    assert spks[0].dtype == x.dtype
    assert torch.all(spks[0] > 0)
    assert torch.all(spks[1] == 0.0)
    assert torch.all(spks[2] == 0.0)
    assert torch.all(spks[3] > 0)


def test_qif_learnable_parameters():
    n = QIF(
        beta=0.01,
        learnable_v_rest=True,
        learnable_v_thresh=True,
        learnable_membrane_resistance=True,
        init_hidden=True,
    )
    for name in ("v_rest", "v_thresh", "membrane_resistance"):
        param = getattr(n, name)
        assert isinstance(param, torch.nn.Parameter)
        assert param.requires_grad
    spk = n(X)
    spk.mean().backward()
    for name in ("v_rest", "v_thresh", "membrane_resistance"):
        grad = getattr(n, name).grad
        assert grad is not None
        assert torch.isfinite(grad)


def test_lif_qif_docstrings():
    assert isinstance(LIF.__doc__, str) and len(LIF.__doc__) > 0
    assert isinstance(QIF.__doc__, str) and len(QIF.__doc__) > 0


def test_state_dtype_fallback_ignores_module_dtype():
    lif = LIF(beta=0.9, init_hidden=True).half()
    for state in (lif.zero_state((B, F)), lif.initial_state((B, F))):
        for t in state:
            assert t.dtype == torch.float32
