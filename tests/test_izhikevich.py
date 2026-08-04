# NOTE: Reference implementation mirrors the paper MATLAB loop (detect-first):
# spike from previous mem -> reset fired mem to c -> u += d*spk -> integrate mem
# -> update u from newly integrated mem. Half-step integration matches the
# neuron's default substeps=2.
import torch
import pytest

from blowtorch_snn import Izhikevich
from blowtorch_snn.neurons.izhikevich import (
    CLASS1_MEMBRANE,
    IzhPreset,
    STANDARD_MEMBRANE,
)

PRESETS = [IzhPreset.RS, IzhPreset.IB, IzhPreset.CH, IzhPreset.FS, IzhPreset.LTS]


def ref_izh_step(x, mem, u, a, b, c, d, threshold, dt=1.0, substeps=2):
    spk = (mem > threshold).to(x.dtype)
    mem = torch.where(spk > 0, torch.full_like(mem, c), mem)
    u = u + d * spk
    h = dt / substeps
    v2, v1, bias = STANDARD_MEMBRANE
    for _ in range(substeps):
        mem = mem + h * (v2 * mem * mem + v1 * mem + bias - u + x)
        u = u + h * a * (b * mem - u)
    return spk, mem, u


def ref_izh_trajectory(preset, x, threshold, dt, substeps):
    p = preset.params
    a, b, c, d = p.a, p.b, p.c, p.d
    mem = torch.full_like(x[0], c)
    u = torch.full_like(x[0], b * c)
    spikes = []
    mems = []
    for t in range(x.shape[0]):
        spk, mem, u = ref_izh_step(x[t], mem, u, a, b, c, d, threshold, dt, substeps)
        spikes.append(spk)
        mems.append(mem)
    return torch.stack(spikes), torch.stack(mems)


@pytest.mark.parametrize("preset", PRESETS)
def test_izhikevich_matches_paper_reference(preset):
    neuron = Izhikevich(preset=preset, init_hidden=False)
    T, B, F = 20, 4, 8
    x = torch.randn(T, B, F) * 5.0
    ref_spk, ref_mem = ref_izh_trajectory(
        preset, x, float(neuron.threshold), neuron.dt, neuron.substeps
    )
    p = preset.params
    mem = torch.full((B, F), p.c)
    u = torch.full((B, F), p.b * p.c)
    out_spk = []
    out_mem = []
    for t in range(T):
        spk, mem, u = neuron(x[t], mem, u)
        out_spk.append(spk)
        out_mem.append(mem)
    assert torch.allclose(torch.stack(out_spk), ref_spk)
    assert torch.allclose(torch.stack(out_mem), ref_mem)


def test_fired_neurons_reset_before_integration():
    neuron = Izhikevich(preset=IzhPreset.RS, init_hidden=False)
    B, F = 2, 4
    mem = torch.full((B, F), 60.0)
    u = torch.full((B, F), -13.0)
    x = torch.zeros(B, F)
    spk, mem, u = neuron(x, mem, u)
    assert torch.all(spk == 1.0)
    assert torch.all(mem < 0.0)


def test_u_receives_d_before_recovery_update():
    neuron = Izhikevich(preset=IzhPreset.RS, init_hidden=False)
    B, F = 1, 1
    mem = torch.full((B, F), 60.0)
    u = torch.full((B, F), -13.0)
    x = torch.zeros(B, F)
    _, mem, u = neuron(x, mem, u)
    assert torch.all(u > -13.0)


def test_preset_unknown_raises():
    with pytest.raises(ValueError):
        Izhikevich(preset="NOPE")  # type: ignore[arg-type]


def test_preset_must_be_enum():
    with pytest.raises(ValueError):
        Izhikevich(preset="RS")  # type: ignore[arg-type]


def test_v_reset_aliases_c():
    neuron = Izhikevich(v_reset=-70.0)
    assert neuron.c == -70.0
    assert neuron.v_reset == -70.0


def test_u_init_default_is_b_times_c():
    neuron = Izhikevich(preset=IzhPreset.FS)
    assert neuron.u_init == neuron.b * neuron.c


def test_half_step_equals_single_dt_half():
    neuron = Izhikevich(substeps=1)
    assert neuron.substeps == 1
    assert neuron.dt == 1.0


def test_all_presets_have_params():
    assert len(list(IzhPreset)) == 27
    for preset in IzhPreset:
        p = preset.params
        assert preset.description
        assert p.membrane in (STANDARD_MEMBRANE, CLASS1_MEMBRANE)
        assert p.u_mode in ("standard", "accommodation")


def test_spike_detection_default_is_pre():
    neuron = Izhikevich()
    assert neuron.spike_detection == "pre"


def test_spike_detection_invalid_raises():
    with pytest.raises(ValueError, match="spike_detection"):
        Izhikevich(spike_detection="mid")  # type: ignore[arg-type]


def test_spike_detection_post_is_canonical_order():
    a, b, c, d = 0.02, 0.2, -65.0, 8.0
    threshold = 30.0
    neuron = Izhikevich(
        a=a, b=b, c=c, d=d, threshold=threshold, spike_detection="post", substeps=2
    )
    x = torch.tensor([10.0, 100.0, -5.0, 50.0])
    mem0 = neuron.c * torch.ones_like(x)
    u0 = neuron.u_init * torch.ones_like(x)
    spk, mem, u = neuron._step(x, mem0.clone(), u0.clone())

    spk_ref, mem_ref, u_ref = ref_izh_post_step(x, mem0, u0, a, b, c, d, threshold)
    assert torch.allclose(spk, spk_ref, atol=1e-2)
    assert torch.allclose(mem, mem_ref, atol=1e-4)
    assert torch.allclose(u, u_ref, atol=1e-4)


def test_spike_detection_pre_matches_paper_order():
    a, b, c, d = 0.02, 0.2, -65.0, 8.0
    threshold = 30.0
    neuron = Izhikevich(a=a, b=b, c=c, d=d, threshold=threshold, spike_detection="pre", substeps=2)
    x = torch.tensor([10.0, 100.0, -5.0, 50.0])
    mem0 = neuron.c * torch.ones_like(x)
    u0 = neuron.u_init * torch.ones_like(x)
    spk, mem, u = neuron._step(x, mem0.clone(), u0.clone())
    spk_ref, mem_ref, u_ref = ref_izh_step(x, mem0, u0, a, b, c, d, threshold)
    assert torch.allclose(spk, spk_ref, atol=1e-2)
    assert torch.allclose(mem, mem_ref, atol=1e-4)
    assert torch.allclose(u, u_ref, atol=1e-4)


def ref_izh_post_step(x, mem, u, a, b, c, d, threshold, dt=1.0, substeps=2):
    h = dt / substeps
    v2, v1, bias = STANDARD_MEMBRANE
    for _ in range(substeps):
        mem = mem + h * (v2 * mem * mem + v1 * mem + bias - u + x)
        u = u + h * a * (b * mem - u)
    spk = (mem > threshold).to(x.dtype)
    mem = torch.where(spk > 0, torch.full_like(mem, c), mem)
    u = u + d * spk
    return spk, mem, u


def test_spike_detection_post_matches_pre_when_no_activity():
    pre = Izhikevich(spike_detection="pre")
    post = Izhikevich(spike_detection="post")
    x = torch.tensor([-1.0, 0.0, 1.0, -2.0])
    mem = torch.full_like(x, -70.0)
    u = torch.full_like(x, -15.0)
    spk_pre, mem_pre, u_pre = pre._step(x, mem.clone(), u.clone())
    spk_post, mem_post, u_post = post._step(x, mem.clone(), u.clone())
    assert torch.equal(spk_pre, spk_post)
    assert torch.allclose(mem_pre, mem_post)
    assert torch.allclose(u_pre, u_post)
