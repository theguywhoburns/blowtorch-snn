import torch

from blowtorch_snn import LIF


def make_pair(**kwargs):
    torch.manual_seed(0)
    hidden = LIF(beta=0.9, init_hidden=True, **kwargs)
    explicit = LIF(beta=0.9, init_hidden=False, **kwargs)
    explicit.load_state_dict(hidden.state_dict())
    return hidden, explicit


def test_step_loop_matches_forward_sequence():
    _, explicit = make_pair()
    x_seq = torch.randn(8, 4, 6)

    result = explicit.forward_sequence(x_seq)
    spikes_seq = result[0]
    final_state = result[1:]

    state = explicit.initial_state_for_sequence(x_seq)
    spikes = []
    for t in range(x_seq.shape[0]):
        spk, state = explicit.step_state(x_seq[t], state)
        spikes.append(spk)

    assert torch.equal(torch.stack(spikes), spikes_seq)
    assert all(torch.equal(a, b) for a, b in zip(state, final_state))


def test_hidden_sequence_matches_explicit_sequence():
    hidden, explicit = make_pair()
    x_seq = torch.randn(8, 4, 6)

    spikes_hidden = hidden.forward_sequence(x_seq)
    spikes_explicit, _ = explicit.forward_sequence(x_seq)
    assert isinstance(spikes_hidden, torch.Tensor)
    assert isinstance(spikes_explicit, torch.Tensor)

    assert torch.equal(spikes_hidden, spikes_explicit)


def test_forward_sequence_output_matches():
    _, explicit = make_pair()
    x_seq = torch.randn(8, 4, 6)

    out = explicit.forward_sequence_output(x_seq, return_states=True)
    result = explicit.forward_sequence(x_seq)
    assert isinstance(result, tuple)
    spikes = result[0]
    final_state = result[1:]

    assert torch.equal(out.spikes, spikes)
    assert out.final_state is not None
    assert all(torch.equal(a, b) for a, b in zip(out.final_state, final_state))
    assert out.states is not None
    assert len(out.states) == x_seq.shape[0]
    assert all(torch.equal(a, b) for a, b in zip(out.states[-1], final_state))


def test_with_states_matches_step_loop():
    _, explicit = make_pair()
    x_seq = torch.randn(6, 3, 5)

    spikes, final_state, states = explicit.forward_sequence_with_states(x_seq)

    state = explicit.initial_state_for_sequence(x_seq)
    for t in range(x_seq.shape[0]):
        spk, state = explicit.step_state(x_seq[t], state)
        assert torch.equal(spikes[t], spk)
        assert states is not None
        assert all(torch.equal(a, b) for a, b in zip(states[t], state))

    assert all(torch.equal(a, b) for a, b in zip(state, final_state))


def test_fused_hook_consumer_matches_default():
    _, plain = make_pair()
    fused = LIF(beta=0.9, init_hidden=False, use_fused_sequence=True)
    fused.load_state_dict(plain.state_dict())

    x_seq = torch.randn(6, 3, 5)
    out_plain = plain.forward_sequence(x_seq)
    out_fused = fused.forward_sequence(x_seq)
    assert isinstance(out_plain, tuple)
    assert isinstance(out_fused, tuple)

    assert torch.equal(out_plain[0], out_fused[0])


# Optional: only run when you want to pay the torch.compile warmup cost.
# def test_compiled_scan_matches_reference():
#     _, explicit = make_pair()
#     explicit.compile_sequence_scan()
#     x_seq = torch.randn(6, 3, 5)
#     out = explicit.forward_sequence(x_seq)
#     ...compare against a fresh uncompiled module...
