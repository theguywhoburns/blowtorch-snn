import pytest
import torch

from blowtorch_snn import LIF


def test_lif_hidden_and_explicit_match():
    torch.manual_seed(0)

    lif_hidden = LIF(beta=0.9, init_hidden=True)
    lif_explicit = LIF(beta=0.9, init_hidden=False)
    lif_explicit.load_state_dict(lif_hidden.state_dict())

    x_seq = torch.randn(5, 2, 4)
    state = lif_explicit.initial_state_for_sequence(x_seq)

    for t in range(x_seq.shape[0]):
        spk_hidden = lif_hidden(x_seq[t])
        spk_explicit, state = lif_explicit.step_state(x_seq[t], state)
        assert torch.equal(spk_hidden, spk_explicit)


def test_preallocated_training_raises():
    lif = LIF(beta=0.9, init_hidden=True, preallocated=True)
    lif.train()

    with pytest.raises(RuntimeError):
        lif(torch.randn(2, 4))
