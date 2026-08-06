import torch
import pytest

from blowtorch_snn import (
    LIF,
    pack_spikes,
    unpack_spikes,
)


def test_round_trip_multiple_of_32():
    spk = torch.randint(0, 2, (10, 5, 1024)).to(torch.float32)
    packed = pack_spikes(spk)
    assert packed.dtype == torch.int32
    assert packed.shape == (10, 5, 32)
    assert torch.equal(unpack_spikes(packed), spk)


def test_round_trip_partial_word():
    spk = torch.randint(0, 2, (3, 7, 20)).to(torch.float32)
    packed = pack_spikes(spk)
    assert packed.shape == (3, 7, 1)
    recovered = unpack_spikes(packed, features=20)
    assert torch.equal(recovered, spk)


def test_packed_memory_is_32x_smaller():
    spk = torch.randn(100, 32, 1024).to(torch.float32)
    packed = pack_spikes(spk)
    assert packed.dtype == torch.int32
    assert packed.numel() * packed.element_size() == spk.numel() * spk.element_size() // 32


def test_non_binary_values_treated_as_spikes():
    spk = torch.tensor([0.0, 2.5, -1.0, 0.0])
    packed = pack_spikes(spk)
    assert torch.equal(unpack_spikes(packed, features=4), torch.tensor([0.0, 1.0, 1.0, 0.0]))


def test_unpack_dtype_defaults_to_default():
    spk = torch.ones(32, dtype=torch.float16)
    assert unpack_spikes(pack_spikes(spk)).dtype == torch.get_default_dtype()


def test_forward_sequence_pack_output_hidden():
    x = torch.randn(50, 16, 256)
    float_lif = LIF(beta=0.9, init_hidden=True, validate=False)
    packed_lif = LIF(beta=0.9, init_hidden=True, validate=False, pack_output=True)
    out = float_lif.forward_sequence(x)
    packed = packed_lif.forward_sequence(x)
    assert isinstance(out, torch.Tensor)
    assert isinstance(packed, torch.Tensor)
    assert packed.dtype == torch.int32
    assert packed.shape == (50, 16, 8)
    assert torch.equal(unpack_spikes(packed), out)


def test_forward_sequence_pack_output_explicit():
    x = torch.randn(50, 16, 256)
    float_lif = LIF(beta=0.9, init_hidden=False, validate=False)
    packed_lif = LIF(beta=0.9, init_hidden=False, validate=False, pack_output=True)
    out = float_lif.forward_sequence(x)
    assert isinstance(out, tuple)
    packed_out = packed_lif.forward_sequence(x)
    assert isinstance(packed_out, tuple)
    packed, *state = packed_out
    assert packed.dtype == torch.int32
    assert torch.equal(unpack_spikes(packed), out[0])
    for a, b in zip(state, out[1:]):
        assert torch.equal(a, b)
