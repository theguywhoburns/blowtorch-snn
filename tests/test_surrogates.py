import torch
import pytest

from blowtorch_snn.surrogate import (
    heaviside,
    fast_sigmoid,
    atan,
    triangular,
    sigmoid,
    gaussian,
    multi_gaussian,
    rectangular,
    straight_through,
    default_spike_grad,
)


@pytest.mark.parametrize("surrogate", [heaviside(), straight_through()])
def test_step_forward(surrogate):
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 3.0])
    spk = surrogate(x)
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
    assert torch.equal(spk, expected)
    assert spk.dtype == x.dtype


def test_heaviside_and_straight_through_identical():
    x = torch.tensor([-1.0, 0.5], requires_grad=True)
    a = heaviside()(x)
    b = straight_through()(x)
    assert torch.equal(a, b)
    a.sum().backward()
    assert x.grad is not None
    assert torch.equal(x.grad, torch.ones_like(x.grad))


def test_step_forward_all_surrogates():
    x = torch.tensor([-2.0, 0.5, 3.0])
    for surrogate in [
        heaviside(),
        fast_sigmoid(),
        atan(),
        triangular(),
        sigmoid(),
        gaussian(),
        multi_gaussian(),
        rectangular(),
        straight_through(),
    ]:
        spk = surrogate(x)
        assert torch.equal(spk, torch.tensor([0.0, 1.0, 1.0]))


def test_backward_flows():
    x = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)
    spk = fast_sigmoid(slope=25.0)(x)
    spk.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_backward_all_new_surrogates_finite():
    for surrogate in [
        sigmoid(slope=25.0),
        gaussian(sigma=1.0),
        multi_gaussian(),
        rectangular(width=1.0),
    ]:
        x = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)
        spk = surrogate(x)
        spk.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_backward_rectangular_zero_outside_window():
    x = torch.tensor([-3.0, 0.0, 3.0], requires_grad=True)
    spk = rectangular(width=1.0)(x)
    spk.sum().backward()
    assert x.grad is not None
    assert x.grad[0] == 0.0
    assert x.grad[2] == 0.0


def test_backward_heaviside_is_identity():
    x = torch.tensor([-1.0, 2.0], requires_grad=True)
    spk = heaviside()(x)
    spk.sum().backward()
    assert x.grad is not None
    assert torch.equal(x.grad, torch.ones_like(x.grad))


def test_backward_atan_finite():
    x = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)
    spk = atan(alpha=2.0)(x)
    spk.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_backward_triangular_supports_zero_outside_threshold():
    x = torch.tensor([-10.0, 0.0, 10.0], requires_grad=True)
    spk = triangular(threshold=1.0)(x)
    spk.sum().backward()
    assert x.grad is not None
    assert x.grad[0] == 0.0
    assert x.grad[2] == 0.0


def test_default_spike_grad_is_callable():
    x = torch.tensor([0.5])
    spk = default_spike_grad(x)
    assert spk.shape == x.shape
