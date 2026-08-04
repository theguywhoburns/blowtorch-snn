import torch
import math
from typing import Callable

SpikeGrad = Callable[[torch.Tensor], torch.Tensor]


class _Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, surrogate_fn):
        ctx.save_for_backward(x)
        ctx.surrogate_fn = surrogate_fn
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (x,) = ctx.saved_tensors
        return ctx.surrogate_fn(x) * grad_outputs[0], None


def _heaviside(x: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(x)


def _fast_sigmoid(x: torch.Tensor, slope: float) -> torch.Tensor:
    return 1 / (slope * torch.abs(x) + 1.0) ** 2


def _atan(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return (alpha / 2) / (1 + (math.pi / 2 * alpha * x).pow(2))


def _triangular(x: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.where(
        x.abs() < threshold, 1 - x.abs() / threshold, torch.zeros_like(x)
    )


def _sigmoid(x: torch.Tensor, slope: float) -> torch.Tensor:
    return torch.sigmoid(slope * x)


def _gaussian(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-0.5 * (x / sigma).pow(2)) / (sigma * math.sqrt(2 * math.pi))


def _multi_gaussian(
    x: torch.Tensor,
    sigma: float,
    height: float,
    separation: float,
) -> torch.Tensor:
    primary = torch.exp(-0.5 * (x / sigma).pow(2))
    secondary = torch.exp(-0.5 * ((x - separation) / sigma).pow(2))
    return (primary + height * secondary) / (
        (1 + height) * sigma * math.sqrt(2 * math.pi)
    )


def _rectangular(x: torch.Tensor, width: float) -> torch.Tensor:
    return torch.where(
        x.abs() < width / 2, torch.full_like(x, 1 / width), torch.zeros_like(x)
    )


def heaviside() -> SpikeGrad:
    """Unit-gradient surrogate: hard Heaviside forward, identity backward
    (the gradient passes through unchanged)."""
    return lambda x: _Spike.apply(x, _heaviside)


def fast_sigmoid(slope: float = 25.0) -> SpikeGrad:
    return lambda x: _Spike.apply(x, lambda x: _fast_sigmoid(x, slope))


def atan(alpha: float = 2.0) -> SpikeGrad:
    return lambda x: _Spike.apply(x, lambda x: _atan(x, alpha))


def triangular(threshold: float = 1.0) -> SpikeGrad:
    return lambda x: _Spike.apply(x, lambda x: _triangular(x, threshold))


def sigmoid(slope: float = 25.0) -> SpikeGrad:
    """Vanilla logistic surrogate gradient: ``sigmoid(slope * x)``."""
    return lambda x: _Spike.apply(x, lambda x: _sigmoid(x, slope))


def gaussian(sigma: float = 1.0) -> SpikeGrad:
    """Gaussian surrogate gradient, normalized so it integrates to one."""
    return lambda x: _Spike.apply(x, lambda x: _gaussian(x, sigma))


def multi_gaussian(
    sigma: float = 1.0,
    height: float = 1.25,
    separation: float = 1.0,
) -> SpikeGrad:
    """Two-gaussian surrogate: a tall primary lobe plus a weaker offset one."""
    return lambda x: _Spike.apply(
        x, lambda x: _multi_gaussian(x, sigma, height, separation)
    )


def rectangular(width: float = 1.0) -> SpikeGrad:
    """Rectangular window of ``1 / width`` inside ``[-width/2, width/2]``."""
    return lambda x: _Spike.apply(x, lambda x: _rectangular(x, width))


def straight_through() -> SpikeGrad:
    """Alias of :func:`heaviside` (identical unit-gradient surrogate)."""
    return heaviside()


default_spike_grad: SpikeGrad = atan()
