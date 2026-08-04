from typing import Optional

import torch

from ..base import SpikingModule, Tensor


def lif_sequence_scan_reference(
    module: SpikingModule,
    x_seq: Tensor,
    state: Optional[tuple[Tensor, ...]] = None,
) -> Tensor | tuple[Tensor, ...]:
    """Reference fallback for future fused implementations."""
    return module._reference_sequence_scan(x_seq, state)


def lif_sequence_scan_triton(
    module: SpikingModule,
    x_seq: Tensor,
    state: Optional[tuple[Tensor, ...]] = None,
) -> Tensor | tuple[Tensor, ...]:
    """Placeholder for a real Triton/CUDA implementation.

    Expected contract:
    - explicit mode: return ``(spikes, *final_state)``
    - hidden mode: return ``spikes``
    """
    raise NotImplementedError("TODO: implement fused LIF sequence scan")


def install_lif_fused_sequence(
    module: SpikingModule,
    backend: str = "triton",
) -> None:
    """Attach a fused sequence implementation to a LIF module."""
    if backend == "reference":
        module._fused_forward_sequence = lambda x_seq, state=None: lif_sequence_scan_reference(
            module, x_seq, state
        )
    elif backend == "triton":
        module._fused_forward_sequence = lambda x_seq, state=None: lif_sequence_scan_triton(
            module, x_seq, state
        )
    else:
        raise ValueError(f"Unknown backend {backend!r}")

    module.use_fused_sequence = True