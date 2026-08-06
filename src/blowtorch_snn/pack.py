"""Bit-packed spike representation for memory efficiency.

Spikes are binary (``0.0``/``1.0``), so the spike tensor is the only part of a
layer's output that can be compressed losslessly: 32 feature indices pack into
one ``int32`` word, a 32x reduction vs. float32 storage.

Packing is non-differentiable, so it is meant for inference and for storing
long spike sequences, not for the training forward pass (the surrogate
gradient needs the float spikes).
"""

from typing import Optional

import torch

Tensor = torch.Tensor

_WORD = 32

# Single-bit masks [1, 2, 4, ..., 2**31], held as an immutable module constant:
# Dynamo treats it as a constant and bakes it into the compiled graph once, so
# calling code traces a single ``.to(device)`` instead of rebuilding the array
# every call. This is not the mutable, device-keyed cache that caused guard
# invalidations earlier -- an immutable constant carries no guards. The last
# entry is -2**31 (int32 wraps 1 << 31), which is the correct bit pattern; the
# python int 2**31 would overflow on construction.
_BIT_MASKS_BASE = torch.tensor(
    [1 << i for i in range(_WORD - 1)] + [-(2 ** (_WORD - 1))], dtype=torch.int32
)


def _bit_masks(device: torch.device) -> Tensor:
    return _BIT_MASKS_BASE.to(device)


def pack_spikes(spk: Tensor) -> Tensor:
    """Pack a binary spike tensor into an ``int32`` bitfield (32 spikes/word).

    The last axis is packed LSB-first: bit ``i`` of each word holds feature
    index ``i`` (bit 0 is the lowest feature). A trailing partial word is
    zero-padded, so ``features`` must be a multiple of 32 for a fully dense
    packing. Non-zero values are treated as spikes.

    Args:
        spk: float/uint8 tensor of ``0``/``1`` values, shape ``(..., F)``.

    Returns:
        ``int32`` tensor, shape ``(..., ceil(F / 32))``.
    """
    features = spk.shape[-1]
    words = (features + _WORD - 1) // _WORD
    pad = words * _WORD - features

    bits = (spk != 0).to(torch.uint8)
    if pad:
        bits = torch.nn.functional.pad(bits, (0, pad))
    bits = bits.reshape(*bits.shape[:-1], words, _WORD).to(torch.int32)

    masks = _bit_masks(spk.device)
    # int32 accumulation: the two's-complement wrap is the bitfield arithmetic.
    return (bits * masks).sum(dim=-1, dtype=torch.int32)


def unpack_spikes(
    packed: Tensor,
    features: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Invert :func:`pack_spikes`, recovering the 0/1 floating spike tensor.

    Args:
        packed: ``int32`` bitfield as produced by :func:`pack_spikes`.
        features: number of spike positions to recover. Required when the
            packed feature count was not a multiple of 32; otherwise defaults
            to ``packed.shape[-1] * 32``.
        dtype: output dtype (default ``torch.get_default_dtype()``).

    Returns:
        Float tensor of ``0.0``/``1.0``, shape ``(..., features)``.
    """
    words = packed.shape[-1]
    width = words * _WORD
    if features is None:
        features = width

    masks = _bit_masks(packed.device)
    bits = (packed[..., :, None] & masks) != 0
    spk = bits.reshape(*packed.shape[:-1], width)
    if dtype is None:
        dtype = torch.get_default_dtype()
    return spk[..., :features].to(dtype)
