"""Lightweight performance-regression guards.

These assert *relative* timing relationships (e.g. `validate=False` is not
slower than `validate=True`) rather than absolute thresholds, so they stay
robust across machines while still catching pathological regressions. Use the
scripts under `benchmarks/` for absolute numbers against a fixed budget.
"""

import time

import torch

from blowtorch_snn import LIF

B, F, T = 8, 512, 40


def _timed(fn, reps=3):
    fn()  # warmup
    best = float("inf")
    for _ in range(reps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - start)
    return best


def test_validate_off_not_slower_than_on():
    x = torch.randn(T, B, F)
    on = LIF(beta=0.9, init_hidden=True, validate=True)
    off = LIF(beta=0.9, init_hidden=True, validate=False)
    t_on = _timed(lambda: on.forward_sequence(x))
    t_off = _timed(lambda: off.forward_sequence(x))
    assert t_off <= t_on * 1.5


def test_preallocated_not_slower_than_standard():
    x = torch.randn(T, B, F)
    standard = LIF(beta=0.9, init_hidden=True, validate=False)
    prealloc = LIF(beta=0.9, init_hidden=True, preallocated=True, validate=False)
    with torch.no_grad():
        t_s = _timed(lambda: standard.forward_sequence(x))
        t_p = _timed(lambda: prealloc.forward_sequence(x))
    assert t_p <= t_s * 2.0