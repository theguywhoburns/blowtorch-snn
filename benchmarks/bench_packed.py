"""Memory and throughput for packed vs. float spike outputs.

Run from the repo root:

    uv run python benchmarks/bench_packed.py

Two sections on CUDA (CPU-only run if no GPU):

1. Memory + eager throughput at large ``T``: with ``pack_output`` (constructor
   flag) spikes are bit-packed per step *inside* the scan, so the full float
   ``(time, batch, features)`` spike stack is never materialized.
2. Compiled throughput at compile-friendly ``T``: ``fast_sequence_()`` of
   float vs. packed. Note the packed scan takes much longer to compile once
   (the int32/uint8 kernels are new graph ops) but is fast at runtime.
"""

import time

import torch

import blowtorch_snn as bsnn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEM_T, BATCH, FEATURES, REPS = 2000, 32, 1024, 10
COMPILE_T = 100


def _timeit(fn):
    for _ in range(3):
        with torch.no_grad():
            fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(REPS):
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.no_grad():
            out = fn()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - start)
    peak = torch.cuda.max_memory_allocated() if DEVICE == "cuda" else 0
    return best, out, peak


def memory_section():
    T = MEM_T
    print(f"-- memory / eager  (time={T} batch={BATCH} features={FEATURES}) --")
    x_seq = torch.randn(T, BATCH, FEATURES, device=DEVICE)
    results = {}
    for label, pack in (("float", False), ("packed", True)):
        torch._dynamo.reset()
        lif = bsnn.LIF(
            beta=0.9, init_hidden=True, validate=False, pack_output=pack
        ).to(DEVICE)
        ft, out, peak = _timeit(lambda: lif.forward_sequence(x_seq))
        results[label] = (T / ft, out.numel() * out.element_size(), peak)
        print(
            f"{label:<8} output {results[label][1] / 2**20:8.2f} MiB   "
            f"{results[label][0]:9,.0f} steps/s   peak {peak / 2**20:6.0f} MiB"
        )
    f_steps, f_bytes, f_peak = results["float"]
    p_steps, p_bytes, p_peak = results["packed"]
    print(f"output compression: {f_bytes / p_bytes:.1f}x")
    print(f"peak compression:   {f_peak / p_peak:.1f}x")
    print(f"eager throughput:   {f_steps / p_steps:.2f}x float/packed")
    print()


def compiled_section():
    T = COMPILE_T
    print(f"-- compiled  (time={T} batch={BATCH} features={FEATURES}) --")
    x_seq = torch.randn(T, BATCH, FEATURES, device=DEVICE)
    for label, pack in (("float", False), ("packed", True)):
        torch._dynamo.reset()
        lif = bsnn.LIF(
            beta=0.9, init_hidden=True, validate=False, pack_output=pack
        ).to(DEVICE)
        lif.fast_sequence_()
        with torch.no_grad():
            t0 = time.perf_counter()
            lif.forward_sequence(x_seq)
            torch.cuda.synchronize()
            compile_ms = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            for _ in range(10):
                lif.forward_sequence(x_seq)
            torch.cuda.synchronize()
            steady_ms = (time.perf_counter() - t0) / 10 * 1e3
        print(
            f"{label:<8} compile {compile_ms:8.0f} ms   steady {steady_ms:8.3f} ms/call"
        )
    print()


def main():
    print(f"device={DEVICE} batch={BATCH} features={FEATURES}")
    memory_section()
    compiled_section()


if __name__ == "__main__":
    main()
