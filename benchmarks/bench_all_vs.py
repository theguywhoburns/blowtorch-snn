"""Benchmark all blowtorch-snn neurons against snnTorch and Norse.

Run from the repo root:

    uv run --group bench python benchmarks/bench_all_vs.py

Each blowtorch neuron runs 100 timesteps through ``forward_sequence`` in
eight variants:
  hidden-eager, hidden-compile, explicit-eager, explicit-compile,
  plus the same four with ``pack_output`` (spikes bit-packed into int32).

norse/snntorch only ship a LIF, so the cross-framework comparison is LIF-only
(eager vs. torch.compile of their cell). Any framework that is missing is
skipped with a note.

All measurements run under ``torch.no_grad()``.
"""

import time

import torch

import blowtorch_snn as bsnn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH, FEATURES, T, REPS, WARMUP = 32, 1024, 100, 7, 3
BETA = 0.9


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _timeit(fn):
    for _ in range(WARMUP):
        fn()
        _sync()
    best = float("inf")
    for _ in range(REPS):
        _sync()
        start = time.perf_counter()
        fn()
        _sync()
        best = min(best, time.perf_counter() - start)
    return best


NEURONS = {
    "LIF": lambda **kw: bsnn.LIF(beta=BETA, **kw),
    "QIF": lambda **kw: bsnn.QIF(beta=0.01, **kw),
    "Izhikevich": lambda **kw: bsnn.Izhikevich(**kw),
    "AdEx": lambda **kw: bsnn.AdEx(beta=BETA, **kw),
    "SRM": lambda **kw: bsnn.SRM(beta=BETA, **kw),
    "HH": lambda **kw: bsnn.HH(**kw),
}


def bench_blowtorch(name: str, variant: str) -> float:
    hidden = variant.startswith("hidden")
    compiled = "compile" in variant
    packed = "packed" in variant

    t = time.perf_counter()
    neuron = NEURONS[name](
        size=FEATURES,
        init_hidden=hidden,
        validate=False,
        pack_output=packed,
    ).to(DEVICE)
    build = time.perf_counter() - t

    compile_t = 0.0
    if compiled:
        # All neurons compile the same shared ``_reference_sequence_scan``
        # code object, so torch.compile's per-code-object cache (8 entries by
        # default) is exhausted once a few distinct neurons/variants compile in
        # one process -- later neurons silently fall back to eager. Reset dynamo
        # so each compiled run gets a clean, unbounded-by-cache budget and every
        # neuron is genuinely measured as compiled, not silently eager. This is
        # cheap because ``fast_sequence_`` compiles with ``mode="default"``
        # (~2s per variant); CUDA-graph capture made this unusably slow, hence
        # ``reduce-overhead`` is never used here.
        torch._dynamo.reset()
        t = time.perf_counter()
        neuron.fast_sequence_(mode="default")
        compile_t = time.perf_counter() - t

    t = time.perf_counter()
    x_seq = torch.randn(T, BATCH, FEATURES, device=DEVICE)
    state = None
    if not hidden:
        state = neuron.initial_state_for_sequence(x_seq)
    setup = time.perf_counter() - t

    def run():
        with torch.no_grad(), bsnn.no_validation():
            if hidden:
                return neuron.forward_sequence(x_seq)
            return neuron.forward_sequence(x_seq, state)

    t = time.perf_counter()
    best = _timeit(run)
    measure = time.perf_counter() - t
    print(
        f"  [{name:>11} {variant:<24}] build={build*1e3:6.1f}ms "
        f"compile={compile_t*1e3:8.1f}ms setup={setup*1e3:6.1f}ms "
        f"measure={measure*1e3:8.1f}ms best={best*1e3:7.3f}ms",
        flush=True,
    )
    return best


def bench_snntorch(compiled: bool) -> float:
    import snntorch as snn

    layer = snn.Leaky(beta=BETA)
    x_seq = torch.randn(T, BATCH, FEATURES, device=DEVICE)
    mem = torch.zeros(BATCH, FEATURES, device=DEVICE)

    if compiled:
        cell = torch.compile(snn.Leaky(beta=BETA).to(DEVICE))

    def step(t, mem):
        return layer(x_seq[t], mem)

    if compiled:
        def run():
            nonlocal mem
            with torch.no_grad():
                for t in range(T):
                    _, mem = cell(x_seq[t], mem)
        return _timeit(run)

    def run():
        nonlocal mem
        with torch.no_grad():
            for t in range(T):
                _, mem = layer(x_seq[t], mem)
    return _timeit(run)


def bench_norse(compiled: bool) -> float:
    import norse.torch as norse
    from norse.torch.module.lif import LIFParameters

    layer = norse.LIFCell(p=LIFParameters(alpha=torch.as_tensor(BETA)))
    x_seq = torch.randn(T, BATCH, FEATURES, device=DEVICE)
    state = layer.initial_state(x_seq[0])

    cell = torch.compile(layer) if compiled else layer

    def run():
        nonlocal state
        with torch.no_grad():
            for t in range(T):
                _, state = cell(x_seq[t], state)
    return _timeit(run)


def report(name, total, base=None):
    ms = total * 1e3
    line = f"{name:<34} {ms:>10.3f} ms  {T / total:>10.0f} steps/s"
    if base is not None and base:
        line += f"  ({total / base:>5.2f}x)"
    print(line)


def main():
    print(f"device={DEVICE} batch={BATCH} features={FEATURES} steps={T}")
    results = {}
    variants = [
        "hidden-eager",
        "hidden-compile",
        "explicit-eager",
        "explicit-compile",
        "hidden-eager-packed",
        "hidden-compile-packed",
        "explicit-eager-packed",
        "explicit-compile-packed",
    ]

    for name in NEURONS:
        for variant in variants:
            key = f"bsnn {name:<10} {variant}"
            try:
                results[key] = bench_blowtorch(name, variant)
                report(key, results[key])
            except Exception as exc:  # noqa: BLE001 - surface failures loudly
                print(f"{key} ERROR: {exc}")
        print()

    base = results.get("bsnn LIF        hidden-eager")
    for label, fn in (
        ("snntorch eager", lambda: bench_snntorch(False)),
        ("snntorch compile", lambda: bench_snntorch(True)),
        ("norse eager", lambda: bench_norse(False)),
        ("norse compile", lambda: bench_norse(True)),
    ):
        key = f"{label}"
        try:
            results[key] = fn()
            report(key, results[key], base)
        except ImportError:
            print(f"{key} SKIPPED (not installed)")
        except Exception as exc:  # noqa: BLE001
            print(f"{key} ERROR: {exc}")


if __name__ == "__main__":
    main()
