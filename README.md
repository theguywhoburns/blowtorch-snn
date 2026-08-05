# Blowtorch-snn

Just a simple lib i'm making

## Bench vs snnTorch / Norse (LIF)

**NOTE: The benchmarks ran on** `NVIDIA GeForce RTX 3050 Laptop GPU 4G`

| library       | mode    | ms    | steps/s | vs blowtorch eager |
| ------------- | ------- | ----- | ------- | ------------------ |
| blowtorch LIF | eager   | 3.36  | 29,761  | 1.00x              |
| blowtorch LIF | compile | 0.67  | 148,338 | 0.20x              |
| snntorch      | eager   | 10.73 | 9,319   | 3.19x              |
| snntorch      | compile | 3.96  | 25,241  | 1.18x              |
| norse         | eager   | 10.20 | 9,800   | 3.04x              |
| norse         | compile | 4.25  | 23,525  | 1.27x              |

Even **non-compiled** blowtorch LIF (3.36 ms) beats **compiled** snnTorch
(3.96 ms) and Norse (4.25 ms). Compiled blowtorch LIF is ~6.9x faster than
snnTorch compiled and ~7.4x faster than Norse compiled, with zero custom
kernels — pure Python math through `torch.compile`.

### Every neuron, all modes

| neuron     | hidden-eager | hidden-compile | explicit-eager | explicit-compile |
| ---------- | ------------ | -------------- | -------------- | ---------------- |
| LIF        | 3.36 ms      | 0.67 ms        | 4.53 ms        | 0.57 ms          |
| QIF        | 6.20 ms      | 0.64 ms        | 5.90 ms        | 0.63 ms          |
| Izhikevich | 16.01 ms     | 0.72 ms        | 16.14 ms       | 0.70 ms          |
| AdEx       | 13.43 ms     | 0.38 ms        | 12.21 ms       | 0.36 ms          |
| SRM        | 7.16 ms      | 0.67 ms        | 6.85 ms        | 0.65 ms          |
| HH         | 42.97 ms     | 0.54 ms        | 45.67 ms       | 0.54 ms          |

> **Ratio convention**: `bench_all_vs.py` prints `framework_time / bsnn_eager_time` as the trailing `(N.Nx)` factor. A value **below 1.0 means
> the framework is faster** than blowtorch eager. The base is the
> `blowtorch LIF hidden-eager` row (the only row every framework can be
> compared against).

## Development

```bash
uv run pytest            # run the test suite
uv run pyright src       # type-check
uv run --group bench python benchmarks/bench_all_vs.py
```
