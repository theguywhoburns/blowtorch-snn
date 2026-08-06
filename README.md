# Blowtorch-snn

Just a simple lib i'm making

## Bench vs snnTorch / Norse (LIF)

**NOTE: The benchmarks ran on** `NVIDIA GeForce RTX 3050 Laptop GPU 4G`

| library       | mode    | ms    | steps/s | vs blowtorch eager |
| ------------- | ------- | ----- | ------- | ------------------ |
| blowtorch LIF | eager   | 3.67  | 27,247  | 1.00x              |
| blowtorch LIF | compile | 0.71  | 140,623 | 0.19x              |
| snntorch      | eager   | 10.86 | 9,211   | 2.96x              |
| snntorch      | compile | 4.19  | 23,869  | 1.14x              |
| norse         | eager   | 9.93  | 10,074  | 2.70x              |
| norse         | compile | 4.53  | 22,060  | 1.24x              |

Even **non-compiled** blowtorch LIF (3.67 ms) beats **compiled** snnTorch
(4.19 ms) and Norse (4.53 ms). Compiled blowtorch LIF is ~5.9x faster than
snnTorch compiled and ~6.4x faster than Norse compiled, with zero custom
kernels — pure Python math through `torch.compile`.

### Every neuron, all modes

| neuron     | hidden-eager | hidden-compile | explicit-eager | explicit-compile |
| ---------- | ------------ | -------------- | -------------- | ---------------- |
| LIF        | 3.67 ms (27k/s)   | 0.71 ms (141k/s)   | 4.74 ms (21k/s)   | 0.57 ms (174k/s)   |
| QIF        | 6.01 ms (17k/s)   | 0.68 ms (147k/s)   | 7.41 ms (14k/s)   | 0.62 ms (161k/s)   |
| Izhikevich | 20.17 ms (5k/s)   | 0.71 ms (141k/s)   | 16.46 ms (6k/s)   | 0.72 ms (139k/s)   |
| AdEx       | 14.80 ms (7k/s)   | 0.38 ms (265k/s)   | 15.21 ms (7k/s)   | 0.37 ms (267k/s)   |
| SRM        | 7.45 ms (13k/s)   | 0.68 ms (147k/s)   | 8.28 ms (12k/s)   | 0.68 ms (146k/s)   |
| HH         | 43.79 ms (2k/s)   | 0.54 ms (187k/s)   | 42.88 ms (2k/s)   | 0.54 ms (187k/s)   |

### Every neuron, packed output (`pack_output=True`)

Spike tensors are bit-packed into `int32` (32 spikes/word) — 32x smaller
output. Compiled packed is as fast as compiled float, while the returned
tensor shrinks by 32x (e.g. 250 MiB → 7.8 MiB for a T=2000, B=32, F=1024
spike stack).

| neuron     | hidden-eager | hidden-compile | explicit-eager | explicit-compile |
| ---------- | ------------ | -------------- | -------------- | ---------------- |
| LIF        | 4.64 ms (22k/s)   | 0.54 ms (187k/s)   | 5.01 ms (20k/s)   | 0.52 ms (191k/s)   |
| QIF        | 6.68 ms (15k/s)   | 0.60 ms (166k/s)   | 7.62 ms (13k/s)   | 0.60 ms (166k/s)   |
| Izhikevich | 19.68 ms (5k/s)   | 0.66 ms (151k/s)   | 17.34 ms (6k/s)   | 0.65 ms (154k/s)   |
| AdEx       | 14.72 ms (7k/s)   | 0.32 ms (310k/s)   | 13.65 ms (7k/s)   | 0.31 ms (318k/s)   |
| SRM        | 9.34 ms (11k/s)   | 0.61 ms (165k/s)   | 7.85 ms (13k/s)   | 0.60 ms (166k/s)   |
| HH         | 44.84 ms (2k/s)   | 0.50 ms (201k/s)   | 43.67 ms (2k/s)   | 0.49 ms (204k/s)   |

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
