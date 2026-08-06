# Blowtorch-snn

Just a simple lib i'm making

## Bench vs snnTorch / Norse (LIF)

**NOTE: The benchmarks ran on** `NVIDIA GeForce RTX 3050 Laptop GPU 4G`

| library       | mode    | ms    | steps/s | peak MiB | vs blowtorch eager |
| ------------- | ------- | ----- | ------- | -------- | ------------------ |
| blowtorch LIF | eager   | 4.57  | 21,861  | 27.4     | 1.00x              |
| blowtorch LIF | compile | 0.56  | 179,091 | 37.8     | 0.12x              |
| snntorch      | eager   | 11.62 | 8,609   | 37.8     | 2.54x              |
| snntorch      | compile | 5.00  | 20,009  | 38.1     | 1.09x              |
| norse         | eager   | 11.09 | 9,017   | 38.4     | 2.42x              |
| norse         | compile | 0.69  | 145,238 | 39.6     | 0.15x              |

Even **non-compiled** blowtorch LIF (4.57 ms) beats **eager** snnTorch
(11.62 ms, ~2.5x) and Norse (11.09 ms, ~2.4x). Compiled blowtorch LIF
(0.56 ms) is ~9x faster than snnTorch compiled (5.00 ms) and still edges out
Norse compiled (0.69 ms) — all with zero custom kernels, pure Python math
through `torch.compile`. All three are measured identically: the timed call
returns the full `(T, B, F)` spike stack and it stays alive during the
measurement (blowtorch's `forward_sequence`, snnTorch's record-and-stack loop,
and the `norse.LIF` sequence module), so the peak GPU memory shown is a fair
comparison — and blowtorch also holds the least while doing so (27.4 MiB vs
37.8 MiB snnTorch / 38.4 MiB Norse).

### Every neuron, all modes

| neuron     | hidden-eager | hidden-compile | explicit-eager | explicit-compile |
| ---------- | ------------ | -------------- | -------------- | ---------------- |
| LIF        | 4.57 ms (22k/s, 27.4) | 0.56 ms (179k/s, 37.8) | 5.06 ms (20k/s, 27.8) | 0.54 ms (185k/s, 37.9) |
| QIF        | 6.51 ms (15k/s, 27.4) | 0.74 ms (135k/s, 37.8) | 6.24 ms (16k/s, 27.8) | 0.74 ms (134k/s, 37.9) |
| Izhikevich | 16.63 ms (6k/s, 28.1) | 0.70 ms (142k/s, 37.9) | 16.65 ms (6k/s, 28.3) | 0.69 ms (146k/s, 38.6) |
| AdEx       | 13.02 ms (8k/s, 28.0) | 0.72 ms (138k/s, 37.9) | 14.63 ms (7k/s, 28.3) | 0.71 ms (141k/s, 38.3) |
| SRM        | 7.52 ms (13k/s, 27.8) | 0.58 ms (171k/s, 37.9) | 7.59 ms (13k/s, 28.3) | 0.59 ms (170k/s, 38.3) |
| HH         | 49.82 ms (2k/s, 29.5) | 0.87 ms (115k/s, 38.1) | 48.66 ms (2k/s, 30.3) | 0.87 ms (115k/s, 39.0) |

### Every neuron, packed output (`pack_output=True`)

Spike tensors are bit-packed into `int32` (32 spikes/word) — 32x smaller
output. Compiled packed is as fast as compiled float *and* lighter: the
compiled scans fuse per-step so the float spike stack never lands in global
memory, dropping peak GPU memory below the compiled float variants. Eager
packed still materializes the float stack before packing, so it is the
heaviest (e.g. 250 MiB → 7.8 MiB for a T=2000, B=32, F=1024 spike stack
*returned*, but the eager run holds the float version while packing).

| neuron     | hidden-eager | hidden-compile | explicit-eager | explicit-compile |
| ---------- | ------------ | -------------- | -------------- | ---------------- |
| LIF        | 4.64 ms (22k/s, 51.0) | 0.49 ms (206k/s, 25.9) | 4.98 ms (20k/s, 51.5) | 0.48 ms (210k/s, 25.6) |
| QIF        | 7.94 ms (13k/s, 51.0) | 0.69 ms (144k/s, 33.3) | 6.89 ms (15k/s, 51.5) | 0.68 ms (148k/s, 33.1) |
| Izhikevich | 17.41 ms (6k/s, 51.5) | 0.64 ms (157k/s, 27.5) | 17.29 ms (6k/s, 52.0) | 0.63 ms (159k/s, 27.1) |
| AdEx       | 14.01 ms (7k/s, 51.1) | 0.67 ms (149k/s, 28.4) | 14.39 ms (7k/s, 52.0) | 0.65 ms (154k/s, 28.3) |
| SRM        | 9.09 ms (11k/s, 51.1) | 0.52 ms (191k/s, 26.1) | 8.33 ms (12k/s, 52.0) | 0.53 ms (189k/s, 25.9) |
| HH         | 46.73 ms (2k/s, 51.4) | 0.88 ms (114k/s, 28.0) | 47.21 ms (2k/s, 53.0) | 0.81 ms (123k/s, 27.9) |

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
