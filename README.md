# Blowtorch-snn

Just a simple lib i'm making

## Bench vs snnTorch / Norse (LIF)

**NOTE: The benchmarks ran on** `NVIDIA GeForce RTX 3050 Laptop GPU 4G`

| library       | mode    | ms    | steps/s | peak MiB | vs blowtorch eager |
| ------------- | ------- | ----- | ------- | -------- | ------------------ |
| blowtorch LIF | eager   | 3.58  | 27,942  | 37.6     | 1.00x              |
| blowtorch LIF | compile | 0.58  | 171,214 | 37.8     | 0.16x              |
| snntorch      | eager   | 11.59 | 8,629   | 13.4     | 3.24x              |
| snntorch      | compile | 3.69  | 27,076  | 13.6     | 1.03x              |
| norse         | eager   | 9.67  | 10,338  | 14.5     | 2.70x              |
| norse         | compile | 3.56  | 28,126  | 13.8     | 0.99x              |

Even **non-compiled** blowtorch LIF (3.58 ms) beats **compiled** snnTorch
(3.69 ms) and Norse (3.56 ms). Compiled blowtorch LIF is ~6.3x faster than
snnTorch compiled and ~6.1x faster than Norse compiled, with zero custom
kernels — pure Python math through `torch.compile`. Peak GPU memory is shown
for each run (snntorch/norse spend theirs differently: their cells return
state per step, so they never hold the spike stack).

### Every neuron, all modes

| neuron     | hidden-eager | hidden-compile | explicit-eager | explicit-compile |
| ---------- | ------------ | -------------- | -------------- | ---------------- |
| LIF        | 3.58 ms (28k/s, 37.6) | 0.58 ms (171k/s, 37.8) | 3.66 ms (27k/s, 38.0) | 0.59 ms (171k/s, 38.1) |
| QIF        | 5.71 ms (18k/s, 37.9) | 0.66 ms (151k/s, 37.8) | 6.57 ms (15k/s, 38.0) | 0.63 ms (160k/s, 38.1) |
| Izhikevich | 17.63 ms (6k/s, 37.8) | 0.71 ms (141k/s, 37.9) | 16.08 ms (6k/s, 38.4) | 0.70 ms (143k/s, 38.3) |
| AdEx       | 14.78 ms (7k/s, 38.1) | 0.39 ms (257k/s, 37.9) | 14.60 ms (7k/s, 38.4) | 0.37 ms (270k/s, 38.3) |
| SRM        | 8.08 ms (12k/s, 37.8) | 0.66 ms (153k/s, 38.3) | 7.73 ms (13k/s, 38.4) | 0.67 ms (149k/s, 38.6) |
| HH         | 46.77 ms (2k/s, 38.4) | 0.56 ms (179k/s, 38.1) | 43.12 ms (2k/s, 39.1) | 0.53 ms (189k/s, 39.0) |

### Every neuron, packed output (`pack_output=True`)

Spike tensors are bit-packed into `int32` (32 spikes/word) — 32x smaller
output. Compiled packed is as fast as compiled float *and* lighter: the
compiled scans fuse per-step so the float spike stack never lands in global
memory, dropping peak GPU memory below the eager variants. Eager packed
still materializes the float stack before packing, so it is the heaviest
(e.g. 250 MiB → 7.8 MiB for a T=2000, B=32, F=1024 spike stack *returned*,
but the eager run holds the float version while packing).

| neuron     | hidden-eager | hidden-compile | explicit-eager | explicit-compile |
| ---------- | ------------ | -------------- | -------------- | ---------------- |
| LIF        | 4.57 ms (22k/s, 50.9) | 0.53 ms (188k/s, 31.6) | 4.23 ms (24k/s, 63.4) | 0.54 ms (185k/s, 31.6) |
| QIF        | 6.58 ms (15k/s, 50.9) | 0.59 ms (170k/s, 34.6) | 6.35 ms (16k/s, 63.7) | 0.59 ms (171k/s, 33.9) |
| Izhikevich | 16.85 ms (6k/s, 50.8) | 0.66 ms (151k/s, 37.8) | 19.13 ms (5k/s, 63.8) | 0.66 ms (152k/s, 37.9) |
| AdEx       | 14.90 ms (7k/s, 50.8) | 0.32 ms (316k/s, 26.1) | 12.87 ms (8k/s, 63.8) | 0.32 ms (317k/s, 25.9) |
| SRM        | 7.42 ms (13k/s, 51.1) | 0.62 ms (161k/s, 38.3) | 8.53 ms (12k/s, 63.8) | 0.59 ms (169k/s, 38.4) |
| HH         | 43.83 ms (2k/s, 51.0) | 0.49 ms (205k/s, 27.9) | 43.46 ms (2k/s, 64.5) | 0.49 ms (204k/s, 27.8) |

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
