# Blowtorch-snn

Just a simple lib i'm making

## Bench vs snnTorch / Norse (LIF)

**NOTE: The benchmarks ran on** `NVIDIA GeForce RTX 3050 Laptop GPU 4G`

| library   | mode    | ms    | steps/s | vs blowtorch eager |
| --------- | ------- | ----- | ------- | ------------------ |
| blowtorch | eager   | 6.28  | 15,913  | 1.00x              |
| blowtorch | compile | 0.58  | 173,234 | 0.09x              |
| snntorch  | eager   | 9.90  | 10,100  | 1.58x              |
| snntorch  | compile | 4.51  | 22,171  | 0.72x              |
| norse     | eager   | 10.48 | 9,546   | 1.67x              |
| norse     | compile | 4.31  | 23,195  | 0.69x              |

Compiled LIF is ~7.8x faster than snnTorch compiled and ~7.5x faster than
Norse compiled, with zero custom kernels — pure Python math through
`torch.compile`.

### Every neuron, compiled (best mode)

| neuron     | ms   | steps/s |
| ---------- | ---- | ------- |
| LIF        | 0.58 | 173,234 |
| SRM        | 0.66 | 151,410 |
| QIF        | 0.70 | 142,653 |
| AdEx       | 0.72 | 139,736 |
| Izhikevich | 0.72 | 138,518 |
| HH         | 0.73 | 136,440 |

> **Ratio convention**: `bench_all_vs.py` prints `framework_time / bsnn_eager_time` as the trailing `(N.Nx)` factor. A value **below 1.0 means
> the framework is faster** than blowtorch eager.

## Development

```bash
uv run pytest            # run the test suite
uv run pyright src       # type-check
uv run --group bench python benchmarks/bench_all_vs.py
```
