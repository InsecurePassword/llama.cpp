# MoE CPU-offload prefetch optimizations

This fork forward-ports two opt-in optimizations from
[`thecodacus/llama.cpp`](https://github.com/thecodacus/llama.cpp), originally developed
against the `fable5/prefetch-experts` branch.

They target large mixture-of-experts models whose expert tensors are kept in system RAM
with `--n-cpu-moe` and copied to a GPU during prompt processing.

## Controls

| Environment variable | Behavior |
| --- | --- |
| `GGML_CUDA_REGISTER_HOST=1` | Registers retained mmap-backed CPU model pages as pinned host memory, allowing direct asynchronous host-to-device transfers where the platform/backend supports it. |
| `GGML_SCHED_PREFETCH_EXPERTS=1` | Uses the default three GPU staging slots and a second backend stream to overlap full expert-tensor uploads with compute for sufficiently large MoE batches. |
| `GGML_SCHED_PREFETCH_EXPERTS=N` | Requests `N` staging slots, clamped to 8. Extra slots consume one maximum-sized expert tensor of device memory each. |

Both optimizations are disabled by default. The expert-prefetch path is selected only for
host-resident weight inputs feeding `GGML_OP_MUL_MAT_ID`, when the routing-id workload is
at least twice the number of experts. Decode behavior is otherwise unchanged.

The original benchmark used Qwen3.6-35B-A3B on an RTX 3060 12 GB with 26 MoE layers on
CPU, prompt length and batch/ubatch of 2048. It reported prompt-processing throughput of
approximately 1143 to 1880 tokens/s after enabling both options. Treat that result as
source-hardware-specific; performance and memory use must be re-measured for each model,
quantization, context length, batch size, PCIe link, CPU memory subsystem, and GPU.

## Example

```bash
GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1 \
./build/bin/llama-bench -m MODEL -ngl 99 -ncmoe 26 -p 2048 -n 0 -r 5 -b 2048 -ub 2048
```

## Source commits

- `20f5994bfeb91d24da328077c4b6095998cc9888` — register retained mmap-backed CPU weights as pinned host memory.
- `1163cb34939fe4a9cb07aec034c5954144497ae9` — overlap offloaded expert uploads with compute.
- `5f83fbbe7c668c59912a1fe09e86a0ef580406c4` — use per-layer slot sizing and fix fallback lifetime handling.

## Current limitation

The imported mmap registration helper currently performs registration only on platforms
exposing `_POSIX_MAPPED_FILES`; on other platforms it safely returns zero. The scheduler
prefetch path is independent of that mmap registration helper.
