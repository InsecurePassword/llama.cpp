# MoE CPU-offload prefetch optimizations

This fork forward-ports two opt-in optimizations from
[`thecodacus/llama.cpp`](https://github.com/thecodacus/llama.cpp), originally developed
against the `fable5/prefetch-experts` branch.

They target mixture-of-experts models whose routed-expert tensors remain in system RAM
with `--n-cpu-moe` and are copied to a GPU for computation.

## Controls

| Environment variable | Behavior |
| --- | --- |
| `GGML_CUDA_REGISTER_HOST=1` | Registers retained mmap-backed CPU model pages as pinned host memory on supported POSIX platforms. |
| `GGML_SCHED_PREFETCH_EXPERTS=1` | Requests the default three-slot GPU staging ring and a second backend stream. |
| `GGML_SCHED_PREFETCH_EXPERTS=N` | Requests `N` staging slots, clamped to 8. At least two slots must fit. |

Both optimizations are disabled by default. The scheduler path applies only to host-resident
weights feeding `GGML_OP_MUL_MAT_ID`. It uses full-tensor prefetch when the routing-ID count
is at least twice the number of experts. Ordinary one-token decode normally stays on selective
expert copying, but larger decode, continuous-batching, or speculative batches can cross the gate.

The staging ring is allocated lazily and is separate from the graph allocator's reported
compute-buffer size. Startup logs report the requested and active slot counts and total staging
allocation. Leave equivalent VRAM headroom when fitting a model.

The original benchmark used Qwen3.6-35B-A3B on an RTX 3060 12 GB with 26 MoE layers on
CPU, prompt length and batch/ubatch of 2048. It reported prompt-processing throughput of
approximately 1143 to 1880 tokens/s after enabling both options. Treat that result as
source-hardware-specific and remeasure each model, quantization, context, batch size, PCIe link,
CPU memory subsystem, and GPU.

## POSIX mmap example

```bash
GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1 ./build/bin/llama-bench -m MODEL -ngl 99 -ncmoe 26 -p 2048 -n 0 -r 5 -b 2048 -ub 2048
```

## Windows example

The mmap registration helper is not available on Windows. Use non-mmap loading so CPU expert
tensors can use the existing device host-buffer path.

```powershell
$env:GGML_SCHED_PREFETCH_EXPERTS = '1'
.\build\bin\Release\llama-bench.exe -m MODEL -ngl 99 -ncmoe 26 -p 2048 -n 0 -r 5 -b 512 -ub 128 --load-mode none
```

## Source commits

- `20f5994bfeb91d24da328077c4b6095998cc9888` - register retained mmap-backed CPU weights as pinned host memory.
- `1163cb34939fe4a9cb07aec034c5954144497ae9` - overlap offloaded expert uploads with compute.
- `5f83fbbe7c668c59912a1fe09e86a0ef580406c4` - use per-layer slot sizing and fix fallback lifetime handling.

## Current scope

The scheduler ring is bound to the first CUDA device and destination buffer type that initializes it.
Splits targeting another device or buffer type use the ordinary copy path. The imported mmap
registration helper performs registration only on platforms exposing `_POSIX_MAPPED_FILES`.
