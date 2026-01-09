# Project 1: Cloud GPU Stack Validation & Performance Regression Suite

---

## Objective

**Hardware Target:** 2x NVIDIA H200 NVL (NVLink Enabled)

To engineer a robust validation and optimization pipeline for high-performance inference clusters. This project shifts left from "model deployment" to "platform certification", ensuring that the underlying hardware physics (NVLink, PCIe, Compute) are fully saturated before application-level tuning begins.

The scope includes:

1. **Hardware Validation:** Automated snapshots of drivers, topology, and bandwidth.
2. **Physics Benchmarking:** Establishing theoretical peak limits for GEMM (Compute) and NCCL (Communication)
3. **Inference Optimization:** Tuning `vLLM` serving parameters (Batching, KV Cache) for H200 specifics.
4. **Forensics** Root-cause analysis of latency spikes using Nsight Systems profiling.

---

## Tech Stack

- **Hardware:** NVIDIA H200 (141GB VRAM), NVLink Switch.
- **Core Infrastructure:** Python 3.12, PyTorch (CUDA 12.x), NVIDIA NCCL.
- **Serving Engine:** vLLM (with Ray for Tensor Parallelism).
- **Observability:** NVIDIA Nsight Systems, `nvidia-smi`, Python `logging`.
- **DevOps:** Bash Scripting for load generation & environment isolation

---

## Key Achievements & Roadmap

### Phase 1: Infrastructure Validation

- [x] **Base Snapshot:** Automated capture of Driver/CUDA/Topology versions to detect drift
- [x] **GPU Compute Benchmark(GEMM / P2P / NCCL):** Establish & compare theoretical and measured performance for Compute (GEMM) and Bandwidth (NCCL)
- [x] **Interconnect P2P & NCCL Benchmark:**
- [x] **Application Sanity:** Successfully initialized vLLM on Qwen with Tensor Parallelism (TP=2) on H200s

### Phase 2: Baseline & Tooling

- [x] **Load Generator:** Built concurrent request script to measure TTFT & TPOT (Time per Output Token).
- [x] **Resource Monitor:** Implemented 100ms-resolution logging for VRAM and GPU Utilization
- [x] **Baseline Establishment:** Recorded control group metrics (Concurrency 1 vs 32) before optimization

### Phase 3: Stress Test & Optimization

- [x] **KV Cache Saturation:** Determined maximum context length before OOM
- [] **Batching Strategy:** Tuned `max_num_steps` and `continuous_batching` for optimal throughput / latency trade-off
- [] **Precision Sweep:** Quantified speedup / accuracy delta between FP16 & BF16 on H200.

### Phase 4: Forensics

- [] **NSight Profiling:** Captured traces to identify CPU-bound gaps in GPU timeline.
- [] **Perf Story:** Documented specific optimization win (+20% throughput via config change)
- [] **Regression Report:** Produced final `Report Card` validating cluster for production traffic

---

## Evidence

### Phase 1 Evidence: Infrastructure Baseline (Snapshot)

**1. System Stack Snapshot:**

- **GPU Model**: 2x NVIDIA H200 NVL (139.8GB each)
- **Driver Version**: 580.105.08 (up to date)
- **CUDA Version**: 13.0 (match)
- **PyTorch version**: 2.9.0+cu128 (compatible, CUDA 13.0)
- **Topology**: Full NVLink (18 Links / GPU)

**2. Topology Matrix**
`nvidia-smi topo -m`

```text
      GPU0    GPU1    GPU2    GPU3
GPU0   X      NV6     NV6     NV6
GPU1  NV6      X      NV6     NV6
GPU2  NV6     NV6      X      NV6  <-- GPU 2 talks to 3 via NVLink
GPU3  NV6     NV6     NV6      X   <-- GPU 3 talks to 2 via NVLink
```

**3. NVLink Health Check**
`nvidia-smi nvlink -s`

- Link Status (0-17) @ 26.562 GB/s
- P2P Status: Bi-Directional P2P access enabled

**4. Compute Performance (GEMM)**

- **Peak Performance**: 720 TFLOPs (bf16, Square 8k)
- **Real World Performance**: ~620 TFLOPs (MLP / Projection shapes)
- **Status**: ~73% of theoretical peak

**5. Interconnect health (P2P & NCCL)**

- **P2P Copy**: 133.26 GBps (2x faster than typical PCIe)
- **NCCL (Small)**: 14.29 GBps (Latency-dominated) (4 MB)
- **NCCL (Large)**: 109.4 GBps (Bandwidth-saturated) (1 GB)

### Phase 2 Evidence: Baseline Throughput (Qwen 1.5B)

\*Measured on 2x H200 NVL using vLLM (Offline mode)

**2.1 Load Generator Throughput for Offline metric**
--- Baseline Report Card (Load Generator) ---
| Concurrency | Throughput (tok/s) |
|-------------|--------------------|
| 1 | 365.43 |
| 16 | 5941.80 |
| 32 | 10898.08 |
| 64 | 18745.13 |
| 128 | 29088.76 |
| 256 | 40503.24 |
| 512 | 47490.86 |
| 1024 | 50148.57 |
| 2048 | 51166.16 |
| 4196 | 51551.06 |

**Analysis:** H200 exhibit near-perfect linear scaling up to concurrency 32 on this small model. This confirms GPU was severely under-utilized at low batch sizes.

**2.2 Peak GPU Utilization (Resource Monitor)**

- **Peak GPU Utilization**: 100% (compute-bound)

### Phase 3 Evidence: Stress Test & Optimization (Compute vs Memory)

**3.1 KV Cache Saturation (Context Limit)**

- **Max Stable Content:** > 1,048,576 tokens (1M+)
- 2 H200 can hold >1M tokens of KV Cache for Qwen-1.5B

**3.2 Batching Strategy Sweep**
---Optimization Report ---
| Max Batch | Throughput (tok/s) | Gain |
|-----------|--------------------| --- |
| 16 | 2520.57 | Baseline
| 64 | 8927.50 | + 254%
| 256 | 25579.36 | + 186%
| 512 | 38077.25 | + 48%
| 2048 | 38248.70 | + 0.4%

- 512 request is sweetspot
- increasing to 2048 (x4) only increase speed by 0.4%, not worth memory cost

**3.3 Precision Sweep (H200 Native Support)**
| Dtype | Throughput | Analysis |
| :--- | :--- | :--- |
| float16 | 37,646 tok/s | Baseline |
| **bfloat16** | **37,810 tok/s** | **+0.4% (Preferred for Stability)** |

- Get numerical stability of bf16 without speed penalty
