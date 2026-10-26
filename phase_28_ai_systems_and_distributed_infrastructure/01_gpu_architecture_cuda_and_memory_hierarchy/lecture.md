# Topic 01: GPU Architecture, CUDA & Memory Hierarchy

## 1. Introduction & 12-Year-Old Intuition

Imagine you need to move 1,000 heavy cardboard boxes from a warehouse across town to a school gymnasium.

You have two choices of transport:
1. **The Ferrari (The CPU)**: A single, ultra-fast racecar driven by an Olympic Formula 1 driver. It can carry only **one box at a time**, but it drives at 200 miles per hour, swerves around traffic seamlessly, makes split-second decisions at roundabouts, and delivers that single box in 2 minutes flat. If your task is to deliver one urgent secret letter through winding, unpredictable mountain roads, the Ferrari is unbeatable.
2. **The Cargo Train (The GPU)**: A massive freight train with **1,000 flatbed cars** moving side-by-side on parallel tracks. It drives at a modest 25 miles per hour. It cannot turn sharp corners, and it takes 30 minutes to reach top speed. But on a single trip, it moves **all 1,000 boxes at once**. While the Ferrari takes $1{,}000 \times 2 = 2{,}000$ minutes (over 33 hours) to move all the boxes, the cargo train delivers everything in just 30 minutes.

Deep learning is not about driving an urgent letter through winding roads; it is about moving billions of identical numerical multiplications simultaneously. A Central Processing Unit (CPU) is optimized for **latency**—executing complex sequential control logic with deep speculative branch prediction and massive caches. A Graphics Processing Unit (GPU) is optimized for **throughput**—allocating the vast majority of its silicon area to raw Arithmetic Logic Units (ALUs) executing millions of floating-point operations in parallel.

---

## 2. Mathematical & Architectural Symbol Decoder

| Symbol | Concept / Metric | Units | Physical / Mathematical Meaning |
| :--- | :--- | :--- | :--- |
| $\text{SM}$ | Streaming Multiprocessor | Hardware Unit | Independent GPU core containing Warp Schedulers, Register Files, Tensor Cores, and SRAM. |
| $W$ | Warp Size | Threads ($=32$) | Fundamental atomic execution unit of an NVIDIA GPU; 32 threads executing in lockstep (SIMT). |
| $\text{SIMT}$ | Single Instruction, Multiple Threads | Architectural Paradigm | Execution model where multiple threads execute the exact same instruction stream on distinct data. |
| $I$ | Operational / Arithmetic Intensity | $\text{FLOPs} / \text{Byte}$ | Ratio of floating-point computations performed to bytes of high-bandwidth memory transferred. |
| $P_{\text{peak}}$ | Peak Compute Throughput | $\text{TFLOP/s}$ ($10^{12} \text{ op/s}$) | Theoretical maximum arithmetic capability of all SMs and Tensor Cores combined. |
| $B_{\text{peak}}$ | Peak Memory Bandwidth | $\text{GB/s}$ or $\text{TB/s}$ | Theoretical maximum rate data can be shuttled between HBM/VRAM and the GPU chip. |
| $I^*$ | Roofline Ridge Point | $\text{FLOPs} / \text{Byte}$ | Critical operational intensity threshold $\frac{P_{\text{peak}}}{B_{\text{peak}}}$ separating memory-bound from compute-bound regimes. |
| $D = A \cdot B + C$ | MMA (Matrix Multiply-Accumulate) | Matrix Operation | Fundamental Tensor Core primitive multiplying two low-precision matrices and accumulating into higher precision. |
| $T_{\text{roofline}}$ | Theoretical Execution Lower Bound | Seconds ($\text{s}$) | Minimum physical execution time $\max\left(\frac{\text{FLOPs}}{P_{\text{peak}}}, \frac{\text{Bytes}}{B_{\text{peak}}}\right)$. |

---

## 3. CPU vs GPU Compute Paradigms: Latency vs. Throughput

Silicon design involves fundamental trade-offs governed by power limits and die area.

```
       CPU Core (Latency-Optimized)             GPU SM (Throughput-Optimized)
  +-------------------------------------+   +-------------------------------------+
  |   Branch Predictor & Out-of-Order   |   | [Registers: 64K x 32-bit per SM]    |
  |   Superscalar Dispatch Engine       |   +-------------------------------------+
  +-------------------------------------+   | [Warp Scheduler]  [Warp Scheduler]  |
  | Massive L1 / L2 / L3 Caches (MBs)   |   +-------------------------------------+
  +-------------------------------------+   | ALU ALU ALU ALU | TC  TC  TC  TC    |
  |   Few Fast ALUs (High Clock ~5GHz)  |   | ALU ALU ALU ALU | TC  TC  TC  TC    |
  +-------------------------------------+   | Shared Memory / L1 Data Cache (SRAM)|
                                            +-------------------------------------+
```

### 3.1 The CPU: Latency Minimization
1. **Large Caches**: Multi-megabyte L2 and L3 SRAM caches consume 50%+ of the CPU die area to avoid the catastrophic $\sim 100\text{ ns}$ latency penalty of fetching data from off-chip DDR5 DRAM.
2. **Complex Branch Prediction & Speculative Execution**: Advanced branch predictors, out-of-order execution windows (ROB), and register renaming logic deduce instruction dependencies and execute speculative code paths before branches are resolved.
3. **Low Concurrency, High Frequency**: CPUs feature 8 to 128 cores running at high frequencies ($3.5 - 5.5\text{ GHz}$).

### 3.2 The GPU: Throughput Maximization
1. **ALU Dominance**: GPUs strip away speculative execution engines, branch predictors, and giant multi-level caches. Instead, they pack tens of thousands of lightweight ALUs onto the silicon.
2. **Latency Hiding via Concurrency**: When a warp of 32 threads stalls waiting for memory to arrive from High Bandwidth Memory (HBM) ($300 - 800\text{ clock cycles}$ latency), the hardware warp scheduler does not wait—it instantly swaps in another ready warp in a single clock cycle without context-switch overhead. Concurrency hides latency.
3. **Modest Clock, Massive Parallelism**: Modern GPUs (like NVIDIA H100) run at $1.5 - 2.0\text{ GHz}$ across 14,592 CUDA cores and 456 Fourth-Gen Tensor Cores.

---

## 4. Streaming Multiprocessors (SMs), Warps, and SIMT

The primary architectural unit of an NVIDIA GPU is the **Streaming Multiprocessor (SM)**. An NVIDIA A100 GPU houses 108 SMs; an H100 houses 132 SMs.

### 4.1 The Warp: The Unit of Execution
At the software level, CUDA programmers write code for individual *threads*. At the hardware silicon level, however, threads do not execute independently.

The hardware groups threads into bundles of **$W = 32$ parallel threads** called a **Warp**.
- All 32 threads in a warp share a single Instruction Pointer (IP).
- At each clock cycle, the warp scheduler broadcasts one instruction, and all 32 ALUs in the warp execute that same instruction in lockstep on their respective individual registers. This execution model is known as **SIMT (Single Instruction, Multiple Threads)**.

### 4.2 The Warp Divergence Penalty
What happens when code includes data-dependent conditional branching?
```cuda
// Thread-level code executing in a 32-thread warp
if (threadIdx.x < 16) {
    // Branch Path A: takes 40 clock cycles
    compute_heavy_path_A(data);
} else {
    // Branch Path B: takes 40 clock cycles
    compute_heavy_path_B(data);
}
```

Because all 32 threads share a single instruction pointer, the hardware **cannot** execute Path A and Path B concurrently on different threads within the same warp!
Instead, the hardware must **serialize** the branches using an active execution mask:
1. **Pass 1**: The warp scheduler activates threads $0 - 15$ and masks off threads $16 - 31$. The ALUs execute Path A ($40\text{ cycles}$). During these 40 cycles, half the ALUs sit completely idle.
2. **Pass 2**: The warp scheduler inverts the mask, activating threads $16 - 31$ and masking off threads $0 - 15$. The ALUs execute Path B ($40\text{ cycles}$). The first 16 ALUs now sit completely idle.

$$\text{Total Execution Time} = T_A + T_B = 40 + 40 = 80\text{ cycles}$$

Whereas if all threads had taken the same branch, execution would take only $40\text{ cycles}$. Divergence results in an immediate **$2\times$ slowdown** and $50\%$ wasted compute slots.

---

## 5. Tensor Cores & Mixed-Precision MMA

Traditional CUDA cores execute scalar or vector instructions (e.g., single-precision fused multiply-add: $d = a \cdot b + c$ on 32-bit floats).

In deep learning, almost all compute is matrix multiplication ($\mathbf{Y} = \mathbf{X}\mathbf{W}$). Recognizing this, NVIDIA introduced **Tensor Cores** beginning with the Volta architecture (2017).

### 5.1 The Hardware Matrix Multiply-Accumulate (MMA) Primitive
A Tensor Core does not operate on single scalars; it executes an entire matrix multiply-accumulate operation in hardware per clock cycle:
$$\mathbf{D} = \mathbf{A} \cdot \mathbf{B} + \mathbf{C}$$

For example, a standard Tensor Core MMA tile multiplies an $M \times K$ matrix by a $K \times N$ matrix (typically $16 \times 16 \times 16$):
- Inputs $\mathbf{A}$ and $\mathbf{B}$: Stored in lower-precision formats (FP16 or BF16, 2 bytes per element, or FP8, 1 byte per element).
- Accumulator $\mathbf{C}$ and Output $\mathbf{D}$: Stored and accumulated in full 32-bit precision (FP32) to prevent numerical rounding catastrophe.

```
   Matrix A (16x16)       Matrix B (16x16)          Matrix C (16x16)          Matrix D (16x16)
       [ FP16 ]       x       [ FP16 ]        +         [ FP32 ]        =         [ FP32 ]
  +---------------+       +---------------+         +---------------+         +---------------+
  |               |       |               |         |               |         |               |
  |  16 rows      |       |  16 rows      |         |  16 rows      |         |  16 rows      |
  |  x 16 cols    |       |  x 16 cols    |         |  x 16 cols    |         |  x 16 cols    |
  +---------------+       +---------------+         +---------------+         +---------------+
```

### 5.2 Arithmetic Density Comparison
Consider an NVIDIA H100 GPU:
- Standard FP32 CUDA Cores: $\sim 67\text{ TFLOP/s}$
- FP16/BF16 Tensor Cores with FP32 Accumulation: $\mathbf{989\text{ TFLOP/s}}$ ($>14\times$ faster!)
- FP8 Tensor Cores: $\mathbf{1{,}978\text{ TFLOP/s}}$ ($>29\times$ faster!)

Tensor Cores achieve this order-of-magnitude leap because multiplying a matrix tile on fixed systolic arrays eliminates individual instruction decoding and register operand fetch overhead for every scalar product.

---

## 6. The GPU Memory Hierarchy & Bandwidth Latency Profile

Deep learning workloads are overwhelmingly constrained by the time spent moving bytes between storage levels.

```
  +-------------------------------------------------------------+  Latency      Bandwidth
  | Registers: ~64 KB per SM (Total ~30-40 MB across chip)     |  ~1 cycle     ~25-30 TB/s
  +-------------------------------------------------------------+
                                |
  +-------------------------------------------------------------+
  | Shared Memory / L1 SRAM: ~192-256 KB per SM (~25-30 MB chip)|  ~20 cycles   ~15-19 TB/s
  +-------------------------------------------------------------+
                                |
  +-------------------------------------------------------------+
  | L2 Cache: Shared across all SMs (~50 MB A100, ~50 MB H100)  |  ~100 cycles  ~5-7 TB/s
  +-------------------------------------------------------------+
                                |
  +-------------------------------------------------------------+
  | High Bandwidth Memory (HBM2e / HBM3): Off-chip VRAM (80GB)  |  ~300-800 cyc ~2.0-3.3 TB/s
  +-------------------------------------------------------------+
                                |
  +-------------------------------------------------------------+
  | Host CPU Memory (DDR5 via PCIe Gen5 x16)                    |  ~microsec    ~64 GB/s
  +-------------------------------------------------------------+
```

### 6.1 Register File
- Located directly inside the SM next to the ALUs.
- Massive aggregate bandwidth ($>25\text{ TB/s}$).
- Access takes 1 clock cycle.
- Spilling registers to local memory destroys performance instantly.

### 6.2 Shared Memory / L1 Data Cache (SRAM)
- Ultra-fast on-chip SRAM configurable by the developer in CUDA (`__shared__`).
- Allows threads within the same thread block to cooperate, share tiles of matrices, and avoid redundant trips to main memory.
- Bandwidth: $\approx 15-19\text{ TB/s}$. Latency: $\approx 20\text{ clock cycles}$.
- FlashAttention leverages Shared Memory directly to achieve its legendary speedups.

### 6.3 High Bandwidth Memory (HBM3)
- Stacks of DRAM dies connected to the GPU silicon interposer via through-silicon vias (TSVs).
- Size: 80GB to 144GB.
- Bandwidth: $2.0\text{ TB/s}$ (A100 HBM2e) to $3.35\text{ TB/s}$ (H100 HBM3).
- Latency: $300 - 800\text{ clock cycles}$.

---

## 7. Arithmetic Intensity & The Roofline Model

Why doesn't your model run at peak $989\text{ TFLOP/s}$? The answer is dictated by **The Roofline Model** (Williams, Waterman, Patterson 2009).

### 7.1 Operational (Arithmetic) Intensity $I$
Arithmetic intensity measures how many floating-point operations a kernel performs for every single byte of data it reads from or writes to off-chip HBM:

$$I = \frac{\text{Work (FLOPs)}}{\text{Data Movement (Bytes)}} \quad \left[\frac{\text{FLOP}}{\text{Byte}}\right]$$

### 7.2 The Roofline Inequality
A hardware accelerator has two distinct physical ceilings:
1. **Peak Compute Throughput**: $P_{\text{peak}}$ (in $\text{FLOP/s}$)
2. **Peak Memory Bandwidth**: $B_{\text{peak}}$ (in $\text{Bytes/s}$)

The maximum attainable performance $P$ for an operation with arithmetic intensity $I$ is strictly bounded by:

$$P = \min\left(P_{\text{peak}}, \; I \cdot B_{\text{peak}}\right) \quad [\text{FLOP/s}]$$

```
  Attainable
  Performance
  (TFLOP/s)
      ^
P_peak|----------------------------+============================= (Compute-Bound Ceiling)
      |                           /
      |                          /
      |                         /
      |                        /
      |                       /   Ridge Point I* = P_peak / B_peak
      |                      /
      |                     /
      |   MEMORY-BOUND     /      COMPUTE-BOUND
      |   (Bandwidth      /
      |    Limited)      /
      +-----------------+------------------------------------------> Arithmetic Intensity
      0                I*                                           (FLOPs / Byte)
```

### 7.3 The Ridge Point $I^*$
The **Ridge Point** $I^*$ is the exact arithmetic intensity where the memory bandwidth ceiling intersects the peak compute ceiling:

$$I^* = \frac{P_{\text{peak}}}{B_{\text{peak}}} \quad \left[\frac{\text{FLOP}}{\text{Byte}}\right]$$

- If $I < I^*$: The kernel is **Memory-Bound**. The ALUs spend most of their time idle waiting for data from HBM. Performance grows linearly with memory bandwidth $B_{\text{peak}}$ and arithmetic intensity $I$. Compute upgrades will NOT accelerate this kernel.
- If $I \ge I^*$: The kernel is **Compute-Bound**. The memory bus can feed data faster than the ALUs can crunch it. The kernel runs at or near peak $P_{\text{peak}}$.

### 7.4 Quantitative Hardware Ridge Points

#### NVIDIA A100-SXM4 (80GB HBM2e)
- $P_{\text{peak}} = 312\text{ TFLOP/s} = 312 \times 10^{12}\text{ FLOP/s}$ (FP16 Tensor Core)
- $B_{\text{peak}} = 2{,}039\text{ GB/s} = 2{,}039 \times 10^9\text{ Bytes/s}$
$$I^*_{\text{A100}} = \frac{312 \times 10^{12}}{2{,}039 \times 10^9} \approx 153.0\text{ FLOPs/Byte}$$

#### NVIDIA H100-SXM5 (80GB HBM3)
- $P_{\text{peak}} = 989\text{ TFLOP/s} = 989 \times 10^{12}\text{ FLOP/s}$ (FP16/BF16 Tensor Core)
- $B_{\text{peak}} = 3{,}350\text{ GB/s} = 3{,}350 \times 10^9\text{ Bytes/s}$
$$I^*_{\text{H100}} = \frac{989 \times 10^{12}}{3{,}350 \times 10^9} \approx 295.2\text{ FLOPs/Byte}$$

Notice that on H100, a kernel must perform **nearly 300 FLOPs for every byte fetched** just to saturate the Tensor Cores!

---

## 8. Profiling Neural Network Kernels Under the Roofline

Let us derive the arithmetic intensity of the core layers of modern Transformer architectures (stored in FP16, where each scalar is $2\text{ bytes}$).

### 8.1 Elementwise Operations (ReLU, GELU, Residual Add)
Consider GELU applied to a tensor of $N$ elements:
- Work: $\sim 10\text{ FLOPs}$ per element $\implies \text{Total FLOPs} = 10 N$
- Memory: Read $x$ ($2N\text{ bytes}$), Write $y$ ($2N\text{ bytes}$) $\implies \text{Total Bytes} = 4N$

$$I_{\text{GELU}} = \frac{10 N}{4 N} = 2.5\text{ FLOPs/Byte}$$

Since $2.5 \ll 153.0$ (A100) and $2.5 \ll 295.2$ (H100), **GELU is deeply memory-bound**. On an H100, its maximum attainable performance is:
$$P = 2.5 \times 3{,}350\text{ GB/s} = 8.38\text{ TFLOP/s} \quad (\mathbf{0.85\%}\text{ of peak!})$$
Over $99\%$ of the GPU's raw computing power sits idle during raw elementwise operations!

### 8.2 Normalization Layers (LayerNorm, RMSNorm)
For an input tensor $\mathbf{X} \in \mathbb{R}^{B \times S \times D}$:
- Work: Mean and variance reduction, normalization, scaling, and bias: $\sim 8 D\text{ FLOPs}$ per token.
- Memory: Read $\mathbf{X}$ ($2D\text{ bytes}$), write output ($2D\text{ bytes}$) $= 4D\text{ bytes}$ per token.

$$I_{\text{LayerNorm}} = \frac{8 D}{4 D} = 2.0\text{ FLOPs/Byte}$$

LayerNorm is even more severely memory-bandwidth constrained.

### 8.3 Matrix Multiplication (GEMM: Feed-Forward / Linear Projections)
Consider multiplying $\mathbf{X} \in \mathbb{R}^{M \times K}$ by $\mathbf{W} \in \mathbb{R}^{K \times N}$:
- Work: $2 \cdot M \cdot N \cdot K\text{ FLOPs}$
- Memory: Read $\mathbf{X}$ ($2 M K\text{ B}$), Read $\mathbf{W}$ ($2 K N\text{ B}$), Write $\mathbf{Y}$ ($2 M N\text{ B}$):
$$\text{Bytes} = 2(MK + KN + MN)$$

$$I_{\text{GEMM}} = \frac{2 M N K}{2(MK + KN + MN)} = \frac{M N K}{MK + KN + MN}$$

Divide numerator and denominator by $M N K$:
$$I_{\text{GEMM}} = \frac{1}{\frac{1}{N} + \frac{1}{M} + \frac{1}{K}}$$

For large Transformer projections (e.g. LLaMA-70B where $M = 65{,}536$ tokens, $K = 4{,}096$, $N = 14{,}336$):
$$I_{\text{GEMM}} \approx 3{,}038\text{ FLOPs/Byte}$$

Because $3{,}038 \gg 295.2$, large GEMM operations are firmly **compute-bound**. They utilize $100\%$ of Tensor Core peak throughput ($989\text{ TFLOP/s}$).

---

## 9. Architectural Takeaways for Distributed & High-Performance AI

1. **Kernel Fusion**: Because operations like LayerNorm, GELU, and Residual Add are completely memory-bound, leaving them as separate kernels causes endless round-trips to HBM. **Kernel Fusion** (e.g. FlashAttention, PyTorch `torch.compile`, Triton) fuses these operations into a single kernel that keeps intermediate activations inside fast on-chip SRAM / Shared Memory ($15\text{ TB/s}$), bypassing the slow HBM bus entirely.
2. **Batching Increases Operational Intensity**: As batch size $M$ increases, $I_{\text{GEMM}}$ rises asymptotically toward $\min(K, N)$, transitioning memory-starved matrix-vector operations into compute-saturated matrix-matrix operations.
3. **Avoid Warp Divergence**: Never write thread branches where thread decisions oscillate within groups of 32. Structure control flow so all 32 threads take identical paths.
