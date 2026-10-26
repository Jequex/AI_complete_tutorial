# Topic 01: Exercises & Solutions — The Inference Bottleneck & Roofline Model

---

### Exercise 1: Roofline Analysis of Prefill vs. Decode on NVIDIA H100

An engineering team deploys a 13B parameter dense Transformer in FP16 precision ($e = 2$ bytes per parameter, weight memory $M_{\text{weights}} = 26 \text{ GB}$) on an NVIDIA H100 SXM5 GPU:
- Peak FP16 Tensor Core Compute: $P_{\text{peak}} = 989 \text{ TFLOP/s} = 989 \times 10^{12} \text{ FLOP/s}$.
- HBM3 Memory Bandwidth: $B_{\text{mem}} = 3.35 \text{ TB/s} = 3.35 \times 10^{12} \text{ Bytes/s}$.

Assume linear layer computation dominates total FLOPs ($2 \cdot N_{\text{params}}$ FLOPs per token), and memory traffic is dominated by loading the $26 \text{ GB}$ weights from HBM.

**Questions:**
1. Calculate the ridge point $I^* = P_{\text{peak}} / B_{\text{mem}}$ for this GPU.
2. In the **Prefill Phase**, a single user prompt of $S = 2048$ tokens arrives ($b = 1$). Compute the operational intensity $I_{\text{prefill}}$, the attainable performance $P_{\text{prefill}}$, and the execution time $T_{\text{prefill}}$. Is it memory-bound or compute-bound?
3. In the **Decode Phase**, the engine generates token-by-token for $b = 1$ sequence ($s = 1$). Compute the operational intensity $I_{\text{decode}}$, attainable performance $P_{\text{decode}}$, time per token $T_{\text{decode}}$, and the GPU compute utilization percentage.

---

#### Solution to Exercise 1

**1. Calculate Ridge Point $I^*$:**
$$I^* = \frac{P_{\text{peak}}}{B_{\text{mem}}} = \frac{989 \times 10^{12} \text{ FLOP/s}}{3.35 \times 10^{12} \text{ Bytes/s}} \approx \mathbf{295.22 \text{ FLOPs/Byte}}$$

**2. Prefill Phase Analysis ($b = 1, S = 2048$):**
- Total FLOPs: $2 \times N_{\text{params}} \times S = 2 \times (13 \times 10^9) \times 2048 = 5.3248 \times 10^{13} \text{ FLOPs} = 53.248 \text{ TFLOPs}$.
- Memory loaded: The 26 GB weights are streamed once: $M_{\text{traffic}} \approx 26 \times 10^9 \text{ Bytes}$.
- Operational Intensity:
  $$I_{\text{prefill}} = \frac{5.3248 \times 10^{13} \text{ FLOPs}}{26 \times 10^9 \text{ Bytes}} = \mathbf{2048 \text{ FLOPs/Byte}}$$
- Since $I_{\text{prefill}} = 2048 > I^* = 295.22$, the prefill phase is **strictly Compute-Bound**.
- Attainable Performance:
  $$P_{\text{prefill}} = P_{\text{peak}} = \mathbf{989 \text{ TFLOP/s}}$$
- Execution Time:
  $$T_{\text{prefill}} = \frac{\text{Total FLOPs}}{P_{\text{peak}}} = \frac{5.3248 \times 10^{13}}{989 \times 10^{12}} \approx \mathbf{0.0538 \text{ s}} \approx \mathbf{53.8 \text{ ms}}$$

**3. Decode Phase Analysis ($b = 1, s = 1$):**
- Total FLOPs per step: $2 \times N_{\text{params}} \times 1 = 2 \times (13 \times 10^9) \times 1 = 2.6 \times 10^{10} \text{ FLOPs} = 0.026 \text{ TFLOPs}$.
- Memory loaded: The entire 26 GB weight matrix must still be loaded from HBM: $M_{\text{traffic}} \approx 26 \times 10^9 \text{ Bytes}$.
- Operational Intensity:
  $$I_{\text{decode}} = \frac{2.6 \times 10^{10} \text{ FLOPs}}{26 \times 10^9 \text{ Bytes}} = \mathbf{1.0 \text{ FLOP/Byte}}$$
- Since $I_{\text{decode}} = 1.0 \ll I^* = 295.22$, the decode phase is **severely Memory-Bandwidth Bound**.
- Attainable Performance:
  $$P_{\text{decode}} = B_{\text{mem}} \cdot I_{\text{decode}} = (3.35 \times 10^{12} \text{ Bytes/s}) \times 1.0 \text{ FLOP/Byte} = \mathbf{3.35 \text{ TFLOP/s}}$$
- Time per token:
  $$T_{\text{decode}} = \frac{M_{\text{weights}}}{B_{\text{mem}}} = \frac{26 \times 10^9 \text{ Bytes}}{3.35 \times 10^{12} \text{ Bytes/s}} \approx \mathbf{0.00776 \text{ s}} \approx \mathbf{7.76 \text{ ms}}$$
  (equivalent to generating $1 / 0.00776 \approx 128.8$ tokens/sec).
- GPU Compute Utilization:
  $$\text{Utilization} = \frac{P_{\text{decode}}}{P_{\text{peak}}} = \frac{3.35 \text{ TFLOP/s}}{989 \text{ TFLOP/s}} \approx \mathbf{0.339\%}$$
  Over $99.66\%$ of the H100 tensor cores sit completely idle!

---

### Exercise 2: Analytical KV Cache Memory Scaling and Max Batch Capacity

Consider serving the LLaMA-2-70B model with the following architecture:
- $n_{\text{layers}} = 80$
- $n_{\text{heads}} = 64$ (Query heads)
- $n_{\text{kv\_heads}} = 8$ (Key/Value heads, using Grouped-Query Attention)
- $d_{\text{head}} = 128$
- Data precision: 16-bit FP16 ($e = 2$ bytes per element)

The model weights occupy $M_{\text{weights}} = 140 \text{ GB}$. The serving engine is hosted on an 8 $\times$ 80GB GPU cluster (total 640 GB VRAM).
Runtime framework overhead, CUDA context, and working activation scratchpads consume a fixed $M_{\text{overhead}} = 40 \text{ GB}$ across the cluster.
The remaining VRAM is allocated exclusively to the KV Cache.

**Questions:**
1. Derive the formula for KV cache memory per token per sequence, and compute its exact value in KB/token.
2. How much KV memory would be required if the model used standard Multi-Head Attention ($n_{\text{kv\_heads}} = 64$) instead of GQA? Compute the compression ratio.
3. If all requests have an average context length of $s = 4096$ tokens, calculate the absolute maximum concurrent batch size $b_{\text{max}}$ that can be served without running out of memory under GQA.

---

#### Solution to Exercise 2

**1. KV Cache Footprint per Token:**
For a single sequence ($b=1$) and single token ($s=1$):
$$\text{Scalars per token per layer} = 2 \cdot (n_{\text{kv\_heads}} \cdot d_{\text{head}}) = 2 \cdot (8 \cdot 128) = 2 \times 1024 = 2048 \text{ scalars}$$
Across all $n_{\text{layers}} = 80$ layers with $e = 2$ bytes:
$$\text{Bytes per token} = 2 \cdot e \cdot n_{\text{layers}} \cdot n_{\text{kv\_heads}} \cdot d_{\text{head}} = 4 \cdot 80 \cdot 8 \cdot 128 = \mathbf{327,680 \text{ Bytes}} = \mathbf{320 \text{ KB/token}}$$

**2. Comparison with Multi-Head Attention (MHA):**
Under standard MHA ($n_{\text{kv\_heads}} = 64$):
$$\text{Bytes per token (MHA)} = 4 \cdot 80 \cdot 64 \cdot 128 = 2,621,440 \text{ Bytes} = \mathbf{2560 \text{ KB/token}} = \mathbf{2.5 \text{ MB/token}}$$
$$\text{Compression Ratio} = \frac{\text{Bytes (MHA)}}{\text{Bytes (GQA)}} = \frac{64}{8} = \mathbf{8.0\times}$$
GQA reduces the memory footprint of every stored token by exactly $8\times$.

**3. Maximum Concurrent Batch Size $b_{\text{max}}$ at $s = 4096$:**
- Total Cluster VRAM: $640 \text{ GB}$.
- Memory available for KV Cache:
  $$M_{\text{KV\_avail}} = 640 \text{ GB} - 140 \text{ GB (Weights)} - 40 \text{ GB (Overhead)} = \mathbf{460 \text{ GB}} = 460 \times 10^9 \text{ Bytes}$$
- Memory consumed by 1 sequence of context length $s = 4096$:
  $$M_{\text{seq}} = 327,680 \text{ Bytes/token} \times 4096 \text{ tokens} = 1,342,177,280 \text{ Bytes} \approx \mathbf{1.342 \text{ GB/seq}}$$
- Maximum batch size:
  $$b_{\text{max}} = \left\lfloor \frac{M_{\text{KV\_avail}}}{M_{\text{seq}}} \right\rfloor = \left\lfloor \frac{460 \times 10^9}{1,342,177,280} \right\rfloor = \lfloor 342.72 \rfloor = \mathbf{342 \text{ concurrent sequences}}$$

*(Note: Under MHA, each sequence would take $1.342 \times 8 \approx 10.74 \text{ GB}$, limiting maximum batch size to only $\lfloor 460 / 10.74 \rfloor = 42$ sequences!)*

---

### Exercise 3: Derivation of the Critical Batch Size $b^*$ for Compute Saturation

Show analytically that the minimum batch size $b^*$ required to transition an LLM decode phase from memory-bound to compute-bound on an accelerator is given by:
$$b^* \approx I^* = \frac{P_{\text{peak}}}{B_{\text{mem}}}$$
when sequence lengths are moderate. Explain why production serving systems often fail to reach $b^*$ in low-latency interactive applications.

---

#### Solution to Exercise 3

At each decode step, the system executes forward passes for $b$ sequences, generating $1$ token each.
- The total computations required by linear projections (neglecting self-attention matrix multiplies which are small for moderate $s$):
  $$\text{FLOPs} \approx 2 \cdot b \cdot N_{\text{params}}$$
- The total data transferred across the memory bus:
  $$\text{Bytes} \approx M_{\text{weights}} + M_{\text{KV}}(b, s) = (e \cdot N_{\text{params}}) + M_{\text{KV}}(b, s)$$
For small to moderate sequence lengths $s$, model weights heavily dominate the memory bus traffic: $e \cdot N_{\text{params}} \gg M_{\text{KV}}(b, s)$.
Therefore:
$$I(b) = \frac{\text{FLOPs}}{\text{Bytes}} \approx \frac{2 \cdot b \cdot N_{\text{params}}}{e \cdot N_{\text{params}}} = \frac{2 \cdot b}{e}$$
For FP16 precision ($e = 2$ bytes):
$$I(b) \approx \frac{2 \cdot b}{2} = b \quad \text{FLOPs/Byte}$$

By the Roofline Model, an operation transitions to the compute-bound regime when its operational intensity reaches or exceeds the hardware ridge point:
$$I(b) \ge I^* \iff b \ge I^* = \frac{P_{\text{peak}}}{B_{\text{mem}}}$$
Hence:
$$\boxed{b^* \approx \frac{P_{\text{peak}}}{B_{\text{mem}}}}$$

**Why interactive systems fail to reach $b^*$:**
1. **Interactive SLAs**: End users expect Inter-Token Latencies (ITL) under 30-50 ms. Batches of $b = 300$ increase execution time and queuing latency, violating conversational response budgets.
2. **Poisson Arrival Queues**: Real-world request arrival rates fluctuate dynamically. Maintaining $b \ge 295$ continuously requires buffering and waiting for incoming requests, introducing massive Time to First Token (TTFT) penalties.
3. **KV Cache Memory Saturation**: As computed in Exercise 2, sustaining $b=342$ at 4k context consumes nearly half a terabyte of VRAM. If context lengths spike to 16k or 32k, VRAM quickly runs out before compute saturation can be reached.
