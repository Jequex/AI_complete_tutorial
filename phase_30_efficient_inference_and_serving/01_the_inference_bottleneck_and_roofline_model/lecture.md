# Topic 01: The Inference Bottleneck & The Roofline Model

## 1. Introduction & 12-Year-Old Intuition

Imagine you run an elite translation library with a massive master dictionary that weighs 140 pounds. 

Two very different customers walk into your library:

1. **The Prefill Customer (The Reading Phase)**:
   The first customer hands you a 1,000-word essay and says, *"Please read this entire prompt and understand its full context."*
   You take the 140-pound dictionary off the high shelf, place it on your desk once, and spend the next two minutes flipping rapidly through its pages, cross-referencing words, digesting sentences, and thinking. Because you are doing thousands of calculations and comparisons for that single dictionary pull, your brain (compute) is working at 100% capacity. You are **compute-bound**.

2. **The Decode Customer (The Writing Phase)**:
   The second customer wants you to compose an essay one single word at a time.
   To write the *first* word, you must lift the 140-pound dictionary from the shelf to your desk, check one rule, write down *"The"*, and put the dictionary back.
   To write the *second* word, you must lift the entire 140-pound dictionary again, check one rule, write down *"cat"*, and put it back.
   To write 500 words, you have to lift that 140-pound dictionary **500 times**! Your brain isn't tired at all—your arms and back are exhausted from hauling the heavy book back and forth across the room. You are **memory-bandwidth bound**.

This is the central paradox and engineering crisis of Large Language Model serving:
- **Prefill (Prompt Processing)** processes all prompt tokens in parallel using dense Matrix-Matrix multiplication ($\text{GEMM}$). It saturates tensor cores and achieves high arithmetic intensity.
- **Decode (Autoregressive Generation)** generates one token per sequence per forward pass using Matrix-Vector multiplication ($\text{GEMV}$). Every single layer's multi-gigabyte weight matrices must be hauled across the GPU's memory bus into cache just to multiply them against a single vector of numbers.

Understanding this hardware reality through the **Roofline Model** is the first step toward building hyper-efficient inference engines like vLLM, TensorRT-LLM, and TGI.

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Units / Domain | Description & Engineering Significance |
| :--- | :--- | :--- | :--- |
| $P$ | Attainable Performance | FLOPs/sec (TFLOP/s) | Actual throughput of floating-point operations achieved on hardware. |
| $P_{\text{peak}}$ | Peak Compute Capacity | FLOPs/sec (TFLOP/s) | Maximum theoretical mathematical operations the GPU tensor cores can execute. |
| $B_{\text{mem}}$ | Memory Bandwidth | Bytes/sec (TB/s or GB/s) | Rate at which data can be transferred from High Bandwidth Memory (HBM) to on-chip SRAM/registers. |
| $I$ | Operational / Arithmetic Intensity | FLOPs / Byte | Number of floating-point operations executed per byte of data loaded from off-chip DRAM. |
| $I^*$ | Roofline Ridge Point | FLOPs / Byte | The critical threshold $I^* = P_{\text{peak}} / B_{\text{mem}}$ separating memory-bound from compute-bound regimes. |
| $b$ | Batch Size | Integer $\ge 1$ | Number of concurrent requests or sequences processed in a single forward pass. |
| $s$ | Sequence Length | Tokens | Current sequence context length (prompt length + generated output tokens). |
| $d$ | Hidden Dimension | Integer ($d = n_{\text{heads}} \cdot d_{\text{head}}$) | Model hidden representation width (e.g., 4096 in LLaMA-7B, 8192 in LLaMA-70B). |
| $n_{\text{layers}}$ | Layer Count | Integer | Number of Transformer blocks stacked in the neural network architecture. |
| $n_{\text{heads}}$ | Attention Heads | Integer | Number of Query heads (or Key/Value heads in Multi-Head Attention). |
| $n_{\text{kv\_heads}}$ | KV Heads | Integer | Number of Key/Value heads (equals $n_{\text{heads}}$ in MHA; equals 1 in MQA; $< n_{\text{heads}}$ in GQA). |
| $d_{\text{head}}$ | Head Dimension | Integer | Per-head projection dimension ($d_{\text{head}} = d / n_{\text{heads}}$, often 64 or 128). |
| $e$ | Element Precision Size | Bytes | Byte width per weight/activation scalar (e.g., 2 bytes for FP16/BF16, 1 byte for FP8, 0.5 bytes for INT4). |
| $\text{Mem}_{\text{KV}}$ | KV Cache Memory | Bytes (GB) | Cumulative high-bandwidth memory dedicated to cached Keys and Values across all active requests. |

---

## 3. The Roofline Model: Hardware Ceiling vs Operational Reality

The **Roofline Model** (Williams, Waterman, & Patterson, 2009) bounds the maximum performance $P$ achievable by an algorithm on a specific computing architecture:

$$\boxed{P(I) = \min\left(P_{\text{peak}}, \; B_{\text{mem}} \cdot I\right)}$$

```
Performance P (TFLOP/s)
    ▲
P_peak │───────────────────────────────┐ Compute-Bound Regime
       │                              /│ (P = P_peak)
       │                             / │
       │                            /  │
       │                           /   │
       │                          /    │
       │  Memory-Bound Regime    /     │
       │  (P = B_mem * I)       /      │
       │                       /       │
       └──────────────────────┴────────┴─────────────────────▶
                             I* = P_peak / B_mem       Operational Intensity I
                                                       (FLOPs / Byte)
```

### 3.1 The Ridge Point $I^*$
The **ridge point** $I^*$ defines the exact operational intensity required to fully saturate the GPU's arithmetic engines:

$$I^* = \frac{P_{\text{peak}}}{B_{\text{mem}}}$$

Let us compute $I^*$ for representative modern datacenter GPUs using 16-bit precision ($\text{FP16} / \text{BF16}$):

1. **NVIDIA A100 SXM4 (80GB HBM2e)**:
   - $P_{\text{peak}} = 312 \times 10^{12} \text{ FLOPs/s} = 312 \text{ TFLOP/s}$ (dense FP16 Tensor Core)
   - $B_{\text{mem}} = 2.039 \times 10^{12} \text{ Bytes/s} = 2.039 \text{ TB/s}$
   - $$I^*_{\text{A100}} = \frac{312 \times 10^{12}}{2.039 \times 10^{12}} \approx \mathbf{153 \text{ FLOPs/Byte}}$$

2. **NVIDIA H100 SXM5 (80GB HBM3)**:
   - $P_{\text{peak}} = 989 \times 10^{12} \text{ FLOPs/s} = 989 \text{ TFLOP/s}$ (dense FP16 without sparsity; 1,979 TFLOP/s with 2:4 sparsity)
   - $B_{\text{mem}} = 3.35 \times 10^{12} \text{ Bytes/s} = 3.35 \text{ TB/s}$
   - $$I^*_{\text{H100}} = \frac{989 \times 10^{12}}{3.35 \times 10^{12}} \approx \mathbf{295 \text{ FLOPs/Byte}}$$

If an operation's intensity $I < I^*$, it is strictly **memory-bandwidth bound**: the arithmetic execution units are starved for data, idling while memory controllers fetch bytes across the physical bus. If $I \ge I^*$, it is **compute-bound**.

---

## 4. Prefill vs. Decode: An Arithmetic Intensity Analysis

Let us analyze a standard Transformer linear projection layer with weight matrix $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ in FP16 precision ($e = 2$ bytes).

- Weight memory footprint: $M_W = 2 \cdot d_{\text{in}} \cdot d_{\text{out}}$ bytes.
- Input tensor $X \in \mathbb{R}^{b \times s \times d_{\text{in}}}$.
- Number of floating-point operations: Each output element requires a dot-product of length $d_{\text{in}}$ (which takes $d_{\text{in}}$ multiplications and $d_{\text{in}}$ additions = $2 d_{\text{in}}$ FLOPs). Total FLOPs:
  $$\text{FLOPs} = 2 \cdot b \cdot s \cdot d_{\text{in}} \cdot d_{\text{out}}$$

Assuming memory reads are dominated by weight matrix streaming (for large models where weights $\gg$ input activations):

$$I = \frac{\text{Total FLOPs}}{\text{Total Bytes Loaded}} \approx \frac{2 \cdot b \cdot s \cdot d_{\text{in}} \cdot d_{\text{out}}}{2 \cdot d_{\text{in}} \cdot d_{\text{out}} + 2 \cdot b \cdot s \cdot (d_{\text{in}} + d_{\text{out}})}$$

When weights dominate ($d_{\text{in}} \cdot d_{\text{out}} \gg b \cdot s \cdot d$), this simplifies to:

$$\boxed{I \approx b \cdot s \quad \text{FLOPs/Byte}}$$

### 4.1 The Prefill Phase (Prompt Processing)
During prefill, the user provides a prompt of length $s = S_{\text{prompt}}$ (e.g., $S_{\text{prompt}} = 1024, 4096$).
All $S_{\text{prompt}}$ tokens are processed simultaneously in a single dense General Matrix Multiply ($\text{GEMM}$):

$$I_{\text{prefill}} \approx b \cdot S_{\text{prompt}}$$

Even for a single request ($b = 1$) with $S_{\text{prompt}} = 2048$:
$$I_{\text{prefill}} \approx 2048 \text{ FLOPs/Byte} \gg I^*_{\text{H100}} (295 \text{ FLOPs/Byte})$$

**Conclusion**: Prefill is deeply located in the **compute-bound** regime. The GPU tensor cores run at close to peak utilization ($>60\text{--}80\%$ Model FLOPs Utilization / MFU).

### 4.2 The Decode Phase (Autoregressive Token Generation)
During decode, the model has already ingested the prompt and is generating token-by-token. At each forward step, exactly **one** new token is processed per sequence ($s = 1$):

$$I_{\text{decode}} \approx b \cdot 1 = b \quad \text{FLOPs/Byte}$$

If a serving system processes a single sequence at a time ($b = 1$):
$$I_{\text{decode}} = 1 \text{ FLOP/Byte} \ll I^*_{\text{H100}} (295 \text{ FLOPs/Byte})$$

On an H100 GPU ($P_{\text{peak}} = 989 \text{ TFLOP/s}, B_{\text{mem}} = 3.35 \text{ TB/s}$), attainable performance at $b=1$ is:
$$P = B_{\text{mem}} \cdot I = 3.35 \times 10^{12} \times 1 = 3.35 \text{ TFLOP/s}$$

$$\text{GPU Compute Efficiency} = \frac{3.35 \text{ TFLOP/s}}{989 \text{ TFLOP/s}} \approx \mathbf{0.34\%}$$

More than **99.6%** of the GPU's expensive tensor cores sit completely idle, waiting for bytes to trickle in from memory!

To push the decode phase into the compute-bound regime on an H100, the batch size must satisfy:
$$b \ge I^* = 295$$

Running a concurrent batch size of $b \ge 295$ sequences sounds trivial until we examine the physical memory footprint of the **KV Cache**.

---

## 5. The KV Cache Memory Footprint

To generate token $t+1$, standard self-attention requires computing attention against all previous tokens $1, 2, \dots, t$:

$$\text{Attention}(Q_t, K_{\le t}, V_{\le t}) = \text{softmax}\left(\frac{Q_t K_{\le t}^\top}{\sqrt{d_{\text{head}}}}\right) V_{\le t}$$

Recomputing $K$ and $V$ activations for all past tokens at every step would incur quadratic $\mathcal{O}(s^2)$ computational complexity. The **KV Cache** stores the past Key and Value vectors in high-bandwidth GPU memory.

### 5.1 Exact Formula Derivation
For each token in each active sequence, at each transformer layer:
- One Key vector: $n_{\text{kv\_heads}} \times d_{\text{head}}$ scalars.
- One Value vector: $n_{\text{kv\_heads}} \times d_{\text{head}}$ scalars.
- Total scalars per token per layer $= 2 \cdot n_{\text{kv\_heads}} \cdot d_{\text{head}}$.
- Each scalar occupies $e$ bytes (in 16-bit FP16/BF16, $e = 2$).

Across $n_{\text{layers}}$ layers, batch size $b$, and sequence length $s$:

$$\boxed{\text{Mem}_{\text{KV}} = 2 \cdot e \cdot n_{\text{layers}} \cdot n_{\text{kv\_heads}} \cdot d_{\text{head}} \cdot b \cdot s \quad \text{Bytes}}$$

For standard 16-bit precision ($e = 2$ bytes), this yields the canonical formula:

$$\boxed{\text{Mem}_{\text{KV}} = 4 \cdot n_{\text{layers}} \cdot n_{\text{kv\_heads}} \cdot d_{\text{head}} \cdot b \cdot s \quad \text{Bytes}}$$

Notice that for standard Multi-Head Attention (MHA), $n_{\text{kv\_heads}} \cdot d_{\text{head}} = d_{\text{model}}$.

### 5.2 Real-World Case Study: LLaMA-2-70B vs LLaMA-1-65B
Consider serving at batch size $b = 64$ with context length $s = 4096$ tokens:

#### Case A: LLaMA-1-65B (Multi-Head Attention)
- $n_{\text{layers}} = 80$, $n_{\text{heads}} = 64$, $d_{\text{head}} = 128$ ($d = 8192$), $e = 2$.
$$\text{KV per token} = 4 \times 80 \times 64 \times 128 = 2,621,440 \text{ bytes} \approx 2.62 \text{ MB/token}$$
$$\text{Mem}_{\text{KV}} = 2.62144 \text{ MB} \times 64 \times 4096 = \mathbf{687.2 \text{ GB!}}$$
The KV cache alone requires nine 80GB A100 GPUs, even before loading the 130GB model weights!

#### Case B: LLaMA-2-70B (Grouped-Query Attention with 8 KV heads)
- $n_{\text{layers}} = 80$, $n_{\text{kv\_heads}} = 8$, $d_{\text{head}} = 128$, $e = 2$.
$$\text{KV per token} = 4 \times 80 \times 8 \times 128 = 327,680 \text{ bytes} \approx 0.328 \text{ MB/token}$$
$$\text{Mem}_{\text{KV}} = 0.32768 \text{ MB} \times 64 \times 4096 = \mathbf{85.9 \text{ GB}}$$
GQA achieves an immediate **$8\times$ reduction** in KV cache memory footprint!

---

## 6. The Fundamental 4-Way Serving Tradeoff

Every production LLM serving system navigates a zero-sum, four-dimensional Pareto frontier:

```
                     [LATENCY]
                   (TTFT & ITL)
                       ▲
                      / \
                     /   \
                    /     \
                   /       \
      [MEMORY] ◀──────────────▶ [THROUGHPUT]
   (KV Cache, Weights)       (Tokens / Sec)
                   \       /
                    \     /
                     \   /
                      \ /
                       ▼
                    [COST]
                 ($ / 1M Tokens)
```

1. **Latency vs. Throughput**:
   - To achieve the lowest possible Inter-Token Latency (ITL), we want a batch size of $b = 1$ so that forward passes execute with zero queuing. But at $b=1$, throughput is abysmal and GPU utilization is $<1\%$.
   - To maximize total system throughput (tokens/sec/GPU), we want large batch sizes ($b = 128\text{--}256$), which drives arithmetic intensity up toward $I^*$. However, each forward pass takes longer (higher compute + larger KV cache scanning), which increases user-facing latency.

2. **Throughput vs. Memory**:
   - Maximizing batch size requires massive amounts of VRAM dedicated to the KV cache. If memory is exhausted, the server must reject incoming requests or swap KV blocks to host CPU memory over PCIe, creating catastrophic latency spikes.

3. **Memory vs. Cost**:
   - Running across more GPUs (Tensor Parallelism) splits the KV cache across devices, expanding available VRAM and allowing higher concurrency. However, more GPUs directly inflate hourly infrastructure costs.

---

## 7. Mathematical Derivations: Latency Limits

### Derivation: Decode Step Execution Time
In the decode phase, at each step we must read:
1. Model weights: $M_{\text{weights}} = \sum \text{Params} \times e$ bytes.
2. Active KV Cache: $M_{\text{KV}}(b, s)$ bytes.

Assuming pure memory-bandwidth boundedness (which holds for $b < I^*$):

$$T_{\text{step}}(b, s) = \frac{M_{\text{weights}} + M_{\text{KV}}(b, s)}{B_{\text{mem}}}$$

For a 70B parameter model in FP16 ($M_{\text{weights}} = 140 \text{ GB}$) served on an 8-GPU node of A100-80GB GPUs with aggregate memory bandwidth $B_{\text{agg}} = 8 \times 2.039 \text{ TB/s} = 16.312 \text{ TB/s}$:

Even if KV cache size is negligible ($s \approx 0$):
$$T_{\text{min\_step}} = \frac{140 \times 10^9 \text{ Bytes}}{16.312 \times 10^{12} \text{ Bytes/s}} \approx 8.58 \text{ ms}$$

Maximum theoretical decode speed for a single user ($b=1$):
$$\text{Max Single-Stream Speed} = \frac{1}{0.0858 \text{ s}} \approx \mathbf{116 \text{ tokens/sec}}$$

No software optimization can generate tokens faster than 116 tokens/second on this cluster without either:
- Compressing the weights (quantization to INT8/INT4), or
- Altering the decoding algorithm (Speculative Decoding).

---

## 8. Summary & Architectural Takeaways

1. **Dual Nature of Generation**: An LLM inference engine must manage two completely different operational modes: compute-bound parallel prefill and memory-bound serial decode.
2. **The Memory Wall**: At small batch sizes, decoding is constrained by how fast bytes move across the HBM bus, leaving modern tensor cores idling at $<1\%$ efficiency.
3. **The KV Cache Tax**: Autoregressive caching trades compute for linear memory growth $\mathcal{O}(b \cdot s)$. At long context windows, the KV cache dwarfs model weights in memory consumption.
4. **The Scaling Direction**: Mitigating the inference bottleneck requires algorithmic solutions (Speculative Decoding to raise arithmetic intensity per memory load) and architectural solutions (PagedAttention to eliminate memory fragmentation).
