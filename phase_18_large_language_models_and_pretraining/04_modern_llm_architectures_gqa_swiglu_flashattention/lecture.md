# Topic 18.4: Modern LLM Architectures (GQA, SwiGLU, FlashAttention)

---

## 🧭 Executive Overview

The original 2017 Transformer architecture (*"Attention Is All You Need"*, Vaswani et al.) was designed for machine translation on sequence lengths of a few hundred tokens. As artificial intelligence scaled to multi-billion-parameter foundation models operating over long contexts (from $8,192$ to $1,000,000$ tokens), three severe computational bottlenecks emerged:

1. **The KV-Cache Inference Bottleneck:** In Multi-Head Attention (MHA), every query head has its own Key and Value head. During autoregressive generation, retrieving these cached keys and values from GPU high-bandwidth memory (HBM) consumes immense memory bandwidth, starving Tensor Cores and creating a severe latency bottleneck.
2. **The Activation Function Expressivity Cap:** Traditional ReLU and GELU feedforward layers operate without continuous gating, limiting non-linear representational capacity per parameter.
3. **The Quadratic HBM Memory Bottleneck ($\mathcal{O}(T^2)$):** Standard dot-product attention materializes a full $T \times T$ attention matrix in GPU memory, repeatedly reading and writing intermediate activations across slow memory buses.

Modern production LLMs (such as LLaMA-2, LLaMA-3, Mistral, Gemma, and PaLM) overcome these bottlenecks through three unified architectural breakthroughs:
- **Grouped-Query Attention (GQA):** Sharing Key/Value heads across groups of Query heads, reducing KV-cache RAM and memory bandwidth by up to **$8\times$** during inference.
- **SwiGLU Activation:** Replacing traditional MLPs with a three-matrix Gated Linear Unit modulated by the SiLU activation, dramatically increasing parameter expressivity while matching standard parameter budgets ($d_{ffn} \approx \frac{8}{3} d_{model}$).
- **FlashAttention:** A hardware-aware tiling algorithm using **Online Softmax** that computes exact attention entirely within on-chip SRAM, reducing HBM memory reads/writes from $\mathcal{O}(T^2)$ to $\mathcal{O}(T)$ and delivering $2\times - 4\times$ wall-clock speedups.

---

## 👨‍🍳 12-Year-Old Analogy: The Chef's High-Speed Cutting Board & The Shared Kitchen Tools

> **Imagine a bustling Michelin-star restaurant kitchen with 64 master chefs trying to prepare a 100-course banquet.**
>
> 1. **Grouped-Query Attention (The Shared Kitchen Tools):**
>    - In the old kitchen (Multi-Head Attention), every single one of the 64 chefs demanded their own individual pepper grinder and salt shaker. The countertops were buried under 64 grinders and 64 shakers, leaving zero room to cook!
>    - In the modern kitchen (Grouped-Query Attention), we put the chefs into 8 teams of 8 chefs. Each team of 8 shares a single high-quality pepper grinder and salt shaker. The countertop clutter instantly drops by **$8\times$**, the chefs move with lightning speed, and the food tastes just as delicious!
>
> 2. **SwiGLU Activation (The Smart Valve):**
>    - In an ordinary blender (ReLU/GELU), you push a button and it blends everything at full blast or turns off.
>    - SwiGLU is like a high-tech blender equipped with a continuous sensor valve (the SiLU gate). It dynamically decides: *"Add $80\%$ sweetness to this vegetable, but completely block the bitterness."* It gives the network fine-grained, smooth multiplicative control over every ingredient!
>
> 3. **FlashAttention (The Chef's Cutting Board vs. The Cold Storage Basement):**
>    - In the old method (Standard Attention), every time a chef cut one slice of tomato, they walked down two flights of stairs to the cold basement (GPU HBM memory), wrote down on a giant $10,000 \times 10,000$ whiteboard how red the tomato was, walked back upstairs, read the whiteboard, and then cut the next slice. The chefs spent $90\%$ of their shift walking up and down stairs!
>    - FlashAttention says: *"Keep the ingredients right here on your lightning-fast cutting board (on-chip SRAM)! Slice the tomato, update the recipe running total in your head using online math, and only walk to the basement once when the entire dish is plated."* 
>    Zero wasted steps, pure speed!

---

## 📖 Mathematical Symbol Decoder

| Symbol | Mathematical Definition | Concrete Physical Meaning | Typical Values / Units |
| :--- | :--- | :--- | :--- |
| $H_Q$ | Number of Query heads | Number of distinct attention subspace representations | $32, 64, \text{ or } 128$ |
| $H_{KV}$ | Number of Key/Value heads | Number of stored Key/Value heads in the KV-cache | $1 \text{ (MQA)}, 8 \text{ (GQA)}, H_Q \text{ (MHA)}$ |
| $G$ | $H_Q / H_{KV}$ | Group size: number of Query heads sharing one KV head | $4, 8, \text{ or } 16$ |
| $d_{model}$ | Model embedding dimension | Width of the Transformer residual stream | $4096 - 16384$ |
| $d_{head}$ | $d_{model} / H_Q$ | Dimensionality of each individual attention head | $64 \text{ or } 128$ |
| $d_{ffn}$ | $\approx \frac{8}{3} d_{model}$ | Intermediate hidden dimension of the SwiGLU feedforward block | $11008 - 28672$ |
| $\text{SiLU}(z)$ | $z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$ | Sigmoid Linear Unit (Swish-1) activation function | Smooth, non-monotonic curve |
| $T$ | Sequence length / context | Total number of tokens in the attention context | $4096 - 131072$ tokens |
| $B_r, B_c$ | Block tile dimensions | Height and width of attention tiles loaded into SRAM | $64 \text{ or } 128$ tokens |
| $m, d$ | Running max & running denominator | FlashAttention online softmax normalization state | Maintained in GPU registers |

---

## 🔍 Deep-Dive: The Three Modern Architectural Pillars

### 1. Grouped-Query Attention (GQA) & The KV-Cache Bottleneck

#### The Autoregressive Inference Dilemma
During pretraining, all $T$ tokens are known in advance, allowing attention to be computed as a parallel matrix multiplication.
However, during **autoregressive generation (inference)**, tokens are generated one-by-one:
1. Generating token $t+1$ requires computing attention between the new query $q_{t+1}$ and **all historical keys and values $k_{1:t}, v_{1:t}$**.
2. To avoid recomputing keys and values for all past tokens at every step, past keys and values are stored in GPU memory as the **KV-Cache**.

#### The Memory Bandwidth Wall
For a model with $L$ layers, sequence length $T$, $H_{KV}$ key/value heads, and head dimension $d_{head}$ in 16-bit precision:
$$\text{KV-Cache Size (bytes)} = 2 \times L \times T \times H_{KV} \times d_{head} \times 2 \text{ bytes}$$

Consider a LLaMA-3 70B model ($L = 80, H_Q = 64, d_{head} = 128$) serving a $128\text{k}$ token context:
- Under standard **Multi-Head Attention (MHA)** ($H_{KV} = 64$):
  $$\text{Cache} = 2 \times 80 \times 131,072 \times 64 \times 128 \times 2 \approx \mathbf{343.6 \text{ Gigabytes per user!}}$$
  A single user's conversation exceeds the total memory of four 80GB GPUs!

Furthermore, at each generation step, generating a single token requires reading all $343.6\text{ GB}$ from HBM across memory buses. At $3.35\text{ TB/s}$ HBM bandwidth (NVIDIA H100), the maximum generation speed is capped at barely $9$ tokens per second regardless of compute capability!

```text
=======================================================================================
Attention Architecture  | KV Heads ($H_{KV}$) | KV-Cache at 128k Context | Reduction
=======================================================================================
Multi-Head Attention    | 64 heads            | 343.60 GB                | Baseline (1x)
Grouped-Query (GQA-8)   | 8 heads             | 42.95 GB                 | 8x Reduction!
Multi-Query (MQA-1)     | 1 head              | 5.37 GB                  | 64x Reduction
=======================================================================================
```

#### How GQA Bridges the Gap
- **Multi-Query Attention (MQA, Shazeer 2019):** Uses only $1$ KV head shared across all $H_Q$ query heads. While reducing memory by $64\times$, MQA suffers from capacity loss on complex retrieval and code tasks because all query heads must attend to identical key representations.
- **Grouped-Query Attention (GQA, Ainslie et al. 2023):** Partitions $H_Q$ query heads into $G$ groups of size $S = H_Q / G$. Each group shares a single Key head and Value head ($H_{KV} = G$).
- LLaMA-3 and Mistral choose $H_{KV} = 8$ (group size $8$). GQA delivers **$8\times$ lower memory and bandwidth** while matching $100\%$ of MHA's benchmark performance!

---

### 2. SwiGLU Gated Activation & Dimension Scaling

Traditional Transformer MLPs utilize two linear projections and a static activation:
$$\text{MLP}_{\text{GELU}}(x) = \text{GELU}(x W_{in}) W_{out}$$
where $W_{in} \in \mathbb{R}^{d_{model} \times 4 d_{model}}$ and $W_{out} \in \mathbb{R}^{4 d_{model} \times d_{model}}$.
Total parameters: $2 \times (d_{model} \times 4 d_{model}) = 8 d_{model}^2$.

#### The SwiGLU Formulation (Shazeer, 2020)
Shazeer introduced Gated Linear Units modulated by the SiLU (Swish-1) activation:
$$\text{SwiGLU}(x) = \left( (x W_{gate}) \odot \text{SiLU}(x W_{up}) \right) W_{down}$$
where $\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$.

- **Multiplicative Gating:** Unlike additive or monotonic activations, the gating signal $(x W_{gate})$ acts as a dynamic continuous filter, smoothly suppressing or amplifying specific semantic dimensions based on token context.
- **Gradient Preservation:** SiLU has a non-zero, smooth derivative everywhere, preventing the "dead neuron" phenomenon of ReLU.

#### Derivation of the $\frac{8}{3} d_{model}$ Parameter Parity
SwiGLU utilizes **three** weight matrices instead of two:
1. Gate projection: $W_{gate} \in \mathbb{R}^{d_{model} \times d_{ffn}}$
2. Up projection: $W_{up} \in \mathbb{R}^{d_{model} \times d_{ffn}}$
3. Down projection: $W_{down} \in \mathbb{R}^{d_{ffn} \times d_{model}}$

Total SwiGLU parameters:
$$\text{Params}_{\text{SwiGLU}} = 3 \cdot d_{model} \cdot d_{ffn}$$

To keep parameter count identical to a standard $4 d_{model}$ MLP ($8 d_{model}^2$ parameters):
$$3 \cdot d_{model} \cdot d_{ffn} = 8 d_{model}^2 \implies d_{ffn} = \frac{8}{3} d_{model} \approx 2.667 d_{model}$$

In practice, $d_{ffn}$ is rounded to the nearest multiple of 256 for NVIDIA Tensor Core memory alignment.
For $d_{model} = 4096$:
$$d_{ffn} = \text{round}\left(\frac{8}{3} \times 4096\right) = 10,922 \xrightarrow{\text{multiple of 256}} \mathbf{11,008}$$
$$\text{Params} = 3 \times 4096 \times 11008 = 135,266,304 \quad (\approx 8.2 d_{model}^2)$$

---

### 3. FlashAttention: Fast and Memory-Efficient Exact Attention

#### The GPU Memory Hierarchy Problem
A modern GPU contains two primary memory tiers:
- **HBM (High-Bandwidth Memory):** 80 GB capacity, but relatively slow ($\approx 2.0 - 3.35 \text{ TB/s}$).
- **SRAM (On-Chip Shared Memory):** Only $\approx 20 \text{ MB}$ per streaming multiprocessor (SM), but blazingly fast ($\approx 19 \text{ TB/s}$).

Standard Attention (Vaswani et al.) executes as three discrete GPU kernel launches:
1. Compute $S = Q K^T / \sqrt{d_k} \in \mathbb{R}^{T \times T}$ $\to$ **Write $S$ to HBM** ($\mathcal{O}(T^2)$ memory writes).
2. Read $S$ from HBM $\to$ Compute $P = \text{softmax}(S)$ $\to$ **Write $P$ to HBM** ($\mathcal{O}(T^2)$ memory writes).
3. Read $P$ and $V$ from HBM $\to$ Compute $O = P V \in \mathbb{R}^{T \times d}$ $\to$ **Write $O$ to HBM**.

At $T = 64,000$, materializing the $T \times T$ float16 attention matrix requires **$8.19\text{ Gigabytes}$ per attention head**! With 32 heads, that is **262 GB of intermediate memory per layer**, causing immediate OOM.

---

#### The FlashAttention Solution: Tiling + Online Softmax
FlashAttention (Dao et al., 2022) computes exact attention **without ever materializing the $T \times T$ matrix in HBM**.

#### Online Softmax Mathematics
Standard softmax requires knowing the global maximum $m = \max_j x_j$ and sum of exponentials $d = \sum_j e^{x_j - m}$ before outputting probabilities.

Milakov & Gimelshein (2018) proved that softmax can be evaluated **incrementally across partitioned blocks**:
Suppose an input vector $x$ is partitioned into two blocks $x^{(1)}$ and $x^{(2)}$:
- Block 1 statistics:
  $$m^{(1)} = \max_j x_j^{(1)}, \quad d^{(1)} = \sum_j e^{x_j^{(1)} - m^{(1)}}$$
- Block 2 statistics:
  $$m^{(2)} = \max_j x_j^{(2)}, \quad d^{(2)} = \sum_j e^{x_j^{(2)} - m^{(2)}}$$
- Merged global maximum:
  $$m^{\text{new}} = \max(m^{(1)}, m^{(2)})$$
- Merged denominator with correction factors:
  $$d^{\text{new}} = d^{(1)} e^{m^{(1)} - m^{\text{new}}} + d^{(2)} e^{m^{(2)} - m^{\text{new}}}$$

Similarly, the unnormalized attention output vector $O$ accumulates incrementally:
$$O^{\text{new}} = O^{(1)} e^{m^{(1)} - m^{\text{new}}} + \left(\sum_j e^{x_j^{(2)} - m^{\text{new}}} V_j^{(2)}\right)$$

At the end of the final block:
$$\text{Output} = \frac{O^{\text{final}}}{d^{\text{final}}}$$

#### Algorithmic Impact
1. Blocks of $Q, K, V$ (size $64 \times 128$) are loaded directly into ultra-fast **SRAM**.
2. Softmax and matrix products are computed entirely inside SRAM registers.
3. Only the final output $O \in \mathbb{R}^{T \times d}$ is written back to HBM.
4. Total HBM read/write volume drops from **$\mathcal{O}(T^2)$ to $\mathcal{O}(T)$**, yielding an exact mathematical result with **$2\times - 4\times$ speedup** and enabling long contexts up to $1\text{M}$ tokens!

---

## 💻 Python Implementation Architecture

The accompanying `experiment.py` script implements:
1. Complete mathematical simulation of the KV-cache across MHA, GQA, and MQA demonstrating the exact $8\times$ compression.
2. Pure Python Grouped-Query Attention execution validating head grouping and broadcasting.
3. SwiGLU activation forward pass and parameter parity verification.
4. FlashAttention tiled Online Softmax proving numerical equivalence against standard monolithic softmax down to $< 10^{-14}$ error.

Run the experiment directly:
```bash
python3 phase_18_large_language_models_and_pretraining/04_modern_llm_architectures_gqa_swiglu_flashattention/experiment.py
```
