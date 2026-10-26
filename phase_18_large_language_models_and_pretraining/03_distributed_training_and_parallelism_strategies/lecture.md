# Topic 18.3: Distributed Training & Parallelism Strategies

---

## 🧭 Executive Overview

In modern artificial intelligence, models are no longer small enough to fit inside the memory of a single computer. When training a 70-billion-parameter foundation model such as LLaMA-2 70B or LLaMA-3 70B, the raw memory required to store the model parameters, their gradients, and the AdamW optimizer states totals **1,120 Gigabytes** of static high-bandwidth memory (HBM).

Because the world's most capable AI accelerators (such as the NVIDIA A100 or H100) have only **80 Gigabytes** of HBM, attempting to load a 70B model onto a single GPU causes an immediate, catastrophic Out-Of-Memory (OOM) error.

To overcome this **Memory Wall**, AI infrastructure engineering relies on **3D Parallelism**—a unified multi-dimensional distributed orchestration strategy combining:
1. **Fully Sharded Data Parallelism (FSDP / ZeRO-1, ZeRO-2, ZeRO-3):** Eliminating redundant memory by sharding optimizer states, gradients, and model weights across cluster ranks.
2. **Tensor Parallelism (Megatron-LM):** Slicing individual matrix multiplication operations across GPUs connected via ultra-high-speed intra-node NVLink interconnects.
3. **Pipeline Parallelism (GPipe / 1F1B):** Partitioning sequential Transformer layers across distinct nodes with interleaved micro-batch scheduling to minimize pipeline bubble overhead.
4. **Mixed Precision Training (BF16 / FP16):** Accelerating tensor core math while maintaining FP32 master weights to preserve numerical stability during microscopic gradient accumulation.

---

## 🏗️ 12-Year-Old Analogy: The Giant Construction Crew Splitting a Skyscraper

> **Imagine you are tasked with building a 100-story mega-skyscraper in just one week.**
>
> **The Problem:**
> The blueprint, the heavy steel beams, the concrete mixers, and the supervisor's instruction logbook weigh 1,000 tons combined. If you hire a single worker and dump all 1,000 tons onto their back, they will instantly be crushed to dust.
>
> **The 3 Parallelism Strategies:**
>
> 1. **Data Parallelism (ZeRO Sharding) — Splitting the Cargo:**
>    Instead of giving every worker their own 1,000-ton backpack containing all tools, you hire 16 workers. You tear the giant blueprint into 16 sections, split the concrete mixers into 16 parts, and give each worker only a tiny 60-pound bag. Whenever a worker needs to pour concrete for a specific pillar, they quickly radio their teammates, borrow the needed tool for 2 seconds, do the job, and hand it back!
>
> 2. **Tensor Parallelism (Megatron-LM) — Slicing a Giant Steel Beam:**
>    You have a colossal 40-foot steel girder that needs to be lifted into place. A single worker cannot lift it. So 4 workers stand side-by-side, each lifting an exact 10-foot segment of the same girder simultaneously. They walk in lockstep, communicating instantly through direct headsets (NVLink) so the beam never bends or drops.
>
> 3. **Pipeline Parallelism (1F1B) — The Assembly Line Floors:**
>    Worker A only lays foundations on Floors 1–25. As soon as Floor 25 is done, Worker B begins building Floors 26–50, while Worker A immediately starts the next building. Instead of everyone standing idle waiting for the whole skyscraper to finish, materials flow up and down in an uninterrupted rhythmic conveyor belt!

---

## 📖 Mathematical Symbol Decoder

| Symbol | Mathematical Definition | Concrete Physical Meaning | Typical Values / Units |
| :--- | :--- | :--- | :--- |
| $N$ | Non-embedding parameter count | Total learnable weights of the foundation model | $7 \times 10^9 - 4.05 \times 10^{11}$ |
| $N_d$ | Data parallel world size | Number of GPU devices participating in data/ZeRO sharding | $8, 16, 64, \dots, 16384$ GPUs |
| $M_{\text{static}}$ | $2N + 2N + 12N = 16N$ bytes | Total static memory footprint required by AdamW training | Bytes (1,120 GB for 70B) |
| $TP$ | Tensor parallel world size | Number of GPUs splitting intra-layer weight matrices | Typically $2, 4, \text{ or } 8$ (NVLink domain) |
| $PP$ | Pipeline parallel stages | Number of sequential layer chunks across nodes | Typically $4, 8, \text{ or } 16$ |
| $m$ | Number of microbatches | Number of chunks a global batch is split into for pipeline flow | $m \ge 4 \cdot PP$ |
| $F_{\text{bubble}}$ | $\frac{PP - 1}{m + PP - 1}$ | Fractional idle time spent waiting on pipeline boundary bubbles | Fraction $\in [0.05, 0.50]$ |
| $S_{\text{loss}}$ | Dynamic loss scale factor | Constant used in FP16 training to prevent gradient underflow | $2^{10} - 2^{24}$ |
| $\text{All-Reduce}$ | $\sum_{i=1}^P x_i$ across all ranks | Collective communication primitive summing vectors across GPUs | Ring or Tree topology |

---

## 🧱 The Memory Wall: Why 70B Cannot Fit on an 80GB GPU

During training with the standard AdamW optimizer, every single parameter $w$ generates a cascade of memory allocations:

```text
========================================================================================
State Component               | Precision | Bytes per Param | 70B Model Footprint
========================================================================================
Model Weights ($W$)           | FP16/BF16 | 2 bytes         | 140 GB
Gradients ($\nabla_W L$)      | FP16/BF16 | 2 bytes         | 140 GB
AdamW Master Weights ($W^*$)  | FP32      | 4 bytes         | 280 GB
AdamW 1st Momentum ($m_t$)    | FP32      | 4 bytes         | 280 GB
AdamW 2nd Momentum ($v_t$)    | FP32      | 4 bytes         | 280 GB
----------------------------------------------------------------------------------------
Total Static Memory           | Mixed     | 16 bytes        | 1,120 GB
========================================================================================
```

### Why FP32 Master Weights Are Mathematically Mandatory
In mixed precision, forward and backward operations execute in 16-bit precision (FP16 or BF16) to leverage Tensor Cores. However, updating weights directly in 16 bits fails mathematically:
$$\Delta w = -\eta \frac{m_t}{\sqrt{v_t} + \epsilon}$$
With learning rate $\eta \approx 10^{-4}$ and small gradient signals, $|\Delta w|$ can easily be on the order of $10^{-7}$. 

In standard FP16 or BF16, machine epsilon is $\epsilon_{\text{machine}} \approx 2^{-10} \approx 9.7 \times 10^{-4}$ or $2^{-7} \approx 7.8 \times 10^{-3}$. When adding $10^{-7}$ to a weight of magnitude $1.0$:
$$1.0 + 10^{-7} \xrightarrow{\text{16-bit float}} 1.0$$
The update underflows completely and truncates to zero! The model never learns. 

To prevent this catastrophic stagnation, optimizers maintain a **full 32-bit single-precision master copy** ($4N$ bytes) where updates accumulate accurately, and cast weights back to 16 bits only for forward/backward steps.

---

## 🔀 3D Parallelism Architecture

To distribute training across hundreds or thousands of GPUs, engineers deploy a 3D grid of parallelism:
$$\text{Total GPUs} = \text{Data Parallel} (DP) \times \text{Tensor Parallel} (TP) \times \text{Pipeline Parallel} (PP)$$

```text
                    ┌─────────────────────────┐
                    │      3D Parallelism     │
                    └────────────┬────────────┘
         ┌───────────────────────┼──────────────────────┐
         ▼                       ▼                      ▼
  Data Parallelism        Tensor Parallelism     Pipeline Parallelism
  (ZeRO-1, 2, 3 / FSDP)   (Megatron-LM)          (GPipe / 1F1B)
  Shards memory across    Splits W_Q,K,V,O       Partitions layers
  data-replica ranks      across intra-node GPUs across inter-node GPUs
```

---

### 1. Data Parallelism & The ZeRO Hierarchy (Rajbhandari et al., 2020)
In classical Distributed Data Parallel (DDP), every GPU holds an identical copy of all $16N$ bytes of weights, gradients, and optimizer states. ZeRO (Zero Redundancy Optimizer) eliminates this redundancy in three stages:

#### ZeRO-1 ($P_{os}$): Sharded Optimizer States
- Model weights ($2N$) and gradients ($2N$) remain replicated on every GPU.
- AdamW optimizer states ($12N$) are sharded evenly across the $N_d$ GPUs.
$$\text{Memory}_{\text{ZeRO-1}} = 4N + \frac{12N}{N_d}$$
For a 70B model on 16 GPUs: $4(70) + \frac{12(70)}{16} = 280 + 52.5 = 332.5 \text{ GB}$ (Still OOM on 80GB GPUs).

#### ZeRO-2 ($P_{os+g}$): Sharded Optimizer States & Gradients
- Model weights ($2N$) remain replicated.
- Gradients ($2N$) and optimizer states ($12N$) are sharded across $N_d$ GPUs.
$$\text{Memory}_{\text{ZeRO-2}} = 2N + \frac{14N}{N_d}$$
For 70B on 16 GPUs: $2(70) + \frac{14(70)}{16} = 140 + 61.25 = 201.25 \text{ GB}$ (Still OOM).

#### ZeRO-3 ($P_{os+g+p}$ / Fully Sharded Data Parallel - FSDP): Full Sharding
- Model weights ($2N$), gradients ($2N$), and optimizer states ($12N$) are **all sharded across $N_d$ GPUs**.
$$\text{Memory}_{\text{ZeRO-3}} = \frac{16N}{N_d}$$
For 70B on 16 GPUs: $\frac{16(70)}{16} = \mathbf{70.0 \text{ GB}}$!
**70.0 GB fits cleanly inside an 80 GB A100/H100 GPU!**

#### The Communication Trade-Off:
- **DDP Communication:** At the end of the backward pass, an `All-Reduce` sums gradients across all devices. Total volume = $2N$ words transferred.
- **ZeRO-3 Communication:** 
  1. Before each forward layer: `All-Gather` weights ($1N$ words).
  2. Before each backward layer: `All-Gather` weights ($1N$ words).
  3. After backward pass: `Reduce-Scatter` gradients ($1N$ words).
  Total volume = $3N$ words transferred.
  ZeRO-3 introduces an exact **$50\%$ communication overhead** ($\frac{3N}{2N} = 1.5\times$) over vanilla DDP.

---

### 2. Megatron-LM Tensor Parallelism (Shoeybi et al., 2019)
Within a single physical server (e.g., an 8-GPU DGX node), GPUs communicate via NVLink at blistering speeds ($600 - 900 \text{ GB/s}$). Tensor Parallelism splits individual weight matrices across GPUs to execute parallel GEMM operations.

#### Column-Parallel Linear Layer
We partition weight matrix $W \in \mathbb{R}^{d_{in} \times d_{out}}$ column-wise across $K$ GPUs:
$$W = \begin{bmatrix} W_1 & W_2 & \dots & W_K \end{bmatrix}, \quad W_i \in \mathbb{R}^{d_{in} \times \frac{d_{out}}{K}}$$
The input $X$ is replicated on all GPUs. Each GPU computes:
$$Y_i = X W_i \in \mathbb{R}^{B \times \frac{d_{out}}{K}}$$
**Notice: ZERO COMMUNICATION is required!** 
Furthermore, any element-wise non-linearity $f$ (such as GELU or SiLU) distributes cleanly:
$$f(Y) = f([Y_1 \mid Y_2]) = [f(Y_1) \mid f(Y_2)]$$

#### Row-Parallel Linear Layer
Next, the second linear layer has weight matrix $V \in \mathbb{R}^{d_{out} \times d_{final}}$ partitioned row-wise:
$$V = \begin{bmatrix} V_1 \\ V_2 \\ \vdots \\ V_K \end{bmatrix}, \quad V_i \in \mathbb{R}^{\frac{d_{out}}{K} \times d_{final}}$$
Each GPU computes the partial matrix product:
$$Z_i = Y_i V_i \in \mathbb{R}^{B \times d_{final}}$$
The mathematical total is the sum over all ranks:
$$Z = X W V = \sum_{i=1}^K Y_i V_i = \sum_{i=1}^K Z_i$$
To assemble $Z$, the GPUs execute an **`All-Reduce (SUM)`** operation across the TP group.

```text
[Input X] ──┬──> [ W_1 (Col Split) ] ──> [ GELU ] ──> [ V_1 (Row Split) ] ──┐
            │                                                                 │
            └──> [ W_2 (Col Split) ] ──> [ GELU ] ──> [ V_2 (Row Split) ] ──┼──> [ All-Reduce SUM ] ──> [ Output Z ]
```

By pairing Column-Parallel with Row-Parallel projections:
- **Multi-Head Attention:** $W_Q, W_K, W_V$ are column-parallel (splitting heads across GPUs); $W_O$ is row-parallel. **Exactly 1 All-Reduce.**
- **MLP Block:** $W_{gate}, W_{up}$ are column-parallel; $W_{down}$ is row-parallel. **Exactly 1 All-Reduce.**
Total communication: **Only 2 All-Reduces in the forward pass and 2 All-Reduces in the backward pass per Transformer layer!**

---

### 3. Pipeline Parallelism (PP) & The 1F1B Schedule
When a model has 80 or 120 layers, even Tensor Parallelism across 8 GPUs is insufficient. Pipeline Parallelism partitions layers sequentially across nodes (e.g. GPU 0 handles layers 1–20, GPU 1 handles layers 21–40).

#### The Pipeline Bubble Problem
If a batch is passed sequentially through $p$ pipeline stages, early GPUs sit idle waiting for the backward pass to return from later GPUs.
If a batch is divided into $m$ microbatches, the fraction of idle time (the **pipeline bubble**) is:
$$F_{\text{bubble}} = \frac{p - 1}{m + p - 1}$$
To keep bubble overhead below $10\%$, systems configure $m \ge 8p$.

#### GPipe vs 1F1B Scheduling
- **GPipe:** Executes all $m$ forward microbatches, then all $m$ backward microbatches. Memory bottleneck: all $m$ intermediate activation tensors must be held in RAM simultaneously, causing activation OOM.
- **1F1B (One-Forward-One-Backward):** After a brief startup warm-up, each GPU executes **one forward microbatch followed immediately by one backward microbatch**. Once a backward microbatch finishes, its activations are immediately freed! Peak activation memory drops from $\mathcal{O}(m)$ to $\mathcal{O}(p)$.

---

## ⚡ Mixed Precision: FP16 vs BF16

```text
Bit Layout Comparison:
FP32:  [Sign: 1 bit] [Exponent:  8 bits] [Mantissa / Fraction: 23 bits]   Range: 10^(±38)
FP16:  [Sign: 1 bit] [Exponent:  5 bits] [Mantissa / Fraction: 10 bits]   Range: 10^(±5)  <-- Underflow hazard!
BF16:  [Sign: 1 bit] [Exponent:  8 bits] [Mantissa / Fraction:  7 bits]   Range: 10^(±38)  <-- Match FP32 range!
```

- **FP16 Limitations:** With only 5 exponent bits, FP16 can only represent numbers down to $2^{-14} \approx 6.1 \times 10^{-5}$. LLM gradient magnitudes frequently drop to $10^{-6}$ or $10^{-7}$, causing complete numerical underflow to $0.0$. Training requires complex **dynamic loss scaling** ($S \approx 2^{16}$ or $2^{20}$).
- **BF16 Dominance:** Bfloat16 preserves all 8 exponent bits of FP32. Its dynamic range ($10^{-38}$ to $10^{38}$) completely prevents underflow, eliminating loss scaling entirely and making training stable out-of-the-box.

---

## 💻 Python Implementation Architecture

The accompanying `experiment.py` script implements:
1. Exact mathematical memory footprint evaluation of 7B, 13B, 34B, and 70B models under DDP, ZeRO-1, ZeRO-2, and ZeRO-3 across cluster sizes up to 64 GPUs.
2. Distributed communication volume formulas verifying the $1.5\times$ ZeRO-3 overhead.
3. Pure Python Megatron-LM Tensor Parallelism simulating column-parallel and row-parallel linear layers and proving numerical equivalence with monolithic matrix multiplication down to $< 10^{-14}$ error.
4. Pipeline parallelism bubble fraction and utilization simulator.

Run the experiment directly:
```bash
python3 phase_18_large_language_models_and_pretraining/03_distributed_training_and_parallelism_strategies/experiment.py
```
