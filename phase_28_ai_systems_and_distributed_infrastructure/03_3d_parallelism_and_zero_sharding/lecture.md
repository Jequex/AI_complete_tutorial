# Topic 03: 3D Parallelism & ZeRO Sharding

## 1. Introduction & 12-Year-Old Intuition

Imagine you run a publishing company that needs to produce a 10,000-page encyclopedia in one day.

No single printer has enough memory or ink to print the whole encyclopedia at once. How do you organize your factory?

1. **Data Parallelism (DP) — The Multiple Photocopy Machines**:
   You have 8 identical printing presses. If the entire encyclopedia *could* fit on one machine, each machine prints copies for different customers (micro-batches) at the same time. At the end of the day, all 8 printers compare notes and calibrate ink levels (gradient all-reduce). But what if the book is so huge it doesn't fit on one machine?
2. **Pipeline Parallelism (PP) — The Assembly Line**:
   You slice the 10,000-page book into 8 consecutive chapters. Machine 1 prints Chapter 1, binds it, and slides it down a conveyor belt to Machine 2, which prints Chapter 2, and so on.
   - *The Catch (The Bubble)*: When the day starts, Machine 8 sits completely idle twiddling its thumbs waiting for Chapters 1 through 7 to arrive! At the end of the day, Machine 1 sits idle waiting for the last book to exit Machine 8. This idle waiting time is called the **Pipeline Bubble**.
3. **Tensor Parallelism (TP) — Slicing a Single Giant Page**:
   Inside a single chapter, there is a giant panoramic picture so wide that it exceeds the width of any printer bed. You slice the picture down the middle: Machine A prints the left half, and Machine B prints the right half simultaneously. Before gluing the book together, you instantly stitch the two halves together with tape (NVLink All-Reduce). Because this tape-stitching happens on *every single page*, the two printers must sit right next to each other on the exact same desk!
4. **ZeRO Sharding — The Shared Library Card**:
   Instead of every worker keeping an entire encyclopedia dictionary, grammar guide, and thesaurus on their desk (optimizer states and parameters), the 8 workers split the dictionary into 8 parts: Worker 1 keeps words A–C, Worker 2 keeps D–F. When Worker 1 needs a word from D–F, they ask Worker 2, read it, and put it right back. Nobody's desk is cluttered, and massive books fit easily!

---

## 2. Mathematical Symbol Decoder

| Symbol | Concept / Metric | Formal Domain | Physical / Architectural Meaning |
| :--- | :--- | :--- | :--- |
| $\Phi$ | Model Parameter Count | $\mathbb{Z}^+$ | Total number of trainable scalar weights in the neural network. |
| $N$ | World Size (Cluster Scale) | $\mathbb{Z}^+$ | Total number of participating GPUs. |
| $p$ | Pipeline Stages | $\mathbb{Z}^+$ | Number of consecutive partitions dividing the model depth (PP degree). |
| $m$ | Number of Micro-batches | $\mathbb{Z}^+$ | Number of small data chunks per global training step. |
| $F$ | Pipeline Bubble Fraction | $[0, 1]$ | Proportion of total compute time wasted due to pipeline startup/drain stalls. |
| $T$ | Tensor Parallel Degree | $\mathbb{Z}^+$ | Number of GPU slices dividing a single layer (Megatron TP degree, typically 2, 4, or 8). |
| $D$ | Data Parallel Degree | $\mathbb{Z}^+$ | Number of replicated model instances (satisfying $N = D \times p \times T$). |
| $\text{Mem}_{\text{states}}$ | Model State Memory | Bytes | Static memory required to hold parameters, gradients, and optimizer states ($= 16\Phi$ bytes in FP16/FP32 mixed precision). |

---

## 3. The Model State Memory Crisis: Why 16$\Phi$ Bytes?

When training a model with $\Phi$ parameters using mixed precision (FP16 or BF16 weights and FP32 Adam optimizer), where does the memory go?

```
+---------------------------------------------------------------------------------+
| MODEL STATES MEMORY BREAKDOWN (16 bytes per parameter)                          |
+---------------------------------------------------------------------------------+
| FP16 Model Weights:   2 bytes x Phi                                             |
+---------------------------------------------------------------------------------+
| FP16 Gradients:       2 bytes x Phi                                             |
+---------------------------------------------------------------------------------+
| FP32 Adam States:     12 bytes x Phi                                            |
|   - FP32 Master Weights:     4 bytes x Phi (prevents underflow / rounding error)|
|   - FP32 First Momentum (m): 4 bytes x Phi                                      |
|   - FP32 Second Momentum (v):4 bytes x Phi                                      |
+---------------------------------------------------------------------------------+
| TOTAL MODEL STATE MEMORY = (2 + 2 + 12) * Phi = 16 * Phi bytes                  |
+---------------------------------------------------------------------------------+
```

### The Concrete Math for LLaMA-70B:
$$\Phi = 70 \times 10^9\text{ parameters}$$
$$\text{Memory}_{\text{model\_states}} = 16 \times 70 \times 10^9\text{ bytes} = \mathbf{1{,}120\text{ Gigabytes!}}$$

A flagship NVIDIA H100 GPU has **80 GB** of VRAM. A single GPU cannot even hold the model states, let alone the dynamic activations and KV caches!

---

## 4. The 3D Parallelism Triad

To train models that exceed single-GPU capacity, modern distributed frameworks (Megatron-LM, DeepSpeed, NeMo) compose three orthogonal axes of parallelism:

$$\text{Total World Size } N = \text{Data Parallelism } (D) \times \text{Pipeline Parallelism } (p) \times \text{Tensor Parallelism } (T)$$

```
                                  [ 3D PARALLELISM CUBE ]
                                  
                               +-----------------------+
                              /                       /|
                             /                       / |
                            /       DATA (DP)       /  |
                           +-----------------------+   |
                           |                       |   |
                           |                       |   +
          PIPELINE (PP)    |                       |  /
           (Inter-Node)    |                       | /  TENSOR (TP)
                           |                       |/   (Intra-Node NVLink)
                           +-----------------------+
```

---

### 4.1 Data Parallelism (DDP)
- Replicates the model across $D$ workers.
- Each worker processes a distinct mini-batch $\mathbf{X}_i \in \mathbb{R}^{\frac{B}{D} \times S \times d}$.
- During the backward pass, gradients $\mathbf{g}_i$ are synchronized across workers via **Ring All-Reduce**.
- **PyTorch DDP Optimization (Bucket All-Reduce)**: As layers in the backward pass finish calculating their gradients, PyTorch accumulates them into buckets ($\sim 25\text{ MB}$). It triggers asynchronous All-Reduce on bucket $k$ while the GPU continues computing gradients for layer $k-1$, overlapping communication with computation.
- **Bottleneck**: DDP replicates the entire $16\Phi$ bytes of model states on every GPU.

---

### 4.2 Pipeline Parallelism (PP)
When the model layers do not fit on one GPU, Pipeline Parallelism partitions the $L$ layers sequentially across $p$ pipeline stages:
- Stage 0 executes layers $0 \dots \frac{L}{p} - 1$.
- Stage 1 executes layers $\frac{L}{p} \dots \frac{2L}{p} - 1$, and so on.
- Communication between stages is purely **Point-to-Point (P2P)**: Stage $k$ sends activation tensors to Stage $k+1$, and receives backward gradients from Stage $k+1$.
- Because inter-stage traffic is small and sequential, PP is ideal for **inter-node** communication across InfiniBand networks.

#### The Pipeline Bubble: GPipe vs 1F1B
To keep all stages utilized, a batch of size $B$ is divided into $m$ **micro-batches** ($m \gg p$).

```
GPipe Schedule (All Forward, then All Backward):
Stage 3:          [F0][F1][F2][F3]            [B3][B2][B1][B0]
Stage 2:       [F0][F1][F2][F3]      ...   [B3][B2][B1][B0]
Stage 1:    [F0][F1][F2][F3]            [B3][B2][B1][B0]
Stage 0: [F0][F1][F2][F3]            [B3][B2][B1][B0]
         |<- Bubble ->|                               |<- Bubble ->|

1F1B Schedule (One-Forward-One-Backward):
Stage 3:          [F0]   [F1][B0][F2][B1][F3][B2]   [B3]
Stage 2:       [F0]   [F1][B0][F2][B1][F3]   [B2][B3]
Stage 1:    [F0]   [F1][B0][F2]   [B1][F3][B2][B3]
Stage 0: [F0]   [F1][B0]   [F2][B1][F3][B2][B3]
```

#### Bubble Fraction Derivation:
Total ideal compute time across $m$ microbatches: $T_{\text{ideal}} = 2m \cdot t_{\text{step}}$.
Actual wall-clock time including startup and drain: $T_{\text{actual}} = 2(m + p - 1) \cdot t_{\text{step}}$.

$$F = \frac{\text{Idle Bubble Slots}}{\text{Total Available Slots}} = \frac{p - 1}{m + p - 1} \approx \frac{p - 1}{m} \quad (\text{for } m \gg p)$$

- **Rule of Thumb**: To keep the bubble overhead under $10\%$ ($F < 0.10$), choose $m \ge 8p$.
- **Activation Memory Footprint**:
  - **GPipe**: Stage 0 must store activations for **all $m$ micro-batches** until backward arrives $\implies \mathcal{O}(m)$ activation memory.
  - **1F1B**: A backward pass runs as soon as possible, freeing activations immediately $\implies$ activation memory is capped at **$\mathcal{O}(p)$**, completely independent of $m$!

---

### 4.3 Tensor Parallelism (TP: Megatron-LM)
Tensor Parallelism partitions individual weight matrices across $T$ GPUs within a single layer.

#### The Megatron-LM MLP Split (Shoeybi et al. 2019)
An MLP layer consists of two linear projections:
$$\mathbf{Z} = \text{GeLU}(\mathbf{X} \mathbf{W}_1), \quad \mathbf{Y} = \mathbf{Z} \mathbf{W}_2$$
where $\mathbf{X} \in \mathbb{R}^{B \times S \times H}$, $\mathbf{W}_1 \in \mathbb{R}^{H \times 4H}$, and $\mathbf{W}_2 \in \mathbb{R}^{4H \times H}$.

```
1. Column-Parallel Linear Layer (W1):
   Partition W1 into column slices [W1,1 | W1,2]:
   GPU 1: Z1 = GeLU(X * W1,1)     [Dimension: B x S x 2H]
   GPU 2: Z2 = GeLU(X * W1,2)     [Dimension: B x S x 2H]
   --> Zero communication required between GPU 1 and GPU 2!

2. Row-Parallel Linear Layer (W2):
   Partition W2 into matching row slices [W2,1 / W2,2]:
   GPU 1: Y1 = Z1 * W2,1          [Dimension: B x S x H]
   GPU 2: Y2 = Z2 * W2,2          [Dimension: B x S x H]
   
   Final Output: Y = Y1 + Y2
   --> Exactly ONE All-Reduce across the T GPUs!
```

#### Mathematical Elegance:
By pairing a **Column-Parallel** layer with a **Row-Parallel** layer:
- The non-linear activation (GeLU) is applied locally on each GPU without inter-GPU communication.
- The entire MLP block requires **only 1 All-Reduce** in the forward pass and **only 1 All-Reduce** in the backward pass.
- Similarly, in Multi-Head Attention, $Q, K, V$ projections are Column-Parallel (heads partitioned across GPUs), and the output projection $W_O$ is Row-Parallel, requiring **only 1 All-Reduce**.
- **Crucial Constraint**: Because All-Reduce occurs on every single Transformer layer, TP generates high communication frequency. It is **strictly restricted to intra-node NVLink ($900\text{ GB/s}$)** with $T \in \{2, 4, 8\}$.

---

## 5. ZeRO: The Zero Redundancy Optimizer

The **Zero Redundancy Optimizer (ZeRO)** (Rajbhandari et al. 2020) eliminates the memory redundancies of Data Parallelism without changing the mathematical execution of the model.

```
DDP Baseline:
GPU 0: [ Params (2Phi) | Grads (2Phi) | Optimizer States (12Phi) ] = 16Phi
GPU 1: [ Params (2Phi) | Grads (2Phi) | Optimizer States (12Phi) ] = 16Phi

ZeRO-Stage 1 (Pos): Partition Optimizer States
GPU 0: [ Params (2Phi) | Grads (2Phi) | Opt Slice 0 (12Phi / N)   ] = 4Phi + 12Phi/N
GPU 1: [ Params (2Phi) | Grads (2Phi) | Opt Slice 1 (12Phi / N)   ] = 4Phi + 12Phi/N

ZeRO-Stage 2 (Pos+g): Partition Optimizer States & Gradients
GPU 0: [ Params (2Phi) | Grad Slice 0 (2Phi/N) | Opt Slice 0 (12Phi / N) ] = 2Phi + 14Phi/N
GPU 1: [ Params (2Phi) | Grad Slice 1 (2Phi/N) | Opt Slice 1 (12Phi / N) ] = 2Phi + 14Phi/N

ZeRO-Stage 3 (Pos+g+p): Partition Everything (Params + Grads + Optimizer)
GPU 0: [ Param Slice 0 (2Phi/N) | Grad Slice 0 (2Phi/N) | Opt Slice 0 (12Phi/N) ] = 16Phi / N
GPU 1: [ Param Slice 1 (2Phi/N) | Grad Slice 1 (2Phi/N) | Opt Slice 1 (12Phi/N) ] = 16Phi / N
```

---

### 5.1 ZeRO-Stage 1: Optimizer State Partitioning ($P_{os}$)
- The $12\Phi$ bytes of Adam optimizer states (master weights, $m$, $v$) are sharded across $N$ GPUs:
$$\text{Memory per GPU} = 2\Phi + 2\Phi + \frac{12\Phi}{N} = 4\Phi + \frac{12\Phi}{N}\text{ bytes}$$
- For large $N$, memory drops from $16\Phi$ to $4\Phi$ (a **$4\times$ memory reduction**).
- **Communication Overhead**: **$0\%$ increase**! Instead of All-Reducing gradients and having every GPU update all optimizer states, GPUs perform a Reduce-Scatter on gradients ($1\Phi$), update their local $\frac{1}{N}$ optimizer slice, and All-Gather the updated parameters ($1\Phi$). Total communication is $2\Phi$, identical to standard DDP.

---

### 5.2 ZeRO-Stage 2: Gradient Partitioning ($P_{os+g}$)
- As gradients are computed in the backward pass, they are directly reduced into their assigned GPU bucket via Reduce-Scatter and immediately discarded from the sending GPU:
$$\text{Memory per GPU} = 2\Phi + \frac{2\Phi}{N} + \frac{12\Phi}{N} = 2\Phi + \frac{14\Phi}{N}\text{ bytes}$$
- Memory drops to $\approx 2\Phi$ (an **$8\times$ memory reduction**).
- **Communication Overhead**: **$0\%$ increase** (still $2\Phi$ bytes per step).

---

### 5.3 ZeRO-Stage 3: Parameter Partitioning ($P_{os+g+p}$)
- Even the model parameters $2\Phi$ are sharded across the $N$ GPUs:
$$\text{Memory per GPU} = \frac{2\Phi}{N} + \frac{2\Phi}{N} + \frac{12\Phi}{N} = \frac{16\Phi}{N}\text{ bytes}$$
- **Linear Memory Scaling**: Memory scales perfectly as $\mathcal{O}(1/N)$!
  On 64 GPUs, LLaMA-70B model states drop from $1{,}120\text{ GB}$ down to **$17.5\text{ GB per GPU}$**!

#### How ZeRO-3 Works at Runtime:
1. **Forward Pass**: When Layer $l$ needs to compute, the GPU issues an asynchronous **All-Gather** to collect parameter slice $l$ from the other GPUs. It computes the forward activation, and **immediately deletes the parameters from memory**!
2. **Backward Pass**: When computing gradients for Layer $l$, the GPU issues another **All-Gather** to retrieve parameter slice $l$, computes activations and gradients, immediately discards the parameters, and performs a **Reduce-Scatter** on the gradients.

#### Communication Cost:
- Forward All-Gather: $1\Phi$ bytes
- Backward All-Gather: $1\Phi$ bytes
- Backward Reduce-Scatter: $1\Phi$ bytes
- **Total Communication Volume**: $1\Phi + 1\Phi + 1\Phi = \mathbf{3\Phi\text{ bytes}}$.

Compared to standard DDP ($2\Phi$), ZeRO-3 requires **only a $1.5\times$ increase in communication volume** in exchange for a massive, linear $N$-fold reduction in memory!
