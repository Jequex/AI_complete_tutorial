# Topic 05: Memory Optimization & Activation Checkpointing

## 1. Introduction & 12-Year-Old Intuition

Imagine you are taking a 100-step math exam on a tiny wooden student desk that can only fit **10 sheets of paper at once**.

Each math problem builds directly upon the answer from the problem before it:
$$x_1 = f(x_0), \quad x_2 = f(x_1), \quad \dots, \quad x_{100} = f(x_{99})$$

Once you reach the final step 100, the teacher asks you to **grade your work in reverse** (the backward pass). To grade step 99, you need step 98. To grade step 98, you need step 97, and so on.

### Approach 1: Keep Every Single Sheet (Standard Backpropagation)
You try to keep all 100 sheets of paper on your desk. By step 20, papers are falling off the desk onto the floor, spilling your drink, and causing complete chaos (**Out of Memory / OOM crash**).

### Approach 2: The Bookmark Strategy (Activation Checkpointing)
Instead of keeping all 100 sheets, you keep only **1 sheet every 10 steps** (Step 0, Step 10, Step 20, ..., Step 90). That's only 10 bookmark sheets!
- When you reach Step 100, you need steps 91 to 99 to grade the final block.
- You simply grab bookmark #90 and quickly recalculate steps 91 to 99 right on the spot! You grade them, erase those scratch notes, and throw them away.
- Then you grab bookmark #80, recalculate steps 81 to 89, grade them, and discard them.

Notice what happened:
- Your desk only ever held **at most 20 sheets** (10 bookmarks + 10 scratch sheets) instead of 100 sheets! You cut memory usage by **$80\%$**!
- What was the cost? You had to recalculate the scratch sheets once. Because backward grading takes twice as long as forward solving anyway, recalculating forward adds **only 33% extra work** in exchange for massive memory savings!

---

## 2. Mathematical Symbol Decoder

| Symbol | Concept / Metric | Formal Domain | Physical / Computational Meaning |
| :--- | :--- | :--- | :--- |
| $L$ | Network Depth | Layers ($\mathbb{Z}^+$) | Total number of sequential layers or Transformer blocks in the network. |
| $B$ | Micro-batch Size | Sequences ($\mathbb{Z}^+$) | Number of training sequences processed concurrently on a single GPU. |
| $S$ | Sequence Length (Context) | Tokens ($\mathbb{Z}^+$) | Context window length in tokens ($2\text{k}, 8\text{k}, 32\text{k}, 128\text{k}$). |
| $D$ | Hidden Dimension | $\mathbb{Z}^+$ | Width of hidden representation vectors in the Transformer ($d_{\text{model}}$). |
| $k$ | Checkpoint Interval | Layers ($\mathbb{Z}^+$) | Frequency of retaining activation tensors ($k^* = \sqrt{L}$). |
| $M_{\text{act}}$ | Activation Memory | Bytes | Dynamic memory holding intermediate activations required for gradient calculation. |
| $F_{\text{fwd}}$ | Forward FLOPs | FLOPs | Arithmetic operations required for one forward evaluation of a layer ($= 2BD^2$). |
| $F_{\text{bwd}}$ | Backward FLOPs | FLOPs | Arithmetic operations for backward pass ($= 4BD^2 = 2 F_{\text{fwd}}$). |
| $\mathbf{S}, \mathbf{P}$ | Attention Matrices | $\mathbb{R}^{S \times S}$ | Unnormalized attention scores ($\mathbf{Q}\mathbf{K}^T$) and normalized probabilities ($\text{Softmax}$). |

---

## 3. Dynamic Activation Memory: The Long-Context Crisis

When training neural networks, GPU memory is split into two categories:
1. **Static Memory**: Model weights, gradients, and optimizer states ($16\Phi$ bytes, sharded by ZeRO or Tensor Parallelism).
2. **Dynamic Memory**: Activations generated during the forward pass that must be stored until their corresponding backward layer executes.

### The Transformer Activation Formula:
For a standard Transformer block without cross-attention, the intermediate tensors stored per layer include:
- Multi-Head Attention: $Q, K, V$ projections ($6 B S D$), attention probabilities $\mathbf{P}$ ($2 B H S^2$ where $H$ is heads), attention output ($2 B S D$), residual addition ($2 B S D$).
- Feed-Forward Network: MLP intermediate projections ($16 B S D$), GeLU activation ($8 B S D$), down-projection ($2 B S D$).

Total activation memory per layer in FP16 ($2\text{ bytes/element}$):
$$A_{\text{layer}} \approx B \cdot S \cdot D \left(34 + 2 \frac{H \cdot S}{D}\right) \text{ bytes}$$

Across an $L$-layer network:
$$M_{\text{act}} = L \cdot A_{\text{layer}} \in \mathcal{O}(L \cdot B \cdot S \cdot D + L \cdot B \cdot S^2)$$

### Why Context Length Breaks Memory:
Consider LLaMA-70B ($L = 80$, $D = 8{,}192$, $H = 64$) with batch size $B = 1$:
- At context $S = 2{,}048$: Activation memory $\approx \mathbf{56\text{ GB}}$.
- At context $S = 32{,}768$: Activation memory explodes to $\mathbf{1{,}280\text{ GB}}$!
- At context $S = 128{,}000$: Activation memory reaches $\mathbf{18{,}400\text{ GB}}$!

For long contexts, **activation memory dwarfs model weights by orders of magnitude**. Without activation checkpointing and FlashAttention, long-context LLMs cannot be trained.

---

## 4. Mathematical Derivation of Activation Checkpointing

**Activation Checkpointing** (also known as *Gradient Recomputation*, Chen et al. 2016, Griewank 1992) trades cheap arithmetic operations to drastically compress activation memory.

### 4.1 Checkpoint Formulation
Consider an $L$-layer network:
$$\mathbf{a}_0 \xrightarrow{f_1} \mathbf{a}_1 \xrightarrow{f_2} \mathbf{a}_2 \dots \xrightarrow{f_L} \mathbf{a}_L$$

Instead of storing all $L$ activations:
1. Divide the $L$ layers into segments of length $k$.
2. During the forward pass, **only store activations at segment boundaries**:
   $$\mathcal{C} = \left\{\mathbf{a}_0, \mathbf{a}_k, \mathbf{a}_{2k}, \dots, \mathbf{a}_{L - k}\right\}$$
   Total checkpoints stored in memory $= \frac{L}{k}$.
3. During the backward pass for segment $j \in \{\frac{L}{k} - 1, \dots, 0\}$:
   - Start from checkpoint $\mathbf{a}_{j \cdot k}$.
   - Re-run the forward pass for the $k$ layers within the segment to regenerate intermediate activations $\mathbf{a}_{j \cdot k + 1} \dots \mathbf{a}_{(j+1) \cdot k}$.
   - Backpropagate through the segment, computing gradients.
   - Immediately free the regenerated scratch activations.

### 4.2 Deriving the Optimal Interval $k^* = \sqrt{L}$
The peak activation memory held in GPU VRAM during backward execution consists of:
1. The stored checkpoints: $\frac{L}{k}$
2. The transient recomputed segment activations: $k$

Total Peak Memory:
$$M(k) = \frac{L}{k} + k$$

To find the minimum memory footprint, differentiate $M(k)$ with respect to $k$ and set to zero:
$$\frac{dM}{dk} = -\frac{L}{k^2} + 1 = 0 \implies k^2 = L \implies \mathbf{k^* = \sqrt{L}}$$

Substitute $k^* = \sqrt{L}$ back into $M(k)$:
$$M_{\min} = \frac{L}{\sqrt{L}} + \sqrt{L} = 2\sqrt{L} \in \mathbf{\mathcal{O}(\sqrt{L})}$$

#### Concrete Example ($L = 64$ layers):
- **Standard Backpropagation**: Stores all 64 layer activations $\implies \mathbf{64\text{ memory units}}$.
- **Checkpointed ($k = \sqrt{64} = 8$)**: Stores 8 checkpoints $+ 8$ scratch segment activations $\implies \mathbf{16\text{ memory units}}$!
- **Memory Reduction**: **$75\%$ reduction** in activation memory.

---

### 4.3 Mathematical Proof of the 33.3% Compute Overhead

What is the exact computational cost of recomputing the forward activations?

Let $F_{\text{fwd}}$ be the FLOPs required for one forward pass through a layer (e.g. GEMM: $\mathbf{Y} = \mathbf{X}\mathbf{W}$).
In the backward pass, backpropagation must compute two gradients:
1. Gradient with respect to weights: $\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{X}^T \frac{\partial \mathcal{L}}{\partial \mathbf{Y}}$ ($F_{\text{fwd}}\text{ FLOPs}$)
2. Gradient with respect to inputs: $\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} \mathbf{W}^T$ ($F_{\text{fwd}}\text{ FLOPs}$)

Therefore, the backward pass naturally takes twice the compute of the forward pass:
$$F_{\text{bwd}} = 2 \cdot F_{\text{fwd}}$$

#### Work Comparison:
- **Standard Backpropagation Total Work**:
  $$W_{\text{std}} = F_{\text{fwd}} + F_{\text{bwd}} = F_{\text{fwd}} + 2 F_{\text{fwd}} = \mathbf{3 F_{\text{fwd}}}$$
- **Checkpointed Backpropagation Total Work**:
  Each layer runs the forward pass twice (once initially, once during recomputation):
  $$W_{\text{chkpt}} = F_{\text{fwd}} + F_{\text{recompute}} + F_{\text{bwd}} = F_{\text{fwd}} + F_{\text{fwd}} + 2 F_{\text{fwd}} = \mathbf{4 F_{\text{fwd}}}$$

#### Fractional Compute Overhead:
$$\text{Overhead} = \frac{W_{\text{chkpt}} - W_{\text{std}}}{W_{\text{std}}} = \frac{4 F_{\text{fwd}} - 3 F_{\text{fwd}}}{3 F_{\text{fwd}}} = \frac{1}{3} = \mathbf{33.33\%}$$

By paying a modest $33\%$ increase in compute time, activation memory is slashed from $\mathcal{O}(L)$ down to $\mathcal{O}(\sqrt{L})$, allowing batch size or sequence length to expand multi-fold!

---

## 5. FlashAttention: IO-Awareness & Online Softmax

Even with activation checkpointing, standard Self-Attention suffers from a fatal physical memory bottleneck: the **Quadratic HBM Traffic Penalty**.

### 5.1 Standard Attention Memory Bottleneck
Standard Self-Attention executes three separate operations:
$$\mathbf{S} = \mathbf{Q}\mathbf{K}^T \in \mathbb{R}^{S \times S}$$
$$\mathbf{P} = \text{Softmax}(\mathbf{S}) \in \mathbb{R}^{S \times S}$$
$$\mathbf{O} = \mathbf{P}\mathbf{V} \in \mathbb{R}^{S \times D}$$

```
Standard Attention (Memory Unaware):
+---------------------------------------------------------------------------------+
| GPU High Bandwidth Memory (HBM: 2.0 TB/s)                                       |
|  1. Write S x S matrix S to HBM                                                 |
|  2. Read S x S matrix S from HBM -> compute Softmax -> Write S x S matrix P     |
|  3. Read S x S matrix P from HBM -> multiply by V -> Write Output O             |
+---------------------------------------------------------------------------------+
```

Because off-chip HBM bandwidth ($2-3\text{ TB/s}$) is $10\times$ slower than on-chip SRAM ($15-19\text{ TB/s}$), writing and reading the $S \times S$ intermediate matrix repeatedly throttles the GPU. At $S = 64\text{k}$, an $S \times S$ matrix in FP16 requires **$8.5\text{ GB}$ per head**, generating over $25\text{ GB}$ of slow HBM traffic per layer!

---

### 5.2 The Online Softmax Algorithm
How can you compute Softmax without seeing the whole row of $S$ elements?
Milakov & Gimelshein (2018) discovered **Online Softmax**:

Given a vector split into two blocks $\mathbf{x} = [\mathbf{x}^{(1)}, \mathbf{x}^{(2)}]$:
- Let $m^{(1)} = \max(\mathbf{x}^{(1)})$ and $d^{(1)} = \sum_j e^{x_j^{(1)} - m^{(1)}}$.
- When block 2 arrives, let $m^{(2)} = \max(\mathbf{x}^{(2)})$.
- The new combined maximum is:
  $$m^{\text{new}} = \max(m^{(1)}, m^{(2)})$$
- The updated normalizer is rescaled on-the-fly:
  $$d^{\text{new}} = d^{(1)} \cdot e^{m^{(1)} - m^{\text{new}}} + \sum_j e^{x_j^{(2)} - m^{\text{new}}}$$

This allows Softmax and the output product $\mathbf{O} = \mathbf{P}\mathbf{V}$ to be accumulated incrementally in a single loop!

---

### 5.3 FlashAttention Tiling
**FlashAttention** (Dao et al. 2022, 2023) leverages Online Softmax to fuse the entire attention computation into a single GPU kernel:

```
FlashAttention (Tiled Execution in SRAM):
+---------------------------------------------------------------------------------+
| On-Chip SRAM / Shared Memory (~192 KB per SM, 15 TB/s Bandwidth)                |
|                                                                                 |
|  Tile Q into SRAM (Br x d) <----+                                               |
|  Tile K, V into SRAM (Bc x d) <-+                                               |
|                                                                                 |
|  1. Compute local dot products S_tile = Q_tile * K_tile^T in fast SRAM          |
|  2. Update running Softmax statistics (m, d) using Online Softmax               |
|  3. Accumulate directly into Output O_tile = P_tile * V_tile in SRAM            |
+---------------------------------------------------------------------------------+
                                      |
       Only the final Output O is written back to HBM! (Zero S x S writes)
                                      v
+---------------------------------------------------------------------------------+
| GPU High Bandwidth Memory (HBM: 2.0 TB/s)                                       |
|  Read Q, K, V (O(S) bytes) ------------> Write Output O (O(S) bytes)           |
+---------------------------------------------------------------------------------+
```

### Quantitative Advantages:
1. **Memory IO Complexity**: Drops from **$\mathcal{O}(S^2)$** to **$\mathcal{O}(S)$**.
2. **Speed**: $2\times$ to $4\times$ real wall-clock speedup on A100/H100 compared to standard PyTorch attention.
3. **Memory Footprint**: Does not store the $S \times S$ matrix for the backward pass! In backward, it recomputes attention blocks directly from $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ in SRAM.
4. **Context Length**: Enables scaling context lengths from 2,048 tokens to 128,000+ tokens on standard hardware.
