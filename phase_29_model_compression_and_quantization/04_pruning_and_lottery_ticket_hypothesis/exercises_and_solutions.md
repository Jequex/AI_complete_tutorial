# Topic 04: Exercises & Solutions — Pruning & The Lottery Ticket Hypothesis

---

### Exercise 1: Combinatorics & Metadata Encoding of NVIDIA 2:4 Semi-Structured Sparsity
**Problem:**
NVIDIA Ampere and Hopper architectures feature Sparse Tensor Cores that exploit 2:4 structured sparsity.
In every contiguous group of 4 elements across a matrix row, exactly 2 elements must be zero and 2 are non-zero.
1. Compute the total number of distinct allowable non-zero patterns in a 4-element block.
2. Show that a 4-bit field is sufficient to store the positions of the non-zero elements for two 4-element blocks (i.e. exactly 2 bits per non-zero pair).
3. For a weight matrix of size $1024 \times 1024$ in FP16 (2 bytes per element), compute the total memory storage required under 2:4 sparsity including metadata, and contrast it with full dense storage.

<details>
<summary><strong>Solution</strong></summary>

1. **Combinatorial Configurations:**
   The number of ways to choose 2 non-zero positions out of 4 consecutive slots is given by the binomial coefficient:
   $$\binom{4}{2} = \frac{4!}{2! 2!} = \frac{24}{4} = 6 \text{ allowable patterns}$$
   The six possible binary masks are:
   $$(1, 1, 0, 0), \; (1, 0, 1, 0), \; (1, 0, 0, 1), \; (0, 1, 1, 0), \; (0, 1, 0, 1), \; (0, 0, 1, 1)$$

2. **Index Metadata Encoding:**
   Since there are only 6 valid configurations, $\lceil \log_2(6) \rceil = 3$ bits would suffice to index the configuration.
   In NVIDIA's hardware implementation, each of the 2 non-zero entries stores its local column index within the 4-element block $\{0, 1, 2, 3\}$.
   Because each local index requires $\log_2(4) = 2$ bits, the two non-zero entries together require:
   $$2 \text{ entries} \times 2 \text{ bits/entry} = 4 \text{ bits of metadata per 4-element block}$$
   Equivalently, 1 byte (8 bits) stores the index metadata for two consecutive 4-element blocks (8 original weights).

3. **Storage Comparison on a $1024 \times 1024$ FP16 Matrix:**
   - **Dense FP16 Matrix**:
     $$1024 \times 1024 \times 2 \text{ bytes} = 2,097,152 \text{ bytes} = 2.00 \text{ MB}$$
   - **2:4 Sparse Matrix**:
     - Number of non-zero FP16 weights: $\frac{1024 \times 1024}{2} = 524,288$ weights.
     - Weight storage: $524,288 \times 2 \text{ bytes} = 1,048,576 \text{ bytes} = 1.00 \text{ MB}$.
     - Metadata storage: The matrix contains $\frac{1024 \times 1024}{4} = 262,144$ blocks of 4.
       Each block requires 4 bits (0.5 bytes) of metadata:
       $$262,144 \times 0.5 \text{ bytes} = 131,072 \text{ bytes} = 0.125 \text{ MB}$$
     - Total 2:4 storage:
       $$1.00 \text{ MB} + 0.125 \text{ MB} = 1.125 \text{ MB}$$
   - **Compression Ratio**:
     $$\frac{2.00 \text{ MB}}{1.125 \text{ MB}} \approx 1.78\times \text{ memory footprint reduction}$$
     Coupled with a hardware-guaranteed $2.00\times$ speedup on Sparse Tensor Cores!
</details>

---

### Exercise 2: Mathematical Derivation of the Iterative Magnitude Pruning (IMP) Rate
**Problem:**
Suppose you wish to prune a dense neural network from 0% sparsity to a target final sparsity of $S_{\text{final}} \in (0, 1)$ (e.g. $S_{\text{final}} = 0.80$, meaning 80% of parameters are zero) across $R$ discrete pruning rounds.
In each round $r \in \{1, 2, \dots, R\}$, a constant percentage $p$ of the currently surviving weights is pruned.
1. Express the fraction of surviving weights after round $r$ in terms of $p$.
2. Derive the closed-form formula for the per-round pruning rate $p$ required to achieve final sparsity $S_{\text{final}}$ in exactly $R$ rounds.
3. Compute $p$ numerically for $S_{\text{final}} = 0.85$ and $R = 8$ rounds.

<details>
<summary><strong>Solution</strong></summary>

1. **Surviving Fraction After Round $r$:**
   At round 0 (dense), the surviving parameter fraction is $s_0 = 1.0$.
   At round 1, a fraction $p$ is removed, leaving $s_1 = 1 - p$.
   At round 2, a fraction $p$ of the remaining weights is removed:
   $$s_2 = s_1 (1 - p) = (1 - p)^2$$
   By induction, after $r$ rounds:
   $$s_r = (1 - p)^r$$

2. **Per-Round Pruning Rate $p$:**
   At round $R$, the final surviving fraction is:
   $$s_R = 1 - S_{\text{final}}$$
   Equating the two expressions:
   $$(1 - p)^R = 1 - S_{\text{final}}$$
   Taking the $R$-th root of both sides:
   $$1 - p = (1 - S_{\text{final}})^{1 / R}$$
   Solving for $p$:
   $$\boxed{p = 1 - (1 - S_{\text{final}})^{1 / R}}$$

3. **Numerical Computation:**
   For $S_{\text{final}} = 0.85$ and $R = 8$:
   $$1 - S_{\text{final}} = 1 - 0.85 = 0.15$$
   $$1 - p = (0.15)^{1 / 8} = (0.15)^{0.125} \approx 0.78917$$
   $$p = 1 - 0.78917 = 0.21083 \approx \mathbf{21.08\%}$$
   Pruning approximately $21.1\%$ of active weights in each of the 8 rounds gently reaches 85% overall sparsity without catastrophic network collapse.
</details>

---

### Exercise 3: Structured vs. Unstructured FLOP Reduction Analysis
**Problem:**
Consider a dense fully connected linear layer computing $\mathbf{y} = \mathbf{x} \mathbf{W}$ where $\mathbf{x} \in \mathbb{R}^{1 \times K}$ and $\mathbf{W} \in \mathbb{R}^{K \times N}$.
The layer requires $2KN$ FLOPs (multiply and accumulate for each entry).
1. Suppose we apply 50% unstructured pruning. Explain why standard dense matrix multiplication algorithms (`GEMM`) still execute all $2KN$ FLOPs on unmodified GPUs.
2. Suppose we apply 50% structured column pruning, cutting $N$ down to $N' = 0.5 N$. Calculate the exact FLOP count and latency reduction.

<details>
<summary><strong>Solution</strong></summary>

1. **Unstructured Pruning on Dense Runtimes:**
   Standard dense BLAS / cuBLAS kernels load continuous vectors of numbers from memory into hardware registers and execute SIMD/vector multiply-accumulate instructions (`FMA`).
   Because unstructured zeros are scattered unpredictably throughout the matrix, the hardware still loads the zeros and multiplies by zero:
   $$x_k \times 0.0 + \text{acc} = \text{acc}$$
   The exact same $2KN$ floating-point instructions are executed, and the exact same memory bandwidth is consumed to stream the zero entries from VRAM into registers. Thus, on standard dense execution engines, unstructured pruning yields **zero reduction in FLOPs or runtime latency**.

2. **Structured Column Pruning:**
   When 50% of columns are removed, the weight matrix is physically reshaped at compile time into $\mathbf{W}' \in \mathbb{R}^{K \times (N/2)}$.
   The matrix product is now:
   $$\mathbf{y}' = \mathbf{x} \mathbf{W}'$$
   The required FLOP count is:
   $$2 \times K \times \frac{N}{2} = K N \text{ FLOPs}$$
   This yields a **guaranteed 50% reduction in FLOPs** ($2\times$ reduction) and halves the memory traffic required to stream the layer weights, resulting in an immediate near-$2\times$ speedup on any GPU, CPU, or embedded NPU.
</details>
