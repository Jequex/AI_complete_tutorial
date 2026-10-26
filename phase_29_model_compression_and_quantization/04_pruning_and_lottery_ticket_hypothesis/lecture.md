# Topic 04: Pruning & The Lottery Ticket Hypothesis

## 1. Introduction & 12-Year-Old Intuition

Imagine you are auditioning kids for a competitive high school basketball team.
You invite **100 kids to tryouts** (the dense neural network).
- During the first week of intensive training drills, 15 kids stand out: they have incredible natural hand-eye coordination, quick reflexes, and an innate sense of court awareness.
- At the end of tryouts, the coach cuts the other 85 players and keeps only the **15 star players** (pruning to 85% sparsity).
- When those 15 kids play together as a focused squad, they win the championship!

Now, the coach asks a profound question:
*"Could we have picked these 15 kids on Day 1 without ever inviting the other 85 kids?"*

In 2018, Jonathan Frankle and Michael Carbin investigated this question in neural networks and discovered something startling:
1. **The Winning Ticket**: If you take the exact 15 kids that survived cuts, and reset their skills back to their **exact Day 1 initial form** (rewinding weights $\theta_0$), and train only them from scratch, they train fast and win the championship again!
2. **The Random Control**: If you keep the same roster positions but invite **15 different random kids** off the street (random reinitialization $\theta_0'$), they lack the initial natural chemistry and fail to win games.

This is the **Lottery Ticket Hypothesis**:
> Inside every large, randomly-initialized dense network, there hides a lucky sparse subnetwork—a **"winning ticket"**—that, when trained in isolation from its original starting weights, matches or exceeds the full dense network's accuracy!

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Formal Type | Intuition / Operational Role |
| :--- | :--- | :--- | :--- |
| $\theta \in \mathbb{R}^D$ | Model Parameters | Parameter vector | All trainable weights across all layers in the neural network. |
| $\theta_0 \in \mathbb{R}^D$ | Initial Weights | Parameter vector | Master weight values at time step $t = 0$ (immediately after random initialization). |
| $\mathbf{m} \in \{0, 1\}^D$ | Pruning Mask | Binary vector | Multiplicative indicator mask ($m_i = 1$ active, $m_i = 0$ pruned/zeroed). |
| $\mathbf{m} \odot \theta$ | Masked Subnetwork | Parameter vector | Element-wise Hadamard product applying the sparsity pattern to weights. |
| $s \in [0, 1)$ | Sparsity Ratio | Scalar $\in [0, 1)$ | Percentage of total network weights set permanently to zero ($s = 1 - \frac{\|\mathbf{m}\|_0}{D}$). |
| $\tau \in \mathbb{R}^+$ | Pruning Threshold | Positive real scalar | Magnitude cutoff: weights with $|\theta_i| < \tau$ are zeroed out. |
| $N:M$ | Semi-Structured Sparsity | Ratio constraint | NVIDIA hardware constraint: exactly $N$ out of every $M$ contiguous elements are non-zero (e.g. 2:4). |
| $\theta_k$ | Rewound State | Parameter vector | Weights saved at early epoch $k \approx 500-1000$ used for rewinding deep vision/LLM architectures. |

---

## 3. The Pruning Taxonomy: Unstructured vs. Structured vs. Semi-Structured (2:4)

Not all sparsity is created equal. The mechanical structure of zeroed weights dictates whether pruning actually speeds up physical hardware or merely saves disk space.

```
1. Unstructured Sparsity (Irregular Dots):
[ .  X  .  . ]
[ X  .  X  . ]  ---> High theoretical compression (90%),
[ .  .  .  X ]       Zero acceleration on standard GPUs without sparse libraries.
[ X  X  .  . ]

2. Structured Sparsity (Removed Columns / Rows):
[ X  .  X  . ]       [ X  X ]
[ X  .  X  . ]  ===> [ X  X ]  ---> Physical matrix dimension shrinks!
[ X  .  X  . ]       [ X  X ]       Immediate 2x-4x speedup on any hardware.
[ X  .  X  . ]       [ X  X ]

3. Semi-Structured 2:4 Sparsity (NVIDIA Ampere / Hopper):
[ X  .  X  . ][ .  X  .  X ]   ---> Exactly 2 non-zeros per 4-element block.
[ .  X  X  . ][ X  .  .  X ]        2x throughput natively on NVIDIA Sparse Tensor Cores!
```

### 3.1 Unstructured Pruning (Fine-Grained Magnitude Pruning)
Individual scalar parameters are removed independently without regard to spatial location:
$$m_i = \begin{cases} 1, & |\theta_i| \ge \tau \\ 0, & |\theta_i| < \tau \end{cases}$$
- **Pros**: Highest mathematical parameter reduction (up to 90–95% sparsity on CNNs and Transformers with minimal loss).
- **Cons**: Creates irregular, random memory access patterns. Dense GPU hardware (`GEMM`) cannot skip individual zero multiplications efficiently. Requires compressed sparse formats (CSR/CSC) and specialized sparse matrix libraries (e.g. cuSPARSE).

### 3.2 Structured Pruning (Coarse-Grained Channel/Head Pruning)
Removes entire contiguous architectural blocks:
- **CNNs**: Prunes entire convolution filters or input feature maps.
- **Transformers**: Prunes entire multi-head attention heads, MLP intermediate neurons, or transformer layers.
- **Pros**: Directly shrinks tensor dimensions from $(M \times K)$ to $(M \times K')$. Runs seamlessly on any unmodified standard hardware runtime (CPU, GPU, mobile NPU) with immediate $2\times$ to $4\times$ throughput speedup and memory bandwidth reduction.
- **Cons**: Much more damaging to model expressivity. High structured sparsity (>50–60%) typically incurs steep accuracy drops.

### 3.3 Semi-Structured N:M Sparsity (The Hardware Sweet Spot)
Introduced in NVIDIA's Ampere architecture (A100, RTX 3090, H100), **2:4 Semi-Structured Sparsity** mandates that in every consecutive window of 4 weights across a matrix row, **exactly 2 must be zero and 2 may be non-zero**:
$$\forall k \in \{0, 1, \dots, \frac{K}{4} - 1\}: \quad \sum_{j=0}^3 m_{4k + j} = 2$$
- **Hardware Architecture**: NVIDIA Sparse Tensor Cores store only the 2 non-zero FP16/INT8 values plus a 2-bit index metadata pointer.
- **Benefit**: Doubled mathematical throughput ($2\times$ FLOPS) and halved memory fetch traffic, while retaining virtually identical accuracy to the dense baseline because local granularity is preserved.

---

## 4. Iterative Magnitude Pruning (IMP)

Why does one-shot pruning (pruning 80% of weights all at once) fail?
When 80% of connections vanish simultaneously, internal activation distributions undergo massive covariate shift. Subsequent layers receive sudden, catastrophic signal collapse.

**Iterative Magnitude Pruning (IMP)** removes weights gradually over $T$ sequential cycles:

```
[ Dense Model theta_0 ]
          │
          ▼
   [ Train to theta_T ]
          │
          ▼ ─── Round 1: Prune 20% of weights (Mask m_1)
   [ Retrain theta_T ]
          │
          ▼ ─── Round 2: Prune 20% of remaining weights (Mask m_2)
   [ Retrain theta_T ]
          │
          ▼ ─── Round T: Target Sparsity Reached!
   [ Final Sparse Model ]
```

At each round $t$, the threshold $\tau_t$ is chosen such that a fraction $p$ (e.g. 20%) of currently active weights are zeroed:
$$p_t = 1 - (1 - p_{\text{final}})^{t / T}$$
By interleaving pruning with recovery training epochs, the remaining weights gently adapt their magnitudes to compensate for pruned connections.

---

## 5. The Lottery Ticket Hypothesis: Formal Framework

Formulated by Jonathan Frankle and Michael Carbin (ICLR 2019 Best Paper Award):

> **Theorem (Lottery Ticket Hypothesis):**
> Consider a dense feedforward neural network $f(x; \theta)$ with initial parameters $\theta_0 \sim \mathcal{D}_\theta$.
> When trained with stochastic gradient descent to completion, the network reaches test accuracy $A$ in $t^*$ training iterations.
> Then there exists a binary subnetwork mask $\mathbf{m} \in \{0, 1\}^D$ with sparsity $s = 1 - \frac{\|\mathbf{m}\|_0}{D} \gg 0$ such that the subnetwork $f(x; \mathbf{m} \odot \theta_0)$, trained in isolation, satisfies:
> 1. **Accuracy**: Reaches test accuracy $A' \ge A$.
> 2. **Efficiency**: Converges in training iterations $t \le t^*$.

### 5.1 The Rewinding Protocol
1. Initialize a dense neural network: $\theta_0 \sim \mathcal{P}_{\text{init}}$.
2. Train the network for $T$ steps to reach converged weights $\theta_T$.
3. Prune the lowest magnitude weights by creating mask $\mathbf{m}$:
   $$m_i = \mathbb{I}(|\theta_{T, i}| \ge \tau)$$
4. **The Winning Ticket**: Reset the surviving parameters to their original starting values at step 0:
   $$\theta_{\text{ticket}} = \mathbf{m} \odot \theta_0$$
5. Train $\theta_{\text{ticket}}$ on the task. It will converge to full dense accuracy!

### 5.2 The Random Control Test (Proving the Hypothesis)
To prove that $\theta_0$ contains unique inductive bias and is not merely an arbitrary sparsity topology, Frankle & Carbin tested the **Random Control Ticket**:
1. Take the exact same winning mask $\mathbf{m}$.
2. Re-sample new random initial weights: $\theta_0' \sim \mathcal{P}_{\text{init}}$.
3. Train $\theta_{\text{random}} = \mathbf{m} \odot \theta_0'$.

**Experimental Finding**: The random control ticket $\mathbf{m} \odot \theta_0'$ fails to match baseline accuracy and suffers severe degradation or divergence at high sparsity. The winning ticket requires **BOTH** the structural mask $\mathbf{m}$ **AND** the synchronized initial weight coordinates $\theta_0$.

### 5.3 Weight Rewinding to Epoch $k$ (Deep Architectures & LLMs)
For deeper networks (ResNet-50, VGG-16, Transformers), Frankle et al. (2019) discovered that rewinding all the way back to $t = 0$ sometimes fails due to early training stochasticity.
Instead, they introduced **Weight Rewinding to Iteration $k$**:
$$\theta_{\text{rewind}} = \mathbf{m} \odot \theta_k \quad (k \approx 500 - 1000 \text{ steps})$$
At iteration $k$, the network has settled into a stable optimization basin (the "linear mode connectivity" region), after which the sparse subnetwork trains with complete stability.

---

## 6. Engineering Comparison: Pruning Approaches

| Metric | Unstructured IMP | Structured Pruning | Semi-Structured 2:4 |
| :--- | :--- | :--- | :--- |
| **Sparsity Level** | 80% – 95% | 30% – 50% | Fixed 50% |
| **Accuracy Retention** | Exceptional (Matches dense) | Moderate (Drops past 50%) | Exceptional (<0.5% drop) |
| **Hardware Support** | Specialized sparse kernels | Any standard CPU/GPU | NVIDIA Ampere & Hopper |
| **Speedup Source** | Storage / cuSPARSE | Dense matrix shrinkage | Sparse Tensor Cores ($2\times$ GEMM) |
| **Indices Overhead** | $O(N)$ CSR/CSC pointers | Zero overhead | 2-bit per 2 non-zeros |
