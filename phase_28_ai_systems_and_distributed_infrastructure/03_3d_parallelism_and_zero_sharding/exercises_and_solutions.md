# Topic 03: Exercises & Solutions — 3D Parallelism & ZeRO Sharding

## Exercise 1: Sizing 3D Parallelism for a 175B Parameter Model
**Problem:**
You are sizing a distributed training cluster for a 175-billion parameter model ($\Phi = 175 \times 10^9$) with 96 Transformer layers, hidden dimension $H = 12{,}288$, and 96 attention heads.
- Available Hardware: $N = 512\text{ NVIDIA A100 (80GB)}$ GPUs arranged in 64 HGX nodes (8 GPUs per node connected via NVLink $600\text{ GB/s}$; nodes connected via $200\text{ Gbps}$ InfiniBand HDR).
- Global batch size: 1,536 sequences.

1. Why should the Tensor Parallel degree $T$ be chosen as $T = 8$? Why not $T = 16$?
2. If $T = 8$ and we choose Pipeline Parallel degree $p = 8$, what is the resulting Data Parallel degree $D$?
3. To achieve a pipeline bubble fraction $F \le 8\%$, what is the minimum number of micro-batches $m$ required?
4. What is the micro-batch size per pipeline if the global batch size is 1,536?

<details>
<summary>Click to view Solution</summary>

### Solution:

#### 1. Tensor Parallel Degree $T = 8$:
- Tensor Parallelism requires 2 All-Reduces per Transformer layer (one for Attention, one for MLP). This generates massive communication frequency that requires the ultra-high bandwidth of NVLink ($600\text{ GB/s}$).
- Each physical node contains exactly 8 GPUs connected via NVLink. Choosing $T = 8$ keeps all TP communication strictly within the physical node.
- If $T = 16$, TP would have to cross the InfiniBand network ($25\text{ GB/s}$), creating a catastrophic $24\times$ network bottleneck that degrades training throughput.
- Furthermore, 96 attention heads divided by $T = 8$ yields 12 heads per GPU (an exact integer).

#### 2. Data Parallel Degree $D$:
The total cluster size satisfies:
$$N = D \times p \times T \implies 512 = D \times 8 \times 8 = D \times 64$$
$$D = \frac{512}{64} = \mathbf{8}$$

#### 3. Minimum Micro-Batches $m$:
The pipeline bubble fraction is given by:
$$F = \frac{p - 1}{m + p - 1} \le 0.08$$
Substitute $p = 8$:
$$\frac{7}{m + 7} \le 0.08 \implies 7 \le 0.08m + 0.56 \implies 0.08m \ge 6.44 \implies m \ge \frac{6.44}{0.08} = 80.5$$
Thus, we require at least $m = 81$ micro-batches (or practically $m = 96$ or $128$ for clean divisibility) to maintain $F \le 8\%$.

#### 4. Micro-Batch Size:
Global batch size $B = 1{,}536$.
With Data Parallel degree $D = 8$, each DP replica processes:
$$B_{\text{DP}} = \frac{1{,}536}{8} = 192\text{ sequences}$$
If $m = 96$ micro-batches are used per pipeline:
$$\text{Micro-batch size } b = \frac{192}{96} = \mathbf{2\text{ sequences per micro-batch}}$$
</details>

---

## Exercise 2: Megatron-LM Tensor Parallel All-Reduce Volume
**Problem:**
Consider a Transformer layer with hidden dimension $H = 4{,}096$ and sequence length $S = 2{,}048$ with batch size $B = 4$.
Precision is FP16 ($2\text{ bytes}$ per element).
Tensor Parallel degree is $T = 4$.

1. Identify every point in a single Transformer layer (Attention + MLP) where an All-Reduce operation is required.
2. What is the tensor size (elements and bytes) transmitted in each All-Reduce?
3. Calculate the total data transmitted per GPU for one complete forward and backward pass through this single layer.

<details>
<summary>Click to view Solution</summary>

### Solution:

#### 1. Locations of All-Reduce Operations:
In Megatron-LM tensor partitioning:
- **Attention Block**:
  - $Q, K, V$ projections are Column-Parallel (no communication).
  - Attention scores and context vectors are computed locally per head partition.
  - Output projection $W_O$ is Row-Parallel $\implies$ **1 All-Reduce** at the end of the Attention block.
- **MLP Block**:
  - Gate/Up projection $W_1$ is Column-Parallel (no communication).
  - GeLU non-linearity is applied locally on sliced activations (no communication).
  - Down projection $W_2$ is Row-Parallel $\implies$ **1 All-Reduce** at the end of the MLP block.

Total: **2 All-Reduces in the forward pass** and **2 All-Reduces in the backward pass** (gradient of input) per layer.

#### 2. Tensor Size per All-Reduce:
The output of both the Attention projection and the MLP projection is a tensor of shape $[B, S, H]$:
- Elements $= B \times S \times H = 4 \times 2{,}048 \times 4{,}096 = 33{,}554{,}432\text{ elements}$.
- In FP16 ($2\text{ bytes/element}$):
  $$M_{\text{tensor}} = 33{,}554{,}432 \times 2 = \mathbf{67{,}108{,}864\text{ bytes}} = \mathbf{67.11\text{ MB}}$$

#### 3. Communication Volume per GPU per Layer:
Using Ring All-Reduce across $T = 4$ GPUs, each All-Reduce transfers:
$$V_{\text{AR}} = 2 \cdot \frac{T - 1}{T} M = 2 \cdot \frac{3}{4} \times 67.11\text{ MB} = 1.5 \times 67.11\text{ MB} \approx \mathbf{100.66\text{ MB}}$$
With 2 All-Reduces in forward and 2 in backward:
$$V_{\text{layer}} = 4 \times 100.66\text{ MB} \approx \mathbf{402.66\text{ MB per layer}}$$
For an 80-layer model, this requires $\approx 32.2\text{ GB}$ of data transfer per step, proving why NVLink ($900\text{ GB/s}$) is mandatory.
</details>

---

## Exercise 3: ZeRO-2 vs ZeRO-3 on an 8-GPU Node
**Problem:**
You have a single 8-GPU node ($N = 8$) of NVIDIA A100 (80GB). You want to train a 13-billion parameter model ($\Phi = 13 \times 10^9$) in mixed precision ($2\Phi$ weights, $2\Phi$ grads, $12\Phi$ Adam states).
Activation memory and temporary buffers require $25\text{ GB}$.

1. Calculate the model state memory per GPU under Baseline DDP, ZeRO-1, ZeRO-2, and ZeRO-3.
2. Determine which strategies can successfully train without Out-of-Memory (OOM) errors.
3. Compare the network communication volume per GPU per step between ZeRO-2 and ZeRO-3.

<details>
<summary>Click to view Solution</summary>

### Solution:

#### 1. Model State Memory Calculation:
Total model state bytes $= 16\Phi = 16 \times 13 \times 10^9 = 208\text{ GB}$.
- **Baseline DDP**:
  $$\text{Mem} = 16\Phi = \mathbf{208.0\text{ GB}}$$
- **ZeRO-1 ($P_{os}$)**:
  $$\text{Mem} = 4\Phi + \frac{12\Phi}{N} = (4 \times 13) + \frac{12 \times 13}{8} = 52 + 19.5 = \mathbf{71.5\text{ GB}}$$
- **ZeRO-2 ($P_{os+g}$)**:
  $$\text{Mem} = 2\Phi + \frac{14\Phi}{N} = (2 \times 13) + \frac{14 \times 13}{8} = 26 + 22.75 = \mathbf{48.75\text{ GB}}$$
- **ZeRO-3 ($P_{os+g+p}$)**:
  $$\text{Mem} = \frac{16\Phi}{N} = \frac{208}{8} = \mathbf{26.0\text{ GB}}$$

#### 2. OOM Assessment (Total Budget = 80 GB, Available for Model States = $80 - 25 = 55\text{ GB}$):
- Baseline DDP ($208\text{ GB} > 55\text{ GB}$): **OOM** (Fails).
- ZeRO-1 ($71.5\text{ GB} > 55\text{ GB}$): **OOM** (Fails).
- ZeRO-2 ($48.75\text{ GB} \le 55\text{ GB}$): **Fits!** Total memory $= 48.75 + 25 = 73.75\text{ GB} \le 80\text{ GB}$.
- ZeRO-3 ($26.0\text{ GB} \le 55\text{ GB}$): **Fits with generous margin!** Total memory $= 26 + 25 = 51.0\text{ GB}$.

#### 3. Communication Volume Comparison:
- **ZeRO-2 Volume**: $2\Phi = 2 \times 13 \times 2\text{ bytes} = \mathbf{52\text{ GB}}$ per step.
- **ZeRO-3 Volume**: $3\Phi = 3 \times 13 \times 2\text{ bytes} = \mathbf{78\text{ GB}}$ per step ($1.5\times$ more).
Because ZeRO-2 fits within memory ($73.75\text{ GB} < 80\text{ GB}$), **ZeRO-2 is the optimal choice here**, avoiding the extra $26\text{ GB}$ of parameter gathering communication.
</details>
