# Topic 18.3: Exercises & Solutions — Distributed Training & Parallelism Strategies

---

## Exercise 1: Memory Wall & ZeRO Sharding Sizing for a 175B Model

### Problem
An AI research team prepares to train a 175-billion-parameter model using mixed precision (BF16) and the AdamW optimizer on a cluster of NVIDIA H100 GPUs (80 GB HBM per GPU).
1. Calculate the total monolithic static memory footprint $M_{\text{static}}$ (parameters, gradients, FP32 master weights, and optimizer momentum states).
2. For cluster sizes $N_d \in \{16, 64, 256\}$ GPUs, compute the static memory per GPU under:
   - DDP (ZeRO-0)
   - ZeRO-1 ($P_{os}$)
   - ZeRO-2 ($P_{os+g}$)
   - ZeRO-3 ($P_{os+g+p}$ / FSDP)
3. What is the absolute minimum number of 80GB GPUs required to fit the static state under ZeRO-3?
4. If activation memory requires an additional $20\text{ GB}$ per GPU with selective activation checkpointing, how many GPUs are needed so total memory $\le 80\text{ GB}$?

---

### Solution

#### Step 1: Monolithic Static Memory Footprint
$$\text{Parameters } W \text{ (BF16)}: 175 \times 10^9 \times 2 \text{ bytes} = 350 \text{ GB}$$
$$\text{Gradients } \nabla_W L \text{ (BF16)}: 175 \times 10^9 \times 2 \text{ bytes} = 350 \text{ GB}$$
$$\text{AdamW Master Weights } W^* \text{ (FP32)}: 175 \times 10^9 \times 4 \text{ bytes} = 700 \text{ GB}$$
$$\text{AdamW 1st Momentum } m_t \text{ (FP32)}: 175 \times 10^9 \times 4 \text{ bytes} = 700 \text{ GB}$$
$$\text{AdamW 2nd Momentum } v_t \text{ (FP32)}: 175 \times 10^9 \times 4 \text{ bytes} = 700 \text{ GB}$$
$$M_{\text{static}} = 350 + 350 + 700 + 700 + 700 = 2,800 \text{ GB (2.8 Terabytes)}$$

#### Step 2: Memory per GPU Across Cluster Sizes
Formulas ($N = 175 \text{B}$):
- DDP: $16N = 2,800 \text{ GB}$ on every GPU.
- ZeRO-1: $4N + \frac{12N}{N_d} = 700 + \frac{2100}{N_d}$ GB.
- ZeRO-2: $2N + \frac{14N}{N_d} = 350 + \frac{2450}{N_d}$ GB.
- ZeRO-3: $\frac{16N}{N_d} = \frac{2800}{N_d}$ GB.

Evaluating across cluster sizes:
- **At $N_d = 16$ GPUs:**
  - DDP: $2,800 \text{ GB}$ (OOM)
  - ZeRO-1: $700 + \frac{2100}{16} = 700 + 131.25 = 831.25 \text{ GB}$ (OOM)
  - ZeRO-2: $350 + \frac{2450}{16} = 350 + 153.1 = 503.1 \text{ GB}$ (OOM)
  - ZeRO-3: $\frac{2800}{16} = \mathbf{175.0 \text{ GB}}$ (OOM on 80GB GPUs)
- **At $N_d = 64$ GPUs:**
  - DDP: $2,800 \text{ GB}$ (OOM)
  - ZeRO-1: $700 + \frac{2100}{64} = 700 + 32.8 = 732.8 \text{ GB}$ (OOM)
  - ZeRO-2: $350 + \frac{2450}{64} = 350 + 38.3 = 388.3 \text{ GB}$ (OOM)
  - ZeRO-3: $\frac{2800}{64} = \mathbf{43.75 \text{ GB}}$ (FITS in 80GB HBM!)
- **At $N_d = 256$ GPUs:**
  - ZeRO-3: $\frac{2800}{256} = \mathbf{10.94 \text{ GB}}$ per GPU.

#### Step 3: Minimum Number of GPUs for Static Memory
Under ZeRO-3:
$$\frac{2800}{N_d} \le 80 \implies N_d \ge \frac{2800}{80} = 35 \text{ GPUs}$$
Rounding up to standard powers of two: **minimum 64 GPUs**.

#### Step 4: With 20 GB Activation Budget
$$\text{Max Static Memory Budget} = 80 - 20 = 60 \text{ GB}$$
$$\frac{2800}{N_d} \le 60 \implies N_d \ge \frac{2800}{60} \approx 46.67 \text{ GPUs}$$
Rounding up to powers of two: **64 GPUs** (where static memory is $43.75\text{ GB} + 20\text{ GB} = 63.75\text{ GB} \le 80\text{ GB}$).

---

## Exercise 2: Megatron-LM Tensor Parallelism Communication Overhead

### Problem
Consider a Transformer layer with hidden dimension $d = 8192$, batch size $B = 4$, and sequence length $S = 4096$, partitioned across a Tensor Parallelism group of $K = 8$ GPUs using Megatron-LM.
1. Identify all operations in the forward pass that require collective communication and state the communication primitive.
2. In a standard ring All-Reduce across $K$ GPUs, the volume of data sent and received by each GPU for a tensor of $T$ elements in 16-bit precision (2 bytes) is:
   $$\text{Bytes} = 2 \cdot \left(\frac{K - 1}{K}\right) \cdot T \cdot 2 \text{ bytes}$$
   Calculate the exact number of megabytes transferred per GPU for the Self-Attention output projection $W_O$.
3. Repeat the calculation for the MLP output projection $W_{down}$.
4. If NVLink provides $600\text{ GB/s}$ unidirectional bandwidth per GPU, compute the communication latency per Transformer layer.

---

### Solution

#### Step 1: Communication Operations in Forward Pass
1. **Self-Attention:** Projections $W_Q, W_K, W_V$ are column-parallel. They execute locally without communication. The output projection $W_O$ is row-parallel, producing partial sums that require **one `All-Reduce (SUM)`**.
2. **MLP Block:** Projections $W_{gate}, W_{up}$ are column-parallel, executing locally without communication. The output projection $W_{down}$ is row-parallel, requiring **one `All-Reduce (SUM)`**.
Total forward communication: **2 All-Reduce operations per Transformer layer**.

#### Step 2: Attention All-Reduce Communication Volume
Tensor shape: $(B \times S \times d) = (4 \times 4096 \times 8192) = 134,217,728$ elements ($2^{27}$ elements).
Each element in FP16/BF16 is 2 bytes:
$$\text{Tensor Size} = 134,217,728 \times 2 = 268,435,456 \text{ bytes (256 MB)}$$
Ring All-Reduce volume per GPU:
$$\text{Volume} = 2 \cdot \left(\frac{8 - 1}{8}\right) \times 256 \text{ MB} = 2 \cdot \left(\frac{7}{8}\right) \times 256 \text{ MB} = \frac{7}{4} \times 256 \text{ MB} = 448 \text{ MB}$$

#### Step 3: MLP All-Reduce Communication Volume
The output of $W_{down}$ also has shape $(B \times S \times d)$, identical to the Attention output.
$$\text{MLP All-Reduce Volume} = 448 \text{ MB}$$

#### Step 4: Total Forward Communication & Latency
$$\text{Total Transferred per Layer} = 448 \text{ MB} + 448 \text{ MB} = 896 \text{ MB (0.896 GB)}$$
At $600 \text{ GB/s}$ NVLink interconnect bandwidth:
$$\text{Communication Latency} = \frac{0.896 \text{ GB}}{600 \text{ GB/s}} \approx 1.49 \times 10^{-3} \text{ seconds (1.49 milliseconds)}$$
Because NVLink is so fast ($600\text{ GB/s}$ vs standard $12.5\text{ GB/s}$ 100GbE network), the 1.5 ms communication overhead is completely hidden behind the tens of milliseconds of tensor core compute!

---

## Exercise 3: Pipeline Parallelism Bubble Fraction and 1F1B Memory Optimization

### Problem
A foundation model is distributed across $p = 8$ pipeline stages, processing a global batch split into $m$ microbatches.
1. Formulate the pipeline bubble fraction $F_{\text{bubble}}$ as a function of $p$ and $m$.
2. Calculate $F_{\text{bubble}}$ for $m = 8$, $m = 32$, and $m = 64$.
3. Under GPipe scheduling, how many microbatch activation tensors must Stage 0 hold in memory before its first backward step?
4. Under 1F1B (One-Forward-One-Backward) scheduling, what is the maximum number of activation tensors Stage 0 holds in memory?

---

### Solution

#### Step 1: Pipeline Bubble Formula
In an ideal pipeline of $p$ stages with $m$ microbatches, the total execution time is:
$$T_{\text{total}} = (m + p - 1) \cdot t_{\text{step}}$$
The non-idle compute time is $m \cdot t_{\text{step}}$, meaning the idle bubble time is $(p - 1) \cdot t_{\text{step}}$.
The bubble fraction is:
$$F_{\text{bubble}} = \frac{p - 1}{m + p - 1}$$

#### Step 2: Numerical Evaluation ($p = 8$)
- For $m = 8$:
  $$F_{\text{bubble}} = \frac{8 - 1}{8 + 8 - 1} = \frac{7}{15} \approx 46.67\% \quad (\text{Hardware utilization only } 53.3\%)$$
- For $m = 32$:
  $$F_{\text{bubble}} = \frac{8 - 1}{32 + 8 - 1} = \frac{7}{39} \approx 17.95\% \quad (\text{Utilization } 82.05\%)$$
- For $m = 64$:
  $$F_{\text{bubble}} = \frac{8 - 1}{64 + 8 - 1} = \frac{7}{71} \approx 9.86\% \quad (\text{Utilization } 90.14\%)$$

#### Step 3: GPipe Memory
Under GPipe, all $m$ forward passes are executed before any backward pass begins.
Stage 0 must store activations for **all $m = 64$ microbatches**, requiring $64\times$ the memory of a single microbatch.

#### Step 4: 1F1B Memory
Under 1F1B scheduling, once the pipeline is primed after $p$ warmup steps, each forward pass is immediately followed by a backward pass that frees its activation memory.
Stage 0 holds at most **$p = 8$ microbatch activations** at peak!
1F1B reduces peak activation memory by an exact factor of:
$$\frac{m}{p} = \frac{64}{8} = 8\times \text{ memory reduction}$$
allowing large models to train without running out of activation memory.
