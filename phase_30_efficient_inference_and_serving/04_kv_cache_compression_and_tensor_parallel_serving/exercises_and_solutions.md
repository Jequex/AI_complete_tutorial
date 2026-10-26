# Topic 04: Exercises & Solutions — KV Cache Compression & Tensor Parallel Serving

---

### Exercise 1: Quantitative KV Cache Scaling across MHA, GQA, MQA and Tensor Parallelism

Consider serving a 70B parameter model with:
- $n_{\text{layers}} = 80$
- $H = 64$ query heads
- $d_{\text{head}} = 128$
- Batch size $b = 32$
- Context length $s = 8192$ tokens
- Precision: 16-bit FP16 ($e = 2$ bytes)

**Questions:**
1. Compute the total KV cache memory footprint (in GB) under:
   - (a) Multi-Head Attention ($H_{\text{kv}} = 64$).
   - (b) Grouped-Query Attention ($H_{\text{kv}} = 8$, group size $G = 8$).
   - (c) Multi-Query Attention ($H_{\text{kv}} = 1$).
2. If this model is served across an $N_{\text{tp}} = 8$ GPU node using Tensor Parallelism, compute the KV cache memory stored on *each individual GPU* for (a), (b), and (c).

---

#### Solution to Exercise 1

**1. Total KV Cache Footprint:**
Formula: $\text{Mem}_{\text{KV}} = 4 \cdot n_{\text{layers}} \cdot H_{\text{kv}} \cdot d_{\text{head}} \cdot b \cdot s$ bytes.
Constant factor:
$$4 \times 80 \times 128 \times 32 \times 8192 = 10,737,418,240 \text{ bytes/head} = 10.7374 \text{ GB per KV head}$$

- **(a) Multi-Head Attention ($H_{\text{kv}} = 64$):**
  $$\text{Mem}_{\text{MHA}} = 64 \times 10.7374 \text{ GB} = \mathbf{687.19 \text{ GB}}$$
- **(b) Grouped-Query Attention ($H_{\text{kv}} = 8$):**
  $$\text{Mem}_{\text{GQA}} = 8 \times 10.7374 \text{ GB} = \mathbf{85.90 \text{ GB}} \quad (8.0\times \text{ reduction})$$
- **(c) Multi-Query Attention ($H_{\text{kv}} = 1$):**
  $$\text{Mem}_{\text{MQA}} = 1 \times 10.7374 \text{ GB} = \mathbf{10.74 \text{ GB}} \quad (64.0\times \text{ reduction})$$

**2. Per-GPU Memory Footprint with Tensor Parallelism ($N_{\text{tp}} = 8$):**
Under Megatron Tensor Parallelism, the Key and Value heads are partitioned evenly across the 8 GPUs:
- **(a) MHA on 8 GPUs**:
  $$\text{Per-GPU KV} = \frac{687.19 \text{ GB}}{8} = \mathbf{85.90 \text{ GB/GPU}}$$
  (Note: Exceeds the capacity of an 80GB GPU without even accounting for model weights!)
- **(b) GQA on 8 GPUs**:
  Each GPU hosts $8 / 8 = 1$ KV head:
  $$\text{Per-GPU KV} = \frac{85.90 \text{ GB}}{8} = \mathbf{10.74 \text{ GB/GPU}}$$
  (Readily fits alongside 17.5 GB sharded model weights in an 80GB GPU).
- **(c) MQA on 8 GPUs**:
  Here, $H_{\text{kv}} = 1 < N_{\text{tp}} = 8$. Because a single head cannot be split across GPUs without inter-GPU head communication, MQA typically broadcasts or replicates the single KV head across all ranks:
  $$\text{Per-GPU KV} = \mathbf{10.74 \text{ GB/GPU}}$$

---

### Exercise 2: Heavy Hitter Oracle (H2O) Dynamic Eviction Trace

A sequence of length 10 generates a new token at step $t=11$.
The KV cache budget is fixed at $s_{\text{budget}} = 6$ tokens, with:
- $W_{\text{sink}} = 2$ initial sink tokens (Tokens 1 and 2 are permanently pinned).
- $W_{\text{recency}} = 2$ recency window tokens (Tokens 10 and 11 are pinned).
- The remaining budget is $k_{\text{heavy}} = 2$ heavy hitter tokens.

Before step 11, the active cache contains:
- Sinks: `{Token 1, Token 2}`
- Intermediate candidates: `{Token 5 (score: 4.2), Token 6 (score: 1.1), Token 8 (score: 5.7)}`
- Recency: `{Token 10 (score: 2.0)}`

At step 11, Token 11 arrives (cumulative score initialized to $0.8$). The new attention weights from token 11 to historical tokens are:
- Token 1: 0.15
- Token 2: 0.25
- Token 5: 0.30
- Token 6: 0.05
- Token 8: 0.15
- Token 10: 0.10

**Questions:**
1. Update the cumulative attention scores for all candidate tokens.
2. The cache now holds 7 tokens, exceeding the budget of 6. Identify which token is evicted by the H2O policy.

---

#### Solution to Exercise 2

**1. Update Cumulative Attention Scores:**
- Token 1 (Sink): $S_1$ (pinned, protected from eviction)
- Token 2 (Sink): $S_2$ (pinned, protected from eviction)
- Token 5: $S_5 = 4.2 + 0.30 = \mathbf{4.50}$
- Token 6: $S_6 = 1.1 + 0.05 = \mathbf{1.15}$
- Token 8: $S_8 = 5.7 + 0.15 = \mathbf{5.85}$
- Token 10: $S_{10} = 2.0 + 0.10 = \mathbf{2.10}$
- Token 11: $S_{11} = 0.8$ (currently in recency window)

**2. Candidate Identification and Eviction:**
- Total tokens in cache: 7 tokens.
- Protected by Sink policy ($W_{\text{sink}} = 2$): `{Token 1, Token 2}`.
- Protected by Recency policy ($W_{\text{recency}} = 2$): `{Token 10, Token 11}`.
- Intermediate eligible candidates for eviction: `{Token 5, Token 6, Token 8}`.
- Heavy hitter budget: $k_{\text{heavy}} = 2$.
- Comparing cumulative scores of intermediate candidates:
  - Token 8: $5.85$ (Heavy Hitter 1 &rarr; Retained)
  - Token 5: $4.50$ (Heavy Hitter 2 &rarr; Retained)
  - Token 6: $1.15$ (Lowest score &rarr; **Evicted!**)
- Token 6 is evicted from the KV cache. The cache returns to size 6: `{Token 1, Token 2, Token 5, Token 8, Token 10, Token 11}`.

---

### Exercise 3: Communication Volume of Tensor Parallel Serving

In Megatron Tensor Parallelism with world size $N_{\text{tp}}$, each Transformer layer requires one `All-Reduce` collective after the row-parallel output projection $W_O$, and one `All-Reduce` after the MLP second linear projection $W_2$.
For batch size $b$, sequence length $s$, hidden dimension $d$, and 16-bit precision ($e = 2$ bytes):
1. Derive the bytes transmitted per GPU in a ring All-Reduce:
   $$\text{Bytes}_{\text{All-Reduce}} = 2 \cdot \left(\frac{N_{\text{tp}} - 1}{N_{\text{tp}}}\right) \cdot (b \cdot s \cdot d \cdot e)$$
2. For $b = 32, d = 8192, N_{\text{tp}} = 8$:
   Compare the communication volume per forward pass across 80 layers during:
   - (a) The **Prefill Phase** ($s = 2048$).
   - (b) The **Decode Phase** ($s = 1$).

---

#### Solution to Exercise 3

**1. Communication Formula:**
For $N_{\text{tp}}$ GPUs, standard ring All-Reduce transmits $2 \cdot \frac{N_{\text{tp}} - 1}{N_{\text{tp}}} \cdot M$ bytes per device, where $M$ is the message buffer size $b \cdot s \cdot d \cdot e$.
There are 2 All-Reduces per layer (Self-Attention + MLP), across $n_{\text{layers}} = 80$ layers:
$$\text{Total Transmitted} = 2 \times 80 \times 2 \times \left(\frac{8 - 1}{8}\right) \cdot (b \cdot s \cdot d \cdot 2) = 280 \cdot b \cdot s \cdot d \cdot 2 \text{ bytes}$$

**2. Numerical Comparison:**
- **(a) Prefill Phase ($s = 2048$):**
  $$\text{Buffer per layer } M = 32 \times 2048 \times 8192 \times 2 = 1,073,741,824 \text{ Bytes} = 1.074 \text{ GB}$$
  $$\text{Total Transmitted} = 160 \times \left(2 \times \frac{7}{8}\right) \times 1.074 \text{ GB} = \mathbf{300.6 \text{ GB}}$$
- **(b) Decode Phase ($s = 1$):**
  $$\text{Buffer per layer } M = 32 \times 1 \times 8192 \times 2 = 524,288 \text{ Bytes} = 512 \text{ KB}$$
  $$\text{Total Transmitted} = 160 \times 1.75 \times 0.512 \text{ MB} = \mathbf{143.36 \text{ MB}}$$
In the decode phase, communication is ultra-lightweight ($143 \text{ MB}$ total across 80 layers), meaning high-speed NVLink easily overlaps communication with computation.
