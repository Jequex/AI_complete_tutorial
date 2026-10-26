# Topic 02: Exercises & Solutions — Distributed Communication Primitives

## Exercise 1: Ring All-Reduce vs Parameter Server at Scale
**Problem:**
You are training a 70-billion parameter model (such as LLaMA-3-70B) in FP16 precision ($2\text{ bytes}$ per parameter).
- Total Gradient Buffer Size: $M = 70 \times 10^9 \times 2\text{ bytes} = 140\text{ GB}$.
- Cluster Setup: $N = 128\text{ GPUs}$ connected via $400\text{ Gbps}$ InfiniBand NDR RDMA ($B = 50\text{ GB/s}$ sustained network throughput per link).

1. Compute the total data sent and received by the central server in a Parameter Server architecture, and calculate the time required to complete gradient synchronization.
2. Compute the total data sent by each GPU in a Ring All-Reduce architecture, and calculate the time required to complete gradient synchronization.
3. What is the speedup of Ring All-Reduce over Parameter Server?

<details>
<summary>Click to view Solution</summary>

### Solution:

#### 1. Parameter Server (PS):
- The central server must receive gradients from $N - 1 = 127$ workers:
  $$V_{\text{recv}} = 127 \times 140\text{ GB} = 17{,}780\text{ GB}$$
- The server must transmit the aggregated weights back to all 127 workers:
  $$V_{\text{send}} = 127 \times 140\text{ GB} = 17{,}780\text{ GB}$$
- Total data handled by server:
  $$V_{\text{PS}} = 17{,}780 + 17{,}780 = \mathbf{35{,}560\text{ GB}} = 35.56\text{ TB}$$
- Assuming the parameter server has a single 400 Gbps network card ($50\text{ GB/s}$):
  $$T_{\text{PS}} = \frac{35{,}560\text{ GB}}{50\text{ GB/s}} = \mathbf{711.2\text{ seconds}} \approx \mathbf{11.85\text{ minutes per step!}}$$
Training is completely non-viable.

#### 2. Ring All-Reduce:
- Total data sent per GPU:
  $$V_{\text{ring}} = 2 \cdot \frac{N - 1}{N} M = 2 \cdot \left(\frac{127}{128}\right) \times 140\text{ GB} \approx 1.984375 \times 140\text{ GB} \approx \mathbf{277.81\text{ GB}}$$
- Because every GPU transmits simultaneously to its neighbor over dedicated $50\text{ GB/s}$ links:
  $$T_{\text{ring}} = \frac{277.81\text{ GB}}{50\text{ GB/s}} \approx \mathbf{5.556\text{ seconds}}$$

#### 3. Speedup:
$$\text{Speedup} = \frac{T_{\text{PS}}}{T_{\text{ring}}} = \frac{711.2\text{ s}}{5.556\text{ s}} \approx \mathbf{128\times}$$
Ring All-Reduce scales perfectly with cluster size.
</details>

---

## Exercise 2: Hockney Model & The Ring-vs-Tree Transition Point
**Problem:**
Under the Hockney communication model ($T = \text{Hops} \cdot \alpha + \text{Volume} / \beta$):
- Ring All-Reduce takes:
  $$T_{\text{ring}}(M) = 2(N - 1)\alpha + 2 \cdot \frac{N - 1}{N} \cdot \frac{M}{\beta}$$
- Binary Tree All-Reduce takes:
  $$T_{\text{tree}}(M) = 2 \log_2(N)\alpha + 2 \cdot \frac{N - 1}{N} \cdot \frac{M}{\beta}$$
Assume a cluster of $N = 64\text{ GPUs}$, network latency $\alpha = 5\ \mu\text{s} = 5 \times 10^{-6}\text{ s}$, and bandwidth $\beta = 50\text{ GB/s} = 50 \times 10^9\text{ B/s}$.

1. Explain physically why Tree All-Reduce has fewer latency hops than Ring All-Reduce.
2. If tree and ring had identical bandwidth factors, Tree would always win. However, in practice, Tree All-Reduce on a ring or flat network suffers from link contention: its effective bandwidth is lower by a factor of 2:
   $$T_{\text{tree\_contention}}(M) = 2 \log_2(N)\alpha + 4 \cdot \frac{N - 1}{N} \cdot \frac{M}{\beta}$$
   Derive the crossover tensor size $M^*$ where Ring All-Reduce becomes faster than Tree All-Reduce.

<details>
<summary>Click to view Solution</summary>

### Solution:

#### 1. Latency Hops:
In a ring of $N$ nodes, a message must traverse the perimeter node-by-node sequentially, requiring $2(N - 1)$ communication steps.
In a tree structure, communication occurs hierarchically across a balanced tree of depth $\log_2(N)$. In recursive doubling, each round doubles the participating nodes, requiring only $2 \log_2(N)$ round-trip hops. For $N = 64$:
- Ring hops: $2(64 - 1) = 126\text{ hops}$.
- Tree hops: $2 \log_2(64) = 2 \times 6 = 12\text{ hops}$ ($>10\times$ fewer startup latencies).

#### 2. Crossover Derivation $M^*$:
Set $T_{\text{ring}}(M^*) = T_{\text{tree\_contention}}(M^*)$:
$$2(N - 1)\alpha + 2 \frac{N - 1}{N}\frac{M^*}{\beta} = 2 \log_2(N)\alpha + 4 \frac{N - 1}{N}\frac{M^*}{\beta}$$

Rearranging terms:
$$2\left[(N - 1) - \log_2(N)\right] \alpha = 2 \frac{N - 1}{N}\frac{M^*}{\beta}$$

Solve for $M^*$:
$$M^* = \frac{\left[(N - 1) - \log_2(N)\right] N \cdot \beta \cdot \alpha}{N - 1}$$

Substitute $N = 64$, $\alpha = 5 \times 10^{-6}\text{ s}$, $\beta = 50 \times 10^9\text{ B/s}$:
- $N - 1 = 63$
- $\log_2(64) = 6$
- Numerator: $(63 - 6) \times 64 \times (50 \times 10^9) \times (5 \times 10^{-6}) = 57 \times 64 \times 250{,}000 = 912{,}000{,}000$
- Denominator: $63$

$$M^* = \frac{912{,}000{,}000}{63} \approx \mathbf{14{,}476{,}190\text{ bytes}} \approx \mathbf{14.48\text{ MB}}$$

**Conclusion**:
- For tensors smaller than $\approx 14.5\text{ MB}$, Tree All-Reduce is faster because latency $\alpha$ dominates.
- For tensors larger than $\approx 14.5\text{ MB}$ (standard for LLM gradient buckets, which are typically $25\text{ MB}$ to $200\text{ MB}$), Ring All-Reduce is faster because bandwidth $\beta$ dominates.
</details>

---

## Exercise 3: Step-by-Step Ring State Tracing
**Problem:**
Consider $N = 4\text{ GPUs}$ synchronizing a buffer of $M = 4$ scalars:
- GPU 0: $[1, 2, 3, 4]$
- GPU 1: $[10, 20, 30, 40]$
- GPU 2: $[100, 200, 300, 400]$
- GPU 3: $[1000, 2000, 3000, 4000]$

Trace the buffer state of all 4 GPUs after:
1. Scatter-Reduce Step 1 ($k = 0$)
2. Scatter-Reduce Step 2 ($k = 1$)
3. Scatter-Reduce Step 3 ($k = 2$, end of Phase 1)
Show which GPU holds the fully reduced scalar for each chunk $c \in \{0, 1, 2, 3\}$.

<details>
<summary>Click to view Solution</summary>

### Solution:

Chunk $c$ corresponds to index $c \in \{0, 1, 2, 3\}$.
In Scatter-Reduce, node $i$ sends chunk $(i - k) \pmod 4$ to node $(i + 1) \pmod 4$.

#### Initial State ($t = 0$):
- GPU 0: `[1, 2, 3, 4]`
- GPU 1: `[10, 20, 30, 40]`
- GPU 2: `[100, 200, 300, 400]`
- GPU 3: `[1000, 2000, 3000, 4000]`

#### Step 1 ($k = 0$): Send chunk $i \pmod 4$ to $(i + 1) \pmod 4$
- GPU 0 sends chunk 0 ($1$) to GPU 1 $\implies$ GPU 1 chunk 0 becomes $10 + 1 = 11$.
- GPU 1 sends chunk 1 ($20$) to GPU 2 $\implies$ GPU 2 chunk 1 becomes $200 + 20 = 220$.
- GPU 2 sends chunk 2 ($300$) to GPU 3 $\implies$ GPU 3 chunk 2 becomes $3000 + 300 = 3300$.
- GPU 3 sends chunk 3 ($4000$) to GPU 0 $\implies$ GPU 0 chunk 3 becomes $4 + 4000 = 4004$.

#### Step 2 ($k = 1$): Send chunk $(i - 1) \pmod 4$ to $(i + 1) \pmod 4$
- GPU 0 sends chunk 3 ($4004$) to GPU 1 $\implies$ GPU 1 chunk 3 becomes $40 + 4004 = 4044$.
- GPU 1 sends chunk 0 ($11$) to GPU 2 $\implies$ GPU 2 chunk 0 becomes $100 + 11 = 111$.
- GPU 2 sends chunk 1 ($220$) to GPU 3 $\implies$ GPU 3 chunk 1 becomes $2000 + 220 = 2220$.
- GPU 3 sends chunk 2 ($3300$) to GPU 0 $\implies$ GPU 0 chunk 2 becomes $3 + 3300 = 3303$.

#### Step 3 ($k = 2$): Send chunk $(i - 2) \pmod 4$ to $(i + 1) \pmod 4$
- GPU 0 sends chunk 2 ($3303$) to GPU 1 $\implies$ GPU 1 chunk 2 becomes $30 + 3303 = \mathbf{3333}$.
- GPU 1 sends chunk 3 ($4044$) to GPU 2 $\implies$ GPU 2 chunk 3 becomes $400 + 4044 = \mathbf{4444}$.
- GPU 2 sends chunk 0 ($111$) to GPU 3 $\implies$ GPU 3 chunk 0 becomes $1000 + 111 = \mathbf{1111}$.
- GPU 3 sends chunk 1 ($2220$) to GPU 0 $\implies$ GPU 0 chunk 1 becomes $2 + 2220 = \mathbf{2222}$.

#### End of Phase 1 Status:
- GPU 3 holds chunk 0: $\mathbf{1111} = 1 + 10 + 100 + 1000$ (Fully reduced!)
- GPU 0 holds chunk 1: $\mathbf{2222} = 2 + 20 + 200 + 2000$ (Fully reduced!)
- GPU 1 holds chunk 2: $\mathbf{3333} = 3 + 30 + 300 + 3000$ (Fully reduced!)
- GPU 2 holds chunk 3: $\mathbf{4444} = 4 + 40 + 400 + 4000$ (Fully reduced!)

Each GPU $i$ holds the complete global sum of chunk $(i + 1) \pmod 4$, ready for Phase 2 All-Gather.
</details>
