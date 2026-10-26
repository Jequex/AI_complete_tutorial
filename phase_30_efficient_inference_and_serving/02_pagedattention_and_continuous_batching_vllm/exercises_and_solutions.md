# Topic 02: Exercises & Solutions — PagedAttention & Continuous Batching

---

### Exercise 1: Quantitative Analysis of Memory Fragmentation

A serving cluster has 40 GB of VRAM reserved for KV caching. A model generates responses for requests whose final context lengths $L$ follow a distribution with mean $\bar{L} = 600$ tokens and maximum possible context $S_{\text{max}} = 4096$ tokens.

**Questions:**
1. Under **Contiguous Pre-allocation**, the engine must conservatively allocate memory for $S_{\text{max}} = 4096$ tokens when a request arrives. Calculate the percentage of memory lost to internal fragmentation. How many concurrent requests can run simultaneously if each token takes $320 \text{ KB}$ of KV cache?
2. Under **PagedAttention** with block size $\mathcal{B}_{\text{size}} = 16$ tokens, calculate the expected memory waste per request and the overall percentage memory waste.
3. Compute the maximum concurrent request capacity under PagedAttention and calculate the throughput capacity gain factor over contiguous allocation.

---

#### Solution to Exercise 1

**1. Contiguous Pre-allocation Analysis:**
- Memory allocated per sequence: $S_{\text{max}} \times 320 \text{ KB} = 4096 \times 320 \text{ KB} = 1,310,720 \text{ KB} = 1.31072 \text{ GB}$.
- Memory actually used per sequence (on average): $\bar{L} \times 320 \text{ KB} = 600 \times 320 \text{ KB} = 192,000 \text{ KB} = 0.192 \text{ GB}$.
- Internal fragmentation waste percentage:
  $$\text{Waste} = 1 - \frac{\bar{L}}{S_{\text{max}}} = 1 - \frac{600}{4096} = 1 - 0.1465 = \mathbf{85.35\%}$$
- Over $85\%$ of the GPU memory sits empty!
- Concurrent request capacity:
  $$N_{\text{contiguous}} = \left\lfloor \frac{40 \text{ GB}}{1.31072 \text{ GB}} \right\rfloor = \mathbf{30 \text{ concurrent requests}}$$

**2. PagedAttention Analysis ($\mathcal{B}_{\text{size}} = 16$):**
- With block size 16, a sequence of length $L$ allocates $\lceil L / 16 \rceil$ blocks.
- Waste only occurs in the final partially filled block:
  $$\mathbb{E}[\text{Wasted Tokens}] = \frac{\mathcal{B}_{\text{size}} - 1}{2} = \frac{15}{2} = 7.5 \text{ tokens}$$
- Memory waste in bytes: $7.5 \text{ tokens} \times 320 \text{ KB/token} = 2,400 \text{ KB} = 2.4 \text{ MB}$.
- Percentage memory waste:
  $$\text{Waste}_{\text{PagedAttention}} = \frac{7.5}{\bar{L}} = \frac{7.5}{600} = \mathbf{1.25\%}$$

**3. Capacity and Throughput Gain:**
- Memory allocated per sequence under PagedAttention:
  $$M_{\text{seq}} = \left(\bar{L} + 7.5\right) \times 320 \text{ KB} = 607.5 \times 320 \text{ KB} \approx 0.1944 \text{ GB}$$
- Concurrent request capacity:
  $$N_{\text{PagedAttention}} = \left\lfloor \frac{40 \text{ GB}}{0.1944 \text{ GB}} \right\rfloor = \mathbf{205 \text{ concurrent requests}}$$
- Capacity gain factor:
  $$\text{Gain Factor} = \frac{N_{\text{PagedAttention}}}{N_{\text{contiguous}}} = \frac{205}{30} \approx \mathbf{6.83\times}$$
PagedAttention supports nearly $7\times$ as many concurrent users on the identical 40 GB GPU memory pool.

---

### Exercise 2: Copy-on-Write (CoW) Physical Block Allocation Trace

A user submits a prompt consisting of 35 tokens. The engine uses PagedAttention with block size $\mathcal{B}_{\text{size}} = 16$.
The user specifies `n=3` (parallel sampling: generate 3 independent responses from the same prompt).

**Questions:**
1. How many physical blocks are allocated for the prompt, and what are their reference counts?
2. Sequence Branch A generates 1 token. Does a Copy-on-Write event occur? Why or why not?
3. Sequence Branch B generates 15 tokens. How many total tokens does Branch B contain, how many blocks does it occupy, and what are the reference counts of all blocks involved?

---

#### Solution to Exercise 2

**1. Prompt Ingestion:**
- Prompt length: 35 tokens.
- Number of blocks required: $\lceil 35 / 16 \rceil = 3$ blocks:
  - Block 0: Tokens 0..15 (Full: 16 tokens) &rarr; assigned physical block $P_1$.
  - Block 1: Tokens 16..31 (Full: 16 tokens) &rarr; assigned physical block $P_2$.
  - Block 2: Tokens 32..34 (Partial: 3 tokens, 13 slots free) &rarr; assigned physical block $P_3$.
- Total physical memory consumed: $3 \times 16 = 48$ token slots.
- When forked into 3 child sequences ($A, B, C$), each sequence's block table points to $[P_1, P_2, P_3]$.
- Reference counts:
  $$\text{ref\_count}(P_1) = 3, \quad \text{ref\_count}(P_2) = 3, \quad \text{ref\_count}(P_3) = 3$$
- No memory is duplicated!

**2. Branch A Generates 1 Token:**
- Branch A needs to append token 36 into its active block.
- Its current active block is $P_3$, which currently has $\text{ref\_count}(P_3) = 3 > 1$.
- Because $P_3$ is shared, Branch A cannot write into $P_3$ without corrupting Branches B and C.
- **Copy-on-Write (CoW) Triggers**:
  1. The block manager allocates a fresh physical block $P_4$.
  2. The 3 existing tokens (tokens 32..34) are copied from $P_3$ into $P_4$.
  3. The new token 35 is appended at index 3 in $P_4$.
  4. Branch A's block table is updated to $[P_1, P_2, P_4]$.
  5. $\text{ref\_count}(P_3)$ is decremented from 3 to 2; $\text{ref\_count}(P_4) = 1$.
- Fully filled blocks $P_1$ and $P_2$ remain shared ($\text{ref\_count} = 3$) with zero copying!

**3. Branch B Generates 15 Tokens:**
- At token 1 of generation, Branch B also triggers CoW on $P_3$, copying tokens 32..34 into new block $P_5$, decrementing $\text{ref\_count}(P_3)$ to 1 (leaving $P_3$ uniquely owned by Branch C).
- Branch B appends 13 tokens into $P_5$, filling it completely ($3 + 13 = 16$ tokens, total length 48).
- For its 14th and 15th generated tokens, Branch B has filled $P_5$ and requires a new logical block (Block 3).
- The block manager allocates physical block $P_6$ ($\text{ref\_count}(P_6) = 1$).
- Final state of Branch B:
  - Total tokens: $35 + 15 = 50$ tokens.
  - Logical Block Table: $[P_1, P_2, P_5, P_6]$.
  - Block counts: 4 physical blocks ($4 \times 16 = 64$ slots).
  - Reference counts: $\text{ref\_count}(P_1) = 3, \text{ref\_count}(P_2) = 3, \text{ref\_count}(P_5) = 1, \text{ref\_count}(P_6) = 1$.

---

### Exercise 3: Throughput Comparison: Static vs. Continuous Batching

Consider a batch of 4 requests arriving at time $t=0$:
- Request 1: prompt 20 tokens, generates 20 tokens (finishes at step 20).
- Request 2: prompt 20 tokens, generates 40 tokens (finishes at step 40).
- Request 3: prompt 20 tokens, generates 60 tokens (finishes at step 60).
- Request 4: prompt 20 tokens, generates 200 tokens (finishes at step 200).

There are 4 additional identical requests waiting in queue $Q_{\text{wait}}$.
Compare the total time to serve all 8 requests under Static Batching versus Continuous Batching assuming each decode step takes $10 \text{ ms}$ and prefill takes $20 \text{ ms}$.

---

#### Solution to Exercise 3

**1. Static Batching:**
- Batch 1 (Requests 1-4) starts with prefill ($20 \text{ ms}$).
- Decode runs until the longest request finishes: $\max(20, 40, 60, 200) = 200$ steps.
- Time for Batch 1: $20 \text{ ms} + (200 \times 10 \text{ ms}) = 2020 \text{ ms}$.
- During steps 21 to 200, Requests 1, 2, and 3 are idle padding!
- Batch 2 (Requests 5-8) can only begin after step 200:
  - Time for Batch 2: $20 \text{ ms} + (200 \times 10 \text{ ms}) = 2020 \text{ ms}$.
- **Total Time (Static)**: $2020 + 2020 = \mathbf{4040 \text{ ms}} = 4.04 \text{ s}$.

**2. Continuous Batching:**
- Requests 1-4 start with prefill ($20 \text{ ms}$).
- At step 20, Request 1 finishes. Request 5 is admitted immediately from $Q_{\text{wait}}$.
- At step 40, Request 2 finishes. Request 6 is admitted immediately.
- At step 60, Request 3 finishes. Request 7 is admitted immediately.
- The GPU execution slots are never padded or idle; new requests begin generating tokens as soon as an active sequence finishes.
- The entire workload of 8 requests completes in approximately $\mathbf{2400 \text{ ms}}$, achieving a $\mathbf{1.68\times}$ wall-clock speedup and significantly reducing average request waiting latency.
