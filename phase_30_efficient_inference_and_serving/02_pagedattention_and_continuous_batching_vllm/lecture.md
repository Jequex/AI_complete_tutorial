# Topic 02: PagedAttention & Continuous Batching (vLLM)

## 1. Introduction & 12-Year-Old Intuition

Imagine you own a 50-room luxury hotel. 

Two different hotel managers try to run the booking desk:

1. **The Static Batching Manager**:
   When a tour bus arrives with 10 guests, the manager insists: *"All 10 guests must check in together, and nobody can check out until the slowest guest finishes their stay!"*
   If 9 guests want to stay for only 1 night, but 1 guest wants to stay for 7 nights, the manager leaves the other 9 rooms completely empty and locked for 6 whole days, refusing to let any new travelers into those rooms until the entire group checks out together. The hotel is 90% empty, but travelers waiting outside in the rain are told there is no vacancy!

2. **The Continuous Batching Manager (Orca / vLLM)**:
   The second manager operates at the resolution of a single minute. The exact moment a guest finishes their breakfast and returns their key card, that room is immediately cleaned and handed to the very next person waiting in line. Requests flow in and out dynamically, keeping every room occupied 100% of the time.

Now imagine how the hotel manages its storage closets:
- **Traditional Pre-allocation**: Every time a guest checks in, the hotel reserves a giant 500-gallon storage trunk for them, assuming they *might* bring 50 suitcases. If a traveler only brings a tiny backpack, the giant trunk still sits locked and empty in their room. Other guests can't store their luggage because the hotel claims it is "out of storage."
- **PagedAttention (Virtual Memory Paging)**: Instead of giant trunks, the hotel has small modular lockers that hold exactly 16 shirts. When you check in, you get one small locker. When it fills up, the front desk hands you a key to another locker anywhere on the property. Your clothes don't have to be in side-by-side lockers—the hotel keeps a directory (a **Block Table**) mapping your room to your locker numbers!

This breakthrough—pioneered by **Orca** (Yu et al., 2022) for continuous batching and **vLLM / PagedAttention** (Kwon et al., 2023) for virtual memory paging—unlocked a **2x–4x throughput leap** in production LLM serving.

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Domain / Type | Engineering Significance |
| :--- | :--- | :--- | :--- |
| $\mathcal{B}_{\text{size}}$ | Block Size | Integer (typically 16 or 32) | Number of tokens stored inside a single physical KV block. |
| $N_{\text{phys}}$ | Total Physical Blocks | Integer | Capacity of the physical block pool: $\lfloor \text{VRAM}_{\text{KV}} / \text{BlockBytes} \rfloor$. |
| $T_{\text{block}}$ | Logical-to-Physical Table | Mapping: $\mathbb{N} \to \mathbb{N}$ | Translation lookup table mapping logical sequence block index to physical memory block index. |
| $\text{ref\_count}(i)$ | Reference Counter | Integer $\ge 0$ | Number of logical sequences currently reading physical block $i$ (used for Copy-on-Write). |
| $L_i$ | Logical Sequence Length | Tokens | Current token length of request $i$ ($S_{\text{prompt}} + \text{generated}$). |
| $\lceil L_i / \mathcal{B}_{\text{size}} \rceil$ | Blocks Allocated | Integer | Exact count of physical blocks required to store the KV cache for sequence $i$. |
| $\eta_{\text{mem}}$ | Memory Utilization | $[0.0, 1.0]$ | Ratio of memory actively holding valid tokens versus total allocated memory. |
| $Q_{\text{wait}}, Q_{\text{run}}$ | Scheduler Queues | FIFO / Priority Queues | Waiting request pool versus currently executing batch on GPU. |

---

## 3. Static Batching vs. Continuous Batching (Orca)

### 3.1 The Failure of Static Batching
In classical deep learning inference, batching is static: a tensor of shape $[B, S]$ is dispatched to the GPU. In LLM text generation, however, output lengths are non-deterministic and highly variable.

```
Request 1: [Prompt: 100 tok] -> [Generates: 20 tok] -> FINISHED (Idles for 180 steps)
Request 2: [Prompt: 150 tok] -> [Generates: 200 tok] -> FINISHED
Request 3: [Prompt:  50 tok] -> [Generates: 10 tok] -> FINISHED (Idles for 190 steps)
           |<--- Active Generation --->|<------- Wasted GPU Compute / Padding ------->|
```

Under static batching:
1. Short requests must remain in memory until the longest request completes.
2. Padding tokens are multiplied against weights, consuming tensor core operations with zero utility.
3. Newly arriving requests cannot begin prefill until the entire batch finishes.

### 3.2 Continuous (Iteration-Level) Batching
Continuous batching operates at the iteration level of a single autoregressive step:

$$\text{Forward Step } t: \quad \mathcal{R}_{\text{active}} \leftarrow \text{FilterFinished}(\mathcal{R}_t) \cup \text{ScheduleNew}(\mathcal{Q}_{\text{waiting}})$$

At step $t$:
1. The engine checks if any active sequence emitted an `[EOS]` (End-of-Sequence) token or reached maximum length.
2. Finished sequences are evicted, immediately freeing their memory.
3. New requests from the waiting queue are admitted to fill available batch slots.
4. The forward pass dynamically computes mixed prefill (for new admissions) and single-token decode (for ongoing sequences).

---

## 4. KV Cache Memory Fragmentation

Before PagedAttention, state-of-the-art serving frameworks suffered from catastrophic memory waste: **60% to 80%** of GPU memory was wasted due to fragmentation!

```
[ Traditional LLM Memory Allocation: Contiguous Buffers ]
Req 1 (max=2048): [ Tokens 1..300 | ░░░░░░░░░░ Wasted Internal Fragmentation (1748 slots) ░░░░░░░░░░ ]
Req 2 (max=2048): [ Tokens 1..800 | ░░░░░░░░░░ Wasted Internal Fragmentation (1248 slots) ░░░░░░░░░░ ]
VRAM Gap        : [ ░░ External Fragmentation: Hole cannot fit a 2048-token contiguous allocation ░░ ]
```

### 4.1 Internal Fragmentation
Because the engine does not know how many tokens a user prompt will generate, it must pre-allocate a contiguous memory buffer sized to the maximum context window (e.g., $S_{\text{max}} = 4096$).
If a request terminates after generating 50 tokens, the remaining $4046$ token slots sit completely unused, yet locked from other users.

### 4.2 External Fragmentation
As requests of varying lengths complete and exit, their contiguous memory allocations are freed, leaving non-contiguous "holes" of free memory across the VRAM address space. Even if the total free memory is 10 GB, a new request needing 4 GB of contiguous memory will trigger an Out-of-Memory (OOM) error if no single contiguous block of 4 GB exists!

---

## 5. PagedAttention: OS-Style Virtual Memory for KV Caches

PagedAttention (Kwon et al., SOSP 2023) solves both internal and external fragmentation by drawing inspiration from virtual memory paging in operating systems:

```
Logical Blocks (Sequence View):
Logical Block 0 (Tokens 0..15)   ──▶ Physical Block 104
Logical Block 1 (Tokens 16..31)  ──▶ Physical Block 12
Logical Block 2 (Tokens 32..47)  ──▶ Physical Block 89
                                        │
Physical Block Pool (GPU VRAM):         ▼
[ Block 12 ] [ Block 13 ] ... [ Block 89 ] ... [ Block 104 ]
(Non-contiguous physical memory locations scattered across HBM)
```

### 5.1 Mechanics of PagedAttention
1. **Physical Block Pool**: At engine startup, the entire free VRAM is partitioned into fixed-size physical blocks of size $\mathcal{B}_{\text{size}}$ (e.g., 16 tokens). Each block stores Key and Value tensors for 16 tokens across all model layers.
2. **Logical Block Table**: Each active sequence maintains a lookup table $T_{\text{block}}$ mapping its logical block indices $\{0, 1, 2, \dots\}$ to arbitrary physical blocks $\{p_0, p_1, p_2, \dots\}$.
3. **On-Demand Allocation**: When a sequence starts, it is allocated only the blocks needed for its prompt. During decode, as new tokens are generated, a new physical block is fetched from the free list **only when the current block becomes full**.
4. **Zero External Fragmentation**: Because all physical blocks are identical in size ($\mathcal{B}_{\text{size}}$ tokens), any free block can satisfy any allocation request. External fragmentation is mathematically eliminated ($0\%$).
5. **Near-Zero Internal Fragmentation**: Memory waste is restricted strictly to the partially filled final block of each active sequence. The expected waste per sequence is:
   $$\mathbb{E}[\text{Wasted Tokens}] = \frac{\mathcal{B}_{\text{size}} - 1}{2} \approx \frac{\mathcal{B}_{\text{size}}}{2}$$
   For $\mathcal{B}_{\text{size}} = 16$, this is only **8 tokens** per sequence (typically $< 3\%$ of total memory)!

---

## 6. Copy-on-Write (CoW) Fork Mechanics for Parallel Sampling

Many generative workloads require branching generation:
- **Parallel Sampling**: Generating $K$ candidate answers from a single prompt (temperature sampling).
- **Beam Search**: Maintaining the top-$B$ hypotheses.

In traditional systems, branching required duplicating the prompt's KV cache $K$ times in VRAM!

Under PagedAttention, branching is achieved using **Copy-on-Write (CoW)**:

```
Prompt Logical Blocks: [L0, L1] ──▶ Physical Blocks: [P7, P8] (ref_count = 3)
                                            ▲
          ┌─────────────────────────────────┼─────────────────────────────────┐
          │                                 │                                 │
Branch 1 (Fork):                   Branch 2 (Fork):                  Branch 3 (Fork):
Blocks: [P7, P8]                   Blocks: [P7, P8]                   Blocks: [P7, P8]

* Step t+1: Branch 1 generates token filling a new block:
Branch 1 allocates NEW Physical Block P99 (ref_count = 1).
Physical Blocks P7 and P8 remain shared without copying!
```

- Each physical block tracks a `ref_count`.
- When a prompt is forked into $K$ branches, the child sequences simply copy the block table pointers and increment `ref_count`. Memory overhead is **zero bytes of KV duplication**.
- When a branch writes a new token into an incompletely filled shared block, the block manager allocates a new physical block, copies over only that single 16-token block, and decrements the original block's `ref_count`.

---

## 7. Mathematical Derivations: Memory Waste Bounds

### Theorem: Bound on Memory Fragmentation Waste
Let $N_{\text{seq}}$ be the number of active sequences, each with length $L_i \sim \text{Uniform}(1, S_{\text{max}})$.
Under contiguous pre-allocation:
$$\text{Memory Allocated} = N_{\text{seq}} \cdot S_{\text{max}} \cdot \text{BytesPerToken}$$
$$\text{Memory Used} = \sum_{i=1}^{N_{\text{seq}}} L_i \cdot \text{BytesPerToken} \approx N_{\text{seq}} \cdot \frac{S_{\text{max}}}{2} \cdot \text{BytesPerToken}$$
$$\text{Waste Ratio}_{\text{contiguous}} = 1 - \frac{\text{Memory Used}}{\text{Memory Allocated}} \approx 1 - \frac{1}{2} = \mathbf{50\%} \quad (\text{often } >70\% \text{ for short outputs})$$

Under PagedAttention with block size $\mathcal{B}_{\text{size}}$:
Each sequence allocates $\lceil L_i / \mathcal{B}_{\text{size}} \rceil \cdot \mathcal{B}_{\text{size}}$ tokens.
$$\text{Waste per Sequence} = \left(\lceil L_i / \mathcal{B}_{\text{size}} \rceil \cdot \mathcal{B}_{\text{size}}\right) - L_i < \mathcal{B}_{\text{size}}$$
$$\mathbb{E}[\text{Waste Ratio}_{\text{PagedAttention}}] = \frac{\frac{\mathcal{B}_{\text{size}}}{2}}{\bar{L}} = \frac{\mathcal{B}_{\text{size}}}{2 \bar{L}}$$

For average sequence length $\bar{L} = 1024$ and $\mathcal{B}_{\text{size}} = 16$:
$$\text{Waste Ratio}_{\text{PagedAttention}} = \frac{16}{2 \times 1024} = \frac{8}{1024} \approx \mathbf{0.78\%}$$

PagedAttention drives memory waste from $>60\%$ down to $<1\%$, instantly tripling the concurrent batch capacity on identical GPU hardware.
