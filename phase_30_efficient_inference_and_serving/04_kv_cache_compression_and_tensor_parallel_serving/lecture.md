# Topic 04: KV Cache Compression & Tensor Parallel Serving

## 1. Introduction & 12-Year-Old Intuition

Imagine you are a detective investigating an intricate, 1,000-page mystery case. 

Every time you uncover a new clue, you must cross-examine all previously collected evidence. 
Your desk is only so wide. If you spread out all 1,000 pages of notes across your desk, you run out of space immediately and your work grinds to a halt.

How do top detectives solve this? They use two strategies:

1. **Strategic Eviction (The Smart Notebook / H2O)**:
   Instead of keeping all 1,000 pages on the table, you realize three things:
   - **The Case Brief (Attention Sinks)**: The first 4 pages contain the suspect list and core premise. You never throw them away.
   - **The Recent Trail (Recency Window)**: The last 10 pages describe where you walked five minutes ago. You need them right now for immediate context.
   - **The Heavy Clues (Heavy Hitters)**: Out of the 986 pages in the middle, only 20 pages contain critical evidence (fingerprint reports, weapon receipts). The other 966 pages are mundane filler (*"The suspect drank a glass of water"*).
   By keeping only the initial brief, the recent notes, and the heavy clues, you reduce your desk clutter by **80%**, while retaining **99% of your deductive power**!

2. **Teamwork (Tensor Parallelism)**:
   What if the case is simply too enormous for one desk? You hire 4 assistant detectives.
   Instead of copying the entire case file 4 times, you rip each page into 4 sections (or divide the attention heads): Detective 1 tracks Motive, Detective 2 tracks Alibis, Detective 3 tracks Forensics, and Detective 4 tracks Timelines.
   Each detective only needs enough desk space for their own specialized quarter of the notes. At the end of every step, they quickly whisper their conclusions to each other (**All-Reduce**) to form the complete picture.

This combination—**architectural compression (GQA)**, **dynamic cache eviction (H2O)**, and **distributed KV sharding (Tensor Parallelism)**—enables modern inference clusters to serve context lengths of 32k, 128k, and 1M tokens without running out of GPU memory.

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Domain / Type | Engineering Significance |
| :--- | :--- | :--- | :--- |
| $H$ | Query Heads | Integer | Number of Query attention heads ($H = n_{\text{heads}}$). |
| $H_{\text{kv}}$ | Key/Value Heads | Integer | Number of Key/Value heads ($H_{\text{kv}} = H$ in MHA, $1$ in MQA, $H/G$ in GQA). |
| $G$ | Group Size | Integer | Ratio $H / H_{\text{kv}}$, the number of query heads sharing a single KV head. |
| $W_{\text{recency}}$ | Recency Window | Integer (e.g. 64) | Fixed number of most recent tokens always preserved in the KV cache. |
| $W_{\text{sink}}$ | Attention Sink Size | Integer (e.g. 4) | Fixed number of initial prompt tokens permanently pinned in the KV cache. |
| $k_{\text{heavy}}$ | Heavy Hitter Budget | Integer | Number of high-attention historical tokens preserved based on cumulative scores. |
| $s_{\text{budget}}$ | Total KV Token Budget | Integer | Compressed memory budget: $s_{\text{budget}} = W_{\text{sink}} + k_{\text{heavy}} + W_{\text{recency}}$. |
| $N_{\text{tp}}$ | Tensor Parallel World Size | Integer (e.g. 2, 4, 8) | Number of GPUs across which attention heads and weight matrices are partitioned. |
| $A_{t, j}$ | Attention Score | Scalar $\in [0, 1]$ | Softmax attention weight from query token $t$ to key token $j$: $\text{softmax}(Q_t K_j^\top / \sqrt{d})$. |

---

## 3. Architectural Compression: MHA vs. MQA vs. GQA

In standard self-attention, each attention head maintains independent Key and Value projection weights:

```
[ Multi-Head Attention (MHA) ]           [ Multi-Query Attention (MQA) ]         [ Grouped-Query Attention (GQA) ]
Query Heads : [Q1] [Q2] [Q3] [Q4]        Query Heads : [Q1] [Q2] [Q3] [Q4]        Query Heads : [Q1] [Q2]  [Q3] [Q4]
Key Heads   : [K1] [K2] [K3] [K4]        Key Head    : [       K1       ]        Key Heads   : [  K1  ]   [  K2  ]
Value Heads : [V1] [V2] [V3] [V4]        Value Head  : [       V1       ]        Value Heads : [  V1  ]   [  V2  ]
KV Memory Ratio: 1.0x (Baseline)         KV Memory Ratio: 1/H (e.g. 0.015x)       KV Memory Ratio: 1/G (e.g. 0.125x)
```

### 3.1 Multi-Query Attention (MQA, Shazeer 2019)
MQA collapses all Key and Value heads into a single shared KV head ($H_{\text{kv}} = 1$):
- KV cache size is reduced by an exact factor of $H$ ($32\times$ to $64\times$).
- However, sharing a single key-value projection across 64 query heads can cause a non-trivial degradation in modeling nuances, cross-lingual capabilities, and code reasoning.

### 3.2 Grouped-Query Attention (GQA, Ainslie et al. 2023)
GQA divides the $H$ query heads into $H_{\text{kv}}$ groups, where each group of $G = H / H_{\text{kv}}$ query heads shares one Key head and one Value head:
- In LLaMA-2-70B: $H = 64, H_{\text{kv}} = 8 \implies G = 8$.
- Memory reduction is exactly $G\times = 8.0\times$.
- Empirical quality is virtually indistinguishable from full MHA while retaining MQA's high inference decode speed and memory savings.

---

## 4. Dynamic KV Cache Eviction: Heavy Hitter Oracle (H2O) & Attention Sinks

Even with GQA, as sequence lengths stretch to $s = 32,768$ or $131,072$, the linear memory growth $\mathcal{O}(s)$ eventually exhausts GPU HBM.

### 4.1 The Attention Sink Phenomenon (Xiao et al., 2023)
Why does autoregressive attention collapse if initial tokens are dropped?
Because softmax requires exponentiated scores to sum to 1:
$$\sum_{j=1}^t \exp\left(\frac{q_t k_j^\top}{\sqrt{d}}\right) = 1$$
Even when a token at step $t$ needs no contextual information from the beginning of the document, the softmax operator must dump residual attention probability mass somewhere. The model learns to use the **first 1 to 4 initial tokens as an "attention sink"** (an information dumping ground).
If these initial tokens are evicted from the KV cache, the softmax denominator becomes destabilized, leading to catastrophic degradation in perplexity!

### 4.2 The Heavy Hitter Oracle (H2O, Zhang et al., 2023)
Research reveals that the vast majority of historical tokens receive near-zero attention after their generation window. Only a small subset of "Heavy Hitter" tokens are consistently revisited.

At step $t$, the cumulative attention score of token $j$ is:

$$S_j = \sum_{\tau=j}^t \sum_{h=1}^H A_{\tau, j}^{(h)}$$

The **H2O Eviction Policy** maintains a fixed token budget $s_{\text{budget}} \ll s$:
1. **Initial Sink Tokens**: Pinned permanently: $\{1, 2, \dots, W_{\text{sink}}\}$ (typically $W_{\text{sink}} = 4$).
2. **Local Recency Window**: Preserves the last $W_{\text{recency}}$ tokens: $\{t - W_{\text{recency}} + 1, \dots, t\}$ (typically $W_{\text{recency}} = 64\text{--}128$).
3. **Heavy Hitters**: From the remaining intermediate tokens, retains the top-$k$ tokens with the highest cumulative scores $S_j$.
4. **Eviction**: When the cache reaches capacity $s_{\text{budget}}$, the token with the lowest cumulative score $S_j$ outside the sink and recency windows is permanently purged.

This reduces the KV footprint by **$60\text{--}80\%$** while retaining $>95\%$ of original model generation quality and task accuracy.

---

## 5. Tensor Parallel Serving: Sharding the KV Cache

When a model is partitioned across $N_{\text{tp}}$ GPUs using Megatron-LM style Tensor Parallelism, linear layers and attention heads are split across devices:

```
Input X (All GPUs hold identical X)
         │
         ├───▶ Column Parallel W_Q, W_K, W_V:
         │     GPU 0: Computes Q_0, K_0, V_0 (Heads 1..H/N) ──▶ Stores KV_0 in local VRAM
         │     GPU 1: Computes Q_1, K_1, V_1 (Heads H/N+1..) ──▶ Stores KV_1 in local VRAM
         │
         ├───▶ Local Scaled Dot-Product Attention:
         │     GPU 0: Softmax(Q_0 K_0^T) V_0 = HeadOutputs_0
         │     GPU 1: Softmax(Q_1 K_1^T) V_1 = HeadOutputs_1
         │
         └───▶ Row Parallel W_O:
               GPU 0: HeadOutputs_0 * W_O^{(0)} = PartialResult_0
               GPU 1: HeadOutputs_1 * W_O^{(1)} = PartialResult_1
                           │
                           ▼
               [ All-Reduce (SUM) across NVLink ]
                           │
                           ▼
               Final Output Y = PartialResult_0 + PartialResult_1
```

### 5.1 KV Cache Memory Sharding
In Tensor Parallelism, **Key and Value heads are partitioned across GPUs**:
- Total KV heads in model $= H_{\text{kv}}$.
- KV heads per GPU $= H_{\text{kv}} / N_{\text{tp}}$.
- Per-GPU KV Cache footprint:
  $$\boxed{\text{Mem}_{\text{KV\_per\_GPU}} = \frac{\text{Mem}_{\text{KV\_total}}}{N_{\text{tp}}} = \frac{4 \cdot n_{\text{layers}} \cdot H_{\text{kv}} \cdot d_{\text{head}} \cdot b \cdot s}{N_{\text{tp}}}}$$

By serving a 70B model across an 8-GPU node ($N_{\text{tp}} = 8$):
1. Model weights per GPU drop from $140 \text{ GB}$ to $17.5 \text{ GB}$.
2. KV cache footprint per GPU is divided by 8.
3. Total cluster memory available for active request batches expands by $8\times$.

---

## 6. Mathematical Derivations: Eviction Bound & Sparsity

### Derivation: Memory Compression Ratio of H2O
Let context length be $s = 4096$. We configure an H2O cache with:
- $W_{\text{sink}} = 4$ tokens.
- $W_{\text{recency}} = 128$ tokens.
- $k_{\text{heavy}} = 380$ tokens.
- Total budget: $s_{\text{budget}} = 4 + 128 + 380 = \mathbf{512 \text{ tokens}}$.

$$\text{Memory Compression Ratio} = \frac{s_{\text{budget}}}{s} = \frac{512}{4096} = \mathbf{\frac{1}{8} = 12.5\%}$$
$$\text{Memory Savings} = 1.0 - 0.125 = \mathbf{87.5\% \text{ reduction}}$$

Because the attention operation complexity per decode step scales linearly with cached keys ($\mathcal{O}(s)$), compressing the cache from 4096 to 512 tokens also yields an immediate **$8\times$ reduction in attention FLOPs and memory bus traffic** during the decode phase!
