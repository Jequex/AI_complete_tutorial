# Topic 22.2: Exercises & Solutions — Document Chunking & Embedding Strategies

---

### Problem 1: Sliding Window Chunk Count & Redundancy Ratio (Easy)

Consider a technical manual consisting of $T = 10,000$ tokens. An engineer implements a fixed sliding window chunker with chunk size $N = 500$ tokens and overlap $K = 100$ tokens.

1. Derive the closed-form formula for the total number of chunks $M(T, N, K)$ and compute the exact value for this manual.
2. Define the **Token Redundancy Ratio** $\rho = \frac{\text{Total Stored Tokens}}{T}$. Compute $\rho$ for $(N=500, K=100)$ and analyze what happens to storage costs and indexing throughput as $K \to N$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Formula for Number of Chunks $M(T, N, K)$:**
The first chunk consumes $N$ tokens. Each subsequent chunk advances the window by stride $S = N - K$ tokens.
The total number of chunks required to cover all $T$ tokens is given by:
$$M(T, N, K) = 1 + \left\lceil \frac{T - N}{N - K} \right\rceil = \left\lceil \frac{T - K}{N - K} \right\rceil$$
Substituting $T = 10,000$, $N = 500$, $K = 100$:
$$S = 500 - 100 = 400$$
$$M = 1 + \left\lceil \frac{10000 - 500}{400} \right\rceil = 1 + \left\lceil \frac{9500}{400} \right\rceil = 1 + \lceil 23.75 \rceil = 1 + 24 = 25 \text{ chunks}$$
The final (25th) chunk covers the remaining $300$ tokens (or can be padded / capped to length $N$).

**Step 2: Token Redundancy Ratio $\rho$:**
The total number of tokens indexed into the vector database across all $M$ chunks is:
$$\text{Total Indexed Tokens} \approx M \times N = 25 \times 500 = 12,500 \text{ tokens}$$
The redundancy ratio is:
$$\rho = \frac{12,500}{10,000} = 1.25 \quad (25\% \text{ storage overhead})$$
As $K \to N$, stride $S = N - K \to 1$:
$$M \approx \frac{T}{1} = T \implies \text{Total Tokens} \approx T \times N \implies \rho = N$$
For $N = 500$, as overlap approaches chunk size, the storage and embedding computation scale by a catastrophic **$500\times$ multiplier**, causing massive index bloat and excessive query duplicate retrieval.
A standard rule of thumb is setting overlap $K \in [0.10 \cdot N, 0.20 \cdot N]$ ($\rho \in [1.11, 1.25]$).
</details>

---

### Problem 2: Dynamic Semantic Chunking with Statistical Cutoffs (Medium)

A semantic chunker processes a sequence of 6 consecutive sentences $(s_1, \dots, s_6)$ from an article. 
The pairwise cosine distances between adjacent sentences $d_i = 1 - \cos(E(s_i), E(s_{i+1}))$ for $i \in \{1, 2, 3, 4, 5\}$ are measured as:
$$\mathbf{d} = [0.12, 0.18, 0.82, 0.15, 0.23]$$

1. Compute the empirical mean $\mu_d$ and sample standard deviation $\sigma_d$ of the distance vector $\mathbf{d}$.
2. Using a statistical threshold $\tau = \mu_d + 1.0 \cdot \sigma_d$, determine where semantic chunk boundaries are created.
3. Contrast this outcome with a naive fixed-character chunker that happens to split exactly between $s_1$ and $s_2$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute Mean and Standard Deviation:**
$$\mu_d = \frac{0.12 + 0.18 + 0.82 + 0.15 + 0.23}{5} = \frac{1.50}{5} = 0.30$$
Deviations from mean $(d_i - \mu_d)$:
$$[-0.18, -0.12, +0.52, -0.15, -0.07]$$
Squared deviations:
$$[0.0324, 0.0144, 0.2704, 0.0225, 0.0049]$$
Sum of squared deviations $= 0.3446$.
Sample standard deviation ($N-1 = 4$):
$$\sigma_d = \sqrt{\frac{0.3446}{4}} = \sqrt{0.08615} \approx 0.2935$$

**Step 2: Threshold Evaluation:**
$$\tau = \mu_d + 1.0 \cdot \sigma_d = 0.30 + 0.2935 = 0.5935$$
Comparing each distance $d_i$ against $\tau = 0.5935$:
- $d_1 = 0.12 < \tau$ (Keep together)
- $d_2 = 0.18 < \tau$ (Keep together)
- $d_3 = 0.82 > \tau$ $\implies$ **BOUNDARY TRIGGERED!**
- $d_4 = 0.15 < \tau$ (Keep together)
- $d_5 = 0.23 < \tau$ (Keep together)

The resulting semantic chunks are:
- **Chunk 1:** $[s_1, s_2, s_3]$
- **Chunk 2:** $[s_4, s_5, s_6]$

**Step 3: Comparison with Fixed-Size Chunker:**
The semantic distance vector revealed an abrupt topic shift between $s_3$ and $s_4$ ($d_3 = 0.82$). 
A naive fixed character chunker would split arbitrarily (e.g. between $s_1$ and $s_2$) based solely on word counts, tearing apart the strongly coherent thought in $s_1 \to s_2$ ($d_1 = 0.12$) while lumping unrelated topics across $s_3$ and $s_4$ into the same chunk.
</details>

---

### Problem 3: Mathematical Proof of Semantic Dilution in Centroid Embeddings (Hard)

Let an embedding function for text passage $c$ be modeled as the normalized sum of its word representations:
$$\mathbf{v}_c = \frac{1}{\sqrt{|c|}} \sum_{t=1}^{|c|} \mathbf{e}_t$$
where word vectors $\mathbf{e}_t$ are zero-mean independent random vectors with covariance $\mathbb{E}[\mathbf{e}_t \mathbf{e}_t^\top] = \frac{1}{d} \mathbf{I}_d$.

Suppose chunk $c$ contains a specific single-token factual needle $\mathbf{e}^*$ (which matches query $q$ with $\mathbf{e}^* \cdot \mathbf{q} = 1.0$), while the remaining $|c| - 1$ tokens in the chunk are unrelated filler words with zero expected correlation to $\mathbf{q}$.

1. Derive the expected cosine similarity $\mathbb{E}[\mathbf{v}_c \cdot \mathbf{q}]$ as a function of chunk length $|c|$.
2. Show that as chunk size $|c| \to \infty$, the retrieval signal decays as $\mathcal{O}(|c|^{-1/2})$.
3. If an embedding model requires a minimum cosine score of $\cos \ge 0.40$ to rank a chunk in the top-5 search results, determine the maximum allowable chunk length $|c|_{\max}$ before the factual needle is missed.

<details>
<summary>View Complete Solution</summary>

**Step 1: Expected Cosine Similarity:**
Let the chunk embedding be:
$$\mathbf{v}_c = \frac{\mathbf{e}^* + \sum_{t=1}^{|c|-1} \mathbf{e}_t}{\|\mathbf{e}^* + \sum_{t=1}^{|c|-1} \mathbf{e}_t\|_2}$$
For large dimension $d$ and unit-scaled tokens, by the central limit theorem and law of large numbers:
$$\mathbb{E}\left[ \left\| \mathbf{e}^* + \sum_{t=1}^{|c|-1} \mathbf{e}_t \right\|_2^2 \right] = \mathbb{E}[\|\mathbf{e}^*\|_2^2] + \sum_{t=1}^{|c|-1} \mathbb{E}[\|\mathbf{e}_t\|_2^2] = 1 + (|c| - 1) = |c|$$
Thus the norm scales as $\sqrt{|c|}$.
Taking the dot product with query unit vector $\mathbf{q}$ (where $\mathbf{e}^* \cdot \mathbf{q} = 1.0$ and $\mathbb{E}[\mathbf{e}_t \cdot \mathbf{q}] = 0$):
$$\mathbb{E}[\mathbf{v}_c \cdot \mathbf{q}] = \frac{\mathbf{e}^* \cdot \mathbf{q} + \sum_{t=1}^{|c|-1} \mathbb{E}[\mathbf{e}_t \cdot \mathbf{q}]}{\sqrt{|c|}} = \frac{1.0 + 0}{\sqrt{|c|}} = \frac{1}{\sqrt{|c|}}$$

**Step 2: Signal Decay:**
$$\mathbb{E}[\cos(\mathbf{v}_c, \mathbf{q})] = \frac{1}{\sqrt{|c|}} = \mathcal{O}(|c|^{-1/2})$$
As chunk size $|c|$ grows, the dot product signal of the crucial factual needle is diluted by the square root of the number of distracting tokens. At $|c| = 100$, the score drops to $0.10$; at $|c| = 1000$, it drops to $0.0316$.

**Step 3: Maximum Allowable Chunk Length:**
To guarantee expected retrieval above the ranking threshold $\tau = 0.40$:
$$\frac{1}{\sqrt{|c|}} \ge 0.40 \implies \sqrt{|c|} \le \frac{1}{0.40} = 2.50 \implies |c| \le (2.50)^2 = 6.25 \approx 6 \text{ tokens}$$
In real embedding models with multi-head self-attention, attention pooling somewhat mitigates pure average dilution, but the fundamental inverse relationship persists: larger chunks dilute isolated factual statements, mathematically proving why fine-grained chunking is mathematically superior for needle-in-a-haystack retrieval.
</details>
