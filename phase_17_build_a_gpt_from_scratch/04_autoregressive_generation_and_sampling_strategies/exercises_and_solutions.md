# Topic 17.4: Exercises & Solutions — Autoregressive Generation & KV-Caching

---

### Problem 1: Manual Top-k and Top-p Nucleus Filtering (Easy)
Suppose a language model outputs the following unnormalized logits over a small vocabulary of 5 tokens:
$$\mathbf{z} = [3.0, 1.0, 0.0, 2.0, -1.0]$$
corresponding to words `['the', 'cat', 'dog', 'is', 'a']`.
Assume temperature $\tau = 1.0$.

1. Compute the raw softmax probabilities $P(w)$ for all 5 words.
2. Apply **Top-$k$ filtering** with $k = 2$. Which words are retained, and what are their renormalized probabilities?
3. Apply **Top-$p$ (Nucleus) filtering** with $p = 0.80$. Which words are retained in the nucleus, and what are their renormalized probabilities?

<details>
<summary>View Complete Solution</summary>

**Step 1: Softmax Probabilities:**
- $\exp(3.0) \approx 20.0855$ ('the')
- $\exp(1.0) \approx 2.7183$ ('cat')
- $\exp(0.0) = 1.0000$ ('dog')
- $\exp(2.0) \approx 7.3891$ ('is')
- $\exp(-1.0) \approx 0.3679$ ('a')
- Sum $= 20.0855 + 2.7183 + 1.0000 + 7.3891 + 0.3679 = 31.5608$

Probabilities:
- $P(\text{'the'}) = 20.0855 / 31.5608 \approx 0.6364$ (63.6%)
- $P(\text{'is'}) = 7.3891 / 31.5608 \approx 0.2341$ (23.4%)
- $P(\text{'cat'}) = 2.7183 / 31.5608 \approx 0.0861$ (8.6%)
- $P(\text{'dog'}) = 1.0000 / 31.5608 \approx 0.0317$ (3.2%)
- $P(\text{'a'}) = 0.3679 / 31.5608 \approx 0.0117$ (1.2%)

**Step 2: Top-k Filtering ($k=2$):**
Sort tokens descending:
1. `'the'`: $0.6364$
2. `'is'`: $0.2341$
Top 2 retained: `['the', 'is']`.
Sum of top 2 probabilities: $0.6364 + 0.2341 = 0.8705$.
Renormalized probabilities:
- $P'(\text{'the'}) = 0.6364 / 0.8705 \approx 0.7311$ (73.1%)
- $P'(\text{'is'}) = 0.2341 / 0.8705 \approx 0.2689$ (26.9%)
All other tokens: $0.0$.

**Step 3: Top-p Nucleus Filtering ($p=0.80$):**
Accumulate sorted probabilities:
1. `'the'`: cumulative $= 0.6364 < 0.80$ (retained)
2. `'is'`: cumulative $= 0.6364 + 0.2341 = 0.8705 \ge 0.80$ (retained, cutoff reached!)
The nucleus $\mathcal{V}^{(0.80)} = \{\text{'the'}, \text{'is'}\}$.
Notice that for this distribution, Top-$p$ ($p=0.80$) dynamically selects exactly the same top 2 words as $k=2$, renormalizing to $73.1\%$ and $26.9\%$.
</details>

---

### Problem 2: Memory Footprint of the KV-Cache (Medium)
Consider serving a GPT-2 Large model ($L = 36$ layers, $d_{\text{model}} = 1280$, $H = 20$ heads, $d_k = 64$) under FP16 (2 bytes per float).
Suppose a server hosts a batch of $B = 16$ concurrent user conversations, each reaching a context length of $T = 2048$ tokens.

1. Write the mathematical formula for the number of elements in the KV-Cache as a function of $B, L, T, d_{\text{model}}$.
2. Calculate the exact number of floating point elements stored in the cache.
3. Calculate the total VRAM in gigabytes (GB) consumed solely by the KV-Cache.

<details>
<summary>View Complete Solution</summary>

**Step 1: KV-Cache Elements Formula:**
For every layer $l \in \{1, \dots, L\}$, we store:
- Key tensor $K \in \mathbb{R}^{B \times T \times d_{\text{model}}}$
- Value tensor $V \in \mathbb{R}^{B \times T \times d_{\text{model}}}$
Total floating point elements:
$$\text{Elements}_{\text{KV}} = 2 \times B \times L \times T \times d_{\text{model}}$$

**Step 2: Numerical Calculation:**
- $B = 16$
- $L = 36$
- $T = 2048$
- $d_{\text{model}} = 1280$
$$\text{Elements}_{\text{KV}} = 2 \times 16 \times 36 \times 2048 \times 1280$$
$$\text{Elements}_{\text{KV}} = 1152 \times 2048 \times 1280 = 3{,}019{,}898{,}880 \text{ floats} \approx 3.02\text{ Billion floats}$$

**Step 3: VRAM Footprint:**
At 2 bytes per float (FP16):
$$\text{Total Bytes} = 3{,}019{,}898{,}880 \times 2 = 6{,}039{,}797{,}760 \text{ bytes}$$
Converting to gigabytes:
$$\text{Metric GB} = \frac{6{,}039{,}797{,}760}{10^9} \approx 6.04\text{ GB}$$
$$\text{Binary GiB} = \frac{6{,}039{,}797{,}760}{1024^3} = 5.62\text{ GiB}$$

The KV-Cache for 16 concurrent users requires ~6 GB of GPU VRAM exclusively for dynamic context cache, highlighting why modern inference engines utilize PagedAttention and Multi-Query Attention (MQA).
</details>

---

### Problem 3: Asymptotic Speedup Proof of KV-Caching (Hard)
Let $M$ denote the number of new tokens generated autoregressively following a prompt of length $T_{\text{prompt}}$.
In self-attention:
- Matrix multiplication $Q K^T$ for a sequence of length $t$ costs $2 t^2 d$ FLOPs.
- With KV-caching, generating at step $t$ uses a single query vector ($1 \times d$), costing $2 t d$ FLOPs.

1. Compute the exact summation of attention FLOPs required to generate $M$ tokens for Naive re-evaluation vs KV-Cached generation (assume $T_{\text{prompt}} = 0$ for simplicity).
2. Compute the asymptotic ratio $\frac{\text{FLOPs}_{\text{Naive}}}{\text{FLOPs}_{\text{Cached}}}$ as $M \to \infty$.
3. For $M = 1000$ tokens, calculate the theoretical speedup factor.

<details>
<summary>View Complete Solution</summary>

**Step 1: Exact FLOPs Summations:**
- **Naive generation:** At step $t \in \{1, \dots, M\}$, the model recomputes self-attention over all $t$ tokens:
  $$\text{FLOPs}_{\text{Naive}} = \sum_{t=1}^M 2 t^2 d = 2 d \sum_{t=1}^M t^2 = 2 d \frac{M(M+1)(2M+1)}{6} = \frac{d}{3} M(M+1)(2M+1) \approx \frac{2}{3} d M^3$$

- **KV-Cached generation:** At step $t$, only 1 new query vector attends to all $t$ stored keys:
  $$\text{FLOPs}_{\text{Cached}} = \sum_{t=1}^M 2 t d = 2 d \sum_{t=1}^M t = 2 d \frac{M(M+1)}{2} = d M(M+1) \approx d M^2$$

**Step 2: Asymptotic Speedup Ratio:**
$$\lim_{M \to \infty} \frac{\text{FLOPs}_{\text{Naive}}}{\text{FLOPs}_{\text{Cached}}} = \lim_{M \to \infty} \frac{\frac{2}{3} d M^3}{d M^2} = \frac{2}{3} M$$

The speedup scales **linearly with the number of generated tokens $M$**!

**Step 3: Evaluation at $M = 1000$:**
$$\text{Speedup} = \frac{\frac{1}{3} M (M+1)(2M+1)}{M(M+1)} = \frac{2M + 1}{3} = \frac{2001}{3} = 667\times$$

For a 1000-token response, KV-caching reduces self-attention compute by a factor of over **$667\times$**!
</details>
