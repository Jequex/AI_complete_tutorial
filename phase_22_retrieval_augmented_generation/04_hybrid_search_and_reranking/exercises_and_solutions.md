# Topic 22.4: Exercises & Solutions — Hybrid Search & Reranking

---

### Problem 1: Okapi BM25 Numerical Scoring & Length Penalty Analysis (Easy)

A search engine indexes a corpus where the average document length is $\text{avgdl} = 200$ tokens.
A query term $q$ has an inverse document frequency $\text{IDF}(q) = 3.5$.
Standard BM25 hyperparameters are set: $k_1 = 1.2$ and $b = 0.75$.

Consider two documents that both contain exactly $f(q, D) = 4$ occurrences of query term $q$:
- **Document 1 ($D_1$):** Concise technical snippet with $|D_1| = 100$ tokens.
- **Document 2 ($D_2$):** Verbose report with $|D_2| = 600$ tokens.

1. Compute the BM25 score contribution for term $q$ in $D_1$ and $D_2$.
2. Compute the theoretical maximum score contribution if $f(q, D) \to \infty$ for a document with $|D| = \text{avgdl}$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute BM25 Score Contributions:**
The BM25 formula is:
$$\text{Score}(D, q) = \text{IDF}(q) \cdot \frac{f \cdot (k_1 + 1)}{f + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
Here $k_1 + 1 = 1.2 + 1 = 2.2$, and $f = 4$, so numerator $= 4 \times 2.2 = 8.8$.

- **For Document 1 ($|D_1| = 100$):**
  $$\frac{|D_1|}{\text{avgdl}} = \frac{100}{200} = 0.5$$
  Denominator adjustment:
  $$\text{denom}_1 = 4 + 1.2 \cdot (1 - 0.75 + 0.75 \times 0.5) = 4 + 1.2 \cdot (0.25 + 0.375) = 4 + 1.2 \cdot (0.625) = 4 + 0.75 = 4.75$$
  $$\text{Score}(D_1, q) = 3.5 \cdot \frac{8.8}{4.75} = 3.5 \cdot 1.8526 \approx 6.484$$

- **For Document 2 ($|D_2| = 600$):**
  $$\frac{|D_2|}{\text{avgdl}} = \frac{600}{200} = 3.0$$
  Denominator adjustment:
  $$\text{denom}_2 = 4 + 1.2 \cdot (1 - 0.75 + 0.75 \times 3.0) = 4 + 1.2 \cdot (0.25 + 2.25) = 4 + 1.2 \cdot (2.5) = 4 + 3.0 = 7.0$$
  $$\text{Score}(D_2, q) = 3.5 \cdot \frac{8.8}{7.0} = 3.5 \cdot 1.2571 \approx 4.400$$

Document 1 receives a **47.4% higher score** than Document 2 despite having the exact same term frequency, because $D_1$ has high keyword density while $D_2$ is heavily diluted by filler text.

**Step 2: Asymptotic Saturation Limit for $|D| = \text{avgdl}$:**
When $|D| = \text{avgdl}$, the length term simplifies:
$$1 - b + b(1) = 1.0 \implies \text{denom} = f + k_1$$
$$\lim_{f \to \infty} \frac{f \cdot (k_1 + 1)}{f + k_1} = k_1 + 1$$
$$\text{Max Score} = \text{IDF}(q) \cdot (k_1 + 1) = 3.5 \times 2.2 = 7.700$$
No matter how many thousands of times a spammer repeats term $q$, the maximum score can never exceed $7.70$.
</details>

---

### Problem 2: Reciprocal Rank Fusion (RRF) Dynamics & Smoothing Factor $k$ (Medium)

Consider two retrieval models: Model A (Sparse BM25) and Model B (Dense Bi-Encoder).
Two candidate documents $d_1$ and $d_2$ receive the following 1-based ranks:
- **Document 1 ($d_1$):** $r_A(d_1) = 1$ (Rank 1 in BM25), but $r_B(d_1) = 40$ (Rank 40 in Dense).
- **Document 2 ($d_2$):** $r_A(d_2) = 8$ (Rank 8 in BM25), and $r_B(d_2) = 8$ (Rank 8 in Dense).

1. Compute the fused score $\text{RRF}(d)$ for both documents under unsmoothed rank fusion where $k_{\text{RRF}} = 0$. Which document ranks higher?
2. Compute the fused score under standard Cormack smoothing where $k_{\text{RRF}} = 60$. Which document ranks higher?
3. Explain the mathematical intuition behind why setting $k \approx 60$ protects against single-model outlier noise.

<details>
<summary>View Complete Solution</summary>

**Step 1: Unsmoothed Rank Fusion ($k_{\text{RRF}} = 0$):**
$$\text{RRF}(d) = \frac{1}{r_A(d)} + \frac{1}{r_B(d)}$$
- **For Document 1:**
  $$\text{RRF}(d_1) = \frac{1}{1} + \frac{1}{40} = 1.0 + 0.025 = 1.025$$
- **For Document 2:**
  $$\text{RRF}(d_2) = \frac{1}{8} + \frac{1}{8} = 0.125 + 0.125 = 0.250$$
Under $k = 0$, Document 1 wins decisively ($1.025 > 0.250$) because Rank 1 yields a colossal weight ($1/1 = 1.0$) that completely eclipses all consensus signals from the other model.

**Step 2: Smoothed Rank Fusion ($k_{\text{RRF}} = 60$):**
$$\text{RRF}(d) = \frac{1}{60 + r_A(d)} + \frac{1}{60 + r_B(d)}$$
- **For Document 1:**
  $$\text{RRF}(d_1) = \frac{1}{60 + 1} + \frac{1}{60 + 40} = \frac{1}{61} + \frac{1}{100} \approx 0.016393 + 0.010000 = 0.026393$$
- **For Document 2:**
  $$\text{RRF}(d_2) = \frac{1}{60 + 8} + \frac{1}{60 + 8} = \frac{2}{68} \approx 0.029412$$
Under $k = 60$, **Document 2 wins** ($0.029412 > 0.026393$)!

**Step 3: Mathematical Intuition of $k_{\text{RRF}} \approx 60$:**
When $k = 0$, the ratio between rank 1 and rank 2 is $\frac{1/1}{1/2} = 2.0$ (a 100% penalty for dropping a single spot). 
With $k = 60$, the ratio between rank 1 and rank 2 is $\frac{60+2}{60+1} = \frac{62}{61} \approx 1.016$ (a modest $1.6\%$ difference).
The constant $k = 60$ compresses the hyper-sensitive dynamic range of top positions, allowing broad consensus across multiple independent retrievers (documents with solid top-10 presence across both engines) to triumph over high-variance single-model outliers.
</details>

---

### Problem 3: Low-Rank Bottleneck in Bi-Encoders vs. Full Cross-Attention (Hard)

Let a query $Q$ consist of $L_q$ tokens and document $D$ consist of $L_d$ tokens.
Let token contextual embeddings be $\mathbf{h}_i \in \mathbb{R}^d$ for query tokens and $\mathbf{g}_j \in \mathbb{R}^d$ for document tokens.

1. In a **Bi-Encoder**, mean pooling yields $\mathbf{q} = \frac{1}{L_q} \sum_{i=1}^{L_q} \mathbf{h}_i$ and $\mathbf{d} = \frac{1}{L_d} \sum_{j=1}^{L_d} \mathbf{g}_j$. Show that the score $s_{\text{bi}} = \mathbf{q}^\top \mathbf{d}$ corresponds to the trace of a rank-1 interaction matrix between all token pairs.
2. In a **Cross-Encoder**, full self-attention computes an all-to-all attention matrix $\mathbf{A} \in \mathbb{R}^{(L_q + L_d) \times (L_q + L_d)}$ across $H$ attention heads. Contrast the expressive rank capacity of the Cross-Encoder interaction matrix with the Bi-Encoder.
3. Formulate why a Cross-Encoder can easily distinguish the phrase *"bank of the river"* from *"river bank financial institution"* while an un-reranked Bi-Encoder frequently collapses their representations.

<details>
<summary>View Complete Solution</summary>

**Step 1: Rank-1 Interaction Decomposition of Bi-Encoder:**
$$\mathbf{q}^\top \mathbf{d} = \left( \frac{1}{L_q} \sum_{i=1}^{L_q} \mathbf{h}_i \right)^\top \left( \frac{1}{L_d} \sum_{j=1}^{L_d} \mathbf{g}_j \right) = \frac{1}{L_q L_d} \sum_{i=1}^{L_q} \sum_{j=1}^{L_d} \mathbf{h}_i^\top \mathbf{g}_j$$
Define the token interaction matrix $\mathbf{M} \in \mathbb{R}^{L_q \times L_d}$ where $M_{i, j} = \mathbf{h}_i^\top \mathbf{g}_j$.
Notice that:
$$\mathbf{M} = \mathbf{H} \mathbf{G}^\top$$
where $\mathbf{H} \in \mathbb{R}^{L_q \times d}$ and $\mathbf{G} \in \mathbb{R}^{L_d \times d}$.
However, when mean pooling is applied *prior* to interaction, the score is strictly:
$$s_{\text{bi}} = \mathbf{q}^\top \mathbf{d}$$
This collapses the entire multi-token interaction into the inner product of two single vectors. The score has rank $\min(1, d) = 1$. It cannot model token-level alignments (e.g. matching token $q_3$ specifically with token $d_{14}$ while ignoring $d_{15}$).

**Step 2: Cross-Encoder Expressive Capacity:**
In a Cross-Encoder, the concatenated sequence $[\mathbf{x}_1, \dots, \mathbf{x}_{L_q + L_d}]$ is processed through $L_{\text{layers}}$ transformer blocks.
In each layer and head, the attention matrix:
$$\mathbf{A}_{i, j} = \text{softmax}\left( \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}} \right)$$
can assign non-zero dynamic weights to any arbitrary pair of tokens $(q_i, d_j)$.
Across $H$ heads and $L$ layers, the Cross-Encoder interaction space has rank:
$$\text{Rank}(\text{Cross}) = \min(L_q + L_d, d) \times H \times L_{\text{layers}} \gg 1$$
It explicitly computes multi-hop query-to-document cross-attention paths.

**Step 3: Polysemy and Contextual Nuance Resolution:**
In the Bi-Encoder, the word *"bank"* in the query is pooled into an aggregate summary vector $\mathbf{q}$ alongside other words. 
If the document contains *"river"* and *"bank"*, the Bi-Encoder sees high token cosine alignment with both terms regardless of semantic syntax.
In contrast, the Cross-Encoder's multi-head attention directly connects the query modifier *"financial"* to document token *"bank"*, recognizing that the financial sense of "bank" does not attend to the geological sense of "river bank," driving the score to near zero.
This explains why Cross-Encoders achieve 5% to 15% higher NDCG@10 than Bi-Encoders.
</details>
