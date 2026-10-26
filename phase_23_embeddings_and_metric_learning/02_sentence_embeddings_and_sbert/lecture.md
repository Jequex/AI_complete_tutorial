# Topic 23.2: Sentence Embeddings & Sentence-BERT (SBERT)

---

## 1. Executive Summary & The 12-Year-Old Analogy

### The Cross-Encoder Computational Crisis
With the advent of BERT (Devlin et al., 2018), Transformer models achieved state-of-the-art accuracy in Natural Language Processing. In the standard BERT sentence-pair classification framework (the **Cross-Encoder**), two sentences $s_A$ and $s_B$ are concatenated with special boundary tokens:
$$\mathbf{x} = [\text{CLS}] \circ s_A \circ [\text{SEP}] \circ s_B \circ [\text{SEP}]$$
and passed through all $L$ Transformer layers where full bidirectional cross-attention computes interactions between every token of $s_A$ and every token of $s_B$.

While Cross-Encoders achieve exceptional accuracy on classification benchmarks, they suffer from a **quadratic inference catastrophe** in semantic search and clustering:
- To search a corpus of $N = 10,000$ sentences for the most similar pair requires evaluating all $\frac{N(N - 1)}{2} \approx 50,000,000$ combinations.
- On a high-performance NVIDIA V100 GPU, 50 million BERT forward passes require approximately **65 hours**.
- For large enterprise retrieval collections ($N = 1,000,000$), computing $5 \times 10^{11}$ combinations requires **over 700 years**.

To make semantic search tractable, we must compute **fixed-size sentence embeddings** $\mathbf{u} = f(s_A) \in \mathbb{R}^d$ and $\mathbf{v} = f(s_B) \in \mathbb{R}^d$ independently, allowing pair comparisons via instantaneous vector dot products or sub-millisecond approximate nearest neighbor (ANN) search.

```
+----------------------------------------------------------------------------------------------------+
|                                CROSS-ENCODER VS. BI-ENCODER (SBERT)                                |
|                                                                                                    |
|  CROSS-ENCODER (BERT):                                                                             |
|  [Sentence A] + [Sentence B] ──> [ BERT (Full Cross-Attention) ] ──> Score                         |
|  Complexity: O(N^2) forward passes. For N = 10k: 50,000,000 passes (~65 hours).                    |
|                                                                                                    |
|  BI-ENCODER (Sentence-BERT / SBERT):                                                               |
|  [Sentence A] ──> [ Shared BERT ] ──> [ Mean Pooling ] ──> u ──┐                                   |
|                                                                ├──> Cosine Similarity: u · v       |
|  [Sentence B] ──> [ Shared BERT ] ──> [ Mean Pooling ] ──> v ──┘                                   |
|  Complexity: O(N) forward passes to index corpus, O(1) cosine dot product search (~5 seconds!).    |
+----------------------------------------------------------------------------------------------------+
```

---

### The 12-Year-Old Analogy: Summarizing a Whole Book into GPS Coordinates

Imagine you are a judge at a school book competition:

1. **The Confused Robot (Raw BERT & The Anisotropy Cone):**
   - You ask a robot to compare two books: Book 1 is *"Treasure Island"* (pirates, wooden ships, cannons), and Book 2 is *"The Great Baking Championship"* (flour, strawberry frosting, ovens).
   - If you take raw BERT word averages without special metric tuning, the robot says:
     *"Well, both books are written in English! Both books use the letters E, T, and A! Both books have sentences ending in periods, and both have paragraphs! Therefore, these two books are 95% identical!"*
   - Because raw BERT gets obsessed with grammar and high-frequency syntax words, all sentences get squished into a tiny, cramped corner of the room called a **narrow cone**. Even completely unrelated sentences score over $0.90$ cosine similarity!

2. **The Smart Librarian (Sentence-BERT):**
   - Sentence-BERT acts like an expert librarian who ignores the plain paper, ink, and grammatical filler.
   - SBERT reads the pirate book and distills its unique essence into a single coordinate: `(Adventure: 0.95, Oceans: 0.90, Kitchen: 0.01)`.
   - SBERT reads the baking book and distills it into: `(Adventure: 0.05, Oceans: 0.02, Kitchen: 0.98)`.
   - Now, when you calculate their cosine similarity, SBERT says: *"Similarity is 0.04—these books have completely different topics!"* 
   - SBERT spreads books out across the entire world map (the unit hypersphere) so that true friends land side by side and strangers land on opposite continents.

---

## 2. Mathematical Symbol Decoder

| Symbol | Formal Domain | Description & Meaning |
| :--- | :--- | :--- |
| $s = (w_1, \dots, w_T)$ | Sequence | Input sentence containing $T$ subword tokens. |
| $\mathbf{H} \in \mathbb{R}^{T \times d}$ | Matrix | Final hidden state tensor from Transformer encoder; row $t$ is token vector $\mathbf{h}_t$. |
| $\mathbf{h}_{[\text{CLS}]} \in \mathbb{R}^d$ | Vector | Output representation of the leading classification token $[\text{CLS}] = \mathbf{h}_1$. |
| $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$ | Vector | Pooled sentence embedding vectors for sentences $s_A$ and $s_B$. |
| $\mathbb{S}^{d-1}$ | Riemannian Manifold | The $(d-1)$-dimensional unit hypersphere: $\{\mathbf{x} \in \mathbb{R}^d : \|\mathbf{x}\|_2 = 1\}$. |
| $\hat{\mathbf{u}} = \frac{\mathbf{u}}{\|\mathbf{u}\|_2}$ | Vector | $\ell_2$-normalized sentence embedding lying strictly on $\mathbb{S}^{d-1}$. |
| $\cos(\mathbf{u}, \mathbf{v})$ | $\mathbb{R} \times \mathbb{R} \to [-1, 1]$ | Cosine similarity: $\frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \hat{\mathbf{u}}^\top \hat{\mathbf{v}}$. |
| $\|\mathbf{u} - \mathbf{v}\|$ | Vector | Element-wise absolute difference vector: $[|u_1 - v_1|, \dots, |u_d - v_d|]^\top$. |
| $\mathbf{u} \odot \mathbf{v}$ | Vector | Element-wise Hadamard product vector: $[u_1 v_1, \dots, u_d v_d]^\top$. |
| $\mathbf{W}_{\text{clf}} \in \mathbb{R}^{3d \times k}$ | Matrix | Classification projection head mapping concatenated vector $[\mathbf{u}, \mathbf{v}, |\mathbf{u} - \mathbf{v}|]$ to $k$ classes. |
| $\boldsymbol{\mu} \in \mathbb{R}^d$ | Vector | Empirical mean center of the token/sentence representation distribution. |
| $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$ | Matrix | Empirical covariance matrix of representation vectors. |

---

## 3. The Anisotropy Problem of Raw BERT Embeddings

### 3.1 Definition of Anisotropy
In an **isotropic** embedding space, representations are uniformly distributed in all directions from the origin with identity covariance $\boldsymbol{\Sigma} = \sigma^2 \mathbf{I}_d$. Direction carries maximal entropy, and the expected cosine similarity between two randomly drawn unrelated vectors is zero:
$$\mathbb{E}_{\mathbf{u}, \mathbf{v} \sim \mathcal{U}(\mathbb{S}^{d-1})}[\cos(\mathbf{u}, \mathbf{v})] = 0$$

Empirical analysis by Ethayarajh (2019), Gao et al. (2019), and Li et al. (2020) revealed that **raw pre-trained Transformer embeddings (BERT, RoBERTa, GPT) are severely anisotropic**.
Instead of occupying the full hypersphere, token representations and naive sentence averages collapse into a **narrow, elongated cone** occupying an infinitesimal solid angle of $\mathbb{R}^d$:

```
          ISOTROPIC EMBEDDING SPACE                      ANISOTROPIC CONE (RAW BERT)
          
                 y                                              y
                 │   • (sports)                                 │         /  • (sports)
                 │                                              │        / • • • (all sentences)
         • (law) │         • (food)                             │       / • • • •
    ─────────────┼───────────── x                       ────────┼──────/─────── x
                 │                                              │     /  (narrow cone)
      • (tech)   │   • (music)                                  │    /
                 │                                              │   /
                 
     E[cos(u, v)] ≈ 0.0 (Unrelated sentences)         E[cos(u, v)] > 0.85 (Everything is similar!)
     Full expressive capacity of R^d used.            Severe dimensional collapse into 1st singular vector.
```

### 3.2 Mathematical Proof of the Cosine Artifact under Common Mean
Why does raw average pooling produce high cosine similarity between completely unrelated sentences?
Suppose every representation $\mathbf{u}$ is composed of a massive dominant shared mean vector $\boldsymbol{\mu} \in \mathbb{R}^d$ (encoding language syntax, token frequency bias, and position) plus a small zero-mean semantic variation $\boldsymbol{\epsilon}_u$:
$$\mathbf{u} = \boldsymbol{\mu} + \boldsymbol{\epsilon}_u, \quad \mathbf{v} = \boldsymbol{\mu} + \boldsymbol{\epsilon}_v$$
where $\mathbb{E}[\boldsymbol{\epsilon}_u] = \mathbf{0}$, $\mathbb{E}[\boldsymbol{\epsilon}_u^\top \boldsymbol{\epsilon}_v] = 0$ for unrelated sentences, and $\|\boldsymbol{\epsilon}_u\|_2 \ll \|\boldsymbol{\mu}\|_2$.

Now compute the inner product:
$$\mathbf{u}^\top \mathbf{v} = (\boldsymbol{\mu} + \boldsymbol{\epsilon}_u)^\top (\boldsymbol{\mu} + \boldsymbol{\epsilon}_v) = \|\boldsymbol{\mu}\|_2^2 + \boldsymbol{\mu}^\top \boldsymbol{\epsilon}_v + \boldsymbol{\mu}^\top \boldsymbol{\epsilon}_u + \boldsymbol{\epsilon}_u^\top \boldsymbol{\epsilon}_v$$
Taking the expectation over unrelated sentence pairs:
$$\mathbb{E}[\mathbf{u}^\top \mathbf{v}] = \|\boldsymbol{\mu}\|_2^2 > 0$$

Now evaluate the expected cosine similarity:
$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} \approx \frac{\|\boldsymbol{\mu}\|_2^2}{\sqrt{\|\boldsymbol{\mu}\|_2^2 + \|\boldsymbol{\epsilon}_u\|_2^2} \sqrt{\|\boldsymbol{\mu}\|_2^2 + \|\boldsymbol{\epsilon}_v\|_2^2}} \approx \frac{\|\boldsymbol{\mu}\|_2^2}{\|\boldsymbol{\mu}\|_2^2} \approx 1.0$$

Because $\|\boldsymbol{\mu}\|_2^2$ dwarfs the semantic variance $\|\boldsymbol{\epsilon}\|_2^2$, **raw BERT yields cosine similarities of $0.85 - 0.99$ for sentences that share zero words and discuss completely unrelated topics**! A sentence about quantum physics and a sentence about blueberry pancakes appear nearly identical under raw BERT cosine evaluation.

### 3.3 Root Causes of Anisotropy
1. **Token Frequency Bias:** High-frequency tokens (`the`, `of`, punctuation) appear in almost every sentence context. The gradient updates from these tokens pull all representations toward a common directional vector.
2. **Singular Value Dominance:** Singular Value Decomposition (SVD) of BERT's token embedding matrix reveals that the first 1–3 singular values are up to $50\times$ larger than the remaining singular values. Over 90% of variance is concentrated along a 1D or 2D line.
3. **Pre-training Loss Asymmetry:** Masked Language Modeling (MLM) trains the model to predict masked tokens conditioned on context, never explicitly optimizing for distance geometry across whole sentence sequences.

---

## 4. Sentence-BERT (SBERT) Architecture & Pooling Strategies

In 2019, Nils Reimers and Iryna Gurevych introduced **Sentence-BERT (SBERT)**, modifying pre-trained BERT using a **Siamese and Triplet network structure** to derive semantically meaningful sentence embeddings that can be compared using cosine similarity.

### 4.1 Pooling Mechanisms
Given input sentence $s = (w_1, \dots, w_T)$, the Transformer produces hidden states $\mathbf{H} = [\mathbf{h}_1, \dots, \mathbf{h}_T] \in \mathbb{R}^{T \times d}$. SBERT evaluates three pooling operations to collapse $\mathbf{H}$ into a fixed vector $\mathbf{u} \in \mathbb{R}^d$:

1. **`[CLS]`-Token Pooling:**
   $$\mathbf{u} = \mathbf{h}_{[\text{CLS}]} = \mathbf{h}_1$$
   *Empirical finding:* Fails poorly unless specifically fine-tuned with contrastive objectives; captures next-sentence classification artifact rather than sentence semantics.
2. **Max-Pooling:**
   $$u_j = \max_{t \in \{1, \dots, T\}} h_{t, j} \quad \forall j \in \{1, \dots, d\}$$
   *Empirical finding:* Emphasizes the single most salient token per dimension, but discards sentence syntactic structure.
3. **Mean-Pooling (Default SBERT standard):**
   $$\mathbf{u} = \frac{1}{\sum_{t=1}^T m_t} \sum_{t=1}^T m_t \mathbf{h}_t$$
   where $m_t \in \{0, 1\}$ is the attention mask indicator ($m_t = 0$ for padding tokens).
   *Empirical finding:* **Consistently outperforms `[CLS]` and Max-pooling across all semantic textual similarity (STS) benchmarks.**

```
                           SBERT SIAMESE NETWORK ARCHITECTURE
                           
                 Sentence A                                Sentence B
                     │                                         │
                     ▼                                         ▼
             [ Shared BERT-1 ]                         [ Shared BERT-2 ]
             (Tied Parameters)                         (Tied Parameters)
                     │                                         │
                     ▼                                         ▼
             [ Mean Pooling ]                          [ Mean Pooling ]
                     │                                         │
                     ▼                                         ▼
                 Vector u                                  Vector v
                     │                                         │
                     └───────────────────┬─────────────────────┘
                                         │
                                         ▼
                     Feature Vector: [ u, v, |u - v| ]
                                         │
                                         ▼
                     [ Linear Layer: W_clf in R^(3d x 3) ]
                                         │
                                         ▼
                         Softmax Cross-Entropy Loss
                      (Entailment, Neutral, Contradiction)
```

---

## 5. SBERT Training Objectives

SBERT is fine-tuned on large-scale Natural Language Inference (NLI) datasets: Stanford NLI (SNLI, 570k pairs) and Multi-Genre NLI (MNLI, 430k pairs). Each pair $(s_A, s_B)$ is labeled with one of three classes:
- **Entailment** ($y = 0$): $s_A$ implies $s_B$.
- **Neutral** ($y = 1$): $s_A$ and $s_B$ can both be true without contradiction.
- **Contradiction** ($y = 2$): $s_A$ and $s_B$ cannot both be true.

### 5.1 The Classification Objective Head
Given sentence embeddings $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$:
1. Compute the element-wise absolute difference: $|\mathbf{u} - \mathbf{v}| \in \mathbb{R}^d$, where $(|\mathbf{u} - \mathbf{v}|)_j = |u_j - v_j|$.
2. Concatenate into a $3d$-dimensional vector:
   $$\mathbf{h}_{\text{concat}} = [\mathbf{u}, \mathbf{v}, |\mathbf{u} - \mathbf{v}|] \in \mathbb{R}^{3d}$$
3. Project through a classification weight matrix $\mathbf{W}_{\text{clf}} \in \mathbb{R}^{3d \times k}$ (with $k = 3$) and bias $\mathbf{b} \in \mathbb{R}^k$:
   $$\mathbf{z} = \mathbf{W}_{\text{clf}}^\top [\mathbf{u}, \mathbf{v}, |\mathbf{u} - \mathbf{v}|] + \mathbf{b}$$
4. Minimize the standard cross-entropy loss:
   $$\mathcal{L}_{\text{clf}} = -\sum_{c=1}^k y_c \log \left( \frac{e^{z_c}}{\sum_{j=1}^k e^{z_j}} \right)$$

### 5.2 Why the Difference Vector $|\mathbf{u} - \mathbf{v}|$ is Critical
Why did Reimers & Gurevych include $|\mathbf{u} - \mathbf{v}|$ instead of just $[\mathbf{u}, \mathbf{v}]$?
- $[\mathbf{u}, \mathbf{v}]$ alone provides no explicit interaction between dimensions. A linear layer $\mathbf{W}_{\text{clf}}^\top [\mathbf{u}, \mathbf{v}] = \mathbf{W}_u^\top \mathbf{u} + \mathbf{W}_v^\top \mathbf{v}$ would evaluate the two sentences independently!
- The absolute difference $|\mathbf{u} - \mathbf{v}|$ acts as an element-wise distance metric. If dimension $j$ represents "animal", and $u_j \approx v_j$, then $|u_j - v_j| \approx 0$. If they disagree, $|u_j - v_j|$ is large.
- Reimers & Gurevych ablated feature combinations:
  - $[\mathbf{u}, \mathbf{v}]$: STS Spearman correlation = $67.0$
  - $[\mathbf{u}, \mathbf{v}, \mathbf{u} \odot \mathbf{v}]$: STS Spearman correlation = $70.1$
  - $[\mathbf{u}, \mathbf{v}, |\mathbf{u} - \mathbf{v}|]$: STS Spearman correlation = **$74.9$**
  - $[\mathbf{u}, \mathbf{v}, |\mathbf{u} - \mathbf{v}|, \mathbf{u} \odot \mathbf{v}]$: STS Spearman correlation = **$75.1$**

The difference vector $|\mathbf{u} - \mathbf{v}|$ forces the model to map semantically equivalent sentences to identical vector coordinates, directly dismantling the anisotropy cone!

### 5.3 The Regression Objective (Cosine STS)
For datasets providing real-valued semantic similarity scores $y \in [0, 1]$ (such as the Semantic Textual Similarity Benchmark, STS-B), SBERT is trained directly with Mean Squared Error (MSE) loss on cosine similarity:
$$\mathcal{L}_{\text{reg}} = \left( \cos(\mathbf{u}, \mathbf{v}) - y \right)^2 = \left( \frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} - y \right)^2$$

---

## 6. Geometric Normalization: Projection onto the Hypersphere $\mathbb{S}^{d-1}$

At inference time, downstream vector search engines (e.g., FAISS, Qdrant, Chroma, Pinecone) perform **$\ell_2$-normalization**:
$$\hat{\mathbf{u}} = \frac{\mathbf{u}}{\|\mathbf{u}\|_2}, \quad \hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$
Mapping all sentence embeddings onto the unit hypersphere $\mathbb{S}^{d-1} = \{\mathbf{x} \in \mathbb{R}^d : \|\mathbf{x}\|_2 = 1\}$.

### 6.1 Duality Between Cosine Similarity and Euclidean Distance
On the unit hypersphere, cosine similarity and squared Euclidean distance are strictly linearly equivalent.
Recall the expansion of squared Euclidean distance between unit vectors:
$$\|\hat{\mathbf{u}} - \hat{\mathbf{v}}\|_2^2 = (\hat{\mathbf{u}} - \hat{\mathbf{v}})^\top (\hat{\mathbf{u}} - \hat{\mathbf{v}}) = \|\hat{\mathbf{u}}\|_2^2 + \|\hat{\mathbf{v}}\|_2^2 - 2 \hat{\mathbf{u}}^\top \hat{\mathbf{v}}$$
Since $\|\hat{\mathbf{u}}\|_2 = 1$ and $\|\hat{\mathbf{v}}\|_2 = 1$:
$$\|\hat{\mathbf{u}} - \hat{\mathbf{v}}\|_2^2 = 1 + 1 - 2 \cos(\mathbf{u}, \mathbf{v}) = 2 - 2 \cos(\mathbf{u}, \mathbf{v})$$
Rearranging:
$$\cos(\mathbf{u}, \mathbf{v}) = 1 - \frac{1}{2} \|\hat{\mathbf{u}} - \hat{\mathbf{v}}\|_2^2$$

**Significance:**
- If $\cos(\mathbf{u}, \mathbf{v}) = 1.0 \implies \|\hat{\mathbf{u}} - \hat{\mathbf{v}}\|_2 = 0$ (Identical semantic point).
- If $\cos(\mathbf{u}, \mathbf{v}) = 0.0 \implies \|\hat{\mathbf{u}} - \hat{\mathbf{v}}\|_2 = \sqrt{2} \approx 1.414$ (Orthogonal).
- If $\cos(\mathbf{u}, \mathbf{v}) = -1.0 \implies \|\hat{\mathbf{u}} - \hat{\mathbf{v}}\|_2 = 2.0$ (Antipodal opposite poles).
Maximizing cosine similarity is mathematically identical to minimizing Euclidean distance on the unit hypersphere.

---

## 7. Post-Processing Baselines: BERT-Whitening

Can we fix the anisotropy problem of raw pre-trained BERT without expensive fine-tuning?
In 2021, Su et al. proposed **BERT-Whitening**, applying linear algebra centering and decorrelation to the pooled embeddings:

Given a collection of pooled sentence vectors $\{\mathbf{u}_i\}_{i=1}^M \subset \mathbb{R}^d$:
1. Compute the empirical mean vector $\boldsymbol{\mu}$:
   $$\boldsymbol{\mu} = \frac{1}{M} \sum_{i=1}^M \mathbf{u}_i$$
2. Compute the empirical covariance matrix $\boldsymbol{\Sigma}$:
   $$\boldsymbol{\Sigma} = \frac{1}{M} \sum_{i=1}^M (\mathbf{u}_i - \boldsymbol{\mu})(\mathbf{u}_i - \boldsymbol{\mu})^\top$$
3. Perform Singular Value Decomposition on $\boldsymbol{\Sigma}$:
   $$\boldsymbol{\Sigma} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top$$
4. Construct the whitening transformation matrix $\mathbf{W} \in \mathbb{R}^{d \times d}$:
   $$\mathbf{W} = \mathbf{U} \boldsymbol{\Lambda}^{-1/2}$$
5. Transform every embedding vector:
   $$\tilde{\mathbf{u}} = (\mathbf{u} - \boldsymbol{\mu}) \mathbf{W}$$

The transformed vectors $\tilde{\mathbf{u}}$ have zero mean and identity covariance $\mathbf{I}_d$. This centers the distribution at the origin and expands the squashed cone into an isotropic sphere, raising raw BERT STS performance from $58\%$ to $68\%$ without updating a single neural network weight!

---

## 8. Summary & Transition to Metric Learning

Sentence-BERT transformed dense retrieval by replacing $\mathcal{O}(N^2)$ cross-encoders with $\mathcal{O}(N)$ bi-encoders that map entire sentences into isotropic unit hyperspheres where cosine similarity mirrors human judgment.

In Topic 23.3, we delve deeper into the mathematical loss functions that govern representation geometry: **Metric Learning, Contrastive Loss, Triplet Loss, and Semi-Hard Negative Mining**.
