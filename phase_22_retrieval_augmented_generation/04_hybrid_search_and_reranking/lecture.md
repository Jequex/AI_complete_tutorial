# Topic 22.4: Hybrid Search & Reranking

---

## 1. Executive Summary & The 12-Year-Old Analogy

Neither pure keyword matching (Sparse Search) nor pure semantic vector matching (Dense Search) is sufficient on its own for production RAG systems. Each methodology possesses complementary strengths and fatal blind spots:

- **Sparse Search (BM25 / SPLADE):** Operates on inverted lexical indices. It is unmatched at finding exact alphanumeric tokens, unique technical serial numbers (`ERR-9042`, `CVE-2024-38812`), acronyms, and product SKUs. However, it is **lexically rigid**—if a user searches for *"canine doctor"*, BM25 completely misses a document describing a *"veterinarian"*.
- **Dense Search (Bi-Encoder Embeddings):** Projects queries and documents into a shared continuous semantic vector space. It excels at conceptual generalization, understanding synonyms, and matching conceptual paraphrases. However, it suffers from **semantic blurriness**—rare alphanumeric strings and exact codes are lost in high-dimensional embedding averages.

**Hybrid Search** unifies both paradigms, running Sparse and Dense retrieval in parallel, combining their candidate rankings via **Reciprocal Rank Fusion (RRF)**, and passing the merged candidates to a high-precision **Cross-Encoder Reranker**.

```
+--------------------------------------------------------------------------------------------------+
|                                    SPARSE VS. DENSE RETRIEVAL                                    |
+------------------------------------+---------------------------------+---------------------------+
| Attribute                          | Sparse (Okapi BM25)             | Dense (Bi-Encoder / ANN)  |
+------------------------------------+---------------------------------+---------------------------+
| Representation                     | High-dimensional sparse vector  | Low-dimensional dense vec |
| Vocabulary Invariance              | Exact word / stem match only    | Semantic & conceptual     |
| Alphanumeric Codes & SKUs          | Exceptional precision           | Frequent catastrophic fail|
| Synonyms & Paraphrasing            | Fails (Zero lexical overlap)    | Exceptional recall        |
| Computational Index Substrate      | Inverted File Index (Lucene)    | Vector Graph (HNSW / IVF) |
| Latency                            | < 2 ms                          | 3 - 10 ms                 |
+------------------------------------+---------------------------------+---------------------------+
```

---

### The 12-Year-Old Analogy: The Fast Keyword Detective & The Deep Reading Judge

> Imagine a police detective agency trying to solve a crime.
>
> **The Fast Keyword Detective (BM25):**
> Officer Ctrl+F is a speed demon who searches millions of incident reports for exact words. If you ask for license plate *"TX-904-B"*, he finds it in half a second! But if you ask for *"a stolen luxury sports car"*, he gets completely confused unless the report literally contains the exact words *"stolen"*, *"luxury"*, and *"sports car"*. If the file says *"a hijacked red Ferrari"*, he walks right past it!
>
> **The Deep Reading Detective (Dense Bi-Encoder):**
> Detective Semantic understands meaning and themes. She knows immediately that a *"hijacked red Ferrari"* is a *"stolen luxury sports car"*. But if you give her the license plate *"TX-904-B"*, she shrugs because to her, numbers are just a meaningless soup of characters.
>
> **The RRF Hybrid Team with the Cross-Encoder Judge:**
> You let both detectives search the archives in parallel. They each hand their top 20 candidate files to the Chief Judge (the **Cross-Encoder**). The Judge sits at a desk, carefully places the suspect's description side-by-side with each file, reads every word with full cross-attention, and produces the single best verdict!

---

## 2. Mathematical Symbol Decoder

| Symbol | Formal Domain | Description & Meaning |
| :--- | :--- | :--- |
| $Q = (q_1, \dots, q_m)$ | Term Sequence | User query decomposed into lexical tokens. |
| $D$ | Text Passage | Candidate document chunk containing $|D|$ tokens. |
| $f(q_i, D)$ | Integer ($\mathbb{N}_0$) | Raw term frequency of query token $q_i$ in document $D$. |
| $\text{avgdl}$ | Scalar ($\mathbb{R}^+$) | Average document length across the entire corpus: $\frac{1}{N} \sum_j |D_j|$. |
| $k_1$ | Hyperparameter ($\sim 1.2 - 2.0$) | BM25 term frequency saturation parameter. |
| $b$ | Hyperparameter ($\sim 0.75$) | BM25 document length normalization parameter ($b \in [0, 1]$). |
| $\text{IDF}(q_i)$ | Scalar ($\mathbb{R}$) | Robertson-Spärck Jones Inverse Document Frequency. |
| $r_m(d)$ | Integer ($\mathbb{N}$) | 1-based ordinal rank of document $d$ in ranking system $m$. |
| $k_{\text{RRF}}$ | Constant ($\approx 60$) | Rank smoothing constant preventing top ranks from overwhelming fusion. |
| $\text{RRF}(d)$ | Scalar ($\mathbb{R}^+$) | Final fused reciprocal rank score for document $d$. |

---

## 3. Okapi BM25: Mathematical Formulation

Okapi BM25 scores document $D$ against query $Q$ by combining **term frequency saturation** with **document length normalization**:

$$\text{BM25}(D, Q) = \sum_{i=1}^m \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

### Component 1: Inverse Document Frequency (Robertson-Spärck Jones)
$$\text{IDF}(q_i) = \ln\left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \right)$$
where $N$ is total documents and $n(q_i)$ is the number of documents containing term $q_i$. Common words (e.g. "the", "system") receive near-zero weight; rare terms (e.g. "ERR-9042") receive high weight.

### Component 2: Non-Linear Term Frequency Saturation
Unlike standard TF (which scales linearly), BM25 asymptotes toward $k_1 + 1$ as frequency increases:
$$\lim_{f(q_i, D) \to \infty} \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + \text{const}} = k_1 + 1$$
This prevents a document that repeats a keyword 50 times from artificially dominating the index.

```
BM25 Term Score Contribution
  ^
  |                          Asymptote: (k1 + 1) * IDF
  |                            ---------------------------------
  |                           /
  |                          /
  |                         /  k1 controls the curvature of saturation
  |                        /
  |                       /
  |                      /
0 +------------------------------------------------------------> Term Frequency f
  0          2          4          6          8         10
```

---

## 4. Reciprocal Rank Fusion (RRF)

When merging scores from BM25 (which produces unbounded scores $[0, \infty)$) and Dense Embeddings (which produces cosine scores $[-1, 1]$), raw numerical scores cannot be added directly without distorting calibrations.

**Reciprocal Rank Fusion (RRF)** (Cormack et al., 2009) is an unsupervised, scale-invariant rank aggregation algorithm:
$$\boxed{\text{RRF}(d) = \sum_{m \in \mathcal{M}} \frac{1}{k_{\text{RRF}} + r_m(d)}}$$

Where:
- $\mathcal{M} = \{\text{BM25}, \text{Dense}\}$ is the set of retrieval models.
- $r_m(d) \in \{1, 2, 3, \dots\}$ is the 1-based rank position of document $d$ in retriever $m$.
- $k_{\text{RRF}} \approx 60$ is a smoothing constant that prevents high ranks (e.g. rank 1 vs rank 2) from monopolizing the merged list over documents that perform consistently well across both retrievers.

---

## 5. Two-Stage Retrieval: Bi-Encoder vs. Cross-Encoder

```
[Query q]
   │
   ├──> [BM25 Inverted Index] ──> Top-50 candidates ──┐
   │                                                  ├──> [RRF Fusion] ──> Top-30 Candidates
   └──> [Dense Bi-Encoder ANN] ──> Top-50 candidates ─┘                         │
                                                                                ▼
                                                                     [Cross-Encoder Reranker]
                                                                     Full cross-attention:
                                                                     [CLS] Query [SEP] Doc [EOS]
                                                                                │
                                                                                ▼
                                                                     Final Top-5 Grounded Chunks
```

### Bi-Encoder (First-Stage Fast Candidate Generation)
$$\mathbf{q} = E_Q(q), \quad \mathbf{d} = E_D(d), \quad \text{Score} = \mathbf{q} \cdot \mathbf{d}$$
- Query and document are encoded **independently**.
- Document embeddings can be precomputed and indexed in HNSW.
- Complexity: $\mathcal{O}(d)$ inner product; blazing fast ($< 5 \text{ ms}$).

### Cross-Encoder (Second-Stage High-Precision Reranking)
$$\text{Score} = \text{Transformer}(\text{Concat}(q, d))$$
- Full self-attention across every query token and every document token simultaneously.
- Detects subtle negations, word ordering, and semantic nuances.
- Complexity: $\mathcal{O}((L_q + L_d)^2)$; computationally heavy, but applied only to the top-$K$ candidates ($K \sim 30$).

---

## 6. Summary Checklist

- [x] Sparse search (BM25) isolates exact keywords and alphanumeric codes; Dense search understands conceptual synonyms.
- [x] $k_1$ bounds term frequency saturation; $b$ penalizes overly long documents.
- [x] Reciprocal Rank Fusion (RRF) safely merges disparate rankings without needing score normalization.
- [x] The two-stage paradigm (Bi-Encoder candidate retrieval followed by Cross-Encoder reranking) balances low query latency with maximal precision.
