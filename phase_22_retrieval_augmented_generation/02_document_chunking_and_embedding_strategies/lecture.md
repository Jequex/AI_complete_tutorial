# Topic 22.2: Document Chunking & Embedding Strategies

---

## 1. Executive Summary & The 12-Year-Old Analogy

A document cannot simply be fed into a dense vector index as an unpartitioned monolith. Foundation embedding models (such as `text-embedding-3`, `bge-large`, or `e5-mistral`) operate under strict maximum token limits (typically $512$ to $8192$ tokens). More fundamentally, projecting a 50-page document into a single $1024$-dimensional vector causes **semantic dilution**—averaging together hundreds of distinct concepts into a blurry vector centroid where specific factual needles are lost.

Conversely, chunking a document too aggressively creates **context fragmentation**, where anaphoric pronouns ("it", "they", "this system") are severed from their original referents, leaving the retrieved passage incomprehensible to the LLM.

**Document Chunking** is the algorithmic science of decomposing unstructured text into optimal semantic units that maximize both **retrieval precision** (matching specific user queries) and **generative synthesis** (providing sufficient contextual grounding for the reader).

```
+--------------------------------------------------------------------------------------------------+
|                                    THE CHUNK SIZE DILEMMA                                        |
+------------------------------------+-------------------------------------------------------------+
| Attribute                          | Small Chunks (100 - 250 tokens) | Large Chunks (800 - 2000 tokens) |
+------------------------------------+---------------------------------+---------------------------+
| Retrieval Specificity              | High (sharp cosine similarity)  | Low (diffuse centroid)    |
| Context Retention                  | Low (severed sentences/tables)  | High (self-contained)     |
| Noise-to-Signal in Context Window  | Low (only relevant facts)       | High (irrelevant padding) |
| Context Window Utilization         | Pack 10-20 distinct chunks      | 1-3 chunks max            |
| Failure Mode                       | Context fragmentation           | Semantic dilution         |
+------------------------------------+---------------------------------+---------------------------+
```

---

### The 12-Year-Old Analogy: Chopping an Encyclopedia into Index Cards

> Imagine you have a massive, 1,000-page encyclopedia and you want to organize it onto index cards so you can instantly answer quiz questions.
>
> **The Extreme Small Strategy (Single Sentences):**
> You cut the encyclopedia into single sentences. If someone asks *"How fast did the Apollo 11 rocket travel?"*, you find a card that says: *"It reached 24,791 miles per hour."* But wait! What reached that speed? A cheetah? A sports car? A meteor? Because the sentence was sliced off on its own, you lost the context—you don't know what "It" refers to!
>
> **The Extreme Large Strategy (Entire 50-Page Chapters):**
> You glue 50 pages together onto one giant board. When asked about Apollo 11's speed, the retriever hands you a 50-page board that discusses 1960s politics, NASA budgets, lunar geology, and astronaut biographies. You have to spend 20 minutes reading through thousands of irrelevant words just to find the single number!
>
> **The Goldilocks Chunking Strategy:**
> You neatly slice the encyclopedia into self-contained 250-word index cards that each describe one specific topic (e.g., *"Apollo 11: Propulsion and Velocity"*), and you make sure consecutive cards overlap by 50 words so no sentence gets cut in half. Perfect!

---

## 2. Mathematical Symbol Decoder

| Symbol | Formal Domain | Description & Meaning |
| :--- | :--- | :--- |
| $\mathcal{T} = (w_1, \dots, w_T)$ | Sequence | Sequence of $T$ subword tokens representing the raw document. |
| $N$ | Integer ($\mathbb{N}$) | Chunk length (in tokens or characters). |
| $K$ | Integer ($\mathbb{N}$) | Overlap size ($K < N$) shared between adjacent chunks. |
| $S = N - K$ | Integer ($\mathbb{N}$) | Stride / step size of the sliding window. |
| $\mathcal{C} = \{c_1, \dots, c_M\}$ | Set of Strings | The resulting set of $M = \lceil \frac{T - K}{N - K} \rceil$ segmented chunks. |
| $\Delta_{\text{delim}}$ | Ordered List | Delimiter hierarchy: $\Delta = [\text{"\textbackslash n\textbackslash n"}, \text{"\textbackslash n"}, \text{". "}, \text{" "}, \text{""}]$. |
| $\mathbf{s}_i \in \mathbb{R}^d$ | Vector ($\ell_2$-norm) | Dense semantic embedding of sentence $i$. |
| $\cos(\mathbf{s}_i, \mathbf{s}_{i+1})$ | Scalar ($[-1, 1]$) | Cosine similarity between sentence $i$ and adjacent sentence $i+1$. |
| $\tau_{\text{sem}}$ | Scalar threshold | Semantic distance threshold used to trigger a chunk boundary. |

---

## 3. The Modern Chunking Taxonomy

```
                              +---------------------------------------+
                              |          CHUNKING STRATEGIES          |
                              +---------------------------------------+
                                                  │
         ┌─────────────────────────┬──────────────┴──────────────┬─────────────────────────┐
         ▼                         ▼                             ▼                         ▼
   [Fixed-Size Overlap]       [Recursive Split]        [Document-Structure]       [Semantic Chunking]
   Window of N tokens         Hierarchical delimiters   Markdown headings,        Cosine drop between
   with K overlap tokens      (\n\n -> \n -> . -> space) HTML tags, JSON keys     adjacent sentences
```

### Strategy 1: Fixed-Size Chunking with Sliding Window Overlap
The simplest and most computationally efficient strategy. Given token sequence $\mathcal{T} = (w_1, \dots, w_T)$, window size $N$, and overlap $K$:
$$c_m = (w_{(m-1)S + 1}, \dots, w_{(m-1)S + N}), \quad \text{where } S = N - K$$
- **Advantages:** Deterministic $\mathcal{O}(T)$ linear time complexity; guaranteed chunk size constraints.
- **Disadvantages:** Slices arbitrarily through sentences, tables, code blocks, and logical thoughts.

### Strategy 2: Recursive Character Splitting (LangChain / LlamaIndex Standard)
Applies a hierarchy of delimiters:
$$\Delta = [\text{"\textbackslash n\textbackslash n"}, \text{"\textbackslash n"}, \text{". "}, \text{" "}, \text{""}]$$
The algorithm attempts to split at the highest structural boundary first (paragraphs). If a resulting chunk exceeds $N_{\max}$, it recursively descends to sentences (`. `), then phrases, then words (` `), and finally raw characters (`""`) as a last resort.
- **Invariance:** Never breaks a paragraph unless the paragraph exceeds $N_{\max}$.

### Strategy 3: Document-Structure & Markdown-Aware Splitting
Parses structured syntax directly:
- Markdown: Splits along `# H1`, `## H2`, `### H3` headers, tables, and fenced code blocks (` ``` `).
- HTML: Splits along `<article>`, `<section>`, `<div>`, `<p>`, and `<table>` DOM elements.
- Metadata Inheritance: Prepends heading breadcrumbs to each chunk:
  $$\text{Chunk } c = \text{"[Document > Phase 22 > Section 2] ... actual text ..."}$$

### Strategy 4: Semantic Similarity Boundary Chunking
Rather than relying on arbitrary character counts, semantic chunking dynamically detects **topic shifts**:
1. Split document into individual sentences: $(s_1, s_2, \dots, s_L)$.
2. Compute dense embeddings: $\mathbf{s}_i = E(s_i)$.
3. Compute adjacent cosine distances:
   $$d_i = 1 - \cos(\mathbf{s}_i, \mathbf{s}_{i+1}) = 1 - \frac{\mathbf{s}_i \cdot \mathbf{s}_{i+1}}{\|\mathbf{s}_i\|_2 \|\mathbf{s}_{i+1}\|_2}$$
4. Place a chunk boundary at index $i$ if $d_i$ exceeds a statistical threshold (e.g., $d_i > \mu_d + k \sigma_d$ or $d_i > \tau_{\text{sem}}$).

---

## 4. The Mathematical Trade-Off: Small vs. Large Chunks

Let $q$ be a user query, and let $c^*$ be the gold document passage containing the factual answer. 
The expected cosine similarity between query and chunk can be decomposed into signal and noise:
$$\mathbf{v}_c = \frac{1}{|c|} \sum_{w \in c} \mathbf{e}_w$$

As chunk length $|c|$ increases:
1. **Signal Dilution:** As $|c| \to \infty$, the embedding vector $\mathbf{v}_c$ approaches the mean language corpus vector $\mathbb{E}[\mathbf{e}]$, causing $\cos(\mathbf{v}_q, \mathbf{v}_c)$ to converge to an undifferentiated baseline.
2. **Context Completeness:** As $|c| \to 0$, individual tokens lack surrounding modifiers, causing high false-positive retrieval rates.

```
Cosine Similarity to Specific Query
  ^
1.0 |         Peak (Optimal ~250-500 tokens)
    |            /\
    |           /  \
    |          /    \
    |         /      \  Semantic Dilution (Averaging out)
    |        /        \_________________________
0.0 +---------------------------------------------> Chunk Size (Tokens)
    0       256       512       1024      4096
```

---

## 5. Summary Checklist

- [x] Unpartitioned documents cause semantic dilution and exceed embedding sequence limits.
- [x] Sliding window overlap ($K$ tokens) prevents facts spanning chunk borders from being severed.
- [x] Recursive chunkers preserve document paragraph and sentence hierarchies before resorting to word-level splits.
- [x] Semantic chunkers compute adjacent sentence cosine similarities and split at detected topic boundaries.
- [x] Optimal chunk sizes balance high retrieval precision with complete, self-contained context.
