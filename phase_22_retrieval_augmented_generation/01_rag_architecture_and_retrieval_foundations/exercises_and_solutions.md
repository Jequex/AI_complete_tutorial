# Topic 22.1: Exercises & Solutions — RAG Architecture & Retrieval Foundations

---

### Problem 1: Parametric vs. Non-Parametric Memory Scaling Analysis (Easy)

An enterprise engineering organization has a corporate knowledge base of $N = 10^7$ document passages. Each passage has an average length of $L = 200$ tokens. 
The organization is choosing between two knowledge-integration architectures:
- **Strategy A (Full Parametric Fine-Tuning):** Continual pre-training / fine-tuning of an autoregressive LLM with $|\theta| = 70 \times 10^9$ parameters (LLaMA-70B) on the corpus.
- **Strategy B (Retrieval-Augmented Generation):** Indexing the corpus into a non-parametric vector store using an embedding model with dimension $d = 1024$ (FP16 representation) and serving an unmodified frozen LLM.

1. Calculate the raw storage footprint of the non-parametric vector index in gigabytes (excluding metadata).
2. Suppose the enterprise updates $10,000$ internal policy documents per week. Contrast the computational complexity and turnaround latency of updating Strategy A vs. Strategy B.

<details>
<summary>View Complete Solution</summary>

**Step 1: Raw Storage Footprint of the Vector Index:**
Each vector has dimension $d = 1024$. Under 16-bit floating point precision (FP16), each scalar takes 2 bytes:
$$\text{Bytes per vector} = 1024 \times 2 \text{ bytes} = 2048 \text{ bytes} = 2 \text{ KB}$$
For $N = 10^7$ passages:
$$\text{Total Vector Memory} = 10^7 \times 2048 \text{ bytes} = 2.048 \times 10^{10} \text{ bytes} \approx 20.48 \text{ GB}$$
Even in uncompressed dense format, the entire enterprise index of 10 million passages easily fits in the RAM of a single commercial server or GPU.

**Step 2: Computational Complexity of Updates:**
- **Strategy A (Parametric Update):** 
  Updating parametric weights requires running forward and backward passes through all 70 billion parameters over multiple training epochs. 
  The floating-point operations per token in training are approximately:
  $$\text{FLOPs/token} \approx 6 \times |\theta| = 6 \times (70 \times 10^9) = 4.2 \times 10^{11} \text{ FLOPs}$$
  For $10,000$ documents $\times 200$ tokens $= 2 \times 10^6$ tokens over 3 epochs:
  $$\text{Total Compute} \approx 3 \times (2 \times 10^6) \times (4.2 \times 10^{11}) \approx 2.52 \times 10^{18} \text{ FLOPs}$$
  Beyond raw compute, fine-tuning risks catastrophic forgetting of pre-existing skills and takes hours of distributed GPU cluster orchestration.
- **Strategy B (Non-Parametric Update):**
  Only the $10,000$ modified documents are processed. Each document is passed through the small embedding model (forward pass only, typically $\sim 10^8$ parameters):
  $$\text{Latency per doc} \approx 5 \text{ ms}$$
  The new vectors are inserted into the index in $\mathcal{O}(\log N)$ time, and obsolete vectors are deleted by ID. 
  The update completes in seconds with zero downtime, zero catastrophic forgetting, and negligible financial cost.
</details>

---

### Problem 2: Mathematical Derivation of RAG-Sequence vs. RAG-Token Marginalization (Medium)

Recall the mathematical formulations for RAG-Sequence and RAG-Token models:
- **RAG-Sequence:** $P_{\text{Seq}}(\mathbf{y} \mid q) = \sum_{z \in \mathcal{Z}} P_\eta(z \mid q) \prod_{t=1}^T P_\theta(y_t \mid q, z, y_{<t})$
- **RAG-Token:** $P_{\text{Tok}}(\mathbf{y} \mid q) = \prod_{t=1}^T \left[ \sum_{z \in \mathcal{Z}} P_\eta(z \mid q) P_\theta(y_t \mid q, z, y_{<t}) \right]$

1. Prove whether or not $P_{\text{Seq}}(\mathbf{y} \mid q) = P_{\text{Tok}}(\mathbf{y} \mid q)$ in the general case where $|\mathcal{Z}| > 1$ and $T > 1$.
2. Identify the specific constraint under which the two models become mathematically identical.
3. Explain which model architecture is capable of multi-document synthesis where token $y_1$ is supported by document $z_A$ and token $y_5$ is supported by document $z_B$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Mathematical Proof of Divergence:**
Let $\alpha_z = P_\eta(z \mid q)$ where $\sum_{z \in \mathcal{Z}} \alpha_z = 1$ and $\alpha_z \ge 0$.
Let $\beta_{t, z} = P_\theta(y_t \mid q, z, y_{<t})$.
The formulations simplify to:
$$P_{\text{Seq}}(\mathbf{y} \mid q) = \sum_{z \in \mathcal{Z}} \alpha_z \prod_{t=1}^T \beta_{t, z}$$
$$P_{\text{Tok}}(\mathbf{y} \mid q) = \prod_{t=1}^T \left( \sum_{z \in \mathcal{Z}} \alpha_z \beta_{t, z} \right)$$
For $T = 2$ and $|\mathcal{Z}| = 2$ (with documents $z_1, z_2$):
$$P_{\text{Seq}} = \alpha_1 \beta_{1, 1}\beta_{2, 1} + \alpha_2 \beta_{1, 2}\beta_{2, 2}$$
Expanding $P_{\text{Tok}}$:
$$P_{\text{Tok}} = (\alpha_1 \beta_{1, 1} + \alpha_2 \beta_{1, 2})(\alpha_1 \beta_{2, 1} + \alpha_2 \beta_{2, 2})$$
$$= \alpha_1^2 \beta_{1, 1}\beta_{2, 1} + \alpha_2^2 \beta_{1, 2}\beta_{2, 2} + \alpha_1 \alpha_2 (\beta_{1, 1}\beta_{2, 2} + \beta_{1, 2}\beta_{2, 1})$$
Subtracting the two:
$$P_{\text{Tok}} - P_{\text{Seq}} = \alpha_1 \alpha_2 (\beta_{1, 1}\beta_{2, 2} + \beta_{1, 2}\beta_{2, 1} - \beta_{1, 1}\beta_{2, 1} - \beta_{1, 2}\beta_{2, 2})$$
$$= \alpha_1 \alpha_2 (\beta_{1, 1} - \beta_{1, 2})(\beta_{2, 2} - \beta_{2, 1})$$
In general, $(\beta_{1, 1} - \beta_{1, 2}) \ne 0$ and $(\beta_{2, 2} - \beta_{2, 1}) \ne 0$. Therefore, $P_{\text{Seq}}(\mathbf{y} \mid q) \ne P_{\text{Tok}}(\mathbf{y} \mid q)$.

**Step 2: Constraint for Equivalence:**
The two models become mathematically identical if:
1. $|\mathcal{Z}| = 1$ (only a single document is retrieved), or
2. The retrieval prior is degenerate: $\exists z^* \text{ such that } P_\eta(z^* \mid q) = 1$ and $P_\eta(z \mid q) = 0, \forall z \ne z^*$, or
3. The token predictions are conditionally independent of the document $z$: $P_\theta(y_t \mid q, z, y_{<t}) = P_\theta(y_t \mid q, y_{<t})$.

**Step 3: Multi-Document Cross-Synthesis:**
The **RAG-Token** model natively enables cross-document synthesis. Because the document distribution is marginalized inside the product over time steps $t$, token $y_1$ can have high probability conditioned on $z_A$ while token $y_5$ has high probability conditioned on $z_B$, allowing the generator to merge disparate evidence into a coherent sentence.
</details>

---

### Problem 3: Groundedness & Hallucination Upper Bound (Hard)

Let $\mathcal{A}$ be the universe of factual atomic claims. Let $\mathcal{C} \subset \mathcal{A}$ denote the set of atomic claims verified by the retrieved context. 
Suppose a language model generates an answer composed of $M$ atomic claims $\{a_1, a_2, \dots, a_M\}$.
Under an unaugmented parametric generation regime, the probability that an individual generated claim is true is given by the model's parametric accuracy $\mu_{\text{param}} = P(a_i \in \mathcal{A}) = 0.65$.

When running RAG with strict prompt grounding:
1. If the retriever successfully retrieves relevant context ($E_{\text{ret}} = 1$), the model extracts claims directly from context with fidelity $P(a_i \in \mathcal{C} \mid E_{\text{ret}} = 1) = 0.98$.
2. If the retriever fails ($E_{\text{ret}} = 0$), the model falls back to parametric memory with probability $P_{\text{fallback}} = 0.10$ and refuses to answer with probability $0.90$.

Assume retrieval recall is $P(E_{\text{ret}} = 1) = 0.92$.
1. Derive the overall expected hallucination rate per claim $P(a_i \notin \mathcal{A})$ for RAG vs. Parametric Only.
2. For an answer requiring $M = 5$ independent claims, calculate the probability that the entire response is 100% hallucination-free ($P(\text{All True})$).

<details>
<summary>View Complete Solution</summary>

**Step 1: Expected Hallucination Rate per Claim:**
For **Parametric Only**:
$$P_{\text{param}}(\text{Hallucination}) = 1 - P(a_i \in \mathcal{A}) = 1 - 0.65 = 0.35 \quad (35\%)$$

For **RAG**:
Using the Law of Total Probability over retrieval event $E_{\text{ret}} \in \{0, 1\}$:
$$P(a_i \in \mathcal{A}) = P(a_i \in \mathcal{A} \mid E_{\text{ret}} = 1) P(E_{\text{ret}} = 1) + P(a_i \in \mathcal{A} \mid E_{\text{ret}} = 0) P(E_{\text{ret}} = 0)$$
Since $\mathcal{C} \subset \mathcal{A}$, claims verified by context are guaranteed factual:
$$P(a_i \in \mathcal{A} \mid E_{\text{ret}} = 1) = 0.98$$
When retrieval fails ($E_{\text{ret}} = 0$), the model either generates a parametric claim (which is true with prob 0.65) or refuses. Conditioning on an answer being generated:
$$P(a_i \in \mathcal{A} \mid E_{\text{ret}} = 0, \text{attempted}) = \mu_{\text{param}} = 0.65$$
Thus:
$$P(a_i \in \mathcal{A} \mid \text{attempted}) = (0.98 \times 0.92) + (0.65 \times 0.08) = 0.9016 + 0.0520 = 0.9536$$
The expected hallucination rate under RAG is:
$$P_{\text{RAG}}(\text{Hallucination}) = 1 - 0.9536 = 0.0464 \quad (4.64\%)$$
RAG reduces the per-claim hallucination rate from $35\%$ down to $4.64\%$—an **86.7% relative reduction**.

**Step 2: 5-Claim 100% Hallucination-Free Probability:**
Assuming independent claims:
- **Parametric Only:**
  $$P(\text{All 5 True}) = (\mu_{\text{param}})^5 = (0.65)^5 \approx 0.1160 \quad (11.6\%)$$
  There is an $88.4\%$ probability that at least one hallucination contaminates the answer.
- **RAG Augmented:**
  $$P(\text{All 5 True}) = (0.9536)^5 \approx 0.7892 \quad (78.9\%)$$
  RAG improves the probability of a completely factual multi-sentence answer by nearly **7x** ($11.6\% \to 78.9\%$).
</details>
