# Topic 22.5: Exercises & Solutions — Context Construction & RAG Evaluation

---

### Problem 1: Context Precision (Mean Average Precision) Calculation (Easy)

An evaluation test suite queries a RAG retriever that returns $K = 5$ ranked chunks for a user question.
Two competing retrieval pipelines produce the following binary relevance vectors (where $1$ indicates a truly relevant chunk and $0$ indicates an irrelevant distractor):
- **Pipeline A:** $\mathbf{v}_A = [1, 0, 1, 0, 0]$ (Relevant chunks placed at ranks 1 and 3).
- **Pipeline B:** $\mathbf{v}_B = [0, 1, 0, 1, 0]$ (Relevant chunks placed at ranks 2 and 4).

1. Compute the Precision@$k$ for each position $k \in \{1, 2, 3, 4, 5\}$ for both pipelines.
2. Compute the final **Context Precision (Mean Average Precision)** for Pipeline A and Pipeline B.
3. What does the difference in scores demonstrate regarding the ranking objective of vector retrieval?

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute Precision@$k$:**
Recall that $\text{Precision@}k = \frac{\text{Number of relevant chunks in top } k}{k}$.

- **For Pipeline A ($\mathbf{v}_A = [1, 0, 1, 0, 0]$):**
  - $k = 1$: Chunks relevant $= 1 \implies \text{Prec@}1 = \frac{1}{1} = 1.000$
  - $k = 2$: Chunks relevant $= 1 \implies \text{Prec@}2 = \frac{1}{2} = 0.500$
  - $k = 3$: Chunks relevant $= 2 \implies \text{Prec@}3 = \frac{2}{3} \approx 0.667$
  - $k = 4$: Chunks relevant $= 2 \implies \text{Prec@}4 = \frac{2}{4} = 0.500$
  - $k = 5$: Chunks relevant $= 2 \implies \text{Prec@}5 = \frac{2}{5} = 0.400$

- **For Pipeline B ($\mathbf{v}_B = [0, 1, 0, 1, 0]$):**
  - $k = 1$: Chunks relevant $= 0 \implies \text{Prec@}1 = \frac{0}{1} = 0.000$
  - $k = 2$: Chunks relevant $= 1 \implies \text{Prec@}2 = \frac{1}{2} = 0.500$
  - $k = 3$: Chunks relevant $= 1 \implies \text{Prec@}3 = \frac{1}{3} \approx 0.333$
  - $k = 4$: Chunks relevant $= 2 \implies \text{Prec@}4 = \frac{2}{4} = 0.500$
  - $k = 5$: Chunks relevant $= 2 \implies \text{Prec@}5 = \frac{2}{5} = 0.400$

**Step 2: Compute Mean Average Precision (MAP):**
$$\text{Context Precision} = \frac{\sum_{k=1}^K (\text{Prec@}k \times v_k)}{\sum_{k=1}^K v_k}$$
Total relevant chunks $= 2$ for both pipelines.

- **For Pipeline A:**
  $$\text{Context Precision}_A = \frac{(\text{Prec@}1 \times 1) + (\text{Prec@}3 \times 1)}{2} = \frac{1.000 + 0.667}{2} = \frac{1.667}{2} \approx 0.8335 \quad (83.4\%)$$

- **For Pipeline B:**
  $$\text{Context Precision}_B = \frac{(\text{Prec@}2 \times 1) + (\text{Prec@}4 \times 1)}{2} = \frac{0.500 + 0.500}{2} = \frac{1.000}{2} = 0.5000 \quad (50.0\%)$$

**Step 3: Ranking Objective Analysis:**
Both pipelines retrieved the exact same total number of relevant documents ($2$ out of $5$). 
However, Pipeline A achieved an **83.4% Context Precision** vs Pipeline B's **50.0%**. 
Context Precision explicitly penalizes late retrieval: a relevant chunk at rank 1 is far more valuable than one at rank 4 because transformer attention mechanisms degrade when relevant context is diluted by early distractors.
</details>

---

### Problem 2: Mathematical Proof of Lost-in-the-Middle Reordering Optimization (Medium)

Let the position of a retrieved passage in a prompt context window be normalized to $x \in [0, 1]$, where $x = 0$ is the very start of the prompt and $x = 1$ is the very end.
Empirical observations by Liu et al. (2023) show that the probability of an LLM successfully attending to a fact at position $x$ follows a symmetric parabolic U-shaped attention function:
$$P_{\text{attend}}(x) = 4(x - 0.5)^2 + 0.20 = 4x^2 - 4x + 1.20$$
Notice that at the boundaries ($x = 0$ and $x = 1$), $P_{\text{attend}}(0) = P_{\text{attend}}(1) = 1.20$ (capped at $1.0$), while in the dead-center ($x = 0.5$), $P_{\text{attend}}(0.5) = 0.20$ (an 80% loss in attention fidelity).

Suppose a retriever provides two relevant facts: Fact 1 (highest relevance, weight $w_1 = 0.70$) and Fact 2 (secondary relevance, weight $w_2 = 0.30$).
The system must assign these two facts to two slots from a pool of 5 slots: $x \in \{0.0, 0.25, 0.50, 0.75, 1.0\}$.

1. Evaluate the expected attention score $\mathbb{E}[A] = w_1 P_{\text{attend}}(x_1) + w_2 P_{\text{attend}}(x_2)$ under a **Naive Sequential Layout** where Fact 1 is placed at $x = 0.25$ and Fact 2 is placed at $x = 0.50$.
2. Evaluate $\mathbb{E}[A]$ under the **Primacy-Recency Reordering** where Fact 1 is placed at $x = 0.0$ and Fact 2 is placed at $x = 1.0$.
3. Compute the percentage improvement gained purely through prompt layout restructuring.

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute $P_{\text{attend}}(x)$ values:**
- $x = 0.0$: $P_{\text{attend}}(0.0) = \min(1.0, 4(0 - 0.5)^2 + 0.20) = \min(1.0, 1.20) = 1.00$
- $x = 0.25$: $P_{\text{attend}}(0.25) = 4(0.25 - 0.5)^2 + 0.20 = 4(0.0625) + 0.20 = 0.25 + 0.20 = 0.45$
- $x = 0.50$: $P_{\text{attend}}(0.50) = 4(0.5 - 0.5)^2 + 0.20 = 0.20$
- $x = 0.75$: $P_{\text{attend}}(0.75) = 4(0.75 - 0.5)^2 + 0.20 = 0.45$
- $x = 1.0$: $P_{\text{attend}}(1.0) = 1.00$

**Step 2: Naive Sequential Layout ($x_1 = 0.25, x_2 = 0.50$):**
$$\mathbb{E}[A]_{\text{naive}} = (0.70 \times P_{\text{attend}}(0.25)) + (0.30 \times P_{\text{attend}}(0.50))$$
$$= (0.70 \times 0.45) + (0.30 \times 0.20) = 0.315 + 0.060 = 0.375$$

**Step 3: Primacy-Recency Layout ($x_1 = 0.0, x_2 = 1.0$):**
$$\mathbb{E}[A]_{\text{reordered}} = (0.70 \times P_{\text{attend}}(0.0)) + (0.30 \times P_{\text{attend}}(1.0))$$
$$= (0.70 \times 1.00) + (0.30 \times 1.00) = 0.70 + 0.30 = 1.000$$

**Step 4: Relative Improvement:**
$$\text{Improvement} = \frac{1.000 - 0.375}{0.375} \times 100\% = \frac{0.625}{0.375} \times 100\% \approx 166.7\%$$
Without spending a single dollar on bigger LLMs or better embedding models, merely rearranging the order of retrieved passages increases the expected attention signal by **$166.7\%$**, directly neutralizing the "Lost in the Middle" trap.
</details>

---

### Problem 3: RAG Failure Root-Cause Diagnostic Matrix (Hard)

A production RAG support bot for an airline reservation system exhibits occasional user complaints. 
The engineering team deploys the RAGAS framework and monitors four diagnostic scenarios across test queries.

| Case | Faithfulness | Answer Relevance | Context Precision | Context Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Case 1** | $0.20$ (Low) | $0.90$ (High) | $0.95$ (High) | $0.90$ (High) |
| **Case 2** | $0.98$ (High) | $0.30$ (Low) | $0.90$ (High) | $0.85$ (High) |
| **Case 3** | $0.95$ (High) | $0.85$ (High) | $0.25$ (Low) | $0.35$ (Low) |
| **Case 4** | $0.30$ (Low) | $0.40$ (Low) | $0.20$ (Low) | $0.20$ (Low) |

1. For each Case (1 through 3), diagnose the exact sub-system failure (e.g. Prompt instructions, LLM reasoning capacity, Retriever chunking, Vector index recall).
2. Propose the specific engineering remediation required for Case 1 vs. Case 3.
3. Why is decoupling Faithfulness from Answer Relevance indispensable when auditing hallucination risks?

<details>
<summary>View Complete Solution</summary>

**Step 1: Failure Diagnosis by Case:**
- **Case 1 (Low Faithfulness, High Precision/Recall):**
  - *Diagnosis:* **Generator Hallucination / Over-Confidence Failure.**
  - The retriever performed impeccably (high precision and recall: the correct documents were retrieved at the top of the prompt). However, the generator failed to ground its response on that context, instead hallucinating unverified claims from its parametric memory.
- **Case 2 (Low Relevance, High Faithfulness):**
  - *Diagnosis:* **Instruction Evasion / Prompt Misalignment.**
  - The model faithfully summarized the context without fabricating any false claims, but failed to actually answer the specific question asked by the user (answering a tangent or giving an evasive non-answer).
- **Case 3 (Low Precision/Recall, High Faithfulness):**
  - *Diagnosis:* **Retriever Deficiency / Ingestion Blind Spot.**
  - The retriever failed to find the relevant chunks (poor vector similarity or fragmented chunks). However, the LLM behaved responsibly—it adhered strictly to what little context was provided and refused to invent false facts.

**Step 2: Engineering Remediation:**
- **Remediation for Case 1 (Generator Failure):**
  1. Harden system prompt instructions: `"Answer strictly using only the provided context. If the fact cannot be derived from the context, state 'I do not have sufficient information'."`
  2. Lower generation temperature ($\tau \to 0.0$).
  3. Upgrade generator model capacity (e.g., from an unaligned small 7B model to an instruction-tuned 70B model with high citation discipline).
- **Remediation for Case 3 (Retriever Failure):**
  1. Switch from pure dense search to Hybrid BM25 + Dense + RRF.
  2. Implement semantic chunking with sliding-window overlap to eliminate context fragmentation.
  3. Expand index vocabulary with domain-specific synonym expansions or hypothetical document embeddings (HyDE).

**Step 3: Why Decoupling Faithfulness from Relevance is Critical:**
An answer can be 100% relevant to the user's question while being completely fabricated!
For example:
- Question: *"What is my flight confirmation code?"*
- Hallucinated Answer: *"Your confirmation code is AB1234."*
Answer Relevance is $1.0$ (it directly answered the exact question). But if "AB1234" is made up, Faithfulness is $0.0$.
Conversely, an answer can be 100% faithful to the context while completely failing to answer the question. Only by evaluating both metrics orthogonally can an engineering team pinpoint whether a bug lives in the vector retrieval index or in the generative LLM prompt.
</details>
