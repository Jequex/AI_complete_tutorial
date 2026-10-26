# Topic 22.5: Context Construction & RAG Evaluation

---

## 1. Executive Summary & The 12-Year-Old Analogy

Retrieving the right documents from a vector store is only half the battle. Even when the gold factual evidence is successfully retrieved, how that context is **constructed**, **ordered**, and **injected** into the LLM's prompt dramatically influences whether the model generates a factually accurate answer or a confident hallucination.

Two major discoveries govern modern context engineering:
1. **The "Lost in the Middle" Phenomenon (Liu et al., 2023):** Autoregressive LLMs exhibit a severe **U-shaped attention curve**. They readily recall facts placed at the very beginning of the context (Primacy Bias) and at the very end of the prompt (Recency Bias), but frequently overlook facts placed in the middle of long context windows.
2. **Context Compression & Distillation:** Shoveling raw, unpruned retrieved passages directly into the prompt inflates token costs, increases inference latency, and drowns the attention mechanism in irrelevant noise. Extractive and selective distillation filters irrelevant sentences prior to prompt compilation.

To optimize, monitor, and continuously test RAG pipelines, the AI community relies on the **RAGAS (Retrieval Augmented Generation Assessment)** evaluation framework, decomposing quality into four orthogonal metrics: **Faithfulness**, **Answer Relevance**, **Context Precision**, and **Context Recall**.

```
+--------------------------------------------------------------------------------------------------+
|                                    THE RAGAS EVALUATION MATRIX                                   |
+------------------------------------+---------------------------------+---------------------------+
| Metric                             | Evaluates                       | Mathematical Formulation  |
+------------------------------------+---------------------------------+---------------------------+
| Faithfulness (Groundedness)        | Generator -> Context            | |Verified Claims| / |Total Claims| |
| Answer Relevance                   | Generator -> Query              | Cosine(Embed(Answer), Embed(Query)) |
| Context Precision                  | Retriever -> Ground Truth       | Mean Average Precision (MAP@k) |
| Context Recall                     | Retriever -> Ground Truth       | |Found GT Claims| / |Total GT Claims| |
+------------------------------------+---------------------------------+---------------------------+
```

---

### The 12-Year-Old Analogy: The Grading Rubric for the Research Essay

> Imagine you are a science teacher grading student research reports.
>
> 1. **The "Lost in the Middle" Sandwich Problem:**
> If you give a student a 50-page binder of research, their eyes read the first page carefully (the title and intro) and the last page (the conclusion). If the most important lab result is buried on page 26 in the middle of the stack, the student skims right past it and misses it entirely! To make sure they see it, you paperclip the two most crucial pages to the very front and the very back of the binder.
>
> 2. **The Four-Part Grading Rubric:**
> When grading the student's final paper, you check four boxes:
> - **Faithfulness (Honesty):** Did the student invent any facts? If they claim the rocket flew to Mars, is that fact actually written in their binder, or did they make it up?
> - **Answer Relevance (On-Topic):** Did they actually answer the question you asked? If you asked *"Why did the engine overheat?"*, did they answer about the coolant leak, or did they write a story about their favorite astronaut?
> - **Context Precision (Library Skill):** When they searched the library, were their best books right on top of their desk, or did they bring 10 useless books and only 1 good one buried at the bottom?
> - **Context Recall (Thoroughness):** Did they bring all the necessary facts from the library, or did they leave half the answer sitting on the shelf?

---

## 2. Mathematical Symbol Decoder

| Symbol | Formal Domain | Description & Meaning |
| :--- | :--- | :--- |
| $\mathcal{C} = (c_1, \dots, c_k)$ | Sequence | Sequence of $k$ retrieved context passages. |
| $\mathcal{A}_{\text{ans}}$ | Discrete Set | Set of atomic factual propositions / claims generated in answer $y$. |
| $\mathcal{A}_{\text{ctx}}$ | Discrete Set | Set of factual propositions verified by the retrieved context $\mathcal{C}$. |
| $\mathcal{A}_{\text{GT}}$ | Discrete Set | Ground-truth reference statements defining complete factual answers. |
| $v_k \in \{0, 1\}$ | Binary Flag | Relevance indicator: $1$ if chunk at rank $k$ is relevant, $0$ otherwise. |
| $\text{Prec}@k$ | Scalar ($[0, 1]$) | Proportion of relevant chunks in the top-$k$ positions: $\frac{1}{k} \sum_{i=1}^k v_i$. |
| $\text{MAP}$ | Scalar ($[0, 1]$) | Mean Average Precision across all relevant retrieved chunks. |
| $\text{Faith}(y, \mathcal{C})$ | Metric ($[0, 1]$) | Faithfulness score: ratio of grounded claims to total claims. |
| $\text{Rel}(y, q)$ | Metric ($[0, 1]$) | Answer relevance score: semantic cosine similarity to query. |

---

## 3. The "Lost in the Middle" Effect & Prompt Layout

Liu et al. (2023) demonstrated that transformer language models retrieve and reason over facts with highest fidelity when target information appears at the beginning or end of the input context:

```
Performance / Retrieval Accuracy (%)
  ^
1.0 |  \                                                  /  U-SHAPED ATTENTION CURVE
    |   \                                                /
    |    \                                              /
    |     \____________________________________________/
    |                   "Lost in the Middle"
0.0 +------------------------------------------------------------> Relative Context Position
    0% (Start: Primacy)         50% (Middle)           100% (End: Recency)
```

### Optimal Primacy-Recency Context Reordering
Given a retrieved ranking $[c_1, c_2, c_3, c_4, c_5]$ where $c_1$ is highest relevance:
- **Naive Concatenation:** $[c_1, c_2, c_3, c_4, c_5]$ $\implies$ $c_3$ and $c_4$ sit in the middle dead-zone.
- **Primacy-Recency Reordering:** Alternates chunks between the front and the back:
  $$\text{Reordered Layout} = [c_1, c_3, c_5, c_4, c_2]$$
The single most important passage $c_1$ occupies the front (maximum primacy), while the second most important passage $c_2$ sits right adjacent to the user prompt at the bottom (maximum recency).

---

## 4. The RAGAS Evaluation Framework

```
                          [User Query q]
                             │       │
             ┌───────────────┘       └───────────────┐
             ▼                                       ▼
     [Retrieved Context C]                   [Generated Answer y]
             │                                       │
             ├───────────────────────────────────────┤
             │           FAITHFULNESS                │
             │   (Is answer grounded in context?)    │
             └───────────────────────────────────────┘
```

### Metric 1: Faithfulness (Groundedness)
Decomposes generated answer $y$ into $M$ discrete atomic propositions $\mathcal{A}_{\text{ans}} = \{a_1, a_2, \dots, a_M\}$ and tests whether each proposition is entailed by context $\mathcal{C}$:
$$\boxed{\text{Faithfulness}(y, \mathcal{C}) = \frac{|\{a \in \mathcal{A}_{\text{ans}} : \mathcal{C} \models a\}|}{|\mathcal{A}_{\text{ans}}|}}$$
A faithfulness score of $1.0$ guarantees zero hallucinated assertions.

### Metric 2: Answer Relevance
Quantifies whether the generated answer directly addresses the intent of query $q$ rather than evading or rambling:
$$\boxed{\text{Relevance}(y, q) = \cos(E(y), E(q)) \times \text{Coverage}(y, q)}$$

### Metric 3: Context Precision (Mean Average Precision)
Measures the ranking quality of the retriever. Relevant chunks should be concentrated at the very top of the retrieved list rather than dispersed among irrelevant distractors:
$$\boxed{\text{Context Precision@}K = \frac{\sum_{k=1}^K (\text{Precision@}k \times v_k)}{\sum_{k=1}^K v_k}}$$
where $\text{Precision@}k = \frac{\sum_{i=1}^k v_i}{k}$ and $v_k \in \{0, 1\}$.

### Metric 4: Context Recall
Measures whether the retriever captured all necessary factual components specified by gold reference statements $\mathcal{A}_{\text{GT}}$:
$$\boxed{\text{Context Recall} = \frac{|\{g \in \mathcal{A}_{\text{GT}} : \mathcal{C} \models g\}|}{|\mathcal{A}_{\text{GT}}|}}$$

---

## 5. Summary Checklist

- [x] LLMs suffer from the "Lost in the Middle" effect, attending predominantly to context boundaries.
- [x] Context reordering places top retrieved passages at the beginning (primacy) and end (recency) of the prompt.
- [x] Faithfulness evaluates whether claims are strictly grounded in retrieved passages (eliminating hallucinations).
- [x] Context Precision (MAP) evaluates whether relevant chunks occupy top ranks in the retrieval list.
- [x] Context Recall measures the fraction of ground-truth reference points retrieved.
