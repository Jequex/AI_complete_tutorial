# Topic 20.3: Exercises & Solutions — Instruction Datasets & Multi-Turn Conversations

---

### Problem 1: Self-Instruct ROUGE-L Deduplication Tracing (Easy)

In the Self-Instruct pipeline (Wang et al., 2022), candidate synthetic instructions are evaluated against existing instructions in the dataset $\mathcal{D}$ using ROUGE-L (based on the Longest Common Subsequence, LCS).
A candidate instruction is accepted into the dataset if and only if:
$$\max_{s \in \mathcal{D}} \text{ROUGE-L}(s_{\text{candidate}}, s) < 0.70$$

Recall the ROUGE-L F1 formula:
$$R_{\text{LCS}} = \frac{\text{LCS}(s_1, s_2)}{|s_1|}, \quad P_{\text{LCS}} = \frac{\text{LCS}(s_1, s_2)}{|s_2|}, \quad F_{\text{LCS}} = \frac{2 \cdot R_{\text{LCS}} \cdot P_{\text{LCS}}}{R_{\text{LCS}} + P_{\text{LCS}}}$$
where $|s|$ is the word count of sequence $s$.

Suppose our current dataset contains one existing instruction $s_1$:
$$s_1 = \text{"Write a python function to reverse a list"}$$
A teacher model generates candidate instruction $s_2$:
$$s_2 = \text{"Write a python function to reverse an integer list"}$$

1. Compute $|s_1|$ and $|s_2|$.
2. Find the Longest Common Subsequence (LCS) between $s_1$ and $s_2$ at the word level.
3. Compute the ROUGE-L F1 score.
4. Will $s_2$ be accepted or discarded under the $\tau = 0.70$ threshold?

<details>
<summary>View Complete Solution</summary>

**Step 1: Word Counts:**
- $s_1$: `["Write", "a", "python", "function", "to", "reverse", "a", "list"]` $\implies |s_1| = 8$ words.
- $s_2$: `["Write", "a", "python", "function", "to", "reverse", "an", "integer", "list"]` $\implies |s_2| = 9$ words.

**Step 2: Longest Common Subsequence (LCS):**
The shared ordered words are:
$$\text{LCS}(s_1, s_2) = \text{["Write", "a", "python", "function", "to", "reverse", "list"]}$$
Length: $\text{LCS}(s_1, s_2) = 7$ words (only `"a"` vs `"an integer"` differs).

**Step 3: ROUGE-L Calculation:**
- Recall: $R_{\text{LCS}} = \frac{7}{8} = 0.875$
- Precision: $P_{\text{LCS}} = \frac{7}{9} \approx 0.7778$
- F1 score:
  $$F_{\text{LCS}} = \frac{2 \times \frac{7}{8} \times \frac{7}{9}}{\frac{7}{8} + \frac{7}{9}} = \frac{2 \times 7}{8 + 9} = \frac{14}{17} \approx 0.8235$$

**Step 4: Filtering Decision:**
Since $F_{\text{LCS}} \approx 0.8235 > 0.70$, candidate $s_2$ is considered too semantically and lexically redundant with $s_1$.
**Decision:** Discard $s_2$.
</details>

---

### Problem 2: Multi-Turn Loss Mask Construction (Medium)

Consider a multi-turn conversation with the following turns:
- **Turn 1 (User):** `"What is 5 + 7?"` (5 tokens: $u_1, u_2, u_3, u_4, u_5$)
- **Turn 2 (Assistant):** `"12"` (2 tokens: $a_1, a_2$, where $a_2$ is `<|im_end|>`)
- **Turn 3 (User):** `"Multiply that by 3."` (6 tokens: $u_6, u_7, u_8, u_9, u_{10}, u_{11}$)
- **Turn 4 (Assistant):** `"36"` (2 tokens: $a_3, a_4$, where $a_4$ is `<|im_end|>`)

For simplicity, omit role header tokens and assume the serialized sequence of tokens is:
$$\mathbf{w} = [u_1, u_2, u_3, u_4, u_5, a_1, a_2, u_6, u_7, u_8, u_9, u_{10}, u_{11}, a_3, a_4]$$
Total sequence length is $T = 15$.

1. Identify the inputs $\mathbf{z}_{\text{in}} = \mathbf{w}_{0:14}$ of length 14.
2. Construct the exact 14-element target label vector $\mathbf{t}$ under proper multi-turn loss masking.
3. How many tokens generate active loss? What is the denominator used in the loss averaging?

<details>
<summary>View Complete Solution</summary>

**Step 1: Input Sequence $\mathbf{z}_{\text{in}}$:**
Length 14 (omitting the final token $a_4$):
$$\mathbf{z}_{\text{in}} = [u_1, u_2, u_3, u_4, u_5, a_1, a_2, u_6, u_7, u_8, u_9, u_{10}, u_{11}, a_3]$$

**Step 2: Constructing Target Vector $\mathbf{t}$:**
Each token at index $i$ in $\mathbf{z}_{\text{in}}$ predicts the token at index $i+1$ in $\mathbf{w}$:
- Positions $i=0, 1, 2, 3$: within user query 1 predicting $u_2, u_3, u_4, u_5$. **Masked $\to -100$**.
- Position $i=4$: last token of user query 1 ($u_5$), predicting the first assistant token $a_1$. **Supervised $\to a_1$**.
- Position $i=5$: assistant token $a_1$, predicting assistant end token $a_2$. **Supervised $\to a_2$**.
- Positions $i=6, 7, 8, 9, 10, 11$: within user query 2 predicting $u_6, \dots, u_{11}$. **Masked $\to -100$**.
- Position $i=12$: last token of user query 2 ($u_{11}$), predicting the first token of assistant 2 ($a_3$). **Supervised $\to a_3$**.
- Position $i=13$: assistant token $a_3$, predicting assistant end token $a_4$. **Supervised $\to a_4$**.

Therefore, the target vector is:
$$\mathbf{t} = [-100, -100, -100, -100, a_1, a_2, -100, -100, -100, -100, -100, -100, a_3, a_4]$$

**Step 3: Active Supervision Count:**
Active target entries are $a_1, a_2, a_3, a_4$.
There are exactly $4$ active tokens. The denominator in $\mathcal{L}_{\text{SFT}}$ is $N_{\text{active}} = 4$.
</details>

---

### Problem 3: Function Calling Masking & Environment Leakage Proof (Hard)

In an agentic function-calling model, an engineer proposes the following training protocol:
> *"Because tool outputs are part of the conversation, we should train the model with causal cross-entropy over the `<tool_response>` tokens as well, so the model learns what API responses look like."*

1. Formally prove why supervising `<tool_response>` tokens leads to model hallucinations and degradation of agentic reliability.
2. Outline the exact state machine of supervision for an agent turn executing multiple tool calls.

<details>
<summary>View Complete Solution</summary>

**Step 1: Formal Proof of Environmental Leakage & Hallucination:**
Let $E$ be an external non-deterministic environment (e.g. database, random number generator, real-time stock ticker).
The environment returns an observation $o_t \sim P_{\text{env}}(o \mid c_t)$, where $c_t$ is the model's tool call.
If the model is trained with cross-entropy loss over $o_t$:
$$\mathcal{L}_{\text{tool}} = -\sum_{j} \log P_\theta(o_{t, j} \mid c_t, o_{t, <j})$$
1. **Hallucination Incentive:** The model is penalized for not predicting the exact real-time output of an external server (e.g. predicting today's weather in Berlin as 18°C when the sensor read 19°C). This forces the model to allocate weights to "simulating" external APIs rather than querying them.
2. **Short-Circuiting (Skipping Tool Invocations):** At inference time, instead of emitting a `<tool_call>` tag and yielding control to the host environment, the model attempts to synthesize fake `<tool_response>` blocks internally from its own hallucinated distribution, corrupting factual integrity.

**Conclusion:** Tool responses are stochastic environmental observations. They must be conditioned on as causal context, but must always have target label $-100$.

**Step 2: Exact State Machine for Agent Supervision:**
```text
State 1: System & User Prompt      ──> Target = -100 (MASKED)
State 2: Assistant Thought / Call  ──> Target = Token IDs (SUPERVISED)
State 3: Tool Call Closing Tag     ──> Target = </tool_call> (SUPERVISED)
State 4: Tool API Response Payload ──> Target = -100 (MASKED)
State 5: Assistant Final Answer    ──> Target = Token IDs (SUPERVISED)
State 6: Conversation End Tag     ──> Target = <|im_end|> (SUPERVISED)
```
</details>
