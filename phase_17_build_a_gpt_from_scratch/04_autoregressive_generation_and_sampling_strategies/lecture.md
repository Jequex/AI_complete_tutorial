# Topic 17.4: Autoregressive Generation, Sampling Strategies & KV-Caching

---

## 1. Executive Summary & The 12-Year-Old Analogy

Once a Generative Pre-trained Transformer (GPT) is trained, inference takes the form of an iterative, sequential feedback loop called **Autoregressive Generation**:
1. The user supplies an initial prompt sequence $\mathbf{x}_{1:T} = (x_1, x_2, \dots, x_T)$.
2. The model outputs a probability distribution over the vocabulary for the next token:
   $$P(x_{T+1} = v \mid \mathbf{x}_{1:T})$$
3. A decoding algorithm picks a single token $x_{T+1}$ according to a sampling policy (Greedy, Temperature, Top-$k$, or Top-$p$ Nucleus).
4. The chosen token is appended to the prompt, forming $\mathbf{x}_{1:T+1}$, and the process repeats until an end-of-text token (`<|endoftext|>`) or maximum length is reached.

```
                  THE AUTOREGRESSIVE GENERATION LOOP
                  
   Prompt: "The capital of France is"
                     │
                     ▼
           [ GPT Forward Pass ]
                     │
                     ▼ Logits z ∈ ℝ^V
           [ Sampling Strategy ] (Greedy, Top-k, Top-p, Temp)
                     │
                     ▼
             Sampled: "Paris"
                     │
                     ▼ Append to sequence
   Updated Prompt: "The capital of France is Paris"
                     │
                     ▼ (Repeat until <|endoftext|>)
```

Naively re-feeding the entire growing sequence on every step recomputes all past Key and Value vectors from scratch, causing **quadratic time complexity** $\mathcal{O}(T^2)$ per generated sequence. The **Key-Value (KV) Cache** stores past Key and Value projection vectors in GPU VRAM, allowing the model to process **only the single newest token**, reducing the per-step computation from $\mathcal{O}(T)$ to $\mathcal{O}(1)$ attention activations, and overall sequence generation from $\mathcal{O}(T^2)$ to $\mathcal{O}(T)$!

---

### The 12-Year-Old Analogy: The Storyteller's Magic Dice & The Memory Notebook

Imagine a master fantasy storyteller making up a bedtime adventure:

1. **The Magic Dice (Sampling Strategies):**
   - **Greedy Search (The Boring Robot):** Every time the storyteller picks a word, they pick the single most predictable word on the page. *"The knight went to the castle and the castle was in the castle..."* It repeats itself over and over because it has no imagination!
   - **Temperature (The Wildness Dial):**
     - Low Temperature ($\tau = 0.2$): The storyteller is conservative, sticking closely to obvious facts.
     - High Temperature ($\tau = 2.0$): The storyteller drinks 5 cups of espresso and rolls a wild 100-sided die! Sentences become crazy and chaotic: *"The knight danced with purple galactic waffles!"*
   - **Top-$p$ Nucleus Sampling (The Sensible Filter):**
     - Before rolling the dice, the storyteller looks at the top most likely words until their combined probabilities reach 90% ($p=0.9$). Any nonsensical, weird words in the bottom 10% are instantly thrown in the trash! You get creative storytelling without crazy gibberish.

2. **The Memory Notebook (The KV-Cache):**
   - **The Forgetful Storyteller (No Cache):** Imagine that to write chapter 100, the author re-reads all 99 previous chapters word-by-word before writing every single new sentence! It takes hours just to write one paragraph.
   - **The Genius with a Notebook (KV-Cache):** The author writes key character facts into a notebook once. When writing sentence 101, they only read their quick summary notes, write the sentence, add one note to the pad, and keep moving. Writing stays fast no matter how long the book gets!

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $\mathbf{z} \in \mathbb{R}^V$ | Logits Vector | Unnormalized output scores over the vocabulary for position $T$. |
| $\tau \in \mathbb{R}^+$ | Positive Scalar | Sampling temperature parameter ($\tau \to 0$ approaches greedy argmax). |
| $p_i$ | Probability $\in [0, 1]$ | Softmax probability assigned to vocabulary candidate $i$. |
| $k \in \mathbb{N}$ | Integer | Top-$k$ cutoff parameter; retains the $k$ highest logit tokens. |
| $p \in (0, 1]$ | Probability Threshold | Top-$p$ (nucleus) threshold; retains the smallest set with cumulative mass $\ge p$. |
| $K_{\text{cache}}, V_{\text{cache}}$ | Tensors $\mathbb{R}^{T_{\text{past}} \times d_k}$ | Historical keys and values saved across preceding autoregressive steps. |
| $\mathcal{O}(T^2)$ vs $\mathcal{O}(T)$ | Complexity | Algorithmic time complexity of naive generation vs KV-cached generation. |

---

## 3. Decoding and Sampling Strategies

### 3.1 Greedy Search (Argmax)
Greedy search deterministically chooses the single token with the largest logit:
$$x_{T+1} = \arg\max_{v \in \{0, \dots, V-1\}} z_v$$

**Limitation:** It suffers from repetitive loops (e.g. *"I don't know, I don't know, I don't know..."*) and misses globally high-probability paths because it cannot backtrack.

---

### 3.2 Temperature Scaling
Temperature $\tau > 0$ scales the logits prior to applying softmax:
$$P(x_{T+1} = i) = \frac{\exp(z_i / \tau)}{\sum_{j=1}^V \exp(z_j / \tau)}$$

- **When $\tau \to 0$:** The largest logit dominates completely:
  $$\lim_{\tau \to 0^+} P(x_{T+1} = i) = \begin{cases} 1 & \text{if } i = \arg\max_v z_v \\ 0 & \text{otherwise} \end{cases}$$
- **When $\tau = 1.0$:** Standard unaltered distribution from model pre-training.
- **When $\tau \to \infty$:** The distribution approaches uniform noise over all $V$ tokens:
  $$\lim_{\tau \to \infty} P(x_{T+1} = i) = \frac{1}{V}$$

---

### 3.3 Top-$k$ Truncation
Top-$k$ sampling (Fan et al., 2018) sorts all tokens by logit descending, keeps only the top $k$ candidates, and sets all remaining logits to $-\infty$:
$$z_i' = \begin{cases} z_i & \text{if } \text{rank}(z_i) \le k \\ -\infty & \text{otherwise} \end{cases}$$

**Limitation:** A fixed $k$ is rigid. If the context is strictly constrained (e.g., *"The President of the United..."*), even $k=10$ might include irrelevant tokens. If the context is wide open, $k=10$ may be too restrictive.

---

### 3.4 Top-$p$ (Nucleus) Sampling (Holtzman et al., 2020)
Nucleus sampling dynamically selects the smallest subset of tokens $\mathcal{V}^{(p)} \subset \mathcal{V}$ whose cumulative probability exceeds threshold $p \in (0, 1]$:
$$\sum_{i \in \mathcal{V}^{(p)}} P(x_{T+1} = i) \ge p$$

Tokens outside $\mathcal{V}^{(p)}$ have their probabilities zeroed, and the remaining probabilities are renormalized:
$$P'(x_{T+1} = i) = \begin{cases} \frac{P(x_{T+1} = i)}{\sum_{j \in \mathcal{V}^{(p)}} P(x_{T+1} = j)} & \text{if } i \in \mathcal{V}^{(p)} \\ 0 & \text{otherwise} \end{cases}$$

This dynamically expands the candidate pool when uncertainty is high and contracts it to a single token when confidence is near $1.0$.

---

## 4. The Key-Value (KV) Cache Mechanism

In standard causal self-attention, for token $i$, the query $\mathbf{q}_i$ computes dot products with all keys $\mathbf{k}_1, \dots, \mathbf{k}_i$:
$$\mathbf{c}_i = \sum_{j=1}^i \text{softmax}\left(\frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}}\right) \mathbf{v}_j$$

Notice an essential physical property: **past keys $\mathbf{k}_j$ and past values $\mathbf{v}_j$ do not change as new tokens are generated!**
Because of causal masking, token $j < i$ is completely independent of future token $i$. Re-projecting past tokens through $W_k$ and $W_v$ on every new step is purely redundant work.

```
                          KV-CACHE ARCHITECTURE
                          
Step t (New Token x_t):
                                  ┌───────────────────────────┐
  New Token x_t ──> Embeddings ──>│ Project Q_t, K_t, V_t     │
                                  └─────────────┬─────────────┘
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
              Query Q_t                    New Key K_t                   New Value V_t
                 │                              │                              │
                 │                     ┌────────┴────────┐            ┌────────┴────────┐
                 │                     │ Append to Cache │            │ Append to Cache │
                 │                     └────────┬────────┘            └────────┬────────┘
                 │                              ▼                              ▼
                 │                      K_all = [K_past, K_t]          V_all = [V_past, V_t]
                 │                              │                              │
                 └───────────────( Q_t * K_all^T )                             │
                                        │                                      │
                                  Softmax Weights                              │
                                        │                                      │
                                        └──────────────( A * V_all )───────────┘
                                                           │
                                                           ▼
                                                Context Vector c_t
```

### Algorithmic Complexity Comparison:
- **Without KV-Cache (Naive):**
  Generating $M$ tokens requires processing lengths $1, 2, 3, \dots, M$:
  $$\text{Total Compute} \propto \sum_{t=1}^M t^2 \approx \frac{M^3}{3} \implies \mathcal{O}(M^3)$$
  Per-token latency grows quadratically with sequence length!

- **With KV-Cache:**
  For each step $t$, query length is 1, attending to key length $t$:
  $$\text{Total Compute} \propto \sum_{t=1}^M t \approx \frac{M^2}{2} \implies \mathcal{O}(M^2)$$
  Per-token latency scales linearly $\mathcal{O}(M)$ with sequence length, delivering dramatic real-world speedups.

---

## 5. Summary & Key Takeaways

1. **Autoregressive Loop is Sequential:** Next-token generation is fundamentally an iterative loop feeding outputs back to inputs.
2. **Nucleus Sampling Balances Quality & Creativity:** Combining Temperature ($\tau \approx 0.7-0.8$) and Top-$p$ ($p \approx 0.9$) produces natural, coherent text without greedy repetition.
3. **KV-Cache Eliminates Redundant Projections:** Caching historical $K$ and $V$ tensors transforms quadratic recomputation into fast, constant-query attention.
