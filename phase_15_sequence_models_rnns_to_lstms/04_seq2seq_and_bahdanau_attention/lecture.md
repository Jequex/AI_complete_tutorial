# Topic 15.4: Sequence-to-Sequence (Seq2Seq) & Bahdanau Additive Attention

---

## 1. Executive Summary & The 12-Year-Old Analogy

Prior to 2014, recurrent neural networks were constrained to tasks where input and output sequences had identical length (such as character-level text generation or part-of-speech tagging), or where an entire sequence was mapped to a single static label (sequence classification).

In 2014, Sutskever, Vinyals, & Le introduced the **Sequence-to-Sequence (Seq2Seq)** framework. By separating the network into an **Encoder** and a **Decoder**, Seq2Seq enabled neural networks to map variable-length input sequences $(x_1, \dots, x_{T_x})$ to variable-length output sequences $(y_1, \dots, y_{T_y})$, revolutionizing Machine Translation, Text Summarization, and Speech Recognition.

However, the vanilla Seq2Seq architecture suffered from a crippling theoretical flaw: **The Information Bottleneck**. The entire source sentence—whether 5 words or 80 words long—had to be compressed into a single, fixed-size vector $c = h_{T_x}$. For sentences longer than 20 words, translation quality collapsed.

In 1994 / 2014-2015, Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio introduced **Additive Attention**. Rather than forcing the encoder to compress everything into one fixed vector, the decoder is given direct access to **all intermediate encoder annotations $\{h_1, \dots, h_{T_x}\}$**. At each decoding step, the model computes a dynamic probability distribution $\alpha_{ij}$—a **content-based spotlight**—and constructs an individualized dynamic context vector $c_i = \sum_j \alpha_{ij} h_j$.

```
                    BAHDANAU ADDITIVE ATTENTION ARCHITECTURE:
                    
Decoder Target:                   "le"                           "chat"
                                    ▲                              ▲
                                    │                              │
Decoder Hidden:                   [s_1]                          [s_2]
                                    ▲                              ▲
Context Vector:                    c_1 ───────────┐               c_2
                                    ▲             │                ▲
Attention Weights:   [alpha_11, alpha_12, alpha_13]│ [alpha_21, alpha_22, alpha_23]
                               │    │    │        │          │    │    │
                               ▼    ▼    ▼        │          ▼    ▼    ▼
Encoder Annotations:         [h_1] [h_2] [h_3]    │        [h_1] [h_2] [h_3]
                               ▲     ▲     ▲      │          ▲     ▲     ▲
Encoder Inputs:              "the" "black" "cat"  └────────►"the" "black" "cat"
```

---

### The 12-Year-Old Analogy: The Simultaneous Translator with a Magic Spotlight

Imagine you are hired as a simultaneous translator at the United Nations:

**The Old Vanilla Seq2Seq Way (The Memorization Test):**
A French diplomat speaks a 40-word paragraph. Your boss says: *"You must listen to all 40 words with your eyes closed, write down nothing, memorize the entire paragraph into one single 5-digit number in your head, and then recite the full paragraph in fluent English from that number alone!"*
- If the diplomat says *"Hello"*, you can memorize it easily.
- If the diplomat delivers a 50-word legal treaty on nuclear energy, you forget the names, dates, and specifics from the beginning of the sentence by the time you reach the end! This is the **Information Bottleneck**.

**The Bahdanau Attention Way (The Magic Spotlight):**
Instead of memorizing, the French text is written out on a long whiteboard in front of you:
- When you are about to speak your first English word (*"The"*), you hold a **Magic Spotlight** ($e_{ij}$).
- You point the spotlight at the whiteboard. The word *"Le"* glows with 95% brightness ($\alpha_{i1} = 0.95$), while the rest of the board is dark. You translate *"The"*.
- When you are about to translate *"cat"*, you swing the spotlight over to *"chat"*, which now glows brightly!
- You never have to cram an entire speech into a tiny memory bottleneck. You simply glance back at whatever word you need at the exact moment you need it!

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Dimensions |
| :--- | :--- | :--- |
| $T_x \in \mathbb{N}$ | Sequence Length | Temporal length of source sequence |
| $T_y \in \mathbb{N}$ | Sequence Length | Temporal length of target sequence |
| $x_1, \dots, x_{T_x}$ | Source Tokens | Input token sequence ($x_j \in \{1, \dots, V_{\text{src}}\}$) |
| $h_j \in \mathbb{R}^{d_h}$ | Encoder State | Annotation vector for source token $j$: $h_j = [\vec{h}_j; \overleftarrow{h}_j]$ |
| $s_i \in \mathbb{R}^{d_s}$ | Decoder State | Hidden state of decoder at generation step $i$ |
| $e_{ij} \in \mathbb{R}$ | Scalar Energy | Alignment score measuring relevance of source token $j$ to decoder state $s_{i-1}$ |
| $\alpha_{ij} \in [0, 1]$ | Attention Weight | Normalized probability that target token $i$ aligns with source token $j$ |
| $c_i \in \mathbb{R}^{d_h}$ | Context Vector | Dynamic context vector for decoder step $i$: $\sum_j \alpha_{ij} h_j$ |
| $W_a \in \mathbb{R}^{d_a \times d_s}$ | Attention Weight | Linear projection of decoder query state $s_{i-1}$ |
| $U_a \in \mathbb{R}^{d_a \times d_h}$ | Attention Weight | Linear projection of encoder key annotation $h_j$ |
| $v_a \in \mathbb{R}^{d_a}$ | Attention Vector | Scoring vector mapping alignment features to scalar energy |

---

## 3. The Vanilla Seq2Seq Model & The Information Bottleneck

In Sutskever et al. (2014), the encoder is a recurrent network (LSTM) that updates:
$$h_t = \text{EncoderRNN}(x_t, h_{t-1})$$

The final hidden state $h_{T_x}$ is chosen as the **global context vector**:
$$c = h_{T_x}$$

The decoder generates the target sequence autoregressively:
$$s_i = \text{DecoderRNN}([y_{i-1}; c], s_{i-1})$$
$$P(y_i \mid y_{<i}, x) = \text{softmax}(W_o s_i + b_o)$$

### The Information Bottleneck
A fixed vector $c \in \mathbb{R}^{d}$ has finite Shannon channel capacity:
$$\mathcal{C} = d \times B \text{ bits}$$
where $B$ is the bit precision (e.g. 32 bits for float32).
As the source sentence length $T_x$ increases:
- The amount of semantic, grammatical, and relational information in the source scales as $\mathcal{O}(T_x)$.
- But the channel capacity is strictly $\mathcal{O}(1)$.
Empirically, BLEU scores for vanilla Seq2Seq models dropped steeply from $>30$ for sentences under 15 words to $<10$ for sentences over 40 words.

---

## 4. Bahdanau Additive Attention

Bahdanau et al. eliminated this bottleneck by treating the encoder hidden states as an **external content-addressable memory**:
$$\mathcal{M} = \{h_1, h_2, \dots, h_{T_x}\}$$

At each decoder step $i \in \{1, \dots, T_y\}$:

### 4.1 Alignment Energy (Additive MLP)
The alignment score $e_{ij}$ between decoder query $s_{i-1}$ and encoder key $h_j$ is computed by a single-layer feedforward network with $\tanh$:
$$\boxed{e_{ij} = v_a^T \tanh(W_a s_{i-1} + U_a h_j + b_a)}$$

### 4.2 Attention Distribution (Softmax)
The scores are normalized across all source positions $j \in \{1, \dots, T_x\}$ using a softmax operator:
$$\boxed{\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}}$$

Properties of $\alpha_i$:
1. **Valid Probability Simplex:** $\sum_{j=1}^{T_x} \alpha_{ij} = 1.0$ and $\alpha_{ij} \ge 0$.
2. **Differentiable (Soft Attention):** Because softmax is smooth and differentiable everywhere, the entire attention mechanism can be trained end-to-end via standard backpropagation!

### 4.3 Dynamic Context Vector
The context vector $c_i$ is the expected value of encoder annotations under distribution $\alpha_i$:
$$\boxed{c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j}$$

### 4.4 Decoder State & Emission
The decoder updates its hidden state conditioned on previous target token $y_{i-1}$ and dynamic context $c_i$:
$$s_i = f(s_{i-1}, y_{i-1}, c_i)$$
The emission probability distribution over vocabulary $V$ is:
$$P(y_i \mid y_{<i}, x) = \text{softmax}(W_o [s_i; c_i; y_{i-1}] + b_o)$$

---

## 5. Visualizing Attention Alignment Matrices

A revolutionary byproduct of attention is **interpretability**. By plotting the $T_y \times T_x$ matrix $A = [\alpha_{ij}]$, we obtain an alignment heatmap displaying which source words the model focused on when emitting each target word.

For example, when translating French to English:
- Direct word order (*"l'accord sur la zone economique"* $\to$ *"the agreement on the economic zone"*) produces a strict diagonal path.
- Inverted noun-adjective phrases (*"zone economique"* $\to$ *"economic zone"*) produce a local off-diagonal flip! The model attends to *"economique"* first, then to *"zone"*.

```
TARGET       "economic"      "zone"
SOURCE:
"zone"          0.05          0.92   ◄── Flip!
"economique"    0.91          0.06
```

---

## 6. Numerical Verification & Summary

In `experiment.py`, we implemented a complete Seq2Seq model with Bahdanau Attention from scratch in pure Python:
1. **Sum Normalization Check:** Verified across all decoder steps that $\sum_{j=1}^{T_x} \alpha_{ij} = 1.0000000000$ with absolute error $< 10^{-15}$.
2. **Reversal Alignment Matrix:** Generated the full attention matrix $\alpha_{ij}$ and verified dynamic context vector construction.
3. **Bottleneck Analysis:** Characterized information bottleneck scaling, demonstrating that attention scales memory as $\mathcal{O}(T_x)$, bypassing the fixed $\mathcal{O}(1)$ bottleneck.
