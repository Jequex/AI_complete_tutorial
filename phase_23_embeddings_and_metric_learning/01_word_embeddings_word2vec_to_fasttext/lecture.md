# Topic 23.1: Word Embeddings — From Word2Vec (CBOW & Skip-Gram) to FastText

---

## 1. Executive Summary & The 12-Year-Old Analogy

### The Fundamental Problem of Symbolic Words
In early Natural Language Processing, words were treated as discrete, atomic, orthogonal symbols. In a vocabulary $\mathcal{V}$ of size $|\mathcal{V}| = V$, each word $w_i$ was represented as a standard one-hot basis vector $\mathbf{e}_i \in \{0, 1\}^V$:
$$\mathbf{e}_{\text{cat}} = [1, 0, 0, \dots, 0]^\top, \quad \mathbf{e}_{\text{dog}} = [0, 1, 0, \dots, 0]^\top, \quad \mathbf{e}_{\text{apple}} = [0, 0, 1, \dots, 0]^\top$$

Under this orthogonal representation, the dot product between any two distinct words is strictly zero:
$$\mathbf{e}_{\text{cat}}^\top \mathbf{e}_{\text{dog}} = 0, \quad \mathbf{e}_{\text{cat}}^\top \mathbf{e}_{\text{apple}} = 0$$

The Euclidean distance between all distinct pairs is identically $\sqrt{2}$:
$$\|\mathbf{e}_{\text{cat}} - \mathbf{e}_{\text{dog}}\|_2 = \sqrt{(1)^2 + (-1)^2} = \sqrt{2}$$

This discrete representation suffers from two fatal flaws:
1. **The Semantic Isolation Flaw:** The geometric distance between synonyms or related concepts (`cat` and `dog`) is identical to the distance between completely unrelated concepts (`cat` and `refrigerator`). The model cannot share statistical strength across semantically related words.
2. **The Curse of Dimensionality & Sparsity:** Real-world vocabularies span $10^5$ to $10^7$ words. Maintaining sparse vectors of dimensionality $10^6$ incurs prohibitive memory footprints and combinatorial parameter explosion in downstream layers.

The modern paradigm of representation learning replaces sparse orthogonal indicators with **dense continuous embeddings**: mapping each discrete word into a compact vector space $\mathbb{R}^d$ where $d \ll V$ (typically $d \in [50, 300]$), such that geometric proximity in $\mathbb{R}^d$ encodes semantic and syntactic relationships.

```
+----------------------------------------------------------------------------------------------------+
|                               THE REPRESENTATION LEAP                                              |
|                                                                                                    |
|  Symbolic One-Hot (Dimension V ~ 1,000,000):                                                       |
|  cat   = [1, 0, 0, 0, 0, 0, 0, 0, ..., 0]  ──┐                                                    |
|  feline= [0, 1, 0, 0, 0, 0, 0, 0, ..., 0]  ──┼──> Dot Product = 0.0 (Zero semantic knowledge)     |
|  table = [0, 0, 0, 0, 1, 0, 0, 0, ..., 0]  ──┘                                                    |
|                                                                                                    |
|  Dense Distributed Embeddings (Dimension d ~ 100 - 300):                                           |
|  cat   = [ 0.42, -0.19,  0.88, ...,  0.05] ──┐                                                    |
|  feline= [ 0.40, -0.18,  0.85, ...,  0.07] ──┼──> Cosine Sim = 0.98 (Captures synonymy)          |
|  table = [-0.62,  0.55, -0.12, ..., -0.40] ──┘──> Cosine Sim = -0.15 (Captures distinct domain)   |
+----------------------------------------------------------------------------------------------------+
```

---

### The 12-Year-Old Analogy: The Map of Word Neighborhoods & The Lego Subwords

Imagine every word in the English language is looking for a place to build a house in a massive 3D fantasy world:

1. **Firth's Law of Good Neighbors:**
   - How do you decide who lives where? A wise linguist named John Rupert Firth declared in 1957: *"You shall know a word by the company it keeps."*
   - Words that hang out in the same places build their houses on the same block. Because the words `cat` and `dog` both hang around with words like `leash`, `vet`, `pet`, `fur`, `purr`, and `bark`, they end up building their houses right next to each other on "Pet Boulevard."
   - The word `apple` never hangs around with `leash` or `vet`; it hangs around with `pie`, `crisp`, `sweet`, and `orchard`, so its house is built miles away in "Fruit Valley."

2. **The City Compass (Vector Arithmetic):**
   - Because the city is built based on relationships, walking between neighborhoods follows consistent compass directions!
   - If you start at the house for `Man` and walk across the river to the house for `King`, you just walked along the "Royalty Avenue."
   - If you now go to the house for `Woman` and walk in the exact same compass direction (adding the Royalty vector), you land right at the doorstep of `Queen`!
   - Math works like GPS directions on word meanings: $\mathbf{v}_{\text{King}} - \mathbf{v}_{\text{Man}} + \mathbf{v}_{\text{Woman}} \approx \mathbf{v}_{\text{Queen}}$.

3. **FastText and the Lego Bricks:**
   - In Word2Vec, every house was carved out of a single monolithic boulder. If an alien word showed up—like a brand-new slang term `un-dog-like` or a misspelling like `computerr`—the city guards said: *"We don't recognize this whole boulder! You are `<UNK>` (Unknown) and get banished!"*
   - FastText (Bojanowski et al., 2016) fixes this by building every house out of tiny **character Lego bricks** (`<un`, `ndo`, `dog`, `gli`, `ike>`).
   - Even if the city has never seen the full word `computerr`, it immediately spots the Lego bricks `comp`, `ompu`, `mput`, and `uter`. It immediately guides the new word to live on "Technology Boulevard" right next to `computer`!

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $\mathcal{V}$ | Discrete Set | Word vocabulary of size $|\mathcal{V}| = V$. |
| $d$ | $\mathbb{N}^+$ | Latent embedding dimensionality ($d \ll V$, typically $100 \le d \le 300$). |
| $w_t \in \mathcal{V}$ | Categorical | The central target word at sequence position $t$. |
| $w_c \in \mathcal{V}$ | Categorical | Context word within window $c \in [t - C, t + C] \setminus \{t\}$. |
| $C \in \mathbb{N}^+$ | Integer Hyperparameter | Context window radius (half-width). |
| $\mathbf{W} \in \mathbb{R}^{V \times d}$ | Matrix | Center word input embedding weight matrix; row $i$ is vector $\mathbf{v}_{w_i}$. |
| $\mathbf{W}' \in \mathbb{R}^{V \times d}$ | Matrix | Context word output embedding weight matrix; row $j$ is vector $\mathbf{v}_{w_j}'$. |
| $\mathbf{v}_w \in \mathbb{R}^d$ | Vector | Center word representation vector for word $w$. |
| $\mathbf{v}_w' \in \mathbb{R}^d$ | Vector | Context representation vector for word $w$. |
| $\sigma(z) = \frac{1}{1 + e^{-z}}$ | Function: $\mathbb{R} \to (0, 1)$ | Standard sigmoid logistic activation function. |
| $k \in \mathbb{N}^+$ | Integer Hyperparameter | Number of negative samples drawn per positive training pair. |
| $P_n(w)$ | Probability Distribution | Noise distribution over vocabulary $\mathcal{V}$, specifically unigram distribution $U(w)^{3/4}$. |
| $\mathcal{D}, \mathcal{D}_{\text{neg}}$ | Sample Sets | Sets of observed positive word-context pairs and drawn negative noise pairs. |
| $\mathcal{G}_w$ | Set of Strings | Bag of character $n$-grams (including boundary tags) comprising word $w$. |
| $\mathbf{z}_g \in \mathbb{R}^d$ | Vector | Subword embedding vector corresponding to character $n$-gram $g$. |
| $\text{sim}(\mathbf{u}, \mathbf{v})$ | $\mathbb{R} \times \mathbb{R} \to [-1, 1]$ | Cosine similarity: $\frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$. |

---

## 3. The Distributional Hypothesis & Word2Vec Architectures

### 3.1 The Distributional Hypothesis (Firth 1957)
The mathematical foundation of all semantic vector spaces is the **Distributional Hypothesis**, formalized by linguist J.R. Firth (1957) and Zellig Harris (1954):
> *"The statistical distribution of words in context specifies their semantic content. Words that occur in similar contexts tend to have similar meanings."*

If words $w_a$ and $w_b$ frequently co-occur with context words $\{c_1, c_2, \dots, c_m\}$, the model must map $\mathbf{v}_{w_a}$ and $\mathbf{v}_{w_b}$ to adjacent locations in $\mathbb{R}^d$ so that both maximize the inner product with the context vectors $\mathbf{v}_{c_j}'$.

### 3.2 Word2Vec: CBOW vs. Continuous Skip-Gram
In 2013, Mikolov et al. at Google introduced the **Word2Vec** family of shallow, two-layer neural architectures that made training continuous word embeddings computationally feasible on billions of tokens:

```
          CONTINUOUS BAG-OF-WORDS (CBOW)                      CONTINUOUS SKIP-GRAM
          
     Context Words        Target Word                     Target Word        Context Words
     
      w(t-2) ──┐                                                        ┌──> w(t-2)
      w(t-1) ──┼──> [ SUM / AVG ] ──> w(t)                     w(t) ────┼──> w(t-1)
      w(t+1) ──┤                                                        ├──> w(t+1)
      w(t+2) ──┘                                                        └──> w(t+2)
      
    Objective: Predict w(t) given context             Objective: Predict context given w(t)
    Characteristic: Smooths over context;             Characteristic: Treats each pair independently;
    faster; great for frequent words.                 superior for infrequent words & nuances.
```

1. **Continuous Bag-of-Words (CBOW):**
   - Given a context window of size $C$, CBOW takes the sum or average of the context embeddings:
     $$\mathbf{h} = \frac{1}{2C} \sum_{-C \le j \le C, j \ne 0} \mathbf{v}_{w_{t+j}}$$
   - The model predicts the probability of the central word $w_t$ given $\mathbf{h}$.
   - Because context vectors are summed, word order is discarded (hence "Bag-of-Words"). This averages out noise, making CBOW faster to train and effective for frequent words.

2. **Continuous Skip-Gram:**
   - Inverts the CBOW formulation: given the single center target word $w_t$, the model predicts each individual context word $w_{t+j}$ for $-C \le j \le C, j \ne 0$:
     $$\arg \max_{\theta} \sum_{t=1}^T \sum_{-C \le j \le C, j \ne 0} \log P(w_{t+j} \mid w_t)$$
   - Each context word receives a separate gradient update from $w_t$. Because rare words are not averaged away into a context pool, Skip-Gram preserves fine-grained semantic representations for rare and low-frequency words.

---

## 4. The Softmax Bottleneck & Negative Sampling (SGNS)

### 4.1 The Full Softmax Bottleneck
Under the standard multinomial logistic regression (categorical distribution), the probability of observing context word $w_O$ given center word $w_I$ is defined by the full softmax:
$$P(w_O \mid w_I) = \frac{\exp(\mathbf{v}_{w_O}'^\top \mathbf{v}_{w_I})}{\sum_{w \in \mathcal{V}} \exp(\mathbf{v}_{w}'^\top \mathbf{v}_{w_I})}$$

The log-likelihood for a single training pair $(w_I, w_O)$ is:
$$\log P(w_O \mid w_I) = \mathbf{v}_{w_O}'^\top \mathbf{v}_{w_I} - \log \sum_{w \in \mathcal{V}} \exp(\mathbf{v}_{w}'^\top \mathbf{v}_{w_I})$$

Computing the gradient requires evaluating the denominator sum over all $V$ words:
$$\nabla_{\mathbf{v}_{w_I}} \log P(w_O \mid w_I) = \mathbf{v}_{w_O}' - \sum_{w \in \mathcal{V}} P(w \mid w_I) \mathbf{v}_w'$$

When $V = 1,000,000$ and $d = 300$, calculating $\sum_{w \in \mathcal{V}} \exp(\mathbf{v}_w'^\top \mathbf{v}_{w_I})$ requires $3 \times 10^8$ floating-point multiplications **per single token prediction**. Across a multi-billion token corpus, this full softmax computation is practically impossible.

### 4.2 Skip-Gram with Negative Sampling (SGNS)
To bypass the intractable partition function $\sum_{w \in \mathcal{V}} \exp(\cdot)$, Mikolov et al. (2013) adapted Noise Contrastive Estimation (NCE) into **Negative Sampling (SGNS)**.

Instead of predicting a probability distribution over the entire vocabulary $\mathcal{V}$, SGNS recasts the problem into a set of **independent binary logistic regressions**:
- Is the pair $(w_I, w_O)$ a true observed co-occurrence from text ($\text{Label} = 1$)?
- Or is $(w_I, w_i)$ a fabricated noise pair drawn at random ($\text{Label} = 0$)?

For a target word $w_I$ and positive context word $w_O$, we draw $k$ negative samples $\{w_1, w_2, \dots, w_k\}$ from a noise distribution $P_n(w)$. The objective to **maximize** for this instance is:
$$\mathcal{L}_{\text{SGNS}}(\mathbf{v}_{w_I}, \mathbf{v}_{w_O}', \{\mathbf{v}_{w_i}'\}_{i=1}^k) = \log \sigma(\mathbf{v}_{w_O}'^\top \mathbf{v}_{w_I}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-\mathbf{v}_{w_i}'^\top \mathbf{v}_{w_I}) \right]$$

Since $\sigma(-z) = 1 - \sigma(z)$, this is equivalent to:
$$\mathcal{L}_{\text{SGNS}} = \log \sigma(\mathbf{v}_{w_O}'^\top \mathbf{v}_{w_I}) + \sum_{i=1}^k \log \left( 1 - \sigma(\mathbf{v}_{w_i}'^\top \mathbf{v}_{w_I}) \right)$$

When training by gradient descent, we minimize the negative log-likelihood:
$$\mathcal{J}_{\text{SGNS}} = -\log \sigma(\mathbf{v}_{w_O}'^\top \mathbf{v}_{w_I}) - \sum_{i=1}^k \log \sigma(-\mathbf{v}_{w_i}'^\top \mathbf{v}_{w_I})$$

### 4.3 Rigorous Analytical Gradient Derivation
Let $z_O = \mathbf{v}_{w_O}'^\top \mathbf{v}_{w_I}$ and $z_i = \mathbf{v}_{w_i}'^\top \mathbf{v}_{w_I}$.
Recall the fundamental derivative identity for the logistic sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$:
$$\frac{d}{dz} \log \sigma(z) = \frac{\sigma'(z)}{\sigma(z)} = \frac{\sigma(z)(1 - \sigma(z))}{\sigma(z)} = 1 - \sigma(z)$$
Similarly, for $\log \sigma(-z)$:
$$\frac{d}{dz} \log \sigma(-z) = \frac{-\sigma(-z)(1 - \sigma(-z))}{\sigma(-z)} = -(1 - \sigma(-z)) = -\sigma(z)$$

Now, take the partial derivatives of the objective $\mathcal{L}_{\text{SGNS}}$ with respect to each vector parameter:

1. **Gradient with respect to Context Vector $\mathbf{v}_{w_O}'$ (Positive Sample):**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_{w_O}'} = \frac{\partial \log \sigma(z_O)}{\partial z_O} \frac{\partial z_O}{\partial \mathbf{v}_{w_O}'} = (1 - \sigma(\mathbf{v}_{w_O}'^\top \mathbf{v}_{w_I})) \mathbf{v}_{w_I}$$
   *Interpretation:* The context vector $\mathbf{v}_{w_O}'$ is pulled in the direction of the input vector $\mathbf{v}_{w_I}$, scaled by how surprised the model was ($1 - \sigma$).

2. **Gradient with respect to Negative Sample Context Vectors $\mathbf{v}_{w_i}'$:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_{w_i}'} = \frac{\partial \log \sigma(-z_i)}{\partial z_i} \frac{\partial z_i}{\partial \mathbf{v}_{w_i}'} = -\sigma(\mathbf{v}_{w_i}'^\top \mathbf{v}_{w_I}) \mathbf{v}_{w_I}$$
   *Interpretation:* The negative context vector $\mathbf{v}_{w_i}'$ is pushed directly away from $\mathbf{v}_{w_I}$, scaled by the false positive prediction $\sigma(z_i)$.

3. **Gradient with respect to Center Word Input Vector $\mathbf{v}_{w_I}$:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_{w_I}} = (1 - \sigma(\mathbf{v}_{w_O}'^\top \mathbf{v}_{w_I})) \mathbf{v}_{w_O}' - \sum_{i=1}^k \sigma(\mathbf{v}_{w_i}'^\top \mathbf{v}_{w_I}) \mathbf{v}_{w_i}'$$

**Computational Complexity Transformation:**
- Full Softmax: $\mathcal{O}(V \cdot d)$ operations per word update.
- Negative Sampling: $\mathcal{O}((k + 1) \cdot d)$ operations per word update.
For $V = 10^6, k = 5, d = 300$, negative sampling achieves a **$166,666\times$ speedup per gradient step**!

### 4.4 The 3/4-Power Unigram Noise Distribution
How should negative samples be drawn? If drawn strictly proportional to raw corpus frequencies $U(w) = \frac{\text{count}(w)}{\sum_{w'} \text{count}(w')}$, common stopwords (`the`, `of`, `and`, `is`) would constitute 80% of all negative samples, leaving obscure content words rarely sampled as negatives.

Conversely, if drawn uniformly $P_n(w) = \frac{1}{V}$, rare typos and medical jargon would dominate negative samples.

Mikolov et al. empirically identified an optimal smoothed unigram distribution:
$$P_n(w) = \frac{U(w)^{3/4}}{\sum_{w' \in \mathcal{V}} U(w')^{3/4}}$$

Raising frequency to the power of $0.75$ boosts the relative sampling probability of rare words while dampening extreme stopword dominance:
- If Word A has frequency $0.9$ and Word B has frequency $0.00001$:
  - Ratio $U(A) / U(B) = 90,000$.
  - Under exponent $0.75$: $(0.9)^{0.75} \approx 0.924$, $(0.00001)^{0.75} \approx 0.0001778$.
  - Ratio $(U(A)/U(B))^{0.75} \approx 5,196$ (a $17\times$ boost in relative exposure for rare words!).

---

## 5. Vector Arithmetic & Semantic Geometry

### 5.1 Linear Substructures & Parallelograms
One of the most profound emergent properties of word representations trained via SGNS is that linear relationships between word vectors capture intuitive semantic and syntactic analogies:
$$\mathbf{v}_{\text{King}} - \mathbf{v}_{\text{Man}} + \mathbf{v}_{\text{Woman}} \approx \mathbf{v}_{\text{Queen}}$$
$$\mathbf{v}_{\text{Paris}} - \mathbf{v}_{\text{France}} + \mathbf{v}_{\text{Italy}} \approx \mathbf{v}_{\text{Rome}}$$
$$\mathbf{v}_{\text{Walking}} - \mathbf{v}_{\text{Walk}} + \mathbf{v}_{\text{Swim}} \approx \mathbf{v}_{\text{Swimming}}$$

```
                      Semantic Vector Geometry
                      
             [King] ──────────────────────────> [Queen]
               │         Royalty Direction        │
               │                                  │
  Gender       │                                  │  Gender
  Direction    │                                  │  Direction
  (v_Woman -   │                                  │  (v_Woman -
   v_Man)      ▼                                  ▼   v_Man)
             [Man]  ──────────────────────────> [Woman]
                         Royalty Direction
```

To solve an analogy query $a : b :: c : ?$, we compute:
$$\hat{d} = \arg\max_{w \in \mathcal{V} \setminus \{a, b, c\}} \cos(\mathbf{v}_w, \mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c)$$

### 5.2 Theoretical Justification: Implicit Matrix Factorization (Levy & Goldberg, 2014)
Why do simple inner products and vector differences capture semantic relationships?
In 2014, Omer Levy and Yoav Goldberg proved that **Word2Vec with Skip-Gram and Negative Sampling is implicitly factorizing a shifted Pointwise Mutual Information (PMI) matrix**.

Recall the definition of Pointwise Mutual Information between word $w$ and context $c$:
$$\text{PMI}(w, c) = \log \frac{P(w, c)}{P(w) P(c)}$$

Levy & Goldberg proved that if the embedding dimensionality $d$ is sufficiently large, the optimal inner product $\mathbf{v}_w^\top \mathbf{v}_c'$ that minimizes the global SGNS loss satisfies:
$$\mathbf{v}_w^\top \mathbf{v}_c' = \text{PMI}(w, c) - \log k$$
where $k$ is the number of negative samples.

Because the PMI matrix measures the log-odds ratio of words appearing together compared to independent chance, relationships between concepts correspond to linear differences between their log-probability profiles, naturally yielding the linear parallelogram geometry observed in the embedding space!

---

## 6. FastText: Subword n-Grams & Morphological Generalization

### 6.1 Limitations of Classical Word2Vec
Despite its elegance, standard Word2Vec suffers from two severe limitations:
1. **The Out-of-Vocabulary (OOV) Catastrophe:** Every word is treated as an indivisible atomic integer index. If a word was not seen during training, its vector is completely undefined. The model must assign `<UNK>`, losing all semantic content.
2. **Ignorance of Internal Word Morphology:** The English words `book`, `books`, and `bookkeeper` receive completely independent parameter rows in $\mathbf{W}$. In morphologically rich languages (e.g., German, Turkish, Arabic, Finnish), where words are inflected with cases, tenses, and agglutinative affixes, thousands of distinct token variations appear only a handful of times, preventing Word2Vec from learning reliable representations.

### 6.2 FastText Formulation (Bojanowski et al., 2016)
FastText solves this by modeling each word as a **bag of character $n$-grams**.
For a given word $w$:
1. Enclose the word with boundary symbols `<` and `>` to distinguish subwords from whole words (e.g., `<her>` vs subword `her`).
2. Extract all character $n$-grams of lengths $n \in [n_{\min}, n_{\max}]$ (typically $n \in [3, 6]$).
3. Append the special whole-word token `<w>`.

**Example:** Word `where` with $n = 3$:
$$\mathcal{G}_{\text{where}} = \{ \text{<wh}, \text{whe}, \text{her}, \text{ere}, \text{re>}, \text{<where>} \}$$

Each $n$-gram $g$ is assigned its own vector $\mathbf{z}_g \in \mathbb{R}^d$. The final representation of word $w$ is the **sum of its constituent subword vectors**:
$$\mathbf{u}_w = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g$$

### 6.3 Scoring and OOV Generalization
The scoring function between target word $w$ and context word $c$ becomes:
$$s(w, c) = \mathbf{u}_w^\top \mathbf{v}_c' = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g^\top \mathbf{v}_c'$$

**How FastText solves OOV and Typos:**
- When an unseen word like `unprecedentedly` or a misspelled word like `elpehant` arrives at inference time:
- FastText does not output `<UNK>`. Instead, it extracts the subword $n$-grams:
  $$\mathcal{G}_{\text{elpehant}} = \{ \text{<el}, \text{elp}, \text{lpe}, \text{peh}, \text{eha}, \text{han}, \text{ant}, \text{nt>}, \dots \}$$
- Because $n$-grams like `ant`, `han`, `eha`, and `<el` were trained across hundreds of related words (including `elephant`), the summed vector $\sum_{g} \mathbf{z}_g$ lands directly adjacent to the true concept vector!

---

## 7. Architectural Comparison: Word2Vec vs. GloVe vs. FastText

| Dimension | Word2Vec (SGNS) | GloVe (Pennington et al. 2014) | FastText (Bojanowski et al. 2016) |
| :--- | :--- | :--- | :--- |
| **Learning Paradigm** | Local sliding-window negative sampling | Global word co-occurrence matrix factorization | Local sliding-window with subword $n$-grams |
| **Objective Function** | Binary logistic cross-entropy over pairs | Weighted least squares: $f(X_{ij})(\mathbf{w}_i^\top \tilde{\mathbf{w}}_j - \log X_{ij})^2$ | Binary logistic cross-entropy over subword sums |
| **Subword Modeling** | None (atomic word tokens) | None (atomic word tokens) | Character $n$-grams ($n \in [3, 6]$) |
| **Out-of-Vocabulary (OOV)** | Returns `<UNK>` (fails completely) | Returns `<UNK>` (fails completely) | Synthesizes embedding by summing constituent $n$-grams |
| **Morphological Richness** | Weak (treats prefixes/suffixes independently) | Weak (treats prefixes/suffixes independently) | Exceptional (shares subword parameters across languages) |
| **Typo Resilience** | Zero | Zero | High (shared character $n$-gram overlap) |
| **Training Speed** | Fast ($\mathcal{O}(k \cdot d)$ per token) | Moderate (requires full co-occurrence matrix) | Moderate to Fast (hash collision table over $n$-grams) |

---

## 8. Summary & Transition to Deep Sentence Representations

Word2Vec and FastText revolutionized natural language processing by providing dense, continuous representations that satisfy the Distributional Hypothesis and support vector arithmetic.

However, word embeddings possess a critical limitation: **Context Invariance**.
In Word2Vec and FastText, the word `bank` has **exactly one static vector**, whether it appears in:
- *"The fisherman sat on the river **bank**."* (geographical landform)
- *"I deposited money into my savings **bank**."* (financial institution)

In Topic 23.2, we resolve this limitation by studying deep sentence representations, uncovering why raw average pooling of contextualized BERT representations suffers from the **Anisotropy Cone Problem**, and how **Sentence-BERT (SBERT)** restores an isotropic semantic hypersphere.
