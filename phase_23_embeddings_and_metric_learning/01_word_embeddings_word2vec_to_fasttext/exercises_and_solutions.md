# Topic 23.1: Exercises & Solutions — Word Embeddings (Word2Vec to FastText)

---

### Problem 1: Softmax vs. Negative Sampling Computational Complexity (Easy)

Consider an NLP corpus with a vocabulary size of $|\mathcal{V}| = 500,000$ unique tokens. We train an embedding space of dimensionality $d = 300$ using Continuous Skip-Gram.
1. In a single update step for target word $w_I$ and context word $w_O$ using the standard full softmax:
   $$P(w_O \mid w_I) = \frac{\exp(\mathbf{v}_{w_O}'^\top \mathbf{v}_{w_I})}{\sum_{w=1}^V \exp(\mathbf{v}_w'^\top \mathbf{v}_{w_I})}$$
   calculate the exact number of floating-point multiplication-accumulation operations required to compute the denominator and the full gradient $\nabla_{\mathbf{v}_{w_I}} \log P(w_O \mid w_I)$.
2. Calculate the corresponding floating-point operations if we instead use Skip-Gram with Negative Sampling (SGNS) with $k = 5$ negative samples.
3. Compute the speedup factor achieved by SGNS for this single update step.

<details>
<summary>View Complete Solution</summary>

**Step 1: Full Softmax Computational Load:**
- To evaluate the partition function $\sum_{w=1}^V \exp(\mathbf{v}_w'^\top \mathbf{v}_{w_I})$:
  - We must take the dot product of $\mathbf{v}_{w_I} \in \mathbb{R}^{300}$ with all $V = 500,000$ context vectors $\mathbf{v}_w' \in \mathbb{R}^{300}$.
  - Each dot product takes $d = 300$ multiplications and $299$ additions.
  - Total dot product operations: $V \cdot d = 500,000 \times 300 = 150,000,000$ operations.
  - Exponentiation and summation: $V$ exponentials and $V-1$ additions: $\approx 1,000,000$ operations.
- To compute the gradient with respect to $\mathbf{v}_{w_I}$:
  $$\nabla_{\mathbf{v}_{w_I}} \log P(w_O \mid w_I) = \mathbf{v}_{w_O}' - \sum_{w=1}^V P(w \mid w_I) \mathbf{v}_w'$$
  - Multiplying each probability scalar $P(w \mid w_I)$ by vector $\mathbf{v}_w'$ and accumulating requires $V \cdot d = 500,000 \times 300 = 150,000,000$ operations.
- Total operations for full softmax: $\approx 300,000,000$ FLOPs ($3 \times 10^8$ FLOPs per word pair).

**Step 2: Negative Sampling Computational Load:**
- Under SGNS with $k = 5$ negative samples, we evaluate:
  - $1$ positive dot product: $\mathbf{v}_{w_O}'^\top \mathbf{v}_{w_I}$ ($300$ FLOPs).
  - $k = 5$ negative dot products: $\mathbf{v}_{w_i}'^\top \mathbf{v}_{w_I}$ ($5 \times 300 = 1500$ FLOPs).
  - $6$ sigmoid evaluations $\sigma(z)$.
  - Gradients:
    - $\nabla_{\mathbf{v}_{w_O}'} = (1 - \sigma(z_O))\mathbf{v}_{w_I}$ ($300$ FLOPs).
    - $\nabla_{\mathbf{v}_{w_i}'} = -\sigma(z_i)\mathbf{v}_{w_I}$ for $i=1,\dots,5$ ($5 \times 300 = 1500$ FLOPs).
    - $\nabla_{\mathbf{v}_{w_I}} = (1 - \sigma(z_O))\mathbf{v}_{w_O}' - \sum_{i=1}^5 \sigma(z_i)\mathbf{v}_{w_i}'$ ($6 \times 300 = 1800$ FLOPs).
- Total operations for SGNS: $\approx (2 \times (k + 1) + (k + 1)) \times d = 3 \times 6 \times 300 = 5,400$ FLOPs.

**Step 3: Speedup Factor:**
$$\text{Speedup} = \frac{300,000,000}{5,400} \approx 55,555.5\times$$
SGNS executes over $55,000\times$ faster per training step while yielding equivalent or superior representation quality.
</details>

---

### Problem 2: Step-by-Step Gradient Descent Update under SGNS (Medium)

Let the embedding dimension be $d = 2$.
Suppose the target word is $w_I = \text{"cat"}$ with current input vector:
$$\mathbf{v}_{\text{cat}} = \begin{bmatrix} 0.5 \\ -0.2 \end{bmatrix}$$
The positive context word is $w_O = \text{"purr"}$ with current context vector:
$$\mathbf{v}_{\text{purr}}' = \begin{bmatrix} 0.8 \\ 0.1 \end{bmatrix}$$
We draw $k = 1$ negative sample $w_1 = \text{"tractor"}$ with current context vector:
$$\mathbf{v}_{\text{tractor}}' = \begin{bmatrix} -0.4 \\ 0.6 \end{bmatrix}$$
Use learning rate $\eta = 0.1$.
Recall $\sigma(z) = \frac{1}{1 + e^{-z}}$.

1. Compute the positive logit $z_O = \mathbf{v}_{\text{purr}}'^\top \mathbf{v}_{\text{cat}}$ and prediction $\sigma(z_O)$.
2. Compute the negative logit $z_1 = \mathbf{v}_{\text{tractor}}'^\top \mathbf{v}_{\text{cat}}$ and prediction $\sigma(z_1)$.
3. Compute the current value of the negative sampling loss $\mathcal{J} = -\log \sigma(z_O) - \log \sigma(-z_1)$.
4. Compute the gradient vector $\nabla_{\mathbf{v}_{\text{cat}}} \mathcal{J}$ and update $\mathbf{v}_{\text{cat}} \leftarrow \mathbf{v}_{\text{cat}} - \eta \nabla_{\mathbf{v}_{\text{cat}}} \mathcal{J}$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Positive Logit and Sigmoid:**
$$z_O = \mathbf{v}_{\text{purr}}'^\top \mathbf{v}_{\text{cat}} = (0.8)(0.5) + (0.1)(-0.2) = 0.40 - 0.02 = 0.38$$
$$\sigma(z_O) = \frac{1}{1 + e^{-0.38}} = \frac{1}{1 + 0.68386} \approx 0.5933$$

**Step 2: Negative Logit and Sigmoid:**
$$z_1 = \mathbf{v}_{\text{tractor}}'^\top \mathbf{v}_{\text{cat}} = (-0.4)(0.5) + (0.6)(-0.2) = -0.20 - 0.12 = -0.32$$
$$\sigma(z_1) = \frac{1}{1 + e^{0.32}} = \frac{1}{1 + 1.37713} \approx 0.4207$$
Note that $\sigma(-z_1) = 1 - \sigma(z_1) \approx 1 - 0.4207 = 0.5793$.

**Step 3: Loss Evaluation:**
$$\mathcal{J} = -\log(0.5933) - \log(0.5793) = -(-0.5220) - (-0.5459) = 0.5220 + 0.5459 = 1.0679$$

**Step 4: Gradient Calculation with respect to $\mathbf{v}_{\text{cat}}$:**
Recall that minimizing $\mathcal{J}$ corresponds to gradient:
$$\nabla_{\mathbf{v}_{\text{cat}}} \mathcal{J} = -(1 - \sigma(z_O)) \mathbf{v}_{\text{purr}}' + \sigma(z_1) \mathbf{v}_{\text{tractor}}'$$
Plugging in values:
$$1 - \sigma(z_O) = 1 - 0.5933 = 0.4067$$
$$\sigma(z_1) = 0.4207$$
$$-(1 - \sigma(z_O))\mathbf{v}_{\text{purr}}' = -0.4067 \begin{bmatrix} 0.8 \\ 0.1 \end{bmatrix} = \begin{bmatrix} -0.32536 \\ -0.04067 \end{bmatrix}$$
$$\sigma(z_1)\mathbf{v}_{\text{tractor}}' = 0.4207 \begin{bmatrix} -0.4 \\ 0.6 \end{bmatrix} = \begin{bmatrix} -0.16828 \\ 0.25242 \end{bmatrix}$$
Summing the components:
$$\nabla_{\mathbf{v}_{\text{cat}}} \mathcal{J} = \begin{bmatrix} -0.32536 - 0.16828 \\ -0.04067 + 0.25242 \end{bmatrix} = \begin{bmatrix} -0.49364 \\ 0.21175 \end{bmatrix}$$

**Step 5: Gradient Descent Parameter Update:**
$$\mathbf{v}_{\text{cat}}^{(\text{new})} = \mathbf{v}_{\text{cat}} - \eta \nabla_{\mathbf{v}_{\text{cat}}} \mathcal{J} = \begin{bmatrix} 0.5 \\ -0.2 \end{bmatrix} - 0.1 \begin{bmatrix} -0.49364 \\ 0.21175 \end{bmatrix} = \begin{bmatrix} 0.5 + 0.04936 \\ -0.2 - 0.02118 \end{bmatrix} = \begin{bmatrix} 0.54936 \\ -0.22118 \end{bmatrix}$$
Notice how the vector was adjusted in direction: it was pulled toward $\mathbf{v}_{\text{purr}}'$ and pushed away from $\mathbf{v}_{\text{tractor}}'$.
</details>

---

### Problem 3: FastText Subword Decomposition & Typo Handling (Medium)

Consider the target vocabulary containing the standard English word `learning`.
Suppose FastText is configured with subword character $n$-grams of length $n = 3$ to $n = 4$ plus the boundary tags `<` and `>`.
1. List all 3-gram and 4-gram subword tokens extracted for the word `learning`.
2. Now suppose a user submits a search query containing the typographical error: `leraing`. Extract all 3-gram subwords for `leraing`.
3. Compute the Jaccard similarity between the 3-gram subword sets of `learning` and `leraing`.
4. Explain why standard Word2Vec completely fails on `leraing` while FastText produces an embedding with high cosine similarity to `learning`.

<details>
<summary>View Complete Solution</summary>

**Step 1: Subwords for `learning`:**
Bounded string: `<learning>`
- Length of bounded string: $10$ characters (`<`, `l`, `e`, `a`, `r`, `n`, `i`, `n`, `g`, `>`).
- 3-grams: `<le`, `lea`, `ear`, `arn`, `rni`, `nin`, `ing`, `ng>` (8 subwords).
- 4-grams: `<lea`, `lear`, `earn`, `arni`, `rnin`, `ning`, `ing>` (7 subwords).
- Special whole-word token: `<learning>`.
- Total subwords: $8 + 7 + 1 = 16$ subwords.

**Step 2: 3-grams for typo `leraing`:**
Bounded string: `<leraing>` (9 characters).
- 3-grams: `<le`, `ler`, `era`, `rai`, `ain`, `ing`, `ng>` (7 subwords).

**Step 3: Jaccard Similarity on 3-grams:**
- $S_1 = \{ \text{<le}, \text{lea}, \text{ear}, \text{arn}, \text{rni}, \text{nin}, \text{ing}, \text{ng>} \}$ (size 8).
- $S_2 = \{ \text{<le}, \text{ler}, \text{era}, \text{rai}, \text{ain}, \text{ing}, \text{ng>} \}$ (size 7).
- Intersection $S_1 \cap S_2$:
  - `<le` (common)
  - `ing` (common)
  - `ng>` (common)
  - Size of intersection = 3.
- Union $S_1 \cup S_2$:
  - Size of union = $|S_1| + |S_2| - |S_1 \cap S_2| = 8 + 7 - 3 = 12$.
- Jaccard similarity:
  $$J(S_1, S_2) = \frac{3}{12} = 0.25$$
Furthermore, if 4-grams and 5-grams are also included, both prefix `<le...` and suffix `...ing>` contribute significant shared subword embedding vectors.

**Step 4: Comparison:**
- **Word2Vec:** Queries dictionary $\mathcal{V}$ for exact string `"leraing"`. Finding no match, it returns token ID $0$ (`<UNK>`). The resulting embedding is either random noise or zero, losing all context.
- **FastText:** Computes $\mathbf{u}_{\text{leraing}} = \sum_{g \in \mathcal{G}_{\text{leraing}}} \mathbf{z}_g$. The subword vectors for `<le`, `ing`, and `ng>` were updated across thousands of verbal and gerund contexts during pretraining. These shared subwords anchor $\mathbf{u}_{\text{leraing}}$ in the same semantic quadrant as `learning`, preserving downstream retrieval accuracy.
</details>

---

### Problem 4: Levy & Goldberg's Implicit Shifted PMI Theorem (Hard)

In 2014, Omer Levy and Yoav Goldberg proved that Skip-Gram with Negative Sampling (SGNS) implicitly factorizes a shifted Pointwise Mutual Information (PMI) matrix.

Let $N(w, c)$ denote the number of times word $w$ and context $c$ co-occur in the corpus.
Let $N(w) = \sum_{c'} N(w, c')$ and $N(c) = \sum_{w'} N(w', c)$, with total tokens $D = \sum_{w, c} N(w, c)$.
The empirical probabilities are $\hat{P}(w, c) = \frac{N(w, c)}{D}$, $\hat{P}(w) = \frac{N(w)}{D}$, $\hat{P}(c) = \frac{N(c)}{D}$.
Negative samples are drawn from distribution $P_n(c) = \hat{P}(c)$.

The global SGNS objective summed over the entire corpus is:
$$\mathcal{L} = \sum_{w \in \mathcal{V}} \sum_{c \in \mathcal{V}} N(w, c) \left[ \log \sigma(\mathbf{v}_w^\top \mathbf{v}_c') + k \cdot \mathbb{E}_{c' \sim P_n} [\log \sigma(-\mathbf{v}_w^\top \mathbf{v}_{c'}')] \right]$$

1. Rewrite the objective as a sum over individual $(w, c)$ pairs in terms of the scalar product $x = \mathbf{v}_w^\top \mathbf{v}_c'$.
2. Taking the derivative with respect to $x$, find the optimal value $x^*$ that maximizes the local objective for pair $(w, c)$.
3. Prove that $x^* = \text{PMI}(w, c) - \log k$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Expressing the objective per pair $(w, c)$:**
Under negative sampling, when word $w$ appears in the corpus, context $c$ is sampled as a negative candidate with probability $P_n(c) = \frac{N(c)}{D}$.
Since each positive co-occurrence draws $k$ negatives, the expected number of times context $c$ is sampled as a negative for word $w$ across the corpus is:
$$N(w) \cdot k \cdot P_n(c) = k \frac{N(w) N(c)}{D}$$

Grouping all terms involving the specific pair $(w, c)$ and letting $x = \mathbf{v}_w^\top \mathbf{v}_c'$, the local objective $\ell(x)$ is:
$$\ell(x) = N(w, c) \log \sigma(x) + k \frac{N(w) N(c)}{D} \log \sigma(-x)$$

**Step 2: Differentiating with respect to $x$:**
Using $\frac{d}{dx} \log \sigma(x) = 1 - \sigma(x) = \sigma(-x) = \frac{1}{1 + e^x}$ and $\frac{d}{dx} \log \sigma(-x) = -\sigma(x) = -\frac{e^x}{1 + e^x}$:
$$\frac{\partial \ell}{\partial x} = N(w, c) (1 - \sigma(x)) - k \frac{N(w) N(c)}{D} \sigma(x)$$
Set the derivative to zero:
$$N(w, c) (1 - \sigma(x)) = k \frac{N(w) N(c)}{D} \sigma(x)$$
Divide both sides by $\sigma(x)$:
$$N(w, c) \left( \frac{1 - \sigma(x)}{\sigma(x)} \right) = k \frac{N(w) N(c)}{D}$$
Recall that $\frac{1 - \sigma(x)}{\sigma(x)} = \frac{1 - \frac{1}{1 + e^{-x}}}{\frac{1}{1 + e^{-x}}} = e^{-x}$.
Therefore:
$$N(w, c) e^{-x} = k \frac{N(w) N(c)}{D}$$
Solving for $e^x$:
$$e^x = \frac{N(w, c) \cdot D}{k \cdot N(w) N(c)}$$

**Step 3: Deriving Shifted PMI:**
Taking the natural logarithm of both sides:
$$x^* = \log \left( \frac{N(w, c) \cdot D}{N(w) N(c)} \cdot \frac{1}{k} \right) = \log \left( \frac{N(w, c) / D}{(N(w) / D) \cdot (N(c) / D)} \right) - \log k$$
Recognizing that:
$$\frac{\hat{P}(w, c)}{\hat{P}(w) \hat{P}(c)} = \frac{N(w, c) / D}{(N(w) / D) \cdot (N(c) / D)}$$
This is by definition the Pointwise Mutual Information:
$$\text{PMI}(w, c) = \log \frac{\hat{P}(w, c)}{\hat{P}(w) \hat{P}(c)}$$
Substituting this in:
$$\boxed{x^* = \mathbf{v}_w^\top \mathbf{v}_c' = \text{PMI}(w, c) - \log k}$$
This concludes the proof. Word2Vec with SGNS is fundamentally factorizing the shifted Pointwise Mutual Information matrix.
</details>
