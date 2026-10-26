# Topic 23.2: Exercises & Solutions — Sentence Embeddings & SBERT

---

### Problem 1: Mathematical Proof of the Anisotropy Cosine Floor (Easy)

Suppose two unrelated sentence embedding vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$ are modeled as:
$$\mathbf{u} = \boldsymbol{\mu} + \boldsymbol{\epsilon}_u, \quad \mathbf{v} = \boldsymbol{\mu} + \boldsymbol{\epsilon}_v$$
where $\boldsymbol{\mu} \in \mathbb{R}^d$ is a fixed non-zero mean center with $\|\boldsymbol{\mu}\|_2 = M > 0$.
The perturbations $\boldsymbol{\epsilon}_u, \boldsymbol{\epsilon}_v$ are independent random vectors with zero mean $\mathbb{E}[\boldsymbol{\epsilon}] = \mathbf{0}$, expected squared norm $\mathbb{E}[\|\boldsymbol{\epsilon}_u\|_2^2] = \mathbb{E}[\|\boldsymbol{\epsilon}_v\|_2^2] = \sigma^2$, and $\mathbb{E}[\boldsymbol{\epsilon}_u^\top \boldsymbol{\epsilon}_v] = 0$.

1. Using a first-order Taylor expansion for $\frac{1}{\sqrt{1 + x}} \approx 1 - \frac{1}{2}x$, derive an approximation for the expected cosine similarity $\mathbb{E}[\cos(\mathbf{u}, \mathbf{v})]$ in terms of the ratio $\gamma = \frac{\sigma^2}{M^2}$.
2. If $M = 10.0$ and $\sigma = 2.0$, calculate the numerical value of $\mathbb{E}[\cos(\mathbf{u}, \mathbf{v})]$.
3. What is the expected cosine similarity of truly unrelated sentences in an isotropic space ($M = 0$)? Explain the significance of this difference.

<details>
<summary>View Complete Solution</summary>

**Step 1: Inner Product and Norm Expansions:**
The inner product is:
$$\mathbf{u}^\top \mathbf{v} = (\boldsymbol{\mu} + \boldsymbol{\epsilon}_u)^\top (\boldsymbol{\mu} + \boldsymbol{\epsilon}_v) = \|\boldsymbol{\mu}\|_2^2 + \boldsymbol{\mu}^\top \boldsymbol{\epsilon}_v + \boldsymbol{\mu}^\top \boldsymbol{\epsilon}_u + \boldsymbol{\epsilon}_u^\top \boldsymbol{\epsilon}_v$$
Taking expectation over independent perturbations with zero mean:
$$\mathbb{E}[\mathbf{u}^\top \mathbf{v}] = \|\boldsymbol{\mu}\|_2^2 = M^2$$

Now consider the squared norm:
$$\|\mathbf{u}\|_2^2 = \|\boldsymbol{\mu} + \boldsymbol{\epsilon}_u\|_2^2 = \|\boldsymbol{\mu}\|_2^2 + 2\boldsymbol{\mu}^\top \boldsymbol{\epsilon}_u + \|\boldsymbol{\epsilon}_u\|_2^2$$
Its expectation is $\mathbb{E}[\|\mathbf{u}\|_2^2] = M^2 + \sigma^2 = M^2 (1 + \gamma)$ where $\gamma = \frac{\sigma^2}{M^2}$.
Approximating the norm $\|\mathbf{u}\|_2 \approx M \sqrt{1 + \gamma}$:
$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} \approx \frac{M^2}{M^2 (1 + \gamma)} = \frac{1}{1 + \gamma}$$
Using the Taylor expansion $\frac{1}{1 + \gamma} \approx 1 - \gamma$ for small $\gamma$:
$$\boxed{\mathbb{E}[\cos(\mathbf{u}, \mathbf{v})] \approx 1 - \frac{\sigma^2}{M^2}}$$

**Step 2: Numerical Calculation:**
Given $M = 10.0 \implies M^2 = 100.0$, and $\sigma = 2.0 \implies \sigma^2 = 4.0$:
$$\gamma = \frac{4.0}{100.0} = 0.04$$
$$\mathbb{E}[\cos(\mathbf{u}, \mathbf{v})] \approx \frac{1}{1 + 0.04} \approx 0.9615$$
Even though the semantic content vectors $\boldsymbol{\epsilon}_u$ and $\boldsymbol{\epsilon}_v$ are completely uncorrelated ($\mathbb{E}[\boldsymbol{\epsilon}_u^\top \boldsymbol{\epsilon}_v] = 0$), the raw embeddings exhibit an extraordinarily high cosine similarity of $0.96$!

**Step 3: Isotropic Comparison:**
In an isotropic space ($M = 0$), $\mathbf{u} = \boldsymbol{\epsilon}_u$ and $\mathbf{v} = \boldsymbol{\epsilon}_v$:
$$\mathbb{E}[\mathbf{u}^\top \mathbf{v}] = \mathbb{E}[\boldsymbol{\epsilon}_u^\top \boldsymbol{\epsilon}_v] = 0 \implies \mathbb{E}[\cos(\mathbf{u}, \mathbf{v})] = 0.0$$
In an isotropic space, orthogonal unrelated concepts score $0.0$. In raw BERT, the non-zero mean center $M$ forces all vectors into a narrow cone where cosine similarity is artificially inflated to $>0.90$.
</details>

---

### Problem 2: SBERT NLI Head Gradient with Absolute Difference (Medium)

In the SBERT classification architecture, pooled sentence vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$ are transformed into the concatenated feature vector:
$$\mathbf{h} = [\mathbf{u}, \mathbf{v}, |\mathbf{u} - \mathbf{v}|] \in \mathbb{R}^{3d}$$
Let $\mathbf{d} = |\mathbf{u} - \mathbf{v}| \in \mathbb{R}^d$, where $d_j = |u_j - v_j|$.
The logit vector for $k = 3$ classes is:
$$\mathbf{z} = \mathbf{W}^\top \mathbf{h} + \mathbf{b} \in \mathbb{R}^3$$
Partition the weight matrix $\mathbf{W} \in \mathbb{R}^{3d \times 3}$ into three $d \times 3$ blocks:
$$\mathbf{W} = \begin{bmatrix} \mathbf{W}_u \\ \mathbf{W}_v \\ \mathbf{W}_d \end{bmatrix}$$
so that $\mathbf{z} = \mathbf{W}_u^\top \mathbf{u} + \mathbf{W}_v^\top \mathbf{v} + \mathbf{W}_d^\top \mathbf{d} + \mathbf{b}$.
Let the incoming gradient from the cross-entropy loss with respect to logits $\mathbf{z}$ be $\mathbf{g} = \nabla_{\mathbf{z}} \mathcal{L} \in \mathbb{R}^3$.

1. Derive the gradient of the loss with respect to vector $\mathbf{u}$, namely $\nabla_{\mathbf{u}} \mathcal{L} \in \mathbb{R}^d$.
2. Derive the gradient of the loss with respect to vector $\mathbf{v}$, namely $\nabla_{\mathbf{v}} \mathcal{L} \in \mathbb{R}^d$.
3. Explain mathematically how the subgradient of the absolute difference term $|u_j - v_j|$ guides $\mathbf{u}$ and $\mathbf{v}$ during training.

<details>
<summary>View Complete Solution</summary>

**Step 1: Gradient with respect to $\mathbf{u}$:**
By the multivariate chain rule:
$$\nabla_{\mathbf{u}} \mathcal{L} = \left( \frac{\partial \mathbf{z}}{\partial \mathbf{u}} \right) \mathbf{g} = \mathbf{W}_u \mathbf{g} + \left( \frac{\partial \mathbf{d}}{\partial \mathbf{u}} \right) \mathbf{W}_d \mathbf{g}$$
Let $\mathbf{\delta} = \mathbf{W}_d \mathbf{g} \in \mathbb{R}^d$.
Notice that $d_j = |u_j - v_j|$. The derivative of $d_j$ with respect to $u_j$ is the sign function:
$$\frac{\partial d_j}{\partial u_j} = \text{sgn}(u_j - v_j) = \begin{cases} +1 & \text{if } u_j > v_j \\ -1 & \text{if } u_j < v_j \\ 0 & \text{if } u_j = v_j \end{cases}$$
The Jacobian $\frac{\partial \mathbf{d}}{\partial \mathbf{u}}$ is a diagonal matrix $\text{diag}(\text{sgn}(\mathbf{u} - \mathbf{v}))$.
Therefore:
$$\boxed{\nabla_{\mathbf{u}} \mathcal{L} = \mathbf{W}_u \mathbf{g} + \text{sgn}(\mathbf{u} - \mathbf{v}) \odot (\mathbf{W}_d \mathbf{g})}$$

**Step 2: Gradient with respect to $\mathbf{v}$:**
Similarly, for $\mathbf{v}$:
$$\frac{\partial d_j}{\partial v_j} = -\text{sgn}(u_j - v_j)$$
Thus:
$$\boxed{\nabla_{\mathbf{v}} \mathcal{L} = \mathbf{W}_v \mathbf{g} - \text{sgn}(\mathbf{u} - \mathbf{v}) \odot (\mathbf{W}_d \mathbf{g})}$$

**Step 3: Geometric Interpretation:**
For an entailment pair ($s_A$ entails $s_B$), the target loss encourages the distance term $\mathbf{d} = |\mathbf{u} - \mathbf{v}|$ to shrink toward zero.
The gradient update pulls $u_j$ toward $v_j$ when $u_j > v_j$ (since $\text{sgn} = +1$) and pushes $u_j$ upward toward $v_j$ when $u_j < v_j$ (since $\text{sgn} = -1$).
Simultaneously, for contradictory pairs, the loss encourages distance $\mathbf{d}$ to expand. This element-wise directional force directly pulls synonymous sentences into identical coordinates and expels contradictory ones.
</details>

---

### Problem 3: Derivation of BERT-Whitening (Hard)

Given a set of unnormalized sentence vectors $\{\mathbf{u}_i\}_{i=1}^M \subset \mathbb{R}^d$ with empirical mean $\boldsymbol{\mu} = \frac{1}{M}\sum_{i=1}^M \mathbf{u}_i$ and covariance matrix:
$$\boldsymbol{\Sigma} = \frac{1}{M} \sum_{i=1}^M (\mathbf{u}_i - \boldsymbol{\mu})(\mathbf{u}_i - \boldsymbol{\mu})^\top \in \mathbb{R}^{d \times d}$$
Because $\boldsymbol{\Sigma}$ is symmetric positive semi-definite, its eigendecomposition is:
$$\boldsymbol{\Sigma} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top$$
where $\mathbf{U} \in \mathbb{R}^{d \times d}$ is an orthogonal matrix of eigenvectors ($\mathbf{U}^\top \mathbf{U} = \mathbf{I}$), and $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \dots, \lambda_d)$ contains the positive eigenvalues.

Define the linear transformation:
$$\tilde{\mathbf{u}}_i = \mathbf{W}^\top (\mathbf{u}_i - \boldsymbol{\mu})$$
where $\mathbf{W} = \mathbf{U} \boldsymbol{\Lambda}^{-1/2}$.

1. Prove that the transformed embeddings $\tilde{\mathbf{u}}$ have strictly zero mean.
2. Prove that the covariance matrix of $\tilde{\mathbf{u}}$ is the identity matrix $\mathbf{I}_d$.
3. Explain why this transformation eliminates the anisotropy cone.

<details>
<summary>View Complete Solution</summary>

**Step 1: Proving Zero Mean:**
$$\mathbb{E}[\tilde{\mathbf{u}}] = \frac{1}{M} \sum_{i=1}^M \tilde{\mathbf{u}}_i = \frac{1}{M} \sum_{i=1}^M \mathbf{W}^\top (\mathbf{u}_i - \boldsymbol{\mu}) = \mathbf{W}^\top \left( \frac{1}{M} \sum_{i=1}^M \mathbf{u}_i - \boldsymbol{\mu} \right) = \mathbf{W}^\top (\boldsymbol{\mu} - \boldsymbol{\mu}) = \mathbf{0}$$

**Step 2: Proving Identity Covariance:**
Compute the covariance of the transformed vectors:
$$\text{Cov}(\tilde{\mathbf{u}}) = \frac{1}{M} \sum_{i=1}^M \tilde{\mathbf{u}}_i \tilde{\mathbf{u}}_i^\top = \frac{1}{M} \sum_{i=1}^M \mathbf{W}^\top (\mathbf{u}_i - \boldsymbol{\mu}) (\mathbf{u}_i - \boldsymbol{\mu})^\top \mathbf{W}$$
Factoring $\mathbf{W}^\top$ and $\mathbf{W}$ out of the summation:
$$\text{Cov}(\tilde{\mathbf{u}}) = \mathbf{W}^\top \left( \frac{1}{M} \sum_{i=1}^M (\mathbf{u}_i - \boldsymbol{\mu}) (\mathbf{u}_i - \boldsymbol{\mu})^\top \right) \mathbf{W} = \mathbf{W}^\top \boldsymbol{\Sigma} \mathbf{W}$$
Substitute $\mathbf{W} = \mathbf{U} \boldsymbol{\Lambda}^{-1/2}$ and $\boldsymbol{\Sigma} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top$:
$$\text{Cov}(\tilde{\mathbf{u}}) = (\mathbf{U} \boldsymbol{\Lambda}^{-1/2})^\top (\mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top) (\mathbf{U} \boldsymbol{\Lambda}^{-1/2})$$
Since $\boldsymbol{\Lambda}^{-1/2}$ is diagonal, $(\boldsymbol{\Lambda}^{-1/2})^\top = \boldsymbol{\Lambda}^{-1/2}$.
Using the orthogonality property $\mathbf{U}^\top \mathbf{U} = \mathbf{I}$:
$$\text{Cov}(\tilde{\mathbf{u}}) = \boldsymbol{\Lambda}^{-1/2} (\mathbf{U}^\top \mathbf{U}) \boldsymbol{\Lambda} (\mathbf{U}^\top \mathbf{U}) \boldsymbol{\Lambda}^{-1/2} = \boldsymbol{\Lambda}^{-1/2} \mathbf{I} \boldsymbol{\Lambda} \mathbf{I} \boldsymbol{\Lambda}^{-1/2} = \boldsymbol{\Lambda}^{-1/2} \boldsymbol{\Lambda} \boldsymbol{\Lambda}^{-1/2} = \mathbf{I}_d$$
Thus, $\text{Cov}(\tilde{\mathbf{u}}) = \mathbf{I}_d$.

**Step 3: Why Anisotropy is Eliminated:**
- Subtracting $\boldsymbol{\mu}$ translates the center of mass of the embeddings from the displaced cone apex directly to the origin $\mathbf{0}$.
- Multiplying by $\mathbf{U}^\top$ rotates the coordinate axes to align with the principal directions of variation.
- Multiplying by $\boldsymbol{\Lambda}^{-1/2}$ divides each principal axis by its standard deviation $\sqrt{\lambda_j}$. The hyper-ellipsoid that was stretched $50\times$ along the primary singular direction is sphericalized into a uniform unit sphere.
- Directional variance is equalized in all dimensions, restoring the full expressive rank of $\mathbb{R}^d$.
</details>

---

### Problem 4: Retrieval Scalability Analysis — Cross-Encoder vs. Bi-Encoder (Medium)

You are designing a semantic question-answering search system over a knowledge base of $M = 1,000,000$ documentation paragraphs.
A user submits a query sentence $q$.
- A standard 12-layer BERT Cross-Encoder requires $25$ milliseconds of GPU time per $(q, p)$ pair to compute relevance scores.
- An SBERT Bi-Encoder requires $10$ milliseconds to encode query $q$ into a 768-dimensional vector, after which computing the cosine similarity against all $1,000,000$ pre-computed document embeddings in memory takes $2.5$ milliseconds via matrix-vector multiplication.

1. Calculate the total latency for the Cross-Encoder to evaluate the query against the entire knowledge base.
2. Calculate the total latency for the SBERT Bi-Encoder system.
3. What is the speedup factor?
4. How do real-world production search architectures combine Bi-Encoders and Cross-Encoders to achieve both sub-second latency and maximum accuracy?

<details>
<summary>View Complete Solution</summary>

**Step 1: Cross-Encoder Latency:**
For $M = 1,000,000$ documents:
$$\text{Latency}_{\text{Cross}} = 1,000,000 \times 0.025 \text{ seconds} = 25,000 \text{ seconds} \approx 6.94 \text{ hours!}$$
The user would wait nearly 7 hours for a single query response.

**Step 2: Bi-Encoder Latency:**
All $1,000,000$ document vectors were encoded offline during indexing.
At query time:
- Query encoding: $10 \text{ ms} = 0.010 \text{ seconds}$.
- Dot product matrix multiplication $[1 \times 768] \times [768 \times 1,000,000]$: $2.5 \text{ ms} = 0.0025 \text{ seconds}$.
- Total latency: $10 + 2.5 = 12.5 \text{ ms} = 0.0125 \text{ seconds}$.

**Step 3: Speedup Factor:**
$$\text{Speedup} = \frac{25,000 \text{ s}}{0.0125 \text{ s}} = 2,000,000\times \text{ faster}$$

**Step 4: Two-Stage Re-ranking Cascade:**
Production search engines (e.g., Google, Cohere, Pinecone) use a two-stage cascade:
1. **Stage 1 (Bi-Encoder / SBERT Retrieval):** Scan the full $1,000,000$ corpus using fast vector search / ANN index in $\approx 15 \text{ ms}$ to retrieve the top $K = 100$ candidate documents.
2. **Stage 2 (Cross-Encoder Re-ranking):** Pass only the top 100 $(q, p)$ candidate pairs through the heavy Cross-Encoder: $100 \times 25 \text{ ms} = 2.5 \text{ seconds}$ (or $250 \text{ ms}$ with 10 parallel batches).
This achieves the best of both worlds: high recall across the whole corpus in milliseconds, followed by full cross-attention ranking on the top 100 candidates.
</details>
