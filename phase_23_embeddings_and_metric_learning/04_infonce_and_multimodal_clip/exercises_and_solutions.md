# Topic 23.4: Exercises & Solutions — InfoNCE Loss & Multimodal CLIP

---

### Problem 1: Step-by-Step Proof of the InfoNCE Mutual Information Lower Bound (Hard)

Let $X$ and $Y$ be random variables with joint distribution $p(x, y)$ and marginals $p(x), p(y)$.
Consider a set of samples $\mathcal{X} = \{y_0, y_1, \dots, y_K\}$ where $y_0 \sim p(y \mid x)$ is drawn conditionally given $x$, and $\{y_1, \dots, y_K\} \sim p(y)$ are $K$ independent negative samples drawn from the marginal distribution.
Let $C \in \{0, 1, \dots, K\}$ denote the index of the true positive with uniform prior $P(C = j) = \frac{1}{K + 1}$.

1. Use Bayes' theorem to write the exact true posterior probability $P(C = 0 \mid x, \mathcal{X})$ in terms of the density ratios $\frac{p(y_j \mid x)}{p(y_j)}$.
2. Show that if our scoring model is an optimal density estimator $f(x, y) \propto \frac{p(y \mid x)}{p(y)}$, the categorical cross-entropy loss is:
   $$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}_{x, \mathcal{X}} \left[ \log \frac{f(x, y_0)}{\sum_{j=0}^K f(x, y_j)} \right]$$
3. Apply Jensen's inequality to prove that:
   $$I(X; Y) \ge \log(K) - \mathcal{L}_{\text{InfoNCE}}$$
4. What happens to the lower bound as the number of negative samples $K \to \infty$?

<details>
<summary>View Complete Solution</summary>

**Step 1: Bayes' Theorem for True Posterior:**
The probability of observing $x$ and set $\mathcal{X}$ given that $y_j$ is the true conditional sample ($C = j$) is:
$$p(x, \mathcal{X} \mid C = j) = p(x, y_j) \prod_{l \ne j} p(y_l)$$
Since the prior $P(C = j) = \frac{1}{K + 1}$ is constant across all $j$:
$$P(C = 0 \mid x, \mathcal{X}) = \frac{p(x, y_0) \prod_{l \ne 0} p(y_l)}{\sum_{j=0}^K p(x, y_j) \prod_{l \ne j} p(y_l)}$$
Dividing both numerator and denominator by the product of all marginals $p(x) \prod_{l=0}^K p(y_l)$:
$$\frac{p(x, y_j) \prod_{l \ne j} p(y_l)}{p(x) \prod_{l=0}^K p(y_l)} = \frac{p(x, y_j)}{p(x) p(y_j)} = \frac{p(y_j \mid x)}{p(y_j)}$$
Therefore:
$$P(C = 0 \mid x, \mathcal{X}) = \frac{\frac{p(y_0 \mid x)}{p(y_0)}}{\sum_{j=0}^K \frac{p(y_j \mid x)}{p(y_j)}}$$

**Step 2: InfoNCE Representation:**
Setting $f(x, y) \propto \frac{p(y \mid x)}{p(y)}$, the model's predicted probability is:
$$\hat{P}(C = 0 \mid x, \mathcal{X}) = \frac{f(x, y_0)}{\sum_{j=0}^K f(x, y_j)}$$
The negative log-likelihood loss is:
$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}_{x, \mathcal{X}} \left[ \log \left( \frac{f(x, y_0)}{f(x, y_0) + \sum_{j=1}^K f(x, y_j)} \right) \right] = \mathbb{E}_{x, \mathcal{X}} \left[ \log \left( 1 + \frac{p(y_0)}{p(y_0 \mid x)} \sum_{j=1}^K \frac{p(y_j \mid x)}{p(y_j)} \right) \right]$$

**Step 3: Applying Jensen's Inequality:**
Since $g(z) = \log(1 + c \cdot z)$ is concave for $c > 0$, by Jensen's inequality:
$$\mathbb{E}_{y_{1:K}} \left[ \log \left( 1 + \frac{p(y_0)}{p(y_0 \mid x)} \sum_{j=1}^K \frac{p(y_j \mid x)}{p(y_j)} \right) \right] \le \log \left( 1 + \frac{p(y_0)}{p(y_0 \mid x)} \mathbb{E}_{y_{1:K}} \left[ \sum_{j=1}^K \frac{p(y_j \mid x)}{p(y_j)} \right] \right)$$
Now evaluate the expectation for each independent negative $y_j \sim p(y)$:
$$\mathbb{E}_{y_j \sim p(y)} \left[ \frac{p(y_j \mid x)}{p(y_j)} \right] = \int p(y_j) \frac{p(y_j \mid x)}{p(y_j)} dy_j = \int p(y_j \mid x) dy_j = 1$$
Summing over all $K$ negatives:
$$\mathbb{E}_{y_{1:K}} \left[ \sum_{j=1}^K \frac{p(y_j \mid x)}{p(y_j)} \right] = \sum_{j=1}^K 1 = K$$
Substituting this back:
$$\mathcal{L}_{\text{InfoNCE}} \ge \mathbb{E}_{x, y_0} \left[ \log \left( 1 + K \frac{p(y_0)}{p(y_0 \mid x)} \right) \right] \ge \mathbb{E}_{x, y_0} \left[ \log \left( K \frac{p(y_0)}{p(y_0 \mid x)} \right) \right]$$
Expanding the logarithm:
$$\mathcal{L}_{\text{InfoNCE}} \ge \log(K) + \mathbb{E}_{x, y_0} \left[ \log \frac{p(y_0)}{p(y_0 \mid x)} \right] = \log(K) - \mathbb{E}_{p(x, y)} \left[ \log \frac{p(y \mid x)}{p(y)} \right]$$
Recall that $I(X; Y) = \mathbb{E}_{p(x, y)} \left[ \log \frac{p(y \mid x)}{p(y)} \right]$.
Therefore:
$$\mathcal{L}_{\text{InfoNCE}} \ge \log(K) - I(X; Y) \implies \boxed{I(X; Y) \ge \log(K) - \mathcal{L}_{\text{InfoNCE}}}$$

**Step 4: Asymptotic Behavior as $K \to \infty$:**
As the number of negative distractors $K$ increases, the capacity of the bound $\log(K)$ grows without bound. If $K$ is small (e.g. $K = 1$), $\log(K) = 0$, meaning the lower bound cannot guarantee more than $0$ nats of mutual information. Large batches (e.g., $K = 32,768 \implies \log(K) \approx 10.4$ nats) are mathematically required to learn deep, high-information representations.
</details>

---

### Problem 2: InfoNCE Analytical Gradients with Temperature (Medium)

Let query $q \in \mathbb{R}^d$ and keys $\{k_j\}_{j=0}^K \subset \mathbb{R}^d$ be unit-normalized vectors ($\|q\| = \|k_j\| = 1$).
Let $s_j = q^\top k_j \in [-1, 1]$ be the cosine similarity, with positive key at $j = 0$.
The InfoNCE loss is:
$$\mathcal{L} = -\log \left( \frac{e^{s_0 / \tau}}{\sum_{j=0}^K e^{s_j / \tau}} \right)$$
Define the softmax probabilities $p_j = \frac{e^{s_j / \tau}}{\sum_{m=0}^K e^{s_m / \tau}}$.

1. Derive the partial derivative $\frac{\partial \mathcal{L}}{\partial s_0}$ with respect to the positive key similarity.
2. Derive the partial derivative $\frac{\partial \mathcal{L}}{\partial s_j}$ with respect to a negative key similarity ($j \ge 1$).
3. Derive the full gradient vector with respect to the query embedding $q$, namely $\nabla_q \mathcal{L} \in \mathbb{R}^d$.
4. Analyze the direction of $\nabla_q \mathcal{L}$ and explain how temperature $\tau$ scales the step.

<details>
<summary>View Complete Solution</summary>

**Step 1: Derivative w.r.t Positive Similarity $s_0$:**
Rewrite the loss:
$$\mathcal{L} = -\frac{s_0}{\tau} + \log \left( \sum_{j=0}^K e^{s_j / \tau} \right)$$
Differentiating with respect to $s_0$:
$$\frac{\partial \mathcal{L}}{\partial s_0} = -\frac{1}{\tau} + \frac{1}{\sum_{m=0}^K e^{s_m / \tau}} \left( \frac{1}{\tau} e^{s_0 / \tau} \right) = -\frac{1}{\tau} + \frac{1}{\tau} p_0 = \boxed{-\frac{1}{\tau} (1 - p_0)}$$

**Step 2: Derivative w.r.t Negative Similarity $s_j$ ($j \ge 1$):**
$$\frac{\partial \mathcal{L}}{\partial s_j} = 0 + \frac{1}{\sum_{m=0}^K e^{s_m / \tau}} \left( \frac{1}{\tau} e^{s_j / \tau} \right) = \boxed{+\frac{1}{\tau} p_j}$$

**Step 3: Gradient with respect to Query $q$:**
Since $s_j = q^\top k_j$, we have $\nabla_q s_j = k_j$.
By the multivariate chain rule:
$$\nabla_q \mathcal{L} = \sum_{j=0}^K \frac{\partial \mathcal{L}}{\partial s_j} \nabla_q s_j = -\frac{1}{\tau} (1 - p_0) k_0 + \sum_{j=1}^K \frac{1}{\tau} p_j k_j$$
Since $\sum_{j=0}^K p_j = 1 \implies 1 - p_0 = \sum_{j=1}^K p_j$, we can rewrite:
$$\boxed{\nabla_q \mathcal{L} = \frac{1}{\tau} \left( \sum_{j=1}^K p_j k_j - (1 - p_0) k_0 \right) = \frac{1}{\tau} \left( \sum_{j=0}^K p_j k_j - k_0 \right)}$$

**Step 4: Geometric Interpretation:**
- The gradient update for $q$ under gradient descent is $-\eta \nabla_q \mathcal{L}$:
  $$\Delta q = \frac{\eta}{\tau} \left( k_0 - \sum_{j=0}^K p_j k_j \right)$$
- Query $q$ is pulled directly toward the positive key $k_0$, and pushed away from the probability-weighted center of mass of the negative keys $\sum_{j=1}^K p_j k_j$.
- Lowering temperature $\tau$ acts as a global gradient amplifier ($\frac{1}{\tau}$ scaling) while exponentially concentrating the weights $p_j$ on the nearest (hardest) negatives.
</details>

---

### Problem 3: CLIP Batch Scaling & Memory Footprint (Medium)

In OpenAI's CLIP paper (Radford et al., 2021), the authors pre-trained their models using an extraordinarily large mini-batch size of $N = 32,768$ image-text pairs distributed across 256 GPUs.
Let the shared embedding dimensionality be $d = 512$.

1. For a batch of $N = 32,768$, calculate the number of positive pairs and negative pairs evaluated in a single forward pass.
2. Calculate the memory required to store the full $N \times N$ similarity matrix $\mathbf{S}$ in single-precision floating point (`float32`, 4 bytes per float).
3. If the batch size were reduced to $N = 64$ (common in standard computer vision training), how many negative distractors would each image see?
4. Explain why contrastive pre-training collapses when the batch size $N$ is small, citing the mutual information lower bound $\log(K)$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Count of Positive and Negative Pairs:**
- In an $N \times N$ matrix, the total number of entries is $N^2 = (32,768)^2 = 1,073,741,824$ pairs (over 1 billion pairs!).
- Number of positive pairs (diagonal entries): $N = 32,768$.
- Number of negative distractors: $N^2 - N = 1,073,741,824 - 32,768 = \mathbf{1,073,709,056}$ negative pairs.
- Each individual image is contrasted against $K = N - 1 = 32,767$ distinct negative captions!

**Step 2: Memory Footprint of Similarity Matrix $\mathbf{S}$:**
$$\text{Elements} = 32,768 \times 32,768 = 1,073,741,824$$
$$\text{Bytes} = 1,073,741,824 \times 4 \text{ bytes} = 4,294,967,296 \text{ bytes} = \mathbf{4.0 \text{ GiB}}$$
Storing just the similarity matrix requires 4.0 GiB of VRAM (readily manageable when sharded across multi-GPU clusters).

**Step 3: Small Batch Comparison ($N = 64$):**
For $N = 64$:
$$K = N - 1 = 63 \text{ negative distractors}$$
Each image is contrasted against only 63 captions.

**Step 4: Information-Theoretic Collapse with Small Batches:**
Recall the InfoNCE mutual information bound:
$$I(X; Y) \ge \log(K) - \mathcal{L}$$
- For $N = 64 \implies K = 63 \implies \log(63) \approx 4.14$ nats.
- For $N = 32,768 \implies K = 32,767 \implies \log(32,767) \approx 10.40$ nats.
When $K$ is small, it is trivial for the network to separate the positive from 63 random unrelated negatives without learning fine-grained semantic concepts. The model quickly achieves near-zero loss ($\mathcal{L} \approx 0.1$) but the representations fail to capture rich visual details. A massive batch size forces the model to discriminate against thousands of subtle, near-miss distractors, driving representations to maximum mutual information.
</details>

---

### Problem 4: Zero-Shot Prompt Engineering & Decision Boundary (Medium)

You are deploying CLIP for a zero-shot safety filter to detect dangerous weapons in uploaded images.
The two target classes are:
- Class 1: `"weapon"`
- Class 2: `"toy"`

A user uploads an image $I$ showing a brightly colored plastic water gun.
The vision encoder produces normalized image vector $\hat{\mathbf{I}} \in \mathbb{R}^d$.
Suppose two prompt engineering strategies are tested:
- **Strategy A (Raw labels):**
  - Prompt 1: `"weapon"` $\to \mathbf{w}_{A, 1}$
  - Prompt 2: `"toy"` $\to \mathbf{w}_{A, 2}$
- **Strategy B (Contextualized prompts):**
  - Prompt 1: `"a photo of a deadly dangerous metallic weapon"` $\to \mathbf{w}_{B, 1}$
  - Prompt 2: `"a photo of a plastic colorful toy for children"` $\to \mathbf{w}_{B, 2}$

1. Why does Strategy A frequently misclassify the plastic water gun as a weapon?
2. How does Strategy B adjust the text embeddings $\mathbf{w}_1$ and $\mathbf{w}_2$ to resolve the ambiguity?
3. What is prompt ensembling, and why does averaging representations across multiple sentence templates improve classification robustness?

<details>
<summary>View Complete Solution</summary>

**Step 1: Vulnerability of Strategy A (Raw Labels):**
The single isolated word `"weapon"` was trained in diverse web contexts (video games, news headlines, metaphors like "intellect is a weapon", plastic prop discussions). Its single embedding vector $\mathbf{w}_{A, 1}$ has a broad receptive field that includes weapon-shaped silhouettes regardless of material or color. Since the water gun has the geometric silhouette of a pistol, its image embedding $\hat{\mathbf{I}}$ has high cosine similarity to the shape profile in $\mathbf{w}_{A, 1}$, causing a false positive.

**Step 2: Disambiguation under Strategy B:**
Adding adjectives (`"deadly"`, `"metallic"`, `"dangerous"`) pulls $\mathbf{w}_{B, 1}$ strongly into the visual domain of real firearms (metallic textures, gunmetal grey, crime/military scenes).
Conversely, adding `"plastic colorful toy for children"` pulls $\mathbf{w}_{B, 2}$ into the visual domain of bright neon colors, plastics, and playground settings.
Because the water gun is made of brightly colored plastic, $\hat{\mathbf{I}}^\top \mathbf{w}_{B, 2} \gg \hat{\mathbf{I}}^\top \mathbf{w}_{B, 1}$, correctly classifying the object as a toy.

**Step 3: Prompt Ensembling:**
Individual prompt templates introduce idiosyncratic lexical biases. For example, the template `"a centered photo of a {c}"` emphasizes central composition, while `"a close-up of the {c}"` emphasizes macro texture.
By averaging $M$ diverse prompt embeddings:
$$\mathbf{w}_c^* = \frac{1}{M} \sum_{m=1}^M \frac{g(\text{template}_m(c))}{\|g(\text{template}_m(c))\|}$$
the idiosyncratic stylistic artifacts cancel out as zero-mean noise, isolating the invariant core semantic prototype of class $c$. This consistently increases top-1 accuracy by 3–5% across benchmarks without requiring any model retraining.
</details>
