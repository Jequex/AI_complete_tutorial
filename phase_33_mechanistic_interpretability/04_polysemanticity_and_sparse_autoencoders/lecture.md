# Topic 04: Polysemanticity & Sparse Autoencoders (SAEs)

## 1. Introduction & 12-Year-Old Intuition

Imagine you have a small suitcase that only holds 2 books ($d = 2$). But you are going to summer camp for 5 weeks, and you desperately want to bring 5 different books ($D = 5$). 

If you try to pack 5 full-sized hardcover books flat, the suitcase won't close. So you do something clever: you tear the covers off and interleave the pages at slight angles so that all 5 books fit into the 2-book space.

When you unzip the suitcase at camp, every single pocket contains mixed-up pages from 3 different books! 
- If you reach into Pocket 1, you find a page from a Harry Potter book, a page from a math textbook, and a recipe for chocolate chip cookies.
- This confusion is called **Polysemanticity**: a single pocket (neuron) responds to multiple, completely unrelated ideas.

Why would the brain or an AI do this?
Because **you only read one book at a time**! This is called **Feature Sparsity**. In the real world, you rarely need the concept of *"quantum physics"*, *"corgi dogs"*, and *"18th-century French poetry"* in the exact same sentence. Because features are sparse, the network packs hundreds of thousands of concepts into a small vector space by angling them almost perpendicularly to each other. This is the **Superposition Hypothesis**.

A **Sparse Autoencoder (SAE)** is a mathematical machine that unzips the suitcase and neatly reconstructs the 5 original, pristine, uncorrupted hardcover books!

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Entity | Domain | Interpretability Role |
| :--- | :--- | :--- | :--- |
| $d$ | Model Hidden Dimension | Integer (e.g., 4096) | Ambient vector dimension of the Transformer residual stream. |
| $D$ | Feature Space Dimension | Integer, $D \gg d$ | True count of semantic concepts represented in superposition ($D / d \ge 16$). |
| $f_k \ge 0$ | True Latent Feature Activation | Non-negative scalar | Activation intensity of semantic concept $k$. |
| $p$ | Feature Sparsity Probability | Scalar $\ll 1$ | $P(f_k > 0)$; probability that feature $k$ appears in any given token. |
| $\mathbf{v}_k \in \mathbb{R}^d$ | Feature Direction Vector | $\|\mathbf{v}_k\|_2 = 1$ | Unit vector in residual stream representing concept $k$. |
| $\mathbf{x} \in \mathbb{R}^d$ | Model Activation Vector | $\mathbf{x} = \sum_k f_k \mathbf{v}_k$ | Polysemantic representation observed in the residual stream. |
| $W_{\text{enc}}, \mathbf{b}_{\text{enc}}$ | SAE Encoder Parameters | $W_{\text{enc}} \in \mathbb{R}^{D \times d}, \mathbf{b} \in \mathbb{R}^D$ | Projects polysemantic space $d$ to overcomplete feature space $D$. |
| $\mathbf{z} \in \mathbb{R}^D$ | SAE Latent Activations | $\mathbf{z} = \text{ReLU}(W_{\text{enc}}\mathbf{x} + \mathbf{b}_{\text{enc}})$ | Monosemantic, sparse feature activations. |
| $W_{\text{dec}}, \mathbf{b}_{\text{dec}}$ | SAE Decoder Parameters | $W_{\text{dec}} \in \mathbb{R}^{d \times D}, \mathbf{b} \in \mathbb{R}^d$ | Dictionary columns reconstructing the input: $\hat{\mathbf{x}} = W_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{dec}}$. |
| $\lambda$ | $L_1$ Sparsity Coefficient | Real scalar $> 0$ | Regularization weight penalizing non-zero latent feature firing. |
| $\mathcal{L}_{\text{SAE}}$ | Total SAE Loss | Real scalar $\ge 0$ | Objective balancing reconstruction fidelity ($\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2$) with sparsity. |

---

## 3. The Superposition Hypothesis

In **Toy Models of Superposition** (Elhage et al., 2022), Anthropic researchers formalized why neural networks represent more features than dimensions:

> **The Superposition Hypothesis**: Neural networks represent more independent features than the number of dimensions in their vector space ($D > d$) by assigning features almost-orthogonal directions in high dimensions. The network tolerates small cross-talk noise between non-orthogonal features because features are sparse ($P(f_k \neq 0) \ll 1$).

### 3.1 The Geometry of High Dimensions: Almost-Orthogonality
In $d$-dimensional Euclidean space, the maximum number of strictly orthogonal unit vectors is exactly $d$.
However, by the **Johnson-Lindenstrauss Lemma**, if we tolerate a small inner product interference $|\langle \mathbf{v}_i, \mathbf{v}_j \rangle| \le \epsilon$ between any two vectors, the number of almost-orthogonal vectors that can be packed into $\mathbb{R}^d$ grows **exponentially** with $d$:

$$D \approx \exp\left( \mathcal{O}(\epsilon^2 d) \right) \gg d$$

When $d = 4096$, the residual stream can easily pack millions of distinct semantic directions with negligible cross-talk, provided only a small subset fire simultaneously!

### 3.2 The 2D Toy Model: The Pentagon
In 2 dimensions ($d = 2$), the maximum number of orthogonal vectors is 2 (e.g., standard basis $\mathbf{e}_1, \mathbf{e}_2$).
If the data distribution contains $D = 5$ sparse features with equal importance, the optimal geometric packing that minimizes reconstruction error is a **regular pentagon**:

$$\mathbf{v}_k = \begin{bmatrix} \cos\left(\frac{2\pi k}{5}\right) \\ \sin\left(\frac{2\pi k}{5}\right) \end{bmatrix}, \quad k \in \{0, 1, 2, 3, 4\}$$

Each pair of adjacent vectors forms an angle of $72^\circ$, with cosine similarity $\cos(72^\circ) \approx 0.309$.

---

## 4. The Curse of Polysemanticity

Because features are packed in superposition along non-basis directions, the individual coordinate neurons (basis vectors) are mixtures of multiple features:

$$x_i = \mathbf{e}_i^\top \mathbf{x} = \sum_{k=1}^D f_k (\mathbf{e}_i^\top \mathbf{v}_k)$$

In standard language models (e.g., GPT-4 or Claude):
- A single neuron in Layer 15 might fire strongly for:
  1. Mentions of the Golden Gate Bridge,
  2. Academic discussions of neuroscience,
  3. Python syntax errors involving `IndexError`.

This makes individual neurons fundamentally uninterpretable. Neurons are **not** the fundamental units of thought in a neural network; **directions in activation space** are.

---

## 5. Sparse Autoencoders (SAEs) as Overcomplete Dictionaries

In **Towards Monosemanticity** (Bricken et al., 2023), researchers demonstrated that **Sparse Autoencoders (SAEs)** can extract monosemantic directions from polysemantic residual activations.

An SAE is an overcomplete dictionary learning architecture:

```
Polysemantic Activation x in R^d
           │
      [ W_enc ] (d -> D, where D >> d)
           ▼
     Pre-activations + b_enc
           │
        [ ReLU ]
           ▼
Monosemantic Latent z in R^D (Sparse: most entries are exactly 0.0)
           │
      [ W_dec ] (D -> d, columns normalized to ||w_j|| = 1)
           ▼
Reconstructed Activation x_hat in R^d
```

### 5.1 SAE Mathematical Formulation
1. **Encoder**:
   $$\mathbf{z} = \text{ReLU}\left( W_{\text{enc}} \mathbf{x} + \mathbf{b}_{\text{enc}} \right)$$
   where $W_{\text{enc}} \in \mathbb{R}^{D \times d}$ and $\mathbf{b}_{\text{enc}} \in \mathbb{R}^D$.
2. **Decoder**:
   $$\hat{\mathbf{x}} = W_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{dec}}$$
   where $W_{\text{dec}} \in \mathbb{R}^{d \times D}$ and each column $\mathbf{w}_j^{\text{dec}}$ is constrained to unit norm $\|\mathbf{w}_j^{\text{dec}}\|_2 = 1$.

### 5.2 The Regularized Loss Function
The SAE is trained to minimize the sum of Mean Squared Reconstruction Error and an $L_1$ sparsity penalty:

$$\boxed{\mathcal{L}_{\text{SAE}} = \frac{1}{B} \sum_{i=1}^B \left( \|\mathbf{x}^{(i)} - \hat{\mathbf{x}}^{(i)}\|_2^2 + \lambda \sum_{j=1}^D |z_j^{(i)}| \right)}$$

- **$\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2$**: Enforces that the linear combination of dictionary atoms faithfully reconstructs the full residual stream without losing information.
- **$\lambda \sum_j |z_j|$**: Forces the latent code $\mathbf{z}$ to be extremely sparse (e.g., only 5–30 features active out of 100,000 latents per token).

### 5.3 Why the Unit-Norm Constraint on $W_{\text{dec}}$ is Mandatory
If columns of $W_{\text{dec}}$ were unconstrained:
- The network could scale $W_{\text{dec}}$ by 1000 and divide $\mathbf{z}$ by 1000.
- The reconstruction $\hat{\mathbf{x}} = W_{\text{dec}} \mathbf{z}$ would remain unchanged.
- But the $L_1$ penalty $\lambda \|\mathbf{z}\|_1$ would be artificially reduced by 1000!

Constraining $\|\mathbf{w}_j^{\text{dec}}\|_2 = 1$ eliminates this scaling loophole, forcing the $L_1$ penalty to drive true feature selection.

---

## 6. Monosemanticity: From Chaos to Interpretable Features

Once an SAE is trained on intermediate activations:
1. Every latent dimension $j \in \{1, \dots, D\}$ corresponds to a single, interpretable concept.
2. Anthropic's Golden Gate Claude demonstrated this at scale: a single latent feature corresponds to the Golden Gate Bridge. Clamping this latent feature high causes the model to obsessively mention the bridge in all conversations.
3. As demonstrated in our pure Python `experiment.py`, an SAE trained on 2D superposed pentagon activations cleanly untangles the 5 ground-truth features with mean cosine similarity $> 0.94$.
