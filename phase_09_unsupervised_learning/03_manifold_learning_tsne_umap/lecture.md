# Topic 9.3: Manifold Learning — t-SNE, UMAP, & Non-Linear Embeddings

---

## 1. Executive Summary & The 12-Year-Old Analogy

Linear dimensionality reduction techniques like **PCA** project data onto a flat hyperplane. However, when complex real-world data lives on a curved, twisted surface—such as a Swiss Roll, a handwritten digit manifold, or a biological cell differentiation trajectory—PCA fails catastrophically, smashing distant folds directly on top of each other.

**Manifold Learning** algorithms discover the non-linear intrinsic geometry of high-dimensional data. **t-SNE** (van der Maaten & Hinton, 2008) and **UMAP** (McInnes et al., 2018) are the state-of-the-art non-linear projection algorithms that preserve neighborhood affinities and reveal latent cluster topologies.

```
       PCA FAILS ON CURVED MANIFOLDS, MANIFOLD LEARNING UNROLLS THEM
       
     High-Dimensional Swiss Roll:           2D Unrolled Representation:
          ╭──────╮
        ╭─╯  * * ╰─╮                        * * * * * * * * * * * * * *
        │  *     * │                        (Intrinsic geodesic distances
        ╰────╮ * * │                         are preserved faithfully!)
             ╰─────╯
```

---

### The 12-Year-Old Analogy: Unfolding a Crumpled Paper Map

Imagine you have a detailed 2D paper map of a vast amusement park. You crumple the map tightly into a dense 3D paper ball the size of an apple:

- **What PCA does:** PCA takes a heavy wooden hammer and flattens the crumpled paper ball onto a table. Because it only flattens linearly, the Roller Coaster entrance on page 1 gets squashed directly onto the Haunted House on page 10. Everything overlaps into a meaningless smudge!
- **What Manifold Learning (t-SNE & UMAP) does:**
  - First, it looks at tiny pieces of paper that are right next to each other (local neighbors).
  - Then, it gently unfolds the wrinkles and stretches the creases without tearing the paper.
  - The **Crowding Problem:** In 3D space, a point can have dozens of neighbors in all directions (above, below, left, right, front, back). In a flat 2D plane, there simply isn't enough physical room around a point to place 50 equidistant neighbors!
  - **The Student-t Magic Trick:** To fix the crowding problem, t-SNE uses a "stretchy rubber band" (a heavy-tailed Student-t distribution) that pushes moderate-distance points much further apart on the flat page, giving every cluster the room it needs to breathe.

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $x_i \in \mathbb{R}^D$ | High-d Vector | Observation in high-dimensional ambient space (e.g., $D = 784$ or $20,000$) |
| $y_i \in \mathbb{R}^d$ | Low-d Vector | Embedded coordinate in visualization space (typically $d = 2$ or $3$) |
| $p_{j|i} \in [0, 1]$ | Conditional probability | Probability that $x_i$ picks $x_j$ as its neighbor under a Gaussian centered at $x_i$ |
| $\sigma_i$ | $\mathbb{R}_{>0}$ | Bandwidth of the Gaussian centered at $x_i$, tuned to match target **Perplexity** |
| $\text{Perp}(P_i) = 2^{H(P_i)}$ | Positive scalar | Effective number of nearest neighbors around $x_i$ ($H$ is Shannon entropy) |
| $p_{ij} \in [0, 1]$ | Joint probability | Symmetrized high-dimensional affinity: $p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$ |
| $q_{ij} \in [0, 1]$ | Joint probability | Low-dimensional affinity modeled by Student-t distribution (1 degree of freedom) |
| $D_{\text{KL}}(P \parallel Q)$ | $\mathbb{R}_{\ge 0}$ | Kullback-Leibler divergence objective minimized by t-SNE |
| $\rho_i$ | $\mathbb{R}_{\ge 0}$ | Distance from $x_i$ to its 1st nearest neighbor in UMAP (ensuring local connectivity) |
| $C_{\text{UMAP}}(P, Q)$ | $\mathbb{R}_{\ge 0}$ | Fuzzy set cross-entropy loss function minimized by UMAP |

---

## 3. The Curse of Dimensionality & The Crowding Problem

### 3.1 Distance Concentration in High Dimensions
In high-dimensional space $\mathbb{R}^D$, the volume of a sphere of radius $R$ is proportional to $R^D$. As $D \to \infty$, almost the entire volume of a sphere is concentrated in a paper-thin shell at its outer boundary.

Consequently, pairwise Euclidean distances become uniform:
$$\lim_{D \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} \to 0$$
Points lose distinct relative nearest-neighbor hierarchies when measured with raw Euclidean distances.

### 3.2 The Crowding Problem
Suppose 10 points are mutually equidistant in 10-dimensional space (the vertices of a regular 9-simplex). If we attempt to map these 10 points to a 2D plane, it is geometrically impossible for all 10 points to remain equidistant. In 2D, the area available to place neighbors at distance $r$ grows as $2\pi r$, whereas in 10D, the surface area grows as $S_{9} r^9$.

If we use a Gaussian distribution in both high and low dimensions (as in original SNE), the low-dimensional space becomes violently overcrowded: points from distinct clusters are forced together, destroying cluster boundaries.

---

## 4. t-Distributed Stochastic Neighbor Embedding (t-SNE)

Laurens van der Maaten and Geoffrey Hinton (2008) solved the crowding problem by substituting a **heavy-tailed Student-t distribution** in the low-dimensional embedding space.

### 4.1 High-Dimensional Gaussian Affinities
The conditional probability that $x_i$ picks $x_j$ as its neighbor is proportional to their Gaussian density:
$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \ne i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}, \quad p_{i|i} = 0$$

The bandwidth $\sigma_i$ is determined via binary search such that the Shannon entropy matches the user-specified **Perplexity**:
$$\text{Perp}(P_i) = 2^{H(P_i)}, \quad \text{where } H(P_i) = -\sum_{j \ne i} p_{j|i} \log_2 p_{j|i}$$
Typically, perplexity is chosen between 5 and 50.

To ensure robustness against outliers, probabilities are **symmetrized**:
$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}, \quad \text{ensuring } \sum_{i, j} p_{ij} = 1 \text{ and } p_{ij} = p_{ji}$$

---

### 4.2 Low-Dimensional Student-t Distribution
In the low-dimensional space $\mathbb{R}^2$, pairwise affinities $q_{ij}$ are defined using the Cauchy distribution (Student-t with 1 degree of freedom):
$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \ne l} (1 + \|y_k - y_l\|^2)^{-1}}, \quad q_{ii} = 0$$

```
              GAUSSIAN VS STUDENT-T: THE HEAVY TAIL EFFECT
       Density ^
               │       /\  <-- Gaussian (drops exponentially: e^{-d^2})
               │      /  \
               │     /    \
               │    /      \________  <-- Student-t (drops as 1 / (1 + d^2))
               └──────────────────────> Distance d
       Student-t allows moderate high-d distances to map to MUCH larger 
       low-d distances without incurring extreme probability penalties!
```

---

### 4.3 Objective Function: Kullback-Leibler Divergence
t-SNE minimizes the asymmetric KL divergence from $P$ to $Q$:
$$\mathcal{L}_{\text{t-SNE}} = D_{\text{KL}}(P \parallel Q) = \sum_{i \ne j} p_{ij} \ln \frac{p_{ij}}{q_{ij}}$$

Notice that:
- If $p_{ij}$ is large (nearby points in high-d) and $q_{ij}$ is small (far apart in low-d), the penalty $p_{ij} \ln(p_{ij}/q_{ij})$ is **enormous**.
- If $p_{ij}$ is small (distant points in high-d) and $q_{ij}$ is large, the penalty is small.
- **Result:** t-SNE focuses obsessively on preserving **local neighborhood structure**.

---

### 4.4 Complete Derivation of the t-SNE Analytical Gradient

Let $d_{ij} = \|y_i - y_j\|_2$. Define the unnormalized Student-t kernel:
$$w_{ij} = (1 + \|y_i - y_j\|^2)^{-1}, \quad Z = \sum_{k \ne l} w_{kl} \implies q_{ij} = \frac{w_{ij}}{Z}$$

The loss is:
$$\mathcal{L} = \sum_{j \ne k} p_{jk} \ln p_{jk} - \sum_{j \ne k} p_{jk} \ln w_{jk} + \sum_{j \ne k} p_{jk} \ln Z = \sum_{j \ne k} p_{jk} \ln p_{jk} - \sum_{j \ne k} p_{jk} \ln w_{jk} + \ln Z$$

Taking the derivative with respect to coordinate vector $y_i$:
$$\frac{\partial \mathcal{L}}{\partial y_i} = -\sum_{j \ne k} p_{jk} \frac{1}{w_{jk}} \frac{\partial w_{jk}}{\partial y_i} + \frac{1}{Z} \frac{\partial Z}{\partial y_i}$$

Note that:
$$\frac{\partial w_{ij}}{\partial y_i} = -2(1 + \|y_i - y_j\|^2)^{-2} (y_i - y_j) = -2 w_{ij}^2 (y_i - y_j)$$

Since $w_{jk}$ only depends on $y_i$ when $j = i$ or $k = i$, and $w_{ij} = w_{ji}$:
$$\frac{\partial Z}{\partial y_i} = \sum_{j \ne i} 2 \frac{\partial w_{ij}}{\partial y_i} = -4 \sum_{j \ne i} w_{ij}^2 (y_i - y_j)$$

Substituting into the gradient:
$$\frac{\partial \mathcal{L}}{\partial y_i} = -2 \sum_{j \ne i} p_{ij} \frac{1}{w_{ij}} \left( -2 w_{ij}^2 (y_i - y_j) \right) + \frac{1}{Z} \left( -4 \sum_{j \ne i} w_{ij}^2 (y_i - y_j) \right)$$
$$\frac{\partial \mathcal{L}}{\partial y_i} = 4 \sum_{j \ne i} p_{ij} w_{ij} (y_i - y_j) - 4 \sum_{j \ne i} \left( \frac{w_{ij}}{Z} \right) w_{ij} (y_i - y_j)$$

Recalling that $q_{ij} = \frac{w_{ij}}{Z}$:
$$\frac{\partial \mathcal{L}}{\partial y_i} = 4 \sum_{j \ne i} (p_{ij} - q_{ij}) w_{ij} (y_i - y_j) = 4 \sum_{j \ne i} (p_{ij} - q_{ij})(1 + \|y_i - y_j\|^2)^{-1} (y_i - y_j)$$

> **Physical Interpretation of Gradient:**
> The gradient acts as physical springs between points:
> 1. **Attractive Force ($p_{ij} > q_{ij}$):** If two points are neighbors in high-d ($p_{ij} > 0$) but separated in 2D ($q_{ij} \approx 0$), an attractive spring pulls them together.
> 2. **Repulsive Force ($q_{ij} > p_{ij}$):** If two distant points drift too close in 2D ($q_{ij} > p_{ij}$), an electrostatic repulsion pushes them apart.

---

## 5. Uniform Manifold Approximation and Projection (UMAP)

While t-SNE is brilliant for visualization, it has two drawbacks:
1. It does not preserve global distances between distinct clusters.
2. Computing the partition function $Z = \sum_{k \ne l} w_{kl}$ scales as $\mathcal{O}(N^2)$, making it slow for massive datasets.

**UMAP** (McInnes, Healy, Melville, 2018) resolves these limitations using Riemannian geometry and fuzzy simplicial sets.

### 5.1 High-Dimensional Fuzzy Simplicial Sets
UMAP constructs a weighted k-nearest neighbor graph where edge weights represent the probability that a simplex exists:
$$p_{i|j} = \exp\left( -\frac{\max(0, d(x_i, x_j) - \rho_i)}{\sigma_i} \right)$$
where $\rho_i$ is the distance to the nearest neighbor of $x_i$ (ensuring the manifold is locally connected: $p_{i|i+1} = 1$).

Symmetrization uses the algebraic t-conorm (fuzzy set union):
$$p_{ij} = p_{i|j} + p_{j|i} - p_{i|j} p_{j|i}$$

### 5.2 Low-Dimensional Affinities & Cross-Entropy Loss
In the low-dimensional embedding, affinities are:
$$q_{ij} = \left( 1 + a \|y_i - y_j\|^{2b} \right)^{-1}$$
where $a, b$ are hyperparameters chosen to fit a target minimum distance (`min_dist`).

Instead of KL divergence, UMAP optimizes the **Fuzzy Set Cross-Entropy**:
$$\mathcal{L}_{\text{UMAP}} = \sum_{i \ne j} \left[ \underbrace{p_{ij} \ln \frac{p_{ij}}{q_{ij}}}_{\text{Attraction}} + \underbrace{(1 - p_{ij}) \ln \frac{1 - p_{ij}}{1 - q_{ij}}}_{\text{Repulsion}} \right]$$

> **Why UMAP Preserves Global Structure:**
> The second term $(1 - p_{ij}) \ln \frac{1 - p_{ij}}{1 - q_{ij}}$ explicitly penalizes placing non-neighbors ($p_{ij} = 0$) close to each other in low dimensions ($q_{ij} > 0$). This global repulsive term maintains the macro-scale arrangement of clusters.

---

## 6. Algorithmic Comparison

| Criterion | PCA | t-SNE | UMAP |
| :--- | :--- | :--- | :--- |
| **Mapping Type** | Linear projection | Non-linear probabilistic | Non-linear topological |
| **Local vs Global** | Maximizes global variance | Preserves local neighborhoods | Preserves local + global |
| **Computational Complexity** | $\mathcal{O}(N d^2 + d^3)$ | $\mathcal{O}(N^2)$ ($\mathcal{O}(N \log N)$ Barnes-Hut) | $\mathcal{O}(N \log N)$ (Nearest Neighbor Descent) |
| **Objective Function** | Trace($W^T C W$) | KL Divergence $D_{\text{KL}}(P \parallel Q)$ | Fuzzy Cross-Entropy |
| **Parametric Model?** | Yes ($Y = X W$) | No (transductive, optimizes $Y$) | Semi-parametric (can transform new data) |
| **Hyperparameter Sensitivity** | None ($k$ components) | High (Perplexity, Learning Rate) | Moderate (`n_neighbors`, `min_dist`) |

---

## 7. Summary & Next Steps

When datasets contain curved manifolds, linear projections fail. **t-SNE** uses Gaussian neighbor probabilities and heavy-tailed Student-t distributions to eliminate the crowding problem, creating stunning visual representations of cluster neighborhoods. **UMAP** uses fuzzy simplicial sets and cross-entropy to preserve both fine-grained clusters and macro-scale global geometries at lightning speed.

In **Topic 9.4**, we will investigate **Density Estimation & Anomaly Detection** using Kernel Density Estimation (KDE), Mahalanobis distances, and Isolation Forests.
