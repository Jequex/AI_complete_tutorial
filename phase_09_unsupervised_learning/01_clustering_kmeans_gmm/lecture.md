# Topic 9.1: Clustering — K-Means, Gaussian Mixture Models, & Beyond

---

## 1. Executive Summary & The 12-Year-Old Analogy

In supervised learning, every data point arrives with a target label or teacher telling the model what it is looking at. But in the real world, the vast majority of data is **unlabeled**: customer transaction histories, medical sensor telemetry, genomic sequences, image libraries, and raw text. 

**Clustering** is the foundational task of **Unsupervised Learning**: discovering hidden group structures, natural affinities, and latent partitions inside unannotated data points using only their geometric positions and statistical distributions.

```
       UNSUPERVISED CLUSTERING: DISCOVERING HIDDEN ATTRACTIONS
       
    y ^
      │       * * *                      # # # #
      │     * Cluster 1 *              # Cluster 2 #
      │       * * *                      # # # #
      │
      │                     @ @ @ @
      │                   @ Cluster 3 @
      │                     @ @ @ @
      └────────────────────────────────────────> x
       Data points possess no labels: the algorithm must discover 
       boundaries using geometry (distances) or probability density!
```

---

### The 12-Year-Old Analogy: Sorting Mixed Marbles & Secret Clubs

Imagine you dump a huge bag containing 300 mystery marbles onto your bedroom carpet:

1. **The K-Means Approach (Sorting Marbles by Bins):**
   - You put 3 empty colored cardboard boxes on the carpet.
   - **Step 1 (Assignment):** You pick up every marble and drop it into whichever box is closest to where it landed.
   - **Step 2 (Centroid Update):** You look at all the marbles inside the red box, calculate their average physical center, and slide the red box directly into that center.
   - You repeat this dance until the boxes stop moving!
   - *Limitation:* Every marble is forced 100% into exactly one box. If a marble sits directly halfway between red and blue, K-means forces a hard, arbitrary choice.

2. **The Gaussian Mixture Models & EM Approach (The Secret Clubs):**
   - Instead of rigid wooden boxes, imagine 3 friendly secret clubs on campus (The Chess Club, The Art Studio, The Robotics Squad).
   - Each club has an "epicenter" where members gather, but clubs have diffuse, fuzzy boundaries (some clubs are tight circles, others are stretched into oblong ovals).
   - When a new student arrives, they don't belong 100% to one club. Instead, they receive a **membership badge** with percentages: *"You are 70% Robotics, 25% Chess, and 5% Art."*
   - **Expectation-Maximization (EM):**
     - **E-Step (Guessing Memberships):** Based on the current club locations and shapes, calculate every student's membership percentages (responsibilities).
     - **M-Step (Refining Club Profiles):** Based on everyone's membership percentages, recalculate each club's center, size, and orientation to best embrace its community.

3. **The DBSCAN Approach (Neighborhood Block Parties):**
   - You don't guess the number of clubs in advance! Instead, you say: *"If at least 5 people live within 50 feet of each other, that's an active party (core neighborhood)."*
   - Parties merge if their neighborhoods touch. Anyone standing alone in an empty field miles away is labeled as **noise / outlier**.

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $\mathcal{D} = \{x_1, \dots, x_N\}$ | Set in $\mathbb{R}^d$ | Unlabeled dataset of $N$ observations in $d$-dimensional continuous space |
| $K$ | $\mathbb{Z}_{\ge 1}$ | Number of clusters or mixture components |
| $\mu_k \in \mathbb{R}^d$ | Vector | Center (centroid / mean vector) of cluster $k$ |
| $r_{ik} \in \{0, 1\}$ | Binary indicator | Hard assignment indicator ($r_{ik} = 1$ if $x_i$ belongs to cluster $k$, else $0$) |
| $J$ or $\mathcal{I}$ | $\mathbb{R}_{\ge 0}$ | K-Means objective function (**Inertia** / Within-Cluster Sum of Squares) |
| $z_i \in \{1, \dots, K\}$ | Categorical latent variable | Unobserved cluster identity generating observation $x_i$ |
| $\pi_k = P(z = k)$ | Simplex $\sum \pi_k = 1$ | Prior probability (mixture weight) of component $k$ |
| $\Sigma_k \in \mathbb{R}^{d \times d}$ | Symmetric positive definite | Covariance matrix of Gaussian component $k$ (shape, orientation, scale) |
| $\mathcal{N}(x; \mu_k, \Sigma_k)$ | Density $\mathbb{R}^d \to \mathbb{R}_{>0}$ | Multivariate Gaussian probability density function |
| $\gamma_{ik} \equiv P(z_i = k \mid x_i)$ | $[0, 1]$ | **Responsibility**: posterior probability that component $k$ generated $x_i$ |
| $N_k = \sum_{i=1}^N \gamma_{ik}$ | $\mathbb{R}_{>0}$ | Effective number of data points assigned to component $k$ |
| $\ln p(X \mid \theta)$ | $\mathbb{R}$ | Marginal log-likelihood of the observed dataset under the mixture model |
| $\epsilon, \text{MinPts}$ | $\mathbb{R}_{>0}, \mathbb{Z}_{\ge 1}$ | Neighborhood radius and minimum density threshold for DBSCAN |

---

## 3. K-Means Clustering & Lloyd's Algorithm

### 3.1 Objective Function: Within-Cluster Sum of Squares (Inertia)

Given an unlabelled dataset $\{x_i\}_{i=1}^N \subset \mathbb{R}^d$, K-Means partitions the $N$ observations into $K$ disjoint subsets $C_1, C_2, \dots, C_K$ such that the total within-cluster sum of squared Euclidean distances (known as **Inertia** or Distortion) is minimized:

$$J(r, \mu) = \sum_{i=1}^N \sum_{k=1}^K r_{ik} \|x_i - \mu_k\|_2^2$$

subject to the 1-of-$K$ assignment constraint:
$$\sum_{k=1}^K r_{ik} = 1 \quad \forall i, \quad r_{ik} \in \{0, 1\}$$

### 3.2 Lloyd's Alternating Optimization Algorithm

Minimizing $J(r, \mu)$ over both discrete indicators $\{r_{ik}\}$ and continuous centroids $\{\mu_k\}$ simultaneously is NP-hard. **Lloyd's Algorithm** (1957/1982) is a coordinate descent algorithm that alternates between two sub-problems:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        LLOYD'S ALGORITHM CYCLE                         │
│                                                                        │
│   1. ASSIGNMENT STEP (Voronoi Partitioning):                           │
│      Fix centroids μ_k. Assign each point x_i to the closest center:   │
│      r_{ik} = 1  if k = argmin_j ||x_i - μ_j||^2,  else 0              │
│                                                                        │
│                               │                                        │
│                               ▼                                        │
│                                                                        │
│   2. CENTROID UPDATE STEP (Center of Mass):                            │
│      Fix assignments r_{ik}. Solve ∂J/∂μ_k = 0:                        │
│      μ_k = (sum_{i} r_{ik} x_i) / (sum_{i} r_{ik})                     │
│                                                                        │
│                               │                                        │
│                     Has J converged? (Shift < ε)                       │
│                       NO ──> Loop to Step 1                            │
│                       YES ─> Halt                                      │
└────────────────────────────────────────────────────────────────────────┘
```

#### Mathematical Proof of the Update Step:
Taking the gradient of $J$ with respect to centroid $\mu_k$:
$$\nabla_{\mu_k} J = \nabla_{\mu_k} \sum_{i=1}^N r_{ik} (x_i - \mu_k)^T (x_i - \mu_k) = -2 \sum_{i=1}^N r_{ik} (x_i - \mu_k)$$

Setting the gradient to zero:
$$-2 \sum_{i=1}^N r_{ik} (x_i - \mu_k) = 0 \implies \sum_{i=1}^N r_{ik} x_i = \left(\sum_{i=1}^N r_{ik}\right) \mu_k$$

$$\mu_k^* = \frac{\sum_{i=1}^N r_{ik} x_i}{\sum_{i=1}^N r_{ik}} = \frac{1}{|C_k|} \sum_{x \in C_k} x$$

At each step, $J(r, \mu)$ is non-increasing. Because there are only $K^N$ possible discrete assignments, the algorithm is guaranteed to converge to a local minimum in a finite number of iterations.

### 3.3 The K-Means++ Initialization Guarantee

Standard K-Means with uniform random centroid initialization frequently gets trapped in sub-optimal local minima. 

Arthur & Vassilvitskii (2007) introduced **K-Means++**, which seeds initial centroids according to a distance-squared probability distribution:

1. Select the first centroid $\mu_1$ uniformly at random from $\{x_1, \dots, x_N\}$.
2. For each remaining point $x_i$, compute the squared distance to the nearest already-selected centroid:
   $$D(x_i) = \min_{j < k} \|x_i - \mu_j\|_2$$
3. Choose the next centroid $\mu_k = x_i$ with probability:
   $$P(x_i) = \frac{D(x_i)^2}{\sum_{m=1}^N D(x_m)^2}$$
4. Repeat Steps 2–3 until $K$ centroids have been selected.

> **Theoretical Guarantee:** K-Means++ guarantees an expected approximation ratio of $\mathbb{E}[J] \le 8(\ln K + 2) J_{\text{optimal}}$, fundamentally avoiding catastrophic cluster collisions.

---

## 4. Gaussian Mixture Models (GMM) & Expectation-Maximization (EM)

While K-Means assumes spherical, isotropic clusters with hard binary assignments, **Gaussian Mixture Models (GMM)** generalize clustering to flexible ellipsoids with soft, probabilistic responsibilities.

### 4.1 Generative Story & Probabilistic Formulation

We model the marginal distribution $p(x)$ as a weighted sum of $K$ multivariate Gaussian components:
$$p(x \mid \theta) = \sum_{k=1}^K \pi_k \mathcal{N}(x; \mu_k, \Sigma_k)$$

where:
- $\pi_k = P(z = k)$ are the mixture weights satisfying $\pi_k \ge 0$ and $\sum_{k=1}^K \pi_k = 1$.
- Each component density is parameterized by mean $\mu_k \in \mathbb{R}^d$ and positive definite covariance $\Sigma_k \in \mathbb{R}^{d \times d}$:
$$\mathcal{N}(x; \mu_k, \Sigma_k) = \frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}} \exp\left( -\frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) \right)$$

The observed data log-likelihood across all $N$ independent samples is:
$$\ln p(X \mid \theta) = \sum_{i=1}^N \ln \left( \sum_{k=1}^K \pi_k \mathcal{N}(x_i; \mu_k, \Sigma_k) \right)$$

Because the sum sits **inside** the logarithm, there is no closed-form analytical solution for $\nabla_\theta \ln p(X \mid \theta) = 0$. We must turn to **Expectation-Maximization (EM)**.

### 4.2 Complete Derivation of the EM Algorithm

Let $Z = \{z_1, \dots, z_N\}$ denote the latent variables indicating which mixture component generated each $x_i$.

#### 1. The Evidence Lower Bound (ELBO):
Using Jensen's inequality for any distribution $q(Z)$ over the latent variables:
$$\ln p(X \mid \theta) = \sum_Z q(Z) \ln \frac{p(X, Z \mid \theta)}{q(Z)} + D_{\text{KL}}(q(Z) \parallel p(Z \mid X, \theta))$$

Since $D_{\text{KL}} \ge 0$, the Evidence Lower Bound is:
$$\mathcal{L}(q, \theta) = \sum_Z q(Z) \ln \frac{p(X, Z \mid \theta)}{q(Z)} \le \ln p(X \mid \theta)$$

Equality holds when $q(Z) = p(Z \mid X, \theta)$.

#### 2. The E-Step (Expectation):
We evaluate the posterior responsibility $\gamma_{ik} \equiv P(z_i = k \mid x_i, \theta^{\text{old}})$ using Bayes' Theorem:
$$\gamma_{ik} = \frac{P(z_i = k) p(x_i \mid z_i = k, \theta^{\text{old}})}{\sum_{j=1}^K P(z_i = j) p(x_i \mid z_i = j, \theta^{\text{old}})} = \frac{\pi_k \mathcal{N}(x_i; \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i; \mu_j, \Sigma_j)}$$

#### 3. The M-Step (Maximization):
We maximize the expected complete-data log-likelihood:
$$Q(\theta, \theta^{\text{old}}) = \sum_{i=1}^N \sum_{k=1}^K \gamma_{ik} \left[ \ln \pi_k + \ln \mathcal{N}(x_i; \mu_k, \Sigma_k) \right]$$

- **Updating Means $\mu_k$:**
$$\nabla_{\mu_k} Q = \sum_{i=1}^N \gamma_{ik} \Sigma_k^{-1} (x_i - \mu_k) = 0 \implies \mu_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} x_i$$
where $N_k \equiv \sum_{i=1}^N \gamma_{ik}$ is the effective count of points belonging to cluster $k$.

- **Updating Covariances $\Sigma_k$:**
$$\Sigma_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} (x_i - \mu_k^{\text{new}})(x_i - \mu_k^{\text{new}})^T$$

- **Updating Mixture Weights $\pi_k$ (using Lagrange Multipliers for $\sum \pi_k = 1$):**
Form the Lagrangian $\Lambda(\pi, \lambda) = \sum_{i=1}^N \sum_{k=1}^K \gamma_{ik} \ln \pi_k + \lambda \left(1 - \sum_{k=1}^K \pi_k\right)$.
Taking the derivative with respect to $\pi_k$:
$$\frac{\partial \Lambda}{\partial \pi_k} = \sum_{i=1}^N \frac{\gamma_{ik}}{\pi_k} - \lambda = 0 \implies N_k = \lambda \pi_k$$
Summing over all $k$: $\sum_{k=1}^K N_k = N = \lambda \sum_{k=1}^K \pi_k = \lambda \implies \lambda = N$.
Therefore:
$$\pi_k^{\text{new}} = \frac{N_k}{N}$$

---

## 5. Density-Based & Hierarchical Clustering

### 5.1 DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

While K-Means and GMM assume convex/elliptical shapes, **DBSCAN** (Ester et al., 1996) discovers clusters of **arbitrary shape** (concentric circles, spirals) based on spatial density:

- **$\epsilon$-Neighborhood:** $N_\epsilon(x) = \{x' \in \mathcal{D} \mid \|x - x'\| \le \epsilon\}$.
- **Core Point:** Any point with $|N_\epsilon(x)| \ge \text{MinPts}$.
- **Border Point:** A non-core point that lies within the $\epsilon$-neighborhood of a core point.
- **Noise Point:** Any point that is neither core nor border (outlier).
- **Density-Reachable:** A point $p$ is density-reachable from $q$ if there is a chain of core points connecting them.

```
       DBSCAN POINT CLASSIFICATION
       
         * (Core) ── ε ── * (Core)
        /                     \
       * (Core)                * (Border)
                                   
                                      x (Noise / Outlier)
```

### 5.2 Hierarchical Agglomerative Clustering (HAC)

HAC constructs a bottom-up tree (**dendrogram**) of nested clusters without choosing $K$ up front:
1. Initialize each data point as its own singleton cluster: $C_i = \{x_i\}$.
2. Compute pairwise distance matrix between all clusters.
3. Merge the two clusters with the minimum linkage distance.
4. Repeat until a single root cluster remains.

#### Linkage Criteria:
- **Single Linkage (Minimum Distance):** $D(A, B) = \min_{a \in A, b \in B} d(a, b)$ *(Prone to chaining)*.
- **Complete Linkage (Maximum Distance):** $D(A, B) = \max_{a \in A, b \in B} d(a, b)$ *(Produces compact, spherical clusters)*.
- **Average Linkage (UPGMA):** $D(A, B) = \frac{1}{|A||B|} \sum_{a \in A} \sum_{b \in B} d(a, b)$.
- **Ward's Linkage:** Merges clusters that minimize the increase in total within-cluster variance.

---

## 6. Algorithmic Comparison

| Criterion | K-Means | Gaussian Mixture Models | DBSCAN | Hierarchical (HAC) |
| :--- | :--- | :--- | :--- | :--- |
| **Cluster Geometry** | Convex, spherical Voronoi | Flexible ellipsoids | Arbitrary manifolds | Any geometry (depends on linkage) |
| **Assignment Type** | Hard ($0$ or $1$) | Soft posterior ($\gamma_{ik} \in [0, 1]$) | Hard partition + Noise flag | Hierarchical tree |
| **Specifies $K$ upfront?** | Yes | Yes | No ($\epsilon, \text{MinPts}$) | No (cut dendrogram at threshold) |
| **Outlier Handling** | Severe pull on centroids | Influences covariance | Explicit Noise class | Isolated leaf nodes |
| **Computational Complexity** | $\mathcal{O}(N K d \cdot \text{iters})$ | $\mathcal{O}(N K d^2 \cdot \text{iters})$ | $\mathcal{O}(N \log N)$ (with spatial trees) | $\mathcal{O}(N^2)$ to $\mathcal{O}(N^3)$ |

---

## 7. Summary & Next Steps

Clustering allows machines to uncover the natural taxonomy of the world without human supervision. When clusters are compact and spherical, **K-Means** offers blisteringly fast convergence. When clusters overlap and exhibit directional variance, **GMM with Expectation-Maximization** provides rigorous probabilistic modeling. When geometries are non-linear or contaminated with heavy noise, **DBSCAN** extracts dense manifolds without pre-specifying $K$.

In **Topic 9.2**, we will examine how to compress hyper-dimensional spaces while preserving maximum variance using **Principal Component Analysis (PCA)** and **Singular Value Decomposition (SVD)**.
