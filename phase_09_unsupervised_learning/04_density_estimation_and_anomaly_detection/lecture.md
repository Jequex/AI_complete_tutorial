# Topic 9.4: Density Estimation & Anomaly Detection

---

## 1. Executive Summary & The 12-Year-Old Analogy

In unsupervised learning, we often need to answer two fundamental questions about incoming data without having explicit class labels:
1. **What is the continuous probability distribution that produced this data?** (**Density Estimation**)
2. **Does this new observation belong to the normal process, or is it a bizarre, potentially fraudulent or dangerous outlier?** (**Anomaly Detection**)

From detecting credit card fraud and industrial sensor failures to medical tumor screening and generative modeling, density estimation and anomaly detection provide the statistical bedrock for identifying the unexpected.

```
                  THE SPECTRUM OF NORMALITY VS ANOMALIES
       Density ^
               │               /\  <-- High density mode (Common / Normal)
               │              /  \
               │             /    \
               │            /      \       /\  <-- Secondary mode
               │           /        \_____/  \
               │          /                   \
               └─────────┴─────────────────────┴─────────────*──> Feature x
                                                        Anomaly / Outlier
                                                        (Extremely low density)
```

---

### The 12-Year-Old Analogy: The Airport Scanner Finding Anomalous Baggage

Imagine you are an airport security agent inspecting thousands of passenger suitcases passing through an X-ray scanner:

- **Kernel Density Estimation (KDE) — The Smooth Probability Heatmap:**
  - Instead of guessing a rigid formula (like assuming every suitcase weighs exactly 40 lbs following a perfect bell curve), you place a small "fuzzy glow" (a Gaussian kernel) over every single suitcase you have ever weighed.
  - Where hundreds of bags weigh between 35 and 45 lbs, their glowing clouds overlap into a giant bright sun (high probability density).
  - Where a few small duffel bags weigh 15 lbs, a smaller glowing moon appears (secondary mode).
  - You now have a smooth, continuous map of suitcase weights without making rigid parametric assumptions!
  
- **Mahalanobis Distance — The Correlated Dimension Scanner:**
  - Suppose normal suitcases are either **small and light** (carry-ons: 20 inches tall, 15 lbs) or **large and heavy** (checked luggage: 32 inches tall, 50 lbs). Height and weight are strongly positively correlated.
  - Now a suitcase arrives that is **32 inches tall, but weighs only 2 lbs!**
  - If you look at height alone: 32 inches is completely normal!
  - If you look at weight alone: 2 lbs is within range of a backpack!
  - If you measure standard Euclidean distance from the average suitcase: it looks close!
  - But **Mahalanobis Distance** measures distance *relative to the correlation ellipse*. It immediately sets off the alarm: *"A huge suitcase with zero weight is hollow and suspicious!"*

- **Isolation Forest — The Quick Scissors:**
  - Imagine you have a giant dartboard covered in 1,000 normal inlier dots clustered tightly together, plus 1 lone outlier dot sitting isolated in the top-right corner.
  - You play a game where you take a pair of scissors and make completely random horizontal and vertical cuts across the board.
  - To isolate the lone outlier dot by itself, you only need **1 or 2 cuts**!
  - To isolate a single dot hidden inside the crowded cluster of 1,000 dots, you will need **dozens of tiny cuts**!
  - **Isolation Forest rule:** Anything that isolates quickly near the top of the tree is an anomaly.

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $\mathcal{D} = \{x_1, \dots, x_N\}$ | Sample in $\mathbb{R}^d$ | Dataset of $N$ observations drawn i.i.d. from unknown density $f(x)$ |
| $\hat{f}(x)$ | $\mathbb{R}^d \to \mathbb{R}_{\ge 0}$ | Non-parametric Kernel Density Estimate (KDE) |
| $K(u)$ | $\mathbb{R}^d \to \mathbb{R}_{\ge 0}$ | Kernel function (symmetric, integrates to 1, zero mean) |
| $h \in \mathbb{R}_{>0}$ | Smoothing parameter | **Bandwidth** of the kernel window, governing the bias-variance trade-off |
| $h_{\text{opt}}$ | Optimal bandwidth | Silverman's rule of thumb bandwidth minimizing asymptotic MISE |
| $\mu \in \mathbb{R}^d$ | Vector | Population or empirical sample mean vector |
| $\Sigma \in \mathbb{R}^{d \times d}$ | SPD Matrix | Sample covariance matrix of inlier data |
| $D_M(x)$ | $\mathbb{R}_{\ge 0}$ | **Mahalanobis distance** from observation $x$ to center $\mu$ |
| $\chi_d^2$ | Distribution | Chi-squared distribution with $d$ degrees of freedom (distribution of $D_M^2$) |
| $h(x)$ | Integer $\ge 0$ | Tree isolation depth (path length from root to leaf isolating $x$) |
| $c(n)$ | $\mathbb{R}_{>0}$ | Average path length of unsuccessful search in a Binary Search Tree of $n$ nodes |
| $s(x, n) \in [0, 1]$ | Score | Normalized Isolation Forest anomaly score ($s \to 1$ implies anomaly) |

---

## 3. Kernel Density Estimation (KDE)

### 3.1 Non-Parametric vs Parametric Density Estimation
In parametric estimation (like a single Gaussian $\mathcal{N}(\mu, \sigma^2)$), we assume a rigid functional family. If the true underlying distribution is multimodal, skewed, or bounded, parametric models suffer severe model misspecification.

**Kernel Density Estimation (Parzen-Rosenblatt Window)** is a non-parametric approach that lets the empirical data define the probability density function directly:

$$\hat{f}(x) = \frac{1}{N h^d} \sum_{i=1}^N K\left( \frac{x - x_i}{h} \right)$$

### 3.2 Requirements for a Valid Kernel
For $\hat{f}(x)$ to be a valid probability density function ($\hat{f}(x) \ge 0$ and $\int \hat{f}(x) dx = 1$), the kernel function $K(u)$ must satisfy:
1. Non-negativity: $K(u) \ge 0$ for all $u \in \mathbb{R}^d$.
2. Normalization: $\int_{\mathbb{R}^d} K(u) \, du = 1$.
3. Symmetry & Zero Mean: $\int_{\mathbb{R}^d} u K(u) \, du = 0$.
4. Finite Variance: $\sigma_K^2 = \int_{\mathbb{R}^d} u^2 K(u) \, du < \infty$.

#### Common Kernels:
- **Gaussian Kernel (Smooth, infinite support):**
  $$K(u) = \frac{1}{(2\pi)^{d/2}} \exp\left( -\frac{1}{2} \|u\|_2^2 \right)$$
- **Epanechnikov Kernel (Optimal in Mean Integrated Squared Error):**
  $$K(u) = \frac{3}{4}(1 - u^2) \cdot \mathbb{I}(|u| \le 1)$$
- **Tophat / Boxcar Kernel (Uniform within window):**
  $$K(u) = \frac{1}{2} \cdot \mathbb{I}(|u| \le 1)$$

---

### 3.3 The Bandwidth $h$ & Silverman's Rule of Thumb
The bandwidth $h$ acts as a smoothing scale. It controls the fundamental **bias-variance trade-off**:
- **Undersmoothing ($h \to 0$):** High variance, low bias. The density degenerates into a spiky sum of Dirac delta spikes centered on training points.
- **Oversmoothing ($h \to \infty$):** High bias, low variance. True multimodal peaks are wiped out into a single flat blob.

```
       EFFECT OF BANDWIDTH h ON ESTIMATED DENSITY
       
       Small h (Undersmoothed):        Large h (Oversmoothed):
            /\    /\    /\                     ╭──────────╮
           /  \  /  \  /  \                   │            │
          /    \/    \/    \                  │            │
       Spiky, memorizes noise!          Oversmoothed, modes erased!
```

To find the optimal $h$, we minimize the **Mean Integrated Squared Error (MISE)**:
$$\text{MISE}(h) = \mathbb{E}\left[ \int (\hat{f}(x) - f(x))^2 dx \right] = \text{Asymptotic Bias}^2 + \text{Asymptotic Variance}$$

For a 1D Gaussian target distribution, the exact theoretical minimizer yields **Silverman's Rule of Thumb (1986)**:
$$h_{\text{opt}} = \left( \frac{4 \hat{\sigma}^5}{3 N} \right)^{1/5} \approx 1.06 \, \hat{\sigma} \, N^{-1/5}$$

To make the estimate robust against extreme outliers, Silverman recommended using the interquartile range (IQR):
$$h_{\text{robust}} = 0.9 \cdot \min\left( \hat{\sigma}, \frac{\text{IQR}}{1.34} \right) N^{-1/5}$$

---

## 4. Anomaly Detection via Mahalanobis Distance

### 4.1 Why Euclidean Distance Fails
Standard Euclidean distance $D_E(x, \mu) = \sqrt{(x - \mu)^T (x - \mu)}$ assumes all features have identical unit variance and zero correlation (spherical iso-contours). 

In real-world data, features have different scales and strong covariance. An observation can be within the 1D range of every single feature, yet severely violate the multivariate covariance structure!

```
       EUCLIDEAN DISTANCE VS MAHALANOBIS DISTANCE
       
    x2 ^
       │               * * *   / (Correlation Axis)
       │             * * * *  /
       │           * * * * * /
       │          * * * * * /
       │         * * * *   /
       │                  /  
       │           ! (Anomaly) <── Small Euclidean distance from center,
       │                           BUT violates correlation! Huge D_M!
       └────────────────────────────────────────> x1
```

### 4.2 Mathematical Formulation of Mahalanobis Distance
The **Mahalanobis Distance** scales displacements by the inverse covariance matrix:
$$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

where:
- $\mu = \frac{1}{N} \sum_{i=1}^N x_i \in \mathbb{R}^d$ is the centroid.
- $\Sigma = \frac{1}{N-1} \sum_{i=1}^N (x_i - \mu)(x_i - \mu)^T \in \mathbb{R}^{d \times d}$ is the covariance matrix.

#### The Whitening Transformation:
Let $\Sigma = V \Lambda V^T$ be the spectral decomposition of $\Sigma$, and define $\Sigma^{-1/2} = V \Lambda^{-1/2} V^T$.
The whitened coordinate transformation is:
$$\tilde{x} = \Sigma^{-1/2} (x - \mu)$$

The covariance of $\tilde{x}$ is the identity matrix $I_d$, and the Mahalanobis distance is simply the standard Euclidean distance in the whitened space:
$$D_M(x) = \|\tilde{x}\|_2 = \sqrt{\tilde{x}^T \tilde{x}} = \sqrt{(x - \mu)^T \Sigma^{-1/2} \Sigma^{-1/2} (x - \mu)} = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

### 4.3 Statistical Decision Rule: The Chi-Squared Distribution
If the inliers follow a multivariate normal distribution $x \sim \mathcal{N}(\mu, \Sigma)$, then the squared Mahalanobis distance follows a **Chi-squared distribution** with $d$ degrees of freedom:
$$D_M(x)^2 = (x - \mu)^T \Sigma^{-1} (x - \mu) \sim \chi_d^2$$

To detect anomalies at significance level $\alpha$ (e.g. $\alpha = 0.01$ for a 99% confidence boundary):
$$\text{Decision Rule:} \quad \text{Flag } x \text{ as an anomaly if } D_M(x)^2 > \chi_{d, 1 - \alpha}^2$$

For $d = 2$ dimensions and $\alpha = 0.01$:
$$\chi_{2, 0.99}^2 = -2 \ln(1 - 0.99) = -2 \ln(0.01) \approx 9.2103 \implies D_M(x) > \sqrt{9.2103} \approx 3.035$$

---

## 5. Isolation Forest (iForest)

When data contains non-linear clusters, multimodality, or complex manifolds, global elliptical Mahalanobis boundaries produce false positives. **Isolation Forest** (Liu, Ting, Zhou, 2008) is an ensemble tree algorithm designed specifically for anomaly detection.

### 5.1 The Core Premise: Anomalies Are Few and Different
Traditional anomaly detectors model normal points and declare anything far from the normal model an outlier. Isolation Forest flips the paradigm: it **actively targets anomalies** by isolating them using random recursive axis-aligned hyperplanes:
1. Anomalies require **very few random cuts** to be isolated into a singleton leaf (they reside in sparse regions).
2. Inliers require **many random cuts** to be isolated (they reside in dense clusters).

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ISOLATION FOREST RECURSION                      │
│                                                                        │
│   INLIER REGION (Dense Cluster):        ANOMALY REGION (Isolated):     │
│   Split 1 ──> Still in cluster          Split 1 ──> ISOLATED!          │
│   Split 2 ──> Still in cluster          Path length h(x) = 1           │
│   Split 3 ──> Still in cluster                                         │
│   Split 12 ──> Finally isolated!                                       │
│   Path length h(x) = 12                                                │
└────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Anomaly Score Derivation
Let $h(x)$ denote the path length (number of edges traversed from root to leaf) to isolate observation $x$ in an Isolation Tree.

The average path length of an unsuccessful search in a Binary Search Tree (BST) of $n$ instances is:
$$c(n) = 2 \ln(n - 1) + 2\gamma - \frac{2(n - 1)}{n}$$
where $\gamma \approx 0.5772156649$ is the Euler-Mascheroni constant. $c(n)$ serves as the theoretical normalization factor.

The **Anomaly Score** of an observation $x$ across an ensemble of $T$ isolation trees is:
$$s(x, n) = 2^{-\frac{\mathbb{E}[h(x)]}{c(n)}}$$

#### Interpretation of $s(x, n) \in [0, 1]$:
- If $\mathbb{E}[h(x)] \to 0 \implies s(x, n) \to 2^0 = 1$: **Definite Anomaly** (isolated immediately near tree root).
- If $\mathbb{E}[h(x)] \to c(n) \implies s(x, n) \to 2^{-1} = 0.5$: **Typical Instance** (average tree depth).
- If $\mathbb{E}[h(x)] \to n - 1 \implies s(x, n) \to 2^{-\infty} = 0$: **Strong Inlier** (deepest branch of dense cluster).

---

## 6. Algorithmic Comparison

| Method | Best Suited For | Boundary Geometry | Computational Complexity | Sensitivity to Correlated Features |
| :--- | :--- | :--- | :--- | :--- |
| **Kernel Density (KDE)** | Low dimensions ($d \le 3$), continuous density modeling | Arbitrary non-parametric contours | $\mathcal{O}(N_{\text{train}} \cdot N_{\text{eval}})$ | Moderate |
| **Mahalanobis Distance** | Gaussian-distributed inliers with linear correlations | Ellipsoidal iso-probability contours | $\mathcal{O}(N d^2 + d^3)$ | Excellent (directly inverts covariance) |
| **Isolation Forest** | High-dimensional tabular datasets, multimodal inliers | Axis-aligned box partitions | $\mathcal{O}(T \cdot \psi \log \psi)$ ($\psi \approx 256$) | Moderate (can use oblique trees) |
| **Local Outlier Factor (LOF)** | Datasets with varying local densities across clusters | $k$-nearest neighbor reachability graph | $\mathcal{O}(N^2)$ ($\mathcal{O}(N \log N)$ with KD-tree) | Good |

---

## 7. Summary & Phase 09 Synthesis

In **Phase 09**, we traversed the foundational pillars of Unsupervised Learning & Representation:
1. **Clustering (Topic 9.1):** Hard Voronoi partitions via **K-Means** with K-Means++ seeding, soft probabilistic mixture modeling via **GMM & Expectation-Maximization**, and density-based clustering via **DBSCAN**.
2. **Dimensionality Reduction (Topic 9.2):** Maximizing Rayleigh quotients via Lagrange multipliers in **PCA**, and the foundational equivalence to **Singular Value Decomposition (SVD)** right singular vectors and the Eckart-Young-Mirsky Theorem.
3. **Manifold Learning (Topic 9.3):** Resolving the curse of dimensionality and crowding problem using **t-SNE** with heavy-tailed Student-t distributions, and topological fuzzy simplicial sets in **UMAP**.
4. **Density Estimation & Anomaly Detection (Topic 9.4):** Non-parametric continuous probability modeling with **KDE**, statistical outlier gating with **Mahalanobis Distance**, and tree-based isolation depth in **Isolation Forests**.
