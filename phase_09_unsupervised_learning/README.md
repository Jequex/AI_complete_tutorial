# Phase 09: Unsupervised Learning & Representation

---

## 🧭 Executive Overview

In supervised learning, machines are guided by explicit target labels and error supervision. But in the physical world, the vast majority of real-world data—from genomic sequences and telescope telemetry to visual scenes and raw web text—is completely **unlabeled**.

**Phase 09: Unsupervised Learning & Representation** explores how machines discover latent structures, compact geometric representations, and statistical laws directly from raw unannotated observations. 

This phase covers the four core pillars of unsupervised learning:
1. **Clustering:** Partitioning observations into discrete categories via hard Voronoi tessellations (**K-Means** with K-Means++ seeding) and continuous probabilistic mixtures (**Gaussian Mixture Models** with Expectation-Maximization).
2. **Linear Dimensionality Reduction:** Discovering optimal orthogonal sub-spaces that maximize variance and minimize reconstruction error via **Principal Component Analysis (PCA)** and **Singular Value Decomposition (SVD)**.
3. **Manifold Learning:** Unfolding curved hyper-dimensional surfaces while preserving local neighborhoods and resolving the Crowding Problem via **t-SNE** (Student-t kernels) and topological fuzzy simplicial sets (**UMAP**).
4. **Density Estimation & Anomaly Detection:** Non-parametric continuous probability estimation via **Kernel Density Estimation (KDE)**, geometric outlier detection via **Mahalanobis Distance**, and recursive partitioning via **Isolation Forests**.

---

## 🗺️ Phase Syllabus & Topic Breakdown

```text
phase_09_unsupervised_learning/
├── index.html                                            # Phase Portal Hub & Quickstart
├── README.md                                             # Master Syllabus & Architecture Guide
│
├── 01_clustering_kmeans_gmm/
│   ├── lecture.html / lecture.md                         # K-Means, Lloyd's algorithm, Voronoi partitions, GMM EM derivation, DBSCAN
│   ├── exercises_and_solutions.html / .md                # K-Means hand calculation, Lagrange multiplier M-step, K-Means++, DBSCAN
│   └── experiment.py                                     # Pure Python 3 K-Means & GMM with EM from scratch (asserts monotonic LL increase)
│
├── 02_dimensionality_reduction_pca_svd/
│   ├── lecture.html / lecture.md                         # PCA derivation, Rayleigh quotient, Lagrange multiplier proof, SVD equivalence
│   ├── exercises_and_solutions.html / .md                # 2D covariance hand calculation, orthogonality proof, SVD singular values
│   └── experiment.py                                     # Pure Python 3 PCA engine via Power Iteration & Hotelling Deflation (3D to 2D)
│
├── 03_manifold_learning_tsne_umap/
│   ├── lecture.html / lecture.md                         # Curse of dimensionality, Crowding Problem, Student-t distribution, UMAP
│   ├── exercises_and_solutions.html / .md                # Gaussian affinities, Student-t derivative, t-SNE gradient, UMAP cross-entropy
│   └── experiment.py                                     # Pure Python 3 mini t-SNE gradient descent on 4D synthetic clusters
│
└── 04_density_estimation_and_anomaly_detection/
    ├── lecture.html / lecture.md                         # Parzen windows, Silverman's rule, Mahalanobis distance, Isolation Forests
    ├── exercises_and_solutions.html / .md                # 1D KDE manual calculation, Silverman's bandwidth, Chi-squared thresholding
    └── experiment.py                                     # Pure Python 3 KDE density estimator & Mahalanobis anomaly detector
```

---

## 🔑 Core Mathematical Pillars Mastered in Phase 09

1. **Expectation-Maximization (EM) & Evidence Lower Bound (ELBO):**
   Deriving the alternating optimization cycle for latent variables $Z$:
   $$\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i; \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i; \mu_j, \Sigma_j)}, \quad \mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}, \quad \pi_k = \frac{N_k}{N}$$
   Proving that each iteration monotonically increases the marginal observed data log-likelihood $\ln p(X \mid \theta)$.

2. **Rayleigh Quotient Maximization & Covariance Eigendecomposition:**
   Formulating the PCA objective as maximum variance projection and proving via Lagrange multipliers that the projection axes $w_k$ are the dominant eigenvectors of the sample covariance matrix $C$:
   $$C w = \lambda w, \quad \text{where } C = \frac{1}{N-1} X_c^T X_c$$

3. **Exact Equivalence of SVD and PCA:**
   Proving that the right singular vectors $V$ from $X_c = U \Sigma V^T$ match the principal component directions $W$, that eigenvalues satisfy $\lambda_j = \frac{\sigma_j^2}{N-1}$, and applying the Eckart-Young-Mirsky Theorem to minimize Frobenius reconstruction error.

4. **Resolving the Crowding Problem with Student-t Distributions:**
   Overcoming high-dimensional volume concentration by employing a Cauchy / Student-t distribution with 1 degree of freedom in the low-dimensional embedding space:
   $$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \ne l} (1 + \|y_k - y_l\|^2)^{-1}}$$
   Deriving the analytical gradient of the Kullback-Leibler divergence with attractive and repulsive spring forces.

5. **Covariance-Scaled Anomaly Detection:**
   Formulating the Mahalanobis distance to account for non-spherical feature correlations:
   $$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$
   Connecting $D_M^2$ to the Chi-squared distribution $\chi_d^2$ for statistically grounded thresholding.

---

## ⚡ Quickstart: Executing Phase 09 Labs

All experiments are written in pure Python 3 using only standard libraries (zero external dependencies):

```bash
# Topic 9.1: K-Means & Gaussian Mixture Models with EM from scratch
python3 phase_09_unsupervised_learning/01_clustering_kmeans_gmm/experiment.py

# Topic 9.2: Dimensionality Reduction (PCA & SVD) via Power Iteration & Deflation
python3 phase_09_unsupervised_learning/02_dimensionality_reduction_pca_svd/experiment.py

# Topic 9.3: Manifold Learning (Mini t-SNE with Student-t kernels from scratch)
python3 phase_09_unsupervised_learning/03_manifold_learning_tsne_umap/experiment.py

# Topic 9.4: Kernel Density Estimation & Mahalanobis Anomaly Detection
python3 phase_09_unsupervised_learning/04_density_estimation_and_anomaly_detection/experiment.py
```
