# Topic 05: Information Geometry & Natural Gradients

## 1. Introduction & 12-Year-Old Intuition

Imagine you are looking at a flat paper road map of Greenland and Africa. 

On a standard Mercator projection map, Greenland looks enormous—almost the exact same size as the entire continent of Africa! If an airplane flies 1 inch across Greenland on your paper map, you might think it traveled 2,000 miles. But if it flies 1 inch across central Africa, it actually traveled 2,000 miles. On the paper map, distances are distorted depending on where you are. 

If you use a simple ruler to measure distances on the flat paper map, you will make massive navigation errors. The Earth is **not flat**—it is a curved 2D surface (a Riemannian manifold) embedded in 3D space. To measure real distance between cities, you must use spherical geometry and latitude-dependent rulers.

In machine learning, probability distributions live on a curved space called a **statistical manifold**:
- Imagine two bell curves (Gaussian distributions). Both have standard deviation $\sigma = 100$. If you shift the mean $\mu$ by $1.0$ unit (from $0$ to $1$), the two curves still overlap almost $99.9\%$. To any human, they look identical.
- Now imagine two bell curves with standard deviation $\sigma = 0.01$. If you shift the mean $\mu$ by $1.0$ unit (from $0$ to $1$), they don't overlap at all! They are entirely different, mutually exclusive worlds.

In both cases, the parameter changed by the exact same Euclidean distance: $|\Delta \mu| = 1.0$.
Yet, in terms of information and probability, the first change was tiny, while the second change was catastrophic!

Standard **Euclidean Gradient Descent** treats the parameter space like a flat sheet of paper. It takes steps of equal coordinate length, completely blind to how probability distributions actually change.

**Natural Gradient Descent** (pioneered by Shun-ichi Amari in 1998) replaces the flat ruler with the **Fisher Information Metric**. It measures distance not by numbers in a computer tensor, but by how much the actual probability distribution changes (via Kullback-Leibler divergence). It follows the true shortest path (the **geodesic**) on the manifold of distributions!

---

## 2. Complete Mathematical Symbol Decoder

| Symbol | Mathematical Entity | Dimensionality / Domain | Operational & Research Role |
| :--- | :--- | :--- | :--- |
| $\mathcal{M}$ | Statistical Manifold | Smooth Riemannian manifold | Space of probability distributions $\{p_{\boldsymbol{\theta}} : \boldsymbol{\theta} \in \Theta\}$. |
| $\boldsymbol{\theta}$ | Coordinate System | $\mathbb{R}^d$ | Arbitrary parameterization of the distribution (e.g. mean/variance, logits). |
| $p_{\boldsymbol{\theta}}(x)$ | Probability Density Function | Non-negative density | Model family defining likelihood over data domain $\mathcal{X}$. |
| $\mathbf{s}_{\boldsymbol{\theta}}(x)$ | Score Vector | $\mathbb{R}^d$ | $\nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)$; tangent vector along the manifold. |
| $\mathbf{F}(\boldsymbol{\theta})$ | Fisher Information Matrix (FIM) | $\mathbb{R}^{d \times d}$, symmetric PSD | Riemannian metric tensor $g_{ij}(\boldsymbol{\theta})$ defining intrinsic distance on $\mathcal{M}$. |
| $\mathbb{D}_{\text{KL}}(p \| q)$ | Kullback-Leibler Divergence | $\mathbb{R}_{\ge 0}$ | Asymmetric relative entropy measuring information lost when using $q$ to approximate $p$. |
| $d s^2$ | Riemannian Line Element | Differential quadratic form | $d s^2 = d\boldsymbol{\theta}^\top \mathbf{F}(\boldsymbol{\theta}) d\boldsymbol{\theta} \approx 2 \mathbb{D}_{\text{KL}}(p_{\boldsymbol{\theta}} \| p_{\boldsymbol{\theta} + d\boldsymbol{\theta}})$. |
| $\tilde{\nabla} L(\boldsymbol{\theta})$ | Natural Gradient | $\mathbb{R}^d$ | $\tilde{\nabla} L \triangleq \mathbf{F}(\boldsymbol{\theta})^{-1} \nabla L(\boldsymbol{\theta})$; steepest descent direction on the manifold. |
| $\eta$ | Manifold Step Size | Positive real scalar | Constant step distance measured strictly in KL divergence. |
| $\mathbf{J}_{\boldsymbol{\phi}}$ | Jacobian of Diffeomorphism | $\mathbb{R}^{d \times d}$ | Transformation matrix between alternate coordinate charts $\boldsymbol{\phi}(\boldsymbol{\theta})$. |

---

## 3. The Fisher Information Matrix as a Riemannian Metric

### 3.1 Definition of the Fisher Metric
Let $\mathcal{M} = \{p_{\boldsymbol{\theta}} : \boldsymbol{\theta} \in \Theta \subset \mathbb{R}^d\}$ be a family of probability densities parameterized by $\boldsymbol{\theta}$.
The components of the Fisher Information Matrix $\mathbf{F}(\boldsymbol{\theta})$ are defined as:
$$F_{ij}(\boldsymbol{\theta}) \triangleq \mathbb{E}_{x \sim p_{\boldsymbol{\theta}}}\left[ \frac{\partial \log p_{\boldsymbol{\theta}}(x)}{\partial \theta_i} \frac{\partial \log p_{\boldsymbol{\theta}}(x)}{\partial \theta_j} \right]$$

### 3.2 Equivalence with Negative Expected Hessian
Provided the support of $p_{\boldsymbol{\theta}}$ does not depend on $\boldsymbol{\theta}$ and differentiation can be exchanged with integration:
$$\int p_{\boldsymbol{\theta}}(x) dx = 1 \implies \frac{\partial}{\partial \theta_i} \int p_{\boldsymbol{\theta}}(x) dx = \int \frac{\partial p_{\boldsymbol{\theta}}(x)}{\partial \theta_i} dx = 0$$
Notice that $\frac{\partial p}{\partial \theta_i} = p \frac{\partial \log p}{\partial \theta_i}$. Differentiating a second time:
$$\frac{\partial}{\partial \theta_j} \int p \frac{\partial \log p}{\partial \theta_i} dx = \int \left( \frac{\partial p}{\partial \theta_j} \frac{\partial \log p}{\partial \theta_i} + p \frac{\partial^2 \log p}{\partial \theta_i \partial \theta_j} \right) dx = 0$$
$$\int p \left( \frac{\partial \log p}{\partial \theta_i} \frac{\partial \log p}{\partial \theta_j} \right) dx + \int p \frac{\partial^2 \log p}{\partial \theta_i \partial \theta_j} dx = 0$$
Therefore:
$$F_{ij}(\boldsymbol{\theta}) = -\mathbb{E}_{x \sim p_{\boldsymbol{\theta}}}\left[ \frac{\partial^2 \log p_{\boldsymbol{\theta}}(x)}{\partial \theta_i \partial \theta_j} \right]$$

---

## 4. KL Divergence Second-Order Taylor Expansion

Why is the Fisher Information Matrix the fundamental metric tensor of information geometry?

Consider two infinitesimally close distributions $p_{\boldsymbol{\theta}}$ and $p_{\boldsymbol{\theta} + d\boldsymbol{\theta}}$.
The Kullback-Leibler divergence is:
$$\mathbb{D}_{\text{KL}}(p_{\boldsymbol{\theta}} \| p_{\boldsymbol{\theta} + d\boldsymbol{\theta}}) = \int p_{\boldsymbol{\theta}}(x) \log \frac{p_{\boldsymbol{\theta}}(x)}{p_{\boldsymbol{\theta} + d\boldsymbol{\theta}}(x)} dx = -\int p_{\boldsymbol{\theta}}(x) \left( \log p_{\boldsymbol{\theta} + d\boldsymbol{\theta}}(x) - \log p_{\boldsymbol{\theta}}(x) \right) dx$$

Expand $\log p_{\boldsymbol{\theta} + d\boldsymbol{\theta}}(x)$ as a Taylor series in $d\boldsymbol{\theta}$ around $\boldsymbol{\theta}$:
$$\log p_{\boldsymbol{\theta} + d\boldsymbol{\theta}}(x) = \log p_{\boldsymbol{\theta}}(x) + \sum_i \frac{\partial \log p}{\partial \theta_i} d\theta_i + \frac{1}{2} \sum_{i,j} \frac{\partial^2 \log p}{\partial \theta_i \partial \theta_j} d\theta_i d\theta_j + \mathcal{O}(\|d\boldsymbol{\theta}\|^3)$$

Substitute this back into the KL integral:
$$\mathbb{D}_{\text{KL}}(p_{\boldsymbol{\theta}} \| p_{\boldsymbol{\theta} + d\boldsymbol{\theta}}) = -\sum_i d\theta_i \underbrace{\mathbb{E}\left[ \frac{\partial \log p}{\partial \theta_i} \right]}_{0} - \frac{1}{2} \sum_{i,j} d\theta_i d\theta_j \underbrace{\mathbb{E}\left[ \frac{\partial^2 \log p}{\partial \theta_i \partial \theta_j} \right]}_{-F_{ij}(\boldsymbol{\theta})} + \mathcal{O}(\|d\boldsymbol{\theta}\|^3)$$

```
             The Riemannian Metric of Information Geometry
             
    D_KL( p_θ || p_{θ + dθ} ) = (1/2) · dθ^T F(θ) dθ + O(||dθ||^3)
    
  - At zero separation: D_KL = 0
  - First derivative:   ∇_dθ D_KL |_{dθ=0} = 0 (Flat tangent space)
  - Curvature tensor:   Hessian of KL divergence is IDENTICALLY the Fisher Matrix F(θ)!
```

$$\mathbb{D}_{\text{KL}}(p_{\boldsymbol{\theta}} \| p_{\boldsymbol{\theta} + d\boldsymbol{\theta}}) = \frac{1}{2} d\boldsymbol{\theta}^\top \mathbf{F}(\boldsymbol{\theta}) d\boldsymbol{\theta} + \mathcal{O}(\|d\boldsymbol{\theta}\|^3)$$
The Fisher Information Matrix is **literally the local Riemannian metric tensor** measuring statistical distance $ds^2$ on the manifold of probability distributions!

---

## 5. Natural Gradient Descent (Amari 1998)

In standard Euclidean gradient descent, we define steepest descent as:
$$\Delta \boldsymbol{\theta}_{\text{EGD}} = \arg\min_{\Delta \boldsymbol{\theta}} L(\boldsymbol{\theta} + \Delta \boldsymbol{\theta}) \quad \text{subject to} \quad \|\Delta \boldsymbol{\theta}\|_2^2 \le \epsilon^2$$
Using linear approximation $L(\boldsymbol{\theta} + \Delta \boldsymbol{\theta}) \approx L(\boldsymbol{\theta}) + \nabla L^\top \Delta \boldsymbol{\theta}$, the solution is:
$$\Delta \boldsymbol{\theta}_{\text{EGD}} \propto -\nabla L(\boldsymbol{\theta})$$

In **Natural Gradient Descent**, we constrain the step size using the true intrinsic distance on the statistical manifold (KL divergence):
$$\Delta \boldsymbol{\theta}_{\text{NGD}} = \arg\min_{\Delta \boldsymbol{\theta}} L(\boldsymbol{\theta} + \Delta \boldsymbol{\theta}) \quad \text{subject to} \quad \mathbb{D}_{\text{KL}}(p_{\boldsymbol{\theta}} \| p_{\boldsymbol{\theta} + \Delta \boldsymbol{\theta}}) \le \epsilon$$

Using the second-order Taylor expansion of KL:
$$\min_{\Delta \boldsymbol{\theta}} \nabla L(\boldsymbol{\theta})^\top \Delta \boldsymbol{\theta} \quad \text{subject to} \quad \frac{1}{2} \Delta \boldsymbol{\theta}^\top \mathbf{F}(\boldsymbol{\theta}) \Delta \boldsymbol{\theta} \le \epsilon$$

### 5.1 Analytical Solution via Lagrange Multipliers
Form the Lagrangian with multiplier $\lambda > 0$:
$$\mathcal{L}(\Delta \boldsymbol{\theta}, \lambda) = \nabla L(\boldsymbol{\theta})^\top \Delta \boldsymbol{\theta} + \lambda \left( \frac{1}{2} \Delta \boldsymbol{\theta}^\top \mathbf{F}(\boldsymbol{\theta}) \Delta \boldsymbol{\theta} - \epsilon \right)$$
Setting the gradient with respect to $\Delta \boldsymbol{\theta}$ to zero:
$$\nabla L(\boldsymbol{\theta}) + \lambda \mathbf{F}(\boldsymbol{\theta}) \Delta \boldsymbol{\theta} = \mathbf{0} \implies \Delta \boldsymbol{\theta} = -\frac{1}{\lambda} \mathbf{F}(\boldsymbol{\theta})^{-1} \nabla L(\boldsymbol{\theta})$$

Defining learning rate $\eta \triangleq \frac{1}{\lambda}$, the **Natural Gradient Update** is:
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{F}(\boldsymbol{\theta}_t)^{-1} \nabla L(\boldsymbol{\theta}_t)$$

---

## 6. Parameterization Invariance: The Triumph of Information Geometry

The crowning achievement of Natural Gradient Descent is **Reparameterization Invariance**.

Suppose a researcher chooses coordinates $\boldsymbol{\theta}$ (e.g. mean $\mu$ and standard deviation $\sigma$), while another chooses canonical coordinates $\boldsymbol{\phi}$ (e.g. mean $\mu$ and precision $\tau = 1/\sigma^2$).
Let $\boldsymbol{\phi} = \mathbf{g}(\boldsymbol{\theta})$ be an arbitrary smooth invertible mapping with Jacobian $\mathbf{J} = \frac{\partial \boldsymbol{\phi}}{\partial \boldsymbol{\theta}}$.

### 6.1 Euclidean Gradients are Coordinate-Dependent (Fragile):
$$\nabla_{\boldsymbol{\theta}} L = \mathbf{J}^\top \nabla_{\boldsymbol{\phi}} L$$
An optimization step in $\boldsymbol{\theta}$-space does **NOT** equal the transformed step in $\boldsymbol{\phi}$-space:
$$\mathbf{J} \Delta \boldsymbol{\theta}_{\text{EGD}} = -\eta \mathbf{J} \nabla_{\boldsymbol{\theta}} L = -\eta \mathbf{J} \mathbf{J}^\top \nabla_{\boldsymbol{\phi}} L \neq -\eta \nabla_{\boldsymbol{\phi}} L = \Delta \boldsymbol{\phi}_{\text{EGD}}$$
Standard gradient descent takes completely different geometric paths depending on how the engineer chose to name the variables!

### 6.2 Natural Gradients are Invariant (Intrinsic):
By the tensor transformation law, the Fisher metric transforms as:
$$\mathbf{F}_{\boldsymbol{\theta}} = \mathbf{J}^\top \mathbf{F}_{\boldsymbol{\phi}} \mathbf{J} \implies \mathbf{F}_{\boldsymbol{\theta}}^{-1} = \mathbf{J}^{-1} \mathbf{F}_{\boldsymbol{\phi}}^{-1} (\mathbf{J}^\top)^{-1}$$
Now compute the induced update in $\boldsymbol{\phi}$:
$$\mathbf{J} \Delta \boldsymbol{\theta}_{\text{NGD}} = -\eta \mathbf{J} \left( \mathbf{F}_{\boldsymbol{\theta}}^{-1} \nabla_{\boldsymbol{\theta}} L \right) = -\eta \mathbf{J} \left( \mathbf{J}^{-1} \mathbf{F}_{\boldsymbol{\phi}}^{-1} (\mathbf{J}^\top)^{-1} \mathbf{J}^\top \nabla_{\boldsymbol{\phi}} L \right) = -\eta \mathbf{F}_{\boldsymbol{\phi}}^{-1} \nabla_{\boldsymbol{\phi}} L = \Delta \boldsymbol{\phi}_{\text{NGD}}$$
The update on the probability manifold is **strictly identical** regardless of what coordinate system is used!

---

## 7. Connections to Modern AI Research

- **Trust-Region Policy Optimization (TRPO)**: Direct application of natural gradient in policy space, solving $\mathbf{F}\mathbf{d} = \mathbf{g}$ via conjugate gradient.
- **K-FAC (Kronecker-Factored Approximate Curvature)**: Scales natural gradient to deep neural networks by approximating the Fisher matrix as Kronecker products of layer activation and gradient covariances.
- **Variational Autoencoders & Bayesian Inference**: Natural gradients optimize variational distributions $q_\phi(z|x)$ along geodesics without getting trapped in coordinate singularities.
