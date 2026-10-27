# Topic 03: Exercises & Solutions — Continual Learning & Catastrophic Forgetting

## Exercise 1: Equivalence of Expected Hessian and Fisher Information Matrix

Consider a supervised probabilistic model with conditional likelihood $p_\theta(y \mid x)$.
The empirical negative log-likelihood loss is:

$$\mathcal{L}(\boldsymbol{\theta}) = -\mathbb{E}_{(x, y) \sim p_{\text{data}}} \left[ \ln p_\theta(y \mid x) \right]$$

1. Prove that when the expectation is taken over the model's own distribution $p_\theta(y \mid x)$, the expected Hessian of the negative log-likelihood equals the Fisher Information Matrix:
   $$\mathbb{E}_{y \sim p_\theta(\cdot \mid x)} \left[ -\nabla_\theta^2 \ln p_\theta(y \mid x) \right] = \mathbb{E}_{y \sim p_\theta(\cdot \mid x)} \left[ \nabla_\theta \ln p_\theta(y \mid x) \nabla_\theta \ln p_\theta(y \mid x)^\top \right]$$
2. Explain why the empirical Fisher computed in practice:
   $$\hat{\mathbf{F}} = \frac{1}{N} \sum_{i=1}^N \nabla_\theta \ln p_\theta(y_i \mid x_i) \nabla_\theta \ln p_\theta(y_i \mid x_i)^\top$$
   is always positive semi-definite ($\hat{\mathbf{F}} \succeq 0$), whereas the empirical Hessian $-\frac{1}{N}\sum_i \nabla^2 \ln p_\theta(y_i \mid x_i)$ may have negative eigenvalues away from a local minimum.

---

### Solution 1:

#### 1. Mathematical Proof
Start with the normalization condition of probability densities:
$$\int p_\theta(y \mid x) \, dy = 1$$
Differentiate both sides with respect to $\boldsymbol{\theta}$:
$$\nabla_\theta \int p_\theta(y \mid x) \, dy = \int \nabla_\theta p_\theta(y \mid x) \, dy = \mathbf{0}$$
Using the log-derivative identity $\nabla_\theta p_\theta(y \mid x) = p_\theta(y \mid x) \nabla_\theta \ln p_\theta(y \mid x)$:
$$\int p_\theta(y \mid x) \nabla_\theta \ln p_\theta(y \mid x) \, dy = \mathbb{E}_{y \sim p_\theta} \left[ \nabla_\theta \ln p_\theta(y \mid x) \right] = \mathbf{0}$$
Now differentiate this vector equation a second time with respect to $\boldsymbol{\theta}^\top$:
$$\nabla_\theta \left( \int p_\theta(y \mid x) \nabla_\theta \ln p_\theta(y \mid x)^\top \, dy \right) = \mathbf{0}$$
Applying the product rule under the integral:
$$\int \left( \nabla_\theta p_\theta(y \mid x) \nabla_\theta \ln p_\theta(y \mid x)^\top + p_\theta(y \mid x) \nabla_\theta^2 \ln p_\theta(y \mid x) \right) dy = \mathbf{0}$$
Substitute $\nabla_\theta p_\theta(y \mid x) = p_\theta(y \mid x) \nabla_\theta \ln p_\theta(y \mid x)$:
$$\int p_\theta(y \mid x) \left[ \nabla_\theta \ln p_\theta(y \mid x) \nabla_\theta \ln p_\theta(y \mid x)^\top + \nabla_\theta^2 \ln p_\theta(y \mid x) \right] dy = \mathbf{0}$$
Writing as an expectation:
$$\mathbb{E}_{y \sim p_\theta} \left[ \nabla_\theta \ln p_\theta(y \mid x) \nabla_\theta \ln p_\theta(y \mid x)^\top \right] + \mathbb{E}_{y \sim p_\theta} \left[ \nabla_\theta^2 \ln p_\theta(y \mid x) \right] = \mathbf{0}$$
Rearranging terms yields the identity:
$$\mathbb{E}_{y \sim p_\theta} \left[ -\nabla_\theta^2 \ln p_\theta(y \mid x) \right] = \mathbb{E}_{y \sim p_\theta} \left[ \nabla_\theta \ln p_\theta(y \mid x) \nabla_\theta \ln p_\theta(y \mid x)^\top \right] = \mathbf{F}_x$$
Q.E.D.

#### 2. Positive Semi-Definiteness
The empirical Fisher $\hat{\mathbf{F}}$ is an outer product sum:
$$\hat{\mathbf{F}} = \frac{1}{N} \sum_{i=1}^N \mathbf{g}_i \mathbf{g}_i^\top$$
For any non-zero vector $\mathbf{v} \in \mathbb{R}^P$:
$$\mathbf{v}^\top \hat{\mathbf{F}} \mathbf{v} = \frac{1}{N} \sum_{i=1}^N \mathbf{v}^\top (\mathbf{g}_i \mathbf{g}_i^\top) \mathbf{v} = \frac{1}{N} \sum_{i=1}^N (\mathbf{v}^\top \mathbf{g}_i)^2 \ge 0$$
Hence, $\hat{\mathbf{F}}$ is guaranteed to be positive semi-definite everywhere in parameter space. This ensures that the quadratic EWC penalty $\frac{\lambda}{2} (\boldsymbol{\theta} - \boldsymbol{\theta}_A^*)^\top \mathbf{F} (\boldsymbol{\theta} - \boldsymbol{\theta}_A^*)$ is strictly convex with a well-defined unique minimum.

---

## Exercise 2: Analytical Equilibrium Point Under Quadratic Loss Surfaces

Suppose Task A and Task B have quadratic loss landscapes:

$$\mathcal{L}_A(\boldsymbol{\theta}) = \frac{1}{2} (\boldsymbol{\theta} - \boldsymbol{\theta}_A^*)^\top \mathbf{F}_A (\boldsymbol{\theta} - \boldsymbol{\theta}_A^*)$$

$$\mathcal{L}_B(\boldsymbol{\theta}) = \frac{1}{2} (\boldsymbol{\theta} - \boldsymbol{\theta}_B^*)^\top \mathbf{F}_B (\boldsymbol{\theta} - \boldsymbol{\theta}_B^*)$$

where $\mathbf{F}_A \succ 0$ and $\mathbf{F}_B \succ 0$ are positive definite matrices.

1. Formulate the EWC objective for Task B with penalty hyperparameter $\lambda > 0$.
2. Derive the analytical closed-form solution for the optimal parameters $\boldsymbol{\theta}_{\text{EWC}}^*$.
3. Analyze the asymptotic behavior as $\lambda \to 0$ and as $\lambda \to \infty$.

---

### Solution 2:

#### 1. EWC Objective
$$\mathcal{L}_{\text{EWC}}(\boldsymbol{\theta}) = \frac{1}{2} (\boldsymbol{\theta} - \boldsymbol{\theta}_B^*)^\top \mathbf{F}_B (\boldsymbol{\theta} - \boldsymbol{\theta}_B^*) + \frac{\lambda}{2} (\boldsymbol{\theta} - \boldsymbol{\theta}_A^*)^\top \mathbf{F}_A (\boldsymbol{\theta} - \boldsymbol{\theta}_A^*)$$

#### 2. Derivation of Equilibrium Point
Take the gradient with respect to $\boldsymbol{\theta}$ and set to $\mathbf{0}$:
$$\nabla_\theta \mathcal{L}_{\text{EWC}}(\boldsymbol{\theta}) = \mathbf{F}_B (\boldsymbol{\theta} - \boldsymbol{\theta}_B^*) + \lambda \mathbf{F}_A (\boldsymbol{\theta} - \boldsymbol{\theta}_A^*) = \mathbf{0}$$
Expand terms:
$$\mathbf{F}_B \boldsymbol{\theta} - \mathbf{F}_B \boldsymbol{\theta}_B^* + \lambda \mathbf{F}_A \boldsymbol{\theta} - \lambda \mathbf{F}_A \boldsymbol{\theta}_A^* = \mathbf{0}$$
$$(\mathbf{F}_B + \lambda \mathbf{F}_A) \boldsymbol{\theta} = \mathbf{F}_B \boldsymbol{\theta}_B^* + \lambda \mathbf{F}_A \boldsymbol{\theta}_A^*$$
Since $\mathbf{F}_B \succ 0$ and $\lambda \mathbf{F}_A \succ 0$, their sum $(\mathbf{F}_B + \lambda \mathbf{F}_A)$ is strictly positive definite and invertible:
$$\boldsymbol{\theta}_{\text{EWC}}^* = \left( \mathbf{F}_B + \lambda \mathbf{F}_A \right)^{-1} \left( \mathbf{F}_B \boldsymbol{\theta}_B^* + \lambda \mathbf{F}_A \boldsymbol{\theta}_A^* \right)$$

#### 3. Asymptotic Limits
- **Plasticity Limit ($\lambda \to 0$):**
  $$\lim_{\lambda \to 0} \boldsymbol{\theta}_{\text{EWC}}^* = \mathbf{F}_B^{-1} \mathbf{F}_B \boldsymbol{\theta}_B^* = \boldsymbol{\theta}_B^*$$
  The model completely forgets Task A and converges to the single-task optimum of Task B (naive gradient descent).
- **Stability Limit ($\lambda \to \infty$):**
  Divide numerator and denominator by $\lambda$:
  $$\lim_{\lambda \to \infty} \boldsymbol{\theta}_{\text{EWC}}^* = \lim_{\lambda \to \infty} \left( \frac{1}{\lambda}\mathbf{F}_B + \mathbf{F}_A \right)^{-1} \left( \frac{1}{\lambda}\mathbf{F}_B \boldsymbol{\theta}_B^* + \mathbf{F}_A \boldsymbol{\theta}_A^* \right) = \mathbf{F}_A^{-1} \mathbf{F}_A \boldsymbol{\theta}_A^* = \boldsymbol{\theta}_A^*$$
  The network freezes at $\boldsymbol{\theta}_A^*$, refusing to acquire any new knowledge for Task B (zero plasticity).
- For intermediate $\lambda$, $\boldsymbol{\theta}_{\text{EWC}}^*$ lies along the geodesic manifold connecting the two task minima, weighted by their respective Fisher curvatures.

---

## Exercise 3: Storage Complexity: Full vs. Diagonal Fisher in Foundation Models

Consider applying continual learning to a 7-billion parameter language model (e.g. LLaMA-7B, $P = 7 \times 10^9$ parameters).
Assume each float is stored in standard 32-bit floating point format (4 bytes).

1. Calculate the memory storage required to store the **full** Fisher Information Matrix $\mathbf{F}_A \in \mathbb{R}^{P \times P}$.
2. Calculate the memory storage required to store the **diagonal** Fisher Information vector $\text{diag}(\mathbf{F}_A) \in \mathbb{R}^P$.
3. Discuss the architectural trade-off of the diagonal approximation (why off-diagonal parameter couplings are missed).

---

### Solution 3:

#### 1. Full Fisher Matrix Storage
A full matrix for $P = 7 \times 10^9$ parameters requires $P^2$ entries:
$$N_{\text{elements}} = (7 \times 10^9)^2 = 4.9 \times 10^{19} \text{ floats}$$
Memory footprint in bytes:
$$\text{Memory} = 4.9 \times 10^{19} \times 4 \text{ bytes} \approx 1.96 \times 10^{20} \text{ bytes} \approx 196,000 \text{ Petabytes (196 Exabytes)}$$
This is hundreds of times larger than the storage capacity of all data centers combined on Earth! Storing the full Fisher matrix is mathematically and physically impossible.

#### 2. Diagonal Fisher Storage
The diagonal vector stores only $P$ floats:
$$N_{\text{elements}} = 7 \times 10^9 \text{ floats}$$
Memory footprint in bytes:
$$\text{Memory} = 7 \times 10^9 \times 4 \text{ bytes} = 2.8 \times 10^{10} \text{ bytes} \approx 28 \text{ GB}$$
28 GB easily fits into a single modern GPU (e.g., NVIDIA A100 80GB or H100), making diagonal EWC feasible in practice.

#### 3. Architectural Limitations of Diagonal Approximation
The diagonal assumption treats all parameters as independent:
$$\mathbf{F} \approx \text{diag}(F_1, F_2, \dots, F_P)$$
In reality, neural network parameters are strongly coupled. A change in weight $w_1$ can often be completely compensated by an equal and opposite adjustment in weight $w_2$ without harming Task A (the loss basin is a diagonal valley, not axis-aligned).
By ignoring off-diagonal terms $F_{ij} = \mathbb{E}[\nabla_i \mathcal{L} \nabla_j \mathcal{L}]$, diagonal EWC forces *both* $w_1$ and $w_2$ to remain stationary, over-constraining the model and limiting the available capacity for Task B. Methods like **K-FAC** (Kronecker-factored Approximate Curvature) bridge this gap by approximating weight matrices as tensor products of smaller factor matrices.
