# Topic 13.5: Exercises & Solutions — Deep Regularization Dynamics

---

## Problem Set Overview

This exercise set investigates the mathematical connection between Dropout and data-dependent $L_2$ regularization, data augmentation as tangent propagation, and weight decay dynamics.

---

### Problem 1: Dropout as Adaptive Data-Dependent $L_2$ Regularization

Consider a linear regression model with no bias: $\hat{y} = \mathbf{w}^T \mathbf{x}$.
Let Inverted Dropout be applied directly to the inputs $\mathbf{x} = [x_1, \dots, x_D]^T$ with drop probability $p$.
The dropped input is:
$$\tilde{x}_j = \frac{m_j}{1 - p} x_j, \quad \text{where } m_j \sim \text{Bernoulli}(1 - p) \text{ independently.}$$

The training objective is the expected squared loss over the random dropout mask:
$$\mathcal{L}_{\text{dropout}}(\mathbf{w}) = \mathbb{E}_{\mathbf{m}}\left[ \frac{1}{2} (y - \mathbf{w}^T \tilde{\mathbf{x}})^2 \right]$$

1. Compute the expectation $\mathbb{E}[m_j]$ and variance $\text{Var}(m_j)$ of the Bernoulli mask.
2. Prove that the expected dropout loss can be decomposed exactly into the standard mean squared error plus an explicit data-dependent regularization penalty:
   $$\mathcal{L}_{\text{dropout}}(\mathbf{w}) = \frac{1}{2}(y - \mathbf{w}^T \mathbf{x})^2 + \frac{p}{2(1 - p)} \sum_{j=1}^D w_j^2 x_j^2$$
3. Contrast this penalty with standard $L_2$ weight decay $\frac{\lambda}{2} \sum w_j^2$. Why is dropout adaptive to feature scales?

---

### Solution 1

1. **Moments of Bernoulli Mask:**
   Since $m_j \in \{0, 1\}$ with $P(m_j = 1) = 1 - p$:
   $$\mathbb{E}[m_j] = 1 - p$$
   $$\text{Var}(m_j) = \mathbb{E}[m_j^2] - (\mathbb{E}[m_j])^2 = (1 - p) - (1 - p)^2 = (1 - p)(1 - (1 - p)) = p(1 - p)$$

2. **Decomposition of Expected Loss:**
   Let the prediction error be:
   $$e = y - \mathbf{w}^T \tilde{\mathbf{x}} = y - \sum_{j=1}^D w_j \frac{m_j}{1 - p} x_j$$
   Notice that $\mathbb{E}_{\mathbf{m}}[\tilde{\mathbf{x}}] = \mathbf{x}$, so $\mathbb{E}_{\mathbf{m}}[e] = y - \mathbf{w}^T \mathbf{x}$.
   Using the identity $\mathbb{E}[Z^2] = (\mathbb{E}[Z])^2 + \text{Var}(Z)$:
   $$\mathbb{E}_{\mathbf{m}}[(y - \mathbf{w}^T \tilde{\mathbf{x}})^2] = (y - \mathbf{w}^T \mathbf{x})^2 + \text{Var}_{\mathbf{m}}\left( \sum_{j=1}^D w_j \frac{m_j}{1 - p} x_j \right)$$
   Since the mask elements $m_j$ are independent for each feature $j$:
   $$\text{Var}_{\mathbf{m}}\left( \sum_{j=1}^D w_j \frac{m_j}{1 - p} x_j \right) = \sum_{j=1}^D \left(\frac{w_j x_j}{1 - p}\right)^2 \text{Var}(m_j)$$
   Substitute $\text{Var}(m_j) = p(1 - p)$:
   $$\sum_{j=1}^D \frac{w_j^2 x_j^2}{(1 - p)^2} p(1 - p) = \frac{p}{1 - p} \sum_{j=1}^D w_j^2 x_j^2$$
   Multiplying by the $\frac{1}{2}$ factor:
   $$\mathcal{L}_{\text{dropout}}(\mathbf{w}) = \frac{1}{2}(y - \mathbf{w}^T \mathbf{x})^2 + \frac{p}{2(1 - p)} \sum_{j=1}^D w_j^2 x_j^2$$
   This proves the theorem.

3. **Engineering Interpretation:**
   Standard $L_2$ regularization penalizes all weights uniformly with coefficient $\lambda$.
   In contrast, Dropout penalizes $w_j^2$ proportionally to $x_j^2$ (the variance/energy of feature $j$). Features that exhibit high variance are regularized much more aggressively, preventing the network from over-relying on individual dominant input channels!

---

### Problem 2: Data Augmentation as Tangent Distance Regularization

Let $\mathbf{x} \in \mathbb{R}^D$ be an input image, and let $T_\epsilon(\mathbf{x})$ be an infinitesimal transformation (such as a horizontal shift or slight rotation by angle $\epsilon \ll 1$).
Expanding $T_\epsilon(\mathbf{x})$ via Taylor series around $\epsilon = 0$:
$$T_\epsilon(\mathbf{x}) \approx \mathbf{x} + \epsilon \mathbf{v}(\mathbf{x})$$
where $\mathbf{v}(\mathbf{x}) = \left.\frac{\partial T_\epsilon(\mathbf{x})}{\partial \epsilon}\right|_{\epsilon=0}$ is the tangent vector of the transformation manifold.

1. Show that training on augmented samples $T_\epsilon(\mathbf{x})$ where $\epsilon \sim \mathcal{N}(0, \sigma_\epsilon^2)$ is mathematically equivalent to adding a directional gradient penalty:
   $$\mathcal{L}_{\text{aug}} \approx \mathcal{L}(f(\mathbf{x}), y) + \frac{\sigma_\epsilon^2}{2} \left\| \nabla_{\mathbf{x}} f(\mathbf{x}) \cdot \mathbf{v}(\mathbf{x}) \right\|^2$$
2. What physical prior does this enforce on the neural network?

---

### Solution 2

1. **Taylor Expansion of the Loss:**
   Let $L(x) = \mathcal{L}(f(T_\epsilon(\mathbf{x})), y) \approx \mathcal{L}(f(\mathbf{x} + \epsilon \mathbf{v}), y)$.
   Expanding around $\epsilon = 0$:
   $$L(\epsilon) \approx L(0) + \epsilon \nabla_{\mathbf{x}} L(\mathbf{x})^T \mathbf{v} + \frac{\epsilon^2}{2} \mathbf{v}^T \nabla_{\mathbf{x}}^2 L(\mathbf{x}) \mathbf{v}$$
   Taking expectation over $\epsilon \sim \mathcal{N}(0, \sigma_\epsilon^2)$:
   - $\mathbb{E}[\epsilon] = 0$, so the first-order linear term vanishes.
   - $\mathbb{E}[\epsilon^2] = \sigma_\epsilon^2$.
   Therefore:
   $$\mathbb{E}_\epsilon [L(\epsilon)] = L(0) + \frac{\sigma_\epsilon^2}{2} \mathbf{v}^T \nabla_{\mathbf{x}}^2 L(\mathbf{x}) \mathbf{v}$$
   For squared error or cross-entropy, the Hessian term approximates the squared directional derivative $\frac{\sigma_\epsilon^2}{2} \|\nabla_{\mathbf{x}} f(\mathbf{x}) \cdot \mathbf{v}(\mathbf{x})\|^2$.

2. **Physical Prior:**
   This directional gradient penalty explicitly forces the directional derivative of the model along the transformation vector $\mathbf{v}(\mathbf{x})$ to zero:
   $$\nabla_{\mathbf{x}} f(\mathbf{x}) \cdot \mathbf{v}(\mathbf{x}) \to 0$$
   The network becomes **locally flat and invariant** along the manifold of translations/rotations, preventing predictions from changing when an object shifts slightly in the visual field.

---

### Problem 3: Learning Rate Schedules and Effective Weight Decay

In modern training regimes, learning rate $\alpha_t$ follows a Cosine Annealing schedule from $\alpha_0 = 10^{-3}$ down to $\alpha_{T} = 10^{-6}$.
Recall the decoupled weight decay update:
$$\mathbf{w}_{t+1} = \mathbf{w}_t (1 - \alpha_t \lambda) - \alpha_t \mathbf{g}_t$$

1. If weight decay $\lambda$ is kept constant, what happens to the effective shrinkage per step as $\alpha_t \to 0$?
2. Why is this interaction beneficial in the final epochs of training?

---

### Solution 3

1. **Shrinkage Decay:**
   The parameter shrinkage per step is $\Delta_{\text{decay}} = \alpha_t \lambda \mathbf{w}_t$.
   As $\alpha_t \to 0$, the shrinkage force approaches zero: $\lim_{t \to T} \alpha_t \lambda = 0$.

2. **Late-Stage Convergence Advantage:**
   In early epochs when $\alpha_t$ is large, weight decay strongly compresses weights toward zero, clearing irrelevant parameter pathways and acting as a strong regularizer.
   In the final epochs, as the model nears a flat local minimum, weight decay diminishes, allowing the optimizer to settle precisely into the bottom of the basin without being forcibly pulled toward the origin!
