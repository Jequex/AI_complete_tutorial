# Topic 05: Information Geometry & Natural Gradients — Exercises & Solutions

## Exercise 1: Fisher Information Matrix of the Univariate Gaussian

Let $p(x; \mu, \sigma) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$ be a univariate normal distribution with parameter vector $\boldsymbol{\theta} = [\mu, \sigma]^\top$.

1. Derive the score vector $\mathbf{s}(x) = \left[\frac{\partial \log p}{\partial \mu}, \frac{\partial \log p}{\partial \sigma}\right]^\top$.
2. Compute the four components of the Fisher Information Matrix:
   $$F_{ij} = \mathbb{E}_{x \sim p}\left[ \frac{\partial \log p}{\partial \theta_i} \frac{\partial \log p}{\partial \theta_j} \right]$$
3. Write out $\mathbf{F}(\mu, \sigma)$ and its inverse $\mathbf{F}^{-1}(\mu, \sigma)$.
4. Explain why the metric tensor causes standard Euclidean gradient updates to take excessively huge steps in policy space when $\sigma$ is small.

### Solution 1:
1. **The Log-Likelihood & Score Function**:
   $$\log p(x; \mu, \sigma) = -\frac{1}{2}\ln(2\pi) - \ln \sigma - \frac{(x - \mu)^2}{2\sigma^2}$$
   - Derivative w.r.t. $\mu$:
     $$\frac{\partial \log p}{\partial \mu} = \frac{x - \mu}{\sigma^2}$$
   - Derivative w.r.t. $\sigma$:
     $$\frac{\partial \log p}{\partial \sigma} = -\frac{1}{\sigma} + \frac{(x - \mu)^2}{\sigma^3}$$

2. **Fisher Matrix Expectations**:
   Under $x \sim \mathcal{N}(\mu, \sigma^2)$, let $z = \frac{x - \mu}{\sigma} \sim \mathcal{N}(0, 1)$. Then $\mathbb{E}[z] = 0, \mathbb{E}[z^2] = 1, \mathbb{E}[z^3] = 0, \mathbb{E}[z^4] = 3$.
   - Component $F_{\mu\mu}$:
     $$F_{\mu\mu} = \mathbb{E}\left[ \left(\frac{x - \mu}{\sigma^2}\right)^2 \right] = \frac{1}{\sigma^4} \mathbb{E}[(x - \mu)^2] = \frac{\sigma^2}{\sigma^4} = \frac{1}{\sigma^2}$$
   - Component $F_{\mu\sigma}$:
     $$F_{\mu\sigma} = \mathbb{E}\left[ \frac{x - \mu}{\sigma^2} \left(-\frac{1}{\sigma} + \frac{(x - \mu)^2}{\sigma^3}\right) \right] = -\frac{\mathbb{E}[x - \mu]}{\sigma^3} + \frac{\mathbb{E}[(x - \mu)^3]}{\sigma^5} = 0 + 0 = 0$$
   - Component $F_{\sigma\sigma}$:
     $$F_{\sigma\sigma} = \mathbb{E}\left[ \left(-\frac{1}{\sigma} + \frac{(x - \mu)^2}{\sigma^3}\right)^2 \right] = \frac{1}{\sigma^2} \mathbb{E}[(z^2 - 1)^2] = \frac{1}{\sigma^2} (\mathbb{E}[z^4] - 2\mathbb{E}[z^2] + 1) = \frac{3 - 2 + 1}{\sigma^2} = \frac{2}{\sigma^2}$$

3. **Fisher Information Matrix and Inverse**:
   $$\mathbf{F}(\mu, \sigma) = \begin{bmatrix} \frac{1}{\sigma^2} & 0 \\ 0 & \frac{2}{\sigma^2} \end{bmatrix}, \quad \mathbf{F}^{-1}(\mu, \sigma) = \begin{bmatrix} \sigma^2 & 0 \\ 0 & \frac{\sigma^2}{2} \end{bmatrix}$$

4. **Geometric Significance**:
   When $\sigma \ll 1$, the Riemannian metric component $g_{\mu\mu} = 1/\sigma^2$ is enormous. A tiny coordinate shift $\Delta \mu$ produces a huge statistical separation:
   $$ds^2 \approx \frac{(\Delta \mu)^2}{\sigma^2}$$
   Standard Euclidean GD steps $\Delta \mu = -\eta \frac{\partial L}{\partial \mu}$ are scaled without knowledge of $\sigma$. Natural gradient premultiplies by $\mathbf{F}^{-1}$, automatically damping the update by $\sigma^2$ to ensure constant step distance in distribution space.

---

## Exercise 2: Natural Gradient on Categorical Distributions & Simplex Geometry

Consider a categorical distribution over $K$ outcomes with probabilities $\mathbf{p} = [p_1, \dots, p_K]^\top$ on the probability simplex $\sum_{k=1}^K p_k = 1, p_k > 0$.
Parameterized via unconstrained logits $\boldsymbol{\theta} \in \mathbb{R}^K$ where $p_k = \frac{e^{\theta_k}}{\sum_{j=1}^K e^{\theta_j}}$.

1. Derive the score vector $\frac{\partial \log p_y}{\partial \boldsymbol{\theta}}$ for an observed class $y \in \{1, \dots, K\}$.
2. Prove that the Fisher Information Matrix with respect to logits $\boldsymbol{\theta}$ is:
   $$\mathbf{F}(\boldsymbol{\theta}) = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$$
3. Show that $\mathbf{F}(\boldsymbol{\theta})$ has a 1-dimensional null space and state the null vector.

### Solution 2:
1. **Categorical Score**:
   The log probability of observing category $y$ is $\log p_y = \theta_y - \ln\left(\sum_{j=1}^K e^{\theta_j}\right)$.
   Differentiating with respect to $\theta_k$:
   $$\frac{\partial \log p_y}{\partial \theta_k} = \mathbb{I}(y = k) - p_k$$
   In vector form: $\mathbf{s}_y = \mathbf{e}_y - \mathbf{p}$.

2. **Fisher Information Matrix**:
   $$\mathbf{F}(\boldsymbol{\theta}) = \mathbb{E}_{y \sim \mathbf{p}}\left[ (\mathbf{e}_y - \mathbf{p})(\mathbf{e}_y - \mathbf{p})^\top \right] = \sum_{y=1}^K p_y (\mathbf{e}_y - \mathbf{p})(\mathbf{e}_y - \mathbf{p})^\top$$
   Expanding the outer product:
   $$\mathbf{F}(\boldsymbol{\theta}) = \sum_{y=1}^K p_y \left( \mathbf{e}_y \mathbf{e}_y^\top - \mathbf{e}_y \mathbf{p}^\top - \mathbf{p} \mathbf{e}_y^\top + \mathbf{p}\mathbf{p}^\top \right)$$
   - $\sum_{y=1}^K p_y \mathbf{e}_y \mathbf{e}_y^\top = \text{diag}(\mathbf{p})$.
   - $\sum_{y=1}^K p_y \mathbf{e}_y = \mathbf{p}$, so the cross terms give $-2 \mathbf{p}\mathbf{p}^\top$.
   - $\sum_{y=1}^K p_y (\mathbf{p}\mathbf{p}^\top) = \mathbf{p}\mathbf{p}^\top$.
   Summing these terms:
   $$\mathbf{F}(\boldsymbol{\theta}) = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$$

3. **Null Space**:
   Multiply $\mathbf{F}(\boldsymbol{\theta})$ by the all-ones vector $\mathbf{1} = [1, 1, \dots, 1]^\top$:
   $$\mathbf{F}\mathbf{1} = \text{diag}(\mathbf{p})\mathbf{1} - \mathbf{p}(\mathbf{p}^\top \mathbf{1}) = \mathbf{p} - \mathbf{p}(1) = \mathbf{0}$$
   The null space is spanned by $\mathbf{1}$. This reflects the shift-invariance of the softmax function: adding a constant $c$ to all logits leaves probabilities completely unchanged ($p_{\boldsymbol{\theta} + c \mathbf{1}} = p_{\boldsymbol{\theta}}$).

---

## Exercise 3: Kronecker-Factored Approximate Curvature (K-FAC)

In deep neural networks, a linear layer computes $\mathbf{s} = \mathbf{W}\mathbf{a}$, where $\mathbf{a} \in \mathbb{R}^{d_{\text{in}}}$ is the input activation and $\mathbf{s} \in \mathbb{R}^{d_{\text{out}}}$ is the pre-activation. Let $\mathbf{g} = \nabla_{\mathbf{s}} \log p(y|\mathbf{x})$ be the output gradient.
The gradient with respect to weight matrix $\mathbf{W}$ is the outer product $\nabla_{\mathbf{W}} \log p = \mathbf{g} \mathbf{a}^\top$, or in vectorized form:
$$\mathbf{s}_{\mathbf{W}} = \text{vec}(\nabla_{\mathbf{W}} \log p) = \mathbf{a} \otimes \mathbf{g}$$

1. Write out the exact layer Fisher Information Matrix $\mathbf{F} = \mathbb{E}[(\mathbf{a} \otimes \mathbf{g})(\mathbf{a} \otimes \mathbf{g})^\top]$.
2. State the K-FAC approximation (Martens & Grosse, 2015) assuming statistical independence between activations $\mathbf{a}$ and backpropagated gradients $\mathbf{g}$.
3. Using the Kronecker product inversion property $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$, explain how K-FAC reduces the computational cost of inverting the Fisher matrix from $\mathcal{O}((d_{\text{in}} d_{\text{out}})^3)$ to $\mathcal{O}(d_{\text{in}}^3 + d_{\text{out}}^3)$.

### Solution 3:
1. **Exact Layer Fisher Matrix**:
   Using the Kronecker transpose identity $(\mathbf{u} \otimes \mathbf{v})^\top = \mathbf{u}^\top \otimes \mathbf{v}^\top$ and mixed-product property:
   $$(\mathbf{a} \otimes \mathbf{g})(\mathbf{a} \otimes \mathbf{g})^\top = (\mathbf{a} \otimes \mathbf{g})(\mathbf{a}^\top \otimes \mathbf{g}^\top) = (\mathbf{a}\mathbf{a}^\top) \otimes (\mathbf{g}\mathbf{g}^\top)$$
   Therefore, the exact Fisher matrix is:
   $$\mathbf{F} = \mathbb{E}\left[ (\mathbf{a}\mathbf{a}^\top) \otimes (\mathbf{g}\mathbf{g}^\top) \right]$$

2. **The K-FAC Approximation**:
   K-FAC approximates the expectation of the Kronecker product as the Kronecker product of the individual expectations:
   $$\mathbf{F}_{\text{KFAC}} \triangleq \mathbb{E}[\mathbf{a}\mathbf{a}^\top] \otimes \mathbb{E}[\mathbf{g}\mathbf{g}^\top] \triangleq \mathbf{A} \otimes \mathbf{S}$$
   where $\mathbf{A} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ is the activation covariance and $\mathbf{S} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{out}}}$ is the derivative covariance.

3. **Inversion Complexity**:
   - The exact matrix $\mathbf{F}$ has dimension $(d_{\text{in}} d_{\text{out}}) \times (d_{\text{in}} d_{\text{out}})$. Direct inversion requires:
     $$\mathcal{O}\left( (d_{\text{in}} d_{\text{out}})^3 \right) = \mathcal{O}(d_{\text{in}}^3 d_{\text{out}}^3)$$
     For $d_{\text{in}} = d_{\text{out}} = 4096$, this is $(1.67 \times 10^7)^3 \approx 4.7 \times 10^{21}$ operations (impossible).
   - Under K-FAC:
     $$\mathbf{F}_{\text{KFAC}}^{-1} = (\mathbf{A} \otimes \mathbf{S})^{-1} = \mathbf{A}^{-1} \otimes \mathbf{S}^{-1}$$
     We only need to invert $\mathbf{A}$ ($\mathcal{O}(d_{\text{in}}^3)$) and $\mathbf{S}$ ($\mathcal{O}(d_{\text{out}}^3)$).
     Total operations: $\mathcal{O}(4096^3 + 4096^3) \approx 1.3 \times 10^{11}$ operations, making exact second-order natural gradient updates practical on modern GPUs!
