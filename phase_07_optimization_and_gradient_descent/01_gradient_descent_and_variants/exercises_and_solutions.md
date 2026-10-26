# Topic 7.1: Exercises & Solutions — Gradient Descent & Variants

---

### Problem 1: Critical Learning Rate & Divergence Bound (Medium)

Consider the 2D anisotropic quadratic bowl:
$$f(x, y) = 10 x^2 + y^2$$

1. Compute the gradient $\nabla f(x, y)$ and the Hessian matrix $H = \nabla^2 f(x, y)$.
2. Determine the Lipschitz smoothness constant $L$ of the gradient.
3. Write down the exact coordinate update equations for Gradient Descent with constant learning rate $\eta$: $(x_{t+1}, y_{t+1})$ in terms of $(x_t, y_t)$.
4. Find the exact range of positive step sizes $\eta > 0$ for which the iterates $(x_t, y_t)$ converge to the global minimum $(0, 0)$ from any initial point $(x_0, y_0) \ne (0, 0)$. What happens if $\eta = 0.1$? What happens if $\eta = 0.105$?

#### Solution
1. **Gradient & Hessian:**
   $$\nabla f(x, y) = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix} = \begin{bmatrix} 20 x \\ 2 y \end{bmatrix}$$
   $$H = \nabla^2 f(x, y) = \begin{bmatrix} 20 & 0 \\ 0 & 2 \end{bmatrix}$$

2. **Lipschitz Constant $L$:**
   For a quadratic function $f(\theta) = \frac{1}{2} \theta^T Q \theta$, the gradient is $\nabla f(\theta) = Q \theta$. The Lipschitz constant is given by the spectral norm (maximum eigenvalue) of $Q$:
   $$L = \lambda_{\max}(H) = 20$$

3. **Coordinate Update Equations:**
   $$x_{t+1} = x_t - \eta \frac{\partial f}{\partial x} = x_t - 20\eta x_t = (1 - 20\eta) x_t$$
   $$y_{t+1} = y_t - \eta \frac{\partial f}{\partial y} = y_t - 2\eta y_t = (1 - 2\eta) y_t$$
   By induction:
   $$x_t = (1 - 20\eta)^t x_0, \quad y_t = (1 - 2\eta)^t y_0$$

4. **Convergence Range:**
   For $(x_t, y_t) \to (0, 0)$ as $t \to \infty$, the contraction factors must satisfy:
   $$|1 - 20\eta| < 1 \iff -1 < 1 - 20\eta < 1 \iff 0 < 20\eta < 2 \iff 0 < \eta < \frac{2}{20} = 0.1$$
   $$|1 - 2\eta| < 1 \iff 0 < \eta < 1$$
   The restrictive dimension is $x$, which has higher curvature ($L=20$). Thus, convergence requires:
   $$0 < \eta < 0.1$$
   - **Case $\eta = 0.1$:**
     $1 - 20(0.1) = -1$.
     Then $x_t = (-1)^t x_0$. The $x$ coordinate does not converge; it oscillates indefinitely between $+x_0$ and $-x_0$ with constant amplitude!
   - **Case $\eta = 0.105$:**
     $1 - 20(0.105) = 1 - 2.1 = -1.1$.
     Then $x_t = (-1.1)^t x_0$. Because $|-1.1| > 1$, the iterate oscillates with exponentially exploding amplitude ($|x_t| \to \infty$), causing immediate divergence.

---

### Problem 2: Mini-Batch Gradient Variance Scaling (Easy)

Suppose a dataset contains $N$ samples. Let $g_i = \nabla \ell(f(x_i; \theta), y_i)$ be the gradient on the $i$-th sample. Suppose the sample gradients are independently and identically distributed with population mean $\mathbb{E}[g_i] = \mu \in \mathbb{R}^d$ and covariance matrix $\text{Cov}(g_i) = \Sigma \in \mathbb{R}^{d \times d}$.

Let the mini-batch gradient estimator for batch size $B$ be:
$$g_B = \frac{1}{B} \sum_{k=1}^B g_{i_k}$$
where each index $i_k$ is drawn uniformly and independently at random with replacement.

1. Prove that $g_B$ is an unbiased estimator of $\mu$.
2. Prove that $\text{Cov}(g_B) = \frac{1}{B} \Sigma$.
3. Compute how many times more noisy a single-sample SGD estimator ($B=1$) is compared to a GPU mini-batch of size $B=256$.

#### Solution
1. **Unbiased Expectation:**
   By the linearity of expectation:
   $$\mathbb{E}[g_B] = \mathbb{E}\left[ \frac{1}{B} \sum_{k=1}^B g_{i_k} \right] = \frac{1}{B} \sum_{k=1}^B \mathbb{E}[g_{i_k}] = \frac{1}{B} (B \mu) = \mu$$
   Therefore, $g_B$ is an unbiased estimator of the true gradient $\mu = \nabla \mathcal{L}(\theta)$.

2. **Covariance Derivation:**
   Since the samples are drawn independently, all cross-terms $\text{Cov}(g_{i_j}, g_{i_k}) = 0$ for $j \ne k$:
   $$\text{Cov}(g_B) = \text{Cov}\left( \frac{1}{B} \sum_{k=1}^B g_{i_k} \right) = \frac{1}{B^2} \sum_{k=1}^B \text{Cov}(g_{i_k}) = \frac{1}{B^2} (B \Sigma) = \frac{1}{B} \Sigma$$

3. **Noise Ratio Comparison:**
   For $B=1$, the covariance is $\text{Cov}(g_1) = \Sigma$.
   For $B=256$, the covariance is $\text{Cov}(g_{256}) = \frac{1}{256} \Sigma$.
   The variance of the single-sample SGD gradient is exactly **256 times larger** than that of the $B=256$ mini-batch estimator.

---

### Problem 3: Cosine Annealing Learning Rate Schedule (Easy)

A model is trained for $T_{\max} = 100$ epochs using cosine annealing without restarts.
The initial maximum learning rate is $\eta_{\max} = 10^{-3}$ and the final target minimum learning rate is $\eta_{\min} = 10^{-5}$.

The cosine annealing formula is:
$$\eta_t = \eta_{\min} + \frac{1}{2} (\eta_{\max} - \eta_{\min}) \left(1 + \cos\left(\frac{t \pi}{T_{\max}}\right)\right)$$

1. Compute $\eta_0$ (at epoch 0).
2. Compute $\eta_{25}$ (at epoch 25).
3. Compute $\eta_{50}$ (at epoch 50).
4. Compute $\eta_{100}$ (at the final epoch).

#### Solution
We have $\eta_{\max} - \eta_{\min} = 0.00100 - 0.00001 = 0.00099$.
Thus:
$$\eta_t = 0.00001 + \frac{0.00099}{2} \left(1 + \cos\left(\frac{t \pi}{100}\right)\right) = 0.00001 + 0.000495 \left(1 + \cos\left(\frac{t \pi}{100}\right)\right)$$

1. **At $t = 0$:**
   $\cos(0) = 1 \implies 1 + \cos(0) = 2$.
   $$\eta_0 = 0.00001 + 0.000495(2) = 0.00001 + 0.00099 = 0.00100 = 10^{-3}$$

2. **At $t = 25$:**
   $\cos\left(\frac{25\pi}{100}\right) = \cos\left(\frac{\pi}{4}\right) = \frac{\sqrt{2}}{2} \approx 0.7071$.
   $$1 + \cos(\pi/4) \approx 1.7071$$
   $$\eta_{25} = 0.00001 + 0.000495(1.7071) \approx 0.00001 + 0.0008450 = 0.000855 = 8.55 \times 10^{-4}$$

3. **At $t = 50$:**
   $\cos\left(\frac{50\pi}{100}\right) = \cos\left(\frac{\pi}{2}\right) = 0$.
   $$1 + \cos(\pi/2) = 1.0$$
   $$\eta_{50} = 0.00001 + 0.000495(1.0) = 0.000505 = 5.05 \times 10^{-4}$$

4. **At $t = 100$:**
   $\cos\left(\frac{100\pi}{100}\right) = \cos(\pi) = -1$.
   $$1 + \cos(\pi) = 0$$
   $$\eta_{100} = 0.00001 + 0.000495(0) = 0.00001 = 10^{-5}$$

---

### Problem 4: SGD Noise Temperature & Flat Minima (Hard)

Consider an objective function with two local minima:
- **Sharp Minimum $\theta_A$:** Curvature Hessian eigenvalue $\lambda_A = 100$.
- **Flat Minimum $\theta_B$:** Curvature Hessian eigenvalue $\lambda_B = 1$.

Model the optimization trajectory as a continuous Brownian motion with thermal noise variance $\sigma^2 \propto \frac{\eta}{B}$.

1. Explain why the escape probability from a potential well of depth $\Delta E$ under thermal fluctuations is given by the Arrhenius / Kramers escape rate:
   $$P_{\text{escape}} \propto \exp\left( - \frac{\Delta E}{k_B T_{\text{eff}}} \right), \quad \text{where } T_{\text{eff}} \approx \frac{\eta \Sigma}{2 B}$$
2. If training uses large batch size $B=4096$ with small learning rate $\eta=10^{-4}$, what happens to the effective temperature $T_{\text{eff}}$, and which minimum will the optimizer settle in?
3. If the batch size is reduced to $B=32$ with $\eta=10^{-2}$, how does $T_{\text{eff}}$ change, and why does test generalization improve?

#### Solution
1. **Escape Rate from Potential Wells:**
   According to Kramers' escape rate theory for stochastic differential equations, a particle in a potential well of depth $\Delta E$ subject to Gaussian thermal noise with effective temperature $T_{\text{eff}}$ has escape probability per unit time proportional to:
   $$P_{\text{escape}} \propto \exp\left( -\frac{\Delta E}{T_{\text{eff}}} \right)$$
   For a sharp minimum, the basin is narrow and the curvature eigenvalues $\lambda$ are large. A small displacement kicks the loss up sharply, and high noise variance relative to the basin width easily causes the parameters to jump the barrier out of the basin.

2. **Large Batch / Small Learning Rate Regime:**
   $$T_{\text{eff}} \propto \frac{10^{-4}}{4096} \approx 2.44 \times 10^{-8}$$
   $T_{\text{eff}}$ is near zero (quasi-deterministic gradient descent). With almost no noise, the optimizer will fall into whichever local minimum is closest to initialization, including the sharp minimum $\theta_A$. At $\theta_A$, slight shifts between training and test data lead to massive loss spikes because the curvature is $\lambda_A = 100$.

3. **Small Batch / Moderate Learning Rate Regime:**
   $$T_{\text{eff}} \propto \frac{10^{-2}}{32} = 3.125 \times 10^{-4}$$
   The effective temperature is **over 12,800 times higher**! This intense stochastic noise destabilizes sharp minima: the parameters are repeatedly ejected from $\theta_A$. However, in flat minimum $\theta_B$ (curvature $\lambda_B = 1$), the basin is wide and gentle; the stochastic kicks are absorbed without overcoming the broader boundary. As a result, the optimizer preferentially settles in flat minima, which are robust to data distribution shifts and yield superior test generalization.
