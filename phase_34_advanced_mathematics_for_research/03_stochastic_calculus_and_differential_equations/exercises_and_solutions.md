# Topic 03: Stochastic Calculus & Differential Equations — Exercises & Solutions

## Exercise 1: Exact Moments of Geometric Brownian Motion via Ito's Lemma

Consider the Geometric Brownian Motion (GBM) SDE:
$$dX_t = \mu X_t dt + \sigma X_t dW_t, \quad X_0 > 0$$

1. Using the function $f(X_t) = X_t^p$ for integer $p \ge 1$, apply Ito's Lemma to derive the stochastic differential $d(X_t^p)$.
2. Take the expectation $\mathbb{E}[d(X_t^p)]$ to obtain an ordinary differential equation (ODE) for the $p$-th moment $m_p(t) \triangleq \mathbb{E}[X_t^p]$.
3. Solve this ODE to find the exact analytical formula for $m_1(t) = \mathbb{E}[X_t]$ and $m_2(t) = \mathbb{E}[X_t^2]$, and determine the variance $\text{Var}(X_t)$.

### Solution 1:
1. **Application of Ito's Lemma**:
   Let $f(x) = x^p$.
   - $\frac{\partial f}{\partial t} = 0$
   - $\frac{\partial f}{\partial x} = p x^{p-1}$
   - $\frac{\partial^2 f}{\partial x^2} = p(p-1) x^{p-2}$
   Applying Ito's formula with $dX_t = \mu X_t dt + \sigma X_t dW_t$:
   $$d(X_t^p) = \left( 0 + (\mu X_t)(p X_t^{p-1}) + \frac{1}{2} (\sigma X_t)^2 p(p-1) X_t^{p-2} \right) dt + (\sigma X_t)(p X_t^{p-1}) dW_t$$
   $$d(X_t^p) = \left( p \mu + \frac{1}{2} p(p-1) \sigma^2 \right) X_t^p dt + p \sigma X_t^p dW_t$$

2. **Expectation and Moment ODE**:
   Because the stochastic integral with respect to $dW_t$ is an Ito martingale with zero expectation ($\mathbb{E}\left[\int_0^t p \sigma X_s^p dW_s\right] = 0$):
   $$d \mathbb{E}[X_t^p] = \left( p \mu + \frac{1}{2} p(p-1) \sigma^2 \right) \mathbb{E}[X_t^p] dt$$
   $$\frac{d m_p(t)}{dt} = \left( p \mu + \frac{1}{2} p(p-1) \sigma^2 \right) m_p(t)$$

3. **Closed-Form Moments**:
   Solving the linear ODE with initial condition $m_p(0) = X_0^p$:
   $$m_p(t) = X_0^p \exp\left( \left( p \mu + \frac{1}{2} p(p-1) \sigma^2 \right) t \right)$$
   - For $p = 1$:
     $$\mathbb{E}[X_t] = X_0 e^{\mu t}$$
   - For $p = 2$:
     $$\mathbb{E}[X_t^2] = X_0^2 e^{(2\mu + \sigma^2) t}$$
   - Variance:
     $$\text{Var}(X_t) = \mathbb{E}[X_t^2] - (\mathbb{E}[X_t])^2 = X_0^2 e^{(2\mu + \sigma^2) t} - X_0^2 e^{2\mu t} = \mathbf{X_0^2 e^{2\mu t} \left( e^{\sigma^2 t} - 1 \right)}$$

---

## Exercise 2: Transition Kernel and Score for the Variance Preserving (VP) SDE

In continuous score-based diffusion models (Song et al. 2020), the Variance Preserving SDE is defined as:
$$d\mathbf{x} = -\frac{1}{2} \beta(t) \mathbf{x} dt + \sqrt{\beta(t)} d\mathbf{w}$$
with noise schedule $\beta(t) > 0$.

1. Using integrating factors, integrate the deterministic drift part to show that the conditional mean is $\mathbb{E}[\mathbf{x}_t \mid \mathbf{x}_0] = \mathbf{x}_0 e^{-\frac{1}{2} \int_0^t \beta(s) ds}$.
2. Compute the conditional variance $\text{Var}(\mathbf{x}_t \mid \mathbf{x}_0)$ and prove that the total variance remains preserved (i.e. $\alpha_t^2 + \sigma_t^2 = 1$).
3. Derive the exact conditional score function $\nabla_{\mathbf{x}_t} \log p_{0t}(\mathbf{x}_t \mid \mathbf{x}_0)$.

### Solution 2:
1. **Integrating Factor**:
   Let $\gamma(t) = \frac{1}{2} \int_0^t \beta(s) ds$. Multiply the SDE by the integrating factor $e^{\gamma(t)}$:
   $$d\left( e^{\gamma(t)} \mathbf{x}_t \right) = \gamma'(t) e^{\gamma(t)} \mathbf{x}_t dt + e^{\gamma(t)} d\mathbf{x}_t = \frac{1}{2} \beta(t) e^{\gamma(t)} \mathbf{x}_t dt + e^{\gamma(t)} \left( -\frac{1}{2}\beta(t)\mathbf{x}_t dt + \sqrt{\beta(t)} d\mathbf{w} \right)$$
   $$d\left( e^{\gamma(t)} \mathbf{x}_t \right) = e^{\gamma(t)} \sqrt{\beta(t)} d\mathbf{w}$$
   Integrating from $0$ to $t$:
   $$e^{\gamma(t)} \mathbf{x}_t - \mathbf{x}_0 = \int_0^t e^{\gamma(s)} \sqrt{\beta(s)} d\mathbf{w}_s \implies \mathbf{x}_t = e^{-\gamma(t)} \mathbf{x}_0 + \int_0^t e^{-(\gamma(t) - \gamma(s))} \sqrt{\beta(s)} d\mathbf{w}_s$$
   Taking expectations:
   $$\mathbb{E}[\mathbf{x}_t \mid \mathbf{x}_0] = \mathbf{x}_0 e^{-\frac{1}{2} \int_0^t \beta(s) ds}$$

2. **Conditional Variance**:
   By Ito Isometry, the variance of the stochastic integral is:
   $$\text{Var}(\mathbf{x}_t \mid \mathbf{x}_0) = \int_0^t e^{-2(\gamma(t) - \gamma(s))} \beta(s) ds = e^{-2\gamma(t)} \int_0^t e^{2\gamma(s)} \beta(s) ds$$
   Notice that $\frac{d}{ds} \left[ e^{2\gamma(s)} \right] = 2 \gamma'(s) e^{2\gamma(s)} = \beta(s) e^{2\gamma(s)}$. Therefore:
   $$\int_0^t e^{2\gamma(s)} \beta(s) ds = \left[ e^{2\gamma(s)} \right]_0^t = e^{2\gamma(t)} - 1$$
   Multiplying by $e^{-2\gamma(t)}$:
   $$\text{Var}(\mathbf{x}_t \mid \mathbf{x}_0) = 1 - e^{-2\gamma(t)} = 1 - e^{-\int_0^t \beta(s) ds}$$
   Defining $\alpha_t \triangleq e^{-\gamma(t)}$ and $\sigma_t^2 \triangleq 1 - \alpha_t^2$, we immediately see $\alpha_t^2 + \sigma_t^2 = 1$, which strictly preserves total unit variance.

3. **Conditional Score**:
   The conditional distribution is Gaussian: $\mathbf{x}_t \mid \mathbf{x}_0 \sim \mathcal{N}(\alpha_t \mathbf{x}_0, \sigma_t^2 \mathbf{I})$.
   $$\log p_{0t}(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{d}{2}\ln(2\pi \sigma_t^2) - \frac{\|\mathbf{x}_t - \alpha_t \mathbf{x}_0\|_2^2}{2 \sigma_t^2}$$
   Taking the spatial gradient with respect to $\mathbf{x}_t$:
   $$\nabla_{\mathbf{x}_t} \log p_{0t}(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{\mathbf{x}_t - \alpha_t \mathbf{x}_0}{\sigma_t^2} = -\frac{\mathbf{\epsilon}}{\sigma_t}$$
   where $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \mathbf{\epsilon}$. This directly links the score function to the noise prediction objective in DDPM.

---

## Exercise 3: Probability Current and Fokker-Planck Continuity

The Fokker-Planck equation can be expressed in continuity form:
$$\frac{\partial p(x, t)}{\partial t} + \frac{\partial J(x, t)}{\partial x} = 0$$
where $J(x, t)$ is the **probability current** (or flux).

1. Identify the explicit expression for $J(x, t)$ for SDE $dX_t = \mu(x, t) dt + \sigma(x, t) dW_t$.
2. In steady state ($\partial p / \partial t = 0$), explain why $J(x) = \text{const}$, and why for normalizable probability densities on $\mathbb{R}$ the constant must be exactly zero (detailed balance).
3. If $J(x) = 0$, solve the resulting first-order differential equation for $p_{\infty}(x)$ in terms of potential function $V(x) = -\int \mu(s) ds$ when $\sigma$ is constant.

### Solution 3:
1. **Probability Current Formulation**:
   Rewrite the Fokker-Planck equation:
   $$\frac{\partial p}{\partial t} = -\frac{\partial}{\partial x} \left[ \mu(x, t) p(x, t) - \frac{1}{2} \frac{\partial}{\partial x}(\sigma^2(x, t) p(x, t)) \right]$$
   Comparing with $\frac{\partial p}{\partial t} + \frac{\partial J}{\partial x} = 0$, the probability current is:
   $$J(x, t) = \mu(x, t) p(x, t) - \frac{1}{2} \frac{\partial}{\partial x}\left[\sigma^2(x, t) p(x, t)\right]$$

2. **Zero Current in Equilibrium**:
   At steady state, $\frac{\partial J}{\partial x} = 0 \implies J(x) = C$.
   For a normalizable probability distribution on $\mathbb{R}$, both $p(x) \to 0$ and $\frac{\partial p}{\partial x} \to 0$ as $x \to \pm \infty$.
   Evaluating $J(x)$ at $x \to \infty$ shows that $C = 0$.
   Therefore, at equilibrium, the probability flux vanishes everywhere: $J(x) = 0$.

3. **Boltzmann-Gibbs Stationary Distribution**:
   Setting $J(x) = 0$ with constant diffusion $\sigma$:
   $$\mu(x) p(x) - \frac{\sigma^2}{2} \frac{dp(x)}{dx} = 0 \implies \frac{1}{p(x)} \frac{dp}{dx} = \frac{2 \mu(x)}{\sigma^2}$$
   Integrating both sides:
   $$\ln p(x) = \frac{2}{\sigma^2} \int \mu(x) dx + \text{const} = -\frac{2 V(x)}{\sigma^2} + \text{const}$$
   $$p_{\infty}(x) = \frac{1}{Z} \exp\left( -\frac{2 V(x)}{\sigma^2} \right)$$
   This is precisely the canonical Boltzmann-Gibbs distribution in statistical physics, where the effective temperature is $k_B T = \frac{\sigma^2}{2}$.
