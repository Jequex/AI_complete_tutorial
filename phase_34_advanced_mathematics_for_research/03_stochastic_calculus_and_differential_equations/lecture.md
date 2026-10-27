# Topic 03: Stochastic Calculus & Differential Equations

## 1. Introduction & 12-Year-Old Intuition

Imagine you drop a tiny specks of food coloring into a clear glass of warm water. 

At the microscopic level, trillions of invisible water molecules are violently jiggling around, colliding with the dye particle from every possible direction millions of times per second. If you look through a microscope, the dye particle doesn't move smoothly—it jitters and zigs and zags in a jagged, erratic dance called **Brownian motion**.

Now take two steps back and look at the whole glass with your naked eye:
The jagged chaos disappears! The drop of dye smoothly, predictably expands outwards into a perfect spherical cloud of color until the entire glass has an even tint.

This dual perspective is the entire domain of **Stochastic Calculus**:
1. **The Microscopic Path (SDEs & Ito's Lemma)**: How an individual particle or system trajectory bounces around when buffeted by random noise ($dW_t$).
2. **The Macroscopic Cloud (Fokker-Planck Equation)**: How the probability distribution of where the particle might be spreads out smoothly through space over time like heat.

In modern AI research, this is the exact mathematical foundation behind **Continuous Score-Based Diffusion Models** (such as Stable Diffusion, Midjourney, and DALL-E 3). Diffusion models turn images into pure random Gaussian noise using a forward Stochastic Differential Equation (SDE), and then generate brand new photographic images from scratch by running the microscopic movie backwards using a **Reverse-Time SDE**!

---

## 2. Complete Mathematical Symbol Decoder

| Symbol | Mathematical Entity | Dimensionality / Domain | Operational & Research Role |
| :--- | :--- | :--- | :--- |
| $W_t$ | Standard Brownian Motion | Continuous random process | Wiener process with stationary independent Gaussian increments $W_t - W_s \sim \mathcal{N}(0, t - s)$. |
| $dW_t$ | Stochastic Differential | Infinitesimal increment | Gaussian noise increment with mean zero and variance $dt$: $\mathbb{E}[dW_t] = 0$, $(dW_t)^2 = dt$. |
| $X_t$ | Stochastic State Variable | $\mathbb{R}^d$ or $\mathbb{R}$ | Continuous random trajectory (e.g. latent image representation, asset price). |
| $\boldsymbol{\mu}(X_t, t)$ | Drift Coefficient | $\mathbb{R}^d$ | Deterministic velocity pulling the trajectory in a preferred direction. |
| $\boldsymbol{\sigma}(X_t, t)$ | Diffusion Coefficient | $\mathbb{R}^{d \times m}$ | Volatility / noise amplitude modulating the Brownian fluctuations. |
| $df(t, X_t)$ | Ito Differential | Scalar differential | Full Taylor-Ito expansion including the second-order $(dX_t)^2$ term. |
| $p(x, t)$ | Probability Density Function | Non-negative density | Spatial distribution of particle positions across time: $\int p(x, t) dx = 1$. |
| $\nabla_x \log p_t(x)$ | Score Function | $\mathbb{R}^d$ | Stein score pointing toward regions of higher data probability density. |
| $\mathbf{s}_\theta(x, t)$ | Score Neural Network | Function $\mathbb{R}^d \times [0, T] \to \mathbb{R}^d$ | Denoiser network trained by Denoising Score Matching to approximate $\nabla_x \log p_t(x)$. |
| $\theta$ | Reversion Rate (in OU) | Positive real scalar | Strength of restorative spring force pulling $X_t$ toward long-term mean $\mu$. |

---

## 3. Brownian Motion & The Wiener Process

A standard one-dimensional Brownian motion (Wiener process) $\{W_t\}_{t \ge 0}$ is a continuous stochastic process characterized by:
1. **Initial Condition**: $W_0 = 0$ almost surely.
2. **Independent Increments**: For any partition $0 \le t_1 < t_2 < \dots < t_k$, the increments $W_{t_2} - W_{t_1}, \dots, W_{t_k} - W_{t_{k-1}}$ are mutually independent.
3. **Gaussian Increments**: For $0 \le s < t$, the increment $W_t - W_s \sim \mathcal{N}(0, t - s)$.
4. **Continuous Paths**: $t \mapsto W_t$ is continuous with probability 1.

### Pathological Properties:
- **Nowhere Differentiable**: Almost every realization of $W_t$ is nowhere differentiable. The classical derivative $\frac{dW_t}{dt}$ does not exist in standard calculus.
- **Infinite Variation**: The total variation of $W_t$ over any finite interval $[0, T]$ is infinite with probability 1.
- **Quadratic Variation**: Over any partition $\Pi = \{t_0, t_1, \dots, t_N\}$ of $[0, T]$ with mesh size $|\Pi| \to 0$:
  $$\lim_{|\Pi| \to 0} \sum_{i=1}^N (W_{t_i} - W_{t_{i-1}})^2 = T \quad \text{in } L^2$$
  Symbolically written as the fundamental Ito multiplication table:
  $$(dW_t)^2 = dt, \quad dW_t \cdot dt = 0, \quad (dt)^2 = 0$$

---

## 4. Ito's Lemma: Calculus in a Fluctuating World

In ordinary calculus, if $x(t)$ is smooth and differentiable, the chain rule gives:
$$df = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x} dx$$

In stochastic calculus, because $(dX_t)^2$ contains $(dW_t)^2 = dt$, the second-order Taylor term **cannot be discarded**!

```
                    Ordinary Calculus vs. Ito's Calculus
                    
  Ordinary Smooth Variable x(t):
  dx ~ O(dt)
  (dx)^2 ~ O(dt^2) ───▶ Discarded (Zero in limit dt -> 0)
  df = f_t dt + f_x dx
  
  Stochastic Diffusion Variable X_t:
  dX_t = mu dt + sigma dW_t
  (dX_t)^2 = sigma^2 (dW_t)^2 = sigma^2 dt ~ O(dt)!
  Second-order term survives:
  df = f_t dt + f_x dX_t + (1/2) f_xx (dX_t)^2
```

### 4.1 Statement of Ito's Lemma
Let $X_t$ satisfy the Stochastic Differential Equation:
$$dX_t = \mu(X_t, t) dt + \sigma(X_t, t) dW_t$$
Let $f(t, x): [0, \infty) \times \mathbb{R} \to \mathbb{R}$ be twice continuously differentiable ($C^{1, 2}$). Then the process $Y_t = f(t, X_t)$ is also an Ito process, with stochastic differential:
$$df(t, X_t) = \left( \frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{1}{2} \sigma^2 \frac{\partial^2 f}{\partial x^2} \right) dt + \sigma \frac{\partial f}{\partial x} dW_t$$

### 4.2 Application: Exact Solution to Geometric Brownian Motion (GBM)
Consider the Black-Scholes / population growth SDE:
$$dX_t = \mu X_t dt + \sigma X_t dW_t, \quad X_0 > 0$$
Apply Ito's Lemma to the function $f(x) = \ln(x)$:
1. $\frac{\partial f}{\partial t} = 0$
2. $\frac{\partial f}{\partial x} = \frac{1}{x}$
3. $\frac{\partial^2 f}{\partial x^2} = -\frac{1}{x^2}$
4. Substituting into Ito's formula:
   $$d(\ln X_t) = \left(0 + (\mu X_t)\frac{1}{X_t} + \frac{1}{2}(\sigma X_t)^2\left(-\frac{1}{X_t^2}\right)\right) dt + (\sigma X_t)\frac{1}{X_t} dW_t$$
   $$d(\ln X_t) = \left(\mu - \frac{1}{2}\sigma^2\right) dt + \sigma dW_t$$
5. Integrating from $0$ to $t$:
   $$\ln(X_t) - \ln(X_0) = \left(\mu - \frac{1}{2}\sigma^2\right) t + \sigma W_t$$
   $$X_t = X_0 \exp\left( \left(\mu - \frac{1}{2}\sigma^2\right) t + \sigma W_t \right)$$

---

## 5. The Fokker-Planck (Kolmogorov Forward) Equation

While an individual path $X_t$ is unpredictable, the **probability density function** $p(x, t)$ of finding the particle at position $x$ at time $t$ obeys a deterministic Partial Differential Equation (PDE).

For an SDE $dX_t = \mu(X_t, t) dt + \sigma(X_t, t) dW_t$, the evolution of $p(x, t)$ is given by the **Fokker-Planck Equation**:
$$\frac{\partial p(x, t)}{\partial t} = -\frac{\partial}{\partial x}\Big[\mu(x, t) p(x, t)\Big] + \frac{1}{2} \frac{\partial^2}{\partial x^2}\Big[\sigma(x, t)^2 p(x, t)\Big]$$

### Stationary Distribution of the Ornstein-Uhlenbeck (OU) Process
For $dX_t = \theta (\mu - X_t) dt + \sigma dW_t$ with $\theta > 0$:
At steady state, $\frac{\partial p}{\partial t} = 0$:
$$0 = -\frac{d}{dx}[\theta(\mu - x) p(x)] + \frac{\sigma^2}{2} \frac{d^2 p(x)}{dx^2}$$
Integrating once with vanishing boundary conditions at $\pm \infty$:
$$\theta(x - \mu) p(x) + \frac{\sigma^2}{2} \frac{dp(x)}{dx} = 0 \implies \frac{dp}{p} = -\frac{2\theta(x - \mu)}{\sigma^2} dx$$
Integrating yields the Gaussian distribution:
$$p_{\infty}(x) = \frac{1}{\sqrt{2\pi \frac{\sigma^2}{2\theta}}} \exp\left( -\frac{(x - \mu)^2}{2 \left(\frac{\sigma^2}{2\theta}\right)} \right) = \mathcal{N}\left(\mu, \frac{\sigma^2}{2\theta}\right)$$

---

## 6. Score-Based Diffusion Models: The SDE Framework

In landmark work by Yang Song et al. (2020), diffusion models were unified into continuous-time stochastic differential equations.

```
                   Continuous Score-Based Diffusion SDE
                   
  Data Distribution p_0(x) ─────────────────────────────▶ Prior Noise p_T(x) ~ N(0, I)
      Forward SDE: dx = f(x, t) dt + g(t) dw
      (Gradually perturbs image into pure white noise)
      
  Generated Sample x_0 ~ p_0 ◀─────────────────────────── Prior Noise x_T ~ N(0, I)
      Reverse SDE: dx = [f(x, t) - g(t)^2 ∇_x log p_t(x)] dt + g(t) dw_bar
      (Denoises Gaussian noise into realistic image using learned score s_θ)
```

### 6.1 Anderson's Reverse-Time SDE Theorem
Let $X_t \in \mathbb{R}^d$ be governed by forward SDE:
$$d\mathbf{x} = \mathbf{f}(\mathbf{x}, t) dt + g(t) d\mathbf{w}$$
Anderson (1982) proved that the reverse-time process (traveling backwards from $t=T$ to $t=0$) also satisfies an Ito SDE:
$$d\mathbf{x} = \left[ \mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \right] dt + g(t) d\bar{\mathbf{w}}$$
where $d\bar{\mathbf{w}}$ is a standard Brownian motion running backward in time.

### 6.2 Denoising Score Matching
The only quantity required to simulate the reverse process is the **score function** $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$, the gradient of the log-density.
Since $p_t(\mathbf{x})$ is unknown, neural networks $\mathbf{s}_\theta(\mathbf{x}, t)$ are trained via Denoising Score Matching:
$$\min_\theta \mathbb{E}_{t \sim [0, T], \mathbf{x}_0 \sim p_{\text{data}}, \mathbf{x}_t \sim p_{0t}(\cdot|\mathbf{x}_0)} \left[ \left\| \mathbf{s}_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log p_{0t}(\mathbf{x}_t | \mathbf{x}_0) \right\|_2^2 \right]$$
Once trained, the reverse SDE is solved from $t=T$ to $0$ using Euler-Maruyama, generating high-fidelity images directly from noise!
