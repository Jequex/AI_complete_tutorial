# Topic 7.1: Gradient Descent & Its Variants

---

## 1. Executive Summary & The 12-Year-Old Analogy

All machine learning models—from a simple linear regression predicting housing prices to a 405-billion-parameter Transformer generating code—share a common core challenge: **how do we adjust millions of numerical dials (weights) so that the model's errors become as small as possible?**

This is the problem of **continuous mathematical optimization**. The function that measures our model's errors across a dataset is called the **Loss Function** (or Cost Function), denoted $\mathcal{L}(\theta)$. Because we cannot simply solve $\nabla_\theta \mathcal{L}(\theta) = 0$ with algebra for complex neural networks, we must navigate the landscape iteratively using **Gradient Descent**.

```
                ⛰️ High Loss (Terrible Predictions)
               /  \
              /    \
             /   🚶 \  <- Gradient points UP the steepest hill
            /     ↓  \ <- We step DOWN in the opposite direction (-∇L)
           /          \
          /            \
        🏔️              \_____ 🏖️ Global Minimum (Perfect Predictions!)
```

---

### The 12-Year-Old Analogy: The Foggy Mountain Descent

Imagine you are dropped by helicopter at the top of an unfamiliar, pitch-black mountain covered in dense fog. You cannot see more than two inches ahead of your boots. Your mission is to reach the cozy warm cabin at the very bottom of the valley.

- **How do you move?** You carefully feel the ground beneath your feet with your hiking boots. You determine which direction slopes most steeply downwards, and you take a measured step in that direction. This is **Gradient Descent**: sensing the local slope (gradient) and stepping in the opposite direction.
- **The Step Size (Learning Rate $\eta$):** 
  - If your steps are microscopic (a millimeter per hour), you will freeze to death before reaching the cabin.
  - If you take blind, 50-foot running leaps, you might launch yourself off a cliff or overshoot the entire valley into another mountain range.
  - You need a calibrated stride: confident when the slope is clear, gentle when the valley flattens out.
- **Batch Gradient Descent (The Survey Expedition):** You stop and call every single resident in all 50 surrounding towns on the radio, compute an average of their topographic maps, and only then take a single step. It is mathematically pristine and guaranteed not to err, but it takes 12 hours per step!
- **Stochastic Gradient Descent (The Lone Radio Call):** You radio one random hiker on a walkie-talkie, ask which way they are tilting, and immediately jump in that direction. You move lightning fast, but your path is wildly erratic, zig-zagging like a pinball.
- **Mini-Batch SGD (The Squad of 32):** You travel with a squad of 32 trusted hikers right around you. You quickly poll the 32 people, average their footing, and take a quick, confident step. It balances speed, hardware efficiency, and directional stability.

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Physical Meaning |
| :--- | :--- | :--- |
| $\theta \in \mathbb{R}^d$ | Vector space | Model parameter vector (weights and biases of dimension $d$) |
| $\mathcal{L}(\theta) \in \mathbb{R}$ | Scalar field | Empirical risk / objective loss over the entire dataset of size $N$ |
| $\ell(f(x_i; \theta), y_i)$ | Scalar | Per-sample loss on the $i$-th training instance $(x_i, y_i)$ |
| $\nabla_\theta \mathcal{L}(\theta)$ | $\mathbb{R}^d$ vector | True gradient vector: direction of maximum instantaneous rate of increase |
| $-\nabla_\theta \mathcal{L}(\theta)$ | $\mathbb{R}^d$ vector | Negative gradient: direction of steepest local descent |
| $\eta$ or $\eta_t$ | $\mathbb{R}_{>0}$ | Learning rate (step size hyperparameter at iteration $t$) |
| $\mathcal{B} \subset \{1, \dots, N\}$ | Index set | Mini-batch of sample indices randomly sampled from the dataset |
| $B = |\mathcal{B}|$ | Integer | Mini-batch size (typically $16, 32, 64, \dots, 2048$) |
| $g_t$ | $\mathbb{R}^d$ vector | Stochastic gradient estimate computed over mini-batch $\mathcal{B}_t$ |
| $\xi_t = g_t - \nabla \mathcal{L}(\theta_t)$ | $\mathbb{R}^d$ vector | Gradient noise vector (zero-mean estimation error) |
| $L$ | $\mathbb{R}_{>0}$ | Lipschitz constant of the gradient ($\|\nabla \mathcal{L}(\theta_1) - \nabla \mathcal{L}(\theta_2)\| \le L \|\theta_1 - \theta_2\|$) |
| $H = \nabla^2 \mathcal{L}(\theta)$ | $\mathbb{R}^{d \times d}$ matrix | Hessian matrix of second-order partial derivatives (local curvature) |

---

## 3. First-Order Principles: Why the Negative Gradient?

Why do we step in the direction of $-\nabla \mathcal{L}(\theta)$? Can we mathematically prove that this produces the fastest decrease in loss for an infinitesimal step?

### 3.1 First-Order Taylor Expansion & Cauchy-Schwarz Proof

Let $\mathcal{L}: \mathbb{R}^d \to \mathbb{R}$ be a continuously differentiable function. Consider a small displacement vector $v \in \mathbb{R}^d$ with fixed Euclidean length $\|v\|_2 = 1$, and step size $\epsilon > 0$.

The first-order multivariate Taylor expansion of $\mathcal{L}$ around $\theta$ is:
$$\mathcal{L}(\theta + \epsilon v) = \mathcal{L}(\theta) + \epsilon \nabla \mathcal{L}(\theta)^T v + \mathcal{O}(\epsilon^2)$$

To achieve the maximum decrease in loss, we seek the unit direction $v^*$ that minimizes the directional derivative:
$$v^* = \arg\min_{\|v\|_2 = 1} \nabla \mathcal{L}(\theta)^T v$$

By the **Cauchy-Schwarz Inequality**:
$$|\nabla \mathcal{L}(\theta)^T v| \le \|\nabla \mathcal{L}(\theta)\|_2 \|v\|_2 = \|\nabla \mathcal{L}(\theta)\|_2$$

Therefore:
$$-\|\nabla \mathcal{L}(\theta)\|_2 \le \nabla \mathcal{L}(\theta)^T v \le \|\nabla \mathcal{L}(\theta)\|_2$$

The lower bound $-\|\nabla \mathcal{L}(\theta)\|_2$ is achieved if and only if $v$ is antiparallel (collinear and pointing in the opposite direction) to $\nabla \mathcal{L}(\theta)$:
$$v^* = -\frac{\nabla \mathcal{L}(\theta)}{\|\nabla \mathcal{L}(\theta)\|_2}$$

Scaling $v^*$ by an adaptive step magnitude proportional to $\|\nabla \mathcal{L}(\theta)\|_2$ gives the standard Gradient Descent update rule:
$$\Delta \theta = -\eta \nabla \mathcal{L}(\theta) \implies \theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)$$

$\blacksquare$

---

## 4. The Spectrum of Gradient Descent Variants

In supervised learning with training set $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, the total empirical loss is the arithmetic mean of individual sample losses:
$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell_i(\theta), \quad \text{where } \ell_i(\theta) \equiv \ell(f(x_i; \theta), y_i)$$

The true gradient is:
$$\nabla \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \nabla \ell_i(\theta)$$

```
┌────────────────────────────────────────────────────────────────────────┐
│                        GRADIENT ESTIMATION SPECTRUM                    │
│                                                                        │
│   Full Dataset (N)        Mini-Batch (B)         Single Sample (B=1)  │
│   [■■■■■■■■■■■■■■■■]      [■■■■            ]      [■               ]   │
│                                                                        │
│   Batch GD (BGD)          Mini-Batch SGD          Pure SGD             │
│   Variance: 0             Variance: σ²/B          Variance: σ²         │
│   Compute: O(N)           Compute: O(B)           Compute: O(1)        │
│   Hardware: Memory Bound  Hardware: Ideal GPU     Hardware: Underflow  │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Batch Gradient Descent (BGD)
- **Update Rule:**
  $$\theta_{t+1} = \theta_t - \eta \frac{1}{N} \sum_{i=1}^N \nabla \ell_i(\theta_t)$$
- **Properties:**
  - Deterministic loss reduction on convex landscapes.
  - Zero gradient variance ($\text{Var}(g_t) = 0$).
  - **Fatal Flaw:** Evaluating $\nabla \ell_i$ across billions of tokens or millions of images requires prohibitive memory (OOM) and gigabytes of throughput per parameter update.
  - Gets trapped easily in saddle points or poor local minima because there is no noise to shake it free.

### 4.2 Stochastic Gradient Descent (SGD)
- **Update Rule:**
  At each step $t$, draw a uniform random index $i_t \sim \text{Uniform}(\{1, \dots, N\})$:
  $$\theta_{t+1} = \theta_t - \eta \nabla \ell_{i_t}(\theta_t)$$
- **Properties:**
  - $g_t = \nabla \ell_{i_t}(\theta_t)$ is an **unbiased estimator** of the true gradient:
    $$\mathbb{E}_{i_t}[g_t] = \sum_{j=1}^N P(i_t = j) \nabla \ell_j(\theta_t) = \frac{1}{N} \sum_{j=1}^N \nabla \ell_j(\theta_t) = \nabla \mathcal{L}(\theta_t)$$
  - Extreme gradient variance $\text{Var}(g_t) = \sigma^2$.
  - The trajectory wildly oscillates; requires decreasing learning rates ($\eta_t \to 0$) to converge.
  - Poor GPU utilization: computing one sample at a time wastes massive matrix-multiplication hardware parallelism.

### 4.3 Mini-Batch Stochastic Gradient Descent
- **Update Rule:**
  At step $t$, sample mini-batch $\mathcal{B}_t \subset \{1, \dots, N\}$ of size $B = |\mathcal{B}_t|$ without replacement (or with replacement):
  $$g_t = \frac{1}{B} \sum_{i \in \mathcal{B}_t} \nabla \ell_i(\theta_t)$$
  $$\theta_{t+1} = \theta_t - \eta g_t$$
- **Variance Scaling Law:**
  Assuming sample gradients $\nabla \ell_i$ are independent with variance $\text{Var}(\nabla \ell_i) = \sigma^2$:
  $$\text{Var}(g_t) = \text{Var}\left(\frac{1}{B} \sum_{i=1}^B \nabla \ell_i\right) = \frac{1}{B^2} \sum_{i=1}^B \text{Var}(\nabla \ell_i) = \frac{B \sigma^2}{B^2} = \frac{\sigma^2}{B}$$
- **The Sweet Spot of Deep Learning:**
  - Increasing $B$ reduces variance by a factor of $1/B$.
  - Maximizes GPU tensor core occupancy (e.g. multiples of 32 or 64 align with warp sizes on NVIDIA GPUs).
  - Preserves enough stochastic noise to escape saddle points and shallow basins.

---

## 5. Convergence Analysis & Stability Limits

### 5.1 $L$-Smoothness & The Descent Lemma

A differentiable loss function $\mathcal{L}$ is **$L$-Lipschitz smooth** if its gradient is Lipschitz continuous with constant $L > 0$:
$$\|\nabla \mathcal{L}(\theta_1) - \nabla \mathcal{L}(\theta_2)\| \le L \|\theta_1 - \theta_2\|, \quad \forall \theta_1, \theta_2 \in \mathbb{R}^d$$

This implies an upper quadratic bound on the function (the **Descent Lemma**):
$$\mathcal{L}(\theta_{t+1}) \le \mathcal{L}(\theta_t) + \nabla \mathcal{L}(\theta_t)^T (\theta_{t+1} - \theta_t) + \frac{L}{2} \|\theta_{t+1} - \theta_t\|^2$$

Substituting the gradient update $\theta_{t+1} - \theta_t = -\eta \nabla \mathcal{L}(\theta_t)$:
$$\mathcal{L}(\theta_{t+1}) \le \mathcal{L}(\theta_t) - \eta \|\nabla \mathcal{L}(\theta_t)\|^2 + \frac{L \eta^2}{2} \|\nabla \mathcal{L}(\theta_t)\|^2$$
$$\mathcal{L}(\theta_{t+1}) \le \mathcal{L}(\theta_t) - \eta \left( 1 - \frac{L \eta}{2} \right) \|\nabla \mathcal{L}(\theta_t)\|^2$$

### 5.2 Critical Learning Rate Upper Bound
To guarantee that the loss decreases ($\mathcal{L}(\theta_{t+1}) < \mathcal{L}(\theta_t)$ whenever $\nabla \mathcal{L}(\theta_t) \ne 0$), we require:
$$1 - \frac{L \eta}{2} > 0 \implies \eta < \frac{2}{L}$$

The optimal step size that maximizes the guaranteed per-step decrease is found by maximizing $\eta(1 - L\eta/2)$, yielding:
$$\eta^* = \frac{1}{L} \implies \mathcal{L}(\theta_{t+1}) \le \mathcal{L}(\theta_t) - \frac{1}{2L} \|\nabla \mathcal{L}(\theta_t)\|^2$$

For a quadratic objective $f(\theta) = \frac{1}{2} \theta^T Q \theta$, the Lipschitz constant is the largest eigenvalue of the Hessian matrix: $L = \lambda_{\max}(Q)$. If $\eta \ge \frac{2}{\lambda_{\max}}$, the iterations oscillate with growing amplitude and **diverge to infinity**.

---

## 6. Gradient Noise & Implicit Regularization

Why does SGD generalize better than Batch Gradient Descent on test data, even when both achieve zero training loss?

### 6.1 SGD as a Stochastic Differential Equation (SDE)
We can decompose the mini-batch gradient $g_t$ into the true gradient plus zero-mean noise $\xi_t$:
$$g_t = \nabla \mathcal{L}(\theta_t) + \xi_t, \quad \mathbb{E}[\xi_t] = 0, \quad \text{Cov}(\xi_t) \approx \frac{\Sigma(\theta_t)}{B}$$

In the continuous-time limit ($\eta \to 0$), the discrete update $\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t) - \eta \xi_t$ approximates an **Itô Stochastic Differential Equation** (Langevin dynamics):
$$d\theta_t = -\nabla \mathcal{L}(\theta_t) dt + \sqrt{2 D(\theta_t)} \, dW_t$$

The diffusion coefficient (temperature) of this process scales directly with:
$$T_{\text{temp}} \propto \frac{\eta}{B}$$

```
Sharp Minimum (Overfitting)          Flat Minimum (Generalizing)
        |  |                                \           /
       /    \                                \         /
      /      \                                \_______/
   Very high curvature                    Low curvature (small Hessian eigenvalues)
   Tiny shift in test distribution       Tiny shift in test distribution
   causes catastrophic loss explosion!    maintains low loss!
```

- **Sharp Minima:** Have huge Hessian eigenvalues $\lambda_{\max}$. The gradient noise fluctuations $\xi_t \sim \mathcal{N}\left(0, \frac{\eta}{B}\Sigma\right)$ easily kick the parameters out of narrow ravines.
- **Flat Minima:** Have small curvature. The noise cannot dislodge the parameters from wide basins.
- **Empirical Implication:** A larger learning rate $\eta$ and smaller batch size $B$ increase the noise scale $\eta/B$, driving the optimizer toward flatter regions that generalize substantially better to unseen test distributions.

---

## 7. Learning Rate Schedules

Using a constant learning rate throughout training is suboptimal: early in training, we need large steps to traverse vast plateaus; late in training, stochastic noise prevents convergence to the minimum unless the step size decays.

### 7.1 Robbins-Monro Conditions
For stochastic approximation to asymptotically converge almost surely to a local stationary point ($\theta_t \to \theta^*$ as $t \to \infty$), the learning rate sequence $\{\eta_t\}$ must satisfy:
1. **Infinite Reach:** $\sum_{t=1}^\infty \eta_t = \infty$ (the optimizer must be capable of reaching any point regardless of initialization distance).
2. **Finite Energy / Noise Damping:** $\sum_{t=1}^\infty \eta_t^2 < \infty$ (the accumulated variance of the stochastic gradient steps must extinguish).

*Example:* $\eta_t = \frac{\eta_0}{t}$ satisfies both conditions, but decays too aggressively in deep learning practice.

### 7.2 Practical Modern Schedules

1. **Step Decay:**
   $$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / S \rfloor}, \quad \gamma \in (0, 1), \; S = \text{step epoch period}$$
2. **Exponential Decay:**
   $$\eta_t = \eta_0 \cdot e^{-k t}$$
3. **Cosine Annealing (Loshchilov & Hutter):**
   $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left(1 + \cos\left(\frac{t}{T_{\max}} \pi\right)\right)$$
   Smoothly decreases without abrupt drops, keeping the optimizer active and settling gently into wide minima.
4. **Linear Warmup:**
   For $t \le T_{\text{warmup}}$:
   $$\eta_t = \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}}$$
   Critical for large-batch training and Transformer architectures to prevent destructive updates when initial gradients are uncalibrated.

---

## 8. Summary Table: Optimization Flavors

| Metric / Property | Batch GD | Stochastic GD (SGD) | Mini-Batch SGD |
| :--- | :--- | :--- | :--- |
| **Batch Size $B$** | $N$ (Full Dataset) | $1$ | $16 \le B \le 2048$ |
| **Compute Cost per Step** | $\mathcal{O}(N \cdot d)$ | $\mathcal{O}(d)$ | $\mathcal{O}(B \cdot d)$ |
| **Gradient Variance** | $0$ (Exact) | $\sigma^2$ (Very High) | $\sigma^2 / B$ (Tunable) |
| **Convergence Rate (Convex)** | $\mathcal{O}(1/T)$ | $\mathcal{O}(1/\sqrt{T})$ | $\mathcal{O}(1/\sqrt{T}) + \text{batch gain}$ |
| **Memory Footprint** | Extremely High | Minimal | Predictable, GPU-optimized |
| **Saddle Escape Ability** | Poor (Gets Stuck) | Very High (Erratic) | Strong & Controlled |
| **Generalization Quality** | Often sharp minima | Flat minima | Flat minima (optimal) |
