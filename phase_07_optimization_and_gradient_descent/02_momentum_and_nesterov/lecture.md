# Topic 7.2: Momentum & Nesterov Accelerated Gradient

---

## 1. Executive Summary & The 12-Year-Old Analogy

Standard Gradient Descent behaves like a mass-less particle moving through molasses: the instant the slope levels off or changes direction, the particle stops or jerks perpendicularly. In machine learning loss landscapes, this causes catastrophic slowdowns in **ill-conditioned ravines**—long, narrow canyons where the slope is thousands of times steeper across the walls than down the valley floor.

To overcome this, we endow our optimization particle with **physical momentum and mass**.

```
Standard SGD in a Ravine:                 Momentum in a Ravine:
   | \        / |                           | \        / |
   |   \  /\ /  |  <- Wild oscillations     |   \      /  |  <- Transverse bounces
   |    \/  \   |     across steep walls;   |    \    /   |     cancel out;
   |            |     creeps forward slowly |     \  /    |     accelerates smoothly
   |      ↓     |                           |      ═════> |     down the valley floor!
```

---

### The 12-Year-Old Analogy: The Bobsled on an Icy Half-Pipe

Imagine a V-shaped icy half-pipe that gently slopes downward towards a cabin at the finish line:

1. **Standard SGD (The Feather):** 
   A feather is dropped into the half-pipe. The wind from the steep walls blows it violently back and forth between the left and right walls. Because it has no mass or momentum, 99.9% of its energy is spent bouncing left-to-right, and it barely creeps forward toward the cabin.
2. **Polyak Momentum (The Heavy Iron Sled):**
   You place an 800-pound iron bobsled in the half-pipe. As it slides down, the leftward push from one wall is cancelled out by the rightward push from the opposite wall on the next bounce. Meanwhile, the gentle downward slope pushes in the *same forward direction* at every single instant. Momentum accumulates forward like a runaway freight train!
3. **Nesterov Accelerated Gradient (The Sled Pilot with Binoculars):**
   The heavy iron sled is so fast that when it reaches a sharp turn at the bottom, it might blast right over the banked wall. Nesterov momentum puts a smart pilot with binoculars on the sled. The pilot looks ahead to where the sled's current momentum will carry it a second from now, samples the slope *at that future landing point*, and touches the brakes or steers *before* crashing!

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Physical Meaning |
| :--- | :--- | :--- |
| $\theta_t \in \mathbb{R}^d$ | Vector space | Position vector (parameter weights) at iteration $t$ |
| $v_t \in \mathbb{R}^d$ | Vector space | Velocity vector (accumulated momentum) at iteration $t$ |
| $\beta \in [0, 1)$ | Scalar $\in [0, 1)$ | Momentum coefficient (decay / friction parameter, typically $0.9$) |
| $\eta$ | $\mathbb{R}_{>0}$ | Learning rate (step size multiplier) |
| $H = \nabla^2 f(\theta)$ | Matrix $\mathbb{R}^{d \times d}$ | Hessian matrix of second derivatives (local landscape curvature) |
| $\lambda_{\max}, \lambda_{\min}$ | $\mathbb{R}_{>0}$ | Maximum and minimum eigenvalues of the Hessian matrix |
| $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$ | $\mathbb{R}_{\ge 1}$ | **Condition Number** of the Hessian (degree of landscape anisotropy) |
| $\theta_t + \beta v_t$ | Vector space | Nesterov **Lookahead Position** |
| $\nabla f(\theta_t + \beta v_t)$ | Vector space | Nesterov lookahead gradient vector |
| $\tau = \frac{1}{1 - \beta}$ | Positive scalar | Effective memory horizon (e.g. $\beta=0.9 \implies \tau=10$ steps) |

---

## 3. The Ravine Problem & Ill-Conditioned Curvature

Consider an anisotropic quadratic objective function modeling a long, narrow canyon:
$$f(x, y) = \frac{1}{2} L x^2 + \frac{1}{2} \mu y^2, \quad \text{with } L \gg \mu > 0$$

- In the $x$-direction (across the walls), the curvature is $L$ (steep).
- In the $y$-direction (along the valley floor), the curvature is $\mu$ (shallow).
- The Hessian matrix is $H = \begin{bmatrix} L & 0 \\ 0 & \mu \end{bmatrix}$, and its condition number is:
  $$\kappa = \frac{L}{\mu} \gg 1$$

### 3.1 Standard Gradient Descent Failure
The gradient is $\nabla f(x, y) = [Lx, \, \mu y]^T$. The decoupled coordinate updates under Gradient Descent are:
$$x_{t+1} = (1 - \eta L) x_t, \quad y_{t+1} = (1 - \eta \mu) y_t$$

As proven in Topic 7.1, stability requires $\eta < \frac{2}{L}$ to prevent divergence in the steep $x$-direction. If we choose the optimal stable step size $\eta = \frac{2}{L + \mu} \approx \frac{2}{L}$:
- The $x$-contraction factor is $\left| 1 - \frac{2L}{L+\mu} \right| = \frac{L - \mu}{L + \mu} = \frac{\kappa - 1}{\kappa + 1}$.
- The $y$-contraction factor along the valley floor is:
  $$1 - \eta \mu = 1 - \frac{2\mu}{L+\mu} = 1 - \frac{2}{\kappa + 1} = \frac{\kappa - 1}{\kappa + 1}$$

To reduce the error along the valley floor by a factor of $\epsilon$, standard Gradient Descent requires:
$$\mathcal{O}\left( \kappa \ln(1/\epsilon) \right) \text{ steps}$$
If $\kappa = 10,000$ (common in deep neural networks), standard GD requires tens of thousands of steps!

---

## 4. Polyak Heavy-Ball Momentum

Boris Polyak (1964) proposed modeling optimization as a damped physical particle with mass $m$, subject to a potential field $f(\theta)$ and friction coefficient $\gamma$:
$$m \frac{d^2 \theta}{dt^2} + \gamma \frac{d\theta}{dt} + \nabla f(\theta) = 0$$

Discretizing this differential equation yields the **Polyak Heavy-Ball** update:
$$v_{t+1} = \beta v_t - \eta \nabla f(\theta_t)$$
$$\theta_{t+1} = \theta_t + v_{t+1}$$

### 4.1 Velocity Accumulation & Effective Step Size
Expanding the velocity recurrence recursively:
$$v_t = -\eta \sum_{k=0}^{t-1} \beta^k \nabla f(\theta_{t-1-k})$$

Suppose the gradients in a specific direction are constant and aligned: $\nabla f(\theta) \equiv g$. In the limit as $t \to \infty$:
$$v_\infty = -\eta g \sum_{k=0}^\infty \beta^k = -\frac{\eta}{1 - \beta} g$$

- When $\beta = 0.9$, $\frac{1}{1 - 0.9} = 10$. The effective learning rate in the consistent direction is **accelerated by 10x**!
- In an oscillating direction where gradients flip signs alternate steps ($g, -g, g, -g$), the terms cancel:
  $$v_t \approx -\eta g (1 - \beta + \beta^2 - \beta^3 \dots) = -\frac{\eta}{1 + \beta} g \approx -\frac{\eta}{1.9} g$$
  The oscillations are **damped by ~50%**!

### 4.2 Theoretical Convergence Acceleration
For strongly convex functions with condition number $\kappa$, setting the optimal hyperparameters:
$$\beta^* = \left( \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \right)^2, \quad \eta^* = \frac{4}{(\sqrt{L} + \sqrt{\mu})^2}$$
achieves the accelerated convergence rate:
$$\|\theta_t - \theta^*\| \le C \left( \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \right)^t \|\theta_0 - \theta^*\|$$

The number of iterations required drops from $\mathcal{O}(\kappa)$ to **$\mathcal{O}(\sqrt{\kappa})$**!
If $\kappa = 10,000$, $\sqrt{\kappa} = 100$, yielding a **100-fold theoretical speedup**.

---

## 5. Nesterov Accelerated Gradient (NAG)

While Polyak momentum accelerates dramatically, it often suffers from severe **overshooting**: because velocity accumulates high kinetic energy, the particle rockets past the minimum and oscillates around it before settling.

Yurii Nesterov (1983) introduced a brilliant modification: **lookahead gradient evaluation**.

```
Polyak Momentum:
  θ_t ───────────────> (Step with accumulated velocity + current gradient)
                         ∇f(θ_t) evaluated at current point

Nesterov Momentum:
  θ_t ──[lookahead]──> θ_lookahead = θ_t + β v_t
                             │
                             ▼  Evaluates ∇f(θ_lookahead)
                       Applies brake if slope is rising!
```

### 5.1 Nesterov Formulation
Instead of evaluating the gradient at the current position $\theta_t$, Nesterov evaluates it at the **projected future position** $\theta_t + \beta v_t$:
$$\theta_{\text{lookahead}} = \theta_t + \beta v_t$$
$$v_{t+1} = \beta v_t - \eta \nabla f(\theta_{\text{lookahead}})$$
$$\theta_{t+1} = \theta_t + v_{t+1}$$

### 5.2 Why NAG Acts as an Adaptive Brake
Notice what happens as the optimizer approaches a minimum and starts climbing the opposite wall:
1. $v_t$ is still carrying the particle forward.
2. $\theta_{\text{lookahead}} = \theta_t + \beta v_t$ is pushed further up the rising slope.
3. The gradient $\nabla f(\theta_{\text{lookahead}})$ is steep and points backward, opposite to $v_t$.
4. This strong backward gradient subtracts from $\beta v_t$, **immediately applying the brakes before the particle actually overshoots**!

### 5.3 PyTorch Equivalent Implementation
In PyTorch (`torch.optim.SGD(..., momentum=0.9, nesterov=True)`), storing $\theta_{\text{lookahead}}$ separately is computationally inconvenient. By a change of variables $\hat{\theta}_t = \theta_t + \beta v_t$, PyTorch implements Nesterov momentum using only the standard gradient:
$$v_{t+1} = \beta v_t + \nabla f(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta \left( \nabla f(\theta_t) + \beta v_{t+1} \right)$$
This formulation mathematically matches Nesterov's trajectory while requiring only one gradient evaluation at $\theta_t$ per step.

---

## 6. Summary Comparison: GD vs Polyak vs NAG

| Feature | Standard GD | Polyak Momentum | Nesterov (NAG) |
| :--- | :--- | :--- | :--- |
| **Update Formulation** | $\theta - \eta \nabla f(\theta)$ | $v_{t+1} = \beta v_t - \eta \nabla f(\theta)$ | $v_{t+1} = \beta v_t - \eta \nabla f(\theta + \beta v_t)$ |
| **Gradient Evaluation Point** | Current position $\theta_t$ | Current position $\theta_t$ | Lookahead position $\theta_t + \beta v_t$ |
| **Convergence Rate (Convex)** | $\mathcal{O}(1/T)$ | $\mathcal{O}(1/T)$ | $\mathcal{O}(1/T^2)$ (Optimal first-order) |
| **Convergence Rate (Strongly Convex)**| $\frac{\kappa - 1}{\kappa + 1}$ | $\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$ | $\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$ |
| **Ravine Oscillation Damping** | None (Severe bouncing) | High (Bounces cancel) | Maximum (Lookahead braking) |
| **Overshoot Tendency** | Low | High | Significantly reduced |
