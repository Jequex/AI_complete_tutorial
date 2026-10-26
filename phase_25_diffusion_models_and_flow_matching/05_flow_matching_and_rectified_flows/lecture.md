# Topic 25.5: Flow Matching and Rectified Flows

---

## 1. Executive Summary & The 12-Year-Old Analogy

### Beyond Curved Stochastic Diffusion: Continuous Deterministic Vector Fields
While Denoising Diffusion Probabilistic Models (DDPM) unlocked unprecedented visual realism, they inherit fundamental computational inefficiencies:
1. **Curved, Tortuous Trajectories:** Forward Brownian motion scatters particles randomly; the reverse SDE must take tiny steps (typically 50 to 1,000 evaluations) to prevent numerical drift.
2. **Stochastic Inversion:** Reversing a stochastic process requires careful variance management, preventing deterministic encoding and exact latent reconstruction.

**Flow Matching** (Lipman et al., 2022; Albergo & Vanden-Eijnden, 2022) and **Rectified Flows** (Liu et al., 2022) revolutionize generative modeling by replacing stochastic diffusion with **deterministic Continuous Normalizing Flows (CNFs)** governed by ordinary differential equations (ODEs).

By designing straight-line interpolation trajectories between standard Gaussian noise $x_0 \sim \mathcal{N}(0, \mathbf{I})$ and data $x_1 \sim p_{\text{data}}$, the target velocity vector is constant:
$$x_t = (1 - t) x_0 + t x_1 \implies u_t(x_t \mid x_0, x_1) = \frac{dx_t}{dt} = x_1 - x_0$$
The network $v_\theta(x_t, t)$ learns a straight-line vector field, enabling generation in **1 to 4 Euler steps** without curved trajectory distortion.

```
+----------------------------------------------------------------------------------------------------+
|                               DIFFUSION VS. RECTIFIED FLOW TRAJECTORIES                            |
|                                                                                                    |
|  Curved DDPM Diffusion Trajectory (Requires 50 - 1000 steps):                                      |
|  Noise x_T ───~───┐                                                                                |
|                   │                                                                                |
|                   └───~───┐                                                                        |
|                           └───~───> Data x_0  (Tortuous Brownian path)                             |
|                                                                                                    |
|  Rectified Flow Straight-Line Trajectory (Requires 1 - 4 steps):                                   |
|  Noise x_0 ─────────────────────────────────────────────────────────────> Data x_1                 |
|             Constant Velocity Vector: u_t = (x_1 - x_0)                                            |
|             Euler Step: x_{t+dt} = x_t + dt * v_theta(x_t, t)                                      |
+----------------------------------------------------------------------------------------------------+
```

---

### The 12-Year-Old Analogy: The Mountain Switchbacks vs. The Bullet Train

1. **The Old Mountain Road (Diffusion):**
   - Imagine driving a car from the snowy mountain summit (pure Gaussian noise) down to a beach city (sharp image).
   - In DDPM, the road is a winding, foggy switchback with 1,000 hairpin turns. If you take one step that is too big, your car drives off the cliff into static. You must tap the brakes 100 times just to arrive safely.
2. **The Direct Bullet Train (Rectified Flow):**
   - Now imagine an engineering team drills a laser-straight tunnel straight through the mountain directly to the beach!
   - The bullet train doesn't need to turn the steering wheel left or right. It accelerates in a straight line with constant speed: $v = \text{Destination} - \text{Origin}$.
   - Because the path is completely straight, you can calculate where you will be with just 1 or 2 big leaps! This is why modern generative systems (such as Stable Diffusion 3 and Flux) run at lightning speed.

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $t \in [0, 1]$ | Continuous Time | Normalized time parameter ($t = 0$ is noise, $t = 1$ is clean data). |
| $x_0 \sim p_0$ | Continuous Vector | Base noise distribution, typically standard Gaussian $\mathcal{N}(0, \mathbf{I})$. |
| $x_1 \sim p_1$ | Continuous Vector | Target data distribution $p_{\text{data}}(x)$. |
| $x_t \in \mathbb{R}^D$ | Time-Dependent State | State vector at time $t$ along the continuous probability path $p_t(x)$. |
| $v_\theta(x, t)$ | Vector Field $\mathbb{R}^D \times [0, 1] \to \mathbb{R}^D$ | Neural velocity field parameterizing the ODE $\frac{dx_t}{dt} = v_\theta(x_t, t)$. |
| $u_t(x)$ | Vector Field | Marginal target velocity field that generates probability path $p_t(x)$. |
| $u_t(x \mid x_0, x_1)$ | Conditional Velocity | Velocity conditioned on sample endpoints $(x_0, x_1)$. |
| $L_{FM}(\theta)$ | Intractable Loss | Marginal Flow Matching objective: $\mathbb{E}_{t, x \sim p_t} [\|v_\theta(x, t) - u_t(x)\|^2]$. |
| $L_{CFM}(\theta)$ | Tractable Loss | Conditional Flow Matching objective: $\mathbb{E}_{t, x_0, x_1} [\|v_\theta(x_t, t) - u_t(x_t \mid x_0, x_1)\|^2]$. |
| $\text{div}(v) = \nabla \cdot v$ | Divergence | Spatial trace of velocity Jacobian: $\sum_{i=1}^D \frac{\partial v_i}{\partial x_i}$. |
| $\Delta t = 1/N$ | Discretization Step | Step size for numerical ODE integrators (Euler, Heun, Runge-Kutta). |

---

## 3. Mathematical Foundations & Derivations

### 3.1 Continuous Normalizing Flows & The Continuity Equation
Let $x_t \in \mathbb{R}^D$ follow a continuous deterministic trajectory defined by the Neural Ordinary Differential Equation (Neural ODE; Chen et al., 2018):
$$\frac{dx_t}{dt} = v_\theta(x_t, t), \quad t \in [0, 1]$$
Given a base noise density $p_0(x_0) = \mathcal{N}(x_0; 0, \mathbf{I})$, this ODE pushes probability mass forward along the vector field, generating a time-dependent probability density path $p_t(x)$.

By conservation of probability mass, the density path $p_t(x)$ and velocity field $v(x, t)$ strictly satisfy the **Continuity Equation** from fluid dynamics:
$$\frac{\partial p_t(x)}{\partial t} + \nabla \cdot \left( p_t(x) v_\theta(x, t) \right) = 0$$
where $\nabla \cdot F = \sum_{i=1}^D \frac{\partial F_i}{\partial x_i}$ is the divergence operator.

Using the identity $\nabla \cdot (p v) = p (\nabla \cdot v) + v \cdot \nabla p$, along a trajectory $x_t$:
$$\frac{d \log p_t(x_t)}{dt} = \frac{\partial \log p_t(x_t)}{\partial t} + \nabla \log p_t(x_t) \cdot \frac{dx_t}{dt}$$
$$= \frac{1}{p_t} \frac{\partial p_t}{\partial t} + \frac{1}{p_t} \nabla p_t \cdot v = \frac{1}{p_t} \left( \frac{\partial p_t}{\partial t} + \nabla \cdot (p_t v) - p_t \nabla \cdot v \right) = -\nabla \cdot v(x_t, t)$$
$$\frac{d \log p_t(x_t)}{dt} = -\text{tr}\left( \frac{\partial v_\theta(x_t, t)}{\partial x} \right)$$

#### Why Traditional CNF Training Was Prohibitive:
To train $v_\theta$ via maximum likelihood, one had to numerically integrate this ODE backward across all timesteps and estimate the Jacobian trace $\text{tr}\left(\frac{\partial v_\theta}{\partial x}\right)$ using stochastic Hutchinson estimators at every iteration. This made training CNFs thousands of times slower than diffusion models.

---

### 3.2 The Flow Matching Paradigm (Lipman et al., 2022)
Instead of backpropagating through a numerical ODE solver, Lipman et al. proposed **Flow Matching**: directly regressing the network velocity field $v_\theta(x, t)$ onto a target vector field $u_t(x)$:
$$L_{FM}(\theta) \triangleq \mathbb{E}_{t \sim \mathcal{U}[0, 1], x \sim p_t(x)} \left[ \|v_\theta(x, t) - u_t(x)\|^2 \right]$$

However, $L_{FM}(\theta)$ is intractable because neither the marginal density $p_t(x)$ nor the marginal velocity field $u_t(x)$ is available in closed form.

---

### 3.3 Conditional Flow Matching (CFM) Theorem
To make training tractable, condition the probability path and velocity field on endpoint data samples $x_1 \sim p_{\text{data}}$ and noise $x_0 \sim p_0$:
$$p_t(x) = \int p_t(x \mid x_0, x_1) p_0(x_0) p_{\text{data}}(x_1) \, dx_0 \, dx_1$$

The marginal velocity field $u_t(x)$ is the expectation of the conditional velocity $u_t(x \mid x_0, x_1)$ over the posterior distribution:
$$u_t(x) = \mathbb{E}_{x_0, x_1 \sim q(x_0, x_1 \mid x_t = x)} \left[ u_t(x \mid x_0, x_1) \right] = \int u_t(x \mid x_0, x_1) \frac{p_t(x \mid x_0, x_1) p(x_0, x_1)}{p_t(x)} \, dx_0 \, dx_1$$

Now define the **Conditional Flow Matching (CFM)** objective:
$$L_{CFM}(\theta) \triangleq \mathbb{E}_{t \sim \mathcal{U}[0, 1], x_0 \sim p_0, x_1 \sim p_1, x_t \sim p_t(\cdot \mid x_0, x_1)} \left[ \|v_\theta(x_t, t) - u_t(x_t \mid x_0, x_1)\|^2 \right]$$

#### The Equivalence Theorem (Lipman et al., 2022)
**Theorem:** For any choice of conditional probability path and conditional vector field that generates it:
$$\nabla_\theta L_{CFM}(\theta) \equiv \nabla_\theta L_{FM}(\theta)$$

**Proof:**
Expand the squared Euclidean norms in both objectives:
$$L_{FM}(\theta) = \mathbb{E}_{t, x \sim p_t} [\|v_\theta(x, t)\|^2] - 2 \mathbb{E}_{t, x \sim p_t} [v_\theta(x, t)^\top u_t(x)] + C_1$$
$$L_{CFM}(\theta) = \mathbb{E}_{t, x_0, x_1, x_t} [\|v_\theta(x_t, t)\|^2] - 2 \mathbb{E}_{t, x_0, x_1, x_t} [v_\theta(x_t, t)^\top u_t(x_t \mid x_0, x_1)] + C_2$$
The first terms are identical because $\mathbb{E}_{x_0, x_1} [p_t(x_t \mid x_0, x_1)] = p_t(x_t)$.
For the middle term of $L_{CFM}$:
$$\mathbb{E}_{t, x_0, x_1, x_t} [v_\theta(x_t, t)^\top u_t(x_t \mid x_0, x_1)] = \int_0^1 \int \int \int v_\theta(x_t, t)^\top u_t(x_t \mid x_0, x_1) p_t(x_t \mid x_0, x_1) p(x_0, x_1) \, dx_t \, dx_0 \, dx_1 \, dt$$
Using Fubini's theorem to exchange integration order:
$$= \int_0^1 \int v_\theta(x_t, t)^\top \left[ \int \int u_t(x_t \mid x_0, x_1) \frac{p_t(x_t \mid x_0, x_1) p(x_0, x_1)}{p_t(x_t)} dx_0 dx_1 \right] p_t(x_t) \, dx_t \, dt$$
Notice that the bracketed term is precisely the definition of the marginal velocity $u_t(x_t)$:
$$= \int_0^1 \int v_\theta(x_t, t)^\top u_t(x_t) p_t(x_t) \, dx_t \, dt = \mathbb{E}_{t, x \sim p_t} [v_\theta(x, t)^\top u_t(x)]$$
Because all terms involving $\theta$ are identically equal, $\nabla_\theta L_{CFM}(\theta) = \nabla_\theta L_{FM}(\theta)$.

---

### 3.4 Optimal Transport & Rectified Flow
Among all possible interpolation paths, the simplest and most computationally efficient is **straight-line interpolation** (Optimal Transport geodesic displacement; Liu et al., 2022):
$$x_t = (1 - t) x_0 + t x_1, \quad t \in [0, 1]$$

Differentiating this linear interpolation with respect to time $t$:
$$u_t(x_t \mid x_0, x_1) = \frac{dx_t}{dt} = \frac{d}{dt} \left( (1 - t) x_0 + t x_1 \right) = x_1 - x_0$$

Notice the profound simplicity:
- The target conditional velocity $u_t$ is **independent of time $t$**.
- It is simply the constant displacement vector pointing from noise sample $x_0$ to data sample $x_1$!
- The loss function simplifies to standard mean squared error:
$$L_{\text{Rectified}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0, 1], x_0 \sim \mathcal{N}(0, \mathbf{I}), x_1 \sim p_{\text{data}}} \left[ \left\| v_\theta((1 - t)x_0 + t x_1, t) - (x_1 - x_0) \right\|^2 \right]$$

#### Why Straight Lines Allow Few-Step Inference
Consider numerical ODE integration using the Euler method:
$$x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t)$$
The local truncation error of the Euler method is proportional to the second derivative:
$$\text{Error} \sim \mathcal{O}\left( (\Delta t)^2 \left\| \frac{d^2 x_t}{dt^2} \right\| \right)$$
In Rectified Flows, because trajectories are approximately straight lines, the path curvature is nearly zero:
$$\frac{d^2 x_t}{dt^2} = \frac{d v_\theta}{dt} \approx \mathbf{0}$$
Because curvature vanishes, truncation error drops by orders of magnitude. A solver can take massive time steps ($\Delta t = 0.25$ or even $\Delta t = 1.0$), reducing generative inference from 50 steps down to **1 to 4 steps** with zero quality degradation!
