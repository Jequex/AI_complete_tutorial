# Topic 25.5: Exercises & Solutions — Flow Matching & Rectified Flows

---

### Problem 1: Straight-Line Kinematics & Euler Error (Easy)

In Rectified Flow, the interpolation path between a noise sample $x_0 \in \mathbb{R}^D$ and a data sample $x_1 \in \mathbb{R}^D$ is defined by:
$$x_t = (1 - t) x_0 + t x_1, \quad t \in [0, 1]$$

1. Compute the analytical velocity vector $v(t) = \frac{dx_t}{dt}$.
2. Compute the trajectory acceleration vector $a(t) = \frac{d^2 x_t}{dt^2}$.
3. In Taylor series expansion of Euler numerical integration:
   $$x_{t + \Delta t} = x_t + \Delta t \cdot v(t) + \frac{(\Delta t)^2}{2} a(t) + \mathcal{O}((\Delta t)^3)$$
   evaluate the truncation error when $a(t) \equiv \mathbf{0}$, and explain why 1-step integration can be exact.

<details>
<summary>View Complete Solution</summary>

**Step 1: Velocity Vector:**
$$v(t) = \frac{d}{dt} \left( (1 - t)x_0 + t x_1 \right) = -x_0 + x_1 = x_1 - x_0$$
The velocity is a constant vector independent of time $t$.

**Step 2: Acceleration Vector:**
$$a(t) = \frac{dv}{dt} = \frac{d}{dt} (x_1 - x_0) = \mathbf{0}$$
The acceleration is identically zero everywhere along the path.

**Step 3: Euler Truncation Error:**
Substituting $a(t) = \mathbf{0}$ and higher derivatives $\frac{d^k x}{dt^k} = \mathbf{0}$ for $k \ge 2$:
$$x_{t + \Delta t} = x_t + \Delta t \cdot (x_1 - x_0) + \mathbf{0} + \mathbf{0} = x_t + \Delta t (x_1 - x_0)$$
The local truncation error is **identically zero**.
If a neural network perfectly learns the straight vector field $v_\theta(x, t) = x_1 - x_0$, a single Euler step with $\Delta t = 1.0$:
$$x_1 = x_0 + 1.0 \cdot v_\theta(x_0, 0) = x_0 + (x_1 - x_0) = x_1$$
recovers the exact data sample in **a single step**!
</details>

---

### Problem 2: Derivation of the Continuity Equation Velocity for Expanding Gaussians (Medium)

Consider a 1D probability density path defined by a zero-mean Gaussian whose standard deviation expands linearly over time $t \in [0, 1]$:
$$p_t(x) = \frac{1}{\sqrt{2\pi}\sigma_t} \exp\left( -\frac{x^2}{2\sigma_t^2} \right), \quad \text{where} \quad \sigma_t = 1 + t$$

The Continuity Equation requires:
$$\frac{\partial p_t(x)}{\partial t} + \frac{\partial}{\partial x} \left( p_t(x) v(x, t) \right) = 0$$

1. Compute $\frac{\partial p_t(x)}{\partial t}$.
2. Solve the Continuity Equation to find the exact velocity field $v(x, t)$.
3. Check the boundary condition $v(0, t)$ at the origin $x = 0$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute Time Derivative:**
$$\log p_t(x) = -\frac{1}{2}\log(2\pi) - \log \sigma_t - \frac{x^2}{2\sigma_t^2}$$
$$\frac{\partial \log p_t(x)}{\partial t} = -\frac{\dot{\sigma}_t}{\sigma_t} + \frac{x^2 \dot{\sigma}_t}{\sigma_t^3}$$
Since $\sigma_t = 1 + t$, we have $\dot{\sigma}_t = 1$:
$$\frac{\partial p_t(x)}{\partial t} = p_t(x) \left( -\frac{1}{\sigma_t} + \frac{x^2}{\sigma_t^3} \right)$$

**Step 2: Solve for Velocity $v(x, t)$:**
We test a linear velocity field $v(x, t) = c(t) \cdot x$.
Then:
$$\frac{\partial}{\partial x} (p_t(x) v(x, t)) = v(x, t) \frac{\partial p_t}{\partial x} + p_t(x) \frac{\partial v}{\partial x}$$
We know $\frac{\partial p_t}{\partial x} = p_t(x) \left(-\frac{x}{\sigma_t^2}\right)$, and $\frac{\partial v}{\partial x} = c(t)$:
$$\frac{\partial (p_t v)}{\partial x} = p_t(x) \left( -c(t) \frac{x^2}{\sigma_t^2} + c(t) \right)$$

Plugging this into the Continuity Equation:
$$\frac{\partial p_t}{\partial t} + \frac{\partial (p_t v)}{\partial x} = p_t(x) \left[ \left( -\frac{1}{\sigma_t} + \frac{x^2}{\sigma_t^3} \right) + \left( c(t) - c(t) \frac{x^2}{\sigma_t^2} \right) \right] = 0$$
For this to hold for all $x$, the constant and $x^2$ terms must cancel independently:
$$c(t) - \frac{1}{\sigma_t} = 0 \implies c(t) = \frac{1}{\sigma_t}$$
$$\frac{1}{\sigma_t^3} - \frac{c(t)}{\sigma_t^2} = \frac{1}{\sigma_t^3} - \frac{1}{\sigma_t^3} = 0$$
Both equations match!
Therefore, the velocity field is:
$$v(x, t) = \frac{x}{\sigma_t} = \frac{x}{1 + t}$$

**Step 3: Boundary Condition:**
At $x = 0$, $v(0, t) = 0$. The probability mode does not drift, and particles symmetrically expand outward away from the origin as the variance grows.
</details>

---

### Problem 3: Fubini's Equivalence Proof for Conditional Flow Matching (Hard)

Let $L_{FM}(\theta) = \mathbb{E}_{t, x \sim p_t}[\|v_\theta(x, t) - u_t(x)\|^2]$ and $L_{CFM}(\theta) = \mathbb{E}_{t, x_0, x_1, x_t}[\|v_\theta(x_t, t) - u_t(x_t \mid x_0, x_1)\|^2]$.
Prove from first principles that:
$$\nabla_\theta L_{CFM}(\theta) \equiv \nabla_\theta L_{FM}(\theta)$$

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute Gradients of Both Losses:**
Differentiating $L_{FM}(\theta)$ with respect to $\theta$:
$$\nabla_\theta L_{FM}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0, 1], x \sim p_t(x)} \left[ 2 \left( \nabla_\theta v_\theta(x, t) \right) \left( v_\theta(x, t) - u_t(x) \right) \right]$$
$$= 2 \mathbb{E}_{t, x \sim p_t} [(\nabla_\theta v_\theta) v_\theta] - 2 \mathbb{E}_{t, x \sim p_t} [(\nabla_\theta v_\theta) u_t(x)]$$

Differentiating $L_{CFM}(\theta)$ with respect to $\theta$:
$$\nabla_\theta L_{CFM}(\theta) = 2 \mathbb{E}_{t, x_0, x_1, x_t} [(\nabla_\theta v_\theta(x_t, t)) v_\theta(x_t, t)] - 2 \mathbb{E}_{t, x_0, x_1, x_t} [(\nabla_\theta v_\theta(x_t, t)) u_t(x_t \mid x_0, x_1)]$$

**Step 2: Compare First Terms:**
By definition of the marginal density $p_t(x) = \int p_t(x \mid x_0, x_1) p(x_0, x_1) dx_0 dx_1$:
$$\mathbb{E}_{x_0, x_1 \sim p, x_t \sim p_t(\cdot \mid x_0, x_1)} [f(x_t)] = \mathbb{E}_{x \sim p_t} [f(x)]$$
Therefore:
$$\mathbb{E}_{t, x_0, x_1, x_t} [(\nabla_\theta v_\theta) v_\theta] = \mathbb{E}_{t, x \sim p_t} [(\nabla_\theta v_\theta) v_\theta]$$

**Step 3: Compare Second Terms (The Core Substitution):**
Write the second term of $\nabla_\theta L_{CFM}$ in full integral form:
$$\mathcal{T}_2 = \int_0^1 \int_{\mathbb{R}^D} \int \int (\nabla_\theta v_\theta(x, t)) u_t(x \mid x_0, x_1) p_t(x \mid x_0, x_1) p(x_0, x_1) \, dx_0 \, dx_1 \, dx \, dt$$

Pull $(\nabla_\theta v_\theta(x, t))$ out of the internal $(x_0, x_1)$ integrals:
$$\mathcal{T}_2 = \int_0^1 \int_{\mathbb{R}^D} (\nabla_\theta v_\theta(x, t)) \left[ \int \int u_t(x \mid x_0, x_1) \frac{p_t(x \mid x_0, x_1) p(x_0, x_1)}{p_t(x)} \, dx_0 \, dx_1 \right] p_t(x) \, dx \, dt$$

By definition of conditional expectation, the term inside brackets is exactly the marginal velocity:
$$\int \int u_t(x \mid x_0, x_1) \frac{p_t(x \mid x_0, x_1) p(x_0, x_1)}{p_t(x)} \, dx_0 \, dx_1 \equiv u_t(x)$$

Substituting this gives:
$$\mathcal{T}_2 = \int_0^1 \int_{\mathbb{R}^D} (\nabla_\theta v_\theta(x, t)) u_t(x) p_t(x) \, dx \, dt = \mathbb{E}_{t, x \sim p_t} [(\nabla_\theta v_\theta(x, t)) u_t(x)]$$

**Conclusion:**
Every term of $\nabla_\theta L_{CFM}(\theta)$ equals the corresponding term of $\nabla_\theta L_{FM}(\theta)$ identically.
Hence $\nabla_\theta L_{CFM}(\theta) \equiv \nabla_\theta L_{FM}(\theta)$, proving that training on easy conditional pairs $(x_0, x_1)$ minimizes the intractable marginal generative flow objective.
</details>

---

### Problem 4: Optimal Transport Kinetic Action Minimization (Hard)

Consider the physical kinetic action of moving a particle along trajectory $x(t)$ from $x(0) = x_0$ to $x(1) = x_1$:
$$\mathcal{S}[x] = \int_0^1 \frac{1}{2} \|v(t)\|^2 \, dt = \frac{1}{2} \int_0^1 \left\| \frac{dx}{dt} \right\|^2 \, dt$$

1. Apply the Euler-Lagrange equations to find the path $x(t)$ that minimizes $\mathcal{S}[x]$.
2. Calculate the minimum action value in terms of the Euclidean distance $\|x_1 - x_0\|$.
3. Explain why Rectified Flow trajectories represent 2-Wasserstein Optimal Transport geodesics.

<details>
<summary>View Complete Solution</summary>

**Step 1: Euler-Lagrange Formulation:**
The Lagrangian is:
$$\mathcal{L}(t, x, \dot{x}) = \frac{1}{2} \|\dot{x}\|^2 = \frac{1}{2} \sum_{i=1}^D \dot{x}_i^2$$
The Euler-Lagrange equations state:
$$\frac{d}{dt} \left( \frac{\partial \mathcal{L}}{\partial \dot{x}_i} \right) - \frac{\partial \mathcal{L}}{\partial x_i} = 0$$
Since $\mathcal{L}$ does not depend on position $x_i$, $\frac{\partial \mathcal{L}}{\partial x_i} = 0$:
$$\frac{d}{dt} (\dot{x}_i) - 0 = 0 \implies \ddot{x}_i(t) = 0 \implies \frac{d^2 x}{dt^2} = \mathbf{0}$$

Integrating twice:
$$\dot{x}(t) = \mathbf{c}_1 \implies x(t) = \mathbf{c}_1 t + \mathbf{c}_0$$
Using boundary conditions $x(0) = x_0$ and $x(1) = x_1$:
$$\mathbf{c}_0 = x_0, \quad \mathbf{c}_1 = x_1 - x_0$$
$$x(t) = (1 - t) x_0 + t x_1$$
The unique path that minimizes kinetic action is the **straight-line path with constant velocity**!

**Step 2: Minimum Action Value:**
Along the optimal path, $\dot{x}(t) = x_1 - x_0$:
$$\mathcal{S}_{\text{min}} = \frac{1}{2} \int_0^1 \|x_1 - x_0\|^2 \, dt = \frac{1}{2} \|x_1 - x_0\|^2$$

**Step 3: Optimal Transport Geodesics:**
The 2-Wasserstein distance between distributions is defined as:
$$W_2^2(p_0, p_1) = \inf_{\pi \in \Pi(p_0, p_1)} \mathbb{E}_{(x_0, x_1) \sim \pi} \left[ \|x_1 - x_0\|^2 \right] = 2 \inf \mathbb{E}[\mathcal{S}[x]]$$
By using straight-line trajectories $x_t = (1 - t)x_0 + tx_1$, Rectified Flows follow the minimal-energy displacement interpolation connecting $p_0$ and $p_1$. Particles travel along geodesics with zero wasted sideways acceleration, leading to fast, non-intersecting trajectories that ODE solvers can integrate in minimal steps.
</details>
