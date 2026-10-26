# Topic 24.5: Exercises & Solutions — Wasserstein GAN & Gradient Penalty

---

### Problem 1: Earth Mover's Distance vs. Total Variation on Discrete Spikes (Easy)

Consider two discrete probability distributions on the real line $\mathbb{R}$:
$$P = \frac{1}{2}\delta_1 + \frac{1}{2}\delta_3, \qquad Q = \frac{1}{2}\delta_2 + \frac{1}{2}\delta_4$$
where $\delta_a$ denotes a Dirac point mass at coordinate $a$.

1. Compute the Total Variation Distance $\delta(P, Q) = \frac{1}{2} \sum_{x} |P(x) - Q(x)|$.
2. Compute the Jensen-Shannon Divergence $\text{JSD}(P \parallel Q)$.
3. Compute the Earth Mover's (Wasserstein-1) Distance $\mathcal{W}_1(P, Q)$ by finding the optimal transport coupling $\gamma(x, y)$ that minimizes total transport cost $\sum_{x, y} \gamma(x, y) |x - y|$.
4. Now shift $Q$ further to the right: $Q' = \frac{1}{2}\delta_{10} + \frac{1}{2}\delta_{12}$.
   Re-evaluate Total Variation, JSD, and $\mathcal{W}_1(P, Q')$. What fundamental difference do you observe?

<details>
<summary>View Complete Solution</summary>

**Step 1: Total Variation Distance $\delta(P, Q)$:**
The supports $\{1, 3\}$ and $\{2, 4\}$ are disjoint.
$$\delta(P, Q) = \frac{1}{2} \left[ |0.5 - 0| + |0 - 0.5| + |0.5 - 0| + |0 - 0.5| \right] = \frac{1}{2} [0.5 + 0.5 + 0.5 + 0.5] = 1.0$$
*(Maximum possible total variation distance!)*

**Step 2: Jensen-Shannon Divergence:**
Mixture $M = \frac{1}{4}(\delta_1 + \delta_2 + \delta_3 + \delta_4)$.
$$D_{\text{KL}}(P \parallel M) = \frac{1}{2}\log\left(\frac{1/2}{1/4}\right) + \frac{1}{2}\log\left(\frac{1/2}{1/4}\right) = \log 2$$
$$D_{\text{KL}}(Q \parallel M) = \log 2$$
$$\text{JSD}(P \parallel Q) = \frac{1}{2}(\log 2 + \log 2) = \log 2 \approx 0.69315$$

**Step 3: Wasserstein-1 Distance $\mathcal{W}_1(P, Q)$:**
To transport mass from $P$ to $Q$ with minimum cost:
- Move mass $0.5$ from $x = 1$ to $y = 2$: Cost $= 0.5 \times |1 - 2| = 0.5 \times 1 = 0.5$.
- Move mass $0.5$ from $x = 3$ to $y = 4$: Cost $= 0.5 \times |3 - 4| = 0.5 \times 1 = 0.5$.
$$\mathcal{W}_1(P, Q) = 0.5 + 0.5 = 1.0$$

**Step 4: Shifted Distribution $Q'$:**
For $Q' = 0.5\delta_{10} + 0.5\delta_{12}$:
- Total Variation: Still $\delta(P, Q') = 1.0$ (Saturated).
- JSD: Still $\text{JSD}(P \parallel Q') = \log 2 \approx 0.69315$ (Saturated).
- Wasserstein-1:
  - Move mass $0.5$ from $1$ to $10$: Cost $= 0.5 \times 9 = 4.5$.
  - Move mass $0.5$ from $3$ to $12$: Cost $= 0.5 \times 9 = 4.5$.
  $$\mathcal{W}_1(P, Q') = 4.5 + 4.5 = 9.0$$

**Conclusion:** Total Variation and JSD are completely blind to the fact that $Q'$ is $9\times$ further away than $Q$. Wasserstein distance scales smoothly and linearly with true geometric transport distance ($1.0 \to 9.0$).
</details>

---

### Problem 2: Equivalence of Lipschitz Constant and Gradient Norm (Medium)

Let $f: \mathbb{R}^D \to \mathbb{R}$ be a continuously differentiable function.
1. Define the Lipschitz constant $\|f\|_L$.
2. Prove that $\|f\|_L \le 1$ if and only if the Euclidean norm of its spatial gradient satisfies:
   $$\|\nabla f(x)\|_2 \le 1 \quad \forall x \in \mathbb{R}^D$$
3. Explain why this theorem forms the mathematical foundation of the WGAN-GP gradient penalty.

<details>
<summary>View Complete Solution</summary>

**Step 1: Definition of Lipschitz Constant:**
The Lipschitz constant of a function $f$ on a metric space is defined as:
$$\|f\|_L = \sup_{x \ne y} \frac{|f(x) - f(y)|}{\|x - y\|_2}$$
Thus, $\|f\|_L \le 1 \iff |f(x) - f(y)| \le \|x - y\|_2$ for all $x, y \in \mathbb{R}^D$.

**Step 2: Proof of Equivalence:**

- **Direction 1 ($\|f\|_L \le 1 \implies \|\nabla f(x)\|_2 \le 1$):**
  Choose an arbitrary unit vector $v \in \mathbb{R}^D$ ($\|v\|_2 = 1$) and small scalar $t > 0$.
  By the definition of the directional derivative:
  $$\nabla f(x)^\top v = \lim_{t \to 0} \frac{f(x + t v) - f(x)}{t}$$
  Taking absolute values and using the 1-Lipschitz condition:
  $$|\nabla f(x)^\top v| = \lim_{t \to 0} \frac{|f(x + t v) - f(x)|}{t} \le \lim_{t \to 0} \frac{\|t v\|_2}{t} = \lim_{t \to 0} \frac{t \|v\|_2}{t} = 1$$
  Since this holds for all unit vectors $v$, we choose $v = \frac{\nabla f(x)}{\|\nabla f(x)\|_2}$:
  $$\nabla f(x)^\top \left(\frac{\nabla f(x)}{\|\nabla f(x)\|_2}\right) = \|\nabla f(x)\|_2 \le 1$$

- **Direction 2 ($\|\nabla f(x)\|_2 \le 1 \implies \|f\|_L \le 1$):**
  For any distinct points $x, y \in \mathbb{R}^D$, parameterize the straight line segment between them:
  $$\gamma(t) = (1 - t) x + t y, \quad t \in [0, 1]$$
  By the Fundamental Theorem of Calculus:
  $$f(y) - f(x) = \int_0^1 \frac{d}{dt} f(\gamma(t)) \, dt = \int_0^1 \nabla f(\gamma(t))^\top (y - x) \, dt$$
  Applying the Cauchy-Schwarz inequality inside the integral:
  $$|f(y) - f(x)| \le \int_0^1 |\nabla f(\gamma(t))^\top (y - x)| \, dt \le \int_0^1 \|\nabla f(\gamma(t))\|_2 \|y - x\|_2 \, dt$$
  Since $\|\nabla f(\gamma(t))\|_2 \le 1$ everywhere along the path:
  $$|f(y) - f(x)| \le \|y - x\|_2 \int_0^1 1 \, dt = \|y - x\|_2$$
  Dividing by $\|y - x\|_2$ yields $\|f\|_L \le 1$.

**Step 3: Foundation of WGAN-GP:**
This theorem proves that we can enforce the 1-Lipschitz requirement of the Kantorovich-Rubinstein duality simply by constraining the local Euclidean norm of the critic's gradient $\|\nabla_x f(x)\|_2$ to not exceed $1$, eliminating the need for brute-force weight clipping!
</details>

---

### Problem 3: Derivation of the Unit Gradient Norm on Geodesic Lines (Hard)

Let $x_0 \sim P_r$ and $x_1 \sim P_g$ be paired data points under the optimal transport coupling $\pi^*$ that achieves the infimum in $\mathcal{W}_1(P_r, P_g)$.
Let $f^*$ be the optimal 1-Lipschitz critic function achieving the supremum in the Kantorovich-Rubinstein duality:
$$\mathcal{W}_1(P_r, P_g) = \mathbb{E}_{(x_0, x_1) \sim \pi^*}[f^*(x_0) - f^*(x_1)] = \mathbb{E}_{(x_0, x_1) \sim \pi^*}[\|x_0 - x_1\|_2]$$

1. Show that for any optimal pair $(x_0, x_1)$, it must hold that:
   $$f^*(x_0) - f^*(x_1) = \|x_0 - x_1\|_2$$
2. Let $\hat{x}_t = (1 - t) x_0 + t x_1$ be any point on the straight line segment between $x_0$ and $x_1$ for $t \in (0, 1)$. Show that:
   $$f^*(x_0) - f^*(\hat{x}_t) = t \|x_1 - x_0\|_2$$
3. Prove that the spatial gradient vector of the optimal critic at $\hat{x}_t$ satisfies:
   $$\nabla_{\hat{x}} f^*(\hat{x}_t) = \frac{x_0 - x_1}{\|x_0 - x_1\|_2}$$
   and consequently has unit Euclidean norm: $\|\nabla_{\hat{x}} f^*(\hat{x}_t)\|_2 = 1.0$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Pointwise Equality for Optimal Pairs:**
By the 1-Lipschitz property of $f^*$:
$$f^*(x_0) - f^*(x_1) \le \|x_0 - x_1\|_2 \quad \forall x_0, x_1$$
Taking expectations over the optimal coupling $\pi^*$:
$$\mathbb{E}_{\pi^*}[f^*(x_0) - f^*(x_1)] \le \mathbb{E}_{\pi^*}[\|x_0 - x_1\|_2]$$
However, by the Kantorovich-Rubinstein theorem, the expectation on the left is identically equal to the expectation on the right ($\mathcal{W}_1$).
Since $f^*(x_0) - f^*(x_1) \le \|x_0 - x_1\|_2$ pointwise and their integrals are equal:
$$f^*(x_0) - f^*(x_1) = \|x_0 - x_1\|_2 \quad \pi^*\text{-almost everywhere}$$

**Step 2: Values Along the Line Segment:**
Let $\hat{x}_t = (1 - t)x_0 + t x_1$. Then:
$$\|x_0 - \hat{x}_t\|_2 = t \|x_1 - x_0\|_2, \qquad \|\hat{x}_t - x_1\|_2 = (1 - t) \|x_1 - x_0\|_2$$
By the 1-Lipschitz condition:
$$f^*(x_0) - f^*(\hat{x}_t) \le \|x_0 - \hat{x}_t\|_2 = t \|x_1 - x_0\|_2$$
$$f^*(\hat{x}_t) - f^*(x_1) \le \|\hat{x}_t - x_1\|_2 = (1 - t) \|x_1 - x_0\|_2$$
Adding these two inequalities:
$$(f^*(x_0) - f^*(\hat{x}_t)) + (f^*(\hat{x}_t) - f^*(x_1)) = f^*(x_0) - f^*(x_1) \le \|x_1 - x_0\|_2$$
Since we already proved $f^*(x_0) - f^*(x_1) = \|x_1 - x_0\|_2$, both inequalities must be tight equalities!
Therefore:
$$f^*(x_0) - f^*(\hat{x}_t) = t \|x_1 - x_0\|_2$$
$$f^*(\hat{x}_t) = f^*(x_0) - t \|x_1 - x_0\|_2$$

**Step 3: Differentiating Along the Geodesic:**
Differentiating $f^*(\hat{x}_t)$ with respect to scalar parameter $t$:
$$\frac{d}{dt} f^*(\hat{x}_t) = -\|x_1 - x_0\|_2$$
Applying the multivariable chain rule:
$$\frac{d}{dt} f^*(\hat{x}_t) = \nabla_{\hat{x}} f^*(\hat{x}_t)^\top \frac{d\hat{x}_t}{dt} = \nabla_{\hat{x}} f^*(\hat{x}_t)^\top (x_1 - x_0)$$
Thus:
$$\nabla_{\hat{x}} f^*(\hat{x}_t)^\top (x_1 - x_0) = -\|x_1 - x_0\|_2$$
$$\nabla_{\hat{x}} f^*(\hat{x}_t)^\top \left( \frac{x_0 - x_1}{\|x_0 - x_1\|_2} \right) = 1.0$$

By Cauchy-Schwarz, $\nabla_{\hat{x}} f^*(\hat{x}_t)^\top u \le \|\nabla_{\hat{x}} f^*(\hat{x}_t)\|_2 \|u\|_2 = \|\nabla_{\hat{x}} f^*(\hat{x}_t)\|_2 (1)$.
Since $f^*$ is 1-Lipschitz, $\|\nabla_{\hat{x}} f^*(\hat{x}_t)\|_2 \le 1$.
The only vector satisfying these conditions is:
$$\boxed{\nabla_{\hat{x}} f^*(\hat{x}_t) = \frac{x_0 - x_1}{\|x_0 - x_1\|_2}}$$
$$\boxed{\|\nabla_{\hat{x}} f^*(\hat{x}_t)\|_2 = 1.0}$$

**Conclusion:** The optimal critic's gradient norm is **identically 1.0** everywhere along the straight line connecting real and generated samples. This provides the exact theoretical justification for the WGAN-GP two-sided penalty $\mathbb{E}[(\|\nabla_{\hat{x}} f(\hat{x})\| - 1)^2]$.
</details>
