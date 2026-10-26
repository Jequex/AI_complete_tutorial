# Topic 25.1: Score-Based Models and Langevin Dynamics

---

## 1. Executive Summary & The 12-Year-Old Analogy

### The Fundamental Obstacle of Generative Modeling: The Partition Function
Generative modeling fundamentally seeks to model a probability density $p(x)$ over high-dimensional data $x \in \mathbb{R}^D$ (such as images, audio, or molecular configurations) given an empirical dataset $\mathcal{D} = \{x_1, \dots, x_N\} \sim p_{\text{data}}(x)$.

In standard parametric probability modeling (e.g., Energy-Based Models or Markov Random Fields), the probability density is parameterized via an unnormalized scalar energy function $E_\theta(x) \in \mathbb{R}$:
$$p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z_\theta}, \quad \text{where} \quad Z_\theta = \int_{\mathbb{R}^D} \exp(-E_\theta(x)) \, dx$$

The denominator $Z_\theta$ is the **partition function** (normalizing constant). In any realistic domain where $D \ge 100$, computing $Z_\theta$ requires an intractable numerical integration over a continuous $D$-dimensional volume. Furthermore, maximum likelihood estimation requires evaluating the gradient of the log-likelihood:
$$\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) - \nabla_\theta \log Z_\theta = -\nabla_\theta E_\theta(x) + \mathbb{E}_{p_\theta(\tilde{x})}[\nabla_\theta E_\theta(\tilde{x})]$$
Computing expectations under the unnormalized distribution $p_\theta$ requires costly Markov Chain Monte Carlo (MCMC) iterations at every single optimization step, leading to training instability and mode collapse.

### The Score Paradigm
Score-based generative modeling (Hyvärinen 2005; Song & Ermon 2019) completely bypasses the partition function by shifting attention from the scalar density $p(x)$ to its **vector score function**:
$$s(x) \triangleq \nabla_x \log p(x)$$

Notice what happens when we differentiate the log-density with respect to the input data $x$:
$$s_\theta(x) = \nabla_x \log p_\theta(x) = \nabla_x \left( -E_\theta(x) - \log Z_\theta \right) = -\nabla_x E_\theta(x) - \mathbf{0} = -\nabla_x E_\theta(x)$$
Because the partition function $Z_\theta$ is a constant with respect to $x$ ($\nabla_x Z_\theta = \mathbf{0}$), **the intractable normalizing constant vanishes identically**. A neural network $s_\theta(x): \mathbb{R}^D \to \mathbb{R}^D$ can directly parameterize the score vector field without ever normalizing probabilities.

```
+----------------------------------------------------------------------------------------------------+
|                               THE SCORE PARADIGM SHIFT                                             |
|                                                                                                    |
|  Traditional Density Modeling:                                                                     |
|  p_theta(x) = exp(-E_theta(x)) / Z_theta   <--- Intractable D-dimensional Integral Z_theta!        |
|                                                                                                    |
|  Score Vector Field:                                                                               |
|  s_theta(x) = grad_x log p_theta(x) = -grad_x E_theta(x)  <--- Z_theta vanishes completely!       |
|                                                                                                    |
|            Mode Peak (Local Maximum of Density)                                                    |
|                           ^                                                                        |
|                      /         \                                                                   |
|              ------>             <------                                                           |
|             s(x) > 0             s(x) < 0                                                          |
|       (Vectors point uphill toward high-density data modes)                                        |
+----------------------------------------------------------------------------------------------------+
```

---

### The 12-Year-Old Analogy: Mountain Peaks in a Pitch-Black World

Imagine you are dropped by a parachute into a completely pitch-black mountain range at midnight:
1. **The Impossible Map (The Partition Function):**
   - If someone demands you tell them the exact percentage of all oxygen in the entire world that sits on each square inch of the mountain, you would need to measure every molecule in every valley, ocean, and atmosphere worldwide! That total sum is the **partition function $Z$**. It is impossible to calculate.
2. **The Scent of Pine (The Score Function):**
   - But what if you only want to hike to the nearest campsite located at the top of a peak? You don't need a worldwide map. You just sniff the air or feel the slope of the ground under your boots. The direction of steepest uphill slope pointing toward the campsite is the **Score vector $s(x)$**. It tells you: *"Step 2 feet North and 1 foot East to get closer to safety."*
3. **The Trap of Deep Valleys (Mode Barrier):**
   - If you only follow your feet (standard Langevin dynamics), you might get stuck on a tiny hill and never find the grand mountain where all the other hikers are, because a vast, flat, freezing desert lies between the hills where you cannot smell or feel any slope.
4. **The Magic Rising Fog (Annealing):**
   - To fix this, you spray a massive cloud of warm, scented fog across the entire world (high noise $\sigma_1$). Now you can smell the campfire from miles away, anywhere in the desert! You run toward the general area. As you get closer, the fog slowly dissipates ($\sigma_1 \to \sigma_2 \to \dots \to \sigma_L$). By the time the air is crystal clear, you are standing directly in the center of the camp! This is **Annealed Langevin Dynamics**.

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $x \in \mathbb{R}^D$ | Continuous Vector | Data sample (e.g., $D$-dimensional image or continuous state). |
| $p_{\text{data}}(x)$ | Probability Density | The true, unknown ground-truth data-generating distribution. |
| $p_\theta(x)$ | Probability Density | Model parameterized density distribution with parameters $\theta$. |
| $Z_\theta \in \mathbb{R}^+$ | Positive Scalar | Partition function: $Z_\theta = \int \exp(-E_\theta(x)) dx$. |
| $s(x) \in \mathbb{R}^D$ | Vector Field | True score function: $\nabla_x \log p_{\text{data}}(x)$. |
| $s_\theta(x) \in \mathbb{R}^D$ | Neural Vector Field | Score network parameterized by weights $\theta$: $s_\theta: \mathbb{R}^D \to \mathbb{R}^D$. |
| $J_{ESM}(\theta)$ | Scalar Objective | Explicit Score Matching objective (Hyvärinen 2005). |
| $J_{ISM}(\theta)$ | Scalar Objective | Implicit Score Matching objective (derived via integration by parts). |
| $\text{tr}(\nabla_x s_\theta(x))$ | Scalar Trace | Trace of the Jacobian $\sum_{i=1}^D \frac{\partial s_{\theta, i}(x)}{\partial x_i} = \sum_{i=1}^D \frac{\partial^2 \log p(x)}{\partial x_i^2}$. |
| $q_\sigma(\tilde{x} \mid x)$ | Gaussian Kernel | Perturbation distribution: $\mathcal{N}(\tilde{x}; x, \sigma^2 \mathbf{I})$. |
| $J_{DSM}(\theta)$ | Scalar Objective | Denoising Score Matching objective (Vincent 2011). |
| $\sigma_1 > \dots > \sigma_L$ | Geometric Sequence | Annealed noise levels from large scale $\sigma_1$ down to fine scale $\sigma_L$. |
| $z_t \sim \mathcal{N}(0, \mathbf{I})$ | Standard Normal | Stochastic Brownian motion injection at Langevin step $t$. |
| $\epsilon, \alpha_i \in \mathbb{R}^+$ | Positive Step Sizes | Langevin discretization step sizes. |

---

## 3. Mathematical Foundations & Derivations

### 3.1 The Score Function & Explicit Score Matching
The ground-truth score is defined as $s_{\text{data}}(x) \equiv \nabla_x \log p_{\text{data}}(x)$. We seek a neural network $s_\theta(x)$ that minimizes the expected squared Euclidean difference:
$$J_{ESM}(\theta) = \frac{1}{2} \mathbb{E}_{x \sim p_{\text{data}}(x)} \left[ \|s_\theta(x) - \nabla_x \log p_{\text{data}}(x)\|^2 \right]$$

However, $\nabla_x \log p_{\text{data}}(x)$ is inaccessible because $p_{\text{data}}(x)$ is unknown.

### 3.2 Hyvärinen's Theorem: Derivation of Implicit Score Matching
Aapo Hyvärinen (2005) proved that under mild regularity conditions ($\lim_{\|x\| \to \infty} p_{\text{data}}(x) s_\theta(x) = \mathbf{0}$), $J_{ESM}(\theta)$ can be rewritten without referencing the unknown data score.

**Proof:**
Expand the squared Euclidean norm in $J_{ESM}(\theta)$:
$$\|s_\theta(x) - \nabla_x \log p_{\text{data}}(x)\|^2 = \|s_\theta(x)\|^2 - 2 s_\theta(x)^\top \nabla_x \log p_{\text{data}}(x) + \|\nabla_x \log p_{\text{data}}(x)\|^2$$

Taking expectations over $p_{\text{data}}(x)$:
$$\mathbb{E}_{p_{\text{data}}} [\|s_\theta(x) - \nabla_x \log p_{\text{data}}(x)\|^2] = \mathbb{E}_{p_{\text{data}}} [\|s_\theta(x)\|^2] - 2 \mathbb{E}_{p_{\text{data}}} [s_\theta(x)^\top \nabla_x \log p_{\text{data}}(x)] + C$$
where $C = \mathbb{E}_{p_{\text{data}}}[\|\nabla_x \log p_{\text{data}}(x)\|^2]$ is an independent constant with respect to $\theta$.

Now consider the middle cross-term $\mathcal{I} = \mathbb{E}_{p_{\text{data}}} [s_\theta(x)^\top \nabla_x \log p_{\text{data}}(x)]$. Writing this in integral form:
$$\mathcal{I} = \int_{\mathbb{R}^D} p_{\text{data}}(x) \sum_{i=1}^D s_{\theta, i}(x) \frac{\partial \log p_{\text{data}}(x)}{\partial x_i} \, dx$$

Using the identity $\frac{\partial \log p_{\text{data}}(x)}{\partial x_i} = \frac{1}{p_{\text{data}}(x)} \frac{\partial p_{\text{data}}(x)}{\partial x_i}$:
$$\mathcal{I} = \sum_{i=1}^D \int_{\mathbb{R}^D} s_{\theta, i}(x) \frac{\partial p_{\text{data}}(x)}{\partial x_i} \, dx$$

Apply multivariable integration by parts along dimension $x_i$:
$$\int_{-\infty}^{\infty} s_{\theta, i}(x) \frac{\partial p_{\text{data}}(x)}{\partial x_i} \, dx_i = \underbrace{\left[ s_{\theta, i}(x) p_{\text{data}}(x) \right]_{-\infty}^{\infty}}_{= 0 \text{ (boundary condition)}} - \int_{-\infty}^{\infty} p_{\text{data}}(x) \frac{\partial s_{\theta, i}(x)}{\partial x_i} \, dx_i$$

Summing across all $D$ dimensions:
$$\mathcal{I} = -\int_{\mathbb{R}^D} p_{\text{data}}(x) \sum_{i=1}^D \frac{\partial s_{\theta, i}(x)}{\partial x_i} \, dx = -\mathbb{E}_{p_{\text{data}}(x)} [\text{tr}(\nabla_x s_\theta(x))]$$
where $\nabla_x s_\theta(x) \in \mathbb{R}^{D \times D}$ is the Jacobian matrix of the score network, and $\text{tr}(\nabla_x s_\theta(x)) = \sum_{i=1}^D \frac{\partial s_{\theta, i}(x)}{\partial x_i}$.

Substitute $\mathcal{I}$ back into the objective:
$$J_{ISM}(\theta) = \mathbb{E}_{p_{\text{data}}(x)} \left[ \text{tr}(\nabla_x s_\theta(x)) + \frac{1}{2} \|s_\theta(x)\|^2 \right]$$
**Remarkable Conclusion:** The objective depends *only* on the score network $s_\theta(x)$ and its Jacobian trace evaluated at data samples $x \sim p_{\text{data}}$. No knowledge of $p_{\text{data}}(x)$ is required!

---

### 3.3 The Curse of the Jacobian Trace and Denoising Score Matching (DSM)
While $J_{ISM}(\theta)$ is mathematically exact, evaluating $\text{tr}(\nabla_x s_\theta(x))$ in deep networks requires computing the diagonal entries of the Jacobian. Exact computation requires $D$ separate backward passes with automatic differentiation, which is computationally prohibitive for $D = 3 \times 512 \times 512 \approx 7.8 \times 10^5$.

To overcome this, Pascal Vincent (2011) introduced **Denoising Score Matching (DSM)**. Instead of estimating the score of the clean data, we perturb clean samples $x$ with a known Gaussian noise kernel:
$$q_\sigma(\tilde{x} \mid x) = \mathcal{N}(\tilde{x}; x, \sigma^2 \mathbf{I})$$

The perturbed distribution is $q_\sigma(\tilde{x}) = \int p_{\text{data}}(x) q_\sigma(\tilde{x} \mid x) dx$. Crucially, the **conditional score** of the perturbation kernel is known analytically:
$$\log q_\sigma(\tilde{x} \mid x) = -\frac{D}{2} \log(2\pi \sigma^2) - \frac{\|\tilde{x} - x\|^2}{2\sigma^2}$$
$$\nabla_{\tilde{x}} \log q_\sigma(\tilde{x} \mid x) = -\frac{\tilde{x} - x}{\sigma^2}$$

Vincent proved the foundational equivalence:
$$\mathbb{E}_{x \sim p_{\text{data}}, \tilde{x} \sim q_\sigma(\cdot|x)} \left[ \left\| s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log q_\sigma(\tilde{x} \mid x) \right\|^2 \right] = \mathbb{E}_{\tilde{x} \sim q_\sigma} \left[ \left\| s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log q_\sigma(\tilde{x}) \right\|^2 \right] + \text{const}$$

Thus, the Denoising Score Matching objective simplifies to:
$$J_{DSM}(\theta) = \frac{1}{2} \mathbb{E}_{x \sim p_{\text{data}}, \tilde{x} \sim \mathcal{N}(x, \sigma^2 \mathbf{I})} \left[ \left\| s_\theta(\tilde{x}) + \frac{\tilde{x} - x}{\sigma^2} \right\|^2 \right]$$
This requires zero second-order derivatives and zero Jacobian computations! Training simply regresses the neural network output $s_\theta(\tilde{x})$ toward the analytical vector $-\frac{\tilde{x} - x}{\sigma^2}$.

---

### 3.4 Langevin Dynamics and Annealed Sampling
Once the score function $s_\theta(x) \approx \nabla_x \log p(x)$ is trained, how do we generate new samples?

Langevin Dynamics is a continuous-time stochastic differential equation (SDE) derived from statistical physics:
$$dx_t = \frac{1}{2} \nabla_x \log p(x_t) \, dt + dw_t$$
where $w_t$ is standard Brownian motion. Under mild conditions, the stationary distribution of this SDE is precisely $p(x)$!

Using Euler-Maruyama discretization with step size $\epsilon > 0$:
$$x_{t+1} = x_t + \frac{\epsilon}{2} s_\theta(x_t) + \sqrt{\epsilon} \, z_t, \quad z_t \sim \mathcal{N}(0, \mathbf{I})$$

#### The Low-Density Collapse Problem & Annealing
When data concentrates on a low-dimensional manifold embedded in high-dimensional space (the Manifold Hypothesis):
1. In regions where $p_{\text{data}}(x) \approx 0$, no training samples exist, so $s_\theta(x)$ outputs unconstrained noise.
2. If the data distribution is multimodal with high barriers between modes, Langevin dynamics takes exponential time $\mathcal{O}(e^{\Delta E})$ to transit between modes.

Yang Song and Stefano Ermon (2019) solved this by conditioning the score network on a geometric hierarchy of noise scales:
$$\sigma_1 > \sigma_2 > \dots > \sigma_L$$
where $\sigma_1$ is large enough to cover the entire support space, and $\sigma_L \approx 0$ matches the clean data distribution.

```
Annealed Langevin Dynamics Pipeline:
[Initial Noise: x_0 ~ N(0, sigma_1^2 I)]
       │
       ▼ (sigma_1: High noise, mode bridges are open, global exploration)
[Langevin Steps at sigma_1] ---> Updates: x = x + (alpha_1/2)*s_theta(x, sigma_1) + sqrt(alpha_1)*z
       │
       ▼ (sigma_2: Medium noise, coarse shapes form)
[Langevin Steps at sigma_2]
       │
       ▼ (sigma_L: Low noise, fine details sharpen)
[Langevin Steps at sigma_L]
       │
       ▼
[Final High-Fidelity Data Sample x]
```

Step size is scheduled proportionally to the noise scale:
$$\alpha_i = \epsilon \cdot \left( \frac{\sigma_i}{\sigma_L} \right)^2$$
This ensures the drift magnitude $\frac{\alpha_i}{2} \|s_\theta(x, \sigma_i)\| \sim \mathcal{O}\left( \frac{\alpha_i}{\sigma_i} \right)$ balances correctly against stochastic diffusion $\sqrt{\alpha_i} z_t$.

---

## 4. Architectural Implementation Patterns & Best Practices

1. **Noise-Conditioned Score Networks (NCSN):**
   - Condition the score network on $\sigma_i$ via embedding layers (sinusoidal or MLP) or by scaling the output: $s_\theta(x, \sigma) = \frac{s_\theta(x)}{\sigma}$.
2. **Loss Weighting:**
   - Since $\|\nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x)\|^2 = \frac{\|\tilde{x} - x\|^2}{\sigma^4} \sim \mathcal{O}(\frac{1}{\sigma^2})$, multiply each term by $\sigma_i^2$ so all noise levels contribute equally:
   $$\mathcal{L}(\theta; \{\sigma_i\}_{i=1}^L) = \frac{1}{L} \sum_{i=1}^L \frac{\sigma_i^2}{2} \mathbb{E}\left[ \left\| s_\theta(x + \sigma_i z, \sigma_i) + \frac{z}{\sigma_i} \right\|^2 \right]$$
3. **Number of Steps per Noise Scale:**
   - Practical heuristic: 50 to 100 Langevin iterations per scale, with $L \in [10, 50]$.
