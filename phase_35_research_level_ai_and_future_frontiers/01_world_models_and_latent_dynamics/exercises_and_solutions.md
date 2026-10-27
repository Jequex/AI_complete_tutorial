# Topic 01: Exercises & Solutions — World Models and Latent Dynamics

## Exercise 1: KL Divergence & Latent Information Bottleneck

Consider a world model encoder parameterized as a diagonal multivariate Gaussian posterior:

$$q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}\left(\boldsymbol{\mu}(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}^2(\mathbf{x}))\right)$$

with standard Gaussian prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ where $\mathbf{z} \in \mathbb{R}^d$.

1. Derive the closed-form analytical expression for the Kullback-Leibler divergence $D_{\text{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p(\mathbf{z}))$.
2. Explain the role of the KL term in the ELBO objective as an **information bottleneck** and what occurs when the weighting coefficient $\beta$ on the KL term is set too high ($\beta \gg 1$) versus too low ($\beta \to 0$).

---

### Solution 1:

#### 1. Derivation of the KL Divergence
For two multivariate Gaussians $q(\mathbf{z}) = \mathcal{N}(\boldsymbol{\mu}_q, \boldsymbol{\Sigma}_q)$ and $p(\mathbf{z}) = \mathcal{N}(\boldsymbol{\mu}_p, \boldsymbol{\Sigma}_p)$ in $\mathbb{R}^d$:

$$D_{\text{KL}}(q \parallel p) = \frac{1}{2} \left[ \text{tr}(\boldsymbol{\Sigma}_p^{-1} \boldsymbol{\Sigma}_q) + (\boldsymbol{\mu}_p - \boldsymbol{\mu}_q)^\top \boldsymbol{\Sigma}_p^{-1} (\boldsymbol{\mu}_p - \boldsymbol{\mu}_q) - d + \ln\left(\frac{\det \boldsymbol{\Sigma}_p}{\det \boldsymbol{\Sigma}_q}\right) \right]$$

Setting $\boldsymbol{\mu}_p = \mathbf{0}$, $\boldsymbol{\Sigma}_p = \mathbf{I}$, and $\boldsymbol{\Sigma}_q = \text{diag}(\sigma_1^2, \dots, \sigma_d^2)$:
- $\text{tr}(\boldsymbol{\Sigma}_p^{-1} \boldsymbol{\Sigma}_q) = \sum_{j=1}^d \sigma_j^2$
- $(\boldsymbol{\mu}_p - \boldsymbol{\mu}_q)^\top \boldsymbol{\Sigma}_p^{-1} (\boldsymbol{\mu}_p - \boldsymbol{\mu}_q) = \sum_{j=1}^d \mu_j^2$
- $\det \boldsymbol{\Sigma}_p = 1$, and $\det \boldsymbol{\Sigma}_q = \prod_{j=1}^d \sigma_j^2 \implies \ln\left(\frac{\det \boldsymbol{\Sigma}_p}{\det \boldsymbol{\Sigma}_q}\right) = -\sum_{j=1}^d \ln(\sigma_j^2)$

Combining these terms gives the closed-form sum:

$$D_{\text{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p(\mathbf{z})) = -\frac{1}{2} \sum_{j=1}^d \left( 1 + \ln(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$

#### 2. Information Bottleneck Trade-Off
- **Role**: The KL penalty enforces that the continuous representation $\mathbf{z}$ does not memorize high-frequency pixel noise or act as an unconstrained lookup table. It measures the bits of information transmitted about observation $\mathbf{x}$ through the latent channel.
- **$\beta \gg 1$ (Posterior Collapse)**: The network forces $q_\phi(\mathbf{z} \mid \mathbf{x}) \to \mathcal{N}(\mathbf{0}, \mathbf{I})$, ignoring input $\mathbf{x}$. $\mathbf{z}$ carries 0 bits of information, causing the decoder to generate blurry unconditional averages and crippling downstream planning.
- **$\beta \to 0$ (Representation Drift / Overfitting)**: The latent space fails to regularize into a smooth manifold. Nearby points in latent space no longer map to plausible physical world states, meaning imagination rollouts diverge into unphysical states.

---

## Exercise 2: Open-Loop Error Propagation in Latent Dynamics

Consider a linear latent dynamics model:

$$\mathbf{s}_{t+1} = \mathbf{W} \mathbf{s}_t + \mathbf{B} \mathbf{a}_t + \boldsymbol{\epsilon}_t, \quad \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$$

where errors $\boldsymbol{\epsilon}_t$ are independent and identically distributed across time steps.

1. Express the predicted latent state $\mathbf{s}_H$ at horizon $H$ as a function of the initial state $\mathbf{s}_0$, action sequence $\mathbf{a}_{0:H-1}$, and noise terms.
2. Derive the covariance matrix $\boldsymbol{\Sigma}_H = \text{Cov}(\mathbf{s}_H)$ conditioned on $\mathbf{s}_0$ and actions.
3. State the condition on the spectral radius $\rho(\mathbf{W})$ required to prevent exponential error compounding during long open-loop imagination rollouts.

---

### Solution 2:

#### 1. Unrolling the Recurrence
Unrolling step-by-step:
- $\mathbf{s}_1 = \mathbf{W}\mathbf{s}_0 + \mathbf{B}\mathbf{a}_0 + \boldsymbol{\epsilon}_0$
- $\mathbf{s}_2 = \mathbf{W}^2\mathbf{s}_0 + \mathbf{W}\mathbf{B}\mathbf{a}_0 + \mathbf{B}\mathbf{a}_1 + \mathbf{W}\boldsymbol{\epsilon}_0 + \boldsymbol{\epsilon}_1$
- By induction:

$$\mathbf{s}_H = \mathbf{W}^H \mathbf{s}_0 + \sum_{\tau=0}^{H-1} \mathbf{W}^{H-1-\tau} \mathbf{B} \mathbf{a}_\tau + \sum_{\tau=0}^{H-1} \mathbf{W}^{H-1-\tau} \boldsymbol{\epsilon}_\tau$$

#### 2. Covariance Matrix Derivation
Conditioned on fixed initial state $\mathbf{s}_0$ and deterministic action sequence $\mathbf{a}_{0:H-1}$, only the noise terms $\boldsymbol{\epsilon}_\tau$ contribute to variance:

$$\text{Cov}(\mathbf{s}_H) = \mathbb{E}\left[ \left(\sum_{\tau=0}^{H-1} \mathbf{W}^{H-1-\tau} \boldsymbol{\epsilon}_\tau \right) \left(\sum_{k=0}^{H-1} \mathbf{W}^{H-1-k} \boldsymbol{\epsilon}_k \right)^\top \right]$$

Since $\mathbb{E}[\boldsymbol{\epsilon}_\tau \boldsymbol{\epsilon}_k^\top] = \sigma^2 \mathbf{I} \cdot \delta_{\tau, k}$:

$$\boldsymbol{\Sigma}_H = \sigma^2 \sum_{k=0}^{H-1} \mathbf{W}^k (\mathbf{W}^k)^\top$$

#### 3. Stability Condition
For the accumulated covariance $\boldsymbol{\Sigma}_H$ to remain bounded as $H \to \infty$, the series $\sum_{k=0}^\infty \mathbf{W}^k (\mathbf{W}^k)^\top$ must converge.
By the Gelfand spectral radius formula, this requires:

$$\rho(\mathbf{W}) = \max_i |\lambda_i(\mathbf{W})| < 1$$

If $\rho(\mathbf{W}) \ge 1$, errors explode exponentially with horizon $H$ ($\sim \rho(\mathbf{W})^{2H}$), rendering open-loop imagination rollouts unusable beyond a short planning window. This is why modern world models (like DreamerV3) use normalized representations, residual dampening, or recurrent normalization (LayerNorm) to constrain transition eigenvalues.

---

## Exercise 3: Latent MPC vs. Pixel-Space Simulation Complexity

Suppose an agent executes Model Predictive Control with population size $K = 100$ candidate trajectories, planning horizon $H = 12$, and $I = 3$ iterations of the Cross-Entropy Method.
- Observation space: $128 \times 128 \times 3$ RGB image ($D = 49,152$ floats).
- Latent space: $\mathbf{s} \in \mathbb{R}^{32}$.
- Action space: $\mathbf{a} \in \mathbb{R}^4$.

Calculate and compare the floating point operations (FLOPs) required per planning step under:
1. **Pixel-space generative simulation** (requiring a deconvolutional decoder of $10^7$ FLOPs per step to generate $\hat{\mathbf{x}}_{t+1}$).
2. **Latent-space simulation** (where dynamics $f(\mathbf{s}, \mathbf{a})$ and reward head $R(\mathbf{s}, \mathbf{a})$ take two $64 \times 36$ matrix multiplies, $\approx 5 \times 10^3$ FLOPs per step).

---

### Solution 3:

#### 1. Total Rollout Steps
In both paradigms, the total number of simulated transition steps across iterations is:
$$N_{\text{steps}} = I \times K \times H = 3 \times 100 \times 12 = 3,600 \text{ transitions}$$

#### 2. Pixel-Space MPC Cost
If each simulated transition requires decoding an image to evaluate reward or visual physics:
$$\text{FLOPs}_{\text{pixel}} = 3,600 \times 10^7 = 3.6 \times 10^{10} \text{ FLOPs } (36 \text{ GFLOPs})$$
At a typical embedded robot inference throughput of $1 \text{ TFLOP/s}$, this requires $36 \text{ ms}$ just for planning a single control step, risking real-time lag.

#### 3. Latent-Space MPC Cost
In latent space, observation decoding is completely skipped during planning. The agent only evaluates the compact latent dynamics and reward head:
$$\text{FLOPs}_{\text{latent}} = 3,600 \times (5 \times 10^3) = 1.8 \times 10^7 \text{ FLOPs } (0.018 \text{ GFLOPs})$$

#### 4. Speedup Factor:
$$\text{Speedup} = \frac{3.6 \times 10^{10}}{1.8 \times 10^7} = 2,000\times$$
Latent planning executes 2,000 times faster, permitting high-frequency real-time reactive control while maintaining accurate trajectory predictions.
