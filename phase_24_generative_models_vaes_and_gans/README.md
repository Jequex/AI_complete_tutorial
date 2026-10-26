# Phase 24: Generative Models (VAEs & GANs)

---

## 🧭 Executive Overview

In discriminative modeling, machines calculate conditional probabilities $P(y \mid x)$—predicting categorical labels or continuous targets given input features. **Generative modeling** addresses an infinitely more ambitious problem: uncovering and synthesizing the true, high-dimensional probability distribution $P(x)$ of natural data (e.g. images, audio signals, text sequences).

A generative system enables three fundamental capabilities:
1. **High-Fidelity Synthesis:** Sampling novel, previously unobserved instances $\tilde{x} \sim P_\theta(x)$ that match natural data.
2. **Density Evaluation (when tractable):** Computing the exact or approximate likelihood $P_\theta(x)$ of an observation for anomaly detection and compression.
3. **Continuous Representation Learning:** Mapping complex real-world data manifolds to well-behaved, smooth latent coordinates $z \in \mathcal{Z}$.

Across Phase 24, we investigate the two landmark families that revolutionized modern deep generative modeling: **Variational Autoencoders (VAEs)** and **Generative Adversarial Networks (GANs)**, culminating in the optimal transport formulation of **Wasserstein GANs with Gradient Penalty (WGAN-GP)**.

Every module across Phase 24 is built from first principles in pure Python 3 with **zero external dependencies** (no PyTorch, no NumPy, no SciPy, standard library only).

---

## 🗺️ Phase Syllabus & Topic Breakdown

```text
phase_24_generative_models_vaes_and_gans/
├── index.html                                                    # Interactive Phase Visual Hub & Dark/Light Portal
├── README.md                                                     # Master Architecture Guide & Mathematical Pillars
│
├── 01_taxonomy_of_generative_models_and_autoregressive/
│   ├── lecture.html / lecture.md                                 # Explicit vs Implicit, AR chain rule, empty latent space failure
│   ├── exercises_and_solutions.html / .md                        # AR vs GAN complexity, high-dim empty space, NLL gradients
│   └── experiment.py                                             # Pure Python AR generator vs Deterministic AE void failure (Exit Code 0)
│
├── 02_variational_autoencoders_and_elbo/
│   ├── lecture.html / lecture.md                                 # Intractable posterior, variational inference, ELBO derivations, Gaussian KL
│   ├── exercises_and_solutions.html / .md                        # Analytical KL calculation, ELBO exactness proof, beta-VAE collapse
│   └── experiment.py                                             # Pure Python ELBO decomposition, Gaussian KL, discrete toy verification (Exit Code 0)
│
├── 03_reparameterization_trick_and_latent_space/
│   ├── lecture.html / lecture.md                                 # Stochastic node roadblock, pathwise gradients, high-dim annulus, LERP vs SLERP
│   ├── exercises_and_solutions.html / .md                        # Pathwise backprop steps, REINFORCE variance proof, midpoint norm collapse
│   └── experiment.py                                             # Pure Python mini-VAE from scratch, pathwise backprop, SLERP interpolation (Exit Code 0)
│
├── 04_generative_adversarial_networks_gans/
│   ├── lecture.html / lecture.md                                 # Minimax formulation, optimal discriminator D*, JSD equivalence, mode collapse
│   ├── exercises_and_solutions.html / .md                        # Nash equilibrium value, non-saturating gradient dynamics, 1D discrete oscillation
│   └── experiment.py                                             # Pure Python 1D GAN trained on bimodal Gaussian mixture, JSD tracking (Exit Code 0)
│
└── 05_wasserstein_gan_and_gradient_penalty/
    ├── lecture.html / lecture.md                                 # Manifold hypothesis, disjoint support failure, W1, Kantorovich-Rubinstein, GP
    ├── exercises_and_solutions.html / .md                        # W1 on Dirac spikes, Lipschitz-gradient equivalence, unit norm geodesic proof
    └── experiment.py                                             # Pure Python parallel manifold test, 1-Lipschitz critic with GP (Exit Code 0)
```

---

## 🔑 Core Mathematical Pillars Mastered in Phase 24

### 1. Autoregressive Chain Rule Factorization:
$$\boxed{P(x_1, x_2, \dots, x_T) = \prod_{t=1}^T P(x_t \mid x_{<t})}$$
Allows exact likelihood computation and stable maximum likelihood training via teacher forcing, but scales with $\mathcal{O}(T)$ sequential latency at inference.

### 2. The Evidence Lower Bound (ELBO):
$$\boxed{\log p_\theta(x) = \text{ELBO}(\theta, \phi; x) + D_{\text{KL}}(q_\phi(z \mid x) \parallel p_\theta(z \mid x))}$$
where:
$$\boxed{\text{ELBO}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z \mid x)} [\log p_\theta(x \mid z)] - D_{\text{KL}}(q_\phi(z \mid x) \parallel p(z))}$$
Because $D_{\text{KL}} \ge 0$, $\text{ELBO} \le \log p(x)$. Maximizing the ELBO pushes the lower bound upward while simultaneously minimizing the divergence to the true intractable posterior $p(z \mid x)$.

### 3. Closed-Form Gaussian Prior KL Divergence:
For $q_\phi(z \mid x) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ and prior $p(z) = \mathcal{N}(0, \mathbf{I})$:
$$\boxed{D_{\text{KL}}(q_\phi(z \mid x) \parallel p(z)) = -\frac{1}{2} \sum_{j=1}^J \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)}$$

### 4. The Reparameterization Trick & Pathwise Gradient:
Isolating external stochastic noise $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$ via $z = g_\phi(x, \boldsymbol{\epsilon}) = \boldsymbol{\mu}_\phi(x) + \boldsymbol{\sigma}_\phi(x) \odot \boldsymbol{\epsilon}$:
$$\boxed{\nabla_\phi \mathbb{E}_{z \sim q_\phi(z \mid x)} [f(z)] = \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \left[ \nabla_z f(z) \cdot \nabla_\phi g_\phi(x, \boldsymbol{\epsilon}) \right]}$$
Provides a continuous, low-variance pathwise gradient that enables end-to-end backpropagation through the stochastic bottleneck.

### 5. Optimal Discriminator $D^*(x)$:
In the zero-sum minimax game $\min_G \max_D V(D, G)$, for any fixed generator $G$, the unique global optimal discriminator is:
$$\boxed{D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}}$$

### 6. Minimax Equivalence to Jensen-Shannon Divergence:
Evaluating the value function at $D = D^*$ yields:
$$\boxed{V(D^*, G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \parallel p_g)}$$
The global minimum $-\log 4$ is achieved if and only if $\text{JSD} = 0 \iff p_g = p_{\text{data}}$ almost everywhere, at which point $D^*(x) = 0.5$.

### 7. Kantorovich-Rubinstein Duality & Earth Mover's Distance:
$$\boxed{\mathcal{W}_1(P_r, P_g) = \sup_{\|f\|_L \le 1} \left( \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{y \sim P_g}[f(y)] \right)}$$
Eliminates vanishing gradients on non-overlapping manifolds by providing a continuous, linear transport metric everywhere.

### 8. WGAN-GP Two-Sided Geodesic Gradient Penalty:
$$\boxed{\mathcal{L}_{\text{critic}} = \mathbb{E}[f(\tilde{x})] - \mathbb{E}[f(x)] + \lambda_{\text{GP}} \mathbb{E}_{\hat{x} \sim P_{\hat{x}}} \left[ \left( \|\nabla_{\hat{x}} f(\hat{x})\|_2 - 1 \right)^2 \right]}$$
Enforces 1-Lipschitz continuity on random straight line interpolations $\hat{x} = \epsilon x + (1 - \epsilon) \tilde{x}$ without the pathologies of weight clipping.

---

## 🧪 Verification & Execution

Every experiment script is tested and runs cleanly in standard Python 3:

```bash
# 1. Autoregressive Modeling vs Deterministic Latent Space Failure
python3 phase_24_generative_models_vaes_and_gans/01_taxonomy_of_generative_models_and_autoregressive/experiment.py

# 2. Variational Inference & Analytical ELBO Verification
python3 phase_24_generative_models_vaes_and_gans/02_variational_autoencoders_and_elbo/experiment.py

# 3. Mini-VAE with Reparameterization Trick & Latent Space Interpolation
python3 phase_24_generative_models_vaes_and_gans/03_reparameterization_trick_and_latent_space/experiment.py

# 4. 1D Minimax GAN & Jensen-Shannon Divergence Tracking
python3 phase_24_generative_models_vaes_and_gans/04_generative_adversarial_networks_gans/experiment.py

# 5. Wasserstein GAN & 1-Lipschitz Gradient Penalty (WGAN-GP)
python3 phase_24_generative_models_vaes_and_gans/05_wasserstein_gan_and_gradient_penalty/experiment.py
```
