# Phase 25: Diffusion Models & Flow Matching

---

## 🧭 Executive Overview

Generative modeling in high dimensions has historically wrestled with the tension between sample quality, training stability, and likelihood tractability. Autoregressive models are slow during sequential inference; Variational Autoencoders (VAEs) suffer from blurry samples due to imperfect posterior approximations; and Generative Adversarial Networks (GANs) face notorious mode collapse and training instabilities due to adversarial minimax dynamics.

**Phase 25: Diffusion Models & Flow Matching** explores the modern state-of-the-art paradigm in generative modeling: framing data generation as the progressive inversion of a continuous thermodynamic corruption process or the deterministic integration of optimal transport vector fields.

Every module across Phase 25 is implemented from first principles in pure standard Python 3 with **zero external dependencies** (no PyTorch, no Hugging Face, no NumPy, no external API keys).

Across five core modules, this phase establishes:
1. **Score-Based Models & Langevin Dynamics:** Bypassing the intractable partition function $Z_\theta = \int e^{-E_\theta(x)} dx$ via the Stein score $s_\theta(x) = \nabla_x \log p(x) = -\nabla_x E_\theta(x)$. Hyv&auml;rinen's (2005) Implicit Score Matching derivation via integration by parts, Vincent's (2011) Denoising Score Matching, the manifold hypothesis failure modes in low-density data voids, and Song & Ermon's (2019) Annealed Langevin Dynamics.
2. **DDPM Forward & Reverse Processes:** Non-equilibrium thermodynamics (Sohl-Dickstein et al. 2015) and Denoising Diffusion Probabilistic Models (Ho et al. 2020). The forward Markov chain $q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t \mathbf{I})$, mathematical induction deriving the closed-form shortcut $q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)\mathbf{I})$, and the complete Bayesian derivation of the true posterior $q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\tilde{\boldsymbol{\mu}}_t, \tilde{\beta}_t \mathbf{I})$ by completing the square in Gaussian exponents.
3. **Noise Schedules & Denoising U-Net Architectures:** Why the linear variance schedule prematurely destroys image signal, the Cosine Noise Schedule (Nichol & Dhariwal 2021) with offset $s=0.008$, full derivation decomposing the Variational Lower Bound (ELBO) into Gaussian KL divergences, Jonathan Ho's simplified variational objective $L_{\text{simple}}(\theta) = \mathbb{E}[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(x_t, t)\|^2]$, and U-Net architectural motifs (sinusoidal harmonic time embeddings, residual blocks, and multi-scale skip connections).
4. **Classifier-Free Guidance (CFG) & Latent Diffusion (LDM):** Conditional generation $p(x \mid c)$, Classifier Guidance $\nabla \log p(x_t \mid c) = \nabla \log p(x_t) + \nabla \log p(c \mid x_t)$, Classifier-Free Guidance (CFG; Ho & Salimans 2021) joint training with condition dropout ($p_{\text{uncond}} \approx 0.2$), guidance extrapolation $\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\emptyset + s(\boldsymbol{\epsilon}_c - \boldsymbol{\epsilon}_\emptyset)$, mode sharpening versus diversity trade-offs, and Latent Diffusion Models (Stable Diffusion; Rombach et al. 2022) shifting diffusion into the perceptual latent space of a pretrained VAE for a $48\times$ dimensionality reduction and a $4096\times$ self-attention acceleration.
5. **Flow Matching & Rectified Flows:** Continuous Normalizing Flows (CNFs) and Neural ODEs $\frac{dx_t}{dt} = v_\theta(x_t, t)$, probability mass conservation via the Continuity Equation $\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t v) = 0$, the Flow Matching objective (Lipman et al. 2022), Conditional Flow Matching with optimal transport interpolation $x_t = (1 - t)x_0 + tx_1$ yielding constant target velocity $u_t = x_1 - x_0$, and Rectified Flows (Liu et al. 2022) with zero-curvature trajectories enabling high-fidelity data generation in 1 to 4 numerical Euler steps.

---

## 🗺️ Phase Syllabus & Topic Breakdown

```text
phase_25_diffusion_models_and_flow_matching/
├── index.html                                                    # Interactive Portal Hub & Visual Syllabus
├── README.md                                                     # Master Architecture Guide & Derivations
│
├── 01_score_based_models_and_langevin_dynamics/
│   ├── lecture.html / lecture.md                                 # Partition function elimination, Score Matching, DSM, Annealed Langevin
│   ├── exercises_and_solutions.html / .md                        # Hyvärinen IBP proof, analytical DSM equivalence, step size scaling
│   └── experiment.py                                             # Pure Python Annealed Langevin Sampler on GMM landscape (Exit Code 0)
│
├── 02_ddpm_forward_and_reverse_processes/
│   ├── lecture.html / lecture.md                                 # Forward Markov chains, closed-form jump shortcut, Bayesian posterior proof
│   ├── exercises_and_solutions.html / .md                        # Completing the square proof, terminal step proof, Tweedie score link
│   └── experiment.py                                             # Pure Python Markov vs shortcut Monte Carlo & signal reconstruction (Exit Code 0)
│
├── 03_noise_schedules_and_denoising_unet/
│   ├── lecture.html / lecture.md                                 # Linear vs Cosine schedule, VLB ELBO decomposition, L_simple, U-Net skips
│   ├── exercises_and_solutions.html / .md                        # Cosine boundaries, VLB weight explosion at t->1, analytical weight gradient
│   └── experiment.py                                             # Pure Python 1D Denoising MLP with sinusoidal time emb & sampling (Exit Code 0)
│
├── 04_classifier_free_guidance_and_latent_diffusion/
│   ├── lecture.html / lecture.md                                 # Classifier guidance, CFG extrapolation formula, joint training, LDM VAE
│   ├── exercises_and_solutions.html / .md                        # Bayes' score gradient, CFG vector extrapolation, latent attention speedup
│   └── experiment.py                                             # Pure Python conditional diffusion with CFG guidance sweeps s in {0,1,3} (Exit Code 0)
│
└── 05_flow_matching_and_rectified_flows/
    ├── lecture.html / lecture.md                                 # Neural ODEs, Continuity Equation, Conditional Flow Matching, Rectified Flows
    ├── exercises_and_solutions.html / .md                        # Zero Euler error proof, Continuity velocity derivation, Fubini CFM proof
    └── experiment.py                                             # Pure Python Rectified Flow & Euler ODE solver on 2D ring (Exit Code 0)
```

---

## 🔑 Core Mathematical Pillars Mastered in Phase 25

1. **Partition Function Vanishing Property (The Score Function):**
   $$\boxed{s_\theta(x) = \nabla_x \log p_\theta(x) = \nabla_x \left( -E_\theta(x) - \log Z_\theta \right) = -\nabla_x E_\theta(x)}$$
   Because $Z_\theta = \int e^{-E_\theta(x)} dx$ does not depend on $x$, $\nabla_x \log Z_\theta \equiv \mathbf{0}$. Unnormalized energy models can be trained without computing intractable partition functions.

2. **Hyv&auml;rinen Implicit Score Matching (Integration by Parts):**
   $$\boxed{J_{ISM}(\theta) = \mathbb{E}_{x \sim p_{\text{data}}(x)} \left[ \text{tr}(\nabla_x s_\theta(x)) + \frac{1}{2} \|s_\theta(x)\|^2 \right]}$$
   Eliminates the unknown data score $\nabla_x \log p_{\text{data}}(x)$ by shifting spatial derivatives onto the neural network Jacobian trace $\text{tr}(\nabla_x s_\theta(x))$.

3. **Closed-Form Forward Marginal Shortcut (DDPM):**
   $$\boxed{q(x_t \mid x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)\mathbf{I}\right), \quad \text{where} \quad \alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s}$$
   Allows sampling any arbitrary intermediate noisy state $x_t$ directly from clean data $x_0$ in $\mathcal{O}(1)$ time without simulating preceding steps.

4. **Tractable Bayesian Posterior Mean & Variance:**
   $$\boxed{\tilde{\boldsymbol{\mu}}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t, \quad \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t}$$
   $$\boxed{\tilde{\boldsymbol{\mu}}_t(x_t, \boldsymbol{\epsilon}) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \right)}$$
   Conditioning on clean source $x_0$ renders the reverse transition exactly Gaussian, proving that predicting injected Gaussian noise $\boldsymbol{\epsilon}$ parameterizes the reverse mean.

5. **Jonathan Ho's Simplified Variational Loss:**
   $$\boxed{L_{\text{simple}}(\theta) \triangleq \mathbb{E}_{t \sim \mathcal{U}(1, T), x_0, \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(x_t, t)\|^2 \right]}$$
   Discarding the theoretical VLB weight $\frac{\beta_t^2}{2\sigma_t^2 \alpha_t(1 - \bar{\alpha}_t)}$ down-weights imperceptible noise at tiny $t$ and focuses model capacity on large-scale semantic structure at large $t$.

6. **Classifier-Free Guidance (CFG) Extrapolation:**
   $$\boxed{\tilde{\boldsymbol{\epsilon}}_\theta(x_t, c) = \boldsymbol{\epsilon}_\theta(x_t, \emptyset) + s \cdot \left( \boldsymbol{\epsilon}_\theta(x_t, c) - \boldsymbol{\epsilon}_\theta(x_t, \emptyset) \right)}$$
   Steers samples along the conditioning direction vector $(\boldsymbol{\epsilon}_c - \boldsymbol{\epsilon}_\emptyset)$ with guidance scale $s \ge 1$ without requiring a separate noisy classifier.

7. **Conditional Flow Matching (CFM) Objective & Straight-Line Trajectories:**
   $$\boxed{L_{CFM}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0, 1], x_0 \sim p_0, x_1 \sim p_{\text{data}}} \left[ \|v_\theta(x_t, t) - (x_1 - x_0)\|^2 \right], \quad x_t = (1 - t)x_0 + tx_1}$$
   Learns straight-line vector fields connecting Gaussian noise to data distributions, minimizing 2-Wasserstein kinetic action and enabling numerical ODE solvers to generate samples in 1 to 4 steps.

---

## 🧪 Verification & Execution

Every experiment script is written in clean, standard library Python 3 with zero external dependencies and exits with code 0:

```bash
# 1. Score-Based Models & Annealed Langevin Dynamics (GMM Landscape Sampling)
python3 phase_25_diffusion_models_and_flow_matching/01_score_based_models_and_langevin_dynamics/experiment.py

# 2. DDPM Forward & Reverse Processes (Markov vs Shortcut Monte Carlo & Inversion)
python3 phase_25_diffusion_models_and_flow_matching/02_ddpm_forward_and_reverse_processes/experiment.py

# 3. Noise Schedules & Denoising Networks (Cosine Schedule & 1D Denoising MLP)
python3 phase_25_diffusion_models_and_flow_matching/03_noise_schedules_and_denoising_unet/experiment.py

# 4. Classifier-Free Guidance & Latent Diffusion (CFG Guidance Sweeps s in {0, 1, 3})
python3 phase_25_diffusion_models_and_flow_matching/04_classifier_free_guidance_and_latent_diffusion/experiment.py

# 5. Flow Matching & Rectified Flows (Straight-Line OT Vector Field on 2D Ring)
python3 phase_25_diffusion_models_and_flow_matching/05_flow_matching_and_rectified_flows/experiment.py
```
