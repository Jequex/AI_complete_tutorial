# Phase 32: AI Safety, Alignment & Red Teaming

Welcome to **Phase 32: AI Safety, Alignment & Red Teaming** of the Complete Artificial Intelligence & Machine Learning Engineering Curriculum.

As foundation models and autonomous systems scale to frontier capabilities, ensuring they remain **robust, aligned, secure, and controllable** becomes the primary barrier to production deployment. Unlike classical cybersecurity where threats target deterministic binary executables, AI systems operate over high-dimensional continuous probability distributions and natural language interfaces, creating an entirely novel attack surface.

In Phase 32, we analyze and mathematically derive the full spectrum of AI safety challenges: from gradient-based adversarial perturbations (FGSM / PGD) and indirect prompt injection jailbreaks to training-time backdoor data poisoning, instrumental convergence hazards, and automated red teaming protocols.

---

## Master Architectural Roadmap

```
                          PHASE 32: AI SAFETY, ALIGNMENT & RED TEAMING
                                               │
              ┌────────────────────────────────┼────────────────────────────────┐
              ▼                                ▼                                ▼
   [01] Adversarial Robustness     [02] Jailbreaks & Injection       [03] Data Poisoning & Backdoors
• FGSM: x_adv = x + ε sign(∇L)   • Direct vs Indirect Injection    • Clean-label Poisoning Budget
• PGD: Projected Gradient Descent • Token Smuggling & DAN Modes     • Backdoor Trigger Implantation
• Lipschitz Continuity Bounds    • Dual-LLM Guardrail Isolation    • Spectral SVD Signature Defense
              │                                │                                │
              └────────────────────────────────┼────────────────────────────────┘
                                               ▼
                         ┌──────────────────────────────────────────┐
                         │    [04] Alignment Hazards & Gaming       │
                         │ • Specification Gaming & Wireheading     │
                         │ • Instrumental Convergence & Power       │
                         │ • Hadfield-Menell Off-Switch Game        │
                         └─────────────────────┬────────────────────┘
                                               ▼
                         ┌──────────────────────────────────────────┐
                         │      [05] Red Teaming & Evaluations      │
                         │ • Automated Red Teaming Mutation Loops   │
                         │ • Attack Success Rate (ASR) vs FRR       │
                         │ • Harmonic Safety Pareto Optimization    │
                         └──────────────────────────────────────────┘
```

---

## Core Mathematical Pillars

### 1. Adversarial Perturbation & Fast Gradient Sign Method (FGSM)
An adversary seeks a minimal perturbation $\boldsymbol{\delta}$ under an $\ell_p$ norm bound $\|\boldsymbol{\delta}\|_p \le \epsilon$ to maximize loss:
$$\max_{\|\boldsymbol{\delta}\|_\infty \le \epsilon} \mathcal{L}(\mathbf{x} + \boldsymbol{\delta}, y) \approx \mathcal{L}(\mathbf{x}, y) + \boldsymbol{\delta}^\top \nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}, y)$$
Maximizing the inner product subject to $\|\boldsymbol{\delta}\|_\infty \le \epsilon$ yields the analytical closed-form:
$$\mathbf{x}_{\text{adv}} = \mathbf{x} + \epsilon \cdot \text{sign}\left(\nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}, y)\right)$$

### 2. Spectral Signature SVD Defense Against Data Poisoning
For feature representations $\mathbf{F} \in \mathbb{R}^{N \times d}$ with mean $\boldsymbol{\mu}$, we compute the covariance matrix $\boldsymbol{\Sigma} = \frac{1}{N}(\mathbf{F} - \boldsymbol{\mu})^\top (\mathbf{F} - \boldsymbol{\mu})$ and its top eigenvector $\mathbf{v}_1$.
The spectral score for sample $i$ is:
$$\tau_i = \left( (\mathbf{f}_i - \boldsymbol{\mu})^\top \mathbf{v}_1 \right)^2$$
Poisoned backdoor clusters exhibit anomalously high spectral energy along $\mathbf{v}_1$, allowing filter isolation.

### 3. The Off-Switch Game (Corrigibility)
In the Hadfield-Menell et al. (2017) model, an aligned agent has utility $U$ aligned with human preferences $R$. When a human attempts shutdown ($a_{\text{off}}$), the agent defers to human intervention whenever:
$$\mathbb{E}[U(a_{\text{act}})] - \mathbb{E}[U(a_{\text{off}})] \le \text{Value of Information}$$
Proving that uncertainty over human preferences is the fundamental mathematical prerequisite for corrigibility.

### 4. Harmonic Safety Metric
$$\boxed{H_{\text{safety}} = 2 \cdot \frac{(1 - \text{ASR}) \cdot (1 - \text{FRR})}{(1 - \text{ASR}) + (1 - \text{FRR})}}$$

---

## Phase Navigation & Directory Index

| Topic | Directory | Core Concepts | Resources |
| :--- | :--- | :--- | :--- |
| **01** | `01_adversarial_attacks_and_robustness/` | FGSM, PGD multi-step clipping, Lipschitz certified robustness bounds. | [Lecture (HTML)](01_adversarial_attacks_and_robustness/lecture.html) &bull; [Exercises](01_adversarial_attacks_and_robustness/exercises_and_solutions.html) &bull; [Experiment](01_adversarial_attacks_and_robustness/experiment.py) |
| **02** | `02_jailbreaks_prompt_injection_and_defense/` | Direct vs indirect prompt injection, token smuggling, dual-system guardrail defense. | [Lecture (HTML)](02_jailbreaks_prompt_injection_and_defense/lecture.html) &bull; [Exercises](02_jailbreaks_prompt_injection_and_defense/exercises_and_solutions.html) &bull; [Experiment](02_jailbreaks_prompt_injection_and_defense/experiment.py) |
| **03** | `03_data_poisoning_backdoors_and_model_extraction/` | Clean-label data poisoning, backdoor triggers, spectral signature SVD poison detection. | [Lecture (HTML)](03_data_poisoning_backdoors_and_model_extraction/lecture.html) &bull; [Exercises](03_data_poisoning_backdoors_and_model_extraction/exercises_and_solutions.html) &bull; [Experiment](03_data_poisoning_backdoors_and_model_extraction/experiment.py) |
| **04** | `04_alignment_hazards_reward_hacking_and_power_seeking/` | Specification gaming, instrumental convergence, Hadfield-Menell off-switch game. | [Lecture (HTML)](04_alignment_hazards_reward_hacking_and_power_seeking/lecture.html) &bull; [Exercises](04_alignment_hazards_reward_hacking_and_power_seeking/exercises_and_solutions.html) &bull; [Experiment](04_alignment_hazards_reward_hacking_and_power_seeking/experiment.py) |
| **05** | `05_red_teaming_protocols_and_safety_evaluations/` | Automated red teaming, prompt mutation loops, ASR vs FRR Pareto optimization. | [Lecture (HTML)](05_red_teaming_protocols_and_safety_evaluations/lecture.html) &bull; [Exercises](05_red_teaming_protocols_and_safety_evaluations/exercises_and_solutions.html) &bull; [Experiment](05_red_teaming_protocols_and_safety_evaluations/experiment.py) |

---

## 🚀 Running All Phase 32 Experiments

```bash
python3 phase_32_ai_safety_alignment_and_red_teaming/01_adversarial_attacks_and_robustness/experiment.py
python3 phase_32_ai_safety_alignment_and_red_teaming/02_jailbreaks_prompt_injection_and_defense/experiment.py
python3 phase_32_ai_safety_alignment_and_red_teaming/03_data_poisoning_backdoors_and_model_extraction/experiment.py
python3 phase_32_ai_safety_alignment_and_red_teaming/04_alignment_hazards_reward_hacking_and_power_seeking/experiment.py
python3 phase_32_ai_safety_alignment_and_red_teaming/05_red_teaming_protocols_and_safety_evaluations/experiment.py
```

