# Phase 21: Preference Optimization (RLHF & DPO)

---

## 🧭 Executive Overview

Pre-training across trillions of tokens creates an autoregressive next-token predictor, and Supervised Fine-Tuning (SFT) teaches the model conversational turn-taking and instruction formats. However, SFT treats all training demonstrations as equally optimal, lacking any concept of degree, safety boundaries, or human taste.

**Phase 21: Preference Optimization (RLHF & DPO)** covers the theoretical, mathematical, and algorithmic mechanisms used to align Large Language Models with nuanced human preferences:
- Mathematical foundation of pairwise comparisons over noisy scalar grades via the **Bradley-Terry model**.
- Neural **Reward Model (RM)** architectures $[d_{\text{model}} \to 1]$, pairwise ranking loss, and diagnosing **Reward Hacking / Goodhart's Law**.
- Classic **Reinforcement Learning from Human Feedback (RLHF)** with **Proximal Policy Optimization (PPO)**, token-level **KL divergence penalties**, and **Generalized Advantage Estimation (GAE)** across a 4-model memory topology.
- The revolutionary **Direct Preference Optimization (DPO)** algorithm, deriving the exact closed-form policy under KL constraints, canceling out the partition function $Z(x)$, and eliminating both Reward and Critic networks.
- **Kahneman-Tversky Optimization (KTO)** for unpaired binary feedback (thumbs-up / thumbs-down) grounded in **Prospect Theory**, and **Online Iterative DPO** to prevent off-policy distribution shift.

Every component across this phase is built from scratch in pure Python 3 with **zero external dependencies** (no PyTorch, no Hugging Face, no NumPy, no external APIs).

---

## 🗺️ Phase Syllabus & Topic Breakdown

```text
phase_21_preference_optimization_rlhf_and_dpo/
├── index.html                                                    # Phase Portal Hub & Visual Syllabus
├── README.md                                                     # Master Architecture Guide & Derivations
│
├── 01_human_preference_and_bradley_terry/
│   ├── lecture.html / lecture.md                                 # Human scalar noise, Bradley-Terry logistic model, Plackett-Luce, transitivity
│   ├── exercises_and_solutions.html / .md                        # Bias cancellation proof, MLE gradient & Hessian, IIA property, Condorcet cycles
│   └── experiment.py                                             # Pure Python Bradley-Terry MLE solver, rank recovery, SST verification (Exit Code 0)
│
├── 02_reward_modeling_and_loss_derivation/
│   ├── lecture.html / lecture.md                                 # Reward model head [d_model -> 1], causal EOS extraction, ranking loss, Goodhart's law
│   ├── exercises_and_solutions.html / .md                        # RM parameter arithmetic, margin penalty derivation, score normalization, length bias
│   └── experiment.py                                             # Pure Python RewardModel, pairwise cross-entropy loss, margin growth, Goodhart test (Exit Code 0)
│
├── 03_rlhf_and_ppo_policy_gradients/
│   ├── lecture.html / lecture.md                                 # 4-model memory topology, token-level KL penalty, PPO clipped surrogate, GAE
│   ├── exercises_and_solutions.html / .md                        # 4-model VRAM arithmetic (196 GB), KL decomposition proof, ratio clipping, GAE recursion
│   └── experiment.py                                             # Pure Python 4-model PPO simulation, token KL, GAE recursion, ratio clipping (Exit Code 0)
│
├── 04_direct_preference_optimization_dpo/
│   ├── lecture.html / lecture.md                                 # Rafailov et al. breakthrough, closed-form optimal policy, Z(x) cancellation, DPO loss
│   ├── exercises_and_solutions.html / .md                        # Step-by-step Z(x) cancellation, dynamic error-weighting gradient, beta role, DPO vs SFT
│   └── experiment.py                                             # Pure Python DPO loss engine, implicit reward dynamics, error weighting damping (Exit Code 0)
│
└── 05_kto_and_online_preference_optimization/
    ├── lecture.html / lecture.md                                 # Unpaired binary telemetry, Prospect Theory loss aversion, reference points, Online DPO
    ├── exercises_and_solutions.html / .md                        # Asymmetric loss weighting, dynamic z_ref drift, telemetry economics, on-policy correction
    └── experiment.py                                             # Pure Python KTO loss, Prospect Theory curves, unpaired policy update, Online DPO loop (Exit Code 0)
```

---

## 🔑 Core Mathematical Pillars Mastered in Phase 21

### 1. The Bradley-Terry Preference Likelihood (1952)
$$\boxed{P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l)) = \frac{1}{1 + \exp\left(-(r(x, y_w) - r(x, y_l))\right)}}$$
Guarantees scale-free, translation-invariant preference modeling with Strong Stochastic Transitivity (SST).

### 2. Pairwise Reward Model Ranking Loss
$$\boxed{\mathcal{L}_{\text{RM}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right) \right]}$$
Forces the neural reward model $[d_{\text{model}} \to 1]$ evaluated at terminal token $h_T$ ($<\text{eos}>$) to maximize the score margin between preferred and dispreferred completions.

### 3. The KL-Regularized RLHF Objective
$$\boxed{\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ r_\phi(x, y) \right] - \beta D_{\text{KL}}(\pi_\theta(y \mid x) \parallel \pi_{\text{ref}}(y \mid x))}$$
Prevents catastrophic distribution shift, policy collapse, and reward hacking by anchoring the active policy to the SFT reference model.

### 4. The Closed-Form DPO Loss Function (Rafailov et al., 2023)
$$\boxed{\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]}$$
By algebraically canceling the partition function $Z(x)$, DPO optimizes policy weights directly from preference pairs, entirely eliminating separate Reward and Critic networks.

### 5. Kahneman-Tversky Optimization (KTO) Objective (Ethayarajh et al., 2024)
$$\boxed{\mathcal{L}_{\text{KTO}}(\theta) = \mathbb{E}_{\mathcal{D}_D} \left[ 1 - \sigma\left(\lambda_D (z - z_{\text{ref}})\right) \right] + \lambda_{\text{undesirable}} \mathbb{E}_{\mathcal{D}_U} \left[ 1 - \sigma\left(\lambda_U (z_{\text{ref}} - z)\right) \right]}$$
Enables direct optimization from unpaired binary feedback (thumbs-up / thumbs-down), utilizing Prospect Theory loss aversion ($\lambda_{\text{undesirable}} > 1.0$) relative to moving reference baseline $z_{\text{ref}}$.

---

## 🧪 Experiment Verification Suite

All experiments are executable directly in any standard Python 3 environment without external package installations:

```bash
# Topic 21.1: Bradley-Terry MLE & Transitivity Verification
python3 phase_21_preference_optimization_rlhf_and_dpo/01_human_preference_and_bradley_terry/experiment.py

# Topic 21.2: Neural Reward Model, Pairwise Loss & Goodhart Bias
python3 phase_21_preference_optimization_rlhf_and_dpo/02_reward_modeling_and_loss_derivation/experiment.py

# Topic 21.3: 4-Model RLHF, Token KL Penalty, PPO Clipped Surrogate & GAE
python3 phase_21_preference_optimization_rlhf_and_dpo/03_rlhf_and_ppo_policy_gradients/experiment.py

# Topic 21.4: Direct Preference Optimization (DPO) Closed-Form Loss & Implicit Rewards
python3 phase_21_preference_optimization_rlhf_and_dpo/04_direct_preference_optimization_dpo/experiment.py

# Topic 21.5: KTO Unpaired Optimization & Online Iterative DPO Loop
python3 phase_21_preference_optimization_rlhf_and_dpo/05_kto_and_online_preference_optimization/experiment.py
```
