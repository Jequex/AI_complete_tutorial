# Phase 29: Model Compression & Quantization

Welcome to **Phase 29: Model Compression & Quantization** of the Complete Artificial Intelligence & Machine Learning Engineering Curriculum.

Modern deep neural networks and Large Language Models (LLMs) contain tens to hundreds of billions of continuous parameters ($\text{FP32} / \text{BF16}$). While overparameterization enables emergent reasoning and superhuman representation learning, deploying these models in production exposes severe hardware bottlenecks: memory capacity limits, memory bus bandwidth saturation, and thermal power constraints.

**Model Compression & Quantization** is the mathematical discipline of reducing parameter bit-widths and structural counts while preserving downstream accuracy.

---

## Master Architectural Roadmap

```
                          PHASE 29: MODEL COMPRESSION & QUANTIZATION
                                              │
             ┌────────────────────────────────┴────────────────────────────────┐
             ▼                                                                 ▼
   [01] Uniform Quantization                                         [02] PTQ vs. QAT
• Continuous-to-Integer Mapping                                    • PTQ Calibration: MinMax vs Percentile
• Asymmetric (Affine): q = clip(⌊x/s⌉ + z)                        • KL Divergence Minimization (TensorRT)
• Symmetric Mode: z = 0, s = ||x||_∞ / (2^{b-1}-1)                 • Zero-Gradient Rounding Barrier (d⌊x⌉/dx = 0)
• Bennett's Noise Model: σ_e^2 = s^2 / 12                         • Straight-Through Estimator (STE)
• SQNR Derivation: 6.02b + 1.76 dB                                 • Learned Step Size Quantization (LSQ)
             │                                                                 │
             └────────────────────────────────┬────────────────────────────────┘
                                              │
                                              ▼
                         [03] Advanced Quantization (LLMs)
                      • 2nd-Order Taylor Curvature: ΔL ≈ 1/2 Δw^T H Δw
                      • Optimal Brain Surgeon (OBS) & GPTQ Inverse Hessian Updates
                      • Activation-Aware Weight Quantization (AWQ) Top 1% Salience
                      • SmoothQuant Matrix S: Migrating Outlier Dynamic Range
                                              │
             ┌────────────────────────────────┴────────────────────────────────┐
             ▼                                                                 ▼
      [04] Pruning & Lottery Tickets                                    [05] Knowledge Distillation
• Unstructured vs Structured Sparsity                              • Teacher-Student Paradigm (Hinton et al. 2015)
• NVIDIA Ampere 2:4 Semi-Structured Sparsity                       • Dark Knowledge & Soft Targets
• Iterative Magnitude Pruning (IMP) Schedule                       • Temperature-Scaled Softmax p_i(T)
• The Lottery Ticket Hypothesis (Frankle & Carbin)                 • High-Temperature Limit Logit Matching Proof
• Winning Tickets m ⊙ θ_0 & Weight Rewinding                      • Composite Objective L = (1-α)L_CE + α T^2 L_KD
```

---

## Mathematical Pillars & Core Derivations

### 1. Uniform Quantization & The 6.02 dB Rule
Uniform affine (asymmetric) quantization maps continuous values $x \in [x_{\min}, x_{\max}]$ into integer code $q \in [q_{\min}, q_{\max}]$:
$$s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}, \quad z = \text{clip}\left( \left\lfloor -\frac{x_{\min}}{s} \right\rceil + q_{\min}, \; q_{\min}, \; q_{\max} \right)$$
$$q = \text{clip}\left( \left\lfloor \frac{x}{s} \right\rceil + z, \; q_{\min}, \; q_{\max} \right), \quad \hat{x} = s(q - z)$$

Under Bennett's uniform noise assumption $e \sim U\left(-\frac{s}{2}, \frac{s}{2}\right)$, noise variance is:
$$\sigma_e^2 = \int_{-s/2}^{s/2} e^2 \frac{1}{s} de = \frac{s^2}{12}$$

For a full-scale sinusoid with amplitude $A = 2^{b-1} s$:
$$\text{SQNR} = 10 \log_{10}\left( \frac{A^2 / 2}{s^2 / 12} \right) = 10 \log_{10}(1.5 \cdot 2^{2b}) \approx \mathbf{6.02 \cdot b + 1.76 \text{ dB}}$$
Every additional bit reduces quantization error power by $4\times$ ($+6.02\text{ dB}$).

### 2. The Rounding Barrier & The Straight-Through Estimator (STE)
Because the derivative of rounding is zero almost everywhere ($\frac{d\lfloor u \rceil}{du} = 0 \text{ a.e.}$), standard backpropagation halts.
The **Straight-Through Estimator (STE, Bengio 2013)** approximates the derivative as identity within clipping bounds:
$$\frac{\partial \hat{x}_{\text{fake}}}{\partial x} = \begin{cases} 1, & q_{\min} s \le x \le q_{\max} s \\ 0, & \text{otherwise} \end{cases}$$

### 3. Second-Order Curvature & GPTQ Compensation
Expanding loss around pre-trained weights ($\nabla_{\mathbf{w}} \mathcal{L} \approx \mathbf{0}$):
$$\Delta \mathcal{L} \approx \frac{1}{2} (\mathbf{w} - \hat{\mathbf{w}})^\top \mathbf{H} (\mathbf{w} - \hat{\mathbf{w}}), \quad \mathbf{H} = 2 \mathbf{X}^\top \mathbf{X}$$

Under the Optimal Brain Quantizer formulation, quantizing weight $w_q \to \hat{w}_q$ causes optimal compensation update:
$$\Delta \mathbf{w} = -\frac{w_q - \hat{w}_q}{[\mathbf{H}^{-1}]_{qq}} \mathbf{H}^{-1}_{:, q}$$

### 4. SmoothQuant Outlier Migration
Large models exhibit persistent activation outlier channels. SmoothQuant mathematically factors diagonal scaling matrix $\mathbf{S} = \text{diag}(s_j)$:
$$\mathbf{Y} = \mathbf{X}\mathbf{W} = (\mathbf{X} \mathbf{S}^{-1}) (\mathbf{S}\mathbf{W})$$
$$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1 - \alpha}}$$
With $\alpha = 0.5$, dynamic range is shared equally, unlocking fast W8A8 matrix multiplications on integer Tensor Cores.

### 5. High-Temperature Knowledge Distillation Equivalence
With temperature-scaled softmax $p_i(T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$, as $T \to \infty$:
$$\frac{\partial \mathcal{L}_{\text{KD}}}{\partial z_i^s} = \frac{1}{T} (p_i^s(T) - p_i^t(T)) \approx \frac{1}{C T^2} (z_i^s - z_i^t)$$
Matching soft probabilities at high temperatures is mathematically identical to minimizing the Mean Squared Error between raw unnormalized logits!

---

## Global Symbol Decoder

| Symbol | Name | Formal Type | Meaning / Role |
| :--- | :--- | :--- | :--- |
| $s \in \mathbb{R}^+$ | Scale Factor | Real scalar | Quantization step size $\Delta$ represented by 1 integer increment. |
| $z \in \mathbb{Z}$ | Zero-Point | Integer | Integer code representing real continuous zero ($x = 0.0$). |
| $b$ | Bit-width | Integer $\in \{2, 4, 8, 16\}$ | Bit precision allocated per parameter or activation entry. |
| $\mathbf{H} \in \mathbb{R}^{K \times K}$ | Hessian Matrix | Symmetric PSD matrix | Second-order loss curvature with respect to weight errors ($2 \mathbf{X}^\top \mathbf{X}$). |
| $\mathbf{S}$ | Smoothing Matrix | Diagonal matrix | Channel-wise transform migrating outlier scale from activations to weights. |
| $\mathbf{m} \in \{0, 1\}^D$ | Pruning Mask | Binary vector | Sparsity pattern ($m_i = 1$ active, $m_i = 0$ pruned). |
| $\theta_0$ | Initial Weights | Parameter vector | Master weight state at initialization ($t = 0$), rewound in Lottery Tickets. |
| $T$ | Softmax Temperature | Scalar $\ge 1.0$ | Distillation softening parameter exposing dark knowledge. |
| $\alpha$ | Distillation Weight | Scalar $\in [0, 1]$ | Tradeoff parameter balancing hard labels vs teacher soft targets. |

---

## Phase 29 Curriculum & Topic Directory

| Topic | Primary Focus | Lecture | Exercises | Experiment |
| :--- | :--- | :--- | :--- | :--- |
| **01. Uniform Quantization Mechanics** | Affine/Symmetric, Zero Preservation, SQNR 6.02b Rule | [HTML](01_uniform_quantization_mechanics/lecture.html) &bull; [MD](01_uniform_quantization_mechanics/lecture.md) | [HTML](01_uniform_quantization_mechanics/exercises_and_solutions.html) &bull; [MD](01_uniform_quantization_mechanics/exercises_and_solutions.md) | [Python](01_uniform_quantization_mechanics/experiment.py) |
| **02. PTQ vs. QAT** | MinMax/KL Calibration, STE, LSQ, Rounding Barrier | [HTML](02_post_training_quantization_vs_qat/lecture.html) &bull; [MD](02_post_training_quantization_vs_qat/lecture.md) | [HTML](02_post_training_quantization_vs_qat/exercises_and_solutions.html) &bull; [MD](02_post_training_quantization_vs_qat/exercises_and_solutions.md) | [Python](02_post_training_quantization_vs_qat/experiment.py) |
| **03. Advanced Quantization** | GPTQ, Cholesky, AWQ Salience, SmoothQuant Outliers | [HTML](03_advanced_quantization_gptq_awq_smoothquant/lecture.html) &bull; [MD](03_advanced_quantization_gptq_awq_smoothquant/lecture.md) | [HTML](03_advanced_quantization_gptq_awq_smoothquant/exercises_and_solutions.html) &bull; [MD](03_advanced_quantization_gptq_awq_smoothquant/exercises_and_solutions.md) | [Python](03_advanced_quantization_gptq_awq_smoothquant/experiment.py) |
| **04. Pruning & Lottery Tickets** | Structured vs Unstructured, 2:4 Sparsity, IMP, Rewinding | [HTML](04_pruning_and_lottery_ticket_hypothesis/lecture.html) &bull; [MD](04_pruning_and_lottery_ticket_hypothesis/lecture.md) | [HTML](04_pruning_and_lottery_ticket_hypothesis/exercises_and_solutions.html) &bull; [MD](04_pruning_and_lottery_ticket_hypothesis/exercises_and_solutions.md) | [Python](04_pruning_and_lottery_ticket_hypothesis/experiment.py) |
| **05. Knowledge Distillation** | Teacher-Student, Dark Knowledge, High-T Equivalence, $T^2$ | [HTML](05_knowledge_distillation/lecture.html) &bull; [MD](05_knowledge_distillation/lecture.md) | [HTML](05_knowledge_distillation/exercises_and_solutions.html) &bull; [MD](05_knowledge_distillation/exercises_and_solutions.md) | [Python](05_knowledge_distillation/experiment.py) |

---

## Standalone Terminal Verification

Every experiment script is built in **100% pure Python 3 standard library** with zero external dependencies (no PyTorch, no NumPy, no SciPy, no scikit-learn). Run all experiments:

```bash
# Topic 01: Uniform Quantization Mechanics & Theoretical SQNR Verification
python3 phase_29_model_compression_and_quantization/01_uniform_quantization_mechanics/experiment.py

# Topic 02: Post-Training Quantization vs QAT with Straight-Through Estimator
python3 phase_29_model_compression_and_quantization/02_post_training_quantization_vs_qat/experiment.py

# Topic 03: Advanced Quantization: SmoothQuant Migration & Mini-GPTQ Compensation
python3 phase_29_model_compression_and_quantization/03_advanced_quantization_gptq_awq_smoothquant/experiment.py

# Topic 04: Pruning & The Lottery Ticket Hypothesis (Winning vs Random Ticket)
python3 phase_29_model_compression_and_quantization/04_pruning_and_lottery_ticket_hypothesis/experiment.py

# Topic 05: Knowledge Distillation (Teacher Dark Knowledge & Temperature Scaling)
python3 phase_29_model_compression_and_quantization/05_knowledge_distillation/experiment.py
```
