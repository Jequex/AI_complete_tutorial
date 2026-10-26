# Phase 10: ML Engineering & Evaluation

---

## 🧭 Executive Overview

A machine learning model is not useful merely because it achieves low loss inside an isolated Jupyter notebook. 

While earlier phases explored the mathematical foundations of learning—**Linear Algebra** (Phase 02), **Multivariable Calculus** (Phase 03), **Probability** (Phase 04), **Statistics** (Phase 05), and **Optimization** (Phase 07)—**Phase 10: ML Engineering & Evaluation** bridges the chasm between mathematical theory and production-grade software engineering.

Over 80% of machine learning projects fail when deployed to live traffic. Those failures rarely stem from model architectures; they are caused by:
1. **Subtle Data Leakage:** Preprocessing, feature selection, or target-proxies leaking future information into training.
2. **Ill-conditioned Feature Scaling:** Gradients bouncing off steep elliptical loss canyons.
3. **Misleading Evaluation Metrics:** Relying on accuracy in severe class imbalance, misinterpreting ROC-AUC, or deploying uncalibrated probabilities.
4. **Production Degradation:** Training-serving skew, insecure serialization (`pickle` RCE vulnerabilities), and unmonitored covariate shift / concept drift.

Phase 10 establishes the operational protocols, mathematical proofs, and engineering pipelines required to build reliable, high-performance production AI systems.

---

## 🗺️ Phase Syllabus & Topic Breakdown

```text
phase_10_ml_engineering_and_evaluation/
├── index.html                                       # Phase Portal Hub & Quickstart
├── README.md                                        # Master Syllabus & Architecture Guide
│
├── 01_data_leakage_and_splitting/
│   ├── lecture.html / lecture.md                    # Train/Val/Test partitioning, target leakage, Freedman's paradox proof, Stratified K-Fold
│   ├── exercises_and_solutions.html / .md           # Telecom churn case study, proof of mean leakage covariance, hypergeometric fold analysis
│   └── experiment.py                                # Pure Python 3 simulation proving 95% false validation vs 50% test collapse
│
├── 02_feature_engineering_and_scaling/
│   ├── lecture.html / lecture.md                    # Z-score vs MinMax vs Robust IQR, Hessian condition numbers, OHE, target encoding
│   ├── exercises_and_solutions.html / .md           # Dummy variable trap singularity proof, outlier breakdown point, empirical Bayes smoothing
│   └── experiment.py                                # Pure Python Scalers, OneHotEncoder, GD convergence speedup on conditioned surfaces
│
├── 03_evaluation_metrics_and_calibration/
│   ├── lecture.html / lecture.md                    # Confusion matrix, Precision/Recall, F1 harmonic mean proof, MCC, ROC vs PR, Platt scaling
│   ├── exercises_and_solutions.html / .md           # Wilcoxon-Mann-Whitney rank proof of ROC-AUC, F_beta derivation, Murphy Brier decomposition
│   └── experiment.py                                # Confusion matrix, exact trapezoidal ROC/PR AUC, Brier score, and Platt calibrator from scratch
│
└── 04_ml_pipelines_and_production/
    ├── lecture.html / lecture.md                    # ML Pipelines, training-serving skew, pickle arbitrary code execution vs Safetensors, PSI drift
    ├── exercises_and_solutions.html / .md           # Algebraic proof of PSI symmetric KL divergence, importance weighting under covariate shift
    └── experiment.py                                # Pure Python Pipeline (Imputer + Scaler + Model), safe JSON export, PSI drift detector
```

---

## 🔑 Core Mathematical & Engineering Pillars Mastered in Phase 10

1. **The Split-First Doctrine & Freedman's Paradox:**
   Proving that performing feature selection or scaling prior to dataset partitioning violates sample independence:
   $$\text{Cov}(z_i, x_k) = -\frac{\sigma}{N} \ne 0 \quad (\text{for } i \in \text{train}, k \in \text{test})$$
   Demonstrating how pre-split selection on $D \gg N$ pure Gaussian noise features fabricates 95% validation accuracy out of pure random chance.

2. **Hessian Conditioning & Gradient Descent Acceleration:**
   Connecting feature scaling to the condition number $\kappa(H) = \lambda_{\max}/\lambda_{\min}$ of the Hessian $H = \frac{1}{N}X^T X$. Proving that the maximum stable learning rate is bounded by $\eta < \frac{2}{\lambda_{\max}}$, and showing that standardization compresses the condition number to $\kappa \approx 1$, accelerating convergence by orders of magnitude.

3. **Metric Mathematics: Harmonic Penalties & Rank Equivalences:**
   Proving that the $F_1$-score is the harmonic mean of Precision and Recall, which penalizes asymmetric performance much more severely than the arithmetic mean via the AM-HM inequality $\mathcal{H} \le \mathcal{A}$. Proving the Wilcoxon-Mann-Whitney equivalence to ROC-AUC:
   $$\text{AUC-ROC} = P\left(s(X^+) > s(X^-)\right) = \frac{1}{N_+ N_-} \sum_{i=1}^{N_+} \sum_{j=1}^{N_-} \mathbb{I}(s_i^+ > s_j^-)$$

4. **Probability Calibration & Scoring Rules:**
   Evaluating forecast reliability with the strictly proper Brier score $\text{BS} = \frac{1}{N}\sum (\hat{p}_i - y_i)^2$. Decomposing BS into Reliability, Resolution, and Uncertainty. Applying Platt scaling:
   $$P(Y=1 \mid s) = \frac{1}{1 + \exp(A s + B)}$$
   to calibrate raw uncalibrated margins while preserving rank order.

5. **Production Pipelines, MLOps & Covariate Shift Detection:**
   Encapsulating end-to-end imputation, scaling, and estimation into unified pipelines to eliminate training-serving skew. Establishing safe serialization protocols (JSON, Safetensors, ONNX) to eliminate `pickle` arbitrary code execution exploits. Continuous telemetry monitoring using the Population Stability Index:
   $$\text{PSI} = \sum_{b=1}^B (P_b - Q_b) \ln\left(\frac{P_b}{Q_b}\right) = D_{\text{KL}}(P \parallel Q) + D_{\text{KL}}(Q \parallel P)$$

---

## ⚡ Quickstart: Executing Phase 10 Labs

All experiments are written in pure Python 3 using only the standard library (zero external dependencies):

```bash
# Topic 10.1: Data Leakage Demonstration & Freedman's Paradox (95% fake CV vs 50% test collapse)
python3 phase_10_ml_engineering_and_evaluation/01_data_leakage_and_splitting/experiment.py

# Topic 10.2: Feature Scaling & Encoding Impact on Gradient Descent Conditioning
python3 phase_10_ml_engineering_and_evaluation/02_feature_engineering_and_scaling/experiment.py

# Topic 10.3: Comprehensive Evaluation Metrics, Trapezoidal ROC/PR-AUC & Platt Calibration
python3 phase_10_ml_engineering_and_evaluation/03_evaluation_metrics_and_calibration/experiment.py

# Topic 10.4: Production ML Pipelines, Safe JSON Persistence & Covariate Shift PSI Monitoring
python3 phase_10_ml_engineering_and_evaluation/04_ml_pipelines_and_production/experiment.py
```
