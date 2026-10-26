# Phase 05: Statistics & Estimation for Artificial Intelligence

---

## 🧭 Executive Overview

If **Probability** (Phase 04) is the forward machine that asks:
> *"Given known generation parameters $\theta$, what data $D$ will be produced?"*

Then **Statistics & Estimation** (Phase 05) is the reverse detective that asks:
> *"Given observed data $D$, what hidden world parameters $\theta$ produced it?"*

This inverse problem is the fundamental definition of machine learning. Training an AI model—whether a tiny linear regressor or a 405-billion parameter transformer—is fundamentally an exercise in **statistical parameter estimation**.

---

## 🗺️ Phase Syllabus & Topic Breakdown

```text
phase_05_statistics_and_estimation/
├── index.html                                       # Visual Phase Hub
├── README.md                                        # Master Syllabus & Architecture Guide
│
├── 01_populations_samples_and_estimators/
│   ├── lecture.html / lecture.md                    # Populations, sample statistics, bias-variance, Bessel's correction
│   ├── exercises_and_solutions.html / .md           # Proof of E[S^2] = ((n-1)/n)sigma^2 and SE calculation
│   └── experiment.py                                # Empirical demonstration of Bessel's correction & SE scaling
│
├── 02_confidence_intervals_and_hypothesis_testing/
│   ├── lecture.html / lecture.md                    # Confidence intervals, z/t-tests, p-values, A/B testing AI models
│   ├── exercises_and_solutions.html / .md           # Computing 95% CIs, two-sample t-test, Model A vs B win rate
│   └── experiment.py                                # A/B testing simulator & p-value distribution under null vs alt
│
├── 03_maximum_likelihood_estimation_mle/
│   ├── lecture.html / lecture.md                    # Likelihood vs probability, log-likelihood, Gaussian->MSE, Bernoulli->CE
│   ├── exercises_and_solutions.html / .md           # Step-by-step MLE derivation for Exponential & Poisson distributions
│   └── experiment.py                                # Pure Python numerical MLE solver vs analytical formulas
│
├── 04_map_estimation_and_regularization/
│   ├── lecture.html / lecture.md                    # Prior beliefs, MAP formula, Gaussian prior->L2, Laplace prior->L1
│   ├── exercises_and_solutions.html / .md           # Step-by-step derivation of Ridge & Lasso from Bayesian priors
│   └── experiment.py                                # Overfitting mitigation: MLE vs MAP with L2 & L1 regularization
│
└── 05_bayesian_inference_and_conjugate_priors/
    ├── lecture.html / lecture.md                    # Full posterior distributions, Beta-Binomial conjugate update
    ├── exercises_and_solutions.html / .md           # Analytical Beta posterior derivation & sequential Bayes updates
    └── experiment.py                                # Streaming Bayesian learning simulator with dynamic posterior updates
```

---

## 🔑 Core Mathematical Pillars Mastered in Phase 5

1. **Unbiased Estimators & Bessel's Correction:**
   Understanding why dividing by $n$ underestimates population variance:
   $$\mathbb{E}[S^2_{\text{biased}}] = \frac{n-1}{n}\sigma^2 \implies S^2_{\text{unbiased}} = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$$

2. **Hypothesis Testing & AI Evaluation:**
   Rigorous decision theory for model evaluation: Type I error ($\alpha$), Type II error ($\beta$), statistical power ($1-\beta$), and two-tailed $z/t$-tests for benchmark leaderboard comparisons.

3. **Maximum Likelihood Estimation (MLE):**
   The universal optimization objective of deep learning:
   $$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^n \ln P(x_i | \theta)$$
   - Under Gaussian noise $\implies$ Minimizing **Mean Squared Error (MSE)**.
   - Under Categorical distribution $\implies$ Minimizing **Cross-Entropy (Log Loss)**.

4. **Maximum A Posteriori (MAP) & Regularization:**
   Incorporating prior weight beliefs into objective functions:
   $$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[ \sum_{i=1}^n \ln P(x_i | \theta) + \ln P(\theta) \right]$$
   - Gaussian Prior $\mathcal{N}(0, \sigma_0^2) \implies$ **L2 Regularization (Weight Decay / Ridge)**.
   - Laplace Prior $\text{Laplace}(0, b) \implies$ **L1 Regularization (Lasso / Sparsity)**.

5. **Bayesian Conjugate Inference:**
   Exact sequential probability updating where the posterior belongs to the same family as the prior:
   $$\text{Prior } \text{Beta}(\alpha, \beta) + k \text{ successes in } n \text{ trials} \implies \text{Posterior } \text{Beta}(\alpha + k, \beta + n - k)$$

---

## ⚡ Quickstart: Executing Phase 5 Labs

All experiments are written in pure Python 3 without external dependencies:

```bash
# Topic 5.1: Sample Statistics & Bessel's Correction
python3 phase_05_statistics_and_estimation/01_populations_samples_and_estimators/experiment.py

# Topic 5.2: Confidence Intervals & A/B Testing
python3 phase_05_statistics_and_estimation/02_confidence_intervals_and_hypothesis_testing/experiment.py

# Topic 5.3: Maximum Likelihood Estimation (MLE)
python3 phase_05_statistics_and_estimation/03_maximum_likelihood_estimation_mle/experiment.py

# Topic 5.4: MAP Estimation & Regularization (L1 / L2)
python3 phase_05_statistics_and_estimation/04_map_estimation_and_regularization/experiment.py

# Topic 5.5: Bayesian Streaming Inference & Conjugate Priors
python3 phase_05_statistics_and_estimation/05_bayesian_inference_and_conjugate_priors/experiment.py
```

