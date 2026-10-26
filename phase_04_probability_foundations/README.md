# Phase 4: Probability Foundations for AI

> *"Probability is the mathematical logic of uncertainty. Machine learning models never deal in absolute certainties; they observe noisy evidence, quantify beliefs, and make decisions under incomplete information."*

---

## 🧭 Phase Overview

Artificial intelligence is fundamentally probabilistic. Whether a language model is predicting the next token, a vision model is classifying an ambiguous medical scan, or a diffusion model is denoising a photo from pure static, every prediction is a probability distribution over possible outcomes.

Phase 4 builds the rigorous mathematical foundations of probability theory from first principles. We bridge intuitive analogies with uncompromising mathematical formalisms: sample spaces, Kolmogorov axioms, Bayes' theorem, discrete and continuous random variables, probability density functions, multivariate Gaussians, expectation, covariance, the Central Limit Theorem, and Monte Carlo estimation.

---

## 📚 Curriculum Structure

| Topic Folder | Subject | Core Concepts | Deliverables |
| :--- | :--- | :--- | :--- |
| **[`01_probability_fundamentals_and_bayes/`](./01_probability_fundamentals_and_bayes/)** | **Fundamentals & Bayes' Theorem** | Sample spaces ($\Omega$), events, Kolmogorov axioms, conditional probability, independence, Bayes' rule ($P(\theta\|D)$), and the False Positive Paradox. | `lecture.html`<br>`lecture.md`<br>`exercises_and_solutions.html`<br>`exercises_and_solutions.md`<br>`experiment.py` |
| **[`02_random_variables_pmf_pdf_cdf/`](./02_random_variables_pmf_pdf_cdf/)** | **Random Variables, PMF, PDF & CDF** | Discrete vs continuous variables, PMFs, PDFs (why $P(X=x)=0$ yet density $>0$), Cumulative Distribution Functions (CDF), marginalization, and conditioning. | `lecture.html`<br>`lecture.md`<br>`exercises_and_solutions.html`<br>`exercises_and_solutions.md`<br>`experiment.py` |
| **[`03_core_distributions_in_ai/`](./03_core_distributions_in_ai/)** | **Core Probability Distributions** | Bernoulli, Binomial, Categorical/Multinomial (LLM tokens), Gaussian $\mathcal{N}(\mu, \sigma^2)$, and Multivariate Gaussian $\mathcal{N}(\mu, \Sigma)$ in latent spaces (VAEs & Diffusion). | `lecture.html`<br>`lecture.md`<br>`exercises_and_solutions.html`<br>`exercises_and_solutions.md`<br>`experiment.py` |
| **[`04_expectation_variance_covariance/`](./04_expectation_variance_covariance/)** | **Expectation, Variance & Covariance** | Linearity of expectation ($\mathbb{E}[aX+bY]$), variance $\text{Var}(X) = \mathbb{E}[X^2] - \mu^2$, covariance $\text{Cov}(X, Y)$, Pearson correlation $\rho \in [-1, 1]$, and covariance matrices. | `lecture.html`<br>`lecture.md`<br>`exercises_and_solutions.html`<br>`exercises_and_solutions.md`<br>`experiment.py` |
| **[`05_limit_theorems_and_monte_carlo/`](./05_limit_theorems_and_monte_carlo/)** | **Limit Theorems & Monte Carlo** | Law of Large Numbers (LLN), Central Limit Theorem (CLT), Monte Carlo integration ($O(1/\sqrt{N})$), and probabilistic token sampling (Temperature, Top-K, Top-P Nucleus). | `lecture.html`<br>`lecture.md`<br>`exercises_and_solutions.html`<br>`exercises_and_solutions.md`<br>`experiment.py` |

---

## 🎯 Learning Objectives

By completing Phase 4, you will:
1. **Master Bayesian Reasoning:** Formulate prior beliefs, update them through likelihoods, and understand how machine learning models update parameter estimates upon observing data.
2. **Understand Continuous Densities:** Explain why the probability of a continuous variable hitting an exact single number is zero, and how probability densities integrate over intervals.
3. **Internalize Multivariate Gaussians:** Understand how the covariance matrix $\Sigma$ shapes high-dimensional latent representations in variational autoencoders (VAEs) and generative diffusion models.
4. **Leverage Linearity of Expectation:** Prove and apply expectation rules to derive loss functions and gradient estimators without relying on independence assumptions.
5. **Implement Monte Carlo Engines:** Approximate intractable high-dimensional expectations using stochastic sampling algorithms in pure Python.

---

## 🧪 Running the Code Labs

All experiments in this phase run under standard Python 3 with zero external dependencies:
```bash
# Topic 4.1: Bayes' theorem simulator & medical test paradox
python3 phase_04_probability_foundations/01_probability_fundamentals_and_bayes/experiment.py

# Topic 4.2: PMF, PDF & Inverse CDF sampling engine
python3 phase_04_probability_foundations/02_random_variables_pmf_pdf_cdf/experiment.py

# Topic 4.3: Core distributions & Multivariate Gaussian generator
python3 phase_04_probability_foundations/03_core_distributions_in_ai/experiment.py

# Topic 4.4: Expectation, variance & covariance matrix calculator
python3 phase_04_probability_foundations/04_expectation_variance_covariance/experiment.py

# Topic 4.5: Central Limit Theorem & LLM sampling (Temperature, Top-K, Top-P)
python3 phase_04_probability_foundations/05_limit_theorems_and_monte_carlo/experiment.py
```

