# Phase 08: Classical Machine Learning for Artificial Intelligence

---

## 🧭 Executive Overview

If **Linear Algebra** (Phase 02) gives us the spaces and projections, **Multivariable Calculus** (Phase 03) gives us the gradients, **Probability & Statistics** (Phases 04 & 05) give us the likelihoods and estimation frameworks, and **Optimization** (Phase 07) gives us the search engines...

Then **Classical Machine Learning** (Phase 08) is where all these disciplines assemble into **practical predictive systems**.

Phase 08 establishes the mathematical foundations and algorithmic architectures of supervised learning. We cover the entire classical spectrum:
- Analytical closed-form solutions (The Normal Equation) vs. numerical optimization.
- Modeling continuous outputs via linear regression vs. probabilistic binary events via logistic regression.
- Non-linear partitions via recursive Decision Trees and their variance-reduced Ensembles (Random Forests and Gradient Boosting).
- Geometric margin maximization and infinite-dimensional Mercer Hilbert spaces via Support Vector Machines.
- Probabilistic generative text classifiers (Naive Bayes) and non-parametric lazy learners (K-Nearest Neighbors).

Every topic includes an intuitive 12-year-old analogy, a rigorous mathematical symbol decoder, first-principles step-by-step calculus proofs, interactive HTML/KaTeX lectures, collapsible exercises and solutions, and pure Python 3 implementations with zero third-party dependencies.

---

## 🗺️ Phase Syllabus & Topic Breakdown

```text
phase_08_classical_machine_learning/
├── index.html                                       # Phase Portal Hub & Quickstart
├── README.md                                        # Master Syllabus & Architecture Guide
│
├── 01_linear_regression/
│   ├── lecture.html / lecture.md                    # Normal Equation closed-form derivation w = (X^T X)^(-1) X^T y, Gradient Descent, MSE
│   ├── exercises_and_solutions.html / .md           # Manual 2x2 matrix inversion, residual orthogonality proof, Big-O FLOPs scaling
│   └── experiment.py                                # Pure Python 3 solver: Gauss-Jordan inversion vs GD, orthogonality check
│
├── 02_logistic_regression/
│   ├── lecture.html / lecture.md                    # Sigmoid activation, log-odds (logit), MLE, Binary Cross-Entropy loss, Hessian convexity
│   ├── exercises_and_solutions.html / .md           # Logit calculations, BCE per-sample penalty, manual gradient step, decision boundary
│   └── experiment.py                                # Pure Python 3 binary classifier: stable sigmoid, BCE loss, gradient descent, 100% test acc
│
├── 03_decision_trees_and_ensembles/
│   ├── lecture.html / lecture.md                    # Gini impurity, Shannon entropy, Information Gain, CART, Bagging, Random Forests, GBDT
│   ├── exercises_and_solutions.html / .md           # Step-by-step Gini/Entropy hand math, ensemble variance reduction law, GBDT residuals
│   └── experiment.py                                # Pure Python 3 CART decision tree & Random Forest ensemble on non-linear XOR data
│
├── 04_support_vector_machines/
│   ├── lecture.html / lecture.md                    # Maximum margin 2/||w||, Hard vs Soft margin, Lagrange dual, KKT conditions, RBF kernel trick
│   ├── exercises_and_solutions.html / .md           # Geometric margin hand calculations, KKT support vector identification, Hinge subgradient
│   └── experiment.py                                # Pure Python 3 Linear SVM via Hinge subgradient descent & RBF Kernel Evaluator
│
└── 05_naive_bayes_and_knn/
    ├── lecture.html / lecture.md                    # Bayes' rule, conditional independence, Laplace smoothing, lazy KNN, curse of dimensionality
    ├── exercises_and_solutions.html / .md           # Multinomial spam calculation with smoothing, KNN distance voting, hypercube volume proof
    └── experiment.py                                # Pure Python 3 Naive Bayes spam filter & multi-metric KNN classifier (100% test acc)
```

---

## 🔑 Core Mathematical Pillars Mastered in Phase 8

1. **Closed-Form Normal Equation & Orthogonal Projection:**
   Proving that the Mean Squared Error gradient $\nabla_w J(w) = \frac{1}{N}(X^T X w - X^T y) = \mathbf{0}$ yields:
   $$w^* = (X^T X)^{-1} X^T y$$
   Proving geometrically that the residual vector $e = y - Xw^*$ is strictly orthogonal to the column space of $X$ ($X^T e = \mathbf{0}$).

2. **The Logistic Miracle & Binary Cross-Entropy:**
   Deriving the gradient of the Negative Log-Likelihood for Bernoulli trials under sigmoid activation $\sigma(z) = \frac{1}{1 + e^{-z}}$:
   $$\nabla_w \mathcal{L}_{\text{BCE}} = \frac{1}{N} X^T (\hat{y} - y)$$
   Proving that the BCE loss surface is strictly convex everywhere with positive semi-definite Hessian $H = \frac{1}{N} X^T D X \succeq 0$.

3. **Information Theory & Ensemble Variance Decorrelation:**
   Evaluating split purity via Gini Impurity $G(T) = 1 - \sum p_i^2$ and Shannon Entropy $H(T) = -\sum p_i \log_2 p_i$. Proving the fundamental ensemble variance reduction law:
   $$\text{Var}(\text{Ensemble}) = \rho \sigma^2 + \frac{1 - \rho}{M} \sigma^2$$
   and understanding how Random Forest feature subsampling ($k = \lfloor\sqrt{d}\rfloor$) drives tree correlation $\rho \to 0$.

4. **Maximum Margin Hyperplanes & The Mercer Kernel Trick:**
   Maximizing the geometric margin $\frac{2}{\|w\|_2}$ via quadratic programming with KKT complementary slackness $\alpha_i [y_i(w^T x_i + b) - 1] = 0$. Using Mercer kernels $K(x, x') = \exp(-\gamma \|x - x'\|^2)$ to compute dot products implicitly in infinite-dimensional Hilbert spaces.

5. **Generative Classifiers & The Geometry of High Dimensions:**
   Applying Bayes' rule with conditional independence $P(x \mid y) = \prod P(x_j \mid y)$ in log-space with Laplace smoothing $\hat{\theta}_{jc} = \frac{N_{jc} + 1}{N_c + |V|}$. Proving mathematically why the Curse of Dimensionality ($e_d = r^{1/d}$) forces high-dimensional neighborhoods to span the entire volume of space.

---

## ⚡ Quickstart: Executing Phase 8 Labs

All experiments run in standard Python 3 using only the standard library (no `pip install` required):

```bash
# Topic 8.1: Linear Regression (Gauss-Jordan Inversion vs. Gradient Descent)
python3 phase_08_classical_machine_learning/01_linear_regression/experiment.py

# Topic 8.2: Logistic Regression (BCE Loss, Numerically Stable Sigmoid, GD)
python3 phase_08_classical_machine_learning/02_logistic_regression/experiment.py

# Topic 8.3: Decision Trees & Random Forest Ensemble on Non-Linear XOR
python3 phase_08_classical_machine_learning/03_decision_trees_and_ensembles/experiment.py

# Topic 8.4: Linear Support Vector Machine & RBF Kernel Evaluator
python3 phase_08_classical_machine_learning/04_support_vector_machines/experiment.py

# Topic 8.5: Naive Bayes Spam Filter & K-Nearest Neighbors Classifier
python3 phase_08_classical_machine_learning/05_naive_bayes_and_knn/experiment.py
```
