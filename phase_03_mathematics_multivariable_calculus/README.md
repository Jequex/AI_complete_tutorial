# Phase 3: Mathematics II — Multivariable Calculus for AI

> *"If linear algebra is the skeleton of an artificial brain, calculus is its living muscle. Calculus is the mathematical engine of learning: it quantifies sensitivity, measures slopes on high-dimensional loss landscapes, and guides gradient descent toward optimal intelligence."*

---

## 🧭 Phase Overview

How does a neural network with billions of parameters actually learn? It computes a loss (a single error number) and uses **multivariable calculus** to figure out how adjusting every single individual parameter will nudge that error down.

Phase 3 provides an intuitive yet mathematically uncompromising journey through the calculus of deep learning:
- **Limits, Continuity, and Derivatives:** From geometric slopes to finite difference approximations.
- **Partial Derivatives & The Gradient:** Why the gradient vector $\nabla f$ always points in the direction of steepest ascent, and how gradient descent rolls down foggy loss valleys.
- **The Chain Rule & Backpropagation:** From univariate composite functions to computational graphs and the reverse-mode automatic differentiation algorithm.
- **Jacobian Matrices:** Differentiating vector-valued functions, computing the Softmax layer Jacobian, and mapping transformations between spaces.
- **Hessian Matrices & Curvature:** Second derivatives, Taylor series approximations, Newton's optimization method, saddle points, and diagnosing ill-conditioned loss ravines.

---

## 📚 Curriculum Structure

| Topic Folder | Subject | Core Concepts | Deliverables |
| :--- | :--- | :--- | :--- |
| **[`01_functions_limits_derivatives/`](./01_functions_limits_derivatives/)** | **Functions, Limits & Derivatives** | Limits, continuity, $\frac{df}{dx}$ limit definition, power/product rules, and numerical finite differences. | `lecture.html`<br>`lecture.md`<br>`exercises_and_solutions.html`<br>`exercises_and_solutions.md`<br>`experiment.py` |
| **[`02_partial_derivatives_and_gradients/`](./02_partial_derivatives_and_gradients/)** | **Partial Derivatives & Gradients** | Multivariable functions $f(x, y)$, partial derivatives, directional derivatives, $\nabla f$, and steepest ascent proof. | `lecture.html`<br>`lecture.md`<br>`exercises_and_solutions.html`<br>`exercises_and_solutions.md`<br>`experiment.py` |
| **[`03_the_chain_rule_and_backpropagation/`](./03_the_chain_rule_and_backpropagation/)** | **Chain Rule & Backpropagation** | Composite functions, computational graphs, forward vs reverse accumulation, and full backprop derivation. | `lecture.html`<br>`lecture.md`<br>`exercises_and_solutions.html`<br>`exercises_and_solutions.md`<br>`experiment.py` |
| **[`04_jacobians_and_vector_calculus/`](./04_jacobians_and_vector_calculus/)** | **Jacobians & Vector Transforms** | Vector-valued functions $f: \mathbb{R}^n \to \mathbb{R}^m$, Jacobian matrix $J$, and full Softmax Jacobian derivation. | `lecture.html`<br>`lecture.md`<br>`exercises_and_solutions.html`<br>`exercises_and_solutions.md`<br>`experiment.py` |
| **[`05_hessians_curvature_and_optimization/`](./05_hessians_curvature_and_optimization/)** | **Hessians, Curvature & Newton's Method** | Second-order partials, Hessian $H$, multivariable Taylor series, positive definiteness, Newton's method, and saddle points. | `lecture.html`<br>`lecture.md`<br>`exercises_and_solutions.html`<br>`exercises_and_solutions.md`<br>`experiment.py` |

---

## 🎯 Learning Objectives

By completing Phase 3, you will:
1. **Internalize Derivatives Geometrically:** Understand derivatives as instantaneous local linear approximations and rate-of-change sensitivity dials.
2. **Master the Gradient Vector $\nabla f$:** Prove mathematically why moving in the direction of $-\nabla f$ guarantees the greatest local decrease in loss.
3. **Derive Backpropagation:** Trace how the multivariate chain rule propagates error signals backwards through computational graphs with zero redundant computation.
4. **Compute Vector Jacobians:** Derive and implement the Jacobian matrix for modern neural layers (like Softmax and LayerNorm).
5. **Understand Curvature & Loss Landscapes:** Use the Hessian matrix to understand why standard gradient descent struggles in narrow ravines and how second-order curvature informs optimizers like Adam and Newton-Raphson.

---

## 🧪 Running the Code Labs

All experiments in this phase run under standard Python 3 with zero external dependencies:
```bash
# Topic 3.1: Numerical differentiation & derivative verification
python3 phase_03_mathematics_multivariable_calculus/01_functions_limits_derivatives/experiment.py

# Topic 3.2: 2D Loss surface gradient descent & steepest ascent verification
python3 phase_03_mathematics_multivariable_calculus/02_partial_derivatives_and_gradients/experiment.py

# Topic 3.3: Micro-Autograd engine & Reverse-mode computational graph backprop
python3 phase_03_mathematics_multivariable_calculus/03_the_chain_rule_and_backpropagation/experiment.py

# Topic 3.4: Exact Softmax Jacobian matrix & Vector transformation engine
python3 phase_03_mathematics_multivariable_calculus/04_jacobians_and_vector_calculus/experiment.py

# Topic 3.5: Hessian matrix calculator & Newton's 2nd-order optimization solver
python3 phase_03_mathematics_multivariable_calculus/05_hessians_curvature_and_optimization/experiment.py
```
