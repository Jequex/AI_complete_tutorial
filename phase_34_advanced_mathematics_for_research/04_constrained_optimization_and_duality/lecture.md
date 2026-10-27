# Topic 04: Constrained Optimization & Duality

## 1. Introduction & 12-Year-Old Intuition

Imagine you are rolling a heavy marble down the inside of a smooth wooden salad bowl. If there are no obstacles, the marble will naturally roll all the way down to the exact bottom of the bowl. At the bottom, the slope is zero ($\nabla f = \mathbf{0}$). This is **unconstrained optimization**.

Now, imagine placing a stiff piece of cardboard vertically across the bowl, blocking the bottom. The marble rolls down until it hits the cardboard fence and gets stuck.

Why does it stop?
It stops because the gravitational pull pulling it downhill is **exactly balanced** by the cardboard pushing back against it!
- The cardboard only pushes back if the marble is touching it (if the marble is resting far away from the cardboard, the cardboard exerts $0$ force).
- The cardboard can only push *away* from itself; it cannot suck the marble into it ($\lambda \ge 0$).

These physical balancing forces are called **Lagrange Multipliers** ($\lambda$ and $\nu$), and the rules governing where the marble stops are the celebrated **Karush-Kuhn-Tucker (KKT) Conditions**.

In modern AI research, constraints are everywhere:
- **Safety Alignments & RLHF**: Maximizing reward while ensuring KL divergence from the base policy stays below a threshold $\mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \le \epsilon$.
- **Adversarial Robustness**: Perturbing an image to fool a classifier subject to an $\ell_\infty$ norm constraint $\|\boldsymbol{\delta}\|_\infty \le \epsilon$.
- **Hard Attention & Sparsity**: Enforcing simplex constraints $\sum_i p_i = 1$ and $p_i \ge 0$.
- **Support Vector Machines (SVMs)**: Finding maximum-margin decision boundaries where only the support vectors have non-zero Lagrange multipliers.

---

## 2. Complete Mathematical Symbol Decoder

| Symbol | Mathematical Entity | Dimensionality / Domain | Operational & Research Role |
| :--- | :--- | :--- | :--- |
| $\mathbf{x} \in \mathbb{R}^n$ | Primal Decision Variable | $\mathbb{R}^n$ | Neural network weights, adversarial perturbation, or probability distribution. |
| $f_0(\mathbf{x})$ | Primal Objective Function | $\mathbb{R}^n \to \mathbb{R}$ | Loss function to minimize (or negative reward to maximize). |
| $f_i(\mathbf{x}) \le 0$ | Inequality Constraints | $\mathbb{R}^n \to \mathbb{R}$ for $i=1,\dots,m$ | Feasible region boundaries (e.g. safety bounds, parameter budgets). |
| $h_j(\mathbf{x}) = 0$ | Equality Constraints | $\mathbb{R}^n \to \mathbb{R}$ for $j=1,\dots,p$ | Manifold restrictions (e.g. total probability summing to 1). |
| $\lambda_i \ge 0$ | Dual Inequality Multipliers | Non-negative scalar $\mathbb{R}_{\ge 0}$ | "Shadow prices" / normal reaction forces exerted by inequality boundaries. |
| $\nu_j \in \mathbb{R}$ | Dual Equality Multipliers | Real scalar $\mathbb{R}$ | Reaction forces maintaining exact equality constraints. |
| $\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ | Lagrangian Function | $\mathbb{R}^n \times \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$ | $\mathcal{L} = f_0(\mathbf{x}) + \sum_{i=1}^m \lambda_i f_i(\mathbf{x}) + \sum_{j=1}^p \nu_j h_j(\mathbf{x})$. |
| $g(\boldsymbol{\lambda}, \boldsymbol{\nu})$ | Lagrange Dual Function | $\mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$ | $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$; concave lower bound on $p^*$. |
| $p^*$ | Primal Optimal Value | $\mathbb{R} \cup \{\pm \infty\}$ | Minimum constrained objective: $p^* = \inf_{\mathbf{x} \in \mathcal{D}_{\text{feas}}} f_0(\mathbf{x})$. |
| $d^*$ | Dual Optimal Value | $\mathbb{R} \cup \{\pm \infty\}$ | Maximum dual lower bound: $d^* = \sup_{\boldsymbol{\lambda} \ge \mathbf{0}, \boldsymbol{\nu}} g(\boldsymbol{\lambda}, \boldsymbol{\nu})$. |
| $p^* - d^*$ | Duality Gap | $\mathbb{R}_{\ge 0}$ | Non-negative discrepancy; exactly zero when Strong Duality holds. |

---

## 3. The Primal and Dual Problems

### 3.1 Standard Primal Problem
$$\begin{aligned}
\min_{\mathbf{x}} \quad & f_0(\mathbf{x}) \\
\text{subject to} \quad & f_i(\mathbf{x}) \le 0, \quad i = 1, \dots, m \\
& h_j(\mathbf{x}) = 0, \quad j = 1, \dots, p
\end{aligned}$$

### 3.2 The Lagrangian Mechanics
We form the Lagrangian by augmenting the objective with weighted constraint penalties:
$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) \triangleq f_0(\mathbf{x}) + \sum_{i=1}^m \lambda_i f_i(\mathbf{x}) + \sum_{j=1}^p \nu_j h_j(\mathbf{x})$$

The unconstrained minimax formulation reveals the primal problem:
$$\sup_{\boldsymbol{\lambda} \ge \mathbf{0}, \boldsymbol{\nu}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = \begin{cases} f_0(\mathbf{x}) & \text{if } \mathbf{x} \text{ is feasible} \\ +\infty & \text{if } \mathbf{x} \text{ violates any constraint} \end{cases}$$
Hence:
$$p^* = \inf_{\mathbf{x}} \sup_{\boldsymbol{\lambda} \ge \mathbf{0}, \boldsymbol{\nu}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$$

### 3.3 The Dual Problem & Weak Duality
By switching the order of the infimum and supremum, we obtain the **Lagrange Dual Problem**:
$$d^* = \sup_{\boldsymbol{\lambda} \ge \mathbf{0}, \boldsymbol{\nu}} \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = \sup_{\boldsymbol{\lambda} \ge \mathbf{0}, \boldsymbol{\nu}} g(\boldsymbol{\lambda}, \boldsymbol{\nu})$$

```
                   Weak Duality & The Duality Gap
                   
  Primal Objective Values f_0(x) for feasible x:
  ───────●──────────────────────●───────────[ p* ]
         x_1                    x_2           ▲
                                              │  Duality Gap = p* - d* >= 0
                                              ▼  (Zero under Slater's Condition)
  ───────●──────────────────────●───────────[ d* ]
      (λ_1, ν_1)            (λ_2, ν_2)
  Dual Objective Values g(λ, ν) for λ >= 0:
```

**Weak Duality Theorem**: For *any* optimization problem (even non-convex), the dual optimal value is always a lower bound on the primal optimal value:
$$d^* \le p^*$$
**Proof**: For any feasible $\tilde{\mathbf{x}}$ and any dual feasible $(\boldsymbol{\lambda} \ge \mathbf{0}, \boldsymbol{\nu})$:
$$\sum_{i=1}^m \lambda_i f_i(\tilde{\mathbf{x}}) \le 0 \quad \text{and} \quad \sum_{j=1}^p \nu_j h_j(\tilde{\mathbf{x}}) = 0$$
Therefore:
$$g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) \le \mathcal{L}(\tilde{\mathbf{x}}, \boldsymbol{\lambda}, \boldsymbol{\nu}) \le f_0(\tilde{\mathbf{x}})$$
Taking the supremum over dual variables on the left and infimum over feasible $\tilde{\mathbf{x}}$ on the right:
$$d^* \le p^* \quad \blacksquare$$

---

## 4. Slater's Condition & Strong Duality

When does $d^* = p^*$ (zero duality gap)?

### Slater's Condition:
If the primal problem is **convex** (i.e. $f_0, f_1, \dots, f_m$ are convex, and equality constraints are affine $h_j(\mathbf{x}) = \mathbf{a}_j^\top \mathbf{x} - b_j$), and there exists at least one **strictly feasible point** $\mathbf{x}_{\text{relint}}$:
$$f_i(\mathbf{x}_{\text{relint}}) < 0 \quad (i = 1, \dots, m), \quad \mathbf{C}\mathbf{x}_{\text{relint}} = \mathbf{d}$$
Then **Strong Duality** holds:
$$d^* = p^*$$
and the dual optimal values $(\boldsymbol{\lambda}^*, \boldsymbol{\nu}^*)$ are attained.

---

## 5. The Karush-Kuhn-Tucker (KKT) Conditions

If strong duality holds and the functions are differentiable, any pair of primal-dual optimal points $(\mathbf{x}^*, \boldsymbol{\lambda}^*, \boldsymbol{\nu}^*)$ must satisfy the **four KKT conditions**:

```
                       The Four KKT Pillars
                       
1. Stationarity:            ∇ f_0(x*) + Σ λ_i* ∇ f_i(x*) + Σ ν_j* ∇ h_j(x*) = 0
2. Primal Feasibility:      f_i(x*) <= 0  (i=1..m),   h_j(x*) = 0  (j=1..p)
3. Dual Feasibility:        λ_i* >= 0     (i=1..m)
4. Complementary Slackness: λ_i* · f_i(x*) = 0 for all i
```

### 5.1 Deep Geometric Meaning of Complementary Slackness
$$\lambda_i^* f_i(\mathbf{x}^*) = 0 \quad \text{for every } i = 1, \dots, m$$
Because $\lambda_i^* \ge 0$ and $f_i(\mathbf{x}^*) \le 0$, their product can only be zero if:
- **Case 1 (Inactive Constraint)**: $f_i(\mathbf{x}^*) < 0$ (the optimal point lies strictly inside the boundary). Then **$\lambda_i^* = 0$**. The constraint exerts zero force.
- **Case 2 (Active / Binding Constraint)**: $\lambda_i^* > 0$ (the boundary is actively pushing against the objective). Then **$f_i(\mathbf{x}^*) = 0$**. The optimal point sits exactly on the fence.

In Support Vector Machines (SVMs), this explains why only the data points lying on the margin hyperplanes have $\alpha_i > 0$ (the *support vectors*); all correctly classified points far from the margin have $\alpha_i = 0$!

---

## 6. Numerical Solution Methods: Active Set vs. Interior Point

### 6.1 Primal Active Set Method
- Maintains a working set $\mathcal{W}$ of currently active constraints treated as equalities.
- Solves a clean linear equality-constrained subproblem at each step.
- Converges in a finite number of iterations to the exact global KKT point with machine-precision residuals ($10^{-16}$).

### 6.2 Primal-Dual Interior Point Method (IPM)
- Replaces hard complementary slackness $\lambda_i f_i = 0$ with a smooth perturbed central path:
  $$\lambda_i s_i = \mu, \quad s_i > 0, \quad \lambda_i > 0$$
- Solves the Newton step on the combined primal-dual KKT system.
- Gradually drives the barrier parameter $\mu \to 0$, approaching the optimal boundary from the interior.
