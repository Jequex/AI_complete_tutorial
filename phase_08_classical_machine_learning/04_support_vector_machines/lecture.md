# Topic 8.4: Support Vector Machines & The Kernel Trick

---

## 1. Executive Summary & The 12-Year-Old Analogy

Logistic regression finds *a* hyperplane that separates classes by maximizing likelihood. But in linearly separable datasets, an infinite number of hyperplanes can achieve 100% training accuracy. Which one is best?

**Support Vector Machines (SVMs)** provide a geometrically motivated answer: find the unique decision hyperplane that maximizes the **margin**—the physical Euclidean distance between the hyperplane and the closest training points of either class.

Furthermore, through the mathematical elegance of **Lagrangian Duality** and the **Kernel Trick**, SVMs generalize effortlessly to non-linear classification by implicitly mapping data into infinite-dimensional spaces without ever computing coordinates directly.

```
       Feature x2
           ^
           |       Class +1      [ Margin Width = 2 / ||w|| ]
           |        *      *          /
           |          *  (SV) *      /
       ----+------------*-----------+---- w^T x + b = +1 (Positive Margin Boundary)
           |             \         /
           |   Separating \       /
           |   Hyperplane  \     /
       ----+----------------\---+-------- w^T x + b = 0  (Decision Boundary)
           |                 \ /
           |                  v
       ----+------------------*---------- w^T x + b = -1 (Negative Margin Boundary)
           |             (SV)
           |        o       o
           |          o        Class -1
           +-----------------------------------> Feature x1
```

---

### The 12-Year-Old Analogy: The Widest Street Between Two Rival Neighborhoods

Imagine two rival factions in a city: The Red Wolves and The Blue Bears.
You are the master city planner, and your job is to build a straight street separating the two neighborhoods so that neither side gets into fights.

1. **Any Dividing Line vs. The Widest Street:**  
   You could draw lots of thin fences between them. But if you draw a fence right against a Red Wolf's front porch, a dog barking or a kicked soccer ball will spill over into the Bears' territory immediately (this is **fragile, high-variance overfitting**).  
   Instead, you want to pave the **widest possible boulevard** between them. The centerline of the street is the decision boundary ($w^T x + b = 0$), and the two curbs are the margin boundaries ($+1$ and $-1$).

2. **The Support Vectors (The Only Houses That Matter):**  
   Who determines how wide your boulevard can be? Only the very front-row houses that sit directly on the curb! These critical front-line houses are the **Support Vectors**. If you demolish, repaint, or move every house located five blocks further back, the boulevard doesn't move a single inch.

3. **Soft Margin ($C$ - The Parking Fine):**  
   What if one stubborn Red Wolf refuses to move and sits right in the middle of the street? If you insist on a strictly "Hard Margin", you can't build any straight street at all.  
   Instead, you create a **Soft Margin**: you allow a few houses to sit in the bike lane or across the street, but you charge them a hefty slack penalty ($C$).
   - A huge $C$ makes violations extremely expensive (hard, narrow margin).
   - A small $C$ allows some encroachment in exchange for a wider, smoother highway (soft, robust margin).

4. **The Kernel Trick (The Magic Earthquake):**  
   What if the Red Wolves live in a circle right in the town center, and the Blue Bears surround them in an outer ring? No flat straight road can separate them.  
   Imagine an earthquake magically pushes the center of town straight up into the air, turning the Wolves' zone into a high mountain while leaving the Bears on the ground floor. Now, you can easily slide a flat sheet of cardboard horizontally underneath the mountain to separate them completely! The **Kernel Trick** is mathematical magic that lets us classify in that elevated 3D mountain space without ever having to physically lift the ground.

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $x^{(i)} \in \mathbb{R}^d$ | Feature vector | Coordinates of the $i$-th instance |
| $y^{(i)} \in \{-1, +1\}$ | Discrete label | Binary classification target (note the SVM convention: $\pm 1$) |
| $w \in \mathbb{R}^d, b \in \mathbb{R}$ | Parameter space | Normal vector defining hyperplane orientation and scalar bias offset |
| $\hat{\gamma}^{(i)} = y^{(i)}(w^T x^{(i)} + b)$ | Scalar | Functional margin of sample $i$ |
| $\gamma = \frac{2}{\|w\|_2}$ | $\mathbb{R}_{>0}$ | Total geometric margin width between positive and negative margin boundaries |
| $\xi_i \ge 0$ | Scalar | Slack variable measuring distance of margin violation for sample $i$ |
| $C > 0$ | Hyperparameter | Regularization penalty governing trade-off between margin width and violations |
| $\alpha_i \ge 0$ | Dual variable | Lagrange multiplier associated with sample $i$ |
| $\phi(x)$ | High-dimensional map | Feature mapping function $\mathbb{R}^d \to \mathcal{H}$ |
| $K(x, x') = \langle \phi(x), \phi(x') \rangle$ | Mercer kernel | Inner product evaluation in feature space $\mathcal{H}$ without explicit coordinates |
| $\gamma_{\text{rbf}}$ | Hyperparameter | Bandwidth parameter of Radial Basis Function (RBF) kernel |

---

## 3. Geometric Margin & The Hard-Margin Primal Formulation

Let the separating hyperplane be defined by:
$$\mathcal{H} = \{x \in \mathbb{R}^d : w^T x + b = 0\}$$

The orthogonal Euclidean distance from any arbitrary point $x^{(i)}$ to the hyperplane $\mathcal{H}$ is:
$$\text{dist}(x^{(i)}, \mathcal{H}) = \frac{|w^T x^{(i)} + b|}{\|w\|_2}$$

Taking into account the class label $y^{(i)} \in \{-1, +1\}$, the **geometric margin** of sample $i$ is:
$$\gamma^{(i)} = \frac{y^{(i)}(w^T x^{(i)} + b)}{\|w\|_2}$$

We scale $w$ and $b$ such that the closest points to the hyperplane satisfy:
$$y^{(i)}(w^T x^{(i)} + b) \ge 1 \quad \forall i \in \{1, \dots, N\}$$

For points lying on the margin boundary ($y^{(i)}(w^T x^{(i)} + b) = 1$), their Euclidean distance to the hyperplane is $\frac{1}{\|w\|_2}$.  
Therefore, the **total margin width** between the $+1$ and $-1$ boundaries is:
$$\text{Total Margin} = \frac{1}{\|w\|_2} + \frac{1}{\|w\|_2} = \frac{2}{\|w\|_2}$$

To maximize the margin $\frac{2}{\|w\|_2}$, we equivalently minimize $\frac{1}{2} \|w\|_2^2$:

<div class="highlight">
<strong>Hard-Margin SVM Primal Problem:</strong>
$$\min_{w, b} \frac{1}{2} \|w\|_2^2 \quad \text{subject to } y^{(i)}(w^T x^{(i)} + b) \ge 1 \quad \forall i=1, \dots, N$$
</div>

---

## 4. Soft-Margin SVM & Hinge Loss

When data is not linearly separable, or has outliers, the hard-margin constraints cannot be satisfied. We introduce non-negative **slack variables** $\xi_i \ge 0$:
$$y^{(i)}(w^T x^{(i)} + b) \ge 1 - \xi_i$$

- If $\xi_i = 0$: Point $i$ is correctly classified and outside or on the margin boundary.
- If $0 < \xi_i \le 1$: Point $i$ is correctly classified but violates the margin.
- If $\xi_i > 1$: Point $i$ is misclassified on the wrong side of the decision boundary.

### Primal Formulation with Slack:
$$\min_{w, b, \xi} \frac{1}{2} \|w\|_2^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t. } y^{(i)}(w^T x^{(i)} + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

### Equivalence to Hinge Loss
Notice that when $y^{(i)}(w^T x^{(i)} + b) \ge 1$, the optimal slack is $\xi_i = 0$. Otherwise, $\xi_i = 1 - y^{(i)}(w^T x^{(i)} + b)$.  
Thus, $\xi_i = \max(0, 1 - y^{(i)}(w^T x^{(i)} + b))$.

This allows us to rewrite the soft-margin objective as unconstrained **Hinge Loss + $L_2$ Regularization**:
$$\min_{w, b} \frac{1}{N} \sum_{i=1}^N \max\left(0, 1 - y^{(i)}(w^T x^{(i)} + b)\right) + \frac{\lambda}{2} \|w\|_2^2, \quad \text{where } \lambda = \frac{1}{N C}$$

The subgradient of the hinge loss $\ell(z) = \max(0, 1 - z)$ where $z = y^{(i)}(w^T x^{(i)} + b)$ is:
$$\nabla_w \ell(z) = \begin{cases} \mathbf{0} & \text{if } y^{(i)}(w^T x^{(i)} + b) \ge 1 \\ -y^{(i)} x^{(i)} & \text{if } y^{(i)}(w^T x^{(i)} + b) < 1 \end{cases}$$

This enables optimization via simple, elegant **Subgradient Descent**!

---

## 5. Primal to Dual Transformation via Lagrange Multipliers

To understand how SVMs handle non-linear kernels, we construct the **Lagrangian** of the primal soft-margin problem:
$$\mathcal{L}(w, b, \xi, \alpha, \mu) = \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i - \sum_{i=1}^N \alpha_i \left[ y^{(i)}(w^T x^{(i)} + b) - 1 + \xi_i \right] - \sum_{i=1}^N \mu_i \xi_i$$
where $\alpha_i \ge 0$ and $\mu_i \ge 0$ are Lagrange multipliers.

### Karush-Kuhn-Tucker (KKT) Stationarity Conditions:
1. $\nabla_w \mathcal{L} = w - \sum_{i=1}^N \alpha_i y^{(i)} x^{(i)} = \mathbf{0} \implies w = \sum_{i=1}^N \alpha_i y^{(i)} x^{(i)}$
2. $\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^N \alpha_i y^{(i)} = 0 \implies \sum_{i=1}^N \alpha_i y^{(i)} = 0$
3. $\frac{\partial \mathcal{L}}{\partial \xi_i} = C - \alpha_i - \mu_i = 0 \implies \alpha_i + \mu_i = C \implies 0 \le \alpha_i \le C$

Substituting these back into $\mathcal{L}$ yields the **Wolfe Dual Problem**:

<div class="highlight">
<strong>SVM Dual Formulation:</strong>
$$\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y^{(i)} y^{(j)} \left(x^{(i)} \cdot x^{(j)}\right)$$
$$\text{subject to } \quad 0 \le \alpha_i \le C \quad \forall i, \quad \text{and} \quad \sum_{i=1}^N \alpha_i y^{(i)} = 0$$
</div>

### KKT Complementary Slackness:
$$\alpha_i \left[ y^{(i)}(w^T x^{(i)} + b) - 1 + \xi_i \right] = 0$$

- If $y^{(i)}(w^T x^{(i)} + b) > 1$, then $\alpha_i = 0$: the point plays **zero role** in defining the model!
- If $\alpha_i > 0$, then $y^{(i)}(w^T x^{(i)} + b) \le 1$: the point is a **Support Vector**!

---

## 6. The Kernel Trick

Notice a stunning feature of the Dual formulation: **the features $x$ appear exclusively inside dot products $(x^{(i)} \cdot x^{(j)})$!**

If we map our features into a high-dimensional feature space $\phi(x)$, the dual becomes:
$$\sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y^{(i)} y^{(j)} \langle \phi(x^{(i)}), \phi(x^{(j)}) \rangle$$

By **Mercer's Theorem**, we do not need to explicitly compute or store $\phi(x)$. We only need a **Kernel function** $K(x, x') = \langle \phi(x), \phi(x') \rangle$ that directly computes the inner product:

1. **Linear Kernel:**  
   $$K(x, x') = x^T x'$$
2. **Polynomial Kernel (Degree $d$):**  
   $$K(x, x') = (x^T x' + c)^d$$
3. **Radial Basis Function (RBF / Gaussian) Kernel:**  
   $$K(x, x') = \exp\left( -\gamma \|x - x'\|_2^2 \right)$$

### The RBF Kernel is Infinite-Dimensional!
Using the Taylor series expansion of the exponential function:
$$e^{-\gamma \|x - x'\|^2} = e^{-\gamma \|x\|^2} e^{-\gamma \|x'\|^2} \sum_{k=0}^\infty \frac{(2\gamma x^T x')^k}{k!}$$
The RBF kernel implicitly maps input vectors into an **infinite-dimensional Hilbert space**, enabling SVMs to create arbitrarily complex, smooth, non-linear closed decision boundaries!
