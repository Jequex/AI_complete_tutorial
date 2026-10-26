# Topic 23.3: Exercises & Solutions — Metric Learning & Contrastive Losses

---

### Problem 1: Contrastive Loss Gradient Step & Equilibrium (Easy)

Consider two 2D embedding representations:
$$f(x_1) = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}, \quad f(x_2) = \begin{bmatrix} 4.0 \\ 6.0 \end{bmatrix}$$
with Euclidean margin $m = 8.0$.
Recall the Contrastive Loss formulation:
$$\mathcal{L} = \frac{1}{2} (1 - y) D^2 + \frac{1}{2} y \max(0, m - D)^2$$
where $D = \|f(x_1) - f(x_2)\|_2$.

1. Compute the Euclidean distance $D$.
2. Compute the loss $\mathcal{L}$ and gradient $\nabla_{f(x_1)} \mathcal{L}$ assuming the pair is positive ($y = 0$).
3. Compute the loss $\mathcal{L}$ and gradient $\nabla_{f(x_1)} \mathcal{L}$ assuming the pair is negative ($y = 1$).
4. If $f(x_2)$ was located at $[10.0, 14.0]^\top$ for a negative pair ($y = 1$), what would the loss and gradient be?

<details>
<summary>View Complete Solution</summary>

**Step 1: Distance Calculation:**
$$f(x_1) - f(x_2) = \begin{bmatrix} 1.0 - 4.0 \\ 2.0 - 6.0 \end{bmatrix} = \begin{bmatrix} -3.0 \\ -4.0 \end{bmatrix}$$
$$D = \sqrt{(-3.0)^2 + (-4.0)^2} = \sqrt{9 + 16} = \sqrt{25} = 5.0$$

**Step 2: Positive Pair ($y = 0$):**
$$\mathcal{L}_{\text{pos}} = \frac{1}{2} D^2 = \frac{1}{2} (5.0)^2 = 12.5$$
$$\nabla_{f(x_1)} \mathcal{L}_{\text{pos}} = f(x_1) - f(x_2) = \begin{bmatrix} -3.0 \\ -4.0 \end{bmatrix}$$
An attractive step pulls $f(x_1)$ toward $f(x_2)$.

**Step 3: Negative Pair ($y = 1$) with $D = 5.0, m = 8.0$:**
Since $D = 5.0 < m = 8.0$, the pair violates the margin:
$$\mathcal{L}_{\text{neg}} = \frac{1}{2} (m - D)^2 = \frac{1}{2} (8.0 - 5.0)^2 = \frac{1}{2} (3.0)^2 = 4.5$$
Gradient with respect to $f(x_1)$:
$$\nabla_{f(x_1)} \mathcal{L}_{\text{neg}} = -(m - D) \frac{f(x_1) - f(x_2)}{D} = -(8.0 - 5.0) \frac{1}{5.0} \begin{bmatrix} -3.0 \\ -4.0 \end{bmatrix} = -\frac{3.0}{5.0} \begin{bmatrix} -3.0 \\ -4.0 \end{bmatrix} = \begin{bmatrix} +1.8 \\ +2.4 \end{bmatrix}$$
Notice the direction: $[+1.8, +2.4]^\top$ points directly *away* from $f(x_2)$, repelling $f(x_1)$.

**Step 4: Distant Negative Pair:**
If $f(x_2) = [10.0, 14.0]^\top$:
$$D_{\text{new}} = \sqrt{(1.0 - 10.0)^2 + (2.0 - 14.0)^2} = \sqrt{(-9)^2 + (-12)^2} = \sqrt{81 + 144} = \sqrt{225} = 15.0$$
Since $D_{\text{new}} = 15.0 \ge m = 8.0$:
$$\max(0, m - D_{\text{new}}) = \max(0, 8.0 - 15.0) = 0.0$$
$$\mathcal{L}_{\text{neg}} = 0.0, \quad \nabla_{f(x_1)} \mathcal{L} = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$$
No repulsive force is exerted because the representations are already safely beyond the margin.
</details>

---

### Problem 2: Triplet Loss Negative Mining Classification (Medium)

In a face verification mini-batch, an anchor sample $a$ has a squared distance to its positive pair $p$ of $D_{ap} = \|f(a) - f(p)\|_2^2 = 0.36$.
The margin hyperparameter is $\alpha = 0.40$.
The anchor is evaluated against four candidate negative samples $\{n_1, n_2, n_3, n_4\}$ with squared Euclidean distances:
- $D_{a, n_1} = 0.20$
- $D_{a, n_2} = 0.50$
- $D_{a, n_3} = 0.72$
- $D_{a, n_4} = 0.95$

1. Classify each of the four negatives as **Easy**, **Semi-Hard**, or **Hard**.
2. Calculate the exact Triplet Loss value $\mathcal{L}_{\text{triplet}}(a, p, n_i)$ for each candidate negative.
3. Which candidate negative should be selected under the FaceNet semi-hard mining policy? Explain why selecting $n_1$ can destabilize training.

<details>
<summary>View Complete Solution</summary>

**Step 1: Threshold Definitions:**
- Lower boundary: $D_{ap} = 0.36$.
- Upper boundary: $D_{ap} + \alpha = 0.36 + 0.40 = 0.76$.

Classification:
- $n_1$ ($D = 0.20$): Since $D_{a, n_1} < D_{ap}$ ($0.20 < 0.36$), $n_1$ is a **Hard Negative**.
- $n_2$ ($D = 0.50$): Since $D_{ap} < D_{a, n_2} < D_{ap} + \alpha$ ($0.36 < 0.50 < 0.76$), $n_2$ is a **Semi-Hard Negative**.
- $n_3$ ($D = 0.72$): Since $D_{ap} < D_{a, n_3} < D_{ap} + \alpha$ ($0.36 < 0.72 < 0.76$), $n_3$ is a **Semi-Hard Negative**.
- $n_4$ ($D = 0.95$): Since $D_{a, n_4} > D_{ap} + \alpha$ ($0.95 > 0.76$), $n_4$ is an **Easy Negative**.

**Step 2: Triplet Loss Calculations:**
$$\mathcal{L} = \max(0, D_{ap} - D_{an} + \alpha)$$
- For $n_1$: $\mathcal{L} = \max(0, 0.36 - 0.20 + 0.40) = \max(0, 0.56) = \mathbf{0.56}$
- For $n_2$: $\mathcal{L} = \max(0, 0.36 - 0.50 + 0.40) = \max(0, 0.26) = \mathbf{0.26}$
- For $n_3$: $\mathcal{L} = \max(0, 0.36 - 0.72 + 0.40) = \max(0, 0.04) = \mathbf{0.04}$
- For $n_4$: $\mathcal{L} = \max(0, 0.36 - 0.95 + 0.40) = \max(0, -0.19) = \mathbf{0.00}$

**Step 3: Mining Selection Policy:**
Under FaceNet's semi-hard mining policy, candidate $n_2$ or $n_3$ should be selected (typically the semi-hard negative closest to the positive boundary, which is $n_2$).
Why not $n_1$?
In $n_1$, $D_{an} < D_{ap}$. In early or intermediate training, hard negatives often correspond to noisy or mislabeled images (e.g., two people wearing identical dark sunglasses). If the network is forced to separate $a$ from $n_1$, it receives massive gradient forces that can collapse the feature manifold into a degenerate subspace or cause catastrophic forgetting of previously formed clusters. Semi-hard negatives guarantee that the positive is already closer than the negative, providing a smooth, strictly restorative gradient step.
</details>

---

### Problem 3: Geometric Boundary of Zero Triplet Loss (Hard)

Let anchor $a$ be fixed at the origin $\mathbf{0} \in \mathbb{R}^2$, and let positive $p$ be fixed at $[1, 0]^\top \in \mathbb{R}^2$.
Let margin $\alpha = 1.0$.
Consider the set of all negative points $n = [x, y]^\top \in \mathbb{R}^2$ that produce exactly zero Triplet Loss:
$$\mathcal{L}_{\text{triplet}}(a, p, n) = \max(0, \|a - p\|_2^2 - \|a - n\|_2^2 + \alpha) = 0$$

1. Express the condition for $\mathcal{L} = 0$ as an inequality in terms of coordinates $x$ and $y$.
2. What geometric curve forms the boundary between non-zero loss and zero loss?
3. Contrast this geometric boundary with the zero-loss boundary of Contrastive Loss.

<details>
<summary>View Complete Solution</summary>

**Step 1: Deriving the Coordinate Inequality:**
- Anchor: $a = [0, 0]^\top$.
- Positive: $p = [1, 0]^\top \implies \|a - p\|_2^2 = (1 - 0)^2 + (0 - 0)^2 = 1.0$.
- Negative: $n = [x, y]^\top \implies \|a - n\|_2^2 = x^2 + y^2$.
- Margin: $\alpha = 1.0$.

For Triplet Loss to be zero:
$$\|a - p\|_2^2 - \|a - n\|_2^2 + \alpha \le 0$$
Substitute values:
$$1.0 - (x^2 + y^2) + 1.0 \le 0 \implies 2.0 \le x^2 + y^2 \implies x^2 + y^2 \ge 2.0$$

**Step 2: Boundary Geometry:**
The boundary is defined by:
$$x^2 + y^2 = 2.0$$
This is a **circle centered at the origin (anchor $a$) with radius $R = \sqrt{2} \approx 1.414$**.
Any negative sample situated outside this circle of radius $\sqrt{2}$ yields zero loss.

**Step 3: Contrast with Contrastive Loss:**
- In Contrastive Loss, the margin $m$ is **absolute and invariant** to the positive: a negative must be distance $m$ away from $x_1$ regardless of where the positive $x_2$ is.
- In Triplet Loss, the boundary radius $R = \sqrt{\|a - p\|_2^2 + \alpha}$ is **dynamic and adaptive**:
  - If the model brings the positive very close to the anchor ($\|a - p\|_2^2 \to 0$), the required negative boundary shrinks to $\sqrt{\alpha}$.
  - If the positive is far ($\|a - p\|_2^2 = 5.0$), the negative boundary automatically expands to $\sqrt{5 + 1} = \sqrt{6} \approx 2.45$.
  - Triplet loss enforces *relative margins*, allowing compact clusters in dense classes and broader clusters in diverse classes without forcing a single rigid global radius.
</details>
