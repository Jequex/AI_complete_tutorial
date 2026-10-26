# Topic 9.3: Exercises & Solutions — Manifold Learning (t-SNE & UMAP)

---

### Problem 1: High-Dimensional Gaussian Affinities Hand Calculation (Easy)
Consider three points in 1D space:
$$x_1 = 0.0, \quad x_2 = 1.0, \quad x_3 = 4.0$$
Suppose the Gaussian variance for all points is fixed at $\sigma^2 = 2.0$ (so $2\sigma^2 = 4.0$).

1. Compute the unnormalized weights $w_{j|1} = \exp\left(-\frac{(x_1 - x_j)^2}{2\sigma^2}\right)$ for $j \in \{2, 3\}$.
2. Compute the normalized conditional probabilities $p_{2|1}$ and $p_{3|1}$.
3. Why is $p_{3|1}$ significantly smaller than $p_{2|1}$?

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute unnormalized Gaussian affinities from $x_1 = 0$:**
- For $j = 2$ ($x_2 = 1.0$):
  $$d^2 = (0.0 - 1.0)^2 = 1.0 \implies w_{2|1} = e^{-1.0 / 4.0} = e^{-0.25} \approx 0.7788$$
- For $j = 3$ ($x_3 = 4.0$):
  $$d^2 = (0.0 - 4.0)^2 = 16.0 \implies w_{3|1} = e^{-16.0 / 4.0} = e^{-4.0} \approx 0.0183$$

**Step 2: Normalize conditional probabilities:**
Sum of weights:
$$\sum_{k \ne 1} w_{k|1} = 0.7788 + 0.0183 = 0.7971$$

- $p_{2|1} = \frac{0.7788}{0.7971} \approx 0.9770$ (97.7%)
- $p_{3|1} = \frac{0.0183}{0.7971} \approx 0.0230$ (2.3%)

**Step 3: Explanation:**
Because the Gaussian kernel decays exponentially with squared Euclidean distance ($e^{-d^2}$), points that are 4 units away have almost negligible influence compared to points that are 1 unit away. This ensures that only local neighborhoods dominate the similarity profile.
</details>

---

### Problem 2: Derivative of the Student-t Kernel (Medium)
Let $y_i, y_j \in \mathbb{R}^d$ be low-dimensional coordinate vectors. Define the unnormalized Student-t kernel:
$$w_{ij} = \left( 1 + \|y_i - y_j\|_2^2 \right)^{-1}$$
Prove by multivariable calculus that:
$$\frac{\partial w_{ij}}{\partial y_i} = -2 w_{ij}^2 (y_i - y_j)$$

<details>
<summary>View Complete Solution</summary>

Let $u = 1 + \|y_i - y_j\|_2^2 = 1 + \sum_{m=1}^d (y_{im} - y_{jm})^2$.
Then $w_{ij} = u^{-1}$.

By the chain rule:
$$\frac{\partial w_{ij}}{\partial y_i} = \frac{d(u^{-1})}{du} \cdot \nabla_{y_i} u$$

1. Derivative of $u^{-1}$:
   $$\frac{d(u^{-1})}{du} = -u^{-2} = -w_{ij}^2$$
2. Gradient of $u$ with respect to vector $y_i$:
   $$\nabla_{y_i} u = \nabla_{y_i} \left( 1 + \|y_i - y_j\|_2^2 \right) = 2 (y_i - y_j)$$

Combining the two terms:
$$\frac{\partial w_{ij}}{\partial y_i} = -w_{ij}^2 \cdot 2(y_i - y_j) = -2 w_{ij}^2 (y_i - y_j)$$
$\blacksquare$
</details>

---

### Problem 3: Two-Point 1D t-SNE Gradient Hand Calculation (Medium)
Suppose we have only two points in 1D low-dimensional space: $y_1 = 0.0$ and $y_2 = 2.0$.
Suppose their high-dimensional symmetric probability is $p_{12} = 0.5$ (since $N = 2$, $p_{12} = p_{21} = 0.5$).

1. Compute the unnormalized Student-t weight $w_{12} = (1 + (y_1 - y_2)^2)^{-1}$.
2. Compute the normalized low-dimensional probability $q_{12}$.
3. Compute the analytical gradient $\frac{\partial \mathcal{L}}{\partial y_1}$ of the KL divergence.
4. Interpret the direction of the gradient: does gradient descent push $y_1$ closer to $y_2$ or further away?

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute $w_{12}$:**
$$d^2 = (y_1 - y_2)^2 = (0.0 - 2.0)^2 = 4.0$$
$$w_{12} = \frac{1}{1 + 4.0} = \frac{1}{5.0} = 0.2$$

Since $w_{21} = w_{12} = 0.2$, the partition sum is $Z = w_{12} + w_{21} = 0.4$.

**Step 2: Compute $q_{12}$:**
$$q_{12} = \frac{w_{12}}{Z} = \frac{0.2}{0.4} = 0.5$$

**Step 3: Compute the gradient $\frac{\partial \mathcal{L}}{\partial y_1}$:**
Recall the gradient formula:
$$\frac{\partial \mathcal{L}}{\partial y_1} = 4 (p_{12} - q_{12}) (1 + \|y_1 - y_2\|^2)^{-1} (y_1 - y_2)$$

Substitute the known values:
- $p_{12} = 0.5$
- $q_{12} = 0.5$
- $p_{12} - q_{12} = 0.5 - 0.5 = 0.0$

$$\frac{\partial \mathcal{L}}{\partial y_1} = 4 (0.0) (0.2) (0.0 - 2.0) = 0.0$$

**Step 4: Interpretation:**
Because $q_{12} = p_{12} = 0.5$, the low-dimensional distribution matches the target distribution perfectly! The system is in exact physical equilibrium, so the gradient is zero.
</details>

---

### Problem 4: t-SNE vs UMAP Objective Comparison (Hard)
Contrast the behavior of the t-SNE objective function $\mathcal{L}_{\text{t-SNE}} = \sum_{i \ne j} p_{ij} \ln \frac{p_{ij}}{q_{ij}}$ with the UMAP objective function $\mathcal{L}_{\text{UMAP}} = \sum_{i \ne j} \left[ p_{ij} \ln \frac{p_{ij}}{q_{ij}} + (1 - p_{ij}) \ln \frac{1 - p_{ij}}{1 - q_{ij}} \right]$ when two completely unrelated clusters ($p_{ij} \approx 0$) are placed right next to each other in low dimensions ($q_{ij} \gg 0$).

Why does UMAP preserve global structure significantly better than t-SNE?

<details>
<summary>View Complete Solution</summary>

**In t-SNE:**
When $p_{ij} \approx 0$ (distant points in high-dimensional space) and $q_{ij} \approx 0.1$ (placed close together in 2D):
$$\text{Penalty}_{\text{t-SNE}} = p_{ij} \ln \frac{p_{ij}}{q_{ij}} \approx 0 \cdot \ln(0) = 0$$
t-SNE assesses **almost zero penalty** for placing non-neighbors close to each other! As long as high-dimensional neighbors ($p_{ij} > 0$) stay together, t-SNE is completely indifferent to where distinct clusters end up relative to one another.

**In UMAP:**
When $p_{ij} = 0$ and $q_{ij} > 0$:
The first term vanishes ($0 \cdot \ln(0/q) = 0$), but look at the second term:
$$\text{Penalty}_{\text{UMAP}} = (1 - p_{ij}) \ln \frac{1 - p_{ij}}{1 - q_{ij}} = (1 - 0) \ln \frac{1 - 0}{1 - q_{ij}} = -\ln(1 - q_{ij})$$
If $q_{ij} \to 1$ (non-neighbors placed directly on top of each other), $-\ln(1 - q_{ij}) \to \infty$!

**Conclusion:**
UMAP introduces an explicit global repulsive force that strongly penalizes placing distant points close together in the low-dimensional space. This mathematical penalty forces entire distinct clusters apart, preserving global macro-topology.
</details>
