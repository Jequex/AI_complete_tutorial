# Topic 9.1: Exercises & Solutions — Clustering (K-Means & GMM)

---

### Problem 1: Lloyd's Algorithm Hand Calculation (Easy)
Consider four 1D data points:
$$X = \{1.0, 2.0, 6.0, 7.0\}$$
Suppose we set $K = 2$ and initialize centroids at $\mu_1^{(0)} = 1.0$ and $\mu_2^{(0)} = 5.0$.

1. Compute the cluster assignments $r_{ik}$ for iteration 0.
2. Compute the updated centroids $\mu_1^{(1)}$ and $\mu_2^{(1)}$.
3. Check if the algorithm converges at iteration 1.
4. Compute the final Inertia $J$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Distance calculation and assignment:**
- For $x_1 = 1.0$: $|1.0 - 1.0| = 0$, $|1.0 - 5.0| = 4 \implies x_1 \in C_1$.
- For $x_2 = 2.0$: $|2.0 - 1.0| = 1$, $|2.0 - 5.0| = 3 \implies x_2 \in C_1$.
- For $x_3 = 6.0$: $|6.0 - 1.0| = 5$, $|6.0 - 5.0| = 1 \implies x_3 \in C_2$.
- For $x_4 = 7.0$: $|7.0 - 1.0| = 6$, $|7.0 - 5.0| = 2 \implies x_4 \in C_2$.

Cluster groupings: $C_1 = \{1.0, 2.0\}$, $C_2 = \{6.0, 7.0\}$.

**Step 2: Centroid updates:**
$$\mu_1^{(1)} = \frac{1.0 + 2.0}{2} = 1.5$$
$$\mu_2^{(1)} = \frac{6.0 + 7.0}{2} = 6.5$$

**Step 3: Convergence check:**
Re-evaluating distances with $\mu_1 = 1.5$ and $\mu_2 = 6.5$:
- $x_1 = 1.0$: $|1.0 - 1.5| = 0.5 < |1.0 - 6.5| = 5.5 \implies C_1$.
- $x_2 = 2.0$: $|2.0 - 1.5| = 0.5 < |2.0 - 6.5| = 4.5 \implies C_1$.
- $x_3 = 6.0$: $|6.0 - 6.5| = 0.5 < |6.0 - 1.5| = 4.5 \implies C_2$.
- $x_4 = 7.0$: $|7.0 - 6.5| = 0.5 < |7.0 - 1.5| = 5.5 \implies C_2$.

The cluster memberships do not change. Centroids remain unchanged:
$$\mu_1^{(2)} = 1.5, \quad \mu_2^{(2)} = 6.5$$
The algorithm has converged in exactly 1 iteration!

**Step 4: Final Inertia:**
$$J = \sum_{x \in C_1} (x - 1.5)^2 + \sum_{x \in C_2} (x - 6.5)^2$$
$$J = (1.0 - 1.5)^2 + (2.0 - 1.5)^2 + (6.0 - 6.5)^2 + (7.0 - 6.5)^2$$
$$J = 0.25 + 0.25 + 0.25 + 0.25 = 1.0$$
</details>

---

### Problem 2: Lagrange Multiplier Derivation of GMM Mixture Weights (Medium)
In the M-step of the Expectation-Maximization algorithm for GMMs, we maximize the expected complete-data log-likelihood with respect to mixture weights $\pi = (\pi_1, \dots, \pi_K)$:
$$\max_{\pi} \sum_{i=1}^N \sum_{k=1}^K \gamma_{ik} \ln \pi_k \quad \text{subject to} \quad \sum_{k=1}^K \pi_k = 1, \quad \pi_k \ge 0$$
Using the method of Lagrange multipliers, prove that the optimal update is:
$$\pi_k^* = \frac{N_k}{N}, \quad \text{where } N_k = \sum_{i=1}^N \gamma_{ik}$$

<details>
<summary>View Complete Solution</summary>

**Form the Lagrangian:**
Introduce Lagrange multiplier $\lambda \in \mathbb{R}$ for the equality constraint $\sum_{k=1}^K \pi_k = 1$:
$$\Lambda(\pi, \lambda) = \sum_{i=1}^N \sum_{k=1}^K \gamma_{ik} \ln \pi_k + \lambda \left(1 - \sum_{k=1}^K \pi_k\right)$$

**Take the partial derivative with respect to $\pi_k$:**
$$\frac{\partial \Lambda}{\partial \pi_k} = \sum_{i=1}^N \frac{\gamma_{ik}}{\pi_k} - \lambda = 0$$

Multiply both sides by $\pi_k$:
$$\sum_{i=1}^N \gamma_{ik} = \lambda \pi_k \implies N_k = \lambda \pi_k$$

**Determine the multiplier $\lambda$:**
Sum both sides over all $k = 1, \dots, K$:
$$\sum_{k=1}^K N_k = \sum_{k=1}^K \lambda \pi_k = \lambda \sum_{k=1}^K \pi_k$$

Since $\sum_{k=1}^K \pi_k = 1$, we have:
$$\sum_{k=1}^K N_k = \lambda$$

Recall that for every point $i$, the responsibilities sum to 1 ($\sum_{k=1}^K \gamma_{ik} = 1$):
$$\sum_{k=1}^K N_k = \sum_{k=1}^K \sum_{i=1}^N \gamma_{ik} = \sum_{i=1}^N \left(\sum_{k=1}^K \gamma_{ik}\right) = \sum_{i=1}^N 1 = N$$

Therefore:
$$\lambda = N$$

**Substitute $\lambda = N$ back:**
$$N_k = N \pi_k \implies \pi_k^* = \frac{N_k}{N}$$
$\blacksquare$
</details>

---

### Problem 3: K-Means++ Seeding Probability (Medium)
Suppose we have 4 data points in 2D:
$$x_1 = (0, 0), \quad x_2 = (1, 0), \quad x_3 = (0, 1), \quad x_4 = (10, 10)$$
The first centroid chosen uniformly at random is $\mu_1 = x_1 = (0, 0)$.

1. Compute the squared distance $D(x_i)^2$ from each point to $\mu_1$.
2. Compute the exact probability $P(x_i)$ of selecting each point as the second centroid $\mu_2$.
3. Why does K-Means++ heavily bias the selection toward $x_4$?

<details>
<summary>View Complete Solution</summary>

**Step 1: Squared distances:**
- $D(x_1)^2 = \|(0,0) - (0,0)\|^2 = 0$
- $D(x_2)^2 = \|(1,0) - (0,0)\|^2 = 1^2 + 0^2 = 1$
- $D(x_3)^2 = \|(0,1) - (0,0)\|^2 = 0^2 + 1^2 = 1$
- $D(x_4)^2 = \|(10,10) - (0,0)\|^2 = 10^2 + 10^2 = 200$

Total squared distance:
$$\sum_{m=1}^4 D(x_m)^2 = 0 + 1 + 1 + 200 = 202$$

**Step 2: Probabilities:**
- $P(x_1) = \frac{0}{202} = 0.0$
- $P(x_2) = \frac{1}{202} \approx 0.00495$ (0.495%)
- $P(x_3) = \frac{1}{202} \approx 0.00495$ (0.495%)
- $P(x_4) = \frac{200}{202} \approx 0.9901$ (99.01%)

**Step 3: Intuition:**
Because $x_4$ is distant from $x_1$, $D(x_4)^2$ dominates the sum. K-Means++ ensures that distant, isolated clusters receive their own initial centroid rather than crowding all initial seeds into a single tight cluster.
</details>

---

### Problem 4: DBSCAN Point Categorization (Hard)
Given points on a 1D number line:
$$X = \{1, 2, 2.5, 3, 7, 8, 15\}$$
With parameters $\epsilon = 1.2$ and $\text{MinPts} = 3$:

1. For each point $x \in X$, list all points in its $\epsilon$-neighborhood $N_\epsilon(x)$ and count $|N_\epsilon(x)|$.
2. Identify which points are **Core Points**.
3. Identify which points are **Border Points**.
4. Identify which points are **Noise Points (Outliers)**.
5. List the final clusters formed.

<details>
<summary>View Complete Solution</summary>

**Step 1: Neighborhood evaluation ($|x_j - x_i| \le 1.2$):**
- $x = 1$: Neighbors = $\{1, 2\}$, count = 2.
- $x = 2$: Neighbors = $\{1, 2, 2.5, 3\}$, count = 4.
- $x = 2.5$: Neighbors = $\{2, 2.5, 3\}$, count = 3.
- $x = 3$: Neighbors = $\{2, 2.5, 3\}$, count = 3.
- $x = 7$: Neighbors = $\{7, 8\}$, count = 2.
- $x = 8$: Neighbors = $\{7, 8\}$, count = 2.
- $x = 15$: Neighbors = $\{15\}$, count = 1.

**Step 2: Core Points ($|N_\epsilon(x)| \ge \text{MinPts} = 3$):**
- Points $2$, $2.5$, and $3$ have count $\ge 3$.
- **Core points = $\{2, 2.5, 3\}$.**

**Step 3: Border Points (Non-core points that belong to $N_\epsilon(\text{core})$):**
- Point $1$ has count 2 (< 3, so not core), but $|1 - 2| = 1.0 \le 1.2$, so it lies in the neighborhood of core point $2$.
- Point $7$: neighbors $\{7, 8\}$ are both non-core.
- Point $8$: neighbors $\{7, 8\}$ are both non-core.
- **Border point = $\{1\}$.**

**Step 4: Noise Points (Neither core nor border):**
- Points $7, 8, 15$ are not core and do not touch any core point.
- **Noise points = $\{7, 8, 15\}$.**

**Step 5: Final Clusters:**
- **Cluster 1:** $\{1, 2, 2.5, 3\}$ (Core points $2, 2.5, 3$ form a connected dense component, absorbing border point $1$).
- **Noise:** $\{7, 8, 15\}$.
</details>
