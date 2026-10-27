# Topic 04: Constrained Optimization & Duality — Exercises & Solutions

## Exercise 1: Dual Derivation of the Hard-Margin Support Vector Machine

The primal hard-margin Support Vector Machine (SVM) optimization problem is:
$$\begin{aligned}
\min_{\mathbf{w}, b} \quad & \frac{1}{2} \|\mathbf{w}\|_2^2 \\
\text{subject to} \quad & y_i (\mathbf{w}^\top \mathbf{x}_i + b) \ge 1, \quad i = 1, \dots, N
\end{aligned}$$
where $\mathbf{x}_i \in \mathbb{R}^d$ and $y_i \in \{-1, +1\}$.

1. Form the Lagrangian function $\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha})$ with dual multipliers $\alpha_i \ge 0$.
2. Apply KKT stationarity conditions $\nabla_{\mathbf{w}} \mathcal{L} = \mathbf{0}$ and $\frac{\partial \mathcal{L}}{\partial b} = 0$ to solve for optimal $\mathbf{w}$ in terms of dual variables.
3. Substitute these results back into the Lagrangian to derive the Wolfe dual problem.
4. Using complementary slackness, explain why the decision boundary depends only on the support vectors.

### Solution 1:
1. **The Lagrangian**:
   Rewrite constraints as $1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b) \le 0$:
   $$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2} \|\mathbf{w}\|_2^2 + \sum_{i=1}^N \alpha_i \left( 1 - y_i (\mathbf{w}^\top \mathbf{x}_i + b) \right)$$
   where $\alpha_i \ge 0$.

2. **KKT Stationarity Conditions**:
   - Differentiating with respect to $\mathbf{w}$:
     $$\nabla_{\mathbf{w}} \mathcal{L} = \mathbf{w} - \sum_{i=1}^N \alpha_i y_i \mathbf{x}_i = \mathbf{0} \implies \mathbf{w}^* = \sum_{i=1}^N \alpha_i y_i \mathbf{x}_i$$
   - Differentiating with respect to $b$:
     $$\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^N \alpha_i y_i = 0 \implies \sum_{i=1}^N \alpha_i y_i = 0$$

3. **The Dual Optimization Problem**:
   Substitute $\mathbf{w}^*$ back into the Lagrangian:
   $$\begin{aligned}
   \mathcal{L}(\mathbf{w}^*, b^*, \boldsymbol{\alpha}) &= \frac{1}{2} \left( \sum_{i=1}^N \alpha_i y_i \mathbf{x}_i \right)^\top \left( \sum_{j=1}^N \alpha_j y_j \mathbf{x}_j \right) + \sum_{i=1}^N \alpha_i - \sum_{i=1}^N \alpha_i y_i \mathbf{w}^\top \mathbf{x}_i - b \underbrace{\sum_{i=1}^N \alpha_i y_i}_{0} \\
   &= \frac{1}{2} \sum_{i,j=1}^N \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j + \sum_{i=1}^N \alpha_i - \sum_{i,j=1}^N \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j \\
   &= \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j (\mathbf{x}_i^\top \mathbf{x}_j)
   \end{aligned}$$
   The Wolfe Dual is:
   $$\begin{aligned}
   \max_{\boldsymbol{\alpha}} \quad & \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j (\mathbf{x}_i^\top \mathbf{x}_j) \\
   \text{subject to} \quad & \alpha_i \ge 0, \quad i = 1, \dots, N \\
   & \sum_{i=1}^N \alpha_i y_i = 0
   \end{aligned}$$

4. **Sparsity from Complementary Slackness**:
   KKT complementary slackness requires:
   $$\alpha_i \left( 1 - y_i (\mathbf{w}^\top \mathbf{x}_i + b) \right) = 0$$
   - For all training points that are safely beyond the margin ($y_i(\mathbf{w}^\top \mathbf{x}_i + b) > 1$), the bracket is strictly negative, so $\alpha_i = 0$.
   - Only for the rare critical points lying directly on the margin hyperplanes ($y_i(\mathbf{w}^\top \mathbf{x}_i + b) = 1$) can $\alpha_i > 0$.
   Therefore, $\mathbf{w}^* = \sum_{i \in \text{SV}} \alpha_i y_i \mathbf{x}_i$ is a sparse linear combination of only the support vectors.

---

## Exercise 2: Water-Filling Algorithm via KKT Conditions

Consider allocating total transmit power $P_{\text{total}} > 0$ across $K$ parallel communication channels to maximize total Shannon capacity:
$$\begin{aligned}
\min_{p_1, \dots, p_K} \quad & -\sum_{k=1}^K \ln(1 + \frac{p_k}{\sigma_k^2}) \\
\text{subject to} \quad & \sum_{k=1}^K p_k = P_{\text{total}} \\
& p_k \ge 0, \quad k = 1, \dots, K
\end{aligned}$$
where $\sigma_k^2 > 0$ is the noise variance in channel $k$.

1. State the Lagrangian with multipliers $\nu \in \mathbb{R}$ for the total power equality and $\lambda_k \ge 0$ for non-negativity.
2. Formulate the KKT stationarity condition for each channel $k$.
3. Use complementary slackness to derive the exact analytical **water-filling solution** $p_k^* = \max\left(0, \frac{1}{\nu} - \sigma_k^2\right)$.

### Solution 2:
1. **The Lagrangian**:
   $$\mathcal{L}(\mathbf{p}, \nu, \boldsymbol{\lambda}) = -\sum_{k=1}^K \ln\left(1 + \frac{p_k}{\sigma_k^2}\right) + \nu \left( \sum_{k=1}^K p_k - P_{\text{total}} \right) - \sum_{k=1}^K \lambda_k p_k$$
   where $\lambda_k \ge 0$ and $\nu \in \mathbb{R}$.

2. **KKT Stationarity**:
   Differentiating with respect to $p_k$:
   $$\frac{\partial \mathcal{L}}{\partial p_k} = -\frac{1}{1 + \frac{p_k}{\sigma_k^2}} \cdot \frac{1}{\sigma_k^2} + \nu - \lambda_k = -\frac{1}{p_k + \sigma_k^2} + \nu - \lambda_k = 0$$
   $$\frac{1}{p_k + \sigma_k^2} = \nu - \lambda_k$$

3. **Water-Filling Solution**:
   From complementary slackness: $\lambda_k p_k = 0$ with $\lambda_k \ge 0$:
   - **Case 1**: $p_k^* > 0$. Then $\lambda_k = 0$, which gives:
     $$\frac{1}{p_k^* + \sigma_k^2} = \nu \implies p_k^* = \frac{1}{\nu} - \sigma_k^2$$
     Since $p_k^* > 0$, this requires $\frac{1}{\nu} > \sigma_k^2$.
   - **Case 2**: $p_k^* = 0$. Then $\lambda_k \ge 0$, which gives:
     $$\frac{1}{\sigma_k^2} = \nu - \lambda_k \le \nu \implies \frac{1}{\nu} \le \sigma_k^2$$
   Combining both cases:
   $$p_k^* = \max\left(0, \frac{1}{\nu^*} - \sigma_k^2\right)$$
   where the "water level" $\frac{1}{\nu^*}$ is uniquely determined by solving $\sum_{k=1}^K \max\left(0, \frac{1}{\nu^*} - \sigma_k^2\right) = P_{\text{total}}$. Channels with noise higher than the water level receive zero power.

---

## Exercise 3: Penalty vs. Augmented Lagrangian Formulation

To enforce equality constraint $c(\mathbf{x}) = 0$ in unconstrained neural network training:
1. Contrast the Quadratic Penalty Method $\mathcal{L}_{\rho}(\mathbf{x}) = f(\mathbf{x}) + \frac{\rho}{2} c(\mathbf{x})^2$ with the Augmented Lagrangian Method (ALM):
   $$\mathcal{L}_{\text{ALM}}(\mathbf{x}, \nu) = f(\mathbf{x}) + \nu c(\mathbf{x}) + \frac{\rho}{2} c(\mathbf{x})^2$$
2. Explain why the Quadratic Penalty requires $\rho \to \infty$ (causing catastrophic ill-conditioning of the Hessian), while ALM converges to the exact constrained optimum at a finite $\rho^*$.

### Solution 3:
1. **Stationarity Comparison**:
   - For Quadratic Penalty:
     $$\nabla \mathcal{L}_{\rho}(\mathbf{x}) = \nabla f(\mathbf{x}) + \rho c(\mathbf{x}) \nabla c(\mathbf{x}) = \mathbf{0}$$
     For this to match KKT stationarity $\nabla f(\mathbf{x}^*) + \nu^* \nabla c(\mathbf{x}^*) = \mathbf{0}$, we must have:
     $$\rho c(\mathbf{x}_\rho) \approx \nu^* \implies c(\mathbf{x}_\rho) \approx \frac{\nu^*}{\rho}$$
     To achieve $c(\mathbf{x}) \to 0$, we must send $\rho \to \infty$.
   - For Augmented Lagrangian:
     $$\nabla \mathcal{L}_{\text{ALM}}(\mathbf{x}, \nu) = \nabla f(\mathbf{x}) + (\nu + \rho c(\mathbf{x})) \nabla c(\mathbf{x}) = \mathbf{0}$$
     If $\nu = \nu^*$, stationarity is satisfied at $c(\mathbf{x}^*) = 0$ for *any* finite $\rho > 0$!

2. **Hessian Ill-Conditioning**:
   The Hessian of the quadratic penalty objective is:
   $$\nabla^2 \mathcal{L}_{\rho}(\mathbf{x}) = \nabla^2 f(\mathbf{x}) + \rho \nabla c(\mathbf{x}) \nabla c(\mathbf{x})^\top + \rho c(\mathbf{x}) \nabla^2 c(\mathbf{x})$$
   As $\rho \to \infty$, the eigenvalue along $\nabla c(\mathbf{x})$ scales as $\mathcal{O}(\rho)$, while orthogonal tangent directions remain $\mathcal{O}(1)$. The condition number $\kappa \approx \mathcal{O}(\rho) \to \infty$, paralyzing gradient-based optimizers.
   ALM updates the multiplier via $\nu_{k+1} = \nu_k + \rho c(\mathbf{x}_k)$, converging to the exact KKT point while keeping $\rho$ modest and the Hessian well-conditioned.
