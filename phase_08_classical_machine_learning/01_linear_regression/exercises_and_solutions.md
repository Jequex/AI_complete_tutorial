# Topic 8.1: Exercises & Solutions — Linear Regression & Normal Equation

---

### Exercise 1: Manual Closed-Form Solution via the Normal Equation

**Problem:**  
You are given a toy training set of $N = 3$ samples with a single scalar feature $x$ and target $y$:
$$(x^{(1)}, y^{(1)}) = (1, 2), \quad (x^{(2)}, y^{(2)}) = (2, 5), \quad (x^{(3)}, y^{(3)}) = (3, 7)$$

1. Construct the augmented design matrix $X \in \mathbb{R}^{3 \times 2}$ (with a leading column of 1s for the bias) and target vector $y \in \mathbb{R}^3$.
2. Compute the matrix product $X^T X$ and vector $X^T y$.
3. Invert $X^T X$ using the $2 \times 2$ matrix inversion formula:
   $$\begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$
4. Calculate the optimal parameter vector $w^* = [w_0, w_1]^T = (X^T X)^{-1} X^T y$.
5. Write the final prediction equation $\hat{y}(x)$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Construct $X$ and $y$**
$$X = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix}, \quad y = \begin{bmatrix} 2 \\ 5 \\ 7 \end{bmatrix}$$

**Step 2: Compute $X^T X$ and $X^T y$**
$$X^T X = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix} = \begin{bmatrix} 1+1+1 & 1+2+3 \\ 1+2+3 & 1^2+2^2+3^2 \end{bmatrix} = \begin{bmatrix} 3 & 6 \\ 6 & 14 \end{bmatrix}$$

$$X^T y = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 2 \\ 5 \\ 7 \end{bmatrix} = \begin{bmatrix} 2 + 5 + 7 \\ 1(2) + 2(5) + 3(7) \end{bmatrix} = \begin{bmatrix} 14 \\ 2 + 10 + 21 \end{bmatrix} = \begin{bmatrix} 14 \\ 33 \end{bmatrix}$$

**Step 3: Invert $X^T X$**
$$\det(X^T X) = (3)(14) - (6)(6) = 42 - 36 = 6$$
$$(X^T X)^{-1} = \frac{1}{6} \begin{bmatrix} 14 & -6 \\ -6 & 3 \end{bmatrix} = \begin{bmatrix} 7/3 & -1 \\ -1 & 1/2 \end{bmatrix}$$

**Step 4: Solve for $w^*$**
$$w^* = (X^T X)^{-1} X^T y = \begin{bmatrix} 7/3 & -1 \\ -1 & 1/2 \end{bmatrix} \begin{bmatrix} 14 \\ 33 \end{bmatrix} = \begin{bmatrix} \frac{98}{3} - 33 \\ -14 + \frac{33}{2} \end{bmatrix} = \begin{bmatrix} \frac{98 - 99}{3} \\ \frac{-28 + 33}{2} \end{bmatrix} = \begin{bmatrix} -1/3 \\ 5/2 \end{bmatrix} = \begin{bmatrix} -0.3333 \\ 2.5 \end{bmatrix}$$

**Step 5: Prediction equation**
$$\hat{y}(x) = -\frac{1}{3} + \frac{5}{2} x = 2.5x - 0.3333$$

Check predictions:
- $x=1 \implies \hat{y} = 2.5 - 0.333 = 2.167$ (Actual: 2)
- $x=2 \implies \hat{y} = 5.0 - 0.333 = 4.667$ (Actual: 5)
- $x=3 \implies \hat{y} = 7.5 - 0.333 = 7.167$ (Actual: 7)

$\blacksquare$
</details>

---

### Exercise 2: Manual Gradient Descent Step

**Problem:**  
Using the exact same dataset from Exercise 1:
$$X = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix}, \quad y = \begin{bmatrix} 2 \\ 5 \\ 7 \end{bmatrix}$$
Initialize the parameters at the origin: $w_0 = [0, 0]^T$.  
Let the learning rate be $\eta = 0.05$.

1. Compute the initial model predictions $\hat{y}_0 = X w_0$.
2. Compute the residual error vector $e_0 = \hat{y}_0 - y$.
3. Compute the gradient vector $\nabla_w J(w_0) = \frac{1}{N} X^T (\hat{y}_0 - y)$.
4. Perform one gradient descent update to calculate $w_1$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute initial predictions**
$$\hat{y}_0 = X \begin{bmatrix} 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$$

**Step 2: Compute residual error vector**
$$e_0 = \hat{y}_0 - y = \begin{bmatrix} 0 - 2 \\ 0 - 5 \\ 0 - 7 \end{bmatrix} = \begin{bmatrix} -2 \\ -5 \\ -7 \end{bmatrix}$$

**Step 3: Compute gradient vector**
$$\nabla_w J(w_0) = \frac{1}{3} X^T e_0 = \frac{1}{3} \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} -2 \\ -5 \\ -7 \end{bmatrix} = \frac{1}{3} \begin{bmatrix} -14 \\ -33 \end{bmatrix} = \begin{bmatrix} -4.6667 \\ -11.0 \end{bmatrix}$$

**Step 4: Update parameters**
$$w_1 = w_0 - \eta \nabla_w J(w_0) = \begin{bmatrix} 0 \\ 0 \end{bmatrix} - 0.05 \begin{bmatrix} -4.6667 \\ -11.0 \end{bmatrix} = \begin{bmatrix} +0.2333 \\ +0.5500 \end{bmatrix}$$

Notice that both parameters immediately step in the positive direction, heading toward the optimal values $w^* = [-0.333, 2.5]^T$.

$\blacksquare$
</details>

---

### Exercise 3: Mathematical Proof of Orthogonality and Zero-Mean Residuals

**Problem:**  
Let $w^* = (X^T X)^{-1} X^T y$ be the optimal OLS solution.  
1. Prove that the residual error vector $e = y - X w^*$ is strictly orthogonal to every column of $X$, i.e., $X^T e = \mathbf{0}$.
2. If the model includes an explicit intercept (meaning the first column of $X$ is a vector of ones, $\mathbf{1} = [1, 1, \dots, 1]^T$), prove that the sum of residuals is identically zero: $\sum_{i=1}^N e_i = 0$.

<details>
<summary>View Complete Solution</summary>

**Proof of Part 1:**
Substitute $w^*$ into the residual equation:
$$e = y - X w^* = y - X (X^T X)^{-1} X^T y$$

Now multiply from the left by $X^T$:
$$X^T e = X^T \left( y - X (X^T X)^{-1} X^T y \right) = X^T y - \underbrace{(X^T X) (X^T X)^{-1}}_{I} X^T y = X^T y - X^T y = \mathbf{0}$$
Thus, $X^T e = \mathbf{0}$, proving that the residual vector is strictly orthogonal to every feature vector in the design matrix.

**Proof of Part 2:**
Let the columns of $X$ be denoted by $[x_0, x_1, \dots, x_d]$, where $x_0 = \mathbf{1} = [1, 1, \dots, 1]^T$.  
Since $X^T e = \mathbf{0}$, the dot product of any column of $X$ with $e$ is zero:
$$x_j^T e = 0 \quad \text{for all } j \in \{0, 1, \dots, d\}$$

Specifically, for $j = 0$:
$$x_0^T e = \mathbf{1}^T e = \sum_{i=1}^N 1 \cdot e_i = \sum_{i=1}^N e_i = 0$$
Hence, the sum (and arithmetic mean) of the residuals of an ordinary least squares regression with an intercept is guaranteed to be zero.

$\blacksquare$
</details>

---

### Exercise 4: Computational FLOPs Analysis: Normal Equation vs. Gradient Descent

**Problem:**  
You have a dataset with $N = 10^7$ training instances and $d = 1,000$ features.
- Multiplying $X^T X$ requires $\approx N d^2$ FLOPs.
- Inverting a $d \times d$ matrix requires $\approx \frac{1}{3} d^3$ FLOPs.
- One epoch of Gradient Descent across all $N$ samples requires $\approx 2 N d$ FLOPs.

1. Compute the total FLOPs required by the Normal Equation.
2. Compute the total FLOPs required to run 20 epochs of Gradient Descent.
3. Which method is computationally faster for this dataset, and by what factor?

<details>
<summary>View Complete Solution</summary>

**1. Normal Equation FLOPs:**
- Computing $X^T X$: $N \cdot d^2 = (10^7) \cdot (10^3)^2 = 10^7 \cdot 10^6 = 10^{13}$ FLOPs.
- Matrix inversion $(X^T X)^{-1}$: $\frac{1}{3} d^3 = \frac{1}{3} (10^3)^3 \approx 3.33 \times 10^8$ FLOPs.
- Total Normal Equation FLOPs $\approx 10^{13} + 3.33 \times 10^8 \approx 10^{13}$ FLOPs.

**2. Gradient Descent FLOPs (20 epochs):**
- Per epoch: $2 \cdot N \cdot d = 2 \cdot (10^7) \cdot (10^3) = 2 \times 10^{10}$ FLOPs.
- For 20 epochs: $20 \times (2 \times 10^{10}) = 4 \times 10^{11}$ FLOPs.

**3. Comparison:**
$$\frac{\text{Normal Equation FLOPs}}{\text{Gradient Descent FLOPs}} = \frac{10^{13}}{4 \times 10^{11}} = 25\times$$
Gradient descent is **25 times faster** in terms of computational operations, and consumes exponentially less peak RAM since it does not need to store the $1,000 \times 1,000$ product matrix.

$\blacksquare$
</details>
