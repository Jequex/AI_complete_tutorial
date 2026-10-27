# Topic 2.3: Exercises & Solutions — Vector Spaces, Span, Basis, and Rank

Welcome to the problem set for **Topic 2.3: Vector Spaces, Span, Basis, and Rank**!

Test your understanding of linear independence, the Rank-Nullity Theorem, and LoRA parameter reduction. Work through each exercise before opening the collapsible solution.

---

## 📝 Problem Set

### Problem 1: Testing Linear Independence & Redundancy

Consider the three vectors in $\mathbb{R}^3$:
$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \qquad \mathbf{v}_2 = \begin{bmatrix} 0 \\ 1 \\ 2 \end{bmatrix}, \qquad \mathbf{v}_3 = \begin{bmatrix} 2 \\ 5 \\ 8 \end{bmatrix}$$

1. Set up the vector equation $c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + c_3 \mathbf{v}_3 = \mathbf{0}$ as a system of scalar equations.
2. Solve the system for $c_1, c_2, c_3$. Are the vectors linearly independent or linearly dependent?
3. What is the dimension of $\text{Span}(\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3)$? Describe its geometric shape in $\mathbb{R}^3$.

<details>
<summary>Click to view solution for Problem 1</summary>

### Solution 1: Testing Linear Independence

#### Part 1: Setting up the Linear System
$$c_1 \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} + c_2 \begin{bmatrix} 0 \\ 1 \\ 2 \end{bmatrix} + c_3 \begin{bmatrix} 2 \\ 5 \\ 8 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$$
Row-by-row:
1. $c_1 + 2c_3 = 0 \implies c_1 = -2c_3$
2. $2c_1 + c_2 + 5c_3 = 0$
3. $3c_1 + 2c_2 + 8c_3 = 0$

#### Part 2: Solving the System
Substitute $c_1 = -2c_3$ into (2):
$$2(-2c_3) + c_2 + 5c_3 = 0 \implies -4c_3 + c_2 + 5c_3 = 0 \implies c_2 = -c_3$$
Check in (3):
$$3(-2c_3) + 2(-c_3) + 8c_3 = -6c_3 - 2c_3 + 8c_3 = 0 \quad \text{(Satisfied!)}$$
Letting $c_3 = 1$ gives non-zero solution:
$$c_1 = -2, \quad c_2 = -1, \quad c_3 = 1$$
$$-2 \mathbf{v}_1 - 1 \mathbf{v}_2 + \mathbf{v}_3 = \mathbf{0} \implies \mathbf{v}_3 = 2\mathbf{v}_1 + \mathbf{v}_2$$
**The vectors are Linearly Dependent (redundant)!**

#### Part 3: Dimension and Geometry
Since $\mathbf{v}_3$ is a linear combination of $\mathbf{v}_1$ and $\mathbf{v}_2$, it adds zero new dimensions.
- Number of independent vectors: $2$.
- Dimension of the Span: **$2$**.
- Geometric shape: An infinite **2D flat plane** slicing through the origin in $\mathbb{R}^3$.
</details>

---

### Problem 2: The Rank-Nullity Theorem & Solution Uniqueness

A neural network layer is represented by matrix $A \in \mathbb{R}^{5 \times 8}$ ($5$ outputs, $8$ inputs). You are given that $\text{Rank}(A) = 4$.

1. What is the dimension of the input space $n$?
2. Use the Rank-Nullity Theorem to calculate $\text{Nullity}(A)$.
3. If $A \mathbf{x}_0 = \mathbf{b}$ has a solution, can this solution ever be unique? Explain.

<details>
<summary>Click to view solution for Problem 2</summary>

### Solution 2: Rank-Nullity Theorem & Solution Uniqueness

#### Part 1: Input Space Dimension
Matrix $A$ has $8$ columns, so input space dimension $n = \mathbf{8}$.

#### Part 2: Computing Nullity
From the Rank-Nullity Theorem:
$$\text{Rank}(A) + \text{Nullity}(A) = n$$
$$4 + \text{Nullity}(A) = 8 \implies \text{Nullity}(A) = \mathbf{4}$$
*Meaning:* The null space $\mathcal{N}(A)$ is a $4$-dimensional subspace squashed straight to zero!

#### Part 3: Uniqueness of Solutions
**No, the solution can NEVER be unique!**
For any non-zero vector $\mathbf{z} \in \mathcal{N}(A)$ (where $A\mathbf{z} = \mathbf{0}$):
$$A(\mathbf{x}_0 + \mathbf{z}) = A\mathbf{x}_0 + A\mathbf{z} = \mathbf{b} + \mathbf{0} = \mathbf{b}$$
Because the null space has dimension $4$, there exists an entire 4-dimensional infinite family of solutions!
</details>

---

### Problem 3: Low-Rank Adaptation (LoRA) Parameter Calculation

In a Transformer model, weight matrix $W \in \mathbb{R}^{4096 \times 4096}$ is fine-tuned.
- Full fine-tuning trains all entries of $\Delta W \in \mathbb{R}^{4096 \times 4096}$.
- LoRA factors the update as $\Delta W = B \cdot A$ with rank $r = 8$, where $B \in \mathbb{R}^{4096 \times 8}$ and $A \in \mathbb{R}^{8 \times 4096}$.

1. How many parameters are trained in full fine-tuning?
2. How many parameters are trained in LoRA ($r=8$)?
3. What is the exact percentage reduction in parameters?

<details>
<summary>Click to view solution for Problem 3</summary>

### Solution 3: LoRA Parameter Reduction

#### Part 1: Full Fine-Tuning Parameters
$$\text{Full Parameters} = 4096 \times 4096 = \mathbf{16,777,216\text{ parameters}}$$

#### Part 2: LoRA Parameters ($r = 8$)
$$\text{LoRA Parameters} = (4096 \times 8) + (8 \times 4096) = 32,768 + 32,768 = \mathbf{65,536\text{ parameters}}$$

#### Part 3: Percentage Reduction
$$\text{Reduction} = \left( 1 - \frac{65,536}{16,777,216} \right) \times 100\% = \left( 1 - \frac{1}{256} \right) \times 100\% = \left( 1 - 0.00390625 \right) \times 100\% = \mathbf{99.609\%}$$
*Conclusion:* LoRA eliminates over **$99.6\%$** of trainable parameters!
</details>
