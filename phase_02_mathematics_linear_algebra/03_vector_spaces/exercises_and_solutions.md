# Exercises & Detailed Solutions: Topic 2.3

Welcome to the problem set for **Topic 2.3: Vector Spaces, Span, Basis, and Rank**.

---

## 📝 Problem Set

### Problem 1: Testing Linear Independence & Redundancy
Consider the three vectors in $\mathbb{R}^3$:
$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \qquad \mathbf{v}_2 = \begin{bmatrix} 0 \\ 1 \\ 2 \end{bmatrix}, \qquad \mathbf{v}_3 = \begin{bmatrix} 2 \\ 5 \\ 8 \end{bmatrix}$$

- **Part A:** Set up the vector equation $c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + c_3 \mathbf{v}_3 = \mathbf{0}$ as a system of 3 scalar linear equations.
- **Part B:** Solve the system for scalars $c_1, c_2, c_3$. Are the vectors linearly independent or linearly dependent?
- **Part C:** What is the dimension of $\text{Span}(\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3)$? Describe its geometric shape in $\mathbb{R}^3$.

---

### Problem 2: The Rank-Nullity Theorem & Solution Uniqueness
Suppose a neural network layer is represented by a matrix $A \in \mathbb{R}^{5 \times 8}$ (5 outputs, 8 inputs).
You are told that:
$$\text{Rank}(A) = 4$$

- **Part A:** What is the dimension of the input space $n$?
- **Part B:** Use the Rank-Nullity Theorem to calculate $\text{Nullity}(A)$ (the dimension of the null space $\mathcal{N}(A)$).
- **Part C:** If there exists a vector $\mathbf{x}_0$ such that $A\mathbf{x}_0 = \mathbf{b}$, can this solution ever be unique? Explain using the null space.

---

### Problem 3: Low-Rank Adaptation (LoRA) Parameter Calculation
In a Transformer foundation model (like Llama-3 or GPT-4), the query projection matrix $W_Q$ has input dimension $d = 4096$ and output dimension $d = 4096$.
- During standard full fine-tuning, the update matrix $\Delta W \in \mathbb{R}^{4096 \times 4096}$ is trained directly.
- During LoRA fine-tuning, $\Delta W$ is decomposed as $\Delta W = B \cdot A$, where $B \in \mathbb{R}^{4096 \times r}$ and $A \in \mathbb{R}^{r \times 4096}$ with rank $r = 8$.

- **Part A:** How many trainable parameters are updated in full fine-tuning?
- **Part B:** How many trainable parameters are updated in LoRA with rank $r = 8$?
- **Part C:** What is the exact percentage reduction in trainable parameters?

---

# 💡 Complete Solutions & Walkthroughs

---

### Solution 1: Testing Linear Independence

#### Part A: Setting Up the System
$$c_1 \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} + c_2 \begin{bmatrix} 0 \\ 1 \\ 2 \end{bmatrix} + c_3 \begin{bmatrix} 2 \\ 5 \\ 8 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$$

Expanding row-by-row into 3 scalar equations:
1. $c_1 + 0 c_2 + 2 c_3 = 0 \implies \mathbf{c_1 = -2c_3}$
2. $2c_1 + c_2 + 5c_3 = 0$
3. $3c_1 + 2c_2 + 8c_3 = 0$

---

#### Part B: Solving for Scalars
Substitute $c_1 = -2c_3$ into equation (2):
$$2(-2c_3) + c_2 + 5c_3 = 0$$
$$-4c_3 + c_2 + 5c_3 = 0 \implies c_2 + c_3 = 0 \implies \mathbf{c_2 = -c_3}$$

Now check equation (3) with $c_1 = -2c_3$ and $c_2 = -c_3$:
$$3(-2c_3) + 2(-c_3) + 8c_3 = -6c_3 - 2c_3 + 8c_3 = 0 \quad (\text{Consistent!})$$

Let $c_3 = 1$. Then:
$$c_1 = -2, \quad c_2 = -1, \quad c_3 = 1$$

Test:
$$-2 \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} - 1 \begin{bmatrix} 0 \\ 1 \\ 2 \end{bmatrix} + 1 \begin{bmatrix} 2 \\ 5 \\ 8 \end{bmatrix} = \begin{bmatrix} -2 + 0 + 2 \\ -4 - 1 + 5 \\ -6 - 2 + 8 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$$

Because there exists a non-zero solution $(c_1, c_2, c_3) = (-2, -1, 1) \neq (0, 0, 0)$, the vectors are **Linearly Dependent (Redundant)**! In fact, $\mathbf{v}_3 = 2\mathbf{v}_1 + \mathbf{v}_2$.

---

#### Part C: Dimension of the Span
Since $\mathbf{v}_3$ is a linear combination of $\mathbf{v}_1$ and $\mathbf{v}_2$, it adds no new dimension. Vectors $\mathbf{v}_1$ and $\mathbf{v}_2$ are not multiples of each other, so they form a basis for the span.
- **Dimension:** $\mathbf{2}$
- **Geometric Shape:** A flat **2-dimensional plane** slicing through the origin $(0, 0, 0)$ inside 3D space.

---

### Solution 2: The Rank-Nullity Theorem & Solution Uniqueness

#### Part A: Input Space Dimension
For matrix $A \in \mathbb{R}^{m \times n} = \mathbb{R}^{5 \times 8}$, $n$ is the number of columns (inputs).
Thus, $n = \mathbf{8}$.

#### Part B: Calculating Nullity
From the Rank-Nullity Theorem:
$$\text{Rank}(A) + \text{Nullity}(A) = n$$
$$4 + \text{Nullity}(A) = 8 \implies \mathbf{\text{Nullity}(A) = 4}$$
The null space $\mathcal{N}(A)$ is a **4-dimensional subspace** of the 8D input space.

#### Part C: Uniqueness of Solutions
**The solution can never be unique!**
If $A\mathbf{x}_0 = \mathbf{b}$, take any non-zero vector $\mathbf{z} \in \mathcal{N}(A)$ (which satisfies $A\mathbf{z} = \mathbf{0}$).
Then:
$$A(\mathbf{x}_0 + \mathbf{z}) = A\mathbf{x}_0 + A\mathbf{z} = \mathbf{b} + \mathbf{0} = \mathbf{b}$$
Since the null space has dimension 4, there are infinitely many such vectors $\mathbf{z}$. Hence, there exist **infinitely many solutions** to $A\mathbf{x} = \mathbf{b}$.

---

### Solution 3: LoRA Parameter Calculation

#### Part A: Full Fine-Tuning Parameters
$$N_{\text{full}} = 4096 \times 4096 = \mathbf{16,777,216} \text{ parameters}$$

#### Part B: LoRA Parameters ($r = 8$)
Matrix $B$ has size $4096 \times 8$:
$$N_B = 4096 \times 8 = 32,768$$
Matrix $A$ has size $8 \times 4096$:
$$N_A = 8 \times 4096 = 32,768$$
$$N_{\text{LoRA}} = N_B + N_A = 32,768 + 32,768 = \mathbf{65,536} \text{ parameters}$$

#### Part C: Percentage Reduction
$$\text{Reduction Ratio} = 1 - \frac{65,536}{16,777,216} = 1 - 0.00390625 = 0.99609375 = \mathbf{99.61\%}$$
LoRA reduces the trainable parameters by **$99.61\%$**, allowing billion-parameter models to be fine-tuned on consumer GPUs!

