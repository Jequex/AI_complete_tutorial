# Exercises & Detailed Solutions: Topic 2.4

Welcome to the problem set for **Topic 2.4: Eigenvalues and Eigenvectors**.

---

## 📝 Problem Set

### Problem 1: Full Eigendecomposition by Hand
Consider the symmetric $2 \times 2$ matrix:
$$B = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$

- **Part A:** Form $(B - \lambda I)$ and compute the characteristic polynomial $\det(B - \lambda I) = 0$.
- **Part B:** Solve for the eigenvalues $\lambda_1$ and $\lambda_2$.
- **Part C:** Find the corresponding eigenvectors $\mathbf{v}_1$ and $\mathbf{v}_2$.
- **Part D:** Verify that $\mathbf{v}_1$ and $\mathbf{v}_2$ are mutually orthogonal ($\mathbf{v}_1 \cdot \mathbf{v}_2 = 0$).

---

### Problem 2: Fast Matrix Powers via Diagonalization
Using your results from Problem 1:
- **Part A:** Construct the eigenvector matrix $Q$ and the diagonal eigenvalue matrix $\Lambda$ such that $B = Q \Lambda Q^{-1}$.
- **Part B:** Compute $Q^{-1}$.
- **Part C:** Compute $B^4$ using the diagonalization formula:
  $$B^4 = Q \Lambda^4 Q^{-1}$$
  Show every step of the matrix arithmetic.

---

### Problem 3: The Mathematics of RNN Gradient Explosion
In a simple Recurrent Neural Network (RNN), the hidden state evolves over time according to:
$$\mathbf{h}_t = W \mathbf{h}_{t-1}$$
Assume the initial hidden state $\mathbf{h}_0$ happens to align with an eigenvector $\mathbf{v}$ of $W$ with eigenvalue $\lambda$.

- **Part A:** Prove by induction that $\mathbf{h}_t = \lambda^t \mathbf{h}_0$.
- **Part B:** If $\lambda = 1.10$, compute the scaling factor after $t = 100$ time steps ($\lambda^{100}$). What happens to the signal?
- **Part C:** If $\lambda = 0.90$, compute the scaling factor after $t = 100$ time steps ($\lambda^{100}$). What happens to the signal?

---

# 💡 Complete Solutions & Walkthroughs

---

### Solution 1: Full Eigendecomposition by Hand

#### Part A: Characteristic Polynomial
$$B - \lambda I = \begin{bmatrix} 2 - \lambda & 1 \\ 1 & 2 - \lambda \end{bmatrix}$$
$$\det(B - \lambda I) = (2 - \lambda)(2 - \lambda) - (1 \times 1) = 0$$
$$(4 - 4\lambda + \lambda^2) - 1 = 0 \implies \lambda^2 - 4\lambda + 3 = 0$$

#### Part B: Eigenvalues
Factor the quadratic:
$$(\lambda - 3)(\lambda - 1) = 0$$
$$\mathbf{\lambda_1 = 3, \qquad \lambda_2 = 1}$$

---

#### Part C: Eigenvectors

1. **For $\lambda_1 = 3$:**
   $$(B - 3I)\mathbf{v}_1 = \begin{bmatrix} 2 - 3 & 1 \\ 1 & 2 - 3 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
   $$-v_1 + v_2 = 0 \implies v_1 = v_2$$
   $$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

2. **For $\lambda_2 = 1$:**
   $$(B - 1I)\mathbf{v}_2 = \begin{bmatrix} 2 - 1 & 1 \\ 1 & 2 - 1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
   $$v_1 + v_2 = 0 \implies v_2 = -v_1$$
   $$\mathbf{v}_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$$

---

#### Part D: Orthogonality Verification
Compute the dot product between $\mathbf{v}_1$ and $\mathbf{v}_2$:
$$\mathbf{v}_1 \cdot \mathbf{v}_2 = (1 \times 1) + (1 \times -1) = 1 - 1 = \mathbf{0}$$
The eigenvectors are **orthogonal**! *(The Spectral Theorem guarantees that any real symmetric matrix has mutually orthogonal eigenvectors).*

---

### Solution 2: Fast Matrix Powers via Diagonalization

#### Part A: Matrices $Q$ and $\Lambda$
$$Q = \begin{bmatrix} \mathbf{v}_1 & \mathbf{v}_2 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \qquad \Lambda = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}$$

#### Part B: Computing $Q^{-1}$
For a $2 \times 2$ matrix $M = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$, the inverse is:
$$M^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

For $Q$: $\det(Q) = (1)(-1) - (1)(1) = -1 - 1 = -2$.
$$Q^{-1} = \frac{1}{-2} \begin{bmatrix} -1 & -1 \\ -1 & 1 \end{bmatrix} = \begin{bmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{bmatrix} = \frac{1}{2} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$$

---

#### Part C: Computing $B^4 = Q \Lambda^4 Q^{-1}$
1. Calculate $\Lambda^4$:
   $$\Lambda^4 = \begin{bmatrix} 3^4 & 0 \\ 0 & 1^4 \end{bmatrix} = \begin{bmatrix} 81 & 0 \\ 0 & 1 \end{bmatrix}$$

2. Multiply $Q \Lambda^4$:
   $$Q \Lambda^4 = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} 81 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 81 & 1 \\ 81 & -1 \end{bmatrix}$$

3. Multiply by $Q^{-1} = \frac{1}{2} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$:
   $$B^4 = \frac{1}{2} \begin{bmatrix} 81 & 1 \\ 81 & -1 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$$
   $$\text{Entry (1,1): } \frac{1}{2} (81 \times 1 + 1 \times 1) = \frac{82}{2} = \mathbf{41}$$
   $$\text{Entry (1,2): } \frac{1}{2} (81 \times 1 + 1 \times -1) = \frac{80}{2} = \mathbf{40}$$
   $$\text{Entry (2,1): } \frac{1}{2} (81 \times 1 + -1 \times 1) = \frac{80}{2} = \mathbf{40}$$
   $$\text{Entry (2,2): } \frac{1}{2} (81 \times 1 + -1 \times -1) = \frac{82}{2} = \mathbf{41}$$

$$\mathbf{B^4 = \begin{bmatrix} 41 & 40 \\ 40 & 41 \end{bmatrix}}$$

---

### Solution 3: The Mathematics of RNN Gradient Stability

#### Part A: Proof by Induction
- **Base Case ($t=1$):**
  $$\mathbf{h}_1 = W \mathbf{h}_0 = W \mathbf{v} = \lambda \mathbf{v} = \lambda \mathbf{h}_0 \quad \text{✓ (Holds)}$$
- **Inductive Step:**
  Assume $\mathbf{h}_k = \lambda^k \mathbf{h}_0$. Then:
  $$\mathbf{h}_{k+1} = W \mathbf{h}_k = W (\lambda^k \mathbf{h}_0) = \lambda^k (W \mathbf{h}_0) = \lambda^k (\lambda \mathbf{h}_0) = \lambda^{k+1} \mathbf{h}_0 \quad \text{✓}$$
  Therefore, $\mathbf{h}_t = \lambda^t \mathbf{h}_0$ for all integers $t \ge 1$. $\blacksquare$

---

#### Part B: Behavior for $\lambda = 1.10$
$$\lambda^{100} = (1.10)^{100} \approx \mathbf{13,780.6}$$
The signal magnitude explodes by over **$13,780$ times**! In a neural network, this causes floating-point overflow (`NaN` or `inf`) and ruins training.

---

#### Part C: Behavior for $\lambda = 0.90$
$$\lambda^{100} = (0.90)^{100} \approx \mathbf{0.00002656} = 2.656 \times 10^{-5}$$
The signal is attenuated to near zero. The network completely "forgets" the past inputs, preventing it from learning long-term dependencies.

