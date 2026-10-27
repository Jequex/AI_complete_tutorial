# Topic 2.4: Exercises & Solutions — Eigenvalues and Eigenvectors

Welcome to the problem set for **Topic 2.4: Eigenvalues and Eigenvectors**!

Work through each problem carefully before clicking to reveal the complete step-by-step arithmetic and proof breakdown.

---

## 📝 Problem Set

### Problem 1: Full Eigendecomposition by Hand

Consider the symmetric $2 \times 2$ matrix:
$$B = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$

1. Form $(B - \lambda I)$ and compute the characteristic equation $\det(B - \lambda I) = 0$.
2. Solve for eigenvalues $\lambda_1$ and $\lambda_2$.
3. Find the corresponding eigenvectors $\mathbf{v}_1$ and $\mathbf{v}_2$.
4. Verify that $\mathbf{v}_1$ and $\mathbf{v}_2$ are mutually orthogonal ($\mathbf{v}_1 \cdot \mathbf{v}_2 = 0$).

<details>
<summary>Click to view solution for Problem 1</summary>

### Solution 1: Full Eigendecomposition by Hand

#### Part 1: Characteristic Equation
$$B - \lambda I = \begin{bmatrix} 2 - \lambda & 1 \\ 1 & 2 - \lambda \end{bmatrix}$$
$$\det(B - \lambda I) = (2 - \lambda)^2 - 1 = 0 \implies \lambda^2 - 4\lambda + 4 - 1 = 0$$
$$\lambda^2 - 4\lambda + 3 = 0$$

#### Part 2: Eigenvalues
Factor the quadratic equation:
$$(\lambda - 3)(\lambda - 1) = 0 \implies \mathbf{\lambda_1 = 3, \quad \lambda_2 = 1}$$

#### Part 3: Eigenvectors
- **For $\lambda_1 = 3$:**
  $$(B - 3I)\mathbf{v} = \begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \implies -v_1 + v_2 = 0 \implies \mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$
- **For $\lambda_2 = 1$:**
  $$(B - 1I)\mathbf{v} = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \implies v_1 + v_2 = 0 \implies \mathbf{v}_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$$

#### Part 4: Orthogonality Verification
$$\mathbf{v}_1 \cdot \mathbf{v}_2 = (1 \times 1) + (1 \times -1) = 1 - 1 = \mathbf{0}$$
*Theorem Confirmed:* Symmetric real matrices always produce mutually orthogonal eigenvectors!
</details>

---

### Problem 2: Fast Matrix Powers via Diagonalization

Using $B = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ from Problem 1:
1. Construct eigenvector matrix $Q$ and diagonal matrix $\Lambda$.
2. Compute $Q^{-1}$.
3. Calculate $B^4 = Q \Lambda^4 Q^{-1}$ step-by-step.

<details>
<summary>Click to view solution for Problem 2</summary>

### Solution 2: Fast Matrix Powers via Diagonalization

#### Part 1: Constructing $Q$ and $\Lambda$
$$Q = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \qquad \Lambda = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}$$

#### Part 2: Computing $Q^{-1}$
For $2 \times 2$ matrix $Q = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$, $Q^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$:
$$\det(Q) = (1)(-1) - (1)(1) = -2$$
$$Q^{-1} = -\frac{1}{2} \begin{bmatrix} -1 & -1 \\ -1 & 1 \end{bmatrix} = \begin{bmatrix} 0.5 & 0.5 \\ 0.5 & -0.5 \end{bmatrix}$$

#### Part 3: Computing $B^4$
$$\Lambda^4 = \begin{bmatrix} 3^4 & 0 \\ 0 & 1^4 \end{bmatrix} = \begin{bmatrix} 81 & 0 \\ 0 & 1 \end{bmatrix}$$
$$Q \Lambda^4 = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} 81 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 81 & 1 \\ 81 & -1 \end{bmatrix}$$
$$B^4 = (Q \Lambda^4) Q^{-1} = \begin{bmatrix} 81 & 1 \\ 81 & -1 \end{bmatrix} \begin{bmatrix} 0.5 & 0.5 \\ 0.5 & -0.5 \end{bmatrix}$$
- Row 1, Col 1: $(81 \times 0.5) + (1 \times 0.5) = 40.5 + 0.5 = \mathbf{41}$
- Row 1, Col 2: $(81 \times 0.5) + (1 \times -0.5) = 40.5 - 0.5 = \mathbf{40}$
- Row 2, Col 1: $(81 \times 0.5) + (-1 \times 0.5) = 40.5 - 0.5 = \mathbf{40}$
- Row 2, Col 2: $(81 \times 0.5) + (-1 \times -0.5) = 40.5 + 0.5 = \mathbf{41}$
$$B^4 = \begin{bmatrix} 41 & 40 \\ 40 & 41 \end{bmatrix}$$
</details>

---

### Problem 3: The Mathematics of RNN Gradient Explosion & Collapse

In an RNN, hidden state evolves as $\mathbf{h}_t = W \mathbf{h}_{t-1}$. Assume $\mathbf{h}_0$ aligns with an eigenvector $\mathbf{v}$ with eigenvalue $\lambda$.

1. Prove by induction that $\mathbf{h}_t = \lambda^t \mathbf{h}_0$.
2. For $\lambda = 1.10$, compute $\lambda^{100}$. What happens to the signal?
3. For $\lambda = 0.90$, compute $\lambda^{100}$. What happens to the signal?

<details>
<summary>Click to view solution for Problem 3</summary>

### Solution 3: RNN Gradient Dynamics

#### Part 1: Proof by Induction
- **Base case ($t = 1$):** $\mathbf{h}_1 = W \mathbf{h}_0 = W \mathbf{v} = \lambda \mathbf{v} = \lambda^1 \mathbf{h}_0$.
- **Inductive step:** Assume $\mathbf{h}_k = \lambda^k \mathbf{h}_0$. Then:
  $$\mathbf{h}_{k+1} = W \mathbf{h}_k = W (\lambda^k \mathbf{h}_0) = \lambda^k (W \mathbf{h}_0) = \lambda^k (\lambda \mathbf{h}_0) = \lambda^{k+1} \mathbf{h}_0 \quad \blacksquare$$

#### Part 2: Exploding Signal ($\lambda = 1.10$)
$$\lambda^{100} = (1.10)^{100} \approx \mathbf{13,780.6}$$
*Result:* The signal explodes exponentially by over 13,000x, causing gradient saturation and NaN overflow!

#### Part 3: Vanishing Signal ($\lambda = 0.90$)
$$\lambda^{100} = (0.90)^{100} \approx \mathbf{0.00002656}$$
*Result:* The signal collapses to virtually zero ($0.0026\%$), destroying long-term memory!
</details>
