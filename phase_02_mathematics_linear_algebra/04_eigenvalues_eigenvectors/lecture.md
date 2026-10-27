# Topic 2.4: Eigenvalues and Eigenvectors

When a matrix multiplies a vector, it almost always does two things:
1. It **changes its length** (stretches or shrinks it).
2. It **knocks it off its direction** (rotates it).

$$\mathbf{v} \longrightarrow A\mathbf{v} \quad (\text{new length AND new direction})$$

However, for almost every square matrix, there exist a few magical, special arrows that **refuse to rotate**. When matrix $A$ hits these special arrows, they remain on their exact same line—they only get scaled by a factor $\lambda$!

These special arrows are called **Eigenvectors**, and their stretch factors are called **Eigenvalues**:

$$\boxed{A \mathbf{v} = \lambda \mathbf{v}}$$

---

## 1. 🧒 The 12-Year-Old Analogy: The Spinning Globe & The Wind Tunnel

### The Spinning Globe
Imagine you spin a desktop globe of the Earth as hard as you can:
- London, New York, Tokyo, and Nairobi all whip around in fast circles, changing their spatial direction every millisecond.
- But look at the **North Pole** and the **South Pole**.
- They are sitting right on the axis of rotation. They **never leave their line**!
- In linear algebra, that axis is an **Eigenvector**.

```text
               MOST VECTORS (Rotated)                   EIGENVECTOR (Only Scaled!)
               
                  y                                        y
                  ▲       Av                               ▲
                  │      ●                                 │         Av = λ v
                  │     /                                  │        ●
                  │    /                                   │       /
                  │   ● v                                  │      ● v
                  │  /                                     │     /
                  └──┴──────► x                            └────┴──────► x
               Rotated off its line!                    Stays on its exact line!
```

---

## 2. Deconstructing the Master Equation: $A\mathbf{v} = \lambda\mathbf{v}$

Let us break down every term:

$$\underbrace{A}_{\substack{\text{Square Transformation} \\ \text{Matrix }(n \times n)}} \times \underbrace{\mathbf{v}}_{\substack{\text{Eigenvector} \\ (\mathbf{v} \neq \mathbf{0})}} = \underbrace{\lambda}_{\substack{\text{Eigenvalue} \\ \text{(Scalar stretch factor)}}} \times \underbrace{\mathbf{v}}_{\substack{\text{Original} \\ \text{Eigenvector}}}$$

**🧒 What the math is actually doing:**  
Transforming arrow $\mathbf{v}$ by matrix $A$ produces the exact same result as simply multiplying arrow $\mathbf{v}$ by a single regular number $\lambda$!

- If $\lambda = 2$: The vector doubles in length along its original line.
- If $\lambda = 0.5$: The vector shrinks in half.
- If $\lambda = -1$: The vector flips in the exact opposite direction.
- If $\lambda = 0$: The vector is squashed to complete zero (meaning $\mathbf{v}$ belongs to the Null Space $\mathcal{N}(A)$!).

---

## 3. How to Find Eigenvalues: The Characteristic Polynomial

How do we solve $A\mathbf{v} = \lambda\mathbf{v}$ when both $\lambda$ and $\mathbf{v}$ are unknown?

### Step 1: Move Everything to One Side
$$A\mathbf{v} - \lambda\mathbf{v} = \mathbf{0}$$
Insert the identity matrix $I$ so we can factor out vector $\mathbf{v}$:
$$(A - \lambda I)\mathbf{v} = \mathbf{0}$$

### Step 2: The Singularity Requirement
We need a non-zero vector $\mathbf{v} \neq \mathbf{0}$ to satisfy $(A - \lambda I)\mathbf{v} = \mathbf{0}$.
From Topic 2.2, a matrix can squash a non-zero vector to zero **if and only if its determinant is ZERO**!

$$\boxed{\det(A - \lambda I) = 0}$$

$$\underbrace{\det(A - \lambda I) = 0}_{\text{The Characteristic Equation}}$$

---

## 4. Complete Numerical Walkthrough by Hand

Let us find the eigenvalues and eigenvectors for matrix $A$:
$$A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$$

### Step 1: Form $(A - \lambda I)$
$$A - \lambda I = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix} - \begin{bmatrix} \lambda & 0 \\ 0 & \lambda \end{bmatrix} = \begin{bmatrix} 4 - \lambda & 1 \\ 2 & 3 - \lambda \end{bmatrix}$$

### Step 2: Compute Determinant & Solve for $\lambda$
$$\det(A - \lambda I) = (4 - \lambda)(3 - \lambda) - (1 \times 2) = 0$$
$$\lambda^2 - 7\lambda + 12 - 2 = 0$$
$$\lambda^2 - 7\lambda + 10 = 0$$

Factor the quadratic equation:
$$(\lambda - 5)(\lambda - 2) = 0$$
Our two eigenvalues are:
$$\boxed{\lambda_1 = 5, \qquad \lambda_2 = 2}$$

---

### Step 3: Find Eigenvector for $\lambda_1 = 5$
Plug $\lambda_1 = 5$ into $(A - 5I)\mathbf{v} = \mathbf{0}$:
$$\begin{bmatrix} 4 - 5 & 1 \\ 2 & 3 - 5 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} -1 & 1 \\ 2 & -2 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
$$-v_1 + v_2 = 0 \implies v_1 = v_2$$
Choose $v_1 = 1 \implies v_2 = 1$:
$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

**Verify:**
$$A \mathbf{v}_1 = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 5 \\ 5 \end{bmatrix} = 5 \begin{bmatrix} 1 \\ 1 \end{bmatrix} = 5 \mathbf{v}_1 \quad \text{✅ (Matches!)}$$

---

### Step 4: Find Eigenvector for $\lambda_2 = 2$
Plug $\lambda_2 = 2$ into $(A - 2I)\mathbf{v} = \mathbf{0}$:
$$\begin{bmatrix} 4 - 2 & 1 \\ 2 & 3 - 2 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 2 & 1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
$$2v_1 + v_2 = 0 \implies v_2 = -2v_1$$
Choose $v_1 = 1 \implies v_2 = -2$:
$$\mathbf{v}_2 = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$$

**Verify:**
$$A \mathbf{v}_2 = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix} \begin{bmatrix} 1 \\ -2 \end{bmatrix} = \begin{bmatrix} 2 \\ -4 \end{bmatrix} = 2 \begin{bmatrix} 1 \\ -2 \end{bmatrix} = 2 \mathbf{v}_2 \quad \text{✅ (Matches!)}$$

---

## 5. Diagonalization: The Ultimate Shortcut ($A = Q \Lambda Q^{-1}$)

Stack the eigenvectors into the columns of matrix $Q$:
$$Q = \begin{bmatrix} 1 & 1 \\ 1 & -2 \end{bmatrix}$$
Place eigenvalues along the diagonal of $\Lambda$:
$$\Lambda = \begin{bmatrix} 5 & 0 \\ 0 & 2 \end{bmatrix}$$

Then:
$$\boxed{A = Q \Lambda Q^{-1}}$$

### Computing Matrix Powers ($A^{100}$)
When training deep neural networks or running Recurrent Neural Networks (RNNs) for 100 time steps:
$$A^{100} = (Q \Lambda Q^{-1})(Q \Lambda Q^{-1}) \dots (Q \Lambda Q^{-1}) = Q \Lambda^{100} Q^{-1}$$

$$\Lambda^{100} = \begin{bmatrix} 5^{100} & 0 \\ 0 & 2^{100} \end{bmatrix}$$

---

## 6. Why Eigenvalues Matter for AI: Exploding & Vanishing Gradients

In deep networks, signals pass through repeated matrix multiplications:
$$\mathbf{h}_T \approx W^T \mathbf{h}_0$$
- If the largest eigenvalue $|\lambda_{\max}| > 1$: Signal magnitudes **explode exponentially** to infinity ($1.1^{100} \approx 13,780$).
- If the largest eigenvalue $|\lambda_{\max}| < 1$: Signal magnitudes **vanish exponentially** to zero ($0.9^{100} \approx 0.000026$).
- This insight led to **Spectral Normalization**, **orthogonal initialization**, and **residual connections (ResNets)**!

---

## 7. 📋 Mathematical & Technical Symbol Decoder Table

| Symbol | Pronunciation | Plain English Meaning | Concrete Real-World Analogy / Flashcard Example |
| :---: | :---: | :--- | :--- |
| $A\mathbf{v} = \lambda\mathbf{v}$ | "A v equals lambda v" | The master eigenvalue equation | An arrow that stays on its exact line under a matrix transformation |
| $\lambda$ | "lambda" | The scalar stretch factor along the eigenvector direction | The zoom factor along the axis of rotation |
| $\mathbf{v}$ | "eigenvector v" | The special direction that refuses to rotate | The North Pole on a spinning globe |
| $\det(A - \lambda I) = 0$ | "determinant of A minus lambda I" | Characteristic equation whose roots are eigenvalues | The secret recipe that unlocks the stretch factors |
| $Q \Lambda Q^{-1}$ | "Q Lambda Q inverse" | Eigendecomposition / Diagonalization of a square matrix | Rotate coordinates to align with eigenvectors, stretch, and rotate back |
| $\Lambda$ | "capital Lambda" | Diagonal matrix containing eigenvalues | A clean table where each axis scales independently |
| $|\lambda_{\max}|$ | "lambda max (spectral radius)" | The largest eigenvalue magnitude of a matrix | The maximum speed multiplier of a neural signal |

---

## 8. 📌 Summary & Mathematical Checklist

- [x] **Eigenvectors ($\mathbf{v}$)** are directions that do not rotate under matrix $A$.
- [x] **Eigenvalues ($\lambda$)** are the scalar stretch multipliers along those invariant axes: $A\mathbf{v} = \lambda\mathbf{v}$.
- [x] Solve for eigenvalues using the **Characteristic Equation**: $\det(A - \lambda I) = 0$.
- [x] **Diagonalization ($A = Q \Lambda Q^{-1}$)** simplifies matrix powers to $A^k = Q \Lambda^k Q^{-1}$.
- [x] If $|\lambda_{\max}| > 1$, neural signals explode; if $|\lambda_{\max}| < 1$, signals vanish.

---

Next up: **Run `python3 experiment.py`** in this folder to observe the Power Iteration algorithm converge to the true dominant eigenvalue!
