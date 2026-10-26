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

Let's dissect every character:
* $A \in \mathbb{R}^{n \times n}$: A square transformation matrix.
* $\mathbf{v} \in \mathbb{R}^n$: The **Eigenvector** (must be non-zero: $\mathbf{v} \neq \mathbf{0}$).
* $\lambda \in \mathbb{R}$: The **Eigenvalue** (pronounced *"lambda"*). A scalar stretch factor.
  * If $\lambda = 2$: The vector doubles in length in the same direction.
  * If $\lambda = 0.5$: The vector shrinks in half.
  * If $\lambda = -1$: The vector flips in the exact opposite direction.
  * If $\lambda = 0$: The vector is squashed to complete zero (meaning $\mathbf{v}$ lives in the Null Space!).

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

This equation is called the **Characteristic Equation**. Expanding the determinant produces an $n$-th degree polynomial in $\lambda$ whose roots are the eigenvalues!

---

## 4. Complete Step-by-Step Calculation by Hand

Let:
$$A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$$

### Step 1: Form $(A - \lambda I)$
$$A - \lambda I = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix} - \begin{bmatrix} \lambda & 0 \\ 0 & \lambda \end{bmatrix} = \begin{bmatrix} 4 - \lambda & 1 \\ 2 & 3 - \lambda \end{bmatrix}$$

### Step 2: Compute the Determinant & Set to 0
$$\det(A - \lambda I) = (4 - \lambda)(3 - \lambda) - (1 \times 2) = 0$$
$$\lambda^2 - 7\lambda + 12 - 2 = 0$$
$$\lambda^2 - 7\lambda + 10 = 0$$

Factor the quadratic equation:
$$(\lambda - 5)(\lambda - 2) = 0$$
We have found our two eigenvalues:
$$\boxed{\lambda_1 = 5, \qquad \lambda_2 = 2}$$

---

### Step 3: Find the Eigenvector for $\lambda_1 = 5$
Plug $\lambda_1 = 5$ into $(A - 5I)\mathbf{v} = \mathbf{0}$:
$$\begin{bmatrix} 4 - 5 & 1 \\ 2 & 3 - 5 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} -1 & 1 \\ 2 & -2 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

This gives the equation:
$$-v_1 + v_2 = 0 \implies v_1 = v_2$$
Any multiple of $[1, 1]^T$ is an eigenvector! Let:
$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

**Check:**
$$A \mathbf{v}_1 = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 5 \\ 5 \end{bmatrix} = 5 \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \lambda_1 \mathbf{v}_1 \quad \text{✓ (Matches!)}$$

---

### Step 4: Find the Eigenvector for $\lambda_2 = 2$
Plug $\lambda_2 = 2$ into $(A - 2I)\mathbf{v} = \mathbf{0}$:
$$\begin{bmatrix} 4 - 2 & 1 \\ 2 & 3 - 2 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 2 & 1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

This gives:
$$2v_1 + v_2 = 0 \implies v_2 = -2v_1$$
Choose $v_1 = 1 \implies v_2 = -2$:
$$\mathbf{v}_2 = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$$

**Check:**
$$A \mathbf{v}_2 = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix} \begin{bmatrix} 1 \\ -2 \end{bmatrix} = \begin{bmatrix} 2 \\ -4 \end{bmatrix} = 2 \begin{bmatrix} 1 \\ -2 \end{bmatrix} = \lambda_2 \mathbf{v}_2 \quad \text{✓ (Matches!)}$$

---

## 5. Diagonalization: The Ultimate Shortcut: $A = Q \Lambda Q^{-1}$

If an $n \times n$ matrix $A$ has $n$ linearly independent eigenvectors, we can stack them as the columns of a matrix $Q$:
$$Q = \begin{bmatrix} | & | \\ \mathbf{v}_1 & \mathbf{v}_2 \\ | & | \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & -2 \end{bmatrix}$$

And place the eigenvalues along the diagonal of $\Lambda$:
$$\Lambda = \begin{bmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{bmatrix} = \begin{bmatrix} 5 & 0 \\ 0 & 2 \end{bmatrix}$$

Then the matrix can be factored as:
$$\boxed{A = Q \Lambda Q^{-1}}$$

### Why is this so magical in AI?
Imagine you need to compute $A^{100}$ (e.g. running an RNN for 100 time steps):
$$A^{100} = (Q \Lambda Q^{-1})(Q \Lambda Q^{-1}) \dots (Q \Lambda Q^{-1})$$
Notice that all interior $Q^{-1} Q$ cancel into the identity matrix $I$!
$$\boxed{A^{100} = Q \Lambda^{100} Q^{-1}}$$

To take $\Lambda^{100}$, you simply raise the individual diagonal numbers to the 100th power:
$$\Lambda^{100} = \begin{bmatrix} 5^{100} & 0 \\ 0 & 2^{100} \end{bmatrix}$$
No matrix multiplication loops needed!

---

## 6. Why Eigenvalues Matter for AI & Deep Learning

### 1. Vanishing and Exploding Gradients
In deep networks and Recurrent Neural Networks (RNNs), hidden states are repeatedly multiplied by weight matrix $W$ over $T$ steps:
$$\mathbf{h}_T \approx W^T \mathbf{h}_0$$
- If the largest eigenvalue $|\lambda_{\max}| > 1$: Activations and gradients **explode exponentially to $\infty$**!
- If the largest eigenvalue $|\lambda_{\max}| < 1$: Activations and gradients **vanish exponentially to $0$**!
- Understanding eigenvalues is why AI researchers invented **Spectral Normalization**, **orthogonal initialization**, and **residual skip connections**!

### 2. Principal Component Analysis (PCA)
In Phase 9, we will see that the directions of maximum variance in high-dimensional data are given by the eigenvectors of the covariance matrix!

---

## 📌 Summary Checklist for Topic 2.4

- [x] **Eigenvectors ($\mathbf{v}$)** are directions that do not rotate under transformation $A$.
- [x] **Eigenvalues ($\lambda$)** are the scalar scaling factors along those invariant axes: $A\mathbf{v} = \lambda\mathbf{v}$.
- [x] To find eigenvalues, solve the **characteristic equation**: $\det(A - \lambda I) = 0$.
- [x] **Diagonalization ($A = Q \Lambda Q^{-1}$)** decomposes a matrix into rotation $\to$ scaling $\to$ inverse rotation.
- [x] Powers of a matrix are easily computed: $A^k = Q \Lambda^k Q^{-1}$.
- [x] If $|\lambda| > 1$, signals explode; if $|\lambda| < 1$, signals vanish.

---

Next up: **Run `experiment.py` in this folder** to compute eigenvalues using the Power Iteration algorithm from scratch!

