# Topic 2.3: Vector Spaces, Span, Basis, and Rank

In machine learning, we often hear phrases like:
* *"The embeddings live in a 768-dimensional latent space."*
* *"The data lies on a low-dimensional manifold/subspace."*
* *"This weight matrix has rank 16 (like in LoRA!)."*

What do these concepts actually mean? In this lecture, we explore **Vector Spaces**, **Span**, **Linear Independence**, and the **Four Fundamental Subspaces** of a matrix.

---

## 1. 🧒 The 12-Year-Old Analogy: The Redundant Teammate & The Sandbox

### The Sandbox (Vector Space & Subspaces)
Imagine a giant playground sandbox. You can walk anywhere inside the sandbox.
- If you take 2 steps forward and 3 steps right, you are still inside the sandbox.
- If you run twice as fast, you are still in the sandbox.
- The origin $(0, 0, 0)$ is the big flagpole in the center.

A **Vector Space** is simply a sandbox of arrows where:
1. Adding any two arrows keeps you inside the sandbox (**Closed under addition**).
2. Stretching or reversing any arrow keeps you inside the sandbox (**Closed under scalar multiplication**).
3. The origin $\mathbf{0}$ is always included!

---

### The Redundant Teammate (Linear Independence)
Imagine you are playing a video game where your team needs to explore a 2D map:
- **Player 1** can only move **East/West** ($[1, 0]$).
- **Player 2** can only move **North/South** ($[0, 1]$).
Together, Player 1 and Player 2 can reach **any square on the entire island**!

Now, **Player 3** joins your team. Player 3 can move **Northeast** ($[1, 1]$).
- Can Player 3 reach any territory that Player 1 and Player 2 couldn't already reach?
- **NO!** Because:
  $$\begin{bmatrix} 1 \\ 1 \end{bmatrix} = 1 \cdot \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 1 \cdot \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$
- Player 3 is **linearly dependent** (redundant)! Player 3 adds zero new dimensions to your team's reach.

```text
    LINEARLY INDEPENDENT (Span = 2D Plane)         LINEARLY DEPENDENT (Redundant arrow!)
    
              y                                              y
              ▲                                              ▲   v3 = v1 + v2 (Redundant!)
              │                                              │  ↗
           v2 ●                                           v2 ● /
              │                                              │/
              └───●──► x                                     └───●──► x
                  v1                                             v1
```

---

## 2. Linear Combinations & Span

### Linear Combination
Given a set of vectors $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\}$, a **linear combination** is any vector formed by scaling and adding them:

$$\mathbf{w} = c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_k \mathbf{v}_k = \sum_{i=1}^k c_i \mathbf{v}_i$$

where $c_1, c_2, \dots, c_k \in \mathbb{R}$ are scalar weights.

### Span
The **Span** of a set of vectors is the collection of **all possible vectors** that can ever be reached by linear combinations:

$$\text{Span}(\mathbf{v}_1, \dots, \mathbf{v}_k) = \left\{ \sum_{i=1}^k c_i \mathbf{v}_i \;\middle|\; c_i \in \mathbb{R} \right\}$$

- **A single non-zero vector:** Span is an infinite **1D line** passing through the origin.
- **Two non-parallel vectors in 3D:** Span is an infinite **2D flat sheet (plane)** passing through the origin.
- **Three non-coplanar vectors in 3D:** Span is the entire **3D universe ($\mathbb{R}^3$)**!

---

## 3. Linear Independence & Basis

### Formal Definition of Linear Independence
A set of vectors $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\}$ is **Linearly Independent** if the ONLY combination that yields the zero vector is setting every single scalar to zero:

$$c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_k \mathbf{v}_k = \mathbf{0} \implies c_1 = c_2 = \dots = c_k = 0$$

If there exists even one non-zero set of $c_i$ that equals $\mathbf{0}$, the vectors are **Linearly Dependent**.

---

### What is a Basis?
A **Basis** for a vector space $V$ is a set of vectors $\mathcal{B} = \{\mathbf{b}_1, \dots, \mathbf{b}_d\}$ that satisfies two conditions:
1. They are **Linearly Independent** (no redundant vectors).
2. They **Span** the entire space $V$ (they can reach everywhere in $V$).

> 💡 **The Minimalist Coordinate System:** A basis is the smallest possible set of arrows needed to construct unique coordinates for every point in the space!

The **Dimension** of a vector space is simply the **number of vectors in any basis** for that space.

---

## 4. The Column Space and Null Space of a Matrix

Every matrix $A \in \mathbb{R}^{m \times n}$ has columns:
$$A = \begin{bmatrix} | & | & & | \\ \mathbf{a}_1 & \mathbf{a}_2 & \dots & \mathbf{a}_n \\ | & | & & | \end{bmatrix}$$

When we multiply $A$ by vector $\mathbf{x} = [x_1, \dots, x_n]^T$, we are forming a linear combination of its columns:
$$A \mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \dots + x_n \mathbf{a}_n$$

### 1. The Column Space: $\mathcal{C}(A)$
* **Definition:** The span of all the column vectors of $A$:
  $$\mathcal{C}(A) = \text{Span}(\mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_n)$$
* **Meaning:** $\mathcal{C}(A)$ is the set of all possible outputs $\mathbf{b}$ that the equation $A\mathbf{x} = \mathbf{b}$ can ever reach!
* If a target vector $\mathbf{y}$ is NOT in the column space $\mathcal{C}(A)$, the equation $A\mathbf{x} = \mathbf{y}$ has **zero solutions**!

### 2. The Null Space: $\mathcal{N}(A)$ (The Kernel)
* **Definition:** The set of all input vectors $\mathbf{x}$ that get squashed to complete zero:
  $$\mathcal{N}(A) = \{\mathbf{x} \in \mathbb{R}^n \mid A\mathbf{x} = \mathbf{0}\}$$
* **Meaning:** If a matrix collapses 3D space into a 2D plane, an entire 1D line of inputs was squashed into $\mathbf{0}$. That squashed line is the Null Space!

```text
       INPUT SPACE (ℝⁿ)                               OUTPUT SPACE (ℝᵐ)
       
       ┌────────────────────────┐                    ┌────────────────────────┐
       │                        │                    │                        │
       │    x ──────────────────┼── Matrix A ───────►│  Ax ∈ C(A)             │
       │                        │                    │  (Column Space)        │
       │   [ Null Space N(A) ]  │                    │                        │
       │   all inputs squashed  │                    │                        │
       │   straight to zero! ───┼── Matrix A ───────►│  0 (Origin)            │
       │                        │                    │                        │
       └────────────────────────┘                    └────────────────────────┘
```

---

## 5. Matrix Rank & The Rank-Nullity Theorem

### What is Rank?
The **Rank** of a matrix, written $\text{rank}(A)$, is the **dimension of its column space**:
$$\text{rank}(A) = \text{number of linearly independent columns in } A$$

- If $A \in \mathbb{R}^{m \times n}$, then $\text{rank}(A) \le \min(m, n)$.
- If $\text{rank}(A) = \min(m, n)$, the matrix is called **Full Rank**.

---

### The Rank-Nullity Theorem (The Conservation of Dimensions)
In linear algebra, dimensions cannot vanish into thin air. Every input dimension must either survive into the output column space or get squashed into the null space:

$$\boxed{\text{Rank}(A) + \text{Nullity}(A) = n}$$

where $n$ is the total number of columns (the dimension of the input space).

* **🧒 Example:** Suppose you take a 3D input space ($n = 3$).
  * If the matrix squashes the output into a **2D plane** ($\text{Rank} = 2$), then exactly **$3 - 2 = 1$ dimension** must have collapsed into the Null Space ($\text{Nullity} = 1$).
  * If the matrix squashes everything onto a **1D line** ($\text{Rank} = 1$), then **$3 - 1 = 2$ dimensions** collapsed into the Null Space ($\text{Nullity} = 2$).

---

## 6. Why This Matters for AI: Low-Rank Adaptation (LoRA)

In modern Large Language Models (LLMs), a single weight matrix $W$ in a Transformer block might have size $4096 \times 4096$:
$$\text{Total Parameters} = 4096 \times 4096 = 16,777,216 \text{ numbers!}$$

Training all 16.7 million weights is slow and memory-intensive.

In 2021, researchers discovered that during fine-tuning, the parameter update matrix $\Delta W$ has an **extremely low rank** (e.g. $\text{rank } r = 8$ or $16$):

$$\Delta W = B \cdot A$$
where $B \in \mathbb{R}^{4096 \times r}$ and $A \in \mathbb{R}^{r \times 4096}$.

$$\text{LoRA Trainable Parameters} = (4096 \times 16) + (16 \times 4096) = 131,072 \text{ numbers!}$$
That is a **$99.2\%$ reduction in memory and compute**, all made possible because of vector space rank!

---

## 📌 Summary Checklist for Topic 2.3

- [x] A **Vector Space** is closed under addition and scalar scaling, and contains the origin $\mathbf{0}$.
- [x] **Span** is the set of all possible linear combinations reachable by a set of vectors.
- [x] A set of vectors is **Linearly Independent** if no vector can be written as a combination of the others (zero redundancy).
- [x] A **Basis** is a linearly independent set that spans the space; **Dimension** is the count of basis vectors.
- [x] **Column Space $\mathcal{C}(A)$** is the span of matrix columns (all possible outputs).
- [x] **Null Space $\mathcal{N}(A)$** contains all input vectors squashed to $\mathbf{0}$.
- [x] **Rank-Nullity Theorem:** $\text{Rank}(A) + \text{Nullity}(A) = n$.
- [x] **LoRA** compresses weight updates using low-rank factorizations $B A$.

---

Next up: **Run `experiment.py` in this folder** to compute matrix ranks and detect linear dependency from scratch!

