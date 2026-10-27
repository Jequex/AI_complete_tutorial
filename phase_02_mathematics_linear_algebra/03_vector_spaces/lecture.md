# Topic 2.3: Vector Spaces, Span, Basis, and Rank

In machine learning, we often hear phrases like:
* *"The word embeddings live in a 768-dimensional latent vector space."*
* *"The data manifold lies on an intrinsic low-dimensional subspace."*
* *"This weight update matrix has rank 16 (like in LoRA!)."*

What do these concepts actually mean? In this lecture, we explore **Vector Spaces**, **Span**, **Linear Independence**, and the **Four Fundamental Subspaces** of a matrix using intuitive physical analogies and step-by-step arithmetic.

---

## 1. 🧒 The 12-Year-Old Analogy: The Redundant Teammate & The Sandbox

### The Sandbox (Vector Space & Subspaces)
Imagine a giant playground sandbox with a flagpole right in the center representing the origin $(0, 0, 0)$.
- If you walk 2 steps forward and 3 steps right, you are still inside the sandbox.
- If you run twice as fast in that direction, you are still inside the sandbox.
- The flagpole $(0, 0, 0)$ is always there.

A **Vector Space** is simply a mathematical sandbox of arrows where:
1. Adding any two arrows keeps you inside the sandbox (**Closed under addition**).
2. Stretching or reversing any arrow keeps you inside the sandbox (**Closed under scalar multiplication**).
3. The origin $\mathbf{0}$ is always included!

---

### The Redundant Teammate (Linear Independence)
Imagine Timmy, Maya, and Barnaby are exploring an island in a video game:
- **Player 1 (Timmy)** can only move **East/West**: $[1, 0]^T$.
- **Player 2 (Maya)** can only move **North/South**: $[0, 1]^T$.
Working together, Timmy and Maya can reach **any square on the entire 2D island**!

Now, **Player 3 (Barnaby the puppy)** joins the team. Barnaby can run **Northeast**: $[1, 1]^T$.
- Can Barnaby reach any territory that Timmy and Maya couldn't already reach?
- **NO!** Because:
  $$\begin{bmatrix} 1 \\ 1 \end{bmatrix} = 1 \cdot \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 1 \cdot \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$
- Barnaby is **linearly dependent** (redundant)! He adds zero new dimensions to the team's reach.

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
Given a set of vectors $\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$, a **linear combination** is any recipe formed by scaling and adding them:

$$\mathbf{w} = c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_k \mathbf{v}_k = \sum_{i=1}^k c_i \mathbf{v}_i$$

$$\underbrace{\mathbf{w}}_{\text{New destination arrow}} = \underbrace{c_1 \mathbf{v}_1}_{\text{Walk along arrow 1}} + \underbrace{c_2 \mathbf{v}_2}_{\text{Walk along arrow 2}} + \dots + \underbrace{c_k \mathbf{v}_k}_{\text{Walk along arrow } k}$$

### Span
The **Span** of a set of vectors is the collection of **all possible destinations** you can ever reach using linear combinations:

$$\text{Span}(\mathbf{v}_1, \dots, \mathbf{v}_k) = \left\{ \sum_{i=1}^k c_i \mathbf{v}_i \;\middle|\; c_i \in \mathbb{R} \right\}$$

- **1 non-zero vector:** Span is an infinite **1D line** passing through the origin.
- **2 non-parallel vectors in 3D:** Span is an infinite **2D flat plane** passing through the origin.
- **3 non-coplanar vectors in 3D:** Span is the entire **3D universe ($\mathbb{R}^3$)**!

---

## 3. Linear Independence & Basis

### Formal Definition of Linear Independence
A set of vectors $\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$ is **Linearly Independent** if the ONLY way to return to the origin is to take zero steps along every vector:

$$c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_k \mathbf{v}_k = \mathbf{0} \implies c_1 = c_2 = \dots = c_k = 0$$

If there exists even one non-zero set of multipliers where the arrows loop back to $\mathbf{0}$, the vectors are **Linearly Dependent**.

### What is a Basis?
A **Basis** $\mathcal{B} = \{\mathbf{b}_1, \dots, \mathbf{b}_d\}$ for a vector space $V$ is a minimalist coordinate system:
1. They are **Linearly Independent** (no redundant teammates).
2. They **Span** the entire space $V$ (they can reach everywhere in the sandbox).

The **Dimension** of a vector space is simply the **count of vectors in any basis** for that space.

---

## 4. The Column Space $\mathcal{C}(A)$ and Null Space $\mathcal{N}(A)$

When a matrix $A \in \mathbb{R}^{m \times n}$ multiplies a vector $\mathbf{x} = [x_1, \dots, x_n]^T$, it blends the columns of $A$:

$$A \mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \dots + x_n \mathbf{a}_n$$

### 1. The Column Space: $\mathcal{C}(A)$
$$\mathcal{C}(A) = \text{Span}(\mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_n)$$
$$\underbrace{\mathcal{C}(A)}_{\substack{\text{Column Space} \\ \text{(All possible outputs)}}} = \text{Span} \left( \underbrace{\mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_n}_{\text{Columns of Matrix } A} \right)$$
**🧒 What the math is actually doing:**  
$\mathcal{C}(A)$ is the set of all possible outputs $\mathbf{b}$ that the equation $A\mathbf{x} = \mathbf{b}$ can ever reach. If a target output $\mathbf{y}$ is not in the column space, no input $\mathbf{x}$ can ever produce it!

### 2. The Null Space: $\mathcal{N}(A)$ (The Kernel)
$$\mathcal{N}(A) = \{\mathbf{x} \in \mathbb{R}^n \mid A\mathbf{x} = \mathbf{0}\}$$
**🧒 What the math is actually doing:**  
The Null Space contains all input vectors that get squashed to complete zero by the matrix.

---

## 5. Matrix Rank & The Rank-Nullity Theorem

The **Rank** of a matrix is the dimension of its column space:
$$\text{rank}(A) = \text{number of linearly independent columns in } A$$

### The Rank-Nullity Theorem (Conservation of Dimensions)
In linear algebra, dimensions cannot vanish into thin air. Every input dimension must either survive into the output column space or get squashed into the null space:

$$\text{Rank}(A) + \text{Nullity}(A) = n$$

$$\underbrace{\text{Rank}(A)}_{\substack{\text{Surviving output} \\ \text{dimensions}}} + \underbrace{\text{Nullity}(A)}_{\substack{\text{Dimensions squashed} \\ \text{to complete zero}}} = \underbrace{n}_{\text{Total input dimensions}}$$

### 🔢 Concrete Numerical Walkthrough

Consider matrix $A \in \mathbb{R}^{2 \times 2}$:
$$A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$$

1. **Check Linear Independence of Columns:**
   - Column 1: $\mathbf{a}_1 = [1, 2]^T$
   - Column 2: $\mathbf{a}_2 = [2, 4]^T = 2 \cdot [1, 2]^T$
   - Column 2 is a scalar multiple of Column 1! They point along the exact same line.

2. **Compute Rank:**
   - There is only $1$ linearly independent column:
   $$\text{Rank}(A) = \mathbf{1}$$

3. **Compute Null Space & Nullity:**
   - Solve $A \mathbf{x} = \mathbf{0}$:
     $$\begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} x_1 + 2x_2 \\ 2x_1 + 4x_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
   - This gives $x_1 + 2x_2 = 0 \implies x_1 = -2x_2$.
   - Any vector of the form $\mathbf{x} = x_2 \begin{bmatrix} -2 \\ 1 \end{bmatrix}$ gets squashed to $\mathbf{0}$!
   - The null space is an entire 1D line spanned by $[-2, 1]^T$:
   $$\text{Nullity}(A) = \mathbf{1}$$

4. **Verify Rank-Nullity Theorem:**
   $$\text{Rank}(A) + \text{Nullity}(A) = 1 + 1 = 2 = n \quad \text{(Input dimension verified!)}$$

---

## 6. Why This Matters for AI: Low-Rank Adaptation (LoRA)

In modern Large Language Models (LLMs), a single Transformer weight matrix $W$ might have size $4096 \times 4096$:
$$\text{Total Parameters} = 4096 \times 4096 = \mathbf{16,777,216\text{ parameters}}$$

When fine-tuning on a new task, updating 16.7 million weights is slow and memory-intensive.

In 2021, researchers observed that the update matrix $\Delta W$ has an **extremely low intrinsic rank** ($r = 8$ or $16$):

$$\Delta W = B \cdot A$$

where $B \in \mathbb{R}^{4096 \times r}$ and $A \in \mathbb{R}^{r \times 4096}$.

$$\text{LoRA Trainable Parameters} = (4096 \times 16) + (16 \times 4096) = \mathbf{131,072\text{ parameters}}$$

This is a **$99.2\%$ reduction in memory and compute**, enabling users to fine-tune massive AI models on a single consumer GPU!

---

## 7. 📋 Mathematical & Technical Symbol Decoder Table

| Symbol | Pronunciation | Plain English Meaning | Concrete Real-World Analogy / Flashcard Example |
| :---: | :---: | :--- | :--- |
| $\text{Span}(\mathbf{v}_1, \dots)$ | "span of v1 to vk" | All reachable destinations formed by linear combinations | The entire territory your team can explore on an island |
| $\mathcal{B} = \{\mathbf{b}_1, \dots\}$ | "basis B" | Minimal set of non-redundant arrows that span a space | The standard compass arrows (East and North) |
| $\mathcal{C}(A)$ | "column space of A" | The span of all columns of matrix $A$ | All possible output targets the neural net layer can produce |
| $\mathcal{N}(A)$ | "null space of A" | The set of all input vectors squashed to complete zero | Inputs that disappear completely into a black hole |
| $\text{rank}(A)$ | "rank of A" | Number of linearly independent columns in $A$ | Number of true independent skills on a project team |
| $\text{nullity}(A)$ | "nullity of A" | Dimension of the null space $\mathcal{N}(A)$ | Number of redundant dimensions crushed out of existence |
| $\Delta W = BA$ | "delta W equals B times A" | Low-Rank Adaptation (LoRA) matrix factorization | Replacing a massive 16M table with two slender matrices |

---

## 8. 📌 Summary & Mathematical Checklist

- [x] A **Vector Space** is closed under vector addition and scalar multiplication, and always contains $\mathbf{0}$.
- [x] **Span** is the set of all possible linear combinations reachable by a collection of vectors.
- [x] Vectors are **Linearly Independent** if no vector in the set can be recreated from the others.
- [x] A **Basis** is a minimal independent spanning set; its size is the space's **Dimension**.
- [x] **Column Space $\mathcal{C}(A)$** contains all possible outputs $A\mathbf{x}$.
- [x] **Null Space $\mathcal{N}(A)$** contains all inputs mapped to $\mathbf{0}$.
- [x] **Rank-Nullity Theorem:** $\text{Rank}(A) + \text{Nullity}(A) = n$.
- [x] **LoRA** uses low-rank decomposition $\Delta W = BA$ to reduce fine-tuning memory by $>99\%$.

---

Next up: **Run `python3 experiment.py`** in this folder to test matrix rank algorithms and low-rank LoRA projections live!
