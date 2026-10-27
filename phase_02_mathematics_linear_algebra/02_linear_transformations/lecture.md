# Topic 2.2: Linear Transformations

Most people are taught that a matrix is just a boring spreadsheet of numbers. 

In modern Artificial Intelligence, that view will hold you back. **A matrix is not just data—a matrix is a machine that transforms space.**

When a neural network processes an image or an LLM processes a sentence, each layer multiplies the input vector by a weight matrix $W$:
$$\mathbf{y} = W \mathbf{x}$$
Geometrically, that weight matrix is stretching, twisting, rotating, and warping the coordinate space so that the AI can untangle and classify complex data!

---

## 1. 🧒 The 12-Year-Old Analogy: The Stretchy Rubber Sheet

Imagine you take a sheet of stretchy grid paper made of rubber. You draw a coordinate grid on it with an origin $(0, 0)$ right in the center.

Now, take both hands and stretch the rubber sheet. What rules must you follow for it to be a **Linear Transformation**?

```text
       RULE 1: The origin (0, 0) MUST NEVER MOVE!
       RULE 2: All grid lines must remain straight and evenly spaced!
       (No bending, no curving, no tearing)
```

```text
      BEFORE (Standard Grid)                 AFTER (Transformed Grid: Stretched & Sheared)
      
          │   │   │   │                               /   /   /   /
      ────┼───┼───┼───┼────                       ───/───/───/───/───
          │   │   │   │                             /   /   /   /
      ────┼───●───┼───┼────                       ─/───●───/───/───/─
          │  (0,0)│   │                           /  (0,0)/   /   /
      ────┼───┼───┼───┼────                       ───/───/───/───/───
          │   │   │   │                             /   /   /   /
```

If you respect these two simple rules, you are performing a **Linear Transformation**.

---

## 2. The Formal Mathematical Definition

A function $T: \mathbb{R}^n \to \mathbb{R}^m$ is called a **Linear Transformation** if and only if it satisfies two fundamental properties:

1. **Additivity (Preserves Addition):**
   $$T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$$
   *(Transforming the sum of two arrows gives the same result as transforming each arrow separately and adding them).*

2. **Homogeneity (Preserves Scalar Scaling):**
   $$T(c \mathbf{u}) = c T(\mathbf{u}) \quad \forall c \in \mathbb{R}$$
   *(Scaling an arrow by $c$ and then transforming it gives the same result as transforming the arrow first and then scaling it by $c$).*

Together, they form the **Principle of Superposition**:

$$T(c_1 \mathbf{u} + c_2 \mathbf{v}) = c_1 T(\mathbf{u}) + c_2 T(\mathbf{v})$$

$$\underbrace{T(c_1 \mathbf{u} + c_2 \mathbf{v})}_{\text{Transform a combined journey}} = \underbrace{c_1 T(\mathbf{u})}_{\text{Scaled result of Path 1}} + \underbrace{c_2 T(\mathbf{v})}_{\text{Scaled result of Path 2}}$$

**🧒 What the math is actually doing:**  
It means space does not warp like a funhouse mirror. Walking twice as far along path $\mathbf{u}$ before the stretch sends you twice as far after the stretch!

Every linear transformation can be written as multiplication by a matrix $A$:
$$\boxed{T(\mathbf{x}) = A \mathbf{x}}$$

---

## 3. The Big Secret: Where do $\hat{i}$ and $\hat{j}$ Land?

How do you know what a matrix actually does to all of space? 

You do not need to track billions of points! You only need to track **two tiny standard basis arrows**:
* $\hat{i} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ (pronounced *"i-hat"*: 1 step Right along the $x$-axis)
* $\hat{j} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$ (pronounced *"j-hat"*: 1 step Up along the $y$-axis)

### The Columns of a Matrix are Just the Landing Sites of $\hat{i}$ and $\hat{j}$!

Look at what happens when you multiply matrix $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$ by $\hat{i}$:

$$A \hat{i} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} (a \times 1) + (b \times 0) \\ (c \times 1) + (d \times 0) \end{bmatrix} = \begin{bmatrix} a \\ c \end{bmatrix} \longleftarrow \mathbf{\text{Column 1 of } A!}$$

Now multiply $A$ by $\hat{j}$:

$$A \hat{j} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} (a \times 0) + (b \times 1) \\ (c \times 0) + (d \times 1) \end{bmatrix} = \begin{bmatrix} b \\ d \end{bmatrix} \longleftarrow \mathbf{\text{Column 2 of } A!}$$

> 💡 **Core Insight:** If someone hands you a matrix:
> $$A = \begin{bmatrix} 3 & -2 \\ 1 & 4 \end{bmatrix}$$
> You immediately know what it does to the entire universe:
> - It lands $\hat{i}$ at $[3, 1]^T$.
> - It lands $\hat{j}$ at $[-2, 4]^T$.
> - Every other point $\mathbf{v} = [x, y]^T$ lands at $x \begin{bmatrix} 3 \\ 1 \end{bmatrix} + y \begin{bmatrix} -2 \\ 4 \end{bmatrix}$!

---

## 4. The Classic Zoo of Geometric Transformations

### 1. Scaling (Zooming In and Out)
$$S = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$$
- $\hat{i}$ lands at $[s_x, 0]^T$ (stretched horizontally by $s_x$).
- $\hat{j}$ lands at $[0, s_y]^T$ (stretched vertically by $s_y$).

### 2. Rotation Matrix ($R_\theta$)
How do you rotate all of 2D space counterclockwise by angle $\theta$?

```text
               y
               ▲
               │          [ -sin θ, cos θ ]^T = Landing of ĵ
               │         ●
               │        /|
               │       / |
               │      /  |
               │     /   ● [ cos θ, sin θ ]^T = Landing of î
               │    / θ /
               └────┴──┴──────► x
```

From unit-circle trigonometry:
- $\hat{i} = [1, 0]^T$ rotates to $[\cos\theta, \sin\theta]^T$.
- $\hat{j} = [0, 1]^T$ rotates to $[-\sin\theta, \cos\theta]^T$.

Pack these two landing sites into the columns of a matrix:
$$\boxed{R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}}$$

### 3. Shear (Sliding a Deck of Playing Cards)
Imagine a neat deck of cards on a table. Push the top of the deck sideways: the bottom card stays still while each higher card slides further to the right:
$$A_{\text{shear}} = \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$$

### 4. Projection (Squashing Space Flat)
$$P_x = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$$
It squashes all 2D points onto the 1D $x$-axis. The vertical $y$-coordinate is flattened to zero!

---

## 5. The Determinant: The Area Scaling Multiplier

What is the **Determinant** of a matrix, written as $\det(A)$ or $|A|$?

### 🧒 The 12-Year-Old Analogy: The Stretchy Grid Box
Take a $1 \times 1$ square on your original grid formed by basis arrows $\hat{i}$ and $\hat{j}$.
- Its original area is $1 \times 1 = \mathbf{1}$.

Now apply matrix $A$. The square stretches into a slanted **parallelogram**.
**The Determinant $\det(A)$ is simply the Area of that new parallelogram!**

$$\det(A) = ad - bc$$

$$\underbrace{\det(A)}_{\text{Area scaling multiplier}} = \underbrace{a \cdot d}_{\text{Main diagonal stretch}} - \underbrace{b \cdot c}_{\text{Off-diagonal shear overlap}}$$

### 🔢 Concrete Numerical Walkthrough

Let matrix $A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$ transform vector $\mathbf{v} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$:

1. **Columns as Landing Sites:**
   - $\hat{i} = [1, 0]^T \longrightarrow \begin{bmatrix} 3 \\ 0 \end{bmatrix}$
   - $\hat{j} = [0, 1]^T \longrightarrow \begin{bmatrix} 1 \\ 2 \end{bmatrix}$

2. **Transform Vector $\mathbf{v}$:**
   $$A \mathbf{v} = 2 \cdot \begin{bmatrix} 3 \\ 0 \end{bmatrix} + 1 \cdot \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 6 + 1 \\ 0 + 2 \end{bmatrix} = \begin{bmatrix} 7 \\ 2 \end{bmatrix}$$

3. **Compute the Determinant (Area Scale):**
   $$\det(A) = (3 \times 2) - (1 \times 0) = 6 - 0 = \mathbf{6}$$
   *Interpretation:* The unit square of area $1$ has been stretched into a parallelogram of area **$6$**!

4. **Significance of Determinant Values:**
   - If $\det(A) > 1$: Space expands (areas grow).
   - If $0 < \det(A) < 1$: Space contracts (areas shrink).
   - If $\det(A) = 1$: Rigid movement (e.g. pure rotation preserves area).
   - If $\det(A) = 0$: Space was squashed flat onto a line or point (**irreversible information loss!** $A^{-1}$ does not exist).
   - If $\det(A) < 0$: Space was flipped inside-out like looking in a mirror.

---

## 6. 📋 Mathematical & Technical Symbol Decoder Table

| Symbol | Pronunciation | Plain English Meaning | Concrete Real-World Analogy / Flashcard Example |
| :---: | :---: | :--- | :--- |
| $T(\mathbf{x}) = A\mathbf{x}$ | "T of x equals A x" | Linear transformation mapping vector $\mathbf{x}$ to a new position | Stretching a rubber sheet with a fixed center |
| $\hat{i}$ | "i-hat" | Standard unit basis vector along the $x$-axis ($[1, 0]^T$) | Walking exactly 1 step to the East |
| $\hat{j}$ | "j-hat" | Standard unit basis vector along the $y$-axis ($[0, 1]^T$) | Walking exactly 1 step to the North |
| $R_\theta$ | "R of theta" | 2D counterclockwise rotation matrix by angle $\theta$ | Spinning a wheel or steering wheel by angle $\theta$ |
| $\det(A)$ | "determinant of A" | The scaling factor of area (or volume in 3D) caused by matrix $A$ | How many times bigger a $1 \times 1$ rubber square becomes |
| $A^{-1}$ | "A inverse" | The matrix that completely undoes transformation $A$ | Hitting the "Undo" (Ctrl+Z) button to unstretch the sheet |
| $A_{\text{shear}}$ | "A-shear" | Transformation sliding coordinates parallel to an axis | Slanting a neat deck of playing cards sideways |
| $\theta$ | "theta" | Angle of rotation in radians or degrees | The angle between the two hands of a clock |

---

## 7. 📌 Summary & Mathematical Checklist

- [x] A matrix is an engine that transforms coordinate space: $T(\mathbf{x}) = A\mathbf{x}$.
- [x] The **columns of matrix $A$** are the exact landing coordinates of standard unit vectors $\hat{i}$ and $\hat{j}$.
- [x] **Rotation matrix:** $R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ rotates all of space without altering distances.
- [x] **Determinant ($\det(A) = ad - bc$):** Measures the factor by which area expands, contracts, or flips.
- [x] If $\det(A) = 0$, the transformation collapses dimensions, losing information irreversibly.

---

Next up: **Run `python3 experiment.py`** in this folder to watch ASCII grids scale, rotate, and shear live!
