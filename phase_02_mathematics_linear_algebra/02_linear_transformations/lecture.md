# Topic 2.2: Linear Transformations

Most people are taught that a matrix is just a boring table of numbers. 

In modern Artificial Intelligence, that view will hold you back. **A matrix is not just data—a matrix is a machine that transforms space.**

When a neural network processes an image or an LLM processes a sentence, each layer multiplies the input vector by a weight matrix $W$:
$$\mathbf{y} = W \mathbf{x}$$
Geometrically, that weight matrix is stretching, twisting, rotating, and warping the coordinate space so that the AI can make sense of the data!

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
   *(Scaling an arrow by $c$ and then transforming it gives the same result as transforming the arrow and then scaling it by $c$).*

Together, they can be written as the **Principle of Superposition**:
$$T(c_1 \mathbf{u} + c_2 \mathbf{v}) = c_1 T(\mathbf{u}) + c_2 T(\mathbf{v})$$

Every linear transformation can be represented as multiplication by a matrix $A$:
$$\boxed{T(\mathbf{x}) = A \mathbf{x}}$$

---

## 3. The Big Secret: Where do $\hat{i}$ and $\hat{j}$ Land?

How do you know what a matrix actually does to space? 

You only need to track **two tiny standard basis arrows**:
* $\hat{i} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ (1 step to the Right along the $x$-axis)
* $\hat{j} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$ (1 step Up along the $y$-axis)

### The Columns of a Matrix are Just the Landing Sites of $\hat{i}$ and $\hat{j}$!

Look at what happens when you multiply matrix $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$ by $\hat{i}$:
$$A \hat{i} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} (a \times 1) + (b \times 0) \\ (c \times 1) + (d \times 0) \end{bmatrix} = \begin{bmatrix} a \\ c \end{bmatrix} \longleftarrow \mathbf{\text{Column 1 of } A!}$$

Now multiply $A$ by $\hat{j}$:
$$A \hat{j} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} (a \times 0) + (b \times 1) \\ (c \times 0) + (d \times 1) \end{bmatrix} = \begin{bmatrix} b \\ d \end{bmatrix} \longleftarrow \mathbf{\text{Column 2 of } A!}$$

> 💡 **Core Insight:** If someone gives you a matrix:
> $$A = \begin{bmatrix} 3 & -2 \\ 1 & 4 \end{bmatrix}$$
> You immediately know what it does:
> - It sends the horizontal basis arrow $\hat{i}$ to $[3, 1]^T$.
> - It sends the vertical basis arrow $\hat{j}$ to $[-2, 4]^T$.
> - Every other point in the universe follows along linearly!

---

## 4. The Classic Zoo of Geometric Transformations

### 1. Scaling (Zooming In and Out)
$$S = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$$
- $\hat{i}$ lands at $[s_x, 0]^T$ (stretched along $x$ by $s_x$).
- $\hat{j}$ lands at $[0, s_y]^T$ (stretched along $y$ by $s_y$).
- *Example:* If $s_x = 2, s_y = 3$, horizontal distances double, and vertical distances triple.

---

### 2. Rotation Matrix ($R_\theta$)
How do you rotate all of 2D space counterclockwise by an angle $\theta$?

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
- When $\hat{i} = [1, 0]^T$ rotates by $\theta$, its new tip coordinates are $[\cos\theta, \sin\theta]^T$.
- When $\hat{j} = [0, 1]^T$ rotates by $\theta$, its new tip coordinates are $[-\sin\theta, \cos\theta]^T$.

Pack these two landing sites into the columns of a matrix:
$$\boxed{R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}}$$

---

### 3. Shear (Sliding a Deck of Playing Cards)
Imagine a neat deck of cards sitting on a table. If you push the top of the deck sideways, the bottom card stays still while each higher card slides further to the right:

$$A_{\text{shear}} = \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$$
- $\hat{i} = [1, 0]^T$ stays completely still ($[1, 0]^T$).
- $\hat{j} = [0, 1]^T$ tilts to $[k, 1]^T$.

---

### 4. Projection (Squashing Space Flat)
$$P_x = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$$
- Any vector $\mathbf{v} = [x, y]^T$ becomes:
  $$P_x \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} x \\ 0 \end{bmatrix}$$
- It squashes 2D space down onto the 1D $x$-axis. The $y$-information is permanently lost!

---

## 5. The Determinant: The Area Scaling Factor

What is the **Determinant** of a matrix, written as $\det(A)$ or $|A|$?

### 🧒 The 12-Year-Old Analogy: The Stretchy Grid Box
Take a $1 \times 1$ square on your original grid formed by basis vectors $\hat{i}$ and $\hat{j}$.
- Its original area is $1 \times 1 = \mathbf{1}$.

Now, apply matrix $A$. The square stretches into a slanted **parallelogram**.
**The Determinant $\det(A)$ is simply the Area of that new parallelogram!**

```text
     BEFORE TRANSFORMATION                       AFTER TRANSFORMATION
     Area = 1 x 1 = 1                            Area = |det(A)|
     
       y                                           y
       ▲                                           ▲            Landing of ĵ + î
     1 ┼───┌───────┐                             4 ┼             /───────────●
       │   │Area=1 │                               │            /           /
       │   │       │                             2 ┼           ●           /
       └───┴───────┴──► x                          │          /           /
       0   1                                       └─────────●───────────┴────► x
                                                   0         3
```

---

### 🔬 Derivation of the $2 \times 2$ Determinant Formula
Let $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$. 
$\hat{i}$ lands at $[a, c]^T$, and $\hat{j}$ lands at $[b, d]^T$.

From geometry, the area of a parallelogram defined by vectors $[a, c]^T$ and $[b, d]^T$ can be calculated by enclosing it in a rectangle of width $(a + b)$ and height $(c + d)$, and subtracting the surrounding empty triangles and rectangles:

$$\boxed{\det(A) = ad - bc}$$

### What Does the Value of $\det(A)$ Mean?

| Value of $\det(A)$ | Geometric Meaning in Space | AI Implication |
| :---: | :--- | :--- |
| $\det(A) > 1$ | Space **expands** (areas grow larger) | Gradients can explode if multiplied repeatedly |
| $0 < \det(A) < 1$ | Space **contracts** (areas shrink) | Gradients vanish towards zero |
| $\det(A) = 1$ | Rigid transformation (area preserved, e.g. pure rotation) | Stable representations |
| $\det(A) = 0$ | Space **collapsed** into a flat line or single point | **Irreversible information loss!** Matrix has no inverse ($A^{-1}$ does not exist). |
| $\det(A) < 0$ | Space was **flipped / inverted** (like looking in a mirror) | Orientation flipped |

---

## 📌 Summary Checklist for Topic 2.2

- [x] A matrix $A$ is a **geometric linear transformation** $T(\mathbf{x}) = A\mathbf{x}$.
- [x] Under a linear transformation, the origin $(0,0)$ remains fixed, and grid lines remain parallel and evenly spaced.
- [x] The **columns of matrix $A$** are literally the landing coordinates of the standard basis vectors $\hat{i}$ and $\hat{j}$.
- [x] Rotation matrix: $R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$.
- [x] The **Determinant $\det(A) = ad - bc$** is the factor by which areas scale under transformation.
- [x] If $\det(A) = 0$, space collapses to a lower dimension and the matrix is **singular (non-invertible)**.

---

Next up: **Run `experiment.py` in this folder** to watch a 2D ASCII grid rotate, scale, and shear in real time!

