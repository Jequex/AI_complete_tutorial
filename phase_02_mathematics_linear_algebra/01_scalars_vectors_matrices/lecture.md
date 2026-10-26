# Topic 2.1: Scalars, Vectors, and Matrices

Welcome to the foundation of all machine learning mathematics. 

Whether you are training a neural network to recognize cats, generating photorealistic images with diffusion models, or chatting with a 70-billion-parameter language model, **under the hood, the computer only sees numbers arranged in lists and grids**.

In this lecture, we will demystify scalars, vectors, and matrices using intuitive physical analogies, complete symbol breakdowns, and step-by-step derivations.

---

## 1. 🧒 The 12-Year-Old Analogy: The Treasure Map

Imagine you are playing a pirate treasure hunt game:

1. **A Scalar ($x \in \mathbb{R}$): Just a Single Number**
   - The pirate captain shouts: *"Drink $3$ sips of water!"* or *"The temperature is $78^\circ\text{F}$."*
   - There is no direction, no map, no coordinates. It's just a single magnitude (a solitary number).

2. **A Vector ($\mathbf{x} \in \mathbb{R}^n$): An Arrow on a Map**
   - The map says: *"From the big palm tree, walk **4 paces East** and **3 paces North**."*
   - Notice that you need **two numbers** together to describe this journey: $[4, 3]$. 
   - That arrow has two superpowers: **how long it is** (magnitude: $\sqrt{4^2 + 3^2} = 5$ paces) and **which way it points** (direction: Northeast).
   - In AI, if a vector has 3 numbers, it's a 3D arrow $[x, y, z]$. If it has 4,096 numbers (like in an LLM embedding), it is an arrow pointing in a 4,096-dimensional space!

3. **A Matrix ($A \in \mathbb{R}^{m \times n}$): A Spreadsheet of Multiple Arrows**
   - Suppose 3 different pirates embark on 3 different treasure paths:
     - Pirate 1 walks: $4$ paces East, $3$ paces North
     - Pirate 2 walks: $1$ pace West ($-1$), $5$ paces North
     - Pirate 3 walks: $0$ paces East, $2$ paces South ($-2$)
   - Stack them into a table:
     $$A = \begin{bmatrix} 4 & 3 \\ -1 & 5 \\ 0 & -2 \end{bmatrix}$$
   - A matrix is just a grid of numbers where each row or column can represent a vector, an image of pixels, or the weights of a neural network!

---

## 2. Formal Definitions & Notation

```text
  SCALAR (0D)           VECTOR (1D)                 MATRIX (2D)
  Single number        List / Arrow              Table / Grid of numbers
   
      [ 7 ]           ┌───┐                      ┌───               ───┐
                      │ 4 │                      │  4   3   1   0   8  │
                      │ 3 │                      │ -1   5   2   4   1  │
                      │ 9 │                      │  0  -2   7   3   6  │
                      └───┘                      └───               ───┘
   x ∈ ℝ               x ∈ ℝ³                         A ∈ ℝ³ˣ⁵
 (0 dimensions)      (1 dimension)                 (2 dimensions)
```

### 1. Scalars: $x \in \mathbb{R}$
A scalar is a single real number. 
* Symbol: $\mathbb{R}$ represents the set of all real numbers ($\mathbb{R} = (-\infty, +\infty)$).
* $\in$ means *"is an element of"* (belongs to).
* Examples: $x = 42$, $\alpha = -3.14$, $\lambda = 0.001$.

### 2. Vectors: $\mathbf{x} \in \mathbb{R}^n$
A vector is an ordered sequence of $n$ numbers. By universal convention in modern machine learning, **vectors are written as column vectors**:

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n$$

* The boldface letter $\mathbf{x}$ indicates it is a vector.
* $n$ is the **dimension** (how many numbers are inside).
* $x_i$ is the $i$-th entry (a scalar).
* **Two Interpretations:**
  1. **As a Point in Space:** $\mathbf{x} = [4, 3]$ is the specific point located at coordinates $(4, 3)$.
  2. **As a Directed Arrow:** An arrow starting at the origin $(0, 0)$ and pointing to $(4, 3)$.

### 3. Matrices: $A \in \mathbb{R}^{m \times n}$
A matrix is a 2D rectangular array of numbers with $m$ rows and $n$ columns:

$$A = \begin{bmatrix} a_{11} & a_{12} & \dots & a_{1n} \\ a_{21} & a_{22} & \dots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \dots & a_{mn} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

* **Index Memory Trick:** $a_{\text{row}, \text{column}} \longrightarrow$ Remember **"Roman Catholic" (RC = Row first, Column second)**!
* $a_{23}$ means: Row 2, Column 3.

---

### 📋 Symbol Decoder Table

| Symbol | Pronunciation | Plain English Meaning | Example |
| :---: | :---: | :--- | :--- |
| $\mathbb{R}$ | "Real numbers" | The number line (all decimals, fractions, negatives) | $3.1415, -7, 0$ |
| $x \in \mathbb{R}$ | "$x$ in R" | $x$ is a single real scalar number | $x = 5.2$ |
| $\mathbf{x} \in \mathbb{R}^n$ | "vector x in R-n" | A list of $n$ numbers stacked vertically | $\mathbf{x} = [3, -1, 4]^T \in \mathbb{R}^3$ |
| $A \in \mathbb{R}^{m \times n}$ | "Matrix A in R m-by-n" | A grid of numbers with $m$ rows and $n$ columns | $3 \times 2$ matrix |
| $\mathbf{x}^T$ | "x transpose" | Flip vertical vector into horizontal row | $[1, 2]^T = [1 \quad 2]$ |
| $\|\mathbf{x}\|$ | "norm of x" | The physical length (magnitude) of the vector | $\sqrt{3^2 + 4^2} = 5$ |
| $\mathbf{x} \cdot \mathbf{y}$ | "x dot y" | Dot product (similarity score between two arrows) | $(1)(3) + (2)(4) = 11$ |
| $\cos\theta$ | "cosine theta" | Angle between two arrows (ranges from $-1$ to $+1$) | $\cos(0^\circ) = 1$ |

---

## 3. Fundamental Vector Operations

### 1. Vector Addition
Add corresponding components. Both vectors **must have the identical dimension**:

$$\mathbf{x} + \mathbf{y} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} x_1 + y_1 \\ x_2 + y_2 \end{bmatrix}$$

* **🧒 Geometry (Head-to-Tail Rule):** Walk along arrow $\mathbf{x}$. From where $\mathbf{x}$ ends, walk along arrow $\mathbf{y}$. The sum is the shortcut arrow from the start to the final destination.

### 2. Scalar Multiplication
Multiply every component by scalar $c \in \mathbb{R}$:

$$c \mathbf{x} = c \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} c x_1 \\ c x_2 \end{bmatrix}$$

* **🧒 Geometry:** Stretches the arrow by factor $c$. If $c = 2$, it doubles in length. If $c = -1$, it points in the exact opposite direction.

### 3. Vector Transpose: $(\cdot)^T$
The transpose flips rows into columns, and columns into rows:

$$\mathbf{x} = \begin{bmatrix} 4 \\ 3 \end{bmatrix} \in \mathbb{R}^{2 \times 1} \implies \mathbf{x}^T = \begin{bmatrix} 4 & 3 \end{bmatrix} \in \mathbb{R}^{1 \times 2}$$

---

## 4. Vector Norms: How Long is an Arrow?

In AI, we constantly need to measure **"how big"** an error vector is, or how large a weight vector has grown. A function that measures vector size is called a **Norm**, written as $\|\mathbf{x}\|$.

```text
    L1 NORM (Manhattan)                   L2 NORM (Euclidean)
 Walk on city street grid               Fly straight as the crow flies
 
         ┌───────B                                  B
         │                                         /
         │                                        /
         │                                       /
         A                                      A
 ||x||₁ = |4| + |3| = 7                 ||x||₂ = √(4² + 3²) = 5
```

### 1. The $L_2$ Norm (Euclidean Length) — The Crow's Flight
The standard straight-line distance given by the Pythagorean Theorem:

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{x_1^2 + x_2^2 + \dots + x_n^2}$$

* **Example:** For $\mathbf{x} = [3, 4]^T$:
  $$\|\mathbf{x}\|_2 = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = \mathbf{5}$$

### 2. The $L_1$ Norm (Manhattan / Taxicab Distance)
Imagine driving a taxi in Manhattan, New York. You cannot drive diagonally through buildings; you must drive along city street blocks:

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i| = |x_1| + |x_2| + \dots + |x_n|$$

* **Example:** For $\mathbf{x} = [3, -4]^T$:
  $$\|\mathbf{x}\|_1 = |3| + |-4| = 3 + 4 = \mathbf{7}$$
* **AI Importance:** $L_1$ regularization (Lasso) forces parameters to become exactly zero, creating **sparse models**!

### 3. The $L_\infty$ Norm (Max Norm)
Takes the largest absolute value in the vector:

$$\|\mathbf{x}\|_\infty = \max_{i} |x_i|$$

* **Example:** For $\mathbf{x} = [3, -9, 2]^T$:
  $$\|\mathbf{x}\|_\infty = \max(|3|, |-9|, |2|) = \mathbf{9}$$

---

## 5. The Dot Product: The Beating Heart of AI

The **dot product** (also called inner product) takes two vectors and boils them down into a single scalar number:

$$\mathbf{x} \cdot \mathbf{y} = \mathbf{x}^T \mathbf{y} = \sum_{i=1}^n x_i y_i = x_1 y_1 + x_2 y_2 + \dots + x_n y_n$$

### Numerical Example:
Let $\mathbf{x} = [2, 3]^T$ and $\mathbf{y} = [4, 1]^T$:
$$\mathbf{x} \cdot \mathbf{y} = (2 \times 4) + (3 \times 1) = 8 + 3 = \mathbf{11}$$

---

### 🔬 The Geometric Dot Product Formula & Derivation

The dot product has a breathtaking geometric identity:

$$\boxed{\mathbf{x} \cdot \mathbf{y} = \|\mathbf{x}\| \|\mathbf{y}\| \cos\theta}$$

where $\theta$ is the angle between the two arrows.

```text
                       y
                      ▲
                     /
                    / 
                   /  θ
                  └───────► x
                   proj_x(y)
```

#### Step-by-Step Mathematical Derivation from the Law of Cosines
Let arrows $\mathbf{x}$ and $\mathbf{y}$ form two sides of a triangle. The third side connecting their tips is the difference vector $(\mathbf{x} - \mathbf{y})$.

1. From high school trigonometry, the **Law of Cosines** for any triangle with sides $a, b, c$ and angle $\theta$ opposite to $c$ states:
   $$c^2 = a^2 + b^2 - 2ab \cos\theta$$

2. Substitute the lengths of our vectors:
   $$\|\mathbf{x} - \mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2 \|\mathbf{x}\| \|\mathbf{y}\| \cos\theta \quad \text{--- (Equation 1)}$$

3. Now, expand the left side algebraically using vector dot products:
   $$\|\mathbf{x} - \mathbf{y}\|^2 = (\mathbf{x} - \mathbf{y}) \cdot (\mathbf{x} - \mathbf{y})$$
   $$= \mathbf{x} \cdot \mathbf{x} - 2(\mathbf{x} \cdot \mathbf{y}) + \mathbf{y} \cdot \mathbf{y}$$
   $$= \|\mathbf{x}\|^2 - 2(\mathbf{x} \cdot \mathbf{y}) + \|\mathbf{y}\|^2 \quad \text{--- (Equation 2)}$$

4. Equate Equation 1 and Equation 2:
   $$\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2(\mathbf{x} \cdot \mathbf{y}) = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2 \|\mathbf{x}\| \|\mathbf{y}\| \cos\theta$$

5. Subtract $\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2$ from both sides:
   $$-2(\mathbf{x} \cdot \mathbf{y}) = -2 \|\mathbf{x}\| \|\mathbf{y}\| \cos\theta$$

6. Divide both sides by $-2$:
   $$\mathbf{x} \cdot \mathbf{y} = \|\mathbf{x}\| \|\mathbf{y}\| \cos\theta \quad \blacksquare$$

Look at what this means:
* If the arrows point in the **exact same direction** ($\theta = 0^\circ$): $\cos(0^\circ) = 1$. The dot product is **maximum positive**!
* If the arrows are **perpendicular / orthogonal** ($\theta = 90^\circ$): $\cos(90^\circ) = 0$. The dot product is **zero**!
* If the arrows point in **opposite directions** ($\theta = 180^\circ$): $\cos(180^\circ) = -1$. The dot product is **maximum negative**!

---

## 6. Cosine Similarity: How AI Understands Meaning

If you give ChatGPT the word `"king"` and the word `"queen"`, the AI converts them into 4,096-dimensional embedding vectors $\mathbf{x}_{\text{king}}$ and $\mathbf{x}_{\text{queen}}$.

How does the AI know they have similar meanings? By computing their **Cosine Similarity**:

$$\cos\theta = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}$$

```text
  HIGH SIMILARITY (cos θ ≈ 1.0)              ORTHOGONAL / UNRELATED (cos θ = 0.0)
       "king"       "queen"                         "banana"
         ▲           ▲                                 ▲
          \         /                                  │
           \  θ≈0° /                                   │ θ=90°
            \     /                                    │
             \   /                                     └────────► "quantum physics"
```

Because cosine similarity divides by the lengths of the vectors, it is completely immune to how "long" the text is. It only measures whether the two arrows point in the same conceptual direction!

---

## 7. Matrix Multiplication: $C = AB$

Matrix multiplication is **not** multiplying corresponding numbers. It is calculating dot products between **rows of the first matrix** and **columns of the second matrix**.

Suppose $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$. Their product $C = AB$ has dimension $m \times n$:

$$\boxed{(m \times k) \times (k \times n) = (m \times n)}$$

> ⚠️ **The Golden Dimension Rule:** The **inside dimensions must match** ($k = k$)! If they don't, multiplication is physically impossible.

### Step-by-Step Arithmetic Calculation
Let:
$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \in \mathbb{R}^{2 \times 2}, \qquad B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$

Each entry $c_{ij}$ in product $C$ is the dot product of **Row $i$ of $A$** and **Column $j$ of $B$**:

$$C = \begin{bmatrix} c_{11} & c_{12} \\ c_{21} & c_{22} \end{bmatrix}$$

1. **Calculate $c_{11}$ (Row 1 of $A$ $\cdot$ Column 1 of $B$):**
   $$c_{11} = (1 \times 5) + (2 \times 7) = 5 + 14 = \mathbf{19}$$

2. **Calculate $c_{12}$ (Row 1 of $A$ $\cdot$ Column 2 of $B$):**
   $$c_{12} = (1 \times 6) + (2 \times 8) = 6 + 16 = \mathbf{22}$$

3. **Calculate $c_{21}$ (Row 2 of $A$ $\cdot$ Column 1 of $B$):**
   $$c_{21} = (3 \times 5) + (4 \times 7) = 15 + 28 = \mathbf{43}$$

4. **Calculate $c_{22}$ (Row 2 of $A$ $\cdot$ Column 2 of $B$):**
   $$c_{22} = (3 \times 6) + (4 \times 8) = 18 + 32 = \mathbf{50}$$

Result:
$$C = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

---

## 📌 Summary Checklist for Topic 2.1

- [x] **Scalars ($x \in \mathbb{R}$)** are single numbers.
- [x] **Vectors ($\mathbf{x} \in \mathbb{R}^n$)** are ordered lists of numbers representing arrows with magnitude and direction.
- [x] **Matrices ($A \in \mathbb{R}^{m \times n}$)** are 2D grids with $m$ rows and $n$ columns.
- [x] **$L_2$ Norm ($\|\mathbf{x}\|_2 = \sqrt{\sum x_i^2}$)** measures true geometric Euclidean length.
- [x] **$L_1$ Norm ($\|\mathbf{x}\|_1 = \sum |x_i|$)** measures grid taxicab distance.
- [x] **Dot Product ($\mathbf{x} \cdot \mathbf{y} = \mathbf{x}^T \mathbf{y} = \|\mathbf{x}\| \|\mathbf{y}\| \cos\theta$)** measures arrow alignment.
- [x] **Cosine Similarity ($\frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}$)** powers semantic search and transformer attention.
- [x] **Matrix Multiplication ($C = AB$)** requires inner dimensions to match: $(m \times k) \times (k \times n) \to (m \times n)$.

---

Next up: **Run `experiment.py` in this folder** to build a complete linear algebra engine from scratch and test cosine similarity on real vector embeddings!

