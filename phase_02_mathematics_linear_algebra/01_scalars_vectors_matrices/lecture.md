# Topic 2.1: Scalars, Vectors, and Matrices

Welcome to the foundation of all machine learning mathematics. 

Whether you are training a neural network to recognize cats, generating photorealistic images with diffusion models, or chatting with a 70-billion-parameter language model, **under the hood, the computer only sees numbers arranged in lists and grids**.

In this lecture, we demystify scalars, vectors, and matrices using intuitive physical analogies, complete symbol breakdowns, and step-by-step derivations.

---

## 1. 🧒 The 12-Year-Old Analogy: The Pirate Treasure Map

Imagine you are playing a pirate treasure hunt game with Timmy and Maya:

1. **A Scalar ($x \in \mathbb{R}$): Just a Single Number**
   - The pirate captain shouts: *"Drink $3$ sips of water!"* or *"The temperature on the beach is $78^\circ\text{F}$."*
   - There is no direction, no map, and no coordinates. It is just a solitary number measuring size (magnitude).

2. **A Vector ($\mathbf{x} \in \mathbb{R}^n$): An Arrow on a Map**
   - The treasure map says: *"From the big palm tree, walk **4 paces East** and **3 paces North**."*
   - Notice that you need **two numbers** together to describe this journey: $[4, 3]$. 
   - That arrow has two superpowers: **how long it is** (magnitude: $\sqrt{4^2 + 3^2} = 5$ paces) and **which way it points** (direction: Northeast).
   - In AI, if a vector has 3 numbers, it is a 3D arrow $[x, y, z]$. If it has 4,096 numbers (like in an LLM word embedding), it is an arrow pointing in a 4,096-dimensional conceptual universe!

3. **A Matrix ($A \in \mathbb{R}^{m \times n}$): A Spreadsheet of Multiple Arrows**
   - Suppose 3 different pirates embark on 3 different treasure paths:
     - Pirate 1 walks: $4$ paces East, $3$ paces North
     - Pirate 2 walks: $1$ pace West ($-1$), $5$ paces North
     - Pirate 3 walks: $0$ paces East, $2$ paces South ($-2$)
   - Stack them into a table:
     $$A = \begin{bmatrix} 4 & 3 \\ -1 & 5 \\ 0 & -2 \end{bmatrix}$$
   - A matrix is simply a grid of numbers where each row or column can represent a vector, an image of pixels, or the weights of a neural network!

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
* Symbol: $\mathbb{R}$ represents the set of all real numbers ($-\infty, +\infty$).
* $\in$ means *"is an element of"* (belongs to).
* Examples: $x = 42$, $\alpha = -3.14$, $\lambda = 0.001$.

### 2. Vectors: $\mathbf{x} \in \mathbb{R}^n$
A vector is an ordered sequence of $n$ numbers. By universal convention in modern machine learning, **vectors are written as column vectors**:

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n$$

* The boldface letter $\mathbf{x}$ indicates it is a vector.
* $n$ is the **dimension** (how many numbers are inside).
* $x_i$ is the $i$-th entry (a scalar).
* **Two Interpretations:**
  1. **As a Point in Space:** $\mathbf{x} = [4, 3]^T$ is the specific point located at coordinates $(4, 3)$.
  2. **As a Directed Arrow:** An arrow starting at the origin $(0, 0)$ and pointing to $(4, 3)$.

### 3. Matrices: $A \in \mathbb{R}^{m \times n}$
A matrix is a 2D rectangular array of numbers with $m$ rows and $n$ columns:

$$A = \begin{bmatrix} a_{11} & a_{12} & \dots & a_{1n} \\ a_{21} & a_{22} & \dots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \dots & a_{mn} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

* **Index Memory Trick:** $a_{\text{row}, \text{column}} \longrightarrow$ Remember **"Roman Catholic" (RC = Row first, Column second)**!
* $a_{23}$ means: Row 2, Column 3.

---

## 3. 📋 Mathematical & Technical Symbol Decoder Table

| Symbol | Pronunciation | Plain English Meaning | Concrete Real-World Analogy / Flashcard Example |
| :---: | :---: | :--- | :--- |
| $\mathbb{R}$ | "the real numbers" | The continuous infinite number line | Any decimal, fraction, or integer ($3.14, -7, 0$) |
| $x \in \mathbb{R}$ | "x in R" | $x$ is a single real scalar number | The outdoor temperature ($78^\circ\text{F}$) |
| $\mathbf{x} \in \mathbb{R}^n$ | "vector x in R-n" | A vertical stack of $n$ numbers | Walking 4 paces East and 3 paces North ($[4, 3]^T$) |
| $A \in \mathbb{R}^{m \times n}$ | "Matrix A in R m-by-n" | A 2D grid with $m$ rows and $n$ columns | A spreadsheet of travel paths for 3 different pirates |
| $\mathbf{x}^T$ | "x transpose" | Flips a vertical column vector into a horizontal row | Tilting a vertical stick into a horizontal position |
| $\|\mathbf{x}\|_2$ | "L-2 norm of x" | The straight-line Euclidean distance | Flying straight as the crow flies across a field ($\sqrt{4^2 + 3^2} = 5$) |
| $\|\mathbf{x}\|_1$ | "L-1 norm of x" | Manhattan or taxicab grid distance | Driving a taxi on Manhattan city blocks ($|4| + |3| = 7$) |
| $\|\mathbf{x}\|_\infty$ | "L-infinity norm of x" | Maximum absolute value in the vector | Picking the tallest building in the city skyline |
| $\mathbf{x} \cdot \mathbf{y}$ | "x dot y" | Dot product (similarity alignment score) | How strongly two arrows point in the same direction |
| $\cos\theta$ | "cosine theta" | Angle cosine between two vectors ($-1$ to $+1$) | Two compass needles pointing together vs. apart |
| $\det(A)$ | "determinant of A" | Scaling factor of volume or area under matrix $A$ | How much a rubber sheet stretches or shrinks in area |
| $\text{Tr}(A)$ | "trace of A" | Sum of diagonal entries $\sum A_{ii}$ | Adding up the main diagonal backbone of a square matrix |

---

## 4. Fundamental Vector Operations

### 1. Vector Addition
Add corresponding components. Both vectors **must have identical dimension**:

$$\mathbf{x} + \mathbf{y} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} x_1 + y_1 \\ x_2 + y_2 \end{bmatrix}$$

* **🧒 Geometry (Head-to-Tail Rule):** Walk along arrow $\mathbf{x}$. From where $\mathbf{x}$ ends, walk along arrow $\mathbf{y}$. The sum is the shortcut arrow connecting the start to the final destination!

### 2. Scalar Multiplication
Multiply every component by scalar $c \in \mathbb{R}$:

$$c \mathbf{x} = c \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} c x_1 \\ c x_2 \end{bmatrix}$$

* **🧒 Geometry:** Stretches the arrow by factor $c$. If $c = 2$, it doubles in length. If $c = -1$, it flips to point backwards.

---

## 5. Vector Norms: Measuring the Length of an Arrow

In AI, we constantly measure **how large** an error vector is, or how much weights have grown during regularization. A function that measures vector size is called a **Norm**, written as $\|\mathbf{x}\|$.

```text
    L1 NORM (Manhattan Grid)              L2 NORM (Euclidean Distance)
  Walk along city streets blocks           Fly straight as the crow flies
  
          ┌───────B                                  B
          │                                         /
          │                                        /
          │                                       /
          A                                      A
  ||x||₁ = |4| + |3| = 7                 ||x||₂ = √(4² + 3²) = 5
```

### 1. The $L_2$ Norm (Euclidean Distance)
The standard straight-line distance given by the Pythagorean Theorem:

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{x_1^2 + x_2^2 + \dots + x_n^2}$$

$$\underbrace{\|\mathbf{x}\|_2}_{\text{Euclidean length}} = \sqrt{ \underbrace{x_1^2 + x_2^2 + \dots + x_n^2}_{\text{Square every coordinate and sum}} }$$

**🧒 What the math is actually doing:**  
Square every step you took, add them together, and take the square root. For $[4, 3]^T$, $\sqrt{4^2 + 3^2} = \sqrt{16 + 9} = \sqrt{25} = \mathbf{5}$.

### 2. The $L_1$ Norm (Manhattan Taxicab Distance)
Imagine driving a taxi in Manhattan. You cannot drive diagonally through skyscrapers; you must drive along grid blocks:

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i| = |x_1| + |x_2| + \dots + |x_n|$$

For $[4, -3]^T$: $|4| + |-3| = 4 + 3 = \mathbf{7}$. In AI, $L_1$ regularization (Lasso) drives unimportant weights to exactly zero!

---

## 6. The Dot Product: The Beating Heart of AI

The **dot product** (or inner product) combines two vectors and boils them down into a single scalar number:

$$\mathbf{x} \cdot \mathbf{y} = \mathbf{x}^T \mathbf{y} = \sum_{i=1}^n x_i y_i = x_1 y_1 + x_2 y_2 + \dots + x_n y_n$$

### 🔬 The Geometric Dot Product Formula & Derivation

The dot product satisfies a fundamental geometric identity:

$$\mathbf{x} \cdot \mathbf{y} = \|\mathbf{x}\|_2 \|\mathbf{y}\|_2 \cos\theta$$

$$\underbrace{\mathbf{x} \cdot \mathbf{y}}_{\text{Dot product score}} = \underbrace{\|\mathbf{x}\|_2}_{\text{Length of Arrow 1}} \times \underbrace{\|\mathbf{y}\|_2}_{\text{Length of Arrow 2}} \times \underbrace{\cos\theta}_{\substack{\text{Angle alignment} \\ (-1 \text{ to } +1)}}$$

**🧒 What the math is actually doing:**  
It multiplies the length of the two arrows by how closely their directions match!
- If the arrows point in the **exact same direction** ($\theta = 0^\circ$): $\cos(0^\circ) = 1$. The dot product is **maximum positive**!
- If the arrows are **perpendicular / orthogonal** ($\theta = 90^\circ$): $\cos(90^\circ) = 0$. The dot product is **zero**!
- If the arrows point in **opposite directions** ($\theta = 180^\circ$): $\cos(180^\circ) = -1$. The dot product is **maximum negative**!

---

## 7. Cosine Similarity: How AI Measures Conceptual Meaning

In modern LLMs and vector databases (Pinecone, Chroma, Milvus), text embeddings are compared using **Cosine Similarity**:

$$\cos\theta = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2}$$

$$\underbrace{\cos\theta}_{\substack{\text{Semantic similarity} \\ \text{score between } -1 \text{ and } +1}} = \frac{ \overbrace{\mathbf{x} \cdot \mathbf{y}}^{\text{Raw directional alignment}} }{ \underbrace{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2}_{\text{Lengths of the two vectors}} }$$

**🧒 What the math is actually doing:**  
By dividing by the lengths of the two vectors, cosine similarity ignores how long an article is. It strictly measures whether the two concepts point in the same direction in thought-space!

### 🔢 Concrete Numerical Walkthrough of Cosine Similarity

Let vector $\mathbf{x} = [4, 3]^T$ (e.g. embedding for `"king"`) and $\mathbf{y} = [1, 2]^T$ (e.g. embedding for `"queen"`):

1. **Calculate Dot Product:**
   $$\mathbf{x} \cdot \mathbf{y} = (4 \times 1) + (3 \times 2) = 4 + 6 = \mathbf{10}$$

2. **Calculate Lengths ($L_2$ Norms):**
   $$\|\mathbf{x}\|_2 = \sqrt{4^2 + 3^2} = \sqrt{16 + 9} = \sqrt{25} = \mathbf{5.0}$$
   $$\|\mathbf{y}\|_2 = \sqrt{1^2 + 2^2} = \sqrt{1 + 4} = \sqrt{5} \approx \mathbf{2.236}$$

3. **Calculate Cosine Similarity:**
   $$\cos\theta = \frac{10}{5.0 \times 2.236} = \frac{10}{11.18} \approx \mathbf{0.8944}$$
   *Interpretation:* A cosine score of $0.894$ indicates very strong semantic similarity ($\theta \approx 26.5^\circ$)!

---

## 8. Matrix Multiplication: $C = AB$

Matrix multiplication calculates dot products between **rows of the first matrix** and **columns of the second matrix**.

Suppose $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$. Their product $C = AB$ has dimension $m \times n$:

$$\boxed{(m \times k) \times (k \times n) = (m \times n)}$$

> ⚠️ **The Golden Dimension Rule:** The **inside dimensions must match** ($k = k$)! If they do not, multiplication is physically impossible.

### 🔢 Concrete Numerical Example:
Let:
$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \qquad B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$$

1. $C_{11} = (1 \times 5) + (2 \times 7) = 5 + 14 = \mathbf{19}$
2. $C_{12} = (1 \times 6) + (2 \times 8) = 6 + 16 = \mathbf{22}$
3. $C_{21} = (3 \times 5) + (4 \times 7) = 15 + 28 = \mathbf{43}$
4. $C_{22} = (3 \times 6) + (4 \times 8) = 18 + 32 = \mathbf{50}$

$$C = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

---

## 9. 📌 Summary & Mathematical Checklist

- [x] **Scalars ($x \in \mathbb{R}$)** measure pure magnitude.
- [x] **Vectors ($\mathbf{x} \in \mathbb{R}^n$)** represent arrows with magnitude and direction.
- [x] **Matrices ($A \in \mathbb{R}^{m \times n}$)** represent 2D grids of numbers that transform coordinate spaces.
- [x] **$L_2$ Norm ($\|\mathbf{x}\|_2$)** calculates straight-line Euclidean distance via the Pythagorean theorem.
- [x] **$L_1$ Norm ($\|\mathbf{x}\|_1$)** sums absolute values, driving model sparsity in Lasso regularization.
- [x] **Dot Product ($\mathbf{x} \cdot \mathbf{y} = \|\mathbf{x}\| \|\mathbf{y}\| \cos\theta$)** measures arrow alignment.
- [x] **Cosine Similarity ($\frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}$)** powers vector search and transformer attention.
- [x] **Matrix Multiplication ($AB$)** requires matching inner dimensions: $(M, K) \times (K, N) \to (M, N)$.

---

Next up: **Run `python3 experiment.py`** in this folder to see vector operations, norms, and cosine similarity calculations run from scratch!
