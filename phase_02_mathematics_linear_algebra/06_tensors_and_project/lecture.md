# Topic 2.6: Tensors & Capstone Project

You have mastered scalars (0D numbers), vectors (1D arrows), and matrices (2D tables). 

Now, we unlock the ultimate data structure of modern artificial intelligence: **The Tensor**.

In PyTorch, TensorFlow, and JAX, every input image, audio waveform, text token, and neural network weight is represented as a **Tensor**:
$$X \in \mathbb{R}^{B \times C \times H \times W} \qquad \text{or} \qquad X \in \mathbb{R}^{B \times T \times D}$$

In this capstone lecture for Phase 2, we demystify multidimensional arrays, explain how computers store them in flat physical memory using **Strides**, master **Broadcasting**, and derive **Principal Component Analysis (PCA)** from first principles.

---

## 1. 🧒 The 12-Year-Old Analogy: From a Coin to a Shipping Container

Imagine you are organizing physical objects:
- **0D (Scalar):** A single gold coin on the table ($42$).
- **1D (Vector):** A wooden ruler with numbers spaced along one line ($[1, 2, 3]$).
- **2D (Matrix):** A single sheet of graph paper with rows and columns (a spreadsheet).

What happens when you stack 100 sheets of paper together?
- **3D Tensor (A Book):** Now you have `(Page, Row, Column)`.
  - *Example in AI:* A single color image: `(Channels, Height, Width)`. The 3 pages are **Red, Green, Blue**!

What happens when you pack 50 books inside a shipping container?
- **4D Tensor (A Shipping Container of Books):** Now you have `(Box, Book, Row, Column)`.
  - *Example in AI:* A batch of images sent to a GPU:
    $$X \in \mathbb{R}^{B \times C \times H \times W}$$
    - $B$: Batch size (how many images we process in parallel, e.g. 32).
    - $C$: Channels (3 for Red, Green, Blue).
    - $H$: Height of the image (e.g. 224 pixels).
    - $W$: Width of the image (e.g. 224 pixels).

```text
       0D SCALAR            1D VECTOR               2D MATRIX                3D TENSOR              4D BATCH TENSOR
      Single Number         List of numbers         Table of numbers         Stack of tables        Batch of stacks
      
          [ 42 ]           ┌───┐                  ┌───               ───┐         ┌───────────┐         ┌───┐ ┌───┐ ┌───┐
                           │ 1 │                  │  1   2   3   4   5  │        /           /│         │   │ │   │ │   │
                           │ 2 │                  │  6   7   8   9  10  │       /───────────/ │         │ 1 │ │ 2 │ │ 3 │ ...
                           │ 3 │                  │ 11  12  13  14  15  │       │   RED     │ │         └───┘ └───┘ └───┘
                           └───┘                  └───               ───┘       │   GREEN   │/          B images in parallel
                                                                                │   BLUE    │
                                                                                └───────────┘
```

---

## 2. Common Tensor Shapes in Modern AI

| AI Domain | Tensor Shape | Dimension Breakdown | Concrete Example |
| :--- | :---: | :--- | :--- |
| **Computer Vision (CNNs)** | $\mathbb{R}^{B \times C \times H \times W}$ | Batch $\times$ Channels $\times$ Height $\times$ Width | $(32, 3, 224, 224)$ |
| **Language Models (Transformers)** | $\mathbb{R}^{B \times T \times D}$ | Batch $\times$ Sequence Tokens $\times$ Embedding Dim | $(8, 2048, 4096)$ |
| **Audio Processing** | $\mathbb{R}^{B \times C \times L}$ | Batch $\times$ Audio Channels $\times$ Waveform Samples | $(16, 2, 44100)$ |
| **Video Processing** | $\mathbb{R}^{B \times F \times C \times H \times W}$ | Batch $\times$ Video Frames $\times$ Channels $\times$ Height $\times$ Width | $(4, 16, 3, 224, 224)$ |

---

## 3. How Physical Memory Works: Strides & Contiguity

Computer RAM is physically a single, flat, 1-dimensional ribbon of memory addresses. 
How does PyTorch make a 1D strip of RAM look like a 4D tensor? Through **Shapes** and **Strides**!

$$\text{Memory Index}(i_0, i_1, \dots, i_{k-1}) = \sum_{j=0}^{k-1} i_j \times S_j$$

For a 2D matrix of shape $(M, N)$:
- To move right by 1 column: jump 1 slot ($S_1 = 1$).
- To move down by 1 row: skip an entire row of $N$ items ($S_0 = N$).

$$\text{Shape} = (M, N), \qquad \text{Strides} = (N, 1)$$

---

## 4. Tensor Broadcasting: The Cookie Cutter Stamp

### 🧒 The 12-Year-Old Analogy: The Rubber Stamp
Imagine you have a single rubber stamp with 4 numbers: `[10, 20, 30, 40]`.
You need to add these numbers to 3 separate rows in a ledger.
- You don't buy 3 separate rubber stamps!
- You take the **exact same stamp** and press it down onto Row 1, then Row 2, then Row 3!

In PyTorch, **Broadcasting** does this with zero memory allocation by setting stride $S_0 = 0$!

```text
       Tensor A (3 x 1)           Tensor B (1 x 4)                 Broadcasted Sum (3 x 4)
       
          ┌───┐                      ┌───       ───┐               ┌───               ───┐
          │ 1 │                      │ 10 20 30 40 │               │ 11  21   31   41 │
          │ 2 │          +           └───       ───┘     ───►      │ 12  22   32   42 │
          │ 3 │                                                    │ 13  23   33   43 │
          └───┘                                                    └───               ───┘
```

---

## 5. Principal Component Analysis (PCA): The Dinosaur Shadow Puppet

### 🧒 The 12-Year-Old Analogy: The Dinosaur Shadow Puppet
Imagine you build a detailed 3D wire model of a Tyrannosaurus Rex.
You want to take a 2D photograph of it for a science report:
- If you point the flashlight straight down from the ceiling, the shadow on the floor looks like a round, boring blob of wire.
- If you shine the flashlight from the side, the shadow shows the long jaw, sharp teeth, and tail clearly!
- **Principal Component Analysis (PCA)** is finding the **exact flashlight angle** that casts the largest, most informative shadow of your high-dimensional data cloud!

```text
               y
               ▲                  ●   ● (First Principal Component: Max Variance!)
               │                ●  ● ●
               │              ●  ● ●
               │            ● ● ●
               │          ●  ●
               └─────────●────────────────► x
```

---

### 🔬 Mathematical Derivation of PCA

Given a data matrix $X \in \mathbb{R}^{N \times d}$ ($N$ samples, $d$ features):

#### Step 1: Mean Centering
Subtract the average value of each feature:
$$\tilde{X} = X - \boldsymbol{\mu}$$
where $\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}^{(i)}$.

#### Step 2: The Sample Covariance Matrix $C$
$$C = \frac{1}{N - 1} \tilde{X}^T \tilde{X}$$

$$\underbrace{C}_{\substack{\text{Covariance Matrix} \\ (d \times d)}} = \frac{1}{N - 1} \underbrace{\tilde{X}^T}_{\substack{\text{Centered data} \\ \text{transposed}}} \times \underbrace{\tilde{X}}_{\text{Centered data}}$$

**🧒 What the math is actually doing:**  
$C_{i, j}$ measures whether feature $i$ and feature $j$ rise and fall together. The diagonal entries $C_{i, i}$ are the variances of each individual feature!

#### Step 3: Finding the Principal Axes
We compute the eigenvectors of $C$:
$$C \mathbf{v}_1 = \lambda_1 \mathbf{v}_1$$
- $\mathbf{v}_1$ is the **First Principal Component** (the direction of maximum spread).
- $\lambda_1$ is the variance along that axis.
- The low-dimensional projected coordinates for each sample are:
  $$\mathbf{z} = \tilde{X} V_k$$

---

### 🔢 Concrete Numerical Hand Walkthrough

Let us perform PCA on $N = 3$ data points with $d = 2$ features:
$$\mathbf{x}_1 = \begin{bmatrix} 2 \\ 4 \end{bmatrix}, \qquad \mathbf{x}_2 = \begin{bmatrix} 4 \\ 6 \end{bmatrix}, \qquad \mathbf{x}_3 = \begin{bmatrix} 6 \\ 8 \end{bmatrix}$$

1. **Compute Mean $\boldsymbol{\mu}$:**
   $$\mu_1 = \frac{2 + 4 + 6}{3} = \mathbf{4.0}, \qquad \mu_2 = \frac{4 + 6 + 8}{3} = \mathbf{6.0}$$

2. **Compute Mean-Centered Points $\tilde{X}$:**
   $$\tilde{\mathbf{x}}_1 = \begin{bmatrix} 2 - 4 \\ 4 - 6 \end{bmatrix} = \begin{bmatrix} -2 \\ -2 \end{bmatrix}, \qquad \tilde{\mathbf{x}}_2 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \qquad \tilde{\mathbf{x}}_3 = \begin{bmatrix} 2 \\ 2 \end{bmatrix}$$

3. **Compute Covariance Matrix $C$ ($N - 1 = 2$):**
   $$C = \frac{1}{2} \left( \begin{bmatrix} -2 \\ -2 \end{bmatrix} \begin{bmatrix} -2 & -2 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \end{bmatrix} \begin{bmatrix} 0 & 0 \end{bmatrix} + \begin{bmatrix} 2 \\ 2 \end{bmatrix} \begin{bmatrix} 2 & 2 \end{bmatrix} \right)$$
   $$\begin{bmatrix} -2 \\ -2 \end{bmatrix} \begin{bmatrix} -2 & -2 \end{bmatrix} = \begin{bmatrix} 4 & 4 \\ 4 & 4 \end{bmatrix}, \qquad \begin{bmatrix} 2 \\ 2 \end{bmatrix} \begin{bmatrix} 2 & 2 \end{bmatrix} = \begin{bmatrix} 4 & 4 \\ 4 & 4 \end{bmatrix}$$
   $$C = \frac{1}{2} \begin{bmatrix} 8 & 8 \\ 8 & 8 \end{bmatrix} = \begin{bmatrix} 4 & 4 \\ 4 & 4 \end{bmatrix}$$

4. **Find Eigenvalues & Principal Component:**
   $$\det(C - \lambda I) = (4 - \lambda)^2 - 16 = \lambda^2 - 8\lambda = 0 \implies \mathbf{\lambda_1 = 8, \quad \lambda_2 = 0}$$
   - For $\lambda_1 = 8$:
     $$(C - 8I)\mathbf{v} = \begin{bmatrix} -4 & 4 \\ 4 & -4 \end{bmatrix}\mathbf{v} = \mathbf{0} \implies v_1 = v_2 \implies \mathbf{v}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$
   - The first principal component points diagonally at $45^\circ$ along $[1, 1]^T$!

5. **Project Points onto 1D Line:**
   - $z_1 = \tilde{\mathbf{x}}_1 \cdot \mathbf{v}_1 = (-2 \times \frac{1}{\sqrt{2}}) + (-2 \times \frac{1}{\sqrt{2}}) = \mathbf{-2\sqrt{2}} \approx -2.83$
   - $z_2 = \tilde{\mathbf{x}}_2 \cdot \mathbf{v}_1 = \mathbf{0}$
   - $z_3 = \tilde{\mathbf{x}}_3 \cdot \mathbf{v}_1 = \mathbf{+2\sqrt{2}} \approx +2.83$
   We compressed 2D points into 1D coordinates with **100% of information preserved** because $\lambda_2 = 0$!

---

## 6. 📋 Mathematical & Technical Symbol Decoder Table

| Symbol | Pronunciation | Plain English Meaning | Concrete Real-World Analogy / Flashcard Example |
| :---: | :---: | :--- | :--- |
| $\mathbb{R}^{B \times C \times H \times W}$ | "R B-by-C-by-H-by-W" | 4D batch tensor of images | A shipping container of color picture books |
| $\mathbb{R}^{B \times T \times D}$ | "R B-by-T-by-D" | 3D batch tensor of text sequences | A collection of novels with $T$ words per page |
| $S_j$ | "stride j" | Memory jump in flat RAM to advance by 1 along axis $j$ | Number of floor tiles to skip to move to the next aisle |
| $C = \frac{1}{N-1}\tilde{X}^T\tilde{X}$ | "covariance matrix C" | Matrix measuring pairwise correlations between features | How closely students' math scores track science scores |
| $Z = \tilde{X} V_k$ | "Z equals X-tilde V-sub-k" | Low-dimensional PCA projected coordinates | The 2D shadow cast by a 3D wire dinosaur onto a wall |
| $\boldsymbol{\mu}$ | "mu" | Mean vector of feature averages across all samples | The center point of a cloud of dots |
| $\lambda_1 / \sum \lambda_i$ | "explained variance ratio" | Fraction of dataset information captured by principal axis 1 | The percentage of light captured by the main shadow |

---

## 7. 📌 Summary & Phase 2 Capstone Checklist

- [x] **Tensors** generalize multi-dimensional data structures: scalars (0D), vectors (1D), matrices (2D), batch tensors (3D, 4D, 5D).
- [x] Computer RAM is physically 1D; multidimensional indexing uses **Strides**.
- [x] **Broadcasting** expands dimensions of size 1 with zero memory copies using stride 0.
- [x] **PCA** reduces dimensionality by projecting centered data onto the eigenvectors of the sample covariance matrix (or right singular vectors $V$ of SVD).
- [x] You have completed Phase 2: Mathematics: Linear Algebra from first principles!

---

Next up: **Run the capstone project in `experiment.py`** to execute pure Python tensor arithmetic, SVD, and PCA!
