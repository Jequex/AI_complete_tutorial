# Topic 2.6: Tensors & Capstone Project

You have mastered scalars (0D numbers), vectors (1D lists of arrows), and matrices (2D tables). 

Now, we unlock the ultimate data structure of modern artificial intelligence: **The Tensor**.

In PyTorch, TensorFlow, and JAX, every input image, audio waveform, text token, and neural network weight is represented as a **Tensor**:
$$X \in \mathbb{R}^{B \times C \times H \times W} \qquad \text{or} \qquad X \in \mathbb{R}^{B \times T \times D}$$

In this lecture, we will demystify multidimensional arrays, explain how computers store them in flat physical memory using **Strides**, master **Broadcasting**, and derive **Principal Component Analysis (PCA)**.

---

## 1. 🧒 The 12-Year-Old Analogy: From a Sheet to a Filing Cabinet

Imagine you have a single sheet of lined paper:
- **0D (Scalar):** Writing a single number in one box ($5$).
- **1D (Vector):** Writing a list of numbers along one row ($[1, 2, 3]$).
- **2D (Matrix):** Filling the entire page with rows and columns (a spreadsheet).

What happens when you stack 100 pages together?
- **3D Tensor (A Book):** Now you have `(Page, Row, Column)`.
  - *Example in AI:* A single color image: `(Height, Width, Color Channel)`. The 3 color pages are **Red, Green, Blue**!

What happens when you put 50 books inside a shipping box?
- **4D Tensor (A Shipping Box of Books):** Now you have `(Box, Book, Row, Column)`.
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

| AI Domain | Tensor Shape | Dimension Breakdown | Example Dimensions |
| :--- | :---: | :--- | :--- |
| **Computer Vision (CNNs)** | $\mathbb{R}^{B \times C \times H \times W}$ | Batch $\times$ Channels $\times$ Height $\times$ Width | $(32, 3, 224, 224)$ |
| **Language Models (Transformers)** | $\mathbb{R}^{B \times T \times D}$ | Batch $\times$ Sequence Tokens $\times$ Embedding Dim | $(8, 2048, 4096)$ |
| **Audio Processing** | $\mathbb{R}^{B \times C \times L}$ | Batch $\times$ Audio Channels $\times$ Waveform Samples | $(16, 2, 44100)$ |
| **Video Processing** | $\mathbb{R}^{B \times F \times C \times H \times W}$ | Batch $\times$ Video Frames $\times$ Channels $\times$ Height $\times$ Width | $(4, 16, 3, 224, 224)$ |

---

## 3. How Physical Memory Works: Strides & Contiguity

A computer's physical RAM is not 4-dimensional. **RAM is a single, flat, 1-dimensional line of memory addresses!**

How does PyTorch make a 1D strip of RAM look like a 4D tensor? Through **Shapes** and **Strides**!

```text
       2D View (Shape: 2 x 3)                     1D Physical Memory (Flat RAM)
       
       ┌───             ───┐
       │ a₀₀   a₀₁   a₀₂   │  ─────────►   [ a₀₀ , a₀₁ , a₀₂ , a₁₀ , a₁₁ , a₁₂ ]
       │ a₁₀   a₁₁   a₁₂   │                 0     1     2     3     4     5
       └───             ───┘
```

### What is a Stride?
The **stride** along dimension $k$ tells the computer: *"How many physical memory slots must I jump forward in RAM to advance by 1 in that dimension?"*

For a row-major 2D matrix of shape $(M, N)$:
- To move 1 step along a **column** (rightward): you jump **1 slot** ($\text{stride}_1 = 1$).
- To move 1 step along a **row** (downward): you must skip an entire row of $N$ items ($\text{stride}_0 = N$)!

$$\text{Shape} = (M, N), \qquad \text{Strides} = (N, 1)$$

### Memory Index Formula
To locate element $(i, j, k)$ in a 3D tensor:
$$\text{Memory Address} = (i \times \text{stride}_0) + (j \times \text{stride}_1) + (k \times \text{stride}_2)$$

> 💡 **Why does this matter?** When you call `.transpose()` or `.permute()` in PyTorch, PyTorch **does not copy any data in memory**! It simply swaps the stride numbers. That is why PyTorch tensor operations are instantaneous!

---

## 4. Tensor Broadcasting: How Shapes Auto-Expand

When you add a vector of biases $\mathbf{b} \in \mathbb{R}^{D}$ to an entire batch of representations $X \in \mathbb{R}^{B \times T \times D}$:

$$Y = X + \mathbf{b}$$

The dimensions don't match ($B \times T \times D$ vs $D$). How does Python know what to do? Through **Broadcasting**!

```text
       Tensor A (3 x 1)           Tensor B (1 x 4)                 Broadcasted Sum (3 x 4)
       
          ┌───┐                      ┌───       ───┐               ┌───               ───┐
          │ 1 │                      │ 10 20 30 40 │               │ 11  21   31   41 │
          │ 2 │          +           └───       ───┘     ───►      │ 12  22   32   42 │
          │ 3 │                                                    │ 13  23   33   43 │
          └───┘                                                    └───               ───┘
```

### The Universal Broadcasting Rules
Starting from the **trailing (right-most) dimensions** and working backwards:
1. Two dimensions are compatible if:
   - They are **equal**, OR
   - One of them is **1**.
2. If a dimension is missing in one tensor, it is padded with a $1$ on the left.
3. If a dimension is $1$, it is automatically stretched (duplicated) to match the larger size without consuming extra memory.

---

## 5. Principal Component Analysis (PCA) From First Principles

Now, let us bring together everything we learned in Phase 2—vectors, matrices, covariance, eigenvalues, and SVD—to derive **Principal Component Analysis (PCA)**.

PCA is the most famous dimensionality reduction technique in data science. It answers:
> *"Given data in 1,000 dimensions, what are the top 2 directions where the data varies the most?"*

```text
               y
               ▲                  ●   ● (First Principal Component: Max Variance!)
               │                ●  ● ●
               │              ●  ● ●
               │            ● ● ●
               │          ●  ●
               └─────────●────────────────► x
```

### Step-by-Step Derivation of PCA

#### 1. Mean Centering
Given a data matrix $X \in \mathbb{R}^{N \times d}$ ($N$ samples, $d$ features), we first subtract the mean of each feature:
$$\tilde{X} = X - \mu$$
This ensures the center of the data cloud is at the origin $(0, 0)$.

#### 2. The Sample Covariance Matrix
The covariance matrix $C \in \mathbb{R}^{d \times d}$ measures how features vary together:
$$C = \frac{1}{N - 1} \tilde{X}^T \tilde{X}$$

#### 3. Eigendecomposition of Covariance
Because $C$ is symmetric ($C = C^T$), the Spectral Theorem guarantees it has orthogonal eigenvectors:
$$C \mathbf{v}_i = \lambda_i \mathbf{v}_i$$
- **Eigenvector $\mathbf{v}_1$:** The **First Principal Component** (the direction of greatest variance in the data).
- **Eigenvalue $\lambda_1$:** The variance along that direction!
- **Eigenvector $\mathbf{v}_2$:** The second greatest variance direction, strictly perpendicular to $\mathbf{v}_1$.

#### 4. The Direct SVD Shortcut
Instead of computing the huge $d \times d$ covariance matrix $\tilde{X}^T \tilde{X}$, we can take the SVD of the centered data matrix directly:
$$\tilde{X} = U \Sigma V^T$$
Recall from Topic 2.5:
$$\tilde{X}^T \tilde{X} = V \Sigma^2 V^T$$
**The right singular vectors $V$ are literally the Principal Components!**
And the projected low-dimensional coordinates are simply:
$$Z = \tilde{X} V_k$$

---

## 📌 Summary Checklist for Topic 2.6

- [x] **Tensors** generalize vectors and matrices to arbitrary dimensions: $X \in \mathbb{R}^{d_1 \times d_2 \times \dots \times d_k}$.
- [x] In AI, images are 4D $(B, C, H, W)$ and text is 3D $(B, T, D)$.
- [x] Computers store multidimensional tensors in 1D flat RAM using **Strides**.
- [x] **Broadcasting** expands dimensions of size 1 to enable elementwise arithmetic without copying memory.
- [x] **PCA** projects high-dimensional data onto the top eigenvectors of the covariance matrix, equivalent to the right singular vectors $V$ of SVD.

---

Next up: **Run the capstone project in `experiment.py`** to build the complete pure Python Linear Algebra & Tensor Engine and verify it against NumPy!

