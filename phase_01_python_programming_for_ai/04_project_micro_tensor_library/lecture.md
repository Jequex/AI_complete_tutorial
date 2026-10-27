# Topic 1.4: Capstone Project — Building a Micro-Tensor Library

Welcome to the **Capstone Project for Phase 1: Building a Micro-Tensor Library**!

In this capstone, we pull back the curtain on modern deep learning frameworks. Have you ever wondered what PyTorch, NumPy, or JAX actually do under the hood when you write:
```python
x = Tensor([[1.0, 2.0], [3.0, 4.0]])
y = x @ x.T
print(y.relu())
```
Is a tensor magic? No! In this lecture, we build a multi-dimensional tensor computation engine with flat storage, stride calculations, broadcasting, matrix multiplication, and non-linear activation functions from scratch in pure Python with zero external libraries.

---

## 1. 🧒 The 12-Year-Old Analogy: The Conveyor Belt & The Cardboard Divider

### A Tensor is NOT Nested Lists
Many beginners imagine a 2D matrix as a list of lists: `[[1, 2], [3, 4]]`.
In industrial AI systems (like PyTorch C++ `ATen` or CUDA runtimes), nested lists are completely forbidden because they scatter pointers all across RAM.

Instead, a **Tensor** consists of three simple pieces:
1. **The Conveyor Belt (`storage`):** A single, flat, continuous line of numbers in memory.
2. **The Cardboard Divider (`shape`):** A label telling your brain how many rows and columns to imagine: `(2, 2)`.
3. **The Step Ruler (`strides`):** A measurement telling the robot arm how many steps along the conveyor belt it must skip to move down one row or right one column!

```text
       WHAT HUMAN SEES (2D Grid)             WHAT THE COMPUTER ACTUALLY HAS (1D Conveyor Belt)
       
              Col 0   Col 1                 ┌───────┬───────┬───────┬───────┐
       Row 0 [ 1.0     2.0 ]                │  1.0  │  2.0  │  3.0  │  4.0  │
       Row 1 [ 3.0     4.0 ]                └───────┴───────┴───────┴───────┘
                                             Index 0 Index 1 Index 2 Index 3
                                             
                                             Shape:   (2, 2)
                                             Strides: (2, 1)  (Skip 2 items to go down a row!)
```

---

### 🔬 The Stride Coordinate Formula

To find multi-dimensional coordinate $(r, c)$ on a 1D flat ribbon:

$$\text{flat\_index}(r, c) = r \cdot S_0 + c \cdot S_1$$

Let us break down this formula piece by piece:

$$\underbrace{\text{flat\_index}(r, c)}_{\text{Location on the flat belt}} = \left( \underbrace{r}_{\text{Row coordinate}} \times \underbrace{S_0}_{\substack{\text{Row stride} \\ \text{(stride along dimension 0)}}} \right) + \left( \underbrace{c}_{\text{Column coordinate}} \times \underbrace{S_1}_{\substack{\text{Column stride} \\ \text{(stride along dimension 1)}}} \right)$$

**🧒 What the math is actually doing:**  
To find the number at row $r$ and column $c$, multiply the row number by how big an entire row is ($S_0$), and add the column number ($c$).

---

## 2. Zero-Copy Operations: The Transposition Mirror Trick

### 🧒 The 12-Year-Old Analogy: Flipping Your Head vs. Moving the Bricks
Imagine 6 heavy stone blocks arranged in 2 rows of 3 on your driveway:
- If someone asks you to "transpose" the blocks (make it 3 rows of 2), a naive worker would pick up all 6 heavy blocks and carry them to new positions. That takes lots of energy ($\mathcal{O}(N)$ time).
- A smart kid simply **tilts their head sideways by 90 degrees**! The stone blocks never moved an inch, but you are now looking at 3 rows of 2!

In our tensor engine, **transposing never moves a single number in memory!** We simply swap the shape tuple and the strides tuple:

```python
# Matrix A: Shape (2, 3), Strides (3, 1)
# Storage: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

# Transpose A -> A.T:
new_shape = (3, 2)
new_strides = (1, 3)  # Swapped!
```

$$\underbrace{\text{Time to Transpose} = \mathcal{O}(1)}_{\text{Instantaneous! Zero bytes allocated!}}$$

---

## 3. Matrix Multiplication: The Core Workhorse (`@`)

For two 2D tensors $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$, their matrix product $C = A @ B \in \mathbb{R}^{M \times N}$ is defined entrywise as:

$$C_{i, j} = \sum_{k=0}^{K-1} A_{i, k} \times B_{k, j}$$

Let us decode every single symbol:

$$\underbrace{C_{i, j}}_{\substack{\text{Output number at} \\ \text{Row } i \text{, Column } j}} = \underbrace{\sum_{k=0}^{K-1}}_{\substack{\text{Slide across row } i \\ \text{and down column } j}} \left( \underbrace{A_{i, k}}_{\substack{\text{Entry in Matrix } A \\ \text{at row } i \text{, inner step } k}} \times \underbrace{B_{k, j}}_{\substack{\text{Entry in Matrix } B \\ \text{at inner step } k \text{, column } j}} \right)$$

**🧒 What the math is actually doing:**  
Row $i$ of matrix $A$ takes a flying leap and hugs Column $j$ of matrix $B$. Every matching pair of numbers multiplies together, and they all add up into one single number $C_{i, j}$.

---

### 🔢 Concrete Numerical Walkthrough of Matrix Multiplication

Let us calculate $C = A @ B$ step-by-step:

$$A = \begin{bmatrix} 1.0 & 2.0 \\ 3.0 & 4.0 \end{bmatrix}, \qquad B = \begin{bmatrix} 5.0 & 6.0 \\ 7.0 & 8.0 \end{bmatrix}$$

1. **Calculate Entry $C_{0, 0}$ (Row 0 of $A$, Column 0 of $B$):**
   $$C_{0, 0} = (1.0 \times 5.0) + (2.0 \times 7.0) = 5.0 + 14.0 = \mathbf{19.0}$$

2. **Calculate Entry $C_{0, 1}$ (Row 0 of $A$, Column 1 of $B$):**
   $$C_{0, 1} = (1.0 \times 6.0) + (2.0 \times 8.0) = 6.0 + 16.0 = \mathbf{22.0}$$

3. **Calculate Entry $C_{1, 0}$ (Row 1 of $A$, Column 0 of $B$):**
   $$C_{1, 0} = (3.0 \times 5.0) + (4.0 \times 7.0) = 15.0 + 28.0 = \mathbf{43.0}$$

4. **Calculate Entry $C_{1, 1}$ (Row 1 of $A$, Column 1 of $B$):**
   $$C_{1, 1} = (3.0 \times 6.0) + (4.0 \times 8.0) = 18.0 + 32.0 = \mathbf{50.0}$$

5. **Final Result:**
   $$C = \begin{bmatrix} 19.0 & 22.0 \\ 43.0 & 50.0 \end{bmatrix}$$

---

## 4. Building the Pure Python `MicroTensor` Engine

Here is the clean, self-contained implementation of our `MicroTensor` class:

```python
class MicroTensor:
    def __init__(self, data, shape=None):
        if shape is None:
            # Flatten 2D nested list
            self.shape = (len(data), len(data[0]))
            self.storage = [float(val) for row in data for val in row]
        else:
            self.shape = shape
            self.storage = [float(v) for v in data]
        
        # Calculate standard row-major strides
        self.strides = self._compute_strides(self.shape)

    @staticmethod
    def _compute_strides(shape):
        if len(shape) == 1:
            return (1,)
        return (shape[1], 1)

    def _flat_index(self, r, c):
        return r * self.strides[0] + c * self.strides[1]

    def __getitem__(self, idx):
        r, c = idx
        return self.storage[self._flat_index(r, c)]

    @property
    def T(self):
        """Zero-copy transpose: swaps shape and strides!"""
        transposed = MicroTensor(self.storage, shape=(self.shape[1], self.shape[0]))
        transposed.strides = (self.strides[1], self.strides[0])
        return transposed

    def __matmul__(self, other):
        """Matrix multiplication: C = self @ other"""
        M, K1 = self.shape
        K2, N = other.shape
        assert K1 == K2, f"Inner dimensions ({K1} != {K2}) must match!"
        
        out_storage = []
        for i in range(M):
            for j in range(N):
                val = sum(self[i, k] * other[k, j] for k in range(K1))
                out_storage.append(val)
        return MicroTensor(out_storage, shape=(M, N))

    def relu(self):
        """Rectified Linear Unit activation: max(0, x)"""
        return MicroTensor([max(0.0, x) for x in self.storage], shape=self.shape)

    def __repr__(self):
        rows = []
        for i in range(self.shape[0]):
            row_vals = [f"{self[i, j]:6.2f}" for j in range(self.shape[1])]
            rows.append("[" + ", ".join(row_vals) + "]")
        return "MicroTensor([\n  " + ",\n  ".join(rows) + f"\n], shape={self.shape})"
```

---

## 5. End-to-End Neural Network Forward Pass

Now, let us use our `MicroTensor` to perform a 2-layer neural network forward pass:

$$\mathbf{z}_1 = \mathbf{x} @ \mathbf{W}_1 + \mathbf{b}_1$$
$$\mathbf{a}_1 = \text{ReLU}(\mathbf{z}_1)$$
$$\hat{\mathbf{y}} = \mathbf{a}_1 @ \mathbf{W}_2 + \mathbf{b}_2$$

```python
# 1. Input batch of 1 sample with 2 features:
x = MicroTensor([[1.0, 2.0]])

# 2. Layer 1 Weights (2 inputs -> 2 hidden neurons):
W1 = MicroTensor([[1.0, -1.0], [0.5, 2.0]])

# 3. Layer 1 Forward Pass:
z1 = x @ W1  # [[2.0, 3.0]]
a1 = z1.relu() # [[2.0, 3.0]]

# 4. Layer 2 Weights (2 hidden -> 1 output prediction):
W2 = MicroTensor([[0.5], [-1.0]])

# 5. Final Output:
y_hat = a1 @ W2  # (2.0 * 0.5) + (3.0 * -1.0) = 1.0 - 3.0 = -2.0
print("Network Prediction:\n", y_hat)
```

This completes the circle: every deep learning system on Earth, from simple classifiers to ChatGPT, is built upon this exact foundation of tensors, strides, and matrix products!

---

## 6. 📋 Mathematical & Technical Symbol Decoder Table

| Symbol | Pronunciation | Plain English Meaning | Concrete Real-World Analogy / Flashcard Example |
| :---: | :---: | :--- | :--- |
| `storage` | "storage buffer" | 1D flat continuous memory array holding all raw numbers | A single long conveyor belt holding candy pieces in a row |
| `shape` | "shape tuple" | Dimensions describing the multi-dimensional extents | Cardboard dividers showing how many rows and columns exist |
| `strides` | "strides tuple" | Memory offsets required to step along each axis in 1D RAM | Ruler marks telling a robot arm how far to reach on the belt |
| `@` | "matmul operator" | Operator overloading for matrix-matrix or matrix-vector product | Rows of Matrix A taking a flying leap to hug columns of Matrix B |
| `.T` | "dot T (transpose)" | Zero-copy transpose property created by swapping strides | Tilting your head sideways instead of carrying heavy stone blocks |
| $\text{ReLU}(z)$ | "ray-loo of z" | Rectified Linear Unit activation function: $\max(0, z)$ | A one-way electrical diode that lets positive voltage pass |
| $C_{i,j} = \sum_k A_{ik} B_{kj}$ | "C-i-j equals sum of A times B" | Dot product between row $i$ of Matrix A and column $j$ of Matrix B | Multiplying matching pairs of items and adding up total cost |
| $\mathbf{x} \in \mathbb{R}^{B \times D}$ | "x in R B-by-D" | Batch matrix of $B$ samples, each having $D$ feature numbers | A clipboard holding $B$ patient forms, each with $D$ medical vitals |

---

## 7. 📌 Summary & Capstone Checklist

- [x] A tensor is **not** a nested list; it is a contiguous 1D flat memory buffer coupled with `shape` and `strides`.
- [x] **Transposition** is a zero-copy $\mathcal{O}(1)$ operation achieved simply by swapping shape and stride metadata.
- [x] **Matrix multiplication (`@`)** computes entrywise dot products between rows of $A$ and columns of $B$.
- [x] **Activation functions (`ReLU`)** introduce non-linearities, allowing networks to approximate complex decision boundaries.
- [x] You have constructed an entire micro-deep learning framework from first principles in pure Python!

---

Next up: **Run `python3 experiment.py`** in this folder to see the complete `MicroTensor` test suite and neural network forward pass execute!
