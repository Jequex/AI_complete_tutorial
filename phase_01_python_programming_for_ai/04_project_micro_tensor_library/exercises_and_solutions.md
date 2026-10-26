# Topic 1.4: Exercises & Solutions — Capstone Micro-Tensor Library

---

## Problem Set Overview

| Problem | Focus | Difficulty |
| :--- | :--- | :--- |
| **1** | **Stride Calculus & Zero-Copy Tensor Reshaping** | Medium |
| **2** | **Broadcasting Mechanics for Bias Addition $(M, N) + (N,)$** | Medium |
| **3** | **Mean Squared Error (MSE) Loss Implementation from Scratch** | Challenging |

---

## Problem 1: Stride Calculus & Zero-Copy Tensor Reshaping

### The Context
In PyTorch and NumPy, calling `.reshape()` or `.view()` does not copy data if the underlying buffer is C-contiguous. However, if a tensor has been transposed (`.T`), its strides are non-contiguous, meaning an eager copy must be performed before reshaping.

### Part A: Mathematical Stride Derivation
A tensor has shape $(D_0, D_1, \dots, D_{k-1})$.
1. Write the recurrence relation for C-contiguous strides $(S_0, S_1, \dots, S_{k-1})$.
2. Calculate the strides for shape `(2, 3, 4)`.
3. If tensor $A$ with shape `(2, 3)` has strides `(3, 1)`, and is transposed to $A^T$ with shape `(3, 2)` and strides `(1, 3)`, explain why $A^T$ is no longer C-contiguous.

### Part B: Implementing `.reshape()`
Write the method `reshape(self, new_shape: tuple[int, ...]) -> Tensor` for our `Tensor` class:
- If the tensor is contiguous, return a new `Tensor` view sharing `self.storage` with new strides.
- If the tensor is non-contiguous (e.g. after transposition), allocate a new contiguous copy of storage.

---

### Solution 1

#### Part A: Stride Derivation
1. **Recurrence Relation:**
   $$S_{k-1} = 1$$
   $$S_{i} = S_{i+1} \times D_{i+1} \quad \text{for } i = k-2, k-3, \dots, 0$$
2. **For Shape `(2, 3, 4)`:**
   - $S_2 = 1$
   - $S_1 = S_2 \times 4 = 4$
   - $S_0 = S_1 \times 3 = 12$
   $$\text{Strides} = (12, 4, 1)$$
3. **Contiguity of $A^T$:**
   A 2D tensor of shape $(3, 2)$ is C-contiguous if its stride tuple is $(2, 1)$ (moving down a row skips 2 elements).
   However, $A^T$ has strides $(1, 3)$. Moving to the next column skips 3 elements in memory instead of 1. Because elements in a row are not adjacent in RAM, it is non-contiguous.

#### Part B: Method Implementation
```python
def is_contiguous(self) -> bool:
    """Checks if current strides match canonical C-contiguous strides."""
    return self.strides == self._default_strides(self.shape)

def reshape(self, new_shape: tuple[int, ...]) -> "Tensor":
    # Verify element count matches
    expected_count = 1
    for d in new_shape:
        expected_count *= d
    
    current_count = 1
    for d in self.shape:
        current_count *= d
        
    if expected_count != current_count:
        raise ValueError(f"Cannot reshape {self.shape} ({current_count} elements) into {new_shape} ({expected_count} elements).")

    if self.is_contiguous():
        # Zero-copy view!
        return Tensor(None, shape=new_shape, storage=self.storage, strides=self._default_strides(new_shape), offset=self.offset)
    else:
        # Non-contiguous: materialize contiguous copy
        new_storage = self._to_contiguous_list()
        return Tensor(None, shape=new_shape, storage=new_storage)
```

---

## Problem 2: Broadcasting Mechanics for Bias Addition $(M, N) + (N,)$

### The Context
In neural networks, the linear layer output $Z = X W$ produces a 2D matrix of shape $(M, N)$, where $M$ is the batch size and $N$ is the number of neurons. The bias $b$ is a 1D vector of shape $(N,)$. Adding $Z + b$ requires *broadcasting* the bias vector across all $M$ rows of $Z$.

### Mathematical Verification
Given matrix $Z \in \mathbb{R}^{2 \times 3}$ and bias $b \in \mathbb{R}^3$:
$$Z = \begin{bmatrix} 1.0 & 2.0 & 3.0 \\ 4.0 & 5.0 & 6.0 \end{bmatrix}, \quad b = \begin{bmatrix} 0.5 \\ -0.5 \\ 1.0 \end{bmatrix}$$

1. State the resulting shape of $Z + b$.
2. Compute the exact resulting numerical values of $Z + b$ by hand.
3. Write the flat memory loop that computes this sum into an output buffer of size $M \times N$.

---

### Solution 2

1. **Resulting Shape:**
   $$\text{Shape}(Z + b) = (2, 3)$$
2. **Hand Calculation:**
   $$\text{Row 0: } [1.0 + 0.5, 2.0 + (-0.5), 3.0 + 1.0] = [1.5, 1.5, 4.0]$$
   $$\text{Row 1: } [4.0 + 0.5, 5.0 + (-0.5), 6.0 + 1.0] = [4.5, 4.5, 7.0]$$
   $$Z + b = \begin{bmatrix} 1.5 & 1.5 & 4.0 \\ 4.5 & 4.5 & 7.0 \end{bmatrix}$$

3. **Flat Memory Implementation:**
```python
def add_bias(Z: Tensor, b: Tensor) -> Tensor:
    M, N = Z.shape
    out = [0.0] * (M * N)
    for i in range(M):
        for j in range(N):
            out[i * N + j] = Z[i, j] + b[j]
    return Tensor(None, shape=(M, N), storage=out)
```

---

## Problem 3: Mean Squared Error (MSE) Loss Function

### The Context
During supervised learning, a loss function quantifies how far predictions $\hat{y}$ deviate from true targets $y$. For $N$ samples and $K$ output dimensions, the Mean Squared Error (MSE) is:
$$\mathcal{L}_{\text{MSE}}(y, \hat{y}) = \frac{1}{N \times K} \sum_{i=1}^N \sum_{j=1}^K (\hat{y}_{i, j} - y_{i, j})^2$$

### Requirements
1. Implement a standalone function `mse_loss(y_true: Tensor, y_pred: Tensor) -> float` using our `Tensor` class.
2. Verify with test inputs:
   $$y = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}, \quad \hat{y} = \begin{bmatrix} 0.8 & 0.1 \\ 0.2 & 0.9 \end{bmatrix}$$
3. Calculate the exact theoretical loss value by hand.

---

### Solution 3

#### 1. Hand Calculation
Differences $\hat{y} - y$:
- $(0, 0)$: $0.8 - 1.0 = -0.2 \implies (-0.2)^2 = 0.04$
- $(0, 1)$: $0.1 - 0.0 = 0.1 \implies (0.1)^2 = 0.01$
- $(1, 0)$: $0.2 - 0.0 = 0.2 \implies (0.2)^2 = 0.04$
- $(1, 1)$: $0.9 - 1.0 = -0.1 \implies (-0.1)^2 = 0.01$

Sum of Squared Errors:
$$\sum = 0.04 + 0.01 + 0.04 + 0.01 = 0.10$$
Mean over $N \times K = 2 \times 2 = 4$ elements:
$$\mathcal{L}_{\text{MSE}} = \frac{0.10}{4} = 0.0250$$

#### 2. Code Implementation
```python
def mse_loss(y_true: Tensor, y_pred: Tensor) -> float:
    """Calculates Mean Squared Error loss between two identical-shaped tensors."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} != {y_pred.shape}")
    
    # Elementwise subtraction
    diff_storage = [
        (yp - yt) ** 2 
        for yt, yp in zip(y_true._to_contiguous_list(), y_pred._to_contiguous_list())
    ]
    return sum(diff_storage) / len(diff_storage)

# Verification
y_true = Tensor([[1.0, 0.0], [0.0, 1.0]])
y_pred = Tensor([[0.8, 0.1], [0.2, 0.9]])
loss = mse_loss(y_true, y_pred)
assert abs(loss - 0.0250) < 1e-6
print(f"Calculated MSE Loss: {loss:.4f} (Matches exact theoretical 0.0250!)")
```

