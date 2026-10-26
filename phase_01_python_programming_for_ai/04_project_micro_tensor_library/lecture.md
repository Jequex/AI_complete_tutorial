# Topic 1.4: Capstone Project — Building a Micro-Tensor Library

Welcome to **Topic 1.4: Capstone Project — Building a Micro-Tensor Library**. In this capstone project for Phase 1, we demystify modern deep learning frameworks. Have you ever wondered what PyTorch, NumPy, or JAX actually do under the hood when you write:
```python
x = Tensor([[1, 2], [3, 4]])
y = x @ x
print(y)
```
In this lecture, we build a complete, object-oriented, multi-dimensional numerical computing library from scratch in pure Python with zero external dependencies!

---

## 1. Architectural Anatomy of a Tensor

A tensor is **not** a nested list of lists. In real deep learning engines (like PyTorch C++ `ATen`), a tensor consists of three core components:

```
┌────────────────────────────────────────────────────────┐
│                      Tensor Object                     │
├────────────────────────────────────────────────────────┤
│  1. Storage:  [ 1.0, 2.0, 3.0, 4.0 ]  (Flat 1D buffer) │
│  2. Shape:    (2, 2)                  (Dimensions)     │
│  3. Strides:  (2, 1)                  (Step offsets)   │
└────────────────────────────────────────────────────────┘
```

1. **Storage Buffer (`storage`):** A contiguous, one-dimensional array of numbers in memory.
2. **Shape (`shape`):** A tuple of integers $(D_0, D_1, \dots, D_{n-1})$ defining the dimensional extents.
3. **Strides (`strides`):** A tuple of integers $(S_0, S_1, \dots, S_{n-1})$ defining how many positions to skip in the flat 1D buffer to advance by 1 along each dimension.

### The Row-Major Stride Mapping Formula
To access multi-dimensional coordinate $(i_0, i_1, \dots, i_{n-1})$, the flat buffer index is computed as:
$$\text{flat\_index} = \sum_{k=0}^{n-1} i_k \times S_k$$

For a 2D matrix of shape $(R, C)$:
- $S_1 = 1$ (moving right one column steps by 1 element).
- $S_0 = C$ (moving down one row steps over an entire row of $C$ elements).

$$\text{flat\_index}(r, c) = r \times C + c$$

---

## 2. Zero-Copy Operations: Transposition with Strides

One of the most elegant discoveries in computer science is that **transposing a matrix does not require rearranging a single number in memory!**

Consider matrix $A$ of shape $(2, 3)$ with storage `[1, 2, 3, 4, 5, 6]`:
- Initial shape: `(2, 3)`
- Initial strides: `(3, 1)`

To transpose $A$ into $A^T$ of shape $(3, 2)$:
- We simply swap the shape tuple: `(3, 2)`
- And swap the stride tuple: `(1, 3)`!

When reading $A^T$ at coordinate $(r, c)$:
$$\text{flat\_index}(r, c) = r \times 1 + c \times 3$$
- At $(0, 1)$: $0(1) + 1(3) = 3 \implies \text{Value } 4$.
- At $(1, 0)$: $1(1) + 0(3) = 1 \implies \text{Value } 2$.

The entire transposition happens in **$O(1)$ constant time** without copying a single byte!

---

## 3. Matrix Multiplication: The Core Workhorse (`@`)

For two 2D tensors $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$, their matrix product $C = A @ B \in \mathbb{R}^{M \times N}$ is defined entrywise as:
$$C_{i, j} = \sum_{k=0}^{K-1} A_{i, k} \times B_{k, j}$$

Using our `flat_index` function, the multiplication engine simply loops over rows $i$, columns $j$, and inner dimension $k$.

---

## 4. Designing the Micro-Tensor API

Our custom `Tensor` class will support:
1. **Construction:** `Tensor(data, shape=None)` accepts either nested lists or a flat buffer with explicit shape.
2. **Arithmetic:** Elementwise `+`, `-`, `*` with automatic broadcasting.
3. **Linear Algebra:** Matrix multiplication via `@` (`__matmul__`), and transposition via `.T`.
4. **Reductions:** `.sum()` and `.mean()`.
5. **Non-linear Activations:** `.relu()` ($f(x) = \max(0, x)$).
6. **Pretty Printing:** Custom `__repr__` that renders clean 2D grids.

---

## 5. Neural Network Forward Pass from Scratch

Once our `Tensor` class is built, we can construct a 2-layer Multi-Layer Perceptron (MLP) that performs a complete forward pass using native tensor operations:

$$z_1 = x @ W_1 + b_1$$
$$a_1 = \text{ReLU}(z_1)$$
$$\hat{y} = a_1 @ W_2 + b_2$$

This proves that all modern deep learning architectures—from simple MLPs to giant Transformer models—are built entirely upon these fundamental tensor operations!

