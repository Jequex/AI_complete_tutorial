# Topic 1.3: Exercises & Solutions — The Scientific Python Ecosystem

Welcome to the problem set for **Topic 1.3: The Scientific Python Ecosystem**!

Practice stride calculations, vectorization speedups, and DataFrame indexing. Work through each exercise before opening the collapsible solution.

---

## 📝 Problem Set

### Problem 1: Hand-Calculating Stride Offsets and Zero-Copy Transposition

Consider a 2D matrix $A$ of shape $(3, 4)$ stored in row-major order:
$$A = \begin{bmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12
\end{bmatrix}$$

1. In contiguous flat memory, what is the exact index list of these 12 values?
2. State the initial row stride $S_0$ and column stride $S_1$ assuming 1 element = 1 stride unit.
3. Compute by hand the flat index of entry $A[2, 3]$ (the number $12$).
4. When transposed into $A^T$ of shape $(4, 3)$, what are the new strides? Compute the flat index for $A^T[3, 2]$ and confirm it accesses the exact same number $12$ with zero memory copies.

<details>
<summary>Click to view solution for Problem 1</summary>

### Solution 1: Stride Calculations & Zero-Copy Transposition

#### Step 1: Flat Memory Layout
Row-major storage packs rows one after another:
`[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]`

#### Step 2: Initial Strides
- Number of columns $C = 4$.
- Moving down 1 row steps over 4 elements: $S_0 = 4$.
- Moving right 1 column steps over 1 element: $S_1 = 1$.
- Shape: `(3, 4)`, Strides: `(4, 1)`.

#### Step 3: Flat Index for $A[2, 3]$
$$\text{flat\_index}(r=2, c=3) = (r \times S_0) + (c \times S_1) = (2 \times 4) + (3 \times 1) = 8 + 3 = \mathbf{11}$$
Index 11 in 0-indexed flat memory points to the 12th element: **$12$**!

#### Step 4: Transposition to $A^T$
- Transposing simply swaps dimensions and strides!
- New Shape: `(4, 3)`, New Strides: `(1, 4)`.
- Coordinate in transposed matrix: $(r=3, c=2)$.
$$\text{flat\_index}_{A^T}(3, 2) = (3 \times S_0^{\text{new}}) + (2 \times S_1^{\text{new}}) = (3 \times 1) + (2 \times 4) = 3 + 8 = \mathbf{11}$$
Both coordinate lookups land on flat memory index **$11$**, giving value **$12$** in $\mathcal{O}(1)$ time without copying any data!
</details>

---

### Problem 2: Vectorization vs. Loop Asymptotics

A neural network layer computes an activation vector $y_i = \max(0.0, x_i)$ for $N = 10,000,000$ floating-point numbers.
- A pure Python loop takes $42\text{ ns}$ per element due to type-checking overhead and heap pointer dereferencing.
- A SIMD vectorized instruction on an AVX-512 CPU processes 16 single-precision floats per CPU cycle ($0.25\text{ ns}$ per cycle).

1. Calculate the total wall-clock runtime for the pure Python loop in seconds.
2. Calculate the theoretical wall-clock runtime for the SIMD vectorized operation in seconds.
3. What is the overall speedup factor?

<details>
<summary>Click to view solution for Problem 2</summary>

### Solution 2: Vectorization Speedup Calculation

#### Step 1: Pure Python Runtime
$$\text{Time}_{\text{Python}} = 10,000,000 \times 42 \times 10^{-9}\text{ s} = \mathbf{0.42\text{ seconds}}$$

#### Step 2: SIMD Vectorized Runtime
$$\text{Total Cycles} = \frac{10,000,000}{16} = 625,000\text{ cycles}$$
$$\text{Time}_{\text{SIMD}} = 625,000 \times 0.25 \times 10^{-9}\text{ s} = 0.00015625\text{ seconds} \approx \mathbf{0.156\text{ ms}}$$

#### Step 3: Speedup Factor
$$\text{Speedup} = \frac{\text{Time}_{\text{Python}}}{\text{Time}_{\text{SIMD}}} = \frac{0.42}{0.00015625} \approx \mathbf{2,688\times \text{ faster!}}$$
*Takeaway:* This is why neural network libraries strictly rely on vectorized contiguous memory.
</details>

---

### Problem 3: DataFrame Indexing Precision (`.loc` vs. `.iloc`)

Given the following pandas DataFrame `df`:
```text
           loss   accuracy
epoch_3    0.45       0.88
epoch_1    0.92       0.64
epoch_2    0.61       0.79
```

1. What will `df.loc['epoch_1', 'loss']` return?
2. What will `df.iloc[1, 0]` return?
3. What will `df.iloc[0, 1]` return?
4. Write the one-line boolean filter to select all epochs where `accuracy > 0.70`.

<details>
<summary>Click to view solution for Problem 3</summary>

### Solution 3: DataFrame Indexing Precision

1. **`df.loc['epoch_1', 'loss']`:** Returns **`0.92`** (label lookup on row `'epoch_1'`, column `'loss'`).
2. **`df.iloc[1, 0]`:** Returns **`0.92`** (row index 1 is `'epoch_1'`, column index 0 is `'loss'`).
3. **`df.iloc[0, 1]`:** Returns **`0.88`** (row index 0 is `'epoch_3'`, column index 1 is `'accuracy'`).
4. **Boolean Filter:**
   ```python
   df[df["accuracy"] > 0.70]
   ```
</details>
