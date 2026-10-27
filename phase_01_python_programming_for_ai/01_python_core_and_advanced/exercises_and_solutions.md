# Topic 1.1: Exercises & Solutions — Core and Advanced Python for AI

Welcome to the problem set for **Topic 1.1: Core and Advanced Python for AI**!

Work through each problem carefully before clicking to reveal the solution. Every exercise features real-world AI patterns, memory mechanics, and hand derivations.

---

## 📝 Problem Set

### Problem 1: Memory Mutability, Reference Aliasing & Default Arguments (Conceptual & Hand Analysis)

#### Part A: Predict the Output
Read the following Python code snippet carefully:

```python
def train_model(epoch_data, training_history=[]):
    training_history.append(epoch_data)
    return training_history

# Run 1
run_1 = train_model("loss: 0.85")
print("Run 1:", run_1)

# Run 2
run_2 = train_model("loss: 0.62")
print("Run 2:", run_2)

# Identity check
print("run_1 is run_2:", run_1 is run_2)
```

1. What will be printed by `print("Run 1:", run_1)`?
2. What will be printed by `print("Run 2:", run_2)`?
3. What will `run_1 is run_2` evaluate to (`True` or `False`), and why?
4. Rewrite the function signature and body so that each call creates a new, independent history list by default.

#### Part B: The Shallow Multiplication Trap
Consider the following tensor weight matrix initialization:

```python
row = [0.0, 0.0, 0.0]
matrix = [row] * 3

matrix[0][1] = 5.0
print("Matrix:", matrix)
```

1. What will be printed?
2. Explain why changing index `[0][1]` affected all other rows.
3. Show the correct list comprehension idiom to create a true $3 \times 3$ grid of independent float values.

<details>
<summary>Click to view solution for Problem 1</summary>

### Solution 1: Memory Mutability & Reference Aliasing

#### Part A: Default Argument Analysis
1. **Output for Run 1:**
   ```python
   Run 1: ['loss: 0.85']
   ```
2. **Output for Run 2:**
   ```python
   Run 2: ['loss: 0.85', 'loss: 0.62']
   ```
3. **Identity Check:**
   ```python
   run_1 is run_2: True
   ```
   *Explanation:* Python creates default argument objects **only once** when the function definition is compiled into bytecode. The list `[]` was created at one specific memory address. Both `run_1` and `run_2` return references pointing to the exact same list in physical RAM (`id(run_1) == id(run_2)`).
4. **Corrected Implementation:**
   ```python
   def train_model(epoch_data, training_history=None):
       if training_history is None:
           training_history = []
       training_history.append(epoch_data)
       return training_history
   ```

#### Part B: The Shallow Multiplication Trap
1. **Output:**
   ```python
   Matrix: [[0.0, 5.0, 0.0], [0.0, 5.0, 0.0], [0.0, 5.0, 0.0]]
   ```
2. **Explanation:**
   The `[row] * 3` expression duplicates the **pointer** to `row` three times; it does not clone the internal list. All three rows point to the exact same list in memory. Mutating `matrix[0][1]` changes that single shared list.
3. **Correct Idiom:**
   ```python
   matrix = [[0.0 for _ in range(3)] for _ in range(3)]
   matrix[0][1] = 5.0
   # Result: [[0.0, 5.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
   ```
</details>

---

### Problem 2: Hand Calculation of a Linear Layer Forward Pass

Given a linear layer with 2 inputs and 2 outputs:
$$\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$$
where:
$$\mathbf{W} = \begin{bmatrix} 2.0 & -1.0 \\ 0.5 & 3.0 \end{bmatrix}, \qquad \mathbf{x} = \begin{bmatrix} 4.0 \\ 1.0 \end{bmatrix}, \qquad \mathbf{b} = \begin{bmatrix} 0.5 \\ -2.0 \end{bmatrix}$$

1. Break down the matrix-vector multiplication step-by-step for neuron $y_1$.
2. Break down the calculation for neuron $y_2$.
3. Write the final output vector $\mathbf{y}$.

<details>
<summary>Click to view solution for Problem 2</summary>

### Solution 2: Hand Calculation of a Linear Layer Forward Pass

#### Step 1: Neuron 1 ($y_1$)
Multiply Row 1 of $\mathbf{W}$ by $\mathbf{x}$, then add $b_1$:
$$y_1 = (W_{11} \cdot x_1) + (W_{12} \cdot x_2) + b_1$$
$$y_1 = (2.0 \times 4.0) + (-1.0 \times 1.0) + 0.5$$
$$y_1 = 8.0 - 1.0 + 0.5 = \mathbf{7.5}$$

#### Step 2: Neuron 2 ($y_2$)
Multiply Row 2 of $\mathbf{W}$ by $\mathbf{x}$, then add $b_2$:
$$y_2 = (W_{21} \cdot x_1) + (W_{22} \cdot x_2) + b_2$$
$$y_2 = (0.5 \times 4.0) + (3.0 \times 1.0) + (-2.0)$$
$$y_2 = 2.0 + 3.0 - 2.0 = \mathbf{3.0}$$

#### Step 3: Final Output Vector
$$\mathbf{y} = \begin{bmatrix} 7.5 \\ 3.0 \end{bmatrix}$$
</details>

---

### Problem 3: Lazy Batch Streamer & Memory Complexity

Suppose you are training on an image dataset of $N = 100,000$ images, where each image consumes $2\text{ MB}$ of memory.

1. If you load all images into a standard Python list eagerly, how many gigabytes (GB) of RAM will be required?
2. If you use a Python generator function with `yield` to stream mini-batches of size $B = 32$, how many megabytes (MB) of RAM are required to hold each active batch?
3. State the asymptotic memory complexity ($\mathcal{O}(\cdot)$) of the eager list vs. the generator streamer in terms of dataset size $N$.

<details>
<summary>Click to view solution for Problem 3</summary>

### Solution 3: Lazy Batch Streamer & Memory Complexity

#### Step 1: Eager List RAM Calculation
$$\text{Total Memory} = N \times \text{Size per image} = 100,000 \times 2\text{ MB} = 200,000\text{ MB}$$
$$\frac{200,000\text{ MB}}{1,024\text{ MB/GB}} \approx \mathbf{195.31\text{ GB of RAM}}$$
*Analysis:* This will immediately crash almost any standard laptop or GPU workstation with an Out-Of-Memory (OOM) error!

#### Step 2: Generator Streamer RAM Calculation
$$\text{Active Memory} = B \times \text{Size per image} = 32 \times 2\text{ MB} = \mathbf{64\text{ MB of RAM}}$$
*Analysis:* The memory consumed is tiny and completely manageable!

#### Step 3: Asymptotic Big-O Complexity
- Eager List: $\mathcal{O}(N)$ (grows linearly with total dataset size).
- Generator Streamer: $\mathcal{O}(B) = \mathcal{O}(1)$ with respect to total dataset size $N$.
</details>
