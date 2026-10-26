# Topic 1.1: Exercises & Solutions — Core and Advanced Python for AI

---

## Problem Set Overview

| Problem | Focus | Difficulty |
| :--- | :--- | :--- |
| **1** | **Memory Mutability, Reference Aliasing & Default Arguments** | Medium |
| **2** | **Custom PyTorch-style Dataset & Lazy Batch Streamer** | Medium |
| **3** | **Production AI Decorators: Shape Validation & Retry Fallback** | Challenging |

---

## Problem 1: Memory Mutability, Reference Aliasing & Default Arguments

### The Context
In AI training scripts, subtle bugs often emerge from unintended reference aliasing (multiple variables pointing to the same list of loss values or model weights) or default mutable arguments accumulating state across epochs.

### Part A: Predict the Output
Read the following Python code carefully:

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

### Part B: List Copying & Aliasing
Consider the following tensor weight initialization snippet:

```python
row = [0.0, 0.0, 0.0]
matrix = [row] * 3

matrix[0][1] = 5.0
print("Matrix:", matrix)
```

1. What will be printed?
2. Explain why changing index `[0][1]` affected other rows.
3. Show the correct list comprehension idiom to create a true $3 \times 3$ grid of distinct zero floats.

---

### Solution 1

#### Part A: Default Argument Analysis
1. **Output for Run 1:**
   `Run 1: ['loss: 0.85']`
2. **Output for Run 2:**
   `Run 2: ['loss: 0.85', 'loss: 0.62']`
3. **Identity Check:**
   `run_1 is run_2: True`
   *Explanation:* Python creates default arguments **once** when the function definition is compiled into bytecode. The list object `[]` is created in memory once. Both `run_1` and `run_2` reference that identical list in memory (`id(run_1) == id(run_2)`).
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
   `Matrix: [[0.0, 5.0, 0.0], [0.0, 5.0, 0.0], [0.0, 5.0, 0.0]]`
2. **Explanation:**
   The `* 3` operator on `[row]` does not copy the internal list; it creates three references pointing to the exact same list `row` in memory. Modifying `matrix[0][1]` updates that single shared list.
3. **Correct Idiom:**
   ```python
   # Creates 3 independent list objects in memory
   matrix = [[0.0 for _ in range(3)] for _ in range(3)]
   matrix[0][1] = 5.0
   # Matrix: [[0.0, 5.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
   ```

---

## Problem 2: Custom PyTorch-style Dataset & Lazy Batch Streamer

### The Context
Machine learning training requires feeding batches of data into the model. Real-world datasets often do not fit in RAM, necessitating an iterator or generator pipeline.

### Requirements
1. Implement a class `TextClassificationDataset`:
   - `__init__(self, texts: list[str], labels: list[int])`: Validates that `texts` and `labels` have the same length (raise `ValueError` otherwise).
   - `__len__(self) -> int`: Returns the number of items.
   - `__getitem__(self, idx: int) -> dict`: Returns `{"text": texts[idx], "label": labels[idx], "length": len(texts[idx].split())}`.
2. Implement a standalone generator function `stream_batches(dataset, batch_size: int, drop_last: bool = False)`:
   - Uses `yield` to yield a list of batch dictionaries.
   - If `drop_last=True`, discard any final partial batch smaller than `batch_size`.
   - If `drop_last=False`, yield the partial batch.

---

### Solution 2

```python
class TextClassificationDataset:
    def __init__(self, texts: list[str], labels: list[int]):
        if len(texts) != len(labels):
            raise ValueError(
                f"Mismatch: Got {len(texts)} texts and {len(labels)} labels."
            )
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        return {
            "text": text,
            "label": self.labels[idx],
            "length": len(text.split())
        }

def stream_batches(dataset, batch_size: int, drop_last: bool = False):
    """Lazily yields mini-batches from a dataset object."""
    batch = []
    for i in range(len(dataset)):
        batch.append(dataset[i])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch and not drop_last:
        yield batch

# Verification
texts = ["hello world", "artificial intelligence", "deep learning", "transformer attention", "python"]
labels = [0, 1, 1, 1, 0]
ds = TextClassificationDataset(texts, labels)

# With drop_last=False: yields batches of size 2, 2, 1
batches_keep = list(stream_batches(ds, batch_size=2, drop_last=False))
assert len(batches_keep) == 3
assert len(batches_keep[-1]) == 1

# With drop_last=True: yields batches of size 2, 2
batches_drop = list(stream_batches(ds, batch_size=2, drop_last=True))
assert len(batches_drop) == 2
```

---

## Problem 3: Production AI Decorator: Shape Validation & Retry Fallback

### The Context
In production AI inference servers, passing an input with the wrong matrix dimensions will trigger low-level C++ CUDA crashes. Furthermore, intermittent GPU memory spikes can occasionally fail a request. Writing reusable decorators shields your serving endpoint from these issues.

### Requirements
Create a decorator `@validate_input_dim(expected_dim: int)` that:
1. Inspects the first argument `x` passed to the decorated function.
2. Assumes `x` is a nested list representing a matrix, and checks if its number of columns matches `expected_dim`.
3. If it does not match, raises a custom `ShapeMismatchError`.
4. Otherwise, runs the decorated function and returns its result.
5. Uses `functools.wraps` to preserve the function's metadata.

---

### Solution 3

```python
import functools

class ShapeMismatchError(Exception):
    """Raised when tensor dimensions are invalid."""
    pass

def validate_input_dim(expected_dim: int):
    """Decorator factory that verifies input feature dimension."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(x, *args, **kwargs):
            # Verify x has rows and columns
            if not isinstance(x, list) or not x or not isinstance(x[0], list):
                raise TypeError("Input 'x' must be a 2D list matrix.")
            
            actual_dim = len(x[0])
            if actual_dim != expected_dim:
                raise ShapeMismatchError(
                    f"Dimension Error in '{func.__name__}': Expected {expected_dim} features, "
                    f"but received input with {actual_dim} columns."
                )
            return func(x, *args, **kwargs)
        return wrapper
    return decorator

# Usage
@validate_input_dim(expected_dim=4)
def run_model_inference(x: list[list[float]]) -> list[float]:
    """Runs a simulated forward pass on a batch of 4-dimensional vectors."""
    return [sum(row) for row in x]

# Test Valid Input (Batch size 2, feature dim 4)
valid_batch = [[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 1.0, 1.0]]
print("Inference Result:", run_model_inference(valid_batch))  # [10.0, 3.0]

# Test Invalid Input (Batch size 1, feature dim 3)
try:
    invalid_batch = [[1.0, 2.0, 3.0]]
    run_model_inference(invalid_batch)
except ShapeMismatchError as err:
    print("Caught Expected Validation Error:", err)
```

