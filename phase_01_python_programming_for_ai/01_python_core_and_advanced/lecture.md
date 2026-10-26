# Topic 1.1: Core and Advanced Python for AI

Welcome to **Topic 1.1: Core and Advanced Python for AI**. Before we train multi-billion parameter neural networks, we must master Python not just as a scripting tool, but as a high-performance, object-oriented language for managing data pipelines and computational graphs.

---

## 1. The Sticky Nametag Mental Model: Variables, Types, and Memory

### The 12-Year-Old Analogy: Toy Boxes and Nametags
In languages like C or Java, a variable is like a fixed wooden cubby box. If you declare `int x;`, you carved a wooden box that can only ever hold an integer.
In Python, **variables are sticky nametags**, and **values are physical objects floating on a table**.

When you write:
```python
a = [1, 2, 3]
b = a
```
You did **not** clone the list. You just slapped two nametags (`a` and `b`) onto the **exact same toy box** on the table!

```
[Nametag: a] ───┐
                ▼
       ┌─────────────────┐
       │   List Object   │ (Memory Address: 0x104b2a8e0)
       │    [1, 2, 3]    │
       └─────────────────┘
                ▲
[Nametag: b] ───┘
```

If you modify the contents through `b.append(4)`, looking at `a` reveals `[1, 2, 3, 4]`. This behavior is called **pass-by-object-reference** (or *call-by-sharing*).

### Mutability vs. Immutability
Every object in Python is either:
1. **Mutable:** Its internal state can be changed without altering its identity (memory address). Examples: `list`, `dict`, `set`, custom class instances.
2. **Immutable:** Its value is permanently etched in stone. To change it, Python must construct a brand-new object elsewhere in RAM. Examples: `int`, `float`, `str`, `tuple`, `bool`, `frozenset`.

| Data Type | Mutable? | Common AI Use Case |
| :--- | :---: | :--- |
| `int`, `float` | ❌ No | Learning rates, loss scalar values, batch indices |
| `str` | ❌ No | Prompt strings, token text, file paths |
| `tuple` | ❌ No | Tensor dimensions / shapes: `(batch_size, channels, height, width)` |
| `list` | ✅ Yes | Dynamically accumulating generated tokens, loss histories |
| `dict` | ✅ Yes | Model configuration configs, JSON payloads, hyperparameter maps |
| `set` | ✅ Yes | Unique vocabulary tokens, deduplicating document IDs |

### Identity (`is`) vs. Equality (`==`)
- `==` checks **value equality**: *"Do these two boxes hold the same content?"*
- `is` checks **identity**: *"Are these two nametags stuck to the exact same physical box in memory?"* (i.e., `id(x) == id(y)`).

```python
x = [1, 2, 3]
y = [1, 2, 3]
print(x == y)  # True  (same numbers inside)
print(x is y)  # False (two distinct list objects in RAM)

z = x
print(z is x)  # True  (same memory address)
```

> [!WARNING] The Dangerous Default Argument Trap
> In Python, default function arguments are evaluated **once when the function is defined**, not every time it is called.
> ```python
> # ❌ BUG: The same list persists across calls!
> def append_loss(val, history=[]):
>     history.append(val)
>     return history
>
> # ✅ FIX: Use None as a sentinel
> def append_loss(val, history=None):
>     if history is None:
>         history = []
>     history.append(val)
>     return history
> ```

---

## 2. Functions as First-Class Citizens, Closures, and Arguments

In Python, functions are **objects**. You can assign them to variables, store them in dictionaries, pass them into other functions as arguments, and return them from functions.

### Flexible Argument Unpacking: `*args` and `**kwargs`
Neural network layers frequently need to pass arbitrary configuration options down to child modules:

```python
def forward(input_tensor, *args, **kwargs):
    # args is a tuple of positional arguments: (arg1, arg2)
    # kwargs is a dictionary of keyword arguments: {"activation": "relu", "dropout": 0.1}
    pass
```

### Closures: Functions that Remember Their Environment
A **closure** is a nested function that retains access to variables from its enclosing scope, even after the outer function has finished executing:

```python
def make_learning_rate_scheduler(initial_lr: float, decay_factor: float):
    # Enclosing scope variable
    current_lr = initial_lr
    
    def step():
        nonlocal current_lr
        current_lr *= decay_factor
        return current_lr
        
    return step

scheduler = make_learning_rate_scheduler(0.1, 0.9)
print(scheduler())  # 0.09
print(scheduler())  # 0.081
```

---

## 3. Object-Oriented Programming (OOP) and Dunder Methods

Deep learning libraries like PyTorch are built on object-oriented abstractions: `nn.Module`, `Dataset`, `DataLoader`, and `Optimizer`.

### Special "Dunder" (Double Underscore) Methods
Dunder methods allow your custom classes to integrate seamlessly with Python syntax (operators, functions like `len()`, string representations, indexing).

| Dunder Method | Triggered By | AI Relevance |
| :--- | :--- | :--- |
| `__init__(self, ...)` | `Model(...)` | Initializes model weights and layer parameters |
| `__repr__(self)` | `repr(obj)` or terminal print | Developer-friendly debugging representation |
| `__call__(self, x)` | `model(x)` | Makes an instance callable like a mathematical function $f(x)$ |
| `__getitem__(self, idx)` | `dataset[idx]` | Enables indexing, essential for custom PyTorch datasets |
| `__len__(self)` | `len(dataset)` | Returns the total count of training samples |
| `__add__(self, other)` | `a + b` | Operator overloading for vector or tensor addition |
| `__matmul__(self, other)`| `a @ b` | Operator overloading for matrix multiplication |

### Example: A Minimal Neural Network Layer
```python
class LinearLayer:
    """A single fully-connected neural network layer: y = Wx + b"""
    def __init__(self, weights: list[list[float]], bias: list[float]):
        self.weights = weights  # Matrix of shape (out_features, in_features)
        self.bias = bias        # Vector of shape (out_features)

    def __call__(self, x: list[float]) -> list[float]:
        """Calling layer(x) executes the forward pass."""
        out = []
        for row, b in zip(self.weights, self.bias):
            # Compute dot product: sum(w_i * x_i) + b
            dot_product = sum(w * xi for w, xi in zip(row, x)) + b
            out.append(dot_product)
        return out

    def __repr__(self) -> str:
        return f"LinearLayer(in_features={len(self.weights[0])}, out_features={len(self.weights)})"

# Instantiation & execution
layer = LinearLayer(weights=[[0.5, -0.2], [0.1, 0.8]], bias=[0.1, -0.05])
print(layer)             # LinearLayer(in_features=2, out_features=2)
output = layer([1.0, 2.0]) # Invokes __call__
print("Output:", output) # Output: [0.2, 1.65]
```

---

## 4. The Iterator Protocol and Lazy Evaluation

### The Problem: RAM Limits with Big Data
If you train on a 500-Gigabyte text dataset (like Common Crawl), you cannot load it into RAM with:
```python
# 💥 CRASH: Out Of Memory (OOM)!
with open("huge_internet_corpus.txt") as f:
    data = f.readlines() # Attempts to load all 500 GB into memory
```

### The Solution: Iterators and The Iterator Protocol
An **iterable** is any object that can return an **iterator** via `iter(obj)`.
An **iterator** is a bookmark object that produces the next item on demand via `next(iterator)`, and raises `StopIteration` when finished.

```python
class CountDown:
    def __init__(self, start: int):
        self.current = start

    def __iter__(self):
        return self  # The object is its own iterator

    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        val = self.current
        self.current -= 1
        return val

for num in CountDown(3):
    print(num)  # Prints 3, 2, 1
```

---

## 5. Generators: Streaming Infinite Data with `yield`

A **generator** is a function that produces a sequence of results lazily using the `yield` keyword instead of `return`.

When execution hits `yield`, Python:
1. Freezes the function's entire execution state (all local variables and instruction pointer).
2. Yields the value to the caller.
3. Resumes immediately after the `yield` statement on the next call to `next()`.

```python
def batch_generator(data: list, batch_size: int):
    """Yields mini-batches one at a time without copying the entire dataset."""
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]

# Usage
data_points = list(range(10))
batches = batch_generator(data_points, batch_size=4)

print(next(batches))  # [0, 1, 2, 3]
print(next(batches))  # [4, 5, 6, 7]
print(next(batches))  # [8, 9]
```

### Generator Expressions: Memory-Efficient Comprehensions
- List comprehension (eager, allocates memory immediately):
  `squares_list = [x**2 for x in range(1_000_000)]` (takes ~8 MB RAM)
- Generator expression (lazy, calculates on the fly):
  `squares_gen = (x**2 for x in range(1_000_000))` (takes ~100 bytes RAM!)

---

## 6. Decorators: Wrapping and Instrumenting AI Pipelines

A **decorator** is a function that takes another function as input, extends or alters its behavior, and returns a modified function.

### The Syntax Sugar
Writing:
```python
@my_decorator
def train_step():
    pass
```
is completely equivalent to:
```python
def train_step():
    pass
train_step = my_decorator(train_step)
```

### Building an Inference Timer Decorator
When serving models in production, you must monitor the millisecond latency of every inference pass.

```python
import time
import functools

def measure_latency(func):
    """Decorator that measures execution time of an inference or training function."""
    @functools.wraps(func)  # Preserves function name and docstring
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start_time) * 1000  # in ms
        print(f"⏱️ [{func.__name__}] Latency: {elapsed:.3f} ms")
        return result
    return wrapper

@measure_latency
def mock_llm_generate(prompt: str) -> str:
    # Simulate matrix math workload
    total = sum(i * 0.001 for i in range(100_000))
    return f"Response to: '{prompt}' (result: {total:.1f})"

response = mock_llm_generate("Explain quantum computing")
```

---

## 7. Robust Error Handling & Custom AI Exceptions

In production deep learning pipelines, standard generic errors like `ValueError` or `RuntimeError` do not clearly convey AI-specific failures (such as tensor shape mismatches or GPU memory exhaustion).

### Defining Custom Exceptions
```python
class AIError(Exception):
    """Base class for all domain-specific AI errors."""
    pass

class ShapeMismatchError(AIError):
    """Raised when tensor dimensions are incompatible for matrix operations."""
    def __init__(self, shape_a: tuple, shape_b: tuple, operation: str):
        self.shape_a = shape_a
        self.shape_b = shape_b
        self.operation = operation
        super().__init__(
            f"Cannot perform '{operation}' on incompatible shapes {shape_a} and {shape_b}."
        )

# Safe multiplication check
def safe_matrix_multiply(shape_a: tuple[int, int], shape_b: tuple[int, int]):
    if shape_a[1] != shape_b[0]:
        raise ShapeMismatchError(shape_a, shape_b, operation="Matrix Multiplication (A @ B)")
    return (shape_a[0], shape_b[1])

# Usage
try:
    result_shape = safe_matrix_multiply((32, 512), (256, 128))
except ShapeMismatchError as err:
    print(f"⚠️ Caught expected error: {err}")
```

---

## 8. Summary & Mental Checklist

1. **Variables are labels, not boxes:** Assignment never copies data; it binds a name to an object in memory.
2. **Beware mutable defaults:** Never write `def train(history=[])`. Use `None`.
3. **Use dunder methods:** Implement `__call__`, `__len__`, and `__getitem__` to build elegant PyTorch-style components.
4. **Streams over arrays:** Use generators (`yield`) for huge datasets to prevent Out-Of-Memory crashes.
5. **Instrument with decorators:** Measure latency, cache results, and enforce pre-conditions cleanly with `@functools.wraps`.

