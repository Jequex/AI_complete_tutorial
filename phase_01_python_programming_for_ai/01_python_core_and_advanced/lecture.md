# Topic 1.1: Core and Advanced Python for AI

Welcome to **Topic 1.1: Core and Advanced Python for AI**! 

Before we construct multi-billion parameter neural networks, train transformers, or build autonomous AI agents, we must master Python—not merely as a simple scripting calculator, but as a high-performance, object-oriented language for orchestrating data pipelines and computational graphs.

---

## 1. 🧒 The 12-Year-Old Analogy: Toy Boxes and Sticky Nametags

### Variables in Memory: Wooden Cubbies vs. Sticky Nametags
In older languages like C or C++, creating a variable is like building a fixed wooden cubby box into a wall. If you carve an `int` box, it can only ever hold an integer.

In Python, **values are physical toy objects floating on a giant table**, and **variables are sticky nametags** that you peel and stick onto those toys.

```text
       C / JAVA (Fixed Wooden Box)                  PYTHON (Sticky Nametags)
       
       ┌─────────────────────────┐               [Nametag: a] ───┐
       │ int x: [      42      ] │                               ▼
       └─────────────────────────┘                      ┌─────────────────┐
       (Variable IS the memory box)                     │   List Object   │ (Address: 0x104b)
                                                        │    [1, 2, 3]    │
                                                        └─────────────────┘
                                                                 ▲
                                                 [Nametag: b] ───┘
```

Imagine **Timmy** and **Maya** in a playroom:
1. Timmy takes a red plastic bucket containing three marbles: `[1, 2, 3]`.
2. He writes `"a"` on a blue sticky note and slaps it onto the bucket: `a = [1, 2, 3]`.
3. Maya comes along, writes `"b"` on a yellow sticky note, and slaps it onto the **exact same bucket**: `b = a`.
4. Now, Barnaby the puppy drops a green marble `4` into bucket `b`: `b.append(4)`.
5. What happens when Timmy looks inside bucket `"a"`? **He sees all four marbles: `[1, 2, 3, 4]`!**

Timmy did not create two separate buckets. There is only **one bucket in physical memory**, and both nametags are stuck to it. This foundational behavior is called **pass-by-object-reference** (or *call-by-sharing*).

---

### Mutability vs. Immutability: Cardboard Boxes vs. Carved Granite Statues

Every object floating on Python's memory table belongs to one of two families:

1. **Mutable (Cardboard Boxes):** You can open the box, add items, remove items, or paint the inside without destroying the box itself. Its memory address (`id(x)`) stays identical.
   - *Examples:* `list`, `dict`, `set`, custom class instances.
2. **Immutable (Carved Granite Statues):** Once chiselled into granite, its value is permanent. If you want to change a number from $5$ to $6$, Python cannot "edit" the $5$; it must chisel a brand-new $6$ statue elsewhere in memory and move your sticky note over to it!
   - *Examples:* `int`, `float`, `str`, `tuple`, `bool`, `frozenset`.

| Data Type | Mutable? | Real-World Analogy | Crucial AI Workload Use Case |
| :--- | :---: | :--- | :--- |
| `int`, `float` | ❌ No | Carved granite coin | Loss values, learning rates, epoch counters |
| `str` | ❌ No | Printed book page | Token text, system prompts, file paths |
| `tuple` | ❌ No | Sealed plastic blister pack | Fixed tensor dimensions: `(batch_size, channels, height, width)` |
| `list` | ✅ Yes | Open cardboard toy box | Dynamically accumulating generated token IDs, loss logs |
| `dict` | ✅ Yes | Filing cabinet with labeled folders | Model hyperparameter configs, JSON payload bodies |
| `set` | ✅ Yes | Drawstring bag of unique marbles | Vocabulary token lookup, deduplicating document IDs |

---

### Identity (`is`) vs. Value Equality (`==`)

Imagine two golden retriever puppies named Barnaby and Buster:
- Both puppies have golden fur, weigh $30\text{ kg}$, and are $6$ months old.
- In value, they are identical: `barnaby == buster` is **`True`** (same characteristics).
- But are they the *same physical animal*? **No!** If Barnaby takes a nap, Buster is still awake: `barnaby is buster` is **`False`**.

```python
x = [1, 2, 3]
y = [1, 2, 3]

# Value Equality: Do they hold the same numbers?
print(x == y)  # True

# Object Identity: Are they the exact same physical memory object?
print(x is y)  # False (two distinct lists living at different memory addresses!)

z = x
# Now z and x share the exact same physical memory address:
print(z is x)  # True
print(id(z) == id(x))  # True
```

> [!WARNING] The Dangerous Mutable Default Argument Trap
> In Python, default function arguments are created **only once** when the function is defined, **not** every time the function is called!
> ```python
> # ❌ CRITICAL BUG: Every training run shares the SAME history list!
> def train_model(loss, history=[]):
>     history.append(loss)
>     return history
>
> # ✅ PRODUCTION FIX: Use None as an immutable sentinel
> def train_model(loss, history=None):
>     if history is None:
>         history = []
>     history.append(loss)
>     return history
> ```

---

## 2. Functions as First-Class Citizens and Closures

### 🧒 The 12-Year-Old Analogy: Maya's Magic Backpack of Recipes
In many programming languages, functions are rigid static machines bolted to the floor. In Python, **functions are physical recipe cards**! You can:
- Put a recipe card into a variable nametag: `my_tool = compute_loss`.
- Hand a recipe card to another chef as an argument: `run_optimization(my_tool)`.
- Write a recipe that creates and returns *another* brand-new recipe!

A **Closure** is like a recipe card packed inside a **magic backpack**:
- Maya packs a backpack with an initial learning rate and decay rate.
- She hands the backpack to Timmy. 
- Even when Maya leaves the kitchen, the backpack **remembers the temperature inside it**!

```python
def make_learning_rate_scheduler(initial_lr: float, decay_factor: float):
    # Enclosing backpack scope variable
    current_lr = initial_lr
    
    def step():
        nonlocal current_lr  # Modifies the variable in the backpack
        current_lr *= decay_factor
        return current_lr
        
    return step

# Maya prepares the backpack:
scheduler = make_learning_rate_scheduler(initial_lr=0.1, decay_factor=0.9)

# Timmy triggers the step recipe across training epochs:
print(scheduler())  # 0.090  (0.10 * 0.9)
print(scheduler())  # 0.081  (0.09 * 0.9)
print(scheduler())  # 0.0729 (0.081 * 0.9)
```

---

### Flexible Argument Unpacking: `*args` and `**kwargs`
Neural network modules must pass unknown configuration options down through multiple nested layers.
- `*args` (Positional arguments): Timmy's picnic row of unwrapped snacks $\to$ packed into a **tuple**.
- `**kwargs` (Keyword arguments): Maya's labeled bento box $\to$ packed into a **dictionary**.

```python
def forward_module(input_tensor, *args, **kwargs):
    print("Positional arguments tuple:", args)
    print("Keyword dictionary:", kwargs)

forward_module([1.0, 2.0], "normalize", dropout=0.2, activation="relu")
# args   -> ('normalize',)
# kwargs -> {'dropout': 0.2, 'activation': 'relu'}
```

---

## 3. Object-Oriented Programming (OOP) & Dunder Magic

### 🧒 The 12-Year-Old Analogy: Universal LEGO Snap-On Adapters
Why do LEGO bricks snap together effortlessly while random wooden toys do not? Because LEGO bricks have standard **studs and sockets**.

In Python, **Dunder Methods** (methods with **D**ouble **Under**scores like `__init__`, `__call__`, and `__len__`) are universal snap-on LEGO adapters:
- If your class has `__call__`, Python treats your object like a function: `model(x)`.
- If your class has `__len__`, Python's `len(dataset)` tape measure clicks right onto it!
- If your class has `__getitem__`, your object gains indexable drawers: `dataset[0]`.
- If your class has `__matmul__`, you can use the `@` matrix multiplication operator: `A @ B`.

---

### 🔬 Mathematical Formulation of a Linear Layer

Every fully connected neural network layer transforms an input feature vector $\mathbf{x} \in \mathbb{R}^d$ into an output vector $\mathbf{y} \in \mathbb{R}^m$:

$$\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$$

Let us break down every term:

$$\underbrace{\mathbf{y}}_{\substack{\text{Output Vector} \\ (m \times 1)}} = \underbrace{\mathbf{W}}_{\substack{\text{Weight Matrix (Knobs)} \\ (m \times d)}} \times \underbrace{\mathbf{x}}_{\substack{\text{Input Feature Vector} \\ (d \times 1)}} + \underbrace{\mathbf{b}}_{\substack{\text{Bias Offset Vector} \\ (m \times 1)}}$$

For each individual output component $j$ (from $j=1$ to $m$):

$$y_j = \left( \sum_{i=1}^d W_{ji} x_i \right) + b_j$$

$$\underbrace{y_j}_{\text{Output prediction } j} = \underbrace{\sum_{i=1}^d}_{\substack{\text{Add up across} \\ \text{all } d \text{ input features}}} \left( \underbrace{W_{ji}}_{\text{Importance weight}} \times \underbrace{x_i}_{\text{Input value}} \right) + \underbrace{b_j}_{\text{Baseline push}}$$

**🧒 What the math is actually doing:**  
For each output neuron, we take all incoming clues ($x_i$), multiply each clue by how much we trust it ($W_{ji}$), sum them up into a single score, and give it a final baseline nudge ($b_j$).

---

### 🔢 Concrete Numerical Hand Walkthrough

Let us calculate a forward pass by hand with pen and paper using simple numbers:

$$\mathbf{W} = \begin{bmatrix} 0.5 & -0.2 \\ 0.1 & 0.8 \end{bmatrix}, \qquad \mathbf{x} = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}, \qquad \mathbf{b} = \begin{bmatrix} 0.10 \\ -0.05 \end{bmatrix}$$

1. **Calculate Output Neuron 1 ($y_1$):**
   - Take Row 1 of $\mathbf{W}$ and dot-product with $\mathbf{x}$, then add $b_1$:
   $$y_1 = (0.5 \times 1.0) + (-0.2 \times 2.0) + 0.10$$
   $$y_1 = 0.50 - 0.40 + 0.10 = \mathbf{0.20}$$

2. **Calculate Output Neuron 2 ($y_2$):**
   - Take Row 2 of $\mathbf{W}$ and dot-product with $\mathbf{x}$, then add $b_2$:
   $$y_2 = (0.1 \times 1.0) + (0.8 \times 2.0) + (-0.05)$$
   $$y_2 = 0.10 + 1.60 - 0.05 = \mathbf{1.65}$$

3. **Final Output Vector:**
   $$\mathbf{y} = \begin{bmatrix} 0.20 \\ 1.65 \end{bmatrix}$$

---

### Python Implementation of the Linear Layer
Notice how we use `__call__` and `__repr__` to turn our Python class into an intuitive neural network block:

```python
class LinearLayer:
    """A fully connected neural network layer: y = Wx + b"""
    def __init__(self, weights: list[list[float]], bias: list[float]):
        self.weights = weights  # Shape: (out_features, in_features)
        self.bias = bias        # Shape: (out_features)

    def __call__(self, x: list[float]) -> list[float]:
        """Calling layer(x) executes the forward mathematical pass."""
        out = []
        for row, b in zip(self.weights, self.bias):
            # Compute dot product: sum(w_i * x_i) + b
            dot_product = sum(w * xi for w, xi in zip(row, x)) + b
            out.append(dot_product)
        return out

    def __repr__(self) -> str:
        in_f = len(self.weights[0])
        out_f = len(self.weights)
        return f"LinearLayer(in_features={in_f}, out_features={out_f})"

# Test with our exact hand-calculated numbers:
layer = LinearLayer(weights=[[0.5, -0.2], [0.1, 0.8]], bias=[0.10, -0.05])
print(layer)             # LinearLayer(in_features=2, out_features=2)
result = layer([1.0, 2.0])
print("Calculated:", result)  # [0.2, 1.65] (Matches our hand derivation!)
```

---

## 4. The Iterator Protocol & Lazy Evaluation

### 🧒 The 12-Year-Old Analogy: The Candy Dump Truck vs. The Pez Dispenser
Imagine Timmy orders 10,000,000 pieces of candy.
- **Eager Loading (The Dump Truck):** A giant dump truck reverses into Timmy's bedroom and dumps all 10 million candies onto the floor at once. The floor cracks under the weight, the house collapses, and Timmy is buried! This is an **Out Of Memory (OOM)** crash.
- **Lazy Evaluation (The Pez Dispenser):** You give Timmy a pocket dispenser. Whenever he is hungry, he clicks the top: **click $\to$ one single candy pops out**. When he swallows it, he clicks it again. 
Timmy can eat 10 billion candies without his bedroom floor ever collapsing!

```text
       EAGER LIST LOADING (O(N) RAM)                LAZY STREAMING GENERATOR (O(1) RAM)
       
       ┌──────────────────────────────┐              ┌───┐
       │ [ 0, 1, 2, 3, 4, ... 10^7 ]  │              │ 0 │ ──> Consumed, memory freed!
       └──────────────────────────────┘              └───┘
       Requires 800 MB to 50 GB RAM!                 ┌───┐
       💥 System crashes (OOM)!                      │ 1 │ ──> Consumed, memory freed!
                                                     └───┘
                                                     Only 1 item lives in memory! (~100 bytes)
```

In Python, the **Iterator Protocol** consists of two simple rules:
1. `__iter__()`: Returns an iterator object (the Pez dispenser).
2. `__next__()`: Delivers the next item, or raises `StopIteration` when the dispenser is empty.

---

## 5. Streaming Big Data with Generators and `yield`

A **Generator** is a magical Python function that uses the keyword `yield` instead of `return`.

When execution encounters `yield`:
1. Python **freezes** the function in place (remembering all loop variables and line numbers).
2. Hands the value out to the caller.
3. When the caller asks for the next item, Python **unfreezes** and resumes right where it stopped!

```python
def batch_generator(data: list, batch_size: int):
    """Yields mini-batches one at a time without copying the dataset."""
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]

# Usage
data_points = [10, 20, 30, 40, 50, 60, 70]
stream = batch_generator(data_points, batch_size=3)

print(next(stream))  # [10, 20, 30]
print(next(stream))  # [40, 50, 60]
print(next(stream))  # [70]
```

### Memory Footprint Comparison: List vs. Generator
- **List Comprehension (Eager):** `[x**2 for x in range(10_000_000)]` consumes **~800 Megabytes** of RAM immediately.
- **Generator Expression (Lazy):** `(x**2 for x in range(10_000_000))` consumes **~120 bytes** of RAM!

$$\underbrace{\text{RAM}_{\text{list}} = \mathcal{O}(N)}_{\text{Memory scales with data size}} \qquad \longleftrightarrow \qquad \underbrace{\text{RAM}_{\text{generator}} = \mathcal{O}(1)}_{\text{Memory stays constant forever!}}$$

---

## 6. Decorators: Wrapping and Instrumenting AI Pipelines

### 🧒 The 12-Year-Old Analogy: Coach Sarah's Stopwatch Cloak
Imagine Timmy is sprinting on the school track.
He doesn't want to carry a heavy stopwatch in his hands while sprinting.
Instead, **Coach Sarah** stands at the sideline:
1. When Timmy is ready, Coach Sarah presses `START` on her stopwatch.
2. Timmy runs his sprint unimpeded.
3. The instant Timmy crosses the finish line, Coach Sarah presses `STOP` and yells: *"12.4 seconds!"*

A **Decorator** is Coach Sarah's stopwatch wrapped around your function! The inner function doesn't need to know anything about timing; the decorator wraps around it like a protective cloak.

```text
               INCOMING CALL: mock_transformer_forward(tokens)
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │             DECORATOR WRAPPER FUNCTION                 │
       │                                                        │
       │  1. Start stopwatch: t0 = perf_counter()               │
       │                                                        │
       │  2. Execute actual model:                              │
       │     result = original_func(*args, **kwargs)            │
       │                                                        │
       │  3. Stop stopwatch: elapsed = perf_counter() - t0      │
       │  4. Log latency: "⏱️ Latency: 1.25 ms"                 │
       │  5. Return result                                      │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
                          RETURN TO USER
```

```python
import time
import functools

def measure_latency(func):
    """Decorator that measures execution time of an inference pass."""
    @functools.wraps(func)  # Keeps original function name and docstring
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"⏱️ [{func.__name__}] Executed in {elapsed_ms:.4f} ms")
        return result
    return wrapper

@measure_latency
def mock_transformer_forward(tokens: list[int]):
    # Simulate heavy neural network computations
    return [t * 2 for t in tokens]

output = mock_transformer_forward([101, 2054, 2003, 102])
```

---

## 7. Custom AI Exceptions & Shape Verification

### 🧒 The 12-Year-Old Analogy: The Shape-Fitting Toy Box
Have you ever seen a baby play with a wooden cube box that has square, circle, and triangle cutouts?
If the toddler tries to jam a triangle block into a round hole, it physically does not fit.

In linear algebra, if Matrix $A$ has shape $(8, 64)$ and Matrix $B$ has shape $(128, 32)$, they **cannot be multiplied**:
- The inner dimensions must match: $64 \neq 128$.
- Instead of letting Python raise a cryptic error 10 hours into training, we create a custom **`ShapeMismatchError`** that sounds a loud alarm immediately!

```python
class AIError(Exception):
    """Base exception class for all AI pipeline errors."""
    pass

class ShapeMismatchError(AIError):
    """Raised when tensor dimensions are incompatible for matrix multiplication."""
    def __init__(self, shape_a: tuple, shape_b: tuple, op: str):
        self.shape_a = shape_a
        self.shape_b = shape_b
        self.op = op
        super().__init__(
            f"Shape Mismatch: Cannot execute '{op}' between shapes {shape_a} and {shape_b}. "
            f"Inner dimensions ({shape_a[1]} != {shape_b[0]}) must match!"
        )

def verify_matmul_compatibility(shape_a: tuple[int, int], shape_b: tuple[int, int]):
    if shape_a[1] != shape_b[0]:
        raise ShapeMismatchError(shape_a, shape_b, op="A @ B")
    return (shape_a[0], shape_b[1])

# Test compatibility
try:
    verify_matmul_compatibility((8, 64), (128, 32))
except ShapeMismatchError as e:
    print(f"❌ Handled Expected Error:\n   {e}")
```

---

## 8. 📋 Mathematical & Technical Symbol Decoder Table

| Symbol | Pronunciation | Plain English Meaning | Concrete Real-World Analogy / Flashcard Example |
| :---: | :---: | :--- | :--- |
| `id(x)` | "I-D of x" | The physical memory address where an object lives in RAM | The GPS coordinate or house street number of a toy box |
| `==` | "equals-equals" | Checks whether two objects contain the same values | Two puppies that have identical fur color and weight |
| `is` | "is" | Checks whether two variables point to the exact same physical object in memory | The exact same puppy tagged with two different collar tags |
| `*args` | "star-args" | Packs extra positional arguments into a tuple | Timmy's picnic bag of unwrapped snacks in a row |
| `**kwargs` | "star-star kwargs" | Packs extra keyword arguments into a key-value dictionary | Maya's bento box with labeled compartments (`sauce="tamari"`) |
| `__init__` | "dunder init" | Class constructor that sets up new instance variables | Assembling a new LEGO kit out of the box |
| `__call__` | "dunder call" | Makes a class instance directly callable with parentheses like `obj(x)` | Adding a big red "Press to Start" button on a toy |
| `__getitem__`| "dunder get-item" | Allows an object to be indexed using square brackets `obj[i]` | Pulling drawer number $i$ out of a dresser |
| `__len__` | "dunder len" | Returns the total count of elements via `len(obj)` | Snapping a tape measure onto your dataset |
| `yield` | "yield" | Pauses a generator function and streams out a single value lazily | A Pez candy dispenser clicking out one treat at a time |
| `@decorator`| "at decorator" | Wraps and augments a function without altering its internal code | Coach Sarah standing by with a stopwatch timing your run |
| $\mathcal{O}(1)$ | "Big-O of 1" | Constant time or memory (never grows, no matter how big data gets) | Looking at the single candy currently held in your hand |
| $\mathcal{O}(N)$ | "Big-O of N" | Linear resource usage (doubling data doubles required memory/time) | Trying to store 10 million candies on your bedroom floor |
| $\mathbf{W}\mathbf{x} + \mathbf{b}$| "W x plus b" | Linear transformation: matrix-vector product plus bias offset | Scaling each clue by trust, summing, and giving a final nudge |

---

## 9. 📌 Summary & Engineering Checklist

- [x] **Variables are sticky nametags**, not wooden cubbies. Assignment (`b = a`) never copies data; it binds another name to the existing object.
- [x] **Beware mutable defaults:** Never write `def train(history=[])`. Always use `history=None` as an immutable sentinel.
- [x] **Dunder methods** (`__call__`, `__len__`, `__getitem__`, `__matmul__`) provide standard LEGO connectors to make custom classes behave like native PyTorch modules.
- [x] **Generators with `yield`** eliminate Out-Of-Memory crashes by streaming large datasets one batch at a time with $\mathcal{O}(1)$ memory.
- [x] **Decorators** cleanly inject instrumentation (timing, caching, validation) around AI functions without cluttering model math.
- [x] **Custom domain exceptions** prevent silent model corruption by catching tensor shape mismatches before expensive training starts.

---

Next up: **Run `python3 experiment.py`** in this folder to see object identity, batch generators, latency decorators, and shape validators execute in real-time!
