# Topic 12.3: Scalar Autograd Engine From Scratch

---

## 1. Executive Summary & The 12-Year-Old Analogy

At the heart of frameworks like PyTorch (`torch.Tensor`) and Andrej Karpathy’s `micrograd` lies an astonishingly elegant software architecture: a lightweight Python class representing a computational node that automatically tracks mathematical operations and records backwards differentiation closures.

In this lecture, we demystify the magic of automatic differentiation by constructing a complete, production-grade scalar autograd engine from first principles in pure Python.

```
                  ┌─────────────────────────────────────────┐
                  │               Value Object              │
                  ├─────────────────────────────────────────┤
                  │  data:      float (numerical value)     │
                  │  grad:      float (accumulated dL/dv)   │
                  │  _prev:     set (pointers to children)  │
                  │  _op:       str ("+", "*", "tanh"...)   │
                  │  _backward: callable closure            │
                  └─────────────────────────────────────────┘
```

---

### The 12-Year-Old Analogy: The Magic Mathematical Echo

Imagine you are standing at the edge of the Grand Canyon shouting mathematical commands:
- You give your friend two magic stones labeled **$x$** and **$w$**.
- When you tell them to multiply the stones ($z = x \times w$), the stones produce a new stone $z$ with a tiny magical walkie-talkie attached to it.
- In that walkie-talkie, the stone records a whisper: *"To tell $x$ its impact, multiply my incoming signal by $w$; to tell $w$, multiply my signal by $x$."*
- You continue building towers of stones until you produce the final golden stone, the Loss $L$.

**The Magic Echo:**
Now, you tap the golden stone with a tuning fork ($L.\text{grad} = 1.0$).
- The golden stone whispers through its walkie-talkie back to the stones that built it.
- Each stone receives the incoming acoustic vibrations, multiplies them by its recorded formula, and **echoes** the vibration backward to its parents.
- If a stone receives echoes from two different directions, it **adds their vibrations together** (it doesn't wipe out the first echo!).
- When the echo finishes traveling all the way back to $x$ and $w$, the strength of the vibration left on each stone is **precisely its mathematical derivative**!

That is an autograd engine: a network of objects that record backward whispering closures during the forward pass and echo gradients backward on demand.

---

## 2. Mathematical Symbol Decoder

| Symbol | Software Attribute | Mathematical Meaning |
| :--- | :--- | :--- |
| $v.data \in \mathbb{R}$ | `self.data` | The numerical forward evaluation of node $v$ |
| $v.grad \in \mathbb{R}$ | `self.grad` | Accumulated adjoint $\frac{\partial L}{\partial v}$ |
| $v.\_prev \subset \mathcal{V}$ | `self._prev` | Direct dependency predecessors $\{u \mid (u, v) \in \mathcal{E}\}$ |
| $v.\_op \in \text{String}$ | `self._op` | Primitive operator symbol (`"+"`, `"-"`, `"*"`, `"relu"`, etc.) |
| $v.\_backward()$ | `self._backward` | Zero-argument closure implementing local vector-Jacobian product |
| $\text{Topo}(\mathcal{G})$ | `topo` list | Linear sequence of nodes satisfying dependency ordering |

---

## 3. Architecture of the `Value` Class

Every differentiable quantity in our engine is an instance of `Value`. It encapsulates five core attributes:

```python
class Value:
    def __init__(self, data: float, _children: tuple = (), _op: str = ""):
        self.data: float = float(data)
        self.grad: float = 0.0
        self._prev: set = set(_children)
        self._op: str = _op
        self._backward = lambda: None
```

### 3.1 Closures and Backward Callbacks
The genius of this design is that **differentiation logic is defined locally inside each mathematical operator**:

When we write `c = a * b`:
```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
        # Multivariable chain rule: dL/da += dL/dc * (dc/da) = out.grad * b
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    out._backward = _backward
    return out
```

Notice how Python closures work:
- `_backward` captures references to `self`, `other`, and `out`.
- During the forward pass, `_backward` is merely created and stored; it does **not** execute.
- When `L.backward()` is called, `out.grad` has already been populated, allowing `self.grad` and `other.grad` to receive their exact contributions!

---

## 4. Complete Library of Differentiable Primitives

### 4.1 Arithmetic Operations

1. **Addition ($z = x + y$):**
   $$\frac{\partial z}{\partial x} = 1.0, \quad \frac{\partial z}{\partial y} = 1.0$$
   ```python
   self.grad += 1.0 * out.grad
   other.grad += 1.0 * out.grad
   ```

2. **Multiplication ($z = x \cdot y$):**
   $$\frac{\partial z}{\partial x} = y, \quad \frac{\partial z}{\partial y} = x$$
   ```python
   self.grad += other.data * out.grad
   other.grad += self.data * out.grad
   ```

3. **Power ($z = x^p$ for constant scalar $p$):**
   $$\frac{d z}{d x} = p \cdot x^{p-1}$$
   ```python
   self.grad += (p * (self.data ** (p - 1.0))) * out.grad
   ```

4. **Negation, Subtraction, Division:**
   Synthesized directly via composition:
   $$-x = x \cdot (-1)$$
   $$x - y = x + (-y)$$
   $$x / y = x \cdot (y^{-1})$$

### 4.2 Transcendental & Activation Functions

1. **Exponential ($z = \exp(x)$):**
   $$\frac{d z}{d x} = \exp(x) = z$$
   ```python
   self.grad += out.data * out.grad
   ```

2. **Natural Logarithm ($z = \ln(x)$):**
   $$\frac{d z}{d x} = \frac{1}{x}$$
   ```python
   self.grad += (1.0 / self.data) * out.grad
   ```

3. **Hyperbolic Tangent ($z = \tanh(x)$):**
   $$\frac{d z}{d x} = 1 - \tanh^2(x) = 1 - z^2$$
   ```python
   self.grad += (1.0 - out.data ** 2) * out.grad
   ```

4. **Rectified Linear Unit ($z = \max(0, x)$):**
   $$\frac{d z}{d x} = \begin{cases} 1 & x > 0 \\ 0 & x \le 0 \end{cases}$$
   ```python
   self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
   ```

---

## 5. Topological Sort on DAGs via Depth-First Search (DFS)

Why can't we simply traverse the graph in arbitrary order?
Consider the graph:
$$x \to a \to c \to L, \quad x \to b \to c$$
If we evaluate the gradient of $x$ before $a$ and $b$ have finished backpropagating into $x$, the gradient $\bar{x}$ will be incomplete!

We must sort all nodes into a **topological order** such that every node appears after all its ancestors and before all its descendants.

```
                  [ x ] ──► [ a ] ──┐
                    │               ▼
                    └─────► [ b ] ──► [ c ] ──► [ L ]
```

### 5.1 Post-Order DFS Algorithm
```python
def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)  # Append after all children are explored

    build_topo(self)

    # Seed root
    self.grad = 1.0

    # Traverse in reverse topological order (L down to inputs)
    for node in reversed(topo):
        node._backward()
```

1. We start at terminal root $L$.
2. DFS recursively visits all dependencies (`_prev`).
3. A node is added to `topo` only **after** all its dependencies are in `topo`.
4. Traversing `reversed(topo)` guarantees that whenever we invoke `node._backward()`, that node's `.grad` has accumulated every single gradient flowing from all downstream consumers!

---

## 6. The Gradient Accumulation Bug: `+=` vs. `=`

Recall our earlier warning:
```python
# CATASTROPHIC FATAL BUG:
self.grad = ... 

# MATHEMATICALLY MANDATORY:
self.grad += ...
```

When variable $x$ is reused:
$$y = x + x \quad \text{or} \quad y = x \cdot x$$
Node $x$ is in `_prev` of two incoming backward calls.
- If we use `=`, the second call overwrites the first call.
- If we use `+=`, the chain rule additions sum correctly.
- When running multiple training iterations, parameters must be reset via `p.grad = 0.0` before each backward pass!

---

## 7. Building a Neural Network on Top of `Value`

Using only our `Value` class, we can construct full multi-layer perceptrons:

```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):
        # Linear projection + activation:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()
```

When we forward pass through layers of `Neuron` instances, a computational DAG of hundreds of `Value` nodes is dynamically synthesized. Calling `loss.backward()` automatically computes exact gradients for all weights and biases in the network!
