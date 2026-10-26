# Topic 12.1: Computational Graphs & the Multivariable Chain Rule

---

## 1. Executive Summary & The 12-Year-Old Analogy

Modern deep learning architectures contain hundreds of billions of interconnected mathematical operations. How does PyTorch or JAX compute the exact derivative of a scalar loss function with respect to every single parameter in such vast networks without making algebraic errors or exploding in computational complexity?

The foundational solution is the **Computational Graph**: representing complex mathematical formulas as **Directed Acyclic Graphs (DAGs)**. When mathematical expressions are represented as graphs, the calculus of differentiation simplifies into a mechanical, message-passing algorithm governed by the **Multivariable Chain Rule**.

```
                [ x ] ─────────────┐
                  │                ▼
                  │           [ a = x * y ]
                  ▼                │
             [ b = x + y ] ◄───────┘  (Branching & Fan-out)
                  │
                  ▼
             [ L = c(a, b) ]  ──►  Loss Scalar
```

---

### The 12-Year-Old Analogy: The Rube Goldberg Machine & Falling Dominos

Imagine you build an enormous Rube Goldberg machine in your living room:
1. You roll a marble labeled **$x$**.
2. Marble **$x$** rolls down a wooden ramp and strikes a lever that tips a seesaw (**Node $a$**).
3. At the exact same moment, marble **$x$** triggers a tripwire that drops a steel ball down a funnel (**Node $b$**).
4. Both the seesaw and the steel ball collide with a final lever that rings a brass bell (**Node $L$**).

**The Forward Pass:**
You push marble **$x$** with a force of 2 Newtons. The motion cascades forward through the levers and pulleys, and the bell rings with a sound intensity of 47 decibels. That is forward evaluation: data flows along the arrows from inputs to outputs.

**The Backward Pass (Sensitivity Analysis):**
Now, your friend asks: *"If I nudge marble $x$ forward by a tiny fraction of an inch, how much louder will the bell ring?"*

To answer this, you don't need to rebuild the machine. You trace backwards from the bell:
- The bell felt force from **both** the seesaw ($a$) and the steel ball ($b$).
- You ask: *"How much did the seesaw affect the bell?"* (That's $\frac{\partial L}{\partial a}$).
- You ask: *"How much did marble $x$ affect the seesaw?"* (That's $\frac{\partial a}{\partial x}$).
- Multiplying those gives the effect of $x$ along Path 1: $\frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial x}$.
- But marble $x$ also pulled the tripwire to drop ball $b$! The effect along Path 2 is $\frac{\partial L}{\partial b} \cdot \frac{\partial b}{\partial x}$.
- **The Golden Rule:** Because marble $x$ caused ripples along **two independent paths**, its total influence on the bell is the **sum** of both influences:
$$\text{Total Impact} = (\text{Impact via Path } a) + (\text{Impact via Path } b)$$

This is the **Multivariable Chain Rule on Graphs**. If an input branches out to multiple paths, its gradient is the sum of the gradients flowing back from all its downstream children!

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Physical Meaning |
| :--- | :--- | :--- |
| $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ | Graph Theory | Directed Acyclic Graph (DAG) representing the computational program |
| $v_i \in \mathcal{V}$ | Variable / Tensor | Graph node representing an input, intermediate variable, or output |
| $(u, v) \in \mathcal{E}$ | Directed Edge | Causal dependency: node $u$ is an input argument to operation $v$ |
| $\text{Parents}(v)$ | Node Set $\{u \mid (u, v) \in \mathcal{E}\}$ | Predecessors of $v$: the inputs directly required to evaluate $v$ |
| $\text{Children}(u)$ | Node Set $\{v \mid (u, v) \in \mathcal{E}\}$ | Successors of $u$: the downstream nodes that consume $u$ |
| $L \in \mathbb{R}$ | Scalar Field | The scalar objective loss (terminal root of the graph) |
| $\bar{v} \equiv \frac{\partial L}{\partial v}$ | Scalar / Tensor (Adjoint) | The **adjoint** of node $v$: the sensitivity of terminal loss $L$ to perturbations in $v$ |
| $\frac{\partial v}{\partial u}$ | Jacobian / Local Derivative | The local derivative of elementary operation $v$ with respect to input $u$ |
| $\pi \in \Pi(u \to L)$ | Path Sequence | A directed path from node $u$ to terminal node $L$ in the graph |

---

## 3. Computational Graphs as Directed Acyclic Graphs (DAGs)

A computational graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ decomposes an arbitrary mathematical function into a sequence of elementary operations (addition, multiplication, exponentials, activations).

### 3.1 Formal Properties of Computational DAGs
1. **Acyclicity:** There are no directed cycles ($\nexists v_1, v_2, \dots, v_k$ such that $v_1 \to v_2 \to \dots \to v_k \to v_1$). Acyclic structure ensures that every variable can be evaluated in a deterministic causal sequence without circular dependencies.
2. **Topological Ordering:** Because the graph is acyclic, there exists at least one permutation of nodes $(v_1, v_2, \dots, v_N)$ such that for every directed edge $(v_i, v_j) \in \mathcal{E}$, we have $i < j$.
   - **Forward Pass:** Evaluate nodes in topological order ($v_1 \to v_N$).
   - **Backward Pass:** Evaluate adjoints in reverse topological order ($v_N \to v_1$).

```
FORWARD PASS: Topological Sort Order (Left to Right)
Inputs: x, y  ──►  a = x*y  ──►  b = x+y  ──►  c = a*b  ──►  L = c + d
                     │             ▲
                     └─────────────┘
BACKWARD PASS: Reverse Topological Order (Right to Left)
dL/dx, dL/dy  ◄──  dL/da, dL/db  ◄──  dL/dc, dL/dd  ◄──  dL/dL = 1.0
```

---

## 4. The Multivariable Chain Rule on Directed Graphs

In standard single-variable calculus, the chain rule is linear:
$$\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$$

However, in neural networks, intermediate variables are reused across multiple operations. For example, the input feature vector $x$ or hidden activation $h$ is multiplied by weights, passed through residual skip-connections, and fed into attention heads.

### 4.1 Theorem: Total Derivative on Graphs
Let $L$ be a scalar function of intermediate variables $y_1, y_2, \dots, y_k$, where each $y_i$ is a function of variable $x$:
$$L = L(y_1(x), y_2(x), \dots, y_k(x))$$

By the multivariable total derivative theorem:
$$\frac{\partial L}{\partial x} = \sum_{i=1}^k \frac{\partial L}{\partial y_i} \frac{\partial y_i}{\partial x}$$

Expressed in graph-theoretic notation:
$$\boxed{\bar{x}} \equiv \frac{\partial L}{\partial x} = \sum_{y \in \text{Children}(x)} \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} = \sum_{y \in \text{Children}(x)} \bar{y} \cdot \frac{\partial y}{\partial x}$$

### 4.2 Path Expansion Interpretation
Each term in the sum represents the chain rule evaluated along one specific directed path from $x$ to $L$:
$$\frac{\partial L}{\partial x} = \sum_{\pi \in \text{Paths}(x \to L)} \prod_{(u, v) \in \pi} \frac{\partial v}{\partial u}$$

```
                ┌───► [ y1 ] ───► ... ───► [ L ]    (Path 1: product of local derivatives)
                │
       [ x ] ───┼───► [ y2 ] ───► ... ───► [ L ]    (Path 2: product of local derivatives)
                │
                └───► [ y3 ] ───► ... ───► [ L ]    (Path 3: product of local derivatives)

       Total dL/dx = (Path 1 product) + (Path 2 product) + (Path 3 product)
```

Notice a profound computational insight:
- If we compute the derivative by expanding all paths individually, the number of paths through a deep network grows **exponentially** with depth (the combinatorial explosion).
- But if we compute the adjoints $\bar{y}$ backwards in reverse topological order, we compute each intermediate adjoint **once** and reuse it! This reduces exponential complexity $\mathcal{O}(2^D)$ to **linear complexity** $\mathcal{O}(|\mathcal{V}| + |\mathcal{E}|)$.

---

## 5. Concrete Step-by-Step Walkthrough

Consider the multi-branch scalar graph evaluated in our lab:
Inputs: $x = 2.0$, $y = 3.0$
1. $a = x \cdot y = 2 \cdot 3 = 6$
2. $b = x + y = 2 + 3 = 5$
3. $c = a \cdot b = 6 \cdot 5 = 30$
4. $d = b^2 = 5^2 = 25$
5. $e = \exp(x) = \exp(2) \approx 7.389056$
6. $L = c + d - e = 30 + 25 - 7.389056 = 47.610944$

### 5.1 Graph Dependency Analysis
- $\text{Children}(L) = \emptyset$ (Terminal node)
- $\text{Children}(c) = \{L\}$
- $\text{Children}(d) = \{L\}$
- $\text{Children}(e) = \{L\}$
- $\text{Children}(a) = \{c\}$
- $\text{Children}(b) = \{c, d\}$  $\longleftarrow$ **Branching node! Two downstream consumers**
- $\text{Children}(y) = \{a, b\}$  $\longleftarrow$ **Branching input! Two downstream consumers**
- $\text{Children}(x) = \{a, b, e\}$ $\longleftarrow$ **Triple fan-out! Three downstream consumers**

### 5.2 Backward Pass (Reverse Topological Traversal)

**Step 1: Seed terminal loss**
$$\bar{L} = \frac{\partial L}{\partial L} = 1.0$$

**Step 2: Adjoints of $c, d, e$ from $L = c + d - e$**
$$\bar{c} = \bar{L} \cdot \frac{\partial L}{\partial c} = 1.0 \cdot 1 = 1.0$$
$$\bar{d} = \bar{L} \cdot \frac{\partial L}{\partial d} = 1.0 \cdot 1 = 1.0$$
$$\bar{e} = \bar{L} \cdot \frac{\partial L}{\partial e} = 1.0 \cdot (-1) = -1.0$$

**Step 3: Adjoint of $a$ from $c = a \cdot b$**
Node $a$ has one child ($c$):
$$\bar{a} = \bar{c} \cdot \frac{\partial c}{\partial a} = \bar{c} \cdot b = 1.0 \cdot 5 = 5.0$$

**Step 4: Adjoint of $b$ (Summing over Children $\{c, d\}$)**
Node $b$ flows into $c$ (via $c = a \cdot b$) and $d$ (via $d = b^2$):
$$\bar{b} = \bar{c} \cdot \frac{\partial c}{\partial b} + \bar{d} \cdot \frac{\partial d}{\partial b}$$
$$\frac{\partial c}{\partial b} = a = 6, \quad \frac{\partial d}{\partial b} = 2b = 2(5) = 10$$
$$\bar{b} = (1.0 \cdot 6) + (1.0 \cdot 10) = 6 + 10 = 16.0$$

**Step 5: Adjoint of $y$ (Summing over Children $\{a, b\}$)**
Node $y$ flows into $a = x \cdot y$ and $b = x + y$:
$$\bar{y} = \bar{a} \cdot \frac{\partial a}{\partial y} + \bar{b} \cdot \frac{\partial b}{\partial y} = (5.0 \cdot x) + (16.0 \cdot 1.0) = (5.0 \cdot 2) + 16.0 = 10 + 16 = 26.0$$

**Step 6: Adjoint of $x$ (Summing over Children $\{a, b, e\}$)**
Node $x$ flows into $a = x \cdot y$, $b = x + y$, and $e = \exp(x)$:
$$\bar{x} = \bar{a} \cdot \frac{\partial a}{\partial x} + \bar{b} \cdot \frac{\partial b}{\partial x} + \bar{e} \cdot \frac{\partial e}{\partial x}$$
$$\bar{x} = (5.0 \cdot y) + (16.0 \cdot 1.0) + (-1.0 \cdot \exp(x))$$
$$\bar{x} = (5.0 \cdot 3) + 16.0 - \exp(2) = 15 + 16 - 7.389056 = 23.610944$$

Every single manual calculation matches the numerical finite difference approximation down to $10^{-8}$!

---

## 6. The Number One Bug in Autograd Implementations: Overwrite vs. Accumulate

In amateur implementations of backpropagation, the most common catastrophic error is writing:
```python
# CATASTROPHIC BUG:
parent.grad = child.grad * local_derivative
```
instead of:
```python
# CORRECT:
parent.grad += child.grad * local_derivative
```

### Why does this destroy gradients?
If a variable $x$ is used twice—e.g., $f(x) = x \cdot x$ or a residual connection $y = x + F(x)$—the node $x$ has multiple outgoing edges.
- If you use assignment (`=`), the second edge **overwrites** the gradient computed by the first edge!
- The multivariable chain rule mandates $\sum$, which translates in code directly to `+=` (accumulation).
- Before each training iteration, all gradients must be zeroed out (`optimizer.zero_grad()`) so that the accumulation begins fresh from $0$.

---

## 7. Summary & Conceptual Takeaways

1. **Graph Representation:** Mathematical formulas are mapped onto Directed Acyclic Graphs where vertices are operations/variables and directed edges represent data dependencies.
2. **Reverse Topological Ordering:** Gradients must be propagated backwards in reverse topological order so that all children $\bar{y}$ of node $x$ are fully computed before $\bar{x}$ is evaluated.
3. **Branch Fan-Out Means Summation:** Whenever a variable splits into multiple branches, its total derivative is the sum of the derivatives across all branches: $\bar{x} = \sum_{y} \bar{y} \frac{\partial y}{\partial x}$.
4. **Linear Time Efficiency:** Reverse-mode backpropagation computes all parameter derivatives in time proportional to the number of edges $\mathcal{O}(|\mathcal{E}|)$, making deep learning scalable.
