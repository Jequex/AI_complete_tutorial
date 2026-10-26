# Topic 3.3: The Chain Rule, Computational Graphs & Backpropagation

Welcome to **Topic 3.3: The Chain Rule, Computational Graphs & Backpropagation**. If artificial intelligence has a single holy grail equation, it is the **Chain Rule**. Backpropagation—the algorithm that trains every modern neural network, from vision convolutional nets to multi-trillion-parameter foundation models—is nothing more than a clever, reverse-order application of the multivariable chain rule on a computational graph.

---

## 1. Symbol Decoder Table

| Symbol | Mathematical Name | Plain-English Meaning | Child-Friendly Analogy |
| :---: | :--- | :--- | :--- |
| $\frac{dz}{dx}$ | Composite Derivative | Total rate of change of final output $z$ w.r.t input $x$ | How fast the last gear in a toy clock turns when you spin the first gear |
| $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ | Computational Graph | Directed graph representing arithmetic steps | A recipe diagram showing ingredients combining into dishes |
| $\bar{v}_i$ ("v bar") | Adjoint State | Sensitivity of total loss to node $v_i$: $\frac{\partial \mathcal{L}}{\partial v_i}$ | A message whispered backwards: "How much did you contribute to our mistake?" |
| $\text{Reverse Mode}$ | Backpropagation | Computing all gradients in a single backward sweep | Tracing the breadcrumb trail backwards from the treasure to the start |
| $\text{Forward Mode}$ | Forward-Mode Autodiff | Propagating perturbations forward from 1 input to all outputs | Watching a single drop of dye spread through a plumbing system |

---

## 2. The Chain Rule: From Univariate to Multivariable

### 2.1 The Univariate Chain Rule (The Gear Train Analogy)
Imagine three interlocking gear wheels:
- Turning Gear $A$ by 1 rotation spins Gear $B$ by 3 rotations ($\frac{db}{da} = 3$).
- Turning Gear $B$ by 1 rotation spins Gear $C$ by 4 rotations ($\frac{dc}{db} = 4$).

How fast does Gear $C$ spin when you turn Gear $A$ by 1 rotation?
Common sense tells you: $3 \times 4 = 12$ rotations!
Mathematically:
$$\frac{dc}{da} = \frac{dc}{db} \cdot \frac{db}{da} = 4 \times 3 = 12$$

```
   Gear A (x)              Gear B (y)              Gear C (z)
    ╭─────╮                 ╭─────╮                 ╭─────╮
    │  1  │ ──(x3 speed)──▶ │  3  │ ──(x4 speed)──▶ │ 12  │
    ╰─────╯                 ╰─────╯                 ╰─────╯
             dy/dx = 3               dz/dy = 4
                       dz/dx = 3 * 4 = 12
```

---

### 2.2 The Multivariable Chain Rule (Multiple Influence Paths)
In a neural network, a single weight $w$ can influence the final loss $\mathcal{L}$ through **multiple paths** (e.g. feeding into multiple neurons in the next layer).

If $x$ affects intermediate variables $u_1, u_2, \dots, u_K$, and each $u_k$ affects output $z$:

$$\mathbf{\frac{\partial z}{\partial x} = \sum_{k=1}^K \frac{\partial z}{\partial u_k} \frac{\partial u_k}{\partial x}}$$

```
                       ╭──▶ u₁ ──╮
                       │         │
                 x ────┼──▶ u₂ ──┼────▶ z (Loss)
                       │         │
                       ╰──▶ u₃ ──╯
            ∂z/∂x = (∂z/∂u₁)(∂u₁/∂x) + (∂z/∂u₂)(∂u₂/∂x) + (∂z/∂u₃)(∂u₃/∂x)
```
**The Golden Takeaway:** Along a single serial path, derivatives **multiply**. When multiple parallel paths diverge and reconverge, their contributions **add**!

---

## 3. Computational Graphs: The Blueprint of Deep Learning

Modern deep learning frameworks (PyTorch, JAX, TensorFlow) execute mathematics by constructing a **Computational Graph**—a Directed Acyclic Graph (DAG) where:
- **Leaf Nodes** are input features $x$ and trainable weights $w$.
- **Internal Nodes** are basic elementary operations ($+, \times, \exp, \max$).
- **Root Node** is the scalar loss $\mathcal{L}$.

```
  x (2.0) ───┐
             ├──[ * ]──▶ v₁ (6.0) ───┐
  w (3.0) ───┘                       ├──[ + ]──▶ v₂ (7.0) ──▶ Loss ℒ
                                     │
  b (1.0) ───────────────────────────┘
```

Every computation proceeds in two distinct phases:
1. **The Forward Pass:** Compute values from left to right, saving all intermediate activation numbers in memory.
2. **The Backward Pass:** Start at the loss $\mathcal{L}$ with $\frac{\partial \mathcal{L}}{\partial \mathcal{L}} = 1.0$, and flow gradients backwards from right to left using the chain rule.

---

## 4. Forward-Mode vs. Reverse-Mode Autodiff: The Trillion-Dollar Difference

Why did deep learning take off in the 2010s instead of the 1970s? Because computer scientists realized that **Reverse-Mode Automatic Differentiation (Backpropagation)** has an extraordinary computational scaling advantage.

Suppose a model has $N$ input weights and $M$ output losses:

| Autodiff Mode | Traversal Direction | Computational Cost | Optimal When |
| :--- | :---: | :---: | :--- |
| **Forward Mode** | Inputs $\to$ Outputs | $\mathcal{O}(N)$ passes | Few inputs, many outputs ($N \ll M$) |
| **Reverse Mode (Backprop)** | Outputs $\to$ Inputs | $\mathcal{O}(M)$ passes | Many inputs, few outputs ($N \gg M$) |

### The Catastrophic Cost of Forward Mode for Deep Learning:
Consider an LLM with **$N = 70 \text{ Billion parameters}$** and **$M = 1 \text{ scalar loss}$**:
- **Forward-Mode:** Would require running the entire model **70 Billion times** just to compute one gradient step! (This would take centuries).
- **Reverse-Mode:** Runs the forward pass once ($M = 1$), and the backward pass once. In **just two passes**, it computes the exact gradients for all 70 Billion weights simultaneously!

---

## 5. The Mathematical Derivation of Backpropagation

Let $v_i$ represent the scalar value at node $i$ in a computational graph.
Define the **adjoint** (or sensitivity) of node $i$ as:
$$\bar{v}_i = \frac{\partial \mathcal{L}}{\partial v_i}$$

### The Universal Backpropagation Rule:
For any node $i$, its children $\text{Children}(i)$ are all nodes that directly consumed $v_i$ as an input.
By the multivariable chain rule:

$$\mathbf{\bar{v}_i = \sum_{j \in \text{Children}(i)} \bar{v}_j \cdot \frac{\partial v_j}{\partial v_i}}$$

### Step-by-Step Walkthrough:
1. **Initialize:** Set $\bar{v}_{\text{loss}} = \frac{\partial \mathcal{L}}{\partial \mathcal{L}} = 1.0$.
2. **Topological Sort:** Traverse nodes in reverse topological order (from outputs back to inputs).
3. **Local Derivative Multiplication:** For each incoming branch from child $j$, multiply child adjoint $\bar{v}_j$ by the local derivative $\frac{\partial v_j}{\partial v_i}$.
4. **Accumulation:** Sum all incoming branch contributions into $\bar{v}_i$.

This completes backpropagation! Every gradient in PyTorch (`x.grad`, `w.grad`) is computed via this exact algorithm.
