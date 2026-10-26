# Topic 12.3: Exercises & Solutions — Scalar Autograd Engine From Scratch

---

### Problem 1: Implementing the Sigmoid Activation Function (Easy)

The standard logistic sigmoid function is defined as:
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

1. Show analytically that the derivative satisfies:
   $$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$
2. Write a complete Python method `sigmoid(self) -> Value` that could be added to the `Value` class, implementing both the forward calculation and the exact `_backward` closure.

#### Solution

1. **Analytical Derivative Derivation:**
   Let $s = \sigma(x) = (1 + e^{-x})^{-1}$.
   By the chain rule and power rule:
   $$\sigma'(x) = -1 \cdot (1 + e^{-x})^{-2} \cdot \frac{d}{dx}(1 + e^{-x}) = - (1 + e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1 + e^{-x})^2}$$
   We can rewrite the numerator by adding and subtracting 1:
   $$\frac{e^{-x}}{(1 + e^{-x})^2} = \frac{(1 + e^{-x}) - 1}{(1 + e^{-x})^2} = \frac{1 + e^{-x}}{(1 + e^{-x})^2} - \frac{1}{(1 + e^{-x})^2} = \frac{1}{1 + e^{-x}} - \left(\frac{1}{1 + e^{-x}}\right)^2$$
   Factoring out $\sigma(x)$:
   $$\sigma'(x) = \sigma(x) - \sigma(x)^2 = \sigma(x)(1 - \sigma(x))$$

2. **Python Implementation for `Value`:**
   ```python
   def sigmoid(self) -> "Value":
       # Stable forward evaluation
       x = self.data
       # To prevent floating-point overflow for large negative x:
       if x >= 0:
           s = 1.0 / (1.0 + math.exp(-x))
       else:
           exp_x = math.exp(x)
           s = exp_x / (1.0 + exp_x)
       
       out = Value(s, (self,), "sigmoid")

       def _backward():
           # dL/dx += dL/dout * d(out)/dx = out.grad * s * (1 - s)
           self.grad += (s * (1.0 - s)) * out.grad

       out._backward = _backward
       return out
   ```

---

### Problem 2: Hand Simulation of DFS Topological Sort (Medium)

Consider the diamond computational graph:
- Inputs: $x = 2.0$
- Intermediate: $a = x^2$, $b = 3x$
- Output: $c = a + b$

**Questions:**
1. Trace the execution of the recursive function `build_topo(v)` starting from root node $c$.
   List the exact call stack and when each node is added to the `topo` list.
2. What is the resulting order of nodes in `topo`?
3. What is the order in `reversed(topo)`?
4. Explain why executing `_backward()` in `reversed(topo)` order guarantees that node $x$ receives its total accumulated gradient from both $a$ and $b$ before any parent of $x$ would be processed.

#### Solution

1. **DFS Trace:**
   - Call `build_topo(c)`.
     - `c` not in visited $\implies$ mark visited.
     - Explore `c._prev` (nodes $a$ and $b$).
     - Call `build_topo(a)`.
       - `a` not in visited $\implies$ mark visited.
       - Explore `a._prev` (node $x$).
       - Call `build_topo(x)`.
         - `x` not in visited $\implies$ mark visited.
         - Explore `x._prev` ($\emptyset$).
         - **Append $x$ to `topo`**: `topo = [x]`
       - **Append $a$ to `topo`**: `topo = [x, a]`
     - Call `build_topo(b)`.
       - `b` not in visited $\implies$ mark visited.
       - Explore `b._prev` (node $x$).
       - `x` is already in visited! Skip recursive call.
       - **Append $b$ to `topo`**: `topo = [x, a, b]`
     - **Append $c$ to `topo`**: `topo = [x, a, b, c]`

2. **Topological List:**
   `topo = [x, a, b, c]`

3. **Reversed Topological List:**
   `reversed(topo) = [c, b, a, x]` (or `[c, a, b, x]` depending on set iteration order).

4. **Correctness Guarantee:**
   In `reversed(topo)`:
   - First, `c._backward()` runs, distributing $c.\text{grad}$ to $a.\text{grad}$ and $b.\text{grad}$.
   - Next, both $a.\text{backward}()$ and $b.\text{backward}()$ run, each adding their contributions into $x.\text{grad}$.
   - Finally, when $x$ is reached, both downstream paths have fully settled, ensuring $x.\text{grad}$ is complete!

---

### Problem 3: Deriving the Softmax + Cross-Entropy Gradient Cancellation (Hard)

In classification, let $z = [z_1, z_2]^T$ be logits, and let the softmax prediction be:
$$\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^2 e^{z_j}}$$
Let the loss be cross-entropy with ground truth one-hot target $y = [1, 0]^T$:
$$L = -\ln(\hat{y}_1) = -\ln\left(\frac{e^{z_1}}{e^{z_1} + e^{z_2}}\right) = \ln(e^{z_1} + e^{z_2}) - z_1$$

1. If an autograd engine differentiates $L$ as individual primitive nodes ($\exp$, $+$, $\ln$, $-$), write out the chain rule steps for $\frac{\partial L}{\partial z_1}$.
2. Simplify the analytical expression and show that all intermediate terms collapse to:
   $$\frac{\partial L}{\partial z_1} = \hat{y}_1 - 1$$
3. Why do production frameworks like PyTorch fuse `CrossEntropyLoss` into a single custom C++/CUDA kernel rather than decomposing it into individual primitive `Value` operations?

#### Solution

1. **Step-by-Step Chain Rule on Decomposed Graph:**
   Let:
   - $u_1 = \exp(z_1)$, $u_2 = \exp(z_2)$
   - $S = u_1 + u_2$
   - $T = \ln(S)$
   - $L = T - z_1$

   Node $z_1$ fans out into two paths:
   - Path 1: $z_1 \to u_1 \to S \to T \to L$
   - Path 2: $z_1 \to L$ directly (via $-z_1$)

   Applying the multivariable chain rule:
   $$\frac{\partial L}{\partial z_1} = \left( \frac{\partial L}{\partial T} \frac{\partial T}{\partial S} \frac{\partial S}{\partial u_1} \frac{\partial u_1}{\partial z_1} \right) + \frac{\partial L}{\partial z_1\text{ (direct)}}$$
   Evaluating the individual derivatives:
   - $\frac{\partial L}{\partial T} = 1.0$
   - $\frac{\partial T}{\partial S} = \frac{1}{S}$
   - $\frac{\partial S}{\partial u_1} = 1.0$
   - $\frac{\partial u_1}{\partial z_1} = \exp(z_1) = u_1$
   - $\frac{\partial L}{\partial z_1\text{ (direct)}} = -1.0$

   Combining:
   $$\frac{\partial L}{\partial z_1} = \left( 1.0 \cdot \frac{1}{S} \cdot 1.0 \cdot u_1 \right) - 1.0 = \frac{u_1}{S} - 1.0$$

2. **Simplification:**
   Since $\frac{u_1}{S} = \frac{\exp(z_1)}{\exp(z_1) + \exp(z_2)} = \hat{y}_1$:
   $$\frac{\partial L}{\partial z_1} = \hat{y}_1 - 1$$
   More generally, for any class $k$:
   $$\frac{\partial L}{\partial z_k} = \hat{y}_k - y_k$$

3. **Why Production Frameworks Fuse this Kernel:**
   - **Numerical Stability:** If $z_1 = 1000$, $\exp(1000)$ overflows float32 to `+inf`, producing `nan` in the forward pass. A fused kernel applies the Log-Sum-Exp trick: $\ln(\sum e^{z_j}) = m + \ln(\sum e^{z_j - m})$ where $m = \max(z)$.
   - **Memory & Bandwidth:** Decomposing into primitives allocates intermediate tensors for $u_1, u_2, S, T$. A fused kernel writes only the scalar loss and single gradient vector directly in SRAM, reducing GPU memory traffic by $4\times$.
