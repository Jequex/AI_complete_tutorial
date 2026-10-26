# Topic 12.1: Exercises & Solutions — Computational Graphs & the Chain Rule

---

### Problem 1: Manual Reverse-Mode Backprop on a Diamond Graph (Medium)

Consider a scalar computational graph with inputs $x = 3.0$ and $w = 2.0$, and the following sequence of operations:
1. $z = x \cdot w$
2. $h = x + z$
3. $p = z^2$
4. $L = h \cdot p$

**Questions:**
1. Draw the adjacency/dependency relationships (list parents and children of each node).
2. Compute the forward pass values for all intermediate nodes ($z, h, p$) and the final output $L$.
3. Compute the backward pass adjoints $\bar{L}, \bar{p}, \bar{h}, \bar{z}, \bar{w}, \bar{x}$ step-by-step using the multivariable chain rule. Notice which node(s) fan out into multiple children!
4. Check your analytical derivative $\bar{x} = \frac{\partial L}{\partial x}$ using symbolic expansion of $L$ as a direct function of $x$ and $w$.

#### Solution

1. **Dependency Analysis:**
   - $\text{Parents}(z) = \{x, w\}$, $\text{Children}(z) = \{h, p\}$ (Fan-out branch!)
   - $\text{Parents}(h) = \{x, z\}$, $\text{Children}(h) = \{L\}$
   - $\text{Parents}(p) = \{z\}$, $\text{Children}(p) = \{L\}$
   - $\text{Parents}(L) = \{h, p\}$, $\text{Children}(L) = \emptyset$
   - Node $x$ also fans out into $\{z, h\}$!

2. **Forward Pass Evaluation:**
   - $x = 3.0$
   - $w = 2.0$
   - $z = x \cdot w = 3.0 \cdot 2.0 = 6.0$
   - $h = x + z = 3.0 + 6.0 = 9.0$
   - $p = z^2 = 6.0^2 = 36.0$
   - $L = h \cdot p = 9.0 \cdot 36.0 = 324.0$

3. **Backward Pass Evaluation (Reverse Topological Order):**
   - **Seed Output:**
     $$\bar{L} = \frac{\partial L}{\partial L} = 1.0$$
   - **Adjoint $\bar{p}$:**
     $$\bar{p} = \bar{L} \cdot \frac{\partial L}{\partial p} = 1.0 \cdot h = 9.0$$
   - **Adjoint $\bar{h}$:**
     $$\bar{h} = \bar{L} \cdot \frac{\partial L}{\partial h} = 1.0 \cdot p = 36.0$$
   - **Adjoint $\bar{z}$ (Accumulation across children $h$ and $p$):**
     $$\bar{z} = \bar{h} \cdot \frac{\partial h}{\partial z} + \bar{p} \cdot \frac{\partial p}{\partial z}$$
     $$\frac{\partial h}{\partial z} = 1.0, \quad \frac{\partial p}{\partial z} = 2z = 2(6.0) = 12.0$$
     $$\bar{z} = (36.0 \cdot 1.0) + (9.0 \cdot 12.0) = 36.0 + 108.0 = 144.0$$
   - **Adjoint $\bar{w}$:**
     Node $w$ has only one child ($z$):
     $$\bar{w} = \bar{z} \cdot \frac{\partial z}{\partial w} = 144.0 \cdot x = 144.0 \cdot 3.0 = 432.0$$
   - **Adjoint $\bar{x}$ (Accumulation across children $z$ and $h$):**
     Node $x$ feeds into $z = x \cdot w$ and $h = x + z$:
     $$\bar{x} = \bar{z} \cdot \frac{\partial z}{\partial x} + \bar{h} \cdot \frac{\partial h}{\partial x}$$
     $$\frac{\partial z}{\partial x} = w = 2.0, \quad \frac{\partial h}{\partial x} = 1.0$$
     $$\bar{x} = (144.0 \cdot 2.0) + (36.0 \cdot 1.0) = 288.0 + 36.0 = 324.0$$

4. **Symbolic Verification:**
   Express $L$ directly as a polynomial in $x$ and $w$:
   $$z = xw \implies h = x + xw = x(1 + w), \quad p = (xw)^2 = x^2 w^2$$
   $$L = h \cdot p = x(1 + w) \cdot x^2 w^2 = x^3 w^2 (1 + w) = x^3 (w^2 + w^3)$$
   Differentiate with respect to $x$:
   $$\frac{\partial L}{\partial x} = 3 x^2 (w^2 + w^3) = 3(3^2)(2^2 + 2^3) = 3(9)(4 + 8) = 27 \cdot 12 = 324.0$$
   Differentiate with respect to $w$:
   $$\frac{\partial L}{\partial w} = x^3 (2w + 3w^2) = 27(2(2) + 3(4)) = 27(4 + 12) = 27 \cdot 16 = 432.0$$
   The graph backprop adjoints $\bar{x} = 324.0$ and $\bar{w} = 432.0$ match the exact symbolic derivatives identically!

---

### Problem 2: The Assignment Overwrite Bug vs. Accumulation (Easy)

Suppose a software developer writes an autograd engine. For any node $u$ with children $v_1, v_2$, the developer writes:
```python
def backward(self):
    for child in self.children:
        self.grad = child.grad * self.local_derivative(child)
```
1. Suppose $f(x) = x + x$. What is the true derivative $f'(x)$?
2. What value does the developer's buggy engine compute for $f'(x)$?
3. What is the percentage error introduced by this bug?

#### Solution
1. **True Derivative:**
   $$f(x) = x + x = 2x \implies f'(x) = 2.0$$
   In graph terms: $L = x + x$. $L$ has two incoming edges from $x$.
   $$\bar{x} = \bar{L} \cdot \frac{\partial L}{\partial x_{\text{left}}} + \bar{L} \cdot \frac{\partial L}{\partial x_{\text{right}}} = (1.0 \cdot 1.0) + (1.0 \cdot 1.0) = 2.0$$

2. **Buggy Engine Computation:**
   The developer's loop evaluates child 1:
   `self.grad = 1.0 * 1.0 = 1.0`
   Then it loops to child 2 and performs assignment (`=`):
   `self.grad = 1.0 * 1.0 = 1.0` (overwriting the first 1.0)
   The returned gradient is $1.0$.

3. **Percentage Error:**
   $$\text{Error} = \frac{|2.0 - 1.0|}{2.0} \times 100\% = 50\%$$
   The gradient is underestimated by exactly half! In deep networks with shared weights (e.g. RNNs, weight-tied embeddings, self-attention projection weights), this bug causes immediate training collapse.

---

### Problem 3: Residual Connections as Gradient Highways (Hard)

In a Deep Residual Network (ResNet), the transformation between layer $l$ and layer $l+1$ is defined as:
$$x_{l+1} = x_l + \mathcal{F}(x_l; W_l)$$
where $\mathcal{F}$ represents a residual block of convolutions and nonlinearities.

1. Show that for an arbitrary loss $L$, the gradient with respect to $x_l$ satisfies:
   $$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}} \left( I + \frac{\partial \mathcal{F}}{\partial x_l} \right)$$
2. Expand this recursively from layer $K$ back to layer $l$ ($l < K$):
   $$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_K} \left( I + \sum_{j=l}^{K-1} \frac{\partial \mathcal{F}_j}{\partial x_l} \right)$$
3. Explain why this graph topology prevents the vanishing gradient problem even when the network is 1,000 layers deep and the Jacobian $\frac{\partial \mathcal{F}}{\partial x}$ is close to zero.

#### Solution
1. **One-Step Gradient:**
   The computational node for $x_{l+1}$ is an addition: $x_{l+1} = u + v$, where $u = x_l$ (identity skip connection) and $v = \mathcal{F}(x_l)$.
   Both $u$ and $v$ depend on $x_l$. By the multivariable chain rule:
   $$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}} \frac{\partial x_{l+1}}{\partial u} \frac{\partial u}{\partial x_l} + \frac{\partial L}{\partial x_{l+1}} \frac{\partial x_{l+1}}{\partial v} \frac{\partial v}{\partial x_l}$$
   Since $\frac{\partial x_{l+1}}{\partial u} = I$ and $\frac{\partial u}{\partial x_l} = I$, and $\frac{\partial x_{l+1}}{\partial v} = I$ with $\frac{\partial v}{\partial x_l} = \frac{\partial \mathcal{F}}{\partial x_l}$:
   $$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}} (I) + \frac{\partial L}{\partial x_{l+1}} \frac{\partial \mathcal{F}}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}} \left( I + \frac{\partial \mathcal{F}}{\partial x_l} \right)$$

2. **Recursive Expansion Across $K$ Layers:**
   Because $x_K = x_l + \sum_{i=l}^{K-1} \mathcal{F}_i(x_i)$, differentiating $x_K$ with respect to $x_l$ gives:
   $$\frac{\partial x_K}{\partial x_l} = I + \frac{\partial}{\partial x_l} \left( \sum_{i=l}^{K-1} \mathcal{F}_i(x_i) \right)$$
   Multiplying by the upstream adjoint $\frac{\partial L}{\partial x_K}$:
   $$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_K} \left( I + \sum_{i=l}^{K-1} \frac{\partial \mathcal{F}_i}{\partial x_l} \right)$$

3. **Protection Against Vanishing Gradients:**
   In a standard feed-forward network without skip connections:
   $$x_{l+1} = \sigma(W_l x_l) \implies \frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_K} \prod_{j=l}^{K-1} W_j^T \text{diag}(\sigma'_j)$$
   If the spectral norm of $W_j^T \text{diag}(\sigma'_j)$ is less than 1, the product decays exponentially to zero as $(K - l) \to \infty$ ($\lambda^{K-l} \to 0$).
   In a ResNet, the gradient decomposes into:
   $$\frac{\partial L}{\partial x_l} = \underbrace{\frac{\partial L}{\partial x_K}}_{\text{Unobstructed Highway}} + \underbrace{\frac{\partial L}{\partial x_K} \left(\sum_{i=l}^{K-1} \frac{\partial \mathcal{F}_i}{\partial x_l}\right)}_{\text{Residual Perturbations}}$$
   Even if every single intermediate Jacobian $\frac{\partial \mathcal{F}_i}{\partial x_l} \approx 0$, the identity term $I$ guarantees that the gradient $\frac{\partial L}{\partial x_K}$ propagates cleanly back to layer $l$ without attenuation!
