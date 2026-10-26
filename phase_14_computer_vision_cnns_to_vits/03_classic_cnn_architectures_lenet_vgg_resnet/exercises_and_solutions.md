# Topic 14.3: Exercises & Solutions — Classic CNNs & ResNet

---

### Problem 1: Gradient Backpropagation Through a Single Residual Block (Easy)
Consider a 2-dimensional residual block without activations for simplicity:
$$\mathbf{y} = W \mathbf{x} + \mathbf{x} = (W + \mathbf{I}) \mathbf{x}$$

Where:
$$W = \begin{bmatrix} 0.2 & -0.4 \\ 0.1 & -0.3 \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}$$

1. Compute the output vector $\mathbf{y} \in \mathbb{R}^2$.
2. Suppose the upstream loss gradient is $\frac{\partial \mathcal{L}}{\partial \mathbf{y}} = [3.0, -1.0]$. Compute the gradient flowing through the residual branch ($\frac{\partial \mathcal{L}}{\partial \mathbf{y}} W$).
3. Compute the gradient flowing through the identity shortcut ($\frac{\partial \mathcal{L}}{\partial \mathbf{y}} \mathbf{I}$).
4. Compute the total downstream gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$.

<details>
<summary>View Complete Solution</summary>

**Part 1: Forward Pass:**
$$W \mathbf{x} = \begin{bmatrix} 0.2(1.0) - 0.4(2.0) \\ 0.1(1.0) - 0.3(2.0) \end{bmatrix} = \begin{bmatrix} 0.2 - 0.8 \\ 0.1 - 0.6 \end{bmatrix} = \begin{bmatrix} -0.6 \\ -0.5 \end{bmatrix}$$
$$\mathbf{y} = W \mathbf{x} + \mathbf{x} = \begin{bmatrix} -0.6 \\ -0.5 \end{bmatrix} + \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} = \begin{bmatrix} 0.4 \\ 1.5 \end{bmatrix}$$

**Part 2: Residual Branch Gradient:**
Using vector-matrix product:
$$\mathbf{g}_{\text{res}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} W = [3.0, -1.0] \begin{bmatrix} 0.2 & -0.4 \\ 0.1 & -0.3 \end{bmatrix}$$
$$g_{\text{res}, 0} = 3.0(0.2) + (-1.0)(0.1) = 0.6 - 0.1 = 0.5$$
$$g_{\text{res}, 1} = 3.0(-0.4) + (-1.0)(-0.3) = -1.2 + 0.3 = -0.9$$
$$\mathbf{g}_{\text{res}} = [0.5, -0.9]$$

**Part 3: Identity Shortcut Gradient:**
$$\mathbf{g}_{\text{skip}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \mathbf{I} = [3.0, -1.0]$$

**Part 4: Total Downstream Gradient:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \mathbf{g}_{\text{res}} + \mathbf{g}_{\text{skip}} = [0.5, -0.9] + [3.0, -1.0] = \mathbf{[3.5, -1.9]}$$
Notice how the original upstream gradient $[3.0, -1.0]$ directly supports and stabilizes the total gradient.
</details>

---

### Problem 2: Vanishing Gradient Decay Rate: Plain vs. ResNet (Medium)
Consider a deep network with $L$ sequential layers/blocks. Suppose that due to small weights and activations, the Jacobian norm of each layer's transformation is bounded by $\gamma = 0.7 < 1.0$:
$$\left\| \frac{\partial \mathcal{F}_l}{\partial \mathbf{x}_l} \right\| \le \gamma = 0.7$$

1. In a **plain deep network** ($\mathbf{x}_{l+1} = \mathcal{F}_l(\mathbf{x}_l)$), derive an upper bound for the gradient norm ratio $\frac{\|\frac{\partial \mathcal{L}}{\partial \mathbf{x}_0}\|}{\|\frac{\partial \mathcal{L}}{\partial \mathbf{x}_L}\|}$ as a function of $L$.
2. In a **residual network** ($\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}_l(\mathbf{x}_l)$), use the triangle inequality to derive a lower bound for the gradient norm ratio $\frac{\|\frac{\partial \mathcal{L}}{\partial \mathbf{x}_0}\|}{\|\frac{\partial \mathcal{L}}{\partial \mathbf{x}_L}\|}$.
3. For a depth of $L = 34$ layers, calculate the numerical value of the plain network upper bound and compare it to the ResNet behavior.

<details>
<summary>View Complete Solution</summary>

**Part 1: Plain Network Gradient Bound:**
In a plain network, backpropagation multiplies Jacobians:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_0} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \prod_{l=0}^{L-1} \frac{\partial \mathcal{F}_l}{\partial \mathbf{x}_l}$$
Taking norms:
$$\left\| \frac{\partial \mathcal{L}}{\partial \mathbf{x}_0} \right\| \le \left\| \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \right\| \prod_{l=0}^{L-1} \left\| \frac{\partial \mathcal{F}_l}{\partial \mathbf{x}_l} \right\| \le \left\| \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \right\| \gamma^L$$
$$\frac{\|\frac{\partial \mathcal{L}}{\partial \mathbf{x}_0}\|}{\|\frac{\partial \mathcal{L}}{\partial \mathbf{x}_L}\|} \le \gamma^L$$

**Part 2: Residual Network Gradient Bound:**
In ResNet:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_0} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \left( \mathbf{I} + \frac{\partial}{\partial \mathbf{x}_0} \sum_{l=0}^{L-1} \mathcal{F}_l(\mathbf{x}_l) \right)$$
Using the reverse triangle inequality $\|A + B\| \ge |\|A\| - \|B\||$:
$$\left\| \frac{\partial \mathcal{L}}{\partial \mathbf{x}_0} \right\| \ge \left\| \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \right\| \left| 1 - \left\| \frac{\partial}{\partial \mathbf{x}_0} \sum_{l=0}^{L-1} \mathcal{F}_l(\mathbf{x}_l) \right\| \right|$$
Crucially, the identity component $\mathbf{I}$ guarantees that the gradient does not depend on a multiplicative chain $\gamma^L \to 0$. Even when $\mathcal{F} \to 0$, the ratio remains exactly $1.0$!

**Part 3: Numerical Evaluation for $L = 34$:**
For $\gamma = 0.7$ and $L = 34$:
$$\gamma^{34} = (0.7)^{34} \approx 5.41 \times 10^{-6}$$
In the plain network, the gradient is attenuated by a factor of over **$180,000\times$**, rendering the earliest layers completely untrainable. In ResNet, the gradient flow remains bounded and healthy, allowing all 34 layers to learn concurrently.
</details>

---

### Problem 3: The Bottleneck Block: FLOPs & Parameter Analysis (Hard)
Compare two residual block designs for processing an input feature volume of shape $[C=256, H=56, W=56]$ producing an output of shape $[C=256, H=56, W=56]$:

- **Design A (Basic Residual Block):**
  - Layer 1: $3 \times 3$ Conv, 256 filters, padding 1, stride 1.
  - Layer 2: $3 \times 3$ Conv, 256 filters, padding 1, stride 1.
  - Shortcut: Direct identity addition $\mathbf{x} + \mathcal{F}(\mathbf{x})$.

- **Design B (Bottleneck Residual Block):**
  - Layer 1: $1 \times 1$ Conv, 64 filters (squeeze), padding 0, stride 1.
  - Layer 2: $3 \times 3$ Conv, 64 filters (spatial filter), padding 1, stride 1.
  - Layer 3: $1 \times 1$ Conv, 256 filters (expand), padding 0, stride 1.
  - Shortcut: Direct identity addition $\mathbf{x} + \mathcal{F}(\mathbf{x})$.

1. Calculate the total number of convolutional weights (excluding biases) for Design A and Design B.
2. Calculate the total number of MAC operations required for Design A and Design B on a single $56 \times 56$ feature map.
3. Compute the ratio of parameters and FLOPs of Design B relative to Design A.
4. Why is the Bottleneck Block indispensable for very deep networks like ResNet-101 and ResNet-152?

<details>
<summary>View Complete Solution</summary>

**Part 1: Convolutional Weight Counts:**
- **Design A (Basic Block):**
  - Conv1 ($3 \times 3$): $256 \times 256 \times 3 \times 3 = 589,824$
  - Conv2 ($3 \times 3$): $256 \times 256 \times 3 \times 3 = 589,824$
  - Total Design A = $589,824 + 589,824 = \mathbf{1,179,648 \text{ weights}}$

- **Design B (Bottleneck Block):**
  - Conv1 ($1 \times 1$, $256 \to 64$): $64 \times 256 \times 1 \times 1 = 16,384$
  - Conv2 ($3 \times 3$, $64 \to 64$): $64 \times 64 \times 3 \times 3 = 36,864$
  - Conv3 ($1 \times 1$, $64 \to 256$): $256 \times 64 \times 1 \times 1 = 16,384$
  - Total Design B = $16,384 + 36,864 + 16,384 = \mathbf{69,632 \text{ weights}}$

**Part 2: MAC Operations ($H \times W = 56 \times 56 = 3,136$ spatial locations):**
- **Design A:**
  $$\text{MACs}_A = 3,136 \times 1,179,648 = \mathbf{3,699,376,128 \text{ MACs}} \approx 3.70 \text{ GMACs}$$
- **Design B:**
  $$\text{MACs}_B = 3,136 \times 69,632 = \mathbf{218,366,976 \text{ MACs}} \approx 0.218 \text{ GMACs}$$

**Part 3: Ratio Comparison:**
$$\text{Parameter Ratio} = \frac{69,632}{1,179,648} \approx \mathbf{0.0590 \implies 94.1\% \text{ parameter reduction!}}$$
$$\text{Compute Ratio} = \frac{218,366,976}{3,699,376,128} \approx \mathbf{0.0590 \implies 94.1\% \text{ FLOP reduction!}}$$

**Part 4: Architectural Significance:**
The bottleneck block reduces compute and parameters by **nearly $17\times$** compared to a basic block of the same input/output width. This dramatic efficiency gain makes it possible to stack 100 or 150+ layers in ResNet-101 and ResNet-152 while keeping total FLOPs and parameter counts lower than shallower VGG architectures!
</details>
