# Topic 11.4: Exercises & Solutions — Multilayer Perceptron From Scratch

---

### Problem 1: Manual Forward and Backward Pass on XOR (Medium)
Consider an MLP with 2 inputs, 1 hidden layer with 2 neurons (using Tanh), and 1 output neuron (using Sigmoid) for binary classification:
- Architecture: $2 \to 2 \to 1$
- Training sample: $\mathbf{x} = [1.0, 0.0]^T$, ground truth $y = 1.0$
- Parameters:
  $$W^{[1]} = \begin{bmatrix} 0.5 & 0.5 \\ -0.5 & -0.5 \end{bmatrix}, \quad \mathbf{b}^{[1]} = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$$
  $$W^{[2]} = \begin{bmatrix} 1.0 & 1.0 \end{bmatrix}, \quad b^{[2]} = 0.0$$

1. Perform the forward pass to compute $\mathbf{z}^{[1]}$, $\mathbf{a}^{[1]}$, $z^{[2]}$, and prediction $\hat{y} = a^{[2]}$.
2. Compute the initial error $\delta^{[2]} = a^{[2]} - y$.
3. Compute the hidden layer error vector $\boldsymbol{\delta}^{[1]} = (\delta^{[2]} W^{[2]}) \odot (1 - (\mathbf{a}^{[1]})^2)$.
4. Compute the analytical gradients $\frac{\partial \mathcal{L}}{\partial W^{[2]}}$, $\frac{\partial \mathcal{L}}{\partial b^{[2]}}$, $\frac{\partial \mathcal{L}}{\partial W^{[1]}}$, and $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[1]}}$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Forward Pass:**
Hidden pre-activations:
$$z_1^{[1]} = 0.5(1.0) + 0.5(0.0) + 0.0 = 0.5$$
$$z_2^{[1]} = -0.5(1.0) - 0.5(0.0) + 0.0 = -0.5$$
$$\mathbf{z}^{[1]} = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}$$

Hidden activations (using $\tanh(0.5) \approx 0.462117$, $\tanh(-0.5) \approx -0.462117$):
$$\mathbf{a}^{[1]} = \begin{bmatrix} 0.462117 \\ -0.462117 \end{bmatrix}$$

Output pre-activation:
$$z^{[2]} = 1.0(0.462117) + 1.0(-0.462117) + 0.0 = 0.0$$

Output activation:
$$\hat{y} = a^{[2]} = \sigma(0.0) = \frac{1}{1 + e^0} = 0.5$$

**Step 2: Output Error:**
$$\delta^{[2]} = a^{[2]} - y = 0.5 - 1.0 = -0.5$$

**Step 3: Hidden Layer Error:**
$$\delta^{[2]} W^{[2]} = -0.5 \times [1.0, 1.0] = [-0.5, -0.5]$$
Local Tanh derivatives:
$$g'(a_1^{[1]}) = 1 - (0.462117)^2 = 1 - 0.213552 = 0.786448$$
$$g'(a_2^{[1]}) = 1 - (-0.462117)^2 = 0.786448$$

Multiply element-wise:
$$\delta_1^{[1]} = -0.5 \times 0.786448 = -0.393224$$
$$\delta_2^{[1]} = -0.5 \times 0.786448 = -0.393224$$
$$\boldsymbol{\delta}^{[1]} = \begin{bmatrix} -0.393224 \\ -0.393224 \end{bmatrix}$$

**Step 4: Parameter Gradients:**
Output layer:
$$\frac{\partial \mathcal{L}}{\partial W^{[2]}} = \delta^{[2]} (\mathbf{a}^{[1]})^T = -0.5 \times [0.462117, -0.462117] = \begin{bmatrix} -0.231059 & 0.231059 \end{bmatrix}$$
$$\frac{\partial \mathcal{L}}{\partial b^{[2]}} = \delta^{[2]} = -0.5$$

Hidden layer:
$$\frac{\partial \mathcal{L}}{\partial W^{[1]}} = \boldsymbol{\delta}^{[1]} \mathbf{x}^T = \begin{bmatrix} -0.393224 \\ -0.393224 \end{bmatrix} \begin{bmatrix} 1.0 & 0.0 \end{bmatrix} = \begin{bmatrix} -0.393224 & 0.0 \\ -0.393224 & 0.0 \end{bmatrix}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[1]}} = \boldsymbol{\delta}^{[1]} = \begin{bmatrix} -0.393224 \\ -0.393224 \end{bmatrix}$$
</details>

---

### Problem 2: Derivation of Xavier / Glorot Initialization (Medium)
Consider a linear layer $z = \sum_{i=1}^{n_{\text{in}}} w_i x_i$. Assume:
- The inputs $x_i$ are independent and identically distributed with mean 0 and variance $\operatorname{Var}(x)$.
- The weights $w_i$ are independent with mean 0 and variance $\operatorname{Var}(w)$.
- Weights and inputs are mutually independent.

1. Prove that the variance of the pre-activation is:
   $$\operatorname{Var}(z) = n_{\text{in}} \operatorname{Var}(w) \operatorname{Var}(x)$$
2. What condition on $\operatorname{Var}(w)$ ensures that signal variance neither grows nor shrinks during the forward pass ($\operatorname{Var}(z) = \operatorname{Var}(x)$)?
3. Considering the backward pass where gradients propagate with $n_{\text{out}}$, explain why Xavier initialization adopts the harmonic mean compromise:
   $$\operatorname{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$

<details>
<summary>View Complete Solution</summary>

**Step 1: Variance of linear combination:**
$$\mathbb{E}[z] = \sum_{i=1}^{n_{\text{in}}} \mathbb{E}[w_i] \mathbb{E}[x_i] = 0$$
Since all terms are mutually independent:
$$\operatorname{Var}(z) = \sum_{i=1}^{n_{\text{in}}} \operatorname{Var}(w_i x_i)$$
For two independent zero-mean random variables $w_i$ and $x_i$:
$$\operatorname{Var}(w_i x_i) = \mathbb{E}[(w_i x_i)^2] - (\mathbb{E}[w_i x_i])^2 = \mathbb{E}[w_i^2]\mathbb{E}[x_i^2] - 0 = \operatorname{Var}(w) \operatorname{Var}(x)$$
Summing over all $n_{\text{in}}$ terms:
$$\boxed{\operatorname{Var}(z) = n_{\text{in}} \operatorname{Var}(w) \operatorname{Var}(x)}$$

**Step 2: Forward pass stability condition:**
To keep $\operatorname{Var}(z) = \operatorname{Var}(x)$:
$$n_{\text{in}} \operatorname{Var}(w) = 1 \implies \operatorname{Var}(w) = \frac{1}{n_{\text{in}}}$$

**Step 3: Backward pass & Compromise:**
During backpropagation, the gradient with respect to input $x$ is:
$$\frac{\partial \mathcal{L}}{\partial x_k} = \sum_{j=1}^{n_{\text{out}}} w_{jk} \frac{\partial \mathcal{L}}{\partial z_j}$$
Following the exact same derivation, to prevent exploding or vanishing gradients during the backward pass:
$$\operatorname{Var}\left(\frac{\partial \mathcal{L}}{\partial x}\right) = n_{\text{out}} \operatorname{Var}(w) \operatorname{Var}\left(\frac{\partial \mathcal{L}}{\partial z}\right) \implies \operatorname{Var}(w) = \frac{1}{n_{\text{out}}}$$

In general, $n_{\text{in}} \ne n_{\text{out}}$. Glorot and Bengio (2010) proposed taking the harmonic mean of both conditions:
$$\operatorname{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$
For a uniform distribution $\mathcal{U}(-a, a)$, the variance is $\frac{a^2}{3}$. Setting $\frac{a^2}{3} = \frac{2}{n_{\text{in}} + n_{\text{out}}}$ gives:
$$a = \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}$$
This is the celebrated Xavier Uniform initialization formula!
</details>

---

### Problem 3: Hidden Representation Geometry in XOR (Hard)
Let an XOR network have 2 hidden neurons with step / threshold activations $h_1 = \Theta(x_1 + x_2 - 0.5)$ and $h_2 = \Theta(x_1 + x_2 - 1.5)$, where $\Theta(u) = 1$ if $u \ge 0$, else $0$.

1. Construct the truth table mapping the 4 input points $(x_1, x_2) \in \{(0,0), (0,1), (1,0), (1,1)\}$ to hidden coordinates $(h_1, h_2)$.
2. Prove that in the hidden feature space $(h_1, h_2)$, the XOR problem is **linearly separable**.
3. Write down a linear decision rule for the output neuron $y = \Theta(c_1 h_1 + c_2 h_2 + b_0)$ that classifies all four points correctly.

<details>
<summary>View Complete Solution</summary>

**Step 1: Coordinate mapping:**
- For $(0, 0)$:
  $u_1 = 0 + 0 - 0.5 = -0.5 \implies h_1 = 0$
  $u_2 = 0 + 0 - 1.5 = -1.5 \implies h_2 = 0$
  $\implies (h_1, h_2) = (0, 0)$ (Target $y = 0$).
- For $(0, 1)$:
  $u_1 = 0 + 1 - 0.5 = +0.5 \implies h_1 = 1$
  $u_2 = 0 + 1 - 1.5 = -0.5 \implies h_2 = 0$
  $\implies (h_1, h_2) = (1, 0)$ (Target $y = 1$).
- For $(1, 0)$:
  $u_1 = 1 + 0 - 0.5 = +0.5 \implies h_1 = 1$
  $u_2 = 1 + 0 - 1.5 = -0.5 \implies h_2 = 0$
  $\implies (h_1, h_2) = (1, 0)$ (Target $y = 1$).
- For $(1, 1)$:
  $u_1 = 1 + 1 - 0.5 = +1.5 \implies h_1 = 1$
  $u_2 = 1 + 1 - 1.5 = +0.5 \implies h_2 = 1$
  $\implies (h_1, h_2) = (1, 1)$ (Target $y = 0$).

**Step 2: Linear Separability in Hidden Space:**
In the hidden $(h_1, h_2)$ plane:
- Class 1 points: $(1, 0)$ (both $(0, 1)$ and $(1, 0)$ map here!)
- Class 0 points: $(0, 0)$ and $(1, 1)$

Notice that the two positive points collapsed into a single point $(1, 0)$!
Now we just need a line separating $(1, 0)$ from $(0, 0)$ and $(1, 1)$.

**Step 3: Output Decision Rule:**
Consider the linear combination:
$$s = h_1 - 2h_2 - 0.5$$
- For $(0, 0)$: $s = 0 - 0 - 0.5 = -0.5 \le 0 \implies \hat{y} = 0$ ✓
- For $(1, 0)$: $s = 1 - 0 - 0.5 = +0.5 > 0 \implies \hat{y} = 1$ ✓
- For $(1, 1)$: $s = 1 - 2(1) - 0.5 = -1.5 \le 0 \implies \hat{y} = 0$ ✓

All four points are classified perfectly! This demonstrates the geometric essence of hidden layers: **they warp and fold input space until non-linear boundaries become straight lines**.
</details>
