# Topic 11.2: Exercises & Solutions — Activation Functions & Vanishing Gradients

---

### Problem 1: Derivation of the Sigmoid Derivative and Maximum Value (Easy)
Let the logistic sigmoid function be defined as:
$$\sigma(z) = \frac{1}{1 + e^{-z}} = (1 + e^{-z})^{-1}$$

1. Derive the first derivative $\sigma'(z) \equiv \frac{d\sigma}{dz}$ using the quotient or chain rule, and express it strictly in terms of $\sigma(z)$.
2. Prove that $\sigma'(z)$ achieves its global maximum at $z = 0$, and find its maximum value.
3. What is $\sigma'(z)$ as $z \to \infty$ and $z \to -\infty$? Explain the implication for gradient descent.

<details>
<summary>View Complete Solution</summary>

**Step 1: First derivative derivation:**
Using the chain rule:
$$\frac{d\sigma}{dz} = -1(1 + e^{-z})^{-2} \cdot \frac{d}{dz}(1 + e^{-z}) = -(1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$

Rewrite the numerator by adding and subtracting 1:
$$\frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{(1 + e^{-z})} \cdot \frac{e^{-z}}{(1 + e^{-z})} = \frac{1}{1 + e^{-z}} \cdot \frac{(1 + e^{-z}) - 1}{1 + e^{-z}} = \frac{1}{1 + e^{-z}} \left(1 - \frac{1}{1 + e^{-z}}\right)$$

Since $\sigma(z) = \frac{1}{1 + e^{-z}}$:
$$\boxed{\sigma'(z) = \sigma(z)(1 - \sigma(z))}$$

**Step 2: Global maximum:**
Let $u = \sigma(z)$. Since $z \in (-\infty, \infty)$, $u \in (0, 1)$.
We want to maximize $g(u) = u(1 - u) = u - u^2$.
Take the derivative with respect to $u$:
$$\frac{dg}{du} = 1 - 2u = 0 \implies u^* = \frac{1}{2}$$
Second derivative: $\frac{d^2g}{du^2} = -2 < 0$, confirming a local and global maximum.
When $\sigma(z) = \frac{1}{2}$, $\frac{1}{1 + e^{-z}} = \frac{1}{2} \implies 1 + e^{-z} = 2 \implies e^{-z} = 1 \implies z = 0$.
The maximum value of the derivative is:
$$\sigma'(0) = \frac{1}{2} \left(1 - \frac{1}{2}\right) = \frac{1}{4} = 0.25$$

**Step 3: Asymptotics & Gradient Implication:**
As $z \to \pm \infty$:
- When $z \to \infty$, $\sigma(z) \to 1 \implies \sigma'(z) \to 1(1 - 1) = 0$.
- When $z \to -\infty$, $\sigma(z) \to 0 \implies \sigma'(z) \to 0(1 - 0) = 0$.

*Implication:* Whenever a neuron's pre-activation $|z|$ becomes large (due to large inputs or large weights), the derivative approaches 0. The neuron enters **saturation**, completely extinguishing parameter updates regardless of how large the loss error is.
</details>

---

### Problem 2: Tanh vs Sigmoid Relationship & Zero-Centering (Medium)
The hyperbolic tangent is given by:
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

1. Prove algebraically that $\tanh(z) = 2\sigma(2z) - 1$.
2. Prove that $\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z)$.
3. Compare the maximum gradient of $\tanh(z)$ with that of $\sigma(z)$. Why is zero-centering beneficial for training stability?

<details>
<summary>View Complete Solution</summary>

**Step 1: Algebraic proof:**
Start with $2\sigma(2z) - 1$:
$$2\sigma(2z) - 1 = \frac{2}{1 + e^{-2z}} - 1 = \frac{2 - (1 + e^{-2z})}{1 + e^{-2z}} = \frac{1 - e^{-2z}}{1 + e^{-2z}}$$
Multiply numerator and denominator by $e^z$:
$$= \frac{e^z(1 - e^{-2z})}{e^z(1 + e^{-2z})} = \frac{e^z - e^{-z}}{e^z + e^{-z}} = \tanh(z)$$

**Step 2: Derivative of Tanh:**
Using the quotient rule on $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$:
Let $u = e^z - e^{-z} \implies u' = e^z + e^{-z}$.
Let $v = e^z + e^{-z} \implies v' = e^z - e^{-z}$.
$$\frac{d}{dz}\tanh(z) = \frac{u'v - uv'}{v^2} = \frac{(e^z + e^{-z})^2 - (e^z - e^{-z})^2}{(e^z + e^{-z})^2} = 1 - \left(\frac{e^z - e^{-z}}{e^z + e^{-z}}\right)^2 = 1 - \tanh^2(z)$$

**Step 3: Maximum gradient and zero-centering:**
- At $z = 0$, $\tanh(0) = 0$, so $\tanh'(0) = 1 - 0^2 = 1.0$.
- In contrast, $\sigma'(0) = 0.25$. Tanh transmits **4 times larger gradients** at the origin than Sigmoid!
- **Zero-centering:** Because $\tanh(z) \in (-1, 1)$, its mean activation across random data is approximately 0. In contrast, $\sigma(z) \in (0, 1)$ has mean $\approx 0.5$. Non-zero-centered inputs cause the gradients of weights in the subsequent layer to share the same sign ($\frac{\partial \mathcal{L}}{\partial w_i} = \delta a_i$, with $a_i > 0$ always), forcing weights to either all increase or all decrease together. This causes inefficient zigzag trajectories during gradient descent.
</details>

---

### Problem 3: Gradient Ratio in a 12-Layer Deep Network (Hard)
Consider a 12-layer deep network where all weights are initialized with orthogonal matrices having unit operator norm ($\|W^{[l]}\|_2 = 1.0$).
Suppose all pre-activations lie in a typical operating regime centered at $|z| \approx 1.0$.

1. For **Sigmoid**:
   - Compute $\sigma(1.0)$ and the local derivative $\sigma'(1.0)$.
   - Calculate the theoretical attenuation ratio of the gradient from layer 12 back to layer 1:
     $$\mathcal{R}_{\sigma} = \prod_{l=1}^{11} \sigma'(z^{[l]})$$
2. For **ReLU**:
   - Assuming 50% of the neurons are active ($z > 0$) and 50% are inactive ($z \le 0$), compute the expected gradient transmission across active paths.
3. For **SiLU / Swish**:
   - Derive the derivative formula for $\text{SiLU}(z) = z \sigma(z)$.
   - Evaluate $\text{SiLU}'(1.0)$ and compare the gradient ratio $\mathcal{R}_{\text{SiLU}}$ over 11 layers.

<details>
<summary>View Complete Solution</summary>

**Step 1: Sigmoid Attenuation:**
For $z = 1.0$:
$$\sigma(1.0) = \frac{1}{1 + e^{-1}} = \frac{1}{1 + 0.367879} \approx 0.731059$$
$$\sigma'(1.0) = \sigma(1.0)(1 - \sigma(1.0)) = 0.731059 \times 0.268941 \approx 0.196612$$

Across 11 layers:
$$\mathcal{R}_{\sigma} = (0.196612)^{11} \approx 1.70 \times 10^{-8}$$
The gradient signal arriving at layer 1 has shrunk by a factor of over **58 million**! Early layers receive virtually no update signal.

**Step 2: ReLU Gradient Transmission:**
For any active neuron with $z > 0$, $\text{ReLU}'(z) = 1.0$.
Across 11 active layers:
$$\mathcal{R}_{\text{ReLU}} = (1.0)^{11} = 1.0$$
The gradient magnitude is 100% preserved along active paths! Even if 50% of paths are masked, the non-zero gradients propagate with zero numerical decay.

**Step 3: SiLU / Swish Analysis:**
$$\text{SiLU}(z) = z \sigma(z)$$
By the product rule:
$$\frac{d}{dz}[z \sigma(z)] = 1 \cdot \sigma(z) + z \cdot \sigma'(z) = \sigma(z) + z \sigma(z)(1 - \sigma(z)) = \sigma(z)[1 + z(1 - \sigma(z))]$$

Evaluating at $z = 1.0$:
$$\text{SiLU}'(1.0) = 0.731059 \cdot [1 + 1.0(1 - 0.731059)] = 0.731059 \cdot [1 + 0.268941] = 0.731059 \times 1.268941 \approx 0.927670$$

Over 11 layers:
$$\mathcal{R}_{\text{SiLU}} = (0.927670)^{11} \approx 0.428$$
After 11 layers, the gradient is attenuated by only ~57% (gradient remains $\approx 0.43$), which is fully trainable and avoids both explosion and vanishing collapse.
</details>
