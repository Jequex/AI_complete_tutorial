# Topic 01: Adversarial Attacks and Robustness

## 1. Introduction & 12-Year-Old Intuition

Imagine you have a loyal guard dog trained to recognize your front door. No matter if it is raining, sunny, dark, or snowing, the dog instantly knows: *"That is the front door; do not bark at it."*

Now, an invisible trickster approaches. The trickster does not paint the door bright purple or tear it down. Instead, the trickster sprinkles a microscopic, imperceptible layer of yellow and black glitter across the wood. To your human eyes, the door looks 100% identical—not a single human would ever notice the glitter. 

Yet when your guard dog looks at the door, its eyes combine all those tiny glitter reflections. Suddenly, the dog barks in terror, convinced that your front door is actually a 10-foot roaring grizzly bear!

```
     Clean Image [x]                 Adversarial Noise [δ]                Attacked Image [x_adv]
    (Human: Stop Sign)               (Imperceptible Perturbation)          (Human: Stop Sign)
    (Model: Stop Sign 99%)                                                (Model: Speed Limit 80 99%)
   ┌────────────────────┐                 ┌────────────────────┐               ┌────────────────────┐
   │     STOP SIGN      │        +        │  ::. .:  .:: .:..  │       =       │     STOP SIGN      │
   │      [ OCTAGON ]   │   (ε = 0.03)    │ .::. .  .:::  .::  │               │      [ OCTAGON ]   │
   └────────────────────┘                 └────────────────────┘               └────────────────────┘
```

This is an **adversarial example**. 

In classical software engineering, bugs come from syntax errors, buffer overflows, or null pointer dereferences. But in modern deep learning, **the bug is geometric**. Deep neural networks operate in high-dimensional vector spaces ($\mathbb{R}^d$ where $d$ can be thousands or millions of dimensions). In such vast spaces, tiny directional nudges—each individually smaller than the sensor noise of a camera—accumulate across hundreds of weight dot products to shift the network's internal activation state across the decision boundary.

Understanding the mathematics of how these attacks are synthesized and how to certify robustness against them is the first pillar of AI Safety.

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Domain / Type | Meaning & Role in Robustness |
| :--- | :--- | :--- | :--- |
| $x$ | Input Sample | $\mathbb{R}^d$ | Clean feature vector or image tensor (e.g., pixel intensities normalized in $[0, 1]$). |
| $y$ | Ground Truth Label | $\{0, 1\}$ or $\{1, \dots, C\}$ | True categorical label assigned to sample $x$. |
| $f_\theta(x)$ | Neural Classifier | $\mathbb{R}^d \to \mathbb{R}^C$ | Parameterized model outputting class logits or probabilities with weights $\theta$. |
| $\mathcal{L}(f_\theta(x), y)$ | Loss Function | $\mathbb{R}$ | Scalar loss (e.g., cross-entropy) penalizing model error. |
| $\delta$ | Adversarial Perturbation | $\mathbb{R}^d$ | The additive noise vector injected into the input: $x_{\text{adv}} = x + \delta$. |
| $\epsilon$ | Perturbation Budget | $\mathbb{R}^+$ | Maximum allowed magnitude of perturbation under a chosen norm: $\|\delta\|_p \le \epsilon$. |
| $\|\cdot\|_p$ | $\ell_p$-Norm | Functional norm | Metric bounding the attack: $\ell_\infty$ (peak pixel change), $\ell_2$ (Euclidean energy), $\ell_0$ (sparsity). |
| $\nabla_x \mathcal{L}$ | Input Gradient | $\mathbb{R}^d$ | Vector of partial derivatives of loss with respect to *input features*, not weights. |
| $\text{sign}(\cdot)$ | Signum Function | $\mathbb{R}^d \to \{-1, 0, 1\}^d$ | Element-wise sign extractor: returns $+1$ if $z_i > 0$, $-1$ if $z_i < 0$. |
| $\Pi_{\mathcal{S}}(\cdot)$ | Projection Operator | Set operator | Projects an updated vector back into the feasible constrained set $\mathcal{S} = B_p(x, \epsilon) \cap [0, 1]^d$. |
| $L$ | Lipschitz Constant | $\mathbb{R}^+$ | Bound on the maximum rate of change of the model: $\|f(x) - f(x')\| \le L \|x - x'\|$. |
| $r(x)$ | Certified Robustness Radius | $\mathbb{R}^+$ | Exact Euclidean radius around $x$ within which no perturbation can alter the predicted class. |

---

## 3. The Adversarial Threat Model & Formulation

An adversarial attack on an input $x$ with true label $y$ seeks to solve the following constrained optimization problem:

$$\max_{\delta} \mathcal{L}(f_\theta(x + \delta), y) \quad \text{subject to} \quad \|\delta\|_p \le \epsilon \quad \text{and} \quad (x + \delta) \in \mathcal{X}$$

where $\mathcal{X}$ is the valid input domain (e.g., $[0, 1]^d$ for normalized image tensors).

### The $\ell_p$ Norm Ball Geometry
The geometry of the perturbation budget $\|\delta\|_p \le \epsilon$ governs the attacker's power:
1. **$\ell_\infty$ Norm**: $\|\delta\|_\infty = \max_i |\delta_i| \le \epsilon$. Every feature can change by at most $\epsilon$. In images, setting $\epsilon = 8/255 \approx 0.031$ produces modifications completely invisible to the human eye.
2. **$\ell_2$ Norm**: $\|\delta\|_2 = \sqrt{\sum_i \delta_i^2} \le \epsilon$. Bounds total Euclidean perturbation energy.
3. **$\ell_0$ Norm**: $\|\delta\|_0 = |\{i : \delta_i \ne 0\}| \le k$. Bounds the number of features or pixels altered, regardless of magnitude.

---

## 4. The Fast Gradient Sign Method (FGSM)

Introduced by Goodfellow, Shlens, and Szegedy (2014), the **Fast Gradient Sign Method (FGSM)** is a one-step adversarial generation algorithm derived from a first-order Taylor expansion of the loss function.

### Step-by-Step Mathematical Derivation

Consider a linear approximation of the loss function $\mathcal{L}(x + \delta, y)$ around clean point $x$:

$$\mathcal{L}(x + \delta, y) \approx \mathcal{L}(x, y) + \nabla_x \mathcal{L}(x, y)^T \delta$$

To maximize the loss subject to the $\ell_\infty$ constraint $\|\delta\|_\infty \le \epsilon$:

$$\max_{\|\delta\|_\infty \le \epsilon} \left[ \mathcal{L}(x, y) + \sum_{i=1}^d \delta_i \frac{\partial \mathcal{L}}{\partial x_i} \right]$$

Since $\mathcal{L}(x, y)$ is independent of $\delta$, this decomposes into $d$ independent 1-dimensional maximization problems:

$$\max_{-\epsilon \le \delta_i \le \epsilon} \delta_i \frac{\partial \mathcal{L}}{\partial x_i}$$

For each dimension $i$, to make the product $\delta_i \frac{\partial \mathcal{L}}{\partial x_i}$ as positive as possible:
- If $\frac{\partial \mathcal{L}}{\partial x_i} > 0$, choose $\delta_i = +\epsilon$.
- If $\frac{\partial \mathcal{L}}{\partial x_i} < 0$, choose $\delta_i = -\epsilon$.
- If $\frac{\partial \mathcal{L}}{\partial x_i} = 0$, choose $\delta_i = 0$.

Therefore, the exact optimal perturbation $\delta^*$ under the linear approximation is:

$$\delta^* = \epsilon \cdot \text{sign}\left( \nabla_x \mathcal{L}(x, y) \right)$$

Leading to the closed-form FGSM adversarial example:

$$\boxed{x_{\text{adv}} = x + \epsilon \cdot \text{sign}\left(\nabla_x \mathcal{L}(x, y)\right)}$$

### Why Linearity in High Dimensions Causes Fragility
If an input has $d = 10,000$ dimensions and each weight $w_i$ has average magnitude $m$, the dot product change under FGSM is:

$$\Delta z = w^T \delta = \sum_{i=1}^d w_i \cdot \epsilon \cdot \text{sign}(w_i) = \epsilon \sum_{i=1}^d |w_i| = \epsilon \cdot d \cdot m$$

Even if $\epsilon = 0.01$ (tiny imperceptible noise), when $d = 10,000$ and $m = 0.1$, the activation shifts by:

$$\Delta z = 0.01 \times 10,000 \times 0.1 = 10.0$$

A shift of $+10.0$ in logit space changes a sigmoid probability from $\sigma(-5) = 0.0067$ to $\sigma(5) = 0.9933$! **High dimensionality acts as an amplifier for tiny coherent perturbations.**

---

## 5. Projected Gradient Descent (PGD)

While FGSM is fast (requiring only a single backward pass), it frequently fails against non-linear models where the linear Taylor approximation breaks down over large radii $\epsilon$.

Madry et al. (2017) formalized adversarial defense through the lens of robust optimization and introduced **Projected Gradient Descent (PGD)** as the definitive multi-step first-order adversary.

### The PGD Iterative Update Rule
PGD solves the constrained optimization iteratively by taking small gradient steps of size $\alpha < \epsilon$ and projecting the perturbed point back into the $\ell_p$ ball around $x$:

$$\boxed{x^{(t+1)} = \Pi_{B_\epsilon(x) \cap \mathcal{X}} \left( x^{(t)} + \alpha \cdot \text{sign}\left(\nabla_{x^{(t)}} \mathcal{L}(x^{(t)}, y)\right) \right)}$$

where the initial point $x^{(0)}$ is often initialized with uniform random noise inside the $\ell_\infty$ ball:

$$x^{(0)} = x + \mathcal{U}(-\epsilon, \epsilon)$$

### The Projection Operator $\Pi$
For the $\ell_\infty$ norm with image bounds $[0, 1]$, the projection operator is an element-wise clipping operation:

$$\Pi_{B_\epsilon(x) \cap [0, 1]}(z) = \text{clip}\left( z, \; \max(0, x - \epsilon), \; \min(1, x + \epsilon) \right)$$

PGD with $K$ iterations (commonly $K \in [10, 40]$) is recognized as the "universal" first-order adversary: any model robust against PGD is generally robust against all first-order adversarial attacks.

---

## 6. Certified Robustness & Lipschitz Continuity

Empirical defenses often fall prey to **gradient masking** (Athalye et al. 2018), where a model merely makes its gradients vanishing or noisy, creating a false illusion of security against gradient-based attacks while remaining vulnerable to black-box or gradient-free attacks.

To obtain mathematical guarantees that cannot be broken by any attacker, we turn to **certified robustness**.

### Lipschitz Continuity Definition
A function $f: \mathbb{R}^d \to \mathbb{R}^k$ is $L$-Lipschitz continuous with respect to norm $\|\cdot\|_p$ if for all $x, x' \in \mathbb{R}^d$:

$$\|f(x) - f(x')\|_q \le L \|x - x'\|_p$$

The constant $L$ acts as a strict speed limit on how rapidly the model's output can change when its input is perturbed.

### Certified Radius for Linear Classifiers
Consider a binary linear classifier $f(x) = w^T x + b$ where the classification rule is $\hat{y} = \text{sign}(f(x))$.

**Theorem**: For any clean point $x$ where $f(x) \ne 0$, the exact Euclidean distance from $x$ to the decision boundary $w^T x' + b = 0$ is:

$$\boxed{r(x) = \frac{|w^T x + b|}{\|w\|_2}}$$

**Proof**:
The decision hyperplane is $\mathcal{H} = \{x' \in \mathbb{R}^d : w^T x' + b = 0\}$.
The geometric distance $d(x, \mathcal{H})$ is the solution to:

$$\min_{x'} \|x - x'\|_2 \quad \text{subject to} \quad w^T x' + b = 0$$

Let $x' = x + \delta$. Then $w^T (x + \delta) + b = 0 \implies w^T \delta = -(w^T x + b)$.
By the Cauchy-Schwarz inequality:

$$|w^T \delta| \le \|w\|_2 \|\delta\|_2 \implies \|\delta\|_2 \ge \frac{|w^T \delta|}{\|w\|_2} = \frac{|w^T x + b|}{\|w\|_2}$$

Equality is achieved when $\delta$ is parallel to the normal vector $w$:

$$\delta^* = -\frac{w^T x + b}{\|w\|_2^2} w$$

Thus, the minimum perturbation norm required to cross the boundary is exactly $r(x) = \frac{|w^T x + b|}{\|w\|_2}$. $\blacksquare$

**Guarantee**: For any perturbation $\delta$ satisfying $\|\delta\|_2 < r(x)$, the classifier's prediction is guaranteed to remain identical to the clean prediction:

$$\text{sign}(f(x + \delta)) = \text{sign}(f(x)) \quad \forall \|\delta\|_2 < r(x)$$

---

## 7. Adversarial Defense Paradigms

```
                           AI ROBUSTNESS DEFENSES
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
  Empirical Defenses                                     Certified Defenses
  • Adversarial Training (Madry)                         • Exact Verification (SMT / MILP)
    min_θ E [ max_δ L(f_θ(x+δ), y) ]                     • Randomized Smoothing (Cohen et al.)
  • Defensive Distillation (Defeated by C&W)             • Interval Bound Propagation (IBP)
  • Gradient Regularization: ||∇_x L||_2                 • Lipschitz Constrained Networks (1-Lipschitz)
```

1. **Adversarial Training (Min-Max Optimization)**:
   Instead of minimizing standard empirical risk, we solve the robust min-max problem:
   $$\min_\theta \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \max_{\|\delta\|_p \le \epsilon} \mathcal{L}(f_\theta(x + \delta), y) \right]$$
   During each training step, the inner maximization is approximated via PGD-k to generate adversarial samples, which are then passed to the outer minimization to update weights $\theta$.

2. **Randomized Smoothing (Cohen et al. 2019)**:
   For arbitrary deep networks, certified $\ell_2$ robustness is achieved by smoothing the base classifier $f$ with isotropic Gaussian noise $\mathcal{N}(0, \sigma^2 I)$:
   $$g(x) = \arg\max_c \mathbb{P}_{\eta \sim \mathcal{N}(0, \sigma^2 I)}\left( f(x + \eta) = c \right)$$
   If top class $c_A$ has probability $p_A$ and runner-up class has probability $p_B$, the smoothed classifier $g(x)$ is certified robust within radius:
   $$R = \frac{\sigma}{2} \left( \Phi^{-1}(p_A) - \Phi^{-1}(p_B) \right)$$
   where $\Phi^{-1}$ is the inverse standard normal CDF.

---

## 8. Summary & Engineering Takeaways

1. **Vulnerability is Geometric**: Neural network vulnerability to adversarial attacks is an inherent geometric property of high-dimensional linear and semi-linear combinations.
2. **FGSM vs PGD**: FGSM is a single-step Taylor approximation ($\mathcal{O}(1)$ backward pass); PGD is a multi-step iterative projected descent ($\mathcal{O}(K)$ backward passes) and is strictly stronger.
3. **Beware Gradient Masking**: Any defense that claims 100% empirical robustness without certified bounds or without testing under multi-step PGD with random restarts is likely masking gradients.
4. **Certified Robustness**: Lipschitz bounds and randomized smoothing provide mathematically provable invariance guarantees that hold against *any* adversary, regardless of compute budget.
