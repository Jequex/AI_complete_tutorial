# Topic 3.4: Jacobians & Vector Calculus in Deep Learning

Welcome to **Topic 3.4: Jacobians & Vector Calculus in Deep Learning**. 

In single-variable calculus, we studied functions taking one number to one number ($f: \mathbb{R} \to \mathbb{R}$). 
In multivariable calculus, loss functions take a vector of parameters and produce a single scalar error ($f: \mathbb{R}^n \to \mathbb{R}$).

Now we reach the ultimate mathematical tool used across modern neural networks: **vector-valued functions of multiple variables**:
$$\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$$

Every neural network hidden layer, every normalization layer (LayerNorm, RMSNorm), and every classification layer (**Softmax**) is a vector-valued function!

---

## 1. 🧒 The 12-Year-Old Analogy: Maya's Music Mixing Console & Character Morphing

Imagine 12-year-old Maya is a sound producer sitting at a giant **music mixing console** with Timmy.

```text
               Maya's Sound Mixing Console
               
        Inputs: 3 Sliders (x)          Outputs: 3 Speakers (f(x))
        
        [Slider 1: Bass]    ───┐         ╭──▶ [Speaker 1: Front Left]
                               ├──(J)───┼──▶ [Speaker 2: Front Right]
        [Slider 2: Treble]  ───┤         ╰──▶ [Speaker 3: Subwoofer]
        [Slider 3: Reverb]  ───┘
```

1. **The Sliders and the Speakers:**
   - Maya has $n = 3$ input sliders: Bass ($x_1$), Treble ($x_2$), and Reverb ($x_3$).
   - The studio has $m = 3$ output speakers: Front-Left ($f_1$), Front-Right ($f_2$), and Subwoofer ($f_3$).
2. **Nudging a Single Slider:**
   - If Maya nudges the **Bass slider** ($x_1$) upward by 1 centimeter, the Subwoofer ($f_3$) thumps much louder, Front-Left ($f_1$) hums slightly, and Front-Right ($f_2$) stays quiet.
   - If she slides the **Reverb slider** ($x_3$), sound echoes across *all three* speakers!
3. **The Scorecard of Sensitivities (The Jacobian $J$):**
   - How can Timmy record how every single slider influences every single speaker?
   - He makes a table with $m$ rows (one for each speaker) and $n$ columns (one for each slider):
     - **Row $i$** tracks **Speaker $i$**.
     - **Column $j$** tracks **Slider $j$**.
     - **Entry $(i, j)$** answers: *"If Maya nudges Slider $j$ by 1 mm, by how many decibels does Speaker $i$ change?"*
   - That complete sensitivity table is the **Jacobian Matrix**!

---

## 2. 📋 Mathematical Symbol Decoder Table

| Symbol | Pronunciation | Plain English Meaning | Concrete Real-World Analogy / Flashcard Example |
| :---: | :---: | :--- | :--- |
| $\mathbf{x} \in \mathbb{R}^n$ | "vector x in R-n" | Input vector of $n$ numbers | The positions of Maya's $n$ mixing console sliders |
| $\mathbf{f}(\mathbf{x}) \in \mathbb{R}^m$ | "f of x in R-m" | Output vector of $m$ transformed numbers | The volume levels coming out of the $m$ studio speakers |
| $J$ or $\frac{\partial \mathbf{f}}{\partial \mathbf{x}}$ | "Jacobian of f" | Matrix of all $m \times n$ first-order partial derivatives | A master spreadsheet tracking how every slider affects every speaker |
| $J_{ij}$ | "J i j" | Entry in Row $i$, Column $j$: $\frac{\partial f_i}{\partial x_j}$ | Decibel change in Speaker $i$ when Slider $j$ is nudged |
| $\delta_{ij}$ | "Kronecker delta" | Equals $1$ if $i = j$, and $0$ if $i \neq j$ | A light switch: ON (1) if pointing at yourself, OFF (0) for everyone else |
| $\mathbf{z}$ | "logits vector z" | Raw unbounded prediction scores from a neural network | Unnormalized point totals before converting into percentages |
| $\mathbf{p} = \operatorname{softmax}(\mathbf{z})$ | "softmax of z" | Output probability vector (numbers positive and summing to 1.0) | Dividing pie slices so the entire pie equals $100\%$ |
| $\det(J)$ | "determinant of J" | Local volume stretch factor of a space transformation | How many times bigger a cube of dough becomes when rolled out |
| $\mathbf{v}^T J$ | "Vector-Jacobian Product (VJP)" | Reverse-mode backward pass: multiplies upstream error by Jacobian | Whispering the total blame backwards through the layer in $O(n)$ time |

---

## 3. Mathematical Definition of the Jacobian Matrix

Let $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ be a differentiable vector-valued function. The **Jacobian matrix** $J \in \mathbb{R}^{m \times n}$ gathers all partial derivatives into an orderly grid:

$$J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} 
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}$$

### Structural Anatomy:
1. **$m$ Rows (Outputs):** Row $i$ is the gradient of output component $f_i$ transposed: $\nabla f_i(\mathbf{x})^T$.
2. **$n$ Columns (Inputs):** Column $j$ is the rate of change of the whole output vector when input $x_j$ alone is nudged: $\frac{\partial \mathbf{f}}{\partial x_j}$.
3. **Local Linear Approximation:**
   $$\mathbf{f}(\mathbf{x} + \Delta\mathbf{x}) \approx \mathbf{f}(\mathbf{x}) + J \Delta\mathbf{x}$$

---

## 4. The Grand Derivation: The Softmax Jacobian

The **Softmax function** converts $K$ raw logits $\mathbf{z} \in \mathbb{R}^K$ into probabilities $\mathbf{p} \in \mathbb{R}^K$:
$$p_i = \frac{e^{z_i}}{\sum_{k=1}^K e^{z_k}}$$

Every output probability $p_i$ is connected to **every single** input logit $z_j$ through the shared denominator.
Therefore, the derivative is a full $K \times K$ Jacobian matrix:
$$J_{ij} = \frac{\partial p_i}{\partial z_j}$$

Let numerator $u = e^{z_i}$ and denominator $v = \sum_{k=1}^K e^{z_k}$.
Using the Quotient Rule $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$:

### Case 1: Diagonal Elements ($i = j$, Self-Sensitivity)
Here we differentiate $p_i$ with respect to its own logit $z_i$:
$$\frac{\partial u}{\partial z_i} = e^{z_i}, \quad \frac{\partial v}{\partial z_i} = e^{z_i}$$
$$\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i} \left(\sum_k e^{z_k}\right) - e^{z_i} e^{z_i}}{\left(\sum_k e^{z_k}\right)^2} = \frac{e^{z_i}}{\sum_k e^{z_k}} - \left(\frac{e^{z_i}}{\sum_k e^{z_k}}\right)^2 = \mathbf{p_i(1 - p_i)}$$

### Case 2: Off-Diagonal Elements ($i \neq j$, Cross-Sensitivity)
Here we differentiate $p_i$ with respect to a competing logit $z_j$:
Since $u = e^{z_i}$ contains no $z_j$, $\frac{\partial u}{\partial z_j} = 0$.
The denominator still contains $e^{z_j}$, so $\frac{\partial v}{\partial z_j} = e^{z_j}$.
$$\frac{\partial p_i}{\partial z_j} = \frac{0 \cdot v - e^{z_i} e^{z_j}}{v^2} = -\left(\frac{e^{z_i}}{v}\right) \left(\frac{e^{z_j}}{v}\right) = \mathbf{-p_i p_j}$$

---

### Master Softmax Jacobian Formula
Unifying both cases using the Kronecker delta $\delta_{ij}$:

$$\mathbf{J_{ij} = \frac{\partial p_i}{\partial z_j} = p_i (\delta_{ij} - p_j)}$$

#### Line-by-Line Formula Breakdown:
$$\underbrace{J_{ij}}_{\text{Rate of change of } p_i \text{ w.r.t } z_j} = \underbrace{p_i}_{\text{Current probability}} \cdot (\underbrace{\delta_{ij}}_{\text{1 if } i=j, \text{ 0 if } i \neq j} - \underbrace{p_j}_{\text{Competitor probability}})$$

In clean matrix form:
$$\mathbf{J_{\text{softmax}} = \operatorname{diag}(\mathbf{p}) - \mathbf{p} \mathbf{p}^T}$$

---

## 5. The Miracle of Softmax + Cross-Entropy

When training classifiers, Softmax is paired with Cross-Entropy Loss:
$$\mathcal{L} = -\sum_{k=1}^K y_k \ln(p_k)$$
where $\mathbf{y}$ is the one-hot target vector ($y_c = 1$ for the true class, $0$ otherwise).

Applying the multivariable chain rule:
$$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_{i=1}^K \frac{\partial \mathcal{L}}{\partial p_i} \frac{\partial p_i}{\partial z_j} = \sum_{i=1}^K \left(-\frac{y_i}{p_i}\right) \left[ p_i(\delta_{ij} - p_j) \right]$$

The $p_i$ factors cancel out:
$$\frac{\partial \mathcal{L}}{\partial z_j} = -\sum_{i=1}^K y_i(\delta_{ij} - p_j) = -y_j + p_j \underbrace{\sum_{i=1}^K y_i}_{= 1} = \mathbf{p_j - y_j}$$

In vector notation:
$$\mathbf{\nabla_{\mathbf{z}} \mathcal{L} = \mathbf{p} - \mathbf{y}}$$

> [!IMPORTANT] The Golden Collapse
> Even though Softmax involves complex exponentials and a $K \times K$ Jacobian, when combined with Cross-Entropy, all complexities magically vanish. The gradient is simply:
> $$\text{Gradient} = \text{Model's Probability Prediction} - \text{True One-Hot Label}$$

---

## 6. Vector-Jacobian Products (VJP) and Memory Efficiency

In a Large Language Model with vocabulary $V = 128{,}000$, a single token's full Jacobian matrix would have size:
$$128{,}000 \times 128{,}000 \approx 1.6 \times 10^{10} \text{ numbers} \approx \mathbf{65 \text{ Gigabytes of VRAM!}}$$

That is why PyTorch **never** computes or stores explicit Jacobian matrices!
Instead, reverse-mode autodiff directly computes the **Vector-Jacobian Product (VJP)**:
$$\mathbf{v}^T J$$
where $\mathbf{v} = \frac{\partial \mathcal{L}}{\partial \mathbf{p}} \in \mathbb{R}^{1 \times K}$.

For Softmax:
$$\mathbf{v}^T J = \mathbf{v}^T (\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T) = (\mathbf{v} \odot \mathbf{p}) - (\mathbf{v} \cdot \mathbf{p}) \mathbf{p}$$
This requires only **$\mathcal{O}(K)$ memory** instead of $\mathcal{O}(K^2)$, saving hundreds of gigabytes of GPU memory!

---

## 7. Concrete Numerical Walkthrough

Suppose Maya tests a 3-class model where predicted probabilities are:
$$\mathbf{p} = \begin{bmatrix} 0.50 \\ 0.30 \\ 0.20 \end{bmatrix}$$

1. **Calculate Softmax Jacobian Diagonal Entries ($i = j$):**
   - $J_{11} = p_1(1 - p_1) = 0.50(1 - 0.50) = \mathbf{0.25}$
   - $J_{22} = p_2(1 - p_2) = 0.30(1 - 0.30) = \mathbf{0.21}$
   - $J_{33} = p_3(1 - p_3) = 0.20(1 - 0.20) = \mathbf{0.16}$
2. **Calculate Off-Diagonal Entries ($i \neq j$):**
   - $J_{12} = J_{21} = -p_1 p_2 = -(0.50)(0.30) = \mathbf{-0.15}$
   - $J_{13} = J_{31} = -p_1 p_3 = -(0.50)(0.20) = \mathbf{-0.10}$
   - $J_{23} = J_{32} = -p_2 p_3 = -(0.30)(0.20) = \mathbf{-0.06}$
3. **Assemble the Full $3 \times 3$ Jacobian Matrix:**
   $$J = \begin{bmatrix}
   +0.25 & -0.15 & -0.10 \\
   -0.15 & +0.21 & -0.06 \\
   -0.10 & -0.06 & +0.16
   \end{bmatrix}$$
4. **Column-Sum Verification:**
   - Col 1: $+0.25 - 0.15 - 0.10 = \mathbf{0.00}$
   - Col 2: $-0.15 + 0.21 - 0.06 = \mathbf{0.00}$
   - Col 3: $-0.10 - 0.06 + 0.16 = \mathbf{0.00}$
   Every column sums to zero because probabilities must always sum to $1.0$—if one probability expands, the others must shrink to compensate!
