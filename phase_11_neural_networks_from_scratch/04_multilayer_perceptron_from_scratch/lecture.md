# Topic 11.4: Multilayer Perceptrons From Scratch

---

## 1. Executive Summary & The 12-Year-Old Analogy

In 1969, Marvin Minsky and Seymour Papert published their influential book *Perceptrons*, mathematically proving that single-layer perceptrons could never solve non-linearly separable problems like the simple **XOR (Exclusive-OR)** function. This mathematical limitation triggered the devastating first **AI Winter**, stalling neural network research for nearly two decades.

The resurrection came in 1986 with David Rumelhart, Geoffrey Hinton, and Ronald Williams, who demonstrated that **multi-layer perceptrons (MLPs)** equipped with non-linear hidden layers and trained via the **analytical backpropagation algorithm** conquer XOR effortlessly.

In this topic, we assemble every foundational piece developed across Phase 11 into a complete, working Multi-Layer Perceptron engine built in pure Python from first principles—achieving **100% accuracy on XOR and concentric circles**.

```
                THE XOR PROBLEM: SINGLE-LAYER FAILURE VS MLP TRIUMPH
                
     Single-Layer Perceptron (FAIL):             Multi-Layer Perceptron (TRIUMPH):
     No single line can separate (0,1)&(1,0)     Two hidden hyperplanes warp the space,
     from (0,0)&(1,1)! Max accuracy: 75%         isolating classes with 100% accuracy!
     
     x₂ ^                                        x₂ ^
      1 ┼   ● (1)       ○ (0)                     1 ┼   ● (1)  \    ○ (0)
        │      \                                    │     \     \
        │   Impossible                              │      \  ● (1)\
        │    Separator                              │       \       \
      0 ┼   ○ (0)       ● (1)                     0 ┼   ○ (0)\       \ ○ (0)
        └───────┴───────────> x₁                    └─────────┴───────┴───> x₁
        0                   1                       0                     1
```

---

### The 12-Year-Old Analogy: The Complete Brain Assembly

Imagine you are building a mechanical brain from a kit of gears, levers, and rubber bands:

1. **The Single-Layer Limitation (The Rigid Ruler):**
   - A single-layer model is like having a rigid wooden ruler on a flat sheet of paper.
   - If red dots and blue dots are scattered, you can only slide the ruler across in a straight line.
   - If red dots are in the top-left and bottom-right corners (XOR), and blue dots are in the top-right and bottom-left, no matter how you angle your ruler, you always leave at least one dot on the wrong side! You fail with a grade of 50% to 75%.

2. **The Multi-Layer Breakthrough (Origami Paper Folding):**
   - Instead of trying to cut the flat paper with a straight ruler, the **Hidden Layer** folds the paper along two diagonal creases!
   - By folding the sheet in half, the two red corner dots now touch each other on the same side of the crease.
   - Now, the **Output Layer** slides its straight ruler through: snip! A single straight cut effortlessly separates all red dots from all blue dots.
   - **That is what hidden layers do:** they fold, stretch, and warp high-dimensional space so that tangled, non-linear problems become trivial and linearly separable.

3. **The Learning Loop (Forward & Backward):**
   - **Forward Pass:** The brain takes a guess on all examples.
   - **Loss Check:** The referee measures how wrong the guesses were.
   - **Backward Pass (Chain Rule):** Error feedback cascades backwards from the output to each hidden neuron, telling every connection whether to get slightly tighter or looser.
   - **Parameter Update:** All weights slide down the error mountain. Repeat this 1,000 times, and the brain achieves **100% perfection**!

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ | Dataset | Training set of $N$ input-target pairs |
| $L$ | Integer $\ge 2$ | Total number of layers ($L-1$ hidden layers + 1 output layer) |
| $W^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$ | Matrix | Weight matrix of layer $l$ mapping activations from $l-1$ to $l$ |
| $\mathbf{b}^{[l]} \in \mathbb{R}^{n^{[l]}}$ | Vector | Bias vector of layer $l$ providing affine translation offsets |
| $Z^{[l]} \in \mathbb{R}^{B \times n^{[l]}}$ | Matrix | Batch pre-activation tensor: $Z^{[l]} = A^{[l-1]} (W^{[l]})^T + \mathbf{b}^{[l]}$ |
| $A^{[l]} \in \mathbb{R}^{B \times n^{[l]}}$ | Matrix | Batch post-activation tensor: $A^{[l]} = g^{[l]}(Z^{[l]})$ |
| $\delta^{[l]} \in \mathbb{R}^{B \times n^{[l]}}$ | Matrix | Error sensitivity tensor: $\delta^{[l]} \equiv \frac{\partial \mathcal{L}}{\partial Z^{[l]}}$ |
| $\odot$ | Operator | Hadamard (element-wise) matrix/vector product |
| $\eta \in \mathbb{R}_{>0}$ | Scalar | Learning rate step size in Gradient Descent |

---

## 3. The XOR Non-Linear Separability Proof

The Exclusive-OR (XOR) logic gate maps two binary inputs $(x_1, x_2) \in \{0, 1\}^2$ to a binary output $y \in \{0, 1\}$:

| $x_1$ | $x_2$ | Target $y = x_1 \oplus x_2$ |
| :---: | :---: | :---: |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

### 3.1 Mathematical Proof: Why Single Linear Perceptrons Fail
A single-layer perceptron classifies points via a linear hyper-plane $w_1 x_1 + w_2 x_2 + b = 0$, predicting class 1 if $w_1 x_1 + w_2 x_2 + b > 0$, and 0 otherwise.

To correctly classify all four points, the weights $(w_1, w_2, b)$ must satisfy four simultaneous linear inequalities:
1. For $(0, 0) \to 0$: $w_1(0) + w_2(0) + b \le 0 \implies b \le 0$
2. For $(0, 1) \to 1$: $w_1(0) + w_2(1) + b > 0 \implies w_2 + b > 0$
3. For $(1, 0) \to 1$: $w_1(1) + w_2(0) + b > 0 \implies w_1 + b > 0$
4. For $(1, 1) \to 0$: $w_1(1) + w_2(1) + b \le 0 \implies w_1 + w_2 + b \le 0$

Sum inequalities (2) and (3):
$$(w_1 + b) + (w_2 + b) > 0 \implies w_1 + w_2 + 2b > 0$$

Substitute inequality (4), which states $w_1 + w_2 + b \le 0 \implies w_1 + w_2 \le -b$:
$$(-b) + 2b > 0 \implies b > 0$$

Contradiction! Inequality (1) demands $b \le 0$, while inequalities (2), (3), and (4) demand $b > 0$.
No such real numbers $(w_1, w_2, b)$ can exist. A single linear perceptron **provably cannot solve XOR**.

---

## 4. End-to-End Multi-Layer Perceptron Mathematics

To solve XOR and arbitrary non-linear manifolds, we construct an MLP with $L$ layers:

```
  Input X            Layer 1 (Hidden)            Layer 2 (Hidden)          Layer 3 (Output)
 [B, d_in] ──W^[1]──> Z^[1] ──g^[1]──> A^[1] ──W^[2]──> Z^[2] ──g^[2]──> A^[2] ──W^[3]──> Z^[3] ──σ──> ŷ
```

### 4.1 Forward Pass Equations
For batch $A^{[0]} \equiv X \in \mathbb{R}^{B \times d_{\text{in}}}$:

For $l = 1, \dots, L-1$:
$$Z^{[l]} = A^{[l-1]} (W^{[l]})^T + \mathbf{1} (\mathbf{b}^{[l]})^T$$
$$A^{[l]} = \tanh(Z^{[l]})$$

Output Layer ($l = L$):
$$Z^{[L]} = A^{[L-1]} (W^{[L]})^T + \mathbf{1} (\mathbf{b}^{[L]})^T$$
$$\hat{Y} \equiv A^{[L]} = \sigma(Z^{[L]})$$

### 4.2 Analytical Backward Pass Equations
Using Binary Cross-Entropy loss $\mathcal{L} = -\frac{1}{B} \sum [y \ln \hat{y} + (1-y)\ln(1-\hat{y})]$:

1. **Output Error Vector $\delta^{[L]}$:**
   $$\boxed{\delta^{[L]} = A^{[L]} - Y \in \mathbb{R}^{B \times n^{[L]}}}$$

2. **Hidden Layer Error Recursion (for $l = L-1, \dots, 1$):**
   By the multivariable chain rule:
   $$\boxed{\delta^{[l]} = \left( \delta^{[l+1]} W^{[l+1]} \right) \odot g'\left(A^{[l]}\right)}$$
   Where $\odot$ denotes element-wise multiplication, and for Tanh:
   $$g'(A^{[l]}) = 1 - (A^{[l]})^2$$

3. **Parameter Gradients:**
   $$\boxed{\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{1}{B} (\delta^{[l]})^T A^{[l-1]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}}$$
   $$\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \frac{1}{B} \sum_{i=1}^B \delta_i^{[l]} \in \mathbb{R}^{n^{[l]}}}$$

4. **Gradient Descent Parameter Updates:**
   $$W^{[l]} \leftarrow W^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial W^{[l]}}$$
   $$\mathbf{b}^{[l]} \leftarrow \mathbf{b}^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}}$$

---

## 5. Xavier / Glorot Parameter Initialization

If weights are initialized too small, activations and gradients vanish exponentially ($0.01^L \to 0$).
If weights are initialized too large, activations saturate and gradients explode ($10^L \to \infty$).

To maintain constant variance across all layers ($\operatorname{Var}(z^{[l]}) = \operatorname{Var}(z^{[l-1]})$), Xavier Glorot and Yoshua Bengio (2010) derived the optimal uniform initialization bounds:

$$W_{j, k}^{[l]} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, +\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$

Where $n_{\text{in}} = n^{[l-1]}$ and $n_{\text{out}} = n^{[l]}$. Biases are initialized to zeros: $\mathbf{b}^{[l]} = \mathbf{0}$.

---

## 6. Training Dynamics & Decision Boundaries

When an MLP trains on non-linear datasets (such as XOR or Concentric Circles):

1. **Epochs 1–100 (Symmetry Breaking):**
   - The initial weights break symmetry. The loss drops from $\approx 0.693$ ($\ln 2$, random guessing) as the network aligns coarse linear hyperplanes.
2. **Epochs 100–400 (Coordinate Warping):**
   - The hidden layers construct non-linear feature representations.
   - For XOR, the two hidden neurons transform the four vertices into a 2D space where $(0, 1)$ and $(1, 0)$ are grouped together away from $(0, 0)$ and $(1, 1)$.
3. **Epochs 400+ (Convergence):**
   - The output decision boundary isolates the target classes with high confidence ($p > 0.99$).
   - Training loss decreases monotonically below $0.001$, and test accuracy reaches **100%**.

---

## 7. Key Takeaways & Summary
- Single-layer perceptrons provably fail on non-linearly separable problems (Minsky & Papert 1969).
- Multi-Layer Perceptrons solve XOR and arbitrary non-linear boundaries by using hidden layers to fold and warp feature space.
- Backpropagation applies the analytical chain rule backwards, computing exact parameter gradients $\nabla_W \mathcal{L} = \frac{1}{B} (\delta^{[l]})^T A^{[l-1]}$.
- Xavier initialization scales weights inversely with $\sqrt{n_{\text{in}} + n_{\text{out}}}$, preventing early vanishing or exploding gradients.
