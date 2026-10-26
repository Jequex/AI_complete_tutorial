# Topic 14.3: Classic CNN Architectures: LeNet, VGG, and ResNet

---

## 1. Executive Summary & The 12-Year-Old Analogy

Between 1998 and 2015, computer vision underwent a profound architectural revolution. Early architectures like **LeNet-5** established the fundamental pattern of interleaved convolutions and subsampling. In 2012, **AlexNet** triggered the modern deep learning boom by marrying deep convolutions with ReLU non-linearities and GPU parallel computing. In 2014, **VGGNet** demonstrated that elegance lies in modular simplicity: factorizing large spatial filters into stacks of homogeneous $3 \times 3$ convolutions.

However, as researchers pushed networks beyond 20 layers, they ran into a severe wall known as **The Degradation Problem**: deeper plain networks began exhibiting **higher training error** than their shallower counterparts. This was not overfitting; it was optimization failure.

In 2015, Kaiming He and his team at Microsoft Research introduced **ResNet (Residual Networks)**, introducing **identity skip connections** that transform the network from learning unconstrained mappings $\mathcal{H}(\mathbf{x})$ into learning residual corrections $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$. By providing an unhindered **gradient highway**, ResNet enabled networks of 50, 101, and 152 layers to train smoothly, winning the ImageNet competition with an unprecedented $3.57\%$ top-5 error rate (surpassing human performance).

```
                      THE RESIDUAL BLOCK GRADIENT HIGHWAY
                      
                                Identity Shortcut
                     ┌──────────────────────────────────────┐
                     │                                      ▼
             x ──────┴───>[ Weight ]───> ReLU ───>[ Weight ]───(+)───> ReLU ───> y
                                F(x) Residual Branch
```

---

### The 12-Year-Old Analogy: The High-Speed Highway Bypassing City Traffic Jams

Imagine a cross-country message delivery relay across 100 cities:

1. **The Plain Network (The Congested Downtown Grid):**
   - In a vanilla neural network, your delivery van must navigate through downtown city streets in every single city (Block 1, Block 2, ... Block 100).
   - In each city, red lights, stop signs, and traffic jams (multiplications by weights smaller than 1, saturating activations) delay the van.
   - By city 20, the van is completely stuck in gridlock (the vanishing gradient problem!). Messages from the destination never make it back to the start.

2. **The Residual Block (The Elevated Express Flyover):**
   - ResNet engineers built an elevated express flyover (the **identity skip connection**) directly over every city!
   - Your van can cruise down the empty, straight bypass highway at 70 mph without touching any stoplights ($\mathbf{I}$, the identity matrix).
   - In each city, there is an optional side exit called the **Residual Workshop ($\mathcal{F}(\mathbf{x})$)**. If the workshop has a useful modification, it adds it to the van ($\mathcal{F}(\mathbf{x}) + \mathbf{x}$). If the workshop has nothing useful to add, it does nothing ($\mathcal{F}(\mathbf{x}) \approx 0$).

3. **The Miracle on the Way Back (Backpropagation):**
   - When instructions and corrections are sent backward from headquarters, they zoom down the unobstructed express flyover with zero traffic jams. Even across 152 cities, the gradient reaches the starting line instantly!

---

## 2. Mathematical Symbol Decoder

| Symbol | Type / Domain | Mathematical Meaning |
| :--- | :--- | :--- |
| $\mathbf{x} \in \mathbb{R}^d$ | Vector / Tensor | Input representation to a residual block |
| $\mathcal{H}(\mathbf{x})$ | Mapping Function | Underlying desired target mapping to be learned |
| $\mathcal{F}(\mathbf{x}, \{\mathcal{W}_i\})$ | Parameterized Function | Residual mapping function to be fitted by stacked layers |
| $\mathbf{y} \in \mathbb{R}^d$ | Vector / Tensor | Output of residual block: $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$ |
| $\mathbf{I} \in \mathbb{R}^{d \times d}$ | Identity Matrix | Identity operator ($\mathbf{I} \mathbf{v} = \mathbf{v}$), derivative of shortcut $\frac{\partial \mathbf{x}}{\partial \mathbf{x}}$ |
| $\mathcal{L}$ | Scalar | Scalar objective loss function to be minimized |
| $\frac{\partial \mathcal{L}}{\partial \mathbf{y}}$ | Vector / Tensor | Upstream loss gradient arriving from subsequent layers |
| $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$ | Vector / Tensor | Downstream loss gradient passed back to previous layers |
| $L \in \mathbb{Z}_{\ge 1}$ | Integer | Total number of residual blocks in the network |

---

## 3. Architecture Evolution: LeNet to VGGNet

```
                    MILESTONES IN CNN ARCHITECTURE (1998 - 2015)
                    
  1998: LeNet-5     [Input 32x32] ─> 5x5 Conv ─> AvgPool ─> 5x5 Conv ─> AvgPool ─> FC (60k params)
  
  2012: AlexNet     [Input 224x224] ─> 11x11 Conv ─> ReLU ─> MaxPool ─> 5x5 Conv ─> FC (60M params)
  
  2014: VGG-16      Homogeneous 3x3 Conv Stacks (Factorization) ─> MaxPool ─> FC (138M params)
  
  2015: ResNet-50   Deep Residual Stacks (Skip Connections) ─> Global Average Pooling (25.5M params)
```

### 3.1 LeNet-5 (1998)
Pioneered by Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner for handwritten check recognition:
- Input: $32 \times 32$ grayscale digits.
- Convolutional layers with $5 \times 5$ kernels and stride 1.
- Subsampling layers using $2 \times 2$ Average Pooling with trainable coefficients and biases.
- Non-linearities: Scaled hyperbolic tangent ($\tanh$) and sigmoid.
- Output: Euclidean Radial Basis Function (RBF) network.

### 3.2 AlexNet (2012)
Designed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, AlexNet won the ImageNet LSVRC-2012 challenge by an unprecedented margin ($15.3\%$ vs $26.2\%$ top-5 error):
- **ReLU Activations:** Replaced saturating sigmoid/tanh with $f(x) = \max(0, x)$, speeding up gradient descent by $6\times$.
- **GPU Parallelization:** Model partitioned across two NVIDIA GeForce GTX 580 GPUs (3GB each).
- **Overlapping Max Pooling:** $3 \times 3$ pooling with stride 2.
- **Regularization:** Heavy data augmentation (random crops, horizontal flips, PCA color jittering) and Dropout ($p = 0.5$) on fully connected layers.

### 3.3 VGGNet (2014): The Power of $3 \times 3$ Factorization
Karen Simonyan and Andrew Zisserman showed that large spatial filters ($5 \times 5, 7 \times 7, 11 \times 11$) are unnecessary and suboptimal. VGGNet replaced them entirely with stacks of $3 \times 3$ convolutions:

1. **Receptive Field Equivalence:**
   - Stacking two $3 \times 3$ conv layers has an effective receptive field of $5 \times 5$.
   - Stacking three $3 \times 3$ conv layers has an effective receptive field of $7 \times 7$.
2. **Parameter Savings:**
   - Assuming $C$ channels:
     - Two $3 \times 3$ layers: $2 \times (3^2 \cdot C^2) = 18 C^2$ parameters.
     - One $5 \times 5$ layer: $1 \times (5^2 \cdot C^2) = 25 C^2$ parameters.
     - **Savings:** $\frac{25 - 18}{25} = \mathbf{28\% \text{ fewer parameters}}$.
   - For three $3 \times 3$ layers vs. one $7 \times 7$ layer:
     - Three $3 \times 3$ layers: $3 \times 9 C^2 = 27 C^2$.
     - One $7 \times 7$ layer: $49 C^2$.
     - **Savings:** $\frac{49 - 27}{49} = \mathbf{44.9\% \text{ fewer parameters}}$.
3. **Non-Linear Expressivity:**
   - Three stacked $3 \times 3$ layers apply three non-linear ReLU activations instead of a single ReLU in a $7 \times 7$ layer, enabling the network to learn substantially more discriminative features.

---

## 4. The Degradation Problem

When deep learning practitioners attempted to scale VGG-style plain networks to 20, 30, or 56 layers, they encountered a perplexing phenomenon:

```
                  THE DEGRADATION PHENOMENON (HE ET AL., 2015)
                  
          Training Error (%)                   Test Error (%)
       30 │      Plain-56                   30 │      Plain-56
          │    ──────────────                  │    ──────────────
       25 │                                 25 │
          │      Plain-20                      │      Plain-20
       20 │    - - - - - -                  20 │    - - - - - -
          └──────────────────                  └──────────────────
                  Epochs                               Epochs
```

### Why Degradation is NOT Overfitting
- **Overfitting Definition:** Training error is low, but test/generalization error is high.
- **Degradation Reality:** As shown in the graph above, **both training error and test error are higher** for the 56-layer plain network than for the 20-layer plain network!
- If a 56-layer network were simply a 20-layer network with 36 identity layers appended, it would achieve *at least* equal training error. The fact that gradient descent cannot find this solution proves that **plain deep networks suffer from severe optimization gridlock**.

---

## 5. ResNet & The Residual Learning Formulation

### 5.1 Reformulating the Objective
Instead of asking a stack of layers to directly approximate an underlying mapping $\mathcal{H}(\mathbf{x})$, He et al. reformulated the problem:
$$\mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x} \iff \mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$$

Where:
- $\mathbf{x}$: Input representation.
- $\mathcal{F}(\mathbf{x})$: The **residual mapping** to be learned by the stacked weight layers.
- $\mathbf{x}$ is combined with $\mathcal{F}(\mathbf{x})$ via a simple element-wise addition (**identity shortcut**).

### 5.2 Why Residual Mappings are Easier to Optimize
Consider the extreme case where an identity mapping is optimal for a layer:
- In a plain network: The stacked non-linear layers must learn weights $W$ such that $\sigma(W_2 \sigma(W_1 \mathbf{x})) = \mathbf{x}$. Given non-linearities and ill-conditioned loss surfaces, learning identity through non-linear layers is notoriously difficult.
- In a residual block: The network simply drives the weights of $\mathcal{F}(\mathbf{x})$ toward zero ($\mathcal{W} \to 0$). Pushing weights toward zero is the natural tendency of weight decay ($\mathcal{L}_{\text{reg}} = \frac{\lambda}{2} \|W\|^2$)!

---

## 6. Mathematical Proof: The Gradient Highway

The decisive mathematical advantage of ResNet lies in its backward gradient dynamics.

### 6.1 Recursive Forward Representation
Let $\mathbf{x}_l$ be the input to the $l$-th residual block, and $\mathbf{x}_{l+1}$ be its output:
$$\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)$$

Applying this recursion repeatedly from layer $l$ to a deeper layer $L$:
$$\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i)$$

*Observation:* The representation $\mathbf{x}_L$ at any deep layer $L$ is expressed directly as the original representation $\mathbf{x}_l$ plus an additive sum of residual corrections.

### 6.2 Backward Gradient Derivation
By the chain rule of multivariable calculus, the loss gradient with respect to $\mathbf{x}_l$ is:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_l}$$

Substitute the forward expression for $\mathbf{x}_L$:
$$\frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_l} = \frac{\partial}{\partial \mathbf{x}_l} \left( \mathbf{x}_l + \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i) \right) = \mathbf{I} + \frac{\partial}{\partial \mathbf{x}_l} \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i)$$

Therefore:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \left( \mathbf{I} + \frac{\partial}{\partial \mathbf{x}_l} \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i) \right) = \underbrace{\frac{\partial \mathcal{L}}{\partial \mathbf{x}_L}}_{\text{Direct Highway}} + \underbrace{\frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \left( \frac{\partial}{\partial \mathbf{x}_l} \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i) \right)}_{\text{Residual Corrections}}$$

### 6.3 Crucial Insights from the Gradient Highway
1. **The Gradient Never Vanishes:** In a plain network, $\frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_l} = \prod_{i=l}^{L-1} W_i$, which decays exponentially to zero as $L - l \to \infty$. In ResNet, the term $\frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \cdot \mathbf{I}$ flows directly back to $\mathbf{x}_l$ **completely unhindered**, regardless of network depth!
2. **Safe Summation:** For the total gradient to vanish, the term $\frac{\partial}{\partial \mathbf{x}_l} \sum \mathcal{F}$ would have to equal $-\mathbf{I}$ across all dimensions simultaneously—a virtually impossible event in stochastic gradient descent.

---

## 7. The Bottleneck Residual Block

For deeper networks (ResNet-50, ResNet-101, ResNet-152), the standard two-layer $3 \times 3$ residual block becomes computationally expensive. He et al. introduced the **Bottleneck Block**:

```
                       THE BOTTLENECK RESIDUAL BLOCK
                       
                         Input: 256 Channels
                                  │
                 ┌────────────────┼────────────────┐
                 │                │ Shortcut       │
                 │                ▼                │
                 │      [ 1x1 Conv, 64 channels ]  │ (Compression / Squeeze)
                 │                ▼                │
                 │      [ 3x3 Conv, 64 channels ]  │ (Spatial Filtering)
                 │                ▼                │
                 │      [ 1x1 Conv, 256 channels]  │ (Restoration / Expand)
                 │                ▼                │
                 └──────────────>(+)               │
                                  │                │
                                  ▼                ▼
                         Output: 256 Channels
```

1. **$1 \times 1$ Conv (Squeeze):** Compresses channels from 256 down to 64 ($4\times$ reduction).
2. **$3 \times 3$ Conv (Spatial Filtering):** Operates on the compressed 64-channel bottleneck, dramatically reducing FLOPs.
3. **$1 \times 1$ Conv (Expand):** Restores channel capacity back to 256 to match the shortcut dimensions for element-wise addition.
