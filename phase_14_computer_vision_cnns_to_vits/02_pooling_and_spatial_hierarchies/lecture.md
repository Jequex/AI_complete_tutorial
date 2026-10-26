# Topic 14.2: Pooling & Spatial Hierarchies

---

## 1. Executive Summary & The 12-Year-Old Analogy

As convolutional networks process images through sequential layers, two critical architectural needs emerge:
1. **Controlling Dimensionality & Memory:** Maintaining high spatial resolutions ($224 \times 224, 112 \times 112$) across hundreds of channels would require prohibitive GPU memory and compute for downstream layers.
2. **Achieving Translation Invariance & Semantic Abstraction:** For high-level tasks like image classification, we care *whether* a feature (such as an eye, beak, or wheel) is present in a region, not its exact sub-pixel coordinate. If a cat shifts two pixels to the right, the network's classification probability should remain stable.

**Pooling** is a non-linear, fixed-parameter downsampling operation that aggregates local spatial patches into compact summary representations. By progressively reducing spatial height and width while expanding channel depth, pooling constructs a **spatial hierarchy** of representations.

```
                  THE SPATIAL DOWN-SAMPLING PIPELINE
                  
    Input Feature Map [4x4]                      Max-Pooled Output [2x2]
    ┌──────┬──────┬──────┬──────┐                (2x2 window, stride 2)
    │  12  │  20  │  30  │  0   │
    ├──────┼──────┼──────┼──────┤                    ┌──────┬──────┐
    │  8   │  15  │  2   │  14  │    MaxPooling      │  20  │  30  │
    ├──────┼──────┼──────┼──────┤    ─────────>      ├──────┼──────┤
    │  34  │  70  │  37  │  4   │                    │  70  │  112 │
    ├──────┼──────┼──────┼──────┤                    └──────┴──────┘
    │  11  │  9   │  112 │  64  │
    └──────┴──────┴──────┴──────┘
```

---

### The 12-Year-Old Analogy: Summarizing a Photo Album Page by Page

Imagine you are looking at a 1,000-page photo album, and you need to write a quick 1-page summary of every page for a friend:

1. **The Grid of Photographs (The Feature Map):**
   - Each page is divided into a $4 \times 4$ grid of 16 tiny photos.
   
2. **The "Best Highlight" Rule (Max Pooling):**
   - Instead of describing all 4 photos in the top-left quadrant, you only write down the single most exciting photo (the one with the largest fireworks display!).
   - If that firework was slightly to the left or slightly to the right, you still pick it. That is **translation invariance**: tiny movements don't change your summary.

3. **The "Average Vibe" Rule (Average Pooling):**
   - Instead of picking just the highest highlight, you look at all 4 photos and say: *"On average, this whole corner is sunny beach."* You add up their scores and divide by 4.

4. **The Ultimate Whole-Page Summary (Global Average Pooling):**
   - At the very end of the album, you look across an entire channel page with 49 photos. Instead of passing 49 numbers forward, you simply calculate the average brightness of the entire page and send a single number.
   - This prevents you from carrying giant piles of unnecessary paper!

---

## 2. Mathematical Symbol Decoder

| Symbol | Type / Domain | Mathematical Meaning |
| :--- | :--- | :--- |
| $X \in \mathbb{R}^{C \times H \times W}$ | 3D Tensor | Input feature volume with $C$ channels and spatial size $H \times W$ |
| $Y \in \mathbb{R}^{C \times H_{\text{out}} \times W_{\text{out}}}$ | 3D Tensor | Output feature volume after spatial pooling |
| $K \in \mathbb{Z}_{\ge 1}$ | Integer | Spatial pooling window size ($K \times K$) |
| $S \in \mathbb{Z}_{\ge 1}$ | Integer | Stride of the pooling window sweep |
| $\Omega_{i, j}$ | Index Set | Set of input coordinates $(r, c)$ falling inside the window for output $(i, j)$ |
| $M \in \{0, 1\}^{C \times H \times W}$ | Binary Mask | Argmax routing mask recording the location of maximal activations |
| $\mathbf{y}_{\text{GAP}} \in \mathbb{R}^C$ | 1D Vector | Feature vector produced by Global Average Pooling across all $H \times W$ pixels |
| $\frac{\partial \mathcal{L}}{\partial Y}$ | 3D Tensor | Upstream loss gradient arriving at the pooling layer output |
| $\frac{\partial \mathcal{L}}{\partial X}$ | 3D Tensor | Downstream loss gradient routed back to the pooling layer input |

---

## 3. Mathematical Mechanics of Pooling Operations

Unlike convolutional layers, pooling operations contain **zero learnable parameters** ($\text{params} = 0$). They operate on each channel independently (channel-wise invariance).

### 3.1 Max Pooling
Max pooling extracts the maximum value within each local receptive window $\Omega_{i, j}$:
$$Y_{c, i, j} = \max_{(m, n) \in \Omega_{i, j}} X_{c, m, n}$$

Where the local receptive window $\Omega_{i, j}$ spans:
$$\Omega_{i, j} = \left\{ (r, c) \;\middle|\; i \cdot S \le r < i \cdot S + K, \quad j \cdot S \le c < j \cdot S + K \right\}$$

**Why Max Pooling dominates vision backbones:**
- Visual activations from preceding ReLU layers represent the presence or confidence of specific features (e.g. edge detectors, texture filters).
- If a feature is present anywhere in the window, its activation is high ($X_{c, m, n} \gg 0$). Max pooling preserves this strong detection signal while discarding background noise ($X \approx 0$).

### 3.2 Average Pooling
Average pooling computes the arithmetic mean over each local receptive window $\Omega_{i, j}$:
$$Y_{c, i, j} = \frac{1}{|\Omega_{i, j}|} \sum_{(m, n) \in \Omega_{i, j}} X_{c, m, n} = \frac{1}{K^2} \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} X_{c, \, i \cdot S + m, \, j \cdot S + n}$$

Average pooling provides smooth low-pass spatial filtering. While less common in early layers than Max Pooling, it excels in specialized downsampling paths (such as Inception modules and ResNet downsampling branches).

### 3.3 Global Average Pooling (GAP)
Introduced by Min Lin et al. in *Network in Network* (2013) and popularized by ResNet, **Global Average Pooling** collapses the entire spatial dimension of each feature map into a single scalar:
$$y_c = \frac{1}{H \cdot W} \sum_{r=0}^{H-1} \sum_{col=0}^{W-1} X_{c, r, col}, \qquad \forall c \in \{0, \dots, C-1\}$$

```
                   GLOBAL AVERAGE POOLING (GAP) COLLAPSE
                   
      Input Feature Volume [C x H x W]                 GAP Vector [C]
      
           ┌──────┐                                     ┌───┐
          ╱  C   ╱│                                     │y₀ │
         ┌──────┐ │         Global Average              ├───┤
         │      │ │            Pooling                  │y₁ │
         │ HxW  │ │         ─────────────>              ├───┤
         │      │╱                                      │y₂ │
         └──────┘                                       ├───┤
                                                        │y₃ │
                                                        └───┘
```

#### The Architectural Revolution of GAP:
In classic architectures like AlexNet and VGG, the final convolutional volume was flattened into a 1D vector and fed into massive Fully Connected (Dense) layers:
$$\text{Flatten}([512, 7, 7]) = 25,088 \xrightarrow{W \in \mathbb{R}^{4096 \times 25088}} 4096 \implies \mathbf{102.7 \text{ Million Parameters!}}$$
Over $80\%$ of all parameters in VGGNet were trapped in these dense layers, causing extreme vulnerability to overfitting.

By contrast, replacing dense layers with Global Average Pooling directly produces a vector of size $C = 512$, which feeds into a single linear classifier ($512 \times 1000 = 512,000$ parameters)—a **$99.5\%$ parameter reduction** with zero risk of spatial overfitting!

---

## 4. Invariance Properties: Equivariance vs. Invariance

A crucial conceptual distinction in computer vision is that between **equivariance** and **invariance**:

1. **Translation Equivariance (Convolutions):**
   - An operator $f$ is equivariant with respect to a transformation $g$ if transforming the input yields an identically transformed output:
     $$f(g(X)) = g(f(X))$$
   - In convolutions: If you translate an image by 5 pixels, the resulting activations in the feature map shift by exactly 5 pixels.

2. **Translation Invariance (Pooling):**
   - An operator $h$ is invariant with respect to transformation $g$ if transforming the input yields an unchanged output:
     $$h(g(X)) = h(X)$$
   - In Max Pooling: If a high activation shifts within the $K \times K$ pooling window, the maximum value extracted remains unchanged!

```
               EQUIVARIANCE VS. INVARIANCE IN CNN HIERARCHIES
               
   Input X                 Conv Feature Map f(X)        Pooled Feature h(f(X))
   [0, 10, 0, 0]    ───>    [0, 10, 0, 0]        ───>     [10]
   
   Shifted X'               Shifted f(X')                Same Pooled h(f(X'))
   [0, 0, 10, 0]    ───>    [0, 0, 10, 0]        ───>     [10]
                            (Equivariant)                 (Invariant)
```

---

## 5. Backpropagation Through Pooling Layers

Because pooling operations have no learnable parameters, we do not compute gradients with respect to weights ($\frac{\partial \mathcal{L}}{\partial W} = \emptyset$). We only compute the upstream-to-downstream gradient pass: $\frac{\partial \mathcal{L}}{\partial X}$.

### 5.1 Backpropagation Through MaxPooling: Argmax Routing Masks
During the forward pass, for each output coordinate $(c, i, j)$, we record the exact input coordinate $(m^*, n^*)$ that achieved the maximum:
$$(m^*, n^*) = \operatorname{argmax}_{(m, n) \in \Omega_{i, j}} X_{c, m, n}$$

Because the non-maximal elements did not contribute to the output, their local partial derivatives are zero:
$$\frac{\partial Y_{c, i, j}}{\partial X_{c, m, n}} = \begin{cases} 1 & \text{if } (m, n) = (m^*, n^*) \\ 0 & \text{otherwise} \end{cases}$$

By the multivariate chain rule, the gradient flowing back to input pixel $(m, n)$ is:
$$\frac{\partial \mathcal{L}}{\partial X_{c, m, n}} = \sum_{(i, j) \text{ s.t. } (m, n) \in \Omega_{i, j}} \frac{\partial \mathcal{L}}{\partial Y_{c, i, j}} \cdot \mathbf{1}[(m, n) = (m^*, n^*)]$$

*Key Observation:* In non-overlapping pooling ($S = K$), each input pixel belongs to exactly one window. The upstream gradient is routed **exclusively to the single winning argmax neuron**, leaving all other input neurons with zero gradient!

```
                  MAX-POOLING BACKPROPAGATION GRADIENT ROUTING
                  
       Forward Activations               Backward Loss Gradient
          ┌──────┬──────┐                     ┌──────┬──────┐
          │  3   │ 10*  │                     │  0   │ 2.5  │  <-- dL/dY = 2.5
          ├──────┼──────┤                     ├──────┼──────┤       routes solely to
          │  1   │  4   │                     │  0   │  0   │       winning index
          └──────┴──────┘                     └──────┴──────┘
```

### 5.2 Backpropagation Through AvgPooling: Uniform Distribution
For Average Pooling, every element in the window $\Omega_{i, j}$ contributes equally with weight $\frac{1}{K^2}$:
$$\frac{\partial Y_{c, i, j}}{\partial X_{c, m, n}} = \frac{1}{K^2}, \qquad \forall (m, n) \in \Omega_{i, j}$$

Therefore, the upstream gradient is distributed **uniformly** across all $K^2$ input neurons:
$$\frac{\partial \mathcal{L}}{\partial X_{c, m, n}} = \sum_{(i, j) \text{ s.t. } (m, n) \in \Omega_{i, j}} \frac{1}{K^2} \frac{\partial \mathcal{L}}{\partial Y_{c, i, j}}$$

### 5.3 Backpropagation Through Global Average Pooling
For Global Average Pooling, the gradient is scaled by $\frac{1}{H \cdot W}$ and broadcast to every spatial location in channel $c$:
$$\frac{\partial \mathcal{L}}{\partial X_{c, r, col}} = \frac{1}{H \cdot W} \frac{\partial \mathcal{L}}{\partial y_c}$$

---

## 6. Modern Perspectives: Strided Convolutions vs. Pooling

In recent computer vision architectures (e.g. All-CNN, ConvNeXt), explicit pooling layers are often replaced by **strided convolutions** ($K=3, S=2$ or $K=2, S=2$).

| Characteristic | Max / Average Pooling | Strided Convolution ($S=2$) |
| :--- | :--- | :--- |
| **Parameters** | 0 (Fixed, deterministic) | $C_{\text{out}} \times C_{\text{in}} \times K^2$ (Learnable) |
| **Adaptability** | Hardcoded heuristic (max or mean) | Data-driven optimal downsampling |
| **Information Loss** | Discards non-maximal elements | Retains linear combinations across channels |
| **Reversibility** | Harder to invert (unpooling needs masks) | Transposed convolution / Strided adjoint |

Despite this shift, **Global Average Pooling** remains ubiquitous as the standard bridge between spatial backbone representations and linear classification heads.
