# Phase 14: Computer Vision: CNNs to Vision Transformers

---

## 🧭 Executive Overview

Computer vision is the branch of artificial intelligence dedicated to enabling machines to understand, interpret, and extract semantic meaning from visual inputs (images and video). Unlike flat tabular data, visual information exhibits rich **2D spatial locality** and **translation equivariance**: visual motifs like edges, textures, and contours are identical regardless of their coordinates in the image grid.

**Phase 14: Computer Vision: CNNs to Vision Transformers** traces the complete architectural and mathematical evolution of visual representation learning—from 2D discrete convolutions and spatial pooling hierarchies to ResNet's identity shortcut gradient highway and the modern Vision Transformer (ViT) self-attention paradigm.

All concepts are implemented completely from first principles in pure Python 3—with **zero external libraries** (no NumPy, no SciPy, no PyTorch, no OpenCV).

Across four comprehensive modules, this phase covers:
1. **Convolutions & Feature Maps:** The 2D discrete cross-correlation operation, matrix sliding window, spatial dimension arithmetic ($O = \lfloor \frac{W - K + 2P}{S} \rfloor + 1$), multi-channel tensor convolution ($[C_{\text{out}}, C_{\text{in}}, K_h, K_w]$), computational complexity (MACs & FLOPs), Sobel gradient filters, and recursive Receptive Field (RF) propagation ($RF_l = RF_{l-1} + (k_l - 1) J_{l-1}$).
2. **Pooling & Spatial Hierarchies:** Spatial downsampling mechanics: MaxPooling2D, AvgPooling2D, and Global Average Pooling (GAP). Translation invariance vs equivariance, argmax routing masks in backpropagation, uniform gradient distribution, and how GAP eliminates 99.5% of classification head parameters.
3. **Classic CNN Architectures: LeNet, VGG, and ResNet:** The architectural evolution from LeNet-5 to AlexNet and VGGNet. The $3 \times 3$ stacking advantage (28% parameter reduction over $5 \times 5$). The Degradation Problem (higher training error in deep plain networks), ResNet formulation ($\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$), and the mathematical proof of the unhindered Gradient Highway ($\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}}(\frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I})$).
4. **Vision Transformers (ViT):** The visual paradigm shift: "An Image is Worth 16x16 Words". Non-overlapping patch extraction ($N = \frac{HW}{P^2}$), linear projection to dimension $D$, prepending the learnable `[CLS]` token, 1D positional embeddings, Multi-Head Self-Attention, and the trade-off between convolutional inductive bias and foundation-scale Transformer capacity.

---

## 🗺️ Phase Syllabus & Topic Breakdown

```text
phase_14_computer_vision_cnns_to_vits/
├── index.html                                            # Phase Portal Hub & Quickstart
├── README.md                                             # Master Syllabus & Architecture Guide
│
├── 01_convolutions_and_feature_maps/
│   ├── lecture.html / lecture.md                         # 2D cross-correlation, spatial arithmetic, multi-channel convs, FLOPs, RF propagation
│   ├── exercises_and_solutions.html / .md                # Manual conv calculation, AV camera FLOPs analysis, VGG stacking proof
│   └── experiment.py                                     # Pure Python Conv2D, Sobel horizontal/vertical edge detection, RF tracking (Exit Code 0)
│
├── 02_pooling_and_spatial_hierarchies/
│   ├── lecture.html / lecture.md                         # MaxPool, AvgPool, GAP, translation invariance vs equivariance, argmax routing masks
│   ├── exercises_and_solutions.html / .md                # Manual pooling forward/backward, overlapping pooling overlap, VGG vs GAP parameter proof
│   └── experiment.py                                     # Pure Python MaxPool & AvgPool forward/backward with finite-difference check (Exit Code 0)
│
├── 03_classic_cnn_architectures_lenet_vgg_resnet/
│   ├── lecture.html / lecture.md                         # LeNet, AlexNet, VGG factorized convs, Degradation Problem, ResNet Gradient Highway proof
│   ├── exercises_and_solutions.html / .md                # Residual backprop trace, gamma^L vanishing gradient decay, Bottleneck block analysis
│   └── experiment.py                                     # Pure Python 20-block simulation proving ResNet eliminates vanishing gradient (Exit Code 0)
│
└── 04_vision_transformers_vit/
    ├── lecture.html / lecture.md                         # Paradigm shift, patch projection, [CLS] token, 1D pos embeddings, self-attention, inductive bias
    ├── exercises_and_solutions.html / .md                # Manual patch projection & sequence assembly, quadratic resolution scaling, hybrid ViT
    └── experiment.py                                     # Pure Python ViT forward pass (patch slicing, projection, CLS, self-attention) (Exit Code 0)
```

---

## 🔑 Core Mathematical Pillars Mastered in Phase 14

1. **Spatial Output Dimension Arithmetic:**
   $$\boxed{O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1}$$
   Governs spatial downsampling and dimension preservation across convolutional and pooling layers.

2. **Multi-Channel Convolution Volume Tensor:**
   $$\boxed{Y_{k, i, j} = b_k + \sum_{c=0}^{C_{\text{in}}-1} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} X_{c, \, i\cdot S + m, \, j\cdot S + n} \cdot \mathcal{W}_{k, c, m, n}}$$
   Collapses $C_{\text{in}}$ channels into a single scalar response per filter $k$, stacked to form $C_{\text{out}}$ output feature maps.

3. **Recursive Receptive Field Propagation:**
   $$\boxed{RF_l = RF_{l-1} + (k_l - 1) \cdot J_{l-1}, \qquad J_l = J_{l-1} \cdot s_l}$$
   Shows how small local filters compound into large semantic receptive fields over input image space.

4. **The ResNet Gradient Highway Proof:**
   $$\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \left( \mathbf{I} + \frac{\partial}{\partial \mathbf{x}_l} \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i) \right)}$$
   Proves mathematically that the loss gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \cdot \mathbf{I}$ flows backward unhindered through arbitrary network depths, conquering the degradation problem.

5. **Vision Transformer Patch Sequence Assembly:**
   $$\boxed{\mathbf{z}_0 = \left[ \mathbf{x}_{\text{cls}}; \, \mathbf{x}_p^1 E; \, \mathbf{x}_p^2 E; \, \dots; \, \mathbf{x}_p^N E \right] + E_{\text{pos}}}$$
   Converts a 2D image into a sequence of $N+1$ token vectors in $\mathbb{R}^D$ ready for standard Transformer Self-Attention.

---

## 🧪 Quickstart: Running the Labs

All lab scripts run in standard Python 3 with zero external dependencies:

```bash
# Lab 14.1: Convolutions, Sobel edge detection & Receptive Field propagation
python3 phase_14_computer_vision_cnns_to_vits/01_convolutions_and_feature_maps/experiment.py

# Lab 14.2: MaxPooling2D & AvgPooling2D forward/backward gradient checks
python3 phase_14_computer_vision_cnns_to_vits/02_pooling_and_spatial_hierarchies/experiment.py

# Lab 14.3: ResNet Residual Block & 20-block vanishing gradient elimination proof
python3 phase_14_computer_vision_cnns_to_vits/03_classic_cnn_architectures_lenet_vgg_resnet/experiment.py

# Lab 14.4: Vision Transformer patch extraction, [CLS] token & self-attention
python3 phase_14_computer_vision_cnns_to_vits/04_vision_transformers_vit/experiment.py
```
