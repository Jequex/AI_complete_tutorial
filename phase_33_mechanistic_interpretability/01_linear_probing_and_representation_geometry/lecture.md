# Topic 01: Linear Probing & Representation Geometry

## 1. Introduction & 12-Year-Old Intuition

Imagine you are looking at a bustling train station during morning rush hour. Thousands of passengers are moving through the central concourse, carrying briefcases, backpacks, cups of coffee, and tickets. 

To an outside observer staring through the glass ceiling, the crowd looks like an incomprehensible, swirling cloud of chaos. But suppose you hold up a special pair of tinted red glasses: suddenly, you notice that every commuter heading toward Platform 4 is walking along an exact diagonal line across the floor. If you hold up blue glasses, everyone heading toward the exit is walking along a perpendicular line.

The crowd wasn't disorganized at all—it was organized along **geometric directions** in space.

In a deep neural network (specifically a Transformer), the sequence of hidden activations passing from the input to the output is called the **residual stream**. Think of the residual stream as a shared highway or communication bus. As a sentence flows through the network:
1. Early layers notice shallow details (letters, punctuation, word positions).
2. Middle and deep layers extract abstract ideas (sentiment, factual relationships, grammatical gender, truthfulness).

The **Linear Representation Hypothesis** states that deep networks represent human-understandable concepts as **linear directions (1D vectors)** in their high-dimensional hidden spaces. A **linear probe** is simply a mathematical detector (like our tinted glasses) that asks: *"If I look strictly along this single straight direction in layer $l$, can I tell whether the text is positive or negative, or whether a city is in France or Japan?"*

If a simple straight line can separate the concept, the network has organized that fact linearly, making it easily readable by subsequent layers.

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Entity | Dimensionality / Domain | Operational Meaning & Interpretability Role |
| :--- | :--- | :--- | :--- |
| $\mathbf{x}_i$ | Input Token / Embedding | $\mathbb{R}^d$ | Initial representation entering the residual stream at layer 0. |
| $\mathbf{h}_l \in \mathbb{R}^d$ | Residual Stream Activation | $d$-dimensional vector | The state of the residual stream at layer $l$ for a specific token position. |
| $\mathcal{M}_l$ | Layer Module | Function $\mathbb{R}^d \to \mathbb{R}^d$ | The $l$-th Transformer block (combining Multi-Head Attention and MLP). |
| $\mathbf{v}_c$ | Concept Direction Vector | $\mathbb{R}^d, \|\mathbf{v}_c\|_2 = 1$ | Unit vector spanning the 1D subspace encoding semantic concept $c$. |
| $\mathbf{w} \in \mathbb{R}^d, b \in \mathbb{R}$ | Linear Probe Parameters | Weights $\mathbf{w}$, bias $b$ | Hyperplane normal vector and offset defining the decision boundary. |
| $\sigma(z)$ | Sigmoid Activation | Scalar in $(0, 1)$ | $\sigma(z) = \frac{1}{1 + e^{-z}}$, maps linear logits to posterior probabilities. |
| $\mathcal{L}_{\text{BCE}}$ | Binary Cross-Entropy Loss | Scalar $\ge 0$ | Optimization objective for logistic classification probes. |
| $\lambda$ | Regularization Coefficient | Positive real scalar | Weight decay parameter enforcing simplicity in probe weights. |
| $\text{Acc}_{\text{task}}$ | Real Task Probe Accuracy | $[0, 1]$ | Accuracy of the probe trained on true concept labels $y \in \{0, 1\}$. |
| $\text{Acc}_{\text{control}}$ | Control Task Accuracy | $[0, 1]$ | Accuracy of an identical probe trained on random permutation labels. |
| $\text{Sel}$ | Probe Selectivity | $[-1, 1]$ | $\text{Sel} = \text{Acc}_{\text{task}} - \text{Acc}_{\text{control}}$, measuring genuine representation vs probe memorization. |
| $\cos(\theta)$ | Cosine Similarity | $[-1, 1]$ | $\frac{\mathbf{w}^\top \mathbf{v}_c}{\|\mathbf{w}\|_2 \|\mathbf{v}_c\|_2}$, spatial alignment between probe and ground-truth concept vector. |

---

## 3. The Residual Stream as a Shared Communication Bus

In classical deep feedforward networks, each layer completely replaces the previous layer's representation:

$$\mathbf{h}_{l+1} = \sigma(W_l \mathbf{h}_l + \mathbf{b}_l)$$

In modern Transformers with residual connections (He et al., 2016; Vaswani et al., 2017), the architecture enforces an additive formulation:

$$\mathbf{h}_{l+1} = \mathbf{h}_l + \text{Attn}_l(\mathbf{h}_l) + \text{MLP}_l(\mathbf{h}_l + \text{Attn}_l(\mathbf{h}_l))$$

Expanding this recurrence from layer $0$ to layer $L$:

$$\mathbf{h}_L = \mathbf{h}_0 + \sum_{l=0}^{L-1} \Delta \mathbf{h}_l^{\text{attn}} + \sum_{l=0}^{L-1} \Delta \mathbf{h}_l^{\text{mlp}}$$

```
Layer 0 ────▶ [ + ] ────────────▶ [ + ] ────────────▶ ... ────▶ Final Unembed
               ▲                   ▲
               │                   │
         [ Attention 0 ]      [  MLP 0  ]
               ▲                   ▲
               └─────── Read ──────┘
```

### Architectural Implications:
1. **The Highway / Bus**: The residual stream retains earlier information by default. Layers do not need to re-encode the original input; they simply read from the stream, perform isolated computation, and *add* their output vector $\Delta \mathbf{h}$ back to the stream.
2. **Bandwidth Constraint**: Since the vector dimension $d$ (e.g., 4096 in LLaMA-7B) is fixed throughout all layers, every head and sublayer must write to this shared $d$-dimensional space without catastrophically corrupting other signals.

---

## 4. The Linear Representation Hypothesis

Why should semantic features be linear?

> **The Linear Representation Hypothesis (LRH)**: High-level semantic features and factual properties are represented as linear directions in activation space. That is, a feature $c$ corresponds to a direction $\mathbf{v}_c \in \mathbb{R}^d$ such that the presence or magnitude of feature $c$ in activation $\mathbf{h}$ is given by the inner product:
>
> $$\text{val}(c; \mathbf{h}) \approx \mathbf{v}_c^\top \mathbf{h}$$

### Why Linear Directions?
- **Matrix Multiplication as Feature Readout**: In neural networks, downstream layers read the residual stream via matrix-vector multiplication $W \mathbf{h}$. If feature $c$ is aligned with a linear direction $\mathbf{v}_c$, any downstream neuron whose weight row is $\mathbf{w}_j \propto \mathbf{v}_c$ automatically computes:
  
  $$\mathbf{w}_j^\top \mathbf{h} = \alpha (\mathbf{v}_c^\top \mathbf{h}) + \text{noise}$$
  
  Linear encoding allows every downstream attention head and MLP layer to read or ignore features with a single linear operation.
- **Superposition Compatibility**: Linear directions can be packed almost orthogonally (Johnson-Lindenstrauss lemma) into high dimensions, allowing thousands of sparse concepts to coexist simultaneously.

---

## 5. Linear Probes: Formulation & Training

A **linear probe** is a supervised linear model trained on top of frozen intermediate activations $\mathbf{h}_l \in \mathbb{R}^d$.

Given a dataset of $N$ input prompts with intermediate activations $\{\mathbf{h}_l^{(i)}\}_{i=1}^N$ and target concept labels $y^{(i)} \in \{0, 1\}$:

### 5.1 Logistic Regression Probe
The posterior probability predicted by the probe is:

$$P(y=1 \mid \mathbf{h}_l) = \sigma\left(\mathbf{w}^\top \mathbf{h}_l + b\right) = \frac{1}{1 + \exp\left(-(\mathbf{w}^\top \mathbf{h}_l + b)\right)}$$

The regularized objective minimizes the binary cross-entropy loss with an $L_2$ penalty:

$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y^{(i)} \ln \sigma(\mathbf{w}^\top \mathbf{h}_l^{(i)} + b) + (1 - y^{(i)}) \ln(1 - \sigma(\mathbf{w}^\top \mathbf{h}_l^{(i)} + b)) \right] + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$

### 5.2 Gradient Calculation for SGD
Letting $p^{(i)} = \sigma(\mathbf{w}^\top \mathbf{h}_l^{(i)} + b)$, the analytical gradients are:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{1}{N} \sum_{i=1}^N \left(p^{(i)} - y^{(i)}\right) \mathbf{h}_l^{(i)} + \lambda \mathbf{w}$$

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \sum_{i=1}^N \left(p^{(i)} - y^{(i)}\right)$$

The probe is updated via gradient descent:
$$\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{w}}, \quad b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}$$

---

## 6. Ruling Out Probe Memorization: Selectivity & Control Tasks

A critical methodological objection to linear probing is:
> *"Does the probe reveal a feature already utilized by the network, or is the probe simply learning/memorizing the task itself because it has parameters?"*

In seminal work, **Hewitt and Liang (2019)** demonstrated that high probe accuracy does not necessarily prove that the representation linearly encodes the property if the probe has sufficient capacity to memorize arbitrary patterns.

### 6.1 The Control Task Framework
To evaluate whether the probe is merely memorizing or reading true structure:
1. Define a **control task** where the labels $y_{\text{rand}}^{(i)} \sim \text{Bernoulli}(0.5)$ are randomly assigned to inputs, destroying any genuine semantic relationship while maintaining identical input activation distributions.
2. Train an identical probe architecture with identical hyperparameters on the control dataset:
   - Real task accuracy: $\text{Acc}_{\text{task}}$
   - Control task accuracy: $\text{Acc}_{\text{control}}$
3. Define **Probe Selectivity**:
   
   $$\boxed{\text{Selectivity} = \text{Acc}_{\text{task}} - \text{Acc}_{\text{control}}}$$

### 6.2 Decision Criteria
- **High Accuracy, Low Selectivity ($\text{Acc}_{\text{task}} \approx 95\%, \text{Acc}_{\text{control}} \approx 92\%, \text{Sel} \approx 0.03$)**: The probe is memorizing the dataset through high capacity or dimensional overfitting. The representation does not cleanly encode the concept.
- **High Accuracy, High Selectivity ($\text{Acc}_{\text{task}} \approx 99\%, \text{Acc}_{\text{control}} \approx 50\%, \text{Sel} \approx 0.49$)**: The representation genuinely aligns with the concept, and the linear structure is native to the network's internal geometry.

---

## 7. Probe Accuracy vs Representation Cosine Similarity

Does probe accuracy tell the whole story? Not necessarily.

1. **Probe Weight Alignment**: If the network uses direction $\mathbf{v}_c$ to encode the concept, the optimal probe weight vector $\mathbf{w}^*$ should align with $\mathbf{v}_c$:
   
   $$\cos(\theta) = \frac{\mathbf{w}^\top \mathbf{v}_c}{\|\mathbf{w}\|_2 \|\mathbf{v}_c\|_2} \approx 1.0$$

2. **SNR (Signal-to-Noise Ratio)**: If the concept projection has magnitude $s$ and the orthogonal noise in the residual stream has variance $\sigma_{\text{noise}}^2$, the probe accuracy is determined by the separation distance:
   
   $$d' = \frac{2s}{\sigma_{\text{noise}}}$$
   
   Across network depth, as shown in our accompanying `experiment.py`, shallow layers exhibit low selectivity and low cosine similarity, while middle-to-deep layers align tightly ($\cos(\theta) > 0.95$) with maximum selectivity.

---

## 8. Summary of Engineering Best Practices
- **Never evaluate probes on training data only**: Always report test generalization on held-out prompts.
- **Always run randomized label controls**: Calculate selectivity before concluding a concept is linearly represented.
- **Sweep across depth**: Linear emergence is progressive; shallow layers often entangle features non-linearly, whereas late layers disentangle them into linear directions for the final unembedding projection $W_U$.
