# Topic 11.1: Forward Propagation & Neural Network Architectures

---

## 1. Executive Summary & The 12-Year-Old Analogy

A **neural network** is not a mysterious sentient being; at its mathematical core, it is a **composition of parameterized affine transformations interleaved with non-linear activation functions**. 

Before we ever touch modern deep learning frameworks like PyTorch or JAX, we must understand the fundamental physical and mathematical operations that govern every forward pass:
1. How individual inputs are weighted and summed by an artificial neuron.
2. How multiple neurons organize into layers to form hierarchical representations.
3. How matrix multiplication ($Z = A W^T + b$) processes an entire batch of data simultaneously with rigorous tensor dimension bookkeeping.
4. Why the **Universal Approximation Theorem** proves neural networks can approximate any continuous function, and why **depth beats width** in practice.

```
                  THE FEEDFORWARD NEURAL NETWORK ARCHITECTURE
                  
  Input Layer          Hidden Layer 1         Hidden Layer 2         Output Layer
   (Features)           (d_1 neurons)          (d_2 neurons)          (Predictions)
     A^[0]                  A^[1]                  A^[2]                  A^[3]
    ┌────┐                 ┌────┐                 ┌────┐                 ┌────┐
    │ x₁ │ ───────────────>│ a₁ │ ───────────────>│ a₁ │ ───────────────>│ ŷ₁ │
    ├────┤      W^[1]      ├────┤      W^[2]      ├────┤      W^[3]      ├────┤
    │ x₂ │ ───────────────>│ a₂ │ ───────────────>│ a₂ │ ───────────────>│ ŷ₂ │
    ├────┤      b^[1]      ├────┤      b^[2]      ├────┤      b^[3]      └────┘
    │ x₃ │ ───────────────>│ a₃ │ ───────────────>│ a₃ │
    └────┘                 └────┘                 └────┘
    [B, d_in]            [B, d_1]               [B, d_2]               [B, d_out]
```

---

### The 12-Year-Old Analogy: The Relay Race of Brain Cells

Imagine a massive school relay race where messages are passed from one end of the stadium to the other:

1. **The Sensory Observers (Input Layer):**
   - Three students stand at the starting line. One measures temperature ("it's 85°F"), one measures cloud cover ("80% clouds"), and one measures wind speed ("20 mph").

2. **The Committee of Listeners (Hidden Layer):**
   - In the next row, five students wait for messages.
   - Each listener doesn't listen to everyone equally! Listener 1 cares a lot about clouds ($w = 0.9$) but ignores temperature ($w = 0.05$). Listener 2 cares only about wind.
   - Each listener also has their own personal optimism or stubbornness score called a **bias ($b$)**: some listeners need only a whisper to get excited ($b = +5$), while others refuse to speak unless shouted at ($b = -10$).

3. **The Baton Pass (Weighted Sum & Activation):**
   - Each listener multiplies every incoming signal by its assigned importance, adds their personal bias ($z = \sum w_i x_i + b$), and runs it through an excitement filter (the **activation function**).
   - If their excitement crosses a threshold, they shout their updated message forward to the next row of students!

4. **The Final Announcement (Output Layer):**
   - The final students on the podium combine the messages from the committee and make the ultimate call: *"There is a 92% chance of rain!"*

5. **Batching (The Super-Highway):**
   - Instead of running this relay race for one single day, imagine doing it for 100 days at the same time using parallel lanes on a giant running track. That is **batch vectorization**: computing answers for an entire batch of examples in a single parallel sweep!

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $\mathbf{x} \in \mathbb{R}^{d_{\text{in}}}$ | Vector | Single input observation with $d_{\text{in}}$ numerical features |
| $X \in \mathbb{R}^{B \times d_{\text{in}}}$ | Matrix / Tensor | Batch of $B$ training examples stacked as rows |
| $B \in \mathbb{Z}_{\ge 1}$ | Scalar integer | Batch size (number of parallel samples processed) |
| $L \in \mathbb{Z}_{\ge 1}$ | Scalar integer | Total number of parameterized layers in the network |
| $l \in \{1, 2, \dots, L\}$ | Index | Index denoting the current layer (superscript $[l]$) |
| $n^{[l]}$ or $d_l$ | Scalar integer | Number of neurons (hidden dimension) in layer $l$ |
| $W^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$ | Matrix | Weight matrix of layer $l$ (maps from $n^{[l-1]}$ to $n^{[l]}$) |
| $\mathbf{b}^{[l]} \in \mathbb{R}^{n^{[l]}}$ | Vector | Bias vector of layer $l$ (adds independent threshold offsets) |
| $Z^{[l]} \in \mathbb{R}^{B \times n^{[l]}}$ | Matrix | Linear pre-activation tensor: $Z^{[l]} = A^{[l-1]} (W^{[l]})^T + \mathbf{b}^{[l]}$ |
| $g^{[l]}(\cdot)$ | Function | Element-wise non-linear activation function at layer $l$ |
| $A^{[l]} \in \mathbb{R}^{B \times n^{[l]}}$ | Matrix | Post-activation tensor: $A^{[l]} = g^{[l]}(Z^{[l]})$, with $A^{[0]} \equiv X$ |
| $\hat{Y} \equiv A^{[L]} \in \mathbb{R}^{B \times n^{[L]}}$ | Matrix | Network's final predicted outputs / class probabilities |

---

## 3. The Artificial Neuron: Biology vs Mathematical Reality

### 3.1 Biological Inspiration
In 1943, neurophysiologist Warren McCulloch and logician Walter Pitts proposed the first conceptual model of a biological neuron:
- **Dendrites:** Branching fibers that receive biochemical electrical signals from preceding neurons.
- **Soma (Cell Body):** Integrates and sums incoming electrical action potentials.
- **Axon Hillock & Axon:** If the integrated electrical charge exceeds a resting potential threshold, an all-or-none spike (action potential) is triggered down the axon.
- **Synapses:** Specialized junctions that scale signal transmission strength via neurotransmitter vesicles.

```
       BIOLOGICAL NEURON                   ARTIFICIAL NEURON (McCULLOCH-PITTS / PERCEPTRON)
   Dendrites        Soma         Axon           Inputs        Weights      Sum & Bias     Activation
     \   /         ┌─────┐        |               x₁ ─────────> w₁ ───┐
      \ /          │ Sum │        |               x₂ ─────────> w₂ ───┼──> Σ (w_i x_i) + b ──> g(z) ──> a
  ─────●───────────│  &  │────────●──>            x₃ ─────────> w₃ ───┘
      / \          │Fire!│        |                                        (Affine Map)      (Non-linearity)
     /   \         └─────┘        |
```

### 3.2 The Mathematical Reality: Affine Transformation Followed by Non-Linearity
While biological neurons transmit complex continuous temporal spike trains with stochastic refractory periods, an **artificial neuron** is an idealized mathematical function:

$$z = \sum_{j=1}^{d_{\text{in}}} w_j x_j + b = \mathbf{w}^T \mathbf{x} + b$$
$$a = g(z) = g(\mathbf{w}^T \mathbf{x} + b)$$

1. **Affine Map ($\mathbf{w}^T \mathbf{x} + b$):** Computes an inner product between input vector $\mathbf{x}$ and parameter vector $\mathbf{w}$, translating the coordinate origin by scalar bias $b$. Geometrically, this defines a hyper-plane decision boundary $\mathbf{w}^T \mathbf{x} + b = 0$ in $\mathbb{R}^{d_{\text{in}}}$.
2. **Scalar Non-Linear Activation ($g$):** Maps the linear score $z \in \mathbb{R}$ onto a bounded or rectified range (e.g., $(0, 1)$ or $[0, \infty)$), granting the neuron the capacity to express non-linear relationships.

---

## 4. Multi-Layer Feedforward Architecture & Tensor Bookkeeping

When we stack multiple layers, the output of layer $l-1$ becomes the input to layer $l$. 

### 4.1 Forward Propagation Equations (Single Sample)
For a single input $\mathbf{x} \in \mathbb{R}^{d_{\text{in}}}$:

$$\mathbf{a}^{[0]} = \mathbf{x}$$
$$\mathbf{z}^{[1]} = W^{[1]} \mathbf{a}^{[0]} + \mathbf{b}^{[1]}$$
$$\mathbf{a}^{[1]} = g^{[1]}(\mathbf{z}^{[1]})$$
$$\vdots$$
$$\mathbf{z}^{[l]} = W^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$
$$\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})$$
$$\vdots$$
$$\hat{\mathbf{y}} = \mathbf{a}^{[L]} = g^{[L]}(\mathbf{z}^{[L]})$$

### 4.2 Batch Vectorization & Shape Bookkeeping
In modern machine learning, feeding one sample at a time to a CPU or GPU is computationally inefficient due to memory bandwidth bottlenecks. Instead, we stack $B$ observations into a batch data matrix $X \in \mathbb{R}^{B \times d_{\text{in}}}$.

Under the standard deep learning convention (used in PyTorch, JAX, and TensorFlow):
- Batch Activation $A^{[l-1]}$ has shape $[B, d_{\text{in}}]$.
- Layer weight matrix $W^{[l]}$ has shape $[d_{\text{out}}, d_{\text{in}}]$.
- Layer bias vector $\mathbf{b}^{[l]}$ has shape $[d_{\text{out}}]$.

To compute all $B$ pre-activations simultaneously, we transpose $W^{[l]}$:

$$\boxed{Z^{[l]} = A^{[l-1]} (W^{[l]})^T + \mathbf{1}_B (\mathbf{b}^{[l]})^T}$$

Let us track the exact tensor dimensions:

$$\underbrace{Z^{[l]}}_{[B \times d_{\text{out}}]} = \underbrace{A^{[l-1]}}_{[B \times d_{\text{in}}]} \times \underbrace{(W^{[l]})^T}_{[d_{\text{in}} \times d_{\text{out}}]} + \underbrace{\mathbf{b}^{[l]}}_{[1 \times d_{\text{out}} \text{ (broadcast to } B \times d_{\text{out}})}$$

Each element $(i, j)$ of $Z^{[l]}$ represents the pre-activation for sample $i$ at neuron $j$:

$$Z_{i, j}^{[l]} = \sum_{k=1}^{d_{\text{in}}} A_{i, k}^{[l-1]} W_{j, k}^{[l]} + b_j^{[l]}$$

Then the activation function $g^{[l]}$ is applied **element-wise**:
$$A_{i, j}^{[l]} = g^{[l]}\left(Z_{i, j}^{[l]}\right)$$

---

## 5. Universal Approximation: Why Depth Beats Width

### 5.1 The Universal Approximation Theorem (Cybenko 1989, Hornik 1991)
> **Theorem (Universal Approximation):** Let $\sigma(\cdot)$ be any non-constant, bounded, continuous activation function (e.g., Sigmoid). Let $I_m = [0, 1]^m$ denote the $m$-dimensional unit hypercube. Then, for any continuous real-valued function $f \in C(I_m)$ and any tolerance $\epsilon > 0$, there exists a single-hidden-layer neural network with a finite number of neurons $N$:
> 
> $$F(\mathbf{x}) = \sum_{i=1}^N \alpha_i \sigma(\mathbf{w}_i^T \mathbf{x} + b_i)$$
> 
> such that $|F(\mathbf{x}) - f(\mathbf{x})| < \epsilon$ for all $\mathbf{x} \in I_m$.

Later extended by Kurt Hornik (1991) to arbitrary non-polynomial activation functions, this proves that even a network with **one single hidden layer** is mathematically capable of approximating any continuous function to arbitrary precision!

### 5.2 The Catch: Why Shallow Networks Fail in Practice
Although a shallow network can *theoretically* approximate any function, the theorem says nothing about:
1. **The required width $N$:** To approximate complex, highly undulating functions or high-dimensional manifolds, $N$ can grow **exponentially** with the input dimension ($N = \mathcal{O}(2^d)$).
2. **Learnability:** Even if such weights exist, gradient descent from random initialization rarely discovers them.

### 5.3 Why Depth Beats Width: The Power of Composition
Deep neural networks factorize complex functions into a **composition of simpler functions**:

$$f(\mathbf{x}) = f_L(f_{L-1}(\dots f_2(f_1(\mathbf{x}))\dots))$$

1. **Exponential Expressivity:**
   A deep network with $L$ layers can fold and partition the input space into $\mathcal{O}(k^L)$ linear regions using only $\mathcal{O}(k \cdot L)$ parameters (Montufar et al., 2014; Telgarsky, 2016). A shallow network requires $\mathcal{O}(k^L)$ neurons to achieve the same partition complexity!
2. **Hierarchical Feature Abstraction:**
   - Layer 1 learns low-level primitives: edges, gradients, local frequencies.
   - Layer 2 composes edges into parts: corners, textures, motifs.
   - Layer 3 composes parts into objects: wheels, eyes, phonemes.
   - Final layers make decisions based on high-level semantic abstractions.
3. **Invariance and Symmetry:**
   Deep architectures naturally learn translation, scale, and deformation invariances layer by layer, matching the compositional physics of our physical universe.

---

## 6. Step-by-Step Numerical Walkthrough

Consider a 2-layer network (1 hidden layer, 1 output layer):
- Input: $\mathbf{x} = [1.0, 2.0]^T$ ($d_{\text{in}} = 2$)
- Hidden layer: $d_1 = 3$ neurons, linear activation (for inspection)
- Output layer: $d_2 = 1$ neuron

Weights and biases:
$$W^{[1]} = \begin{bmatrix} 0.5 & -0.5 \\ 1.0 & 0.2 \\ -1.5 & 2.0 \end{bmatrix}, \quad \mathbf{b}^{[1]} = \begin{bmatrix} 0.1 \\ -0.2 \\ 0.5 \end{bmatrix}$$

$$W^{[2]} = \begin{bmatrix} 0.4 & -0.8 & 0.5 \end{bmatrix}, \quad b^{[2]} = 0.3$$

### Step 1: Pre-activations at Hidden Layer
$$z_1^{[1]} = 0.5(1.0) + (-0.5)(2.0) + 0.1 = 0.5 - 1.0 + 0.1 = -0.4$$
$$z_2^{[1]} = 1.0(1.0) + 0.2(2.0) - 0.2 = 1.0 + 0.4 - 0.2 = 1.2$$
$$z_3^{[1]} = -1.5(1.0) + 2.0(2.0) + 0.5 = -1.5 + 4.0 + 0.5 = 3.0$$

Therefore:
$$\mathbf{z}^{[1]} = \mathbf{a}^{[1]} = \begin{bmatrix} -0.4 \\ 1.2 \\ 3.0 \end{bmatrix}$$

### Step 2: Output Layer Pre-activation
$$z^{[2]} = W^{[2]} \mathbf{a}^{[1]} + b^{[2]} = 0.4(-0.4) + (-0.8)(1.2) + 0.5(3.0) + 0.3$$
$$z^{[2]} = -0.16 - 0.96 + 1.50 + 0.30 = 0.68$$

Final prediction $\hat{y} = 0.68$. Every forward pass in modern deep learning is simply this exact sequence scaled to millions of dimensions!

---

## 7. Key Takeaways & Summary
- An artificial neuron performs an affine transformation $z = \mathbf{w}^T \mathbf{x} + b$ followed by non-linear squashing $a = g(z)$.
- Batch vectorization stacks $B$ observations into rows, computing $Z = A W^T + b$ in parallel ($[B, d_{\text{in}}] \times [d_{\text{in}}, d_{\text{out}}] \to [B, d_{\text{out}}]$).
- While the Universal Approximation Theorem guarantees that a single wide hidden layer can represent any function, deep networks achieve the same representational power with exponentially fewer parameters.
