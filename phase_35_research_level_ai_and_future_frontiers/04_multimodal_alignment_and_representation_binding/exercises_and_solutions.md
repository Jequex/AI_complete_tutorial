# Topic 04: Exercises & Solutions — Multimodal Alignment & Representation Binding

## Exercise 1: Analytical Gradients of the InfoNCE Loss

Consider the Image-to-Text contrastive loss for a batch of $B$ normalized embedding pairs $(\hat{\mathbf{z}}_{v, i}, \hat{\mathbf{z}}_{t, i})$ on the unit sphere ($\|\hat{\mathbf{z}}_{v, i}\|_2 = \|\hat{\mathbf{z}}_{t, i}\|_2 = 1$):

$$\mathcal{L}_{v2t} = -\frac{1}{B} \sum_{i=1}^B \ln \left( \frac{\exp(\hat{\mathbf{z}}_{v, i}^\top \hat{\mathbf{z}}_{t, i} / \tau)}{\sum_{j=1}^B \exp(\hat{\mathbf{z}}_{v, i}^\top \hat{\mathbf{z}}_{t, j} / \tau)} \right)$$

where $\tau > 0$ is the temperature hyperparameter.

Let $P_{ij} = \frac{\exp(\hat{\mathbf{z}}_{v, i}^\top \hat{\mathbf{z}}_{t, j} / \tau)}{\sum_{k=1}^B \exp(\hat{\mathbf{z}}_{v, i}^\top \hat{\mathbf{z}}_{t, k} / \tau)}$ be the predicted posterior probability distribution across captions for image $i$.

1. Derive the partial gradient $\frac{\partial \mathcal{L}_{v2t}}{\partial \hat{\mathbf{z}}_{v, i}}$ with respect to the normalized vision embedding $\hat{\mathbf{z}}_{v, i}$.
2. Interpret the resulting gradient as a balance of **attractive** and **repulsive** geometric forces in representation space.

---

### Solution 1:

#### 1. Mathematical Derivation
Rewrite the loss for a single image $i$:
$$\ell_i = -\frac{1}{\tau} \hat{\mathbf{z}}_{v, i}^\top \hat{\mathbf{z}}_{t, i} + \ln \left( \sum_{j=1}^B \exp\left( \frac{\hat{\mathbf{z}}_{v, i}^\top \hat{\mathbf{z}}_{t, j}}{\tau} \right) \right)$$
Differentiate $\ell_i$ with respect to $\hat{\mathbf{z}}_{v, i}$:
$$\frac{\partial \ell_i}{\partial \hat{\mathbf{z}}_{v, i}} = -\frac{1}{\tau} \hat{\mathbf{z}}_{t, i} + \frac{1}{\sum_{k=1}^B \exp(\hat{\mathbf{z}}_{v, i}^\top \hat{\mathbf{z}}_{t, k} / \tau)} \sum_{j=1}^B \frac{1}{\tau} \exp\left( \frac{\hat{\mathbf{z}}_{v, i}^\top \hat{\mathbf{z}}_{t, j}}{\tau} \right) \hat{\mathbf{z}}_{t, j}$$
Substitute the definition of $P_{ij}$:
$$\frac{\partial \ell_i}{\partial \hat{\mathbf{z}}_{v, i}} = -\frac{1}{\tau} \hat{\mathbf{z}}_{t, i} + \frac{1}{\tau} \sum_{j=1}^B P_{ij} \hat{\mathbf{z}}_{t, j}$$
Separating the positive term ($j = i$) and negative distractors ($j \neq i$):
$$\frac{\partial \ell_i}{\partial \hat{\mathbf{z}}_{v, i}} = -\frac{1}{\tau} (1 - P_{ii}) \hat{\mathbf{z}}_{t, i} + \frac{1}{\tau} \sum_{j \neq i} P_{ij} \hat{\mathbf{z}}_{t, j}$$
Averaging over the batch ($B$):
$$\frac{\partial \mathcal{L}_{v2t}}{\partial \hat{\mathbf{z}}_{v, i}} = \frac{1}{B \tau} \left( -(1 - P_{ii}) \hat{\mathbf{z}}_{t, i} + \sum_{j \neq i} P_{ij} \hat{\mathbf{z}}_{t, j} \right)$$

#### 2. Geometric Force Interpretation
The gradient descent update moves $\hat{\mathbf{z}}_{v, i}$ in the direction $-\nabla \mathcal{L}$:
$$\Delta \hat{\mathbf{z}}_{v, i} \propto \underbrace{(1 - P_{ii}) \hat{\mathbf{z}}_{t, i}}_{\text{Attractive Force}} - \underbrace{\sum_{j \neq i} P_{ij} \hat{\mathbf{z}}_{t, j}}_{\text{Repulsive Forces}}$$
- **Attractive Force**: Pulls $\hat{\mathbf{z}}_{v, i}$ directly toward its ground-truth caption embedding $\hat{\mathbf{z}}_{t, i}$, with strength proportional to the error $(1 - P_{ii})$. If $P_{ii} \approx 1$, the attractive force vanishes.
- **Repulsive Force**: Pushes $\hat{\mathbf{z}}_{v, i}$ away from all negative captions $\hat{\mathbf{z}}_{t, j}$, weighted by their current probability $P_{ij}$. Hard negative captions that have high accidental similarity receive large repulsive pushes, driving negative pairs apart.

---

## Exercise 2: SigLIP vs. CLIP Distributed Communication Scaling

Suppose a distributed cluster of $K = 128$ GPUs trains a vision-language model.
- Global batch size: $B = 32,768$ (each GPU holds a local batch $b = \frac{B}{K} = 256$ pairs).
- Embedding vector dimension: $d = 768$ in FP32 ($4$ bytes per float).

1. In standard **CLIP**, computing the softmax denominator requires every GPU to collect all $B$ text embeddings from all other nodes via an `All-Gather` collective communication operation. Calculate the volume of network data gathered by each GPU per training step.
2. In **SigLIP**, the loss is formulated as independent pairwise binary classifications:
   $$\mathcal{L}_{\text{SigLIP}} = -\sum_{i=1}^B \sum_{j=1}^B \left[ y_{ij} \ln \sigma(t_{ij}) + (1 - y_{ij}) \ln(1 - \sigma(t_{ij})) \right]$$
   Explain how SigLIP can be implemented via asynchronous ring chunking without an all-node memory synchronization barrier, and compare the communication overhead.

---

### Solution 2:

#### 1. CLIP All-Gather Communication Footprint
In standard CLIP, each GPU must gather the full global text embedding matrix $\mathbf{Z}_t \in \mathbb{R}^{B \times d}$:
$$N_{\text{floats}} = B \times d = 32,768 \times 768 = 25,165,824 \text{ floats}$$
Memory footprint in megabytes:
$$\text{Data} = 25,165,824 \times 4 \text{ bytes} \approx 100.66 \text{ MB per GPU per step}$$
Under a ring-allgather algorithm across $K = 128$ nodes, the total data transmitted over the interconnect per step is:
$$\text{Total Transferred} = \frac{K - 1}{K} \times (B \times d \times 4) \approx 99.88 \text{ MB per worker}$$
At 1,000 steps per minute, network interconnect bandwidth is heavily saturated by gathering these matrices before loss computation can even begin.

#### 2. SigLIP Ring Chunking & Decoupled Scaling
Because SigLIP computes independent sigmoid probabilities $\sigma(z_v^\top z_t)$:
1. A node $k$ does not need all $B$ text embeddings simultaneously in VRAM.
2. Instead, it holds its local $b = 256$ images and receives a rolling buffer of $b = 256$ text embeddings from its neighbor in a ring pipeline.
3. It computes the local $256 \times 256$ block loss, sends the text chunk to the next GPU, and receives the next chunk asynchronously.
4. **Zero Synchronization Barrier**: Training overlap is near $100\%$, and GPUs require only $256 \times 768 \times 4 \text{ bytes} \approx 786 \text{ KB}$ of buffer memory instead of $100 \text{ MB}$, completely eliminating the distributed memory and latency bottleneck.

---

## Exercise 3: The Modality Gap Geometry & The Cone Hypothesis

In 2022, Liang et al. discovered the **Modality Gap**: in trained CLIP models, image embeddings and text embeddings do not intermingle uniformly across the unit hypersphere $\mathbb{S}^{d-1}$. Instead, they cluster into two distinct, non-overlapping geometric **cones**:

$$\mathcal{C}_v = \{ \hat{\mathbf{z}}_v \in \mathbb{S}^{d-1} : \hat{\mathbf{z}}_v^\top \mathbf{c}_v \ge \cos(\theta_v) \}$$
$$\mathcal{C}_t = \{ \hat{\mathbf{z}}_t \in \mathbb{S}^{d-1} : \hat{\mathbf{z}}_t^\top \mathbf{c}_t \ge \cos(\theta_t) \}$$

where $\mathbf{c}_v$ and $\mathbf{c}_t$ are the centroid directions of the vision and text embeddings, and $\mathbf{c}_v^\top \mathbf{c}_t < 0.5$.

1. Explain the primary cause of the modality gap (initialization bias, architecture disparity, and contrastive temperature dynamics).
2. Explain why zero-shot classification and cross-modal retrieval still function with $>95\%$ accuracy despite the presence of this gap.

---

### Solution 3:

#### 1. Causes of the Modality Gap
- **Architecture & Initialization Inductive Bias**: Vision encoders (ViT) and text encoders (Causal/Bidirectional Transformer) initialize with completely different spectral properties. At initialization, images and texts map to orthogonal sub-cones of the sphere.
- **Contrastive Temperature $\tau$**: The InfoNCE objective only cares about **relative pairwise cosine distances** within a batch:
  $$\frac{\hat{\mathbf{z}}_{v, i}^\top \hat{\mathbf{z}}_{t, i} - \hat{\mathbf{z}}_{v, i}^\top \hat{\mathbf{z}}_{t, j}}{\tau}$$
  If $\tau \ll 1$ (e.g. $\tau = 0.01$), a tiny angular difference of $\Delta \cos(\theta) = 0.05$ produces an exponent difference of $\frac{0.05}{0.01} = 5.0$, which yields a massive probability ratio $e^5 \approx 148$. Once positive pairs have a slightly higher dot product than negative pairs, the loss gradients decay to near zero, providing no mathematical incentive to pull the centers of the two modality cones completely into alignment.

#### 2. Why Zero-Shot Classification Remains Highly Accurate
Zero-shot classification relies on **relative rank ordering**, not absolute cosine angles:
$$c^* = \arg\max_{k} \left( \hat{\mathbf{z}}_v^\top \hat{\mathbf{z}}_{t, k} \right)$$
Even if all image-text inner products are shifted by a constant baseline offset (e.g., all similarities lie in $[0.2, 0.4]$ rather than $[0.7, 0.9]$), the relative alignment remains preserved:
$$\hat{\mathbf{z}}_v^\top \hat{\mathbf{z}}_{t, \text{correct}} > \hat{\mathbf{z}}_v^\top \hat{\mathbf{z}}_{t, \text{distractor}}$$
The argmax operator is invariant to monotonic shifts, meaning the modality gap does not impair retrieval or classification performance.
