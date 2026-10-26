# Topic 13.2: Exercises & Solutions — Modules, Data Pipelines & Training Loops

---

## Problem Set Overview

This exercise set explores object-oriented module mechanics, custom `DataLoader` collate functions for dynamic sequence batching, and momentum-accelerated gradient dynamics.

---

### Problem 1: Custom Collate Function for Variable-Length NLP Sequences

When processing natural language text, raw samples from a `Dataset` are integer token lists of varying lengths:
$$\mathbf{s}_1 = [101, 2054, 2003, 102] \quad (\text{length } 4)$$
$$\mathbf{s}_2 = [101, 7592, 102] \quad (\text{length } 3)$$
$$\mathbf{s}_3 = [101, 2023, 2003, 1037, 3899, 102] \quad (\text{length } 6)$$

A standard PyTorch `DataLoader` will fail if default collation is used, because non-uniform tensors cannot be stacked into a single rectangular $(B, L)$ matrix.

**Tasks:**
1. Design a custom `collate_fn` that dynamically pads sequences in a batch to the maximum sequence length $L_{\max}$ in that batch using a designated `pad_token_id = 0`.
2. Construct the binary attention mask matrix $\mathbf{M} \in \{0, 1\}^{B \times L_{\max}}$ where $M_{ij} = 1$ for valid tokens and $0$ for padded positions.
3. Explain why dynamic batch-level padding is computationally superior to static dataset-level padding to a global maximum sequence length.

---

### Solution 1

1. **Custom Collate Implementation (Python 3):**
```python
def dynamic_padding_collate_fn(batch, pad_token_id=0):
    """
    batch: list of tuples (token_ids: list[int], label: int)
    """
    batch_tokens = [item[0] for item in batch]
    batch_labels = [item[1] for item in batch]
    
    # 1. Determine batch-level max length
    max_len = max(len(seq) for seq in batch_tokens)
    B = len(batch)
    
    # 2. Allocate padded tensor and attention mask
    padded_tokens = [[pad_token_id] * max_len for _ in range(B)]
    attention_mask = [[0] * max_len for _ in range(B)]
    
    # 3. Fill values
    for b, seq in enumerate(batch_tokens):
        for t, token in enumerate(seq):
            padded_tokens[b][t] = token
            attention_mask[b][t] = 1
            
    return padded_tokens, attention_mask, batch_labels
```

2. **Generated Attention Mask for the Problem Input:**
   For the 3 sequences with $L_{\max} = 6$:
   $$\mathbf{M} = \begin{bmatrix}
   1 & 1 & 1 & 1 & 0 & 0 \\
   1 & 1 & 1 & 0 & 0 & 0 \\
   1 & 1 & 1 & 1 & 1 & 1
   \end{bmatrix}$$

3. **Computational Advantage of Dynamic Padding:**
   Self-attention in Transformers scales quadratically with sequence length: $\mathcal{O}(B \cdot L^2)$.
   If the global dataset maximum length is $L_{\text{global}} = 512$, but a given batch contains sentences with lengths at most $L_{\text{batch}} = 32$, static padding wastes $(512^2 - 32^2) / 512^2 \approx 99.6\%$ of floating-point operations (FLOPs) processing meaningless pad tokens! Dynamic padding limits computational waste strictly to the maximum length present in each specific mini-batch.

---

### Problem 2: Mathematical Analysis of SGD with Polyak Momentum

Standard SGD updates parameter $\theta$ via:
$$\theta_{t+1} = \theta_t - \alpha g_t$$
where $g_t = \nabla \mathcal{L}(\theta_t)$ and $\alpha > 0$ is the learning rate.

In PyTorch `torch.optim.SGD(..., momentum=\beta)`:
$$v_{t+1} = \beta v_t + g_t$$
$$\theta_{t+1} = \theta_t - \alpha v_{t+1}$$
with initial velocity $v_0 = 0$.

1. Unroll the recurrence relation for $v_t$ to express $v_t$ as an exponentially weighted sum of past gradients $g_0, \dots, g_{t-1}$.
2. Suppose the loss surface is an ill-conditioned quadratic canyon where the gradient along the high-curvature axis oscillates with alternating signs: $g_t^{(y)} = (-1)^t C$. Show how momentum damps these perpendicular oscillations.
3. Compute the effective step size along a persistent ravine where the gradient is constant: $g_t^{(x)} = G > 0$.

---

### Solution 2

1. **Unrolling the Momentum Recurrence:**
   $$v_1 = g_0$$
   $$v_2 = \beta v_1 + g_1 = \beta g_0 + g_1$$
   $$v_3 = \beta v_2 + g_2 = \beta^2 g_0 + \beta g_1 + g_2$$
   By mathematical induction, for step $t$:
   $$v_t = \sum_{k=0}^{t-1} \beta^{t-1-k} g_k$$

2. **Damping Oscillations:**
   Along the oscillatory axis where $g_k = (-1)^k C$:
   $$v_t = C \sum_{k=0}^{t-1} \beta^{t-1-k} (-1)^k$$
   Because $\beta \approx 0.9$ and consecutive terms alternate in sign, adjacent terms largely cancel out:
   $$v_t \approx C (1 - \beta + \beta^2 - \beta^3 + \dots) = \frac{C}{1 + \beta}$$
   For $\beta = 0.9$, $\frac{C}{1 + 0.9} = \frac{C}{1.9} \approx 0.526 C$. The momentum velocity is reduced to roughly half the raw gradient magnitude, preventing catastrophic divergence across steep valley walls.

3. **Effective Step Size in Ravines:**
   Along the ravine axis where $g_k = G$ (constant sign and direction):
   $$v_t = G \sum_{k=0}^{t-1} \beta^{t-1-k} = G \frac{1 - \beta^t}{1 - \beta}$$
   As $t \to \infty$, since $|\beta| < 1$, $\beta^t \to 0$:
   $$v_\infty = \frac{G}{1 - \beta}$$
   The actual parameter step is:
   $$\Delta \theta = -\alpha v_\infty = -\frac{\alpha}{1 - \beta} G$$
   For $\beta = 0.9$:
   $$\frac{\alpha}{1 - 0.9} = 10 \alpha$$
   Momentum effectively accelerates convergence along the flat floor of the valley by an **amplification factor of $10\times$** without requiring an increase in the base learning rate!

---

### Problem 3: State Buffers and Checkpointing Protocol

An engineer adds BatchNorm to an existing architecture:
```python
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.running_mean = torch.zeros(10)  # Bug!
```

**Questions:**
1. Why does defining `self.running_mean = torch.zeros(10)` introduce a silent failure during distributed training and model saving?
2. How should this be written using PyTorch idiomatic patterns?

---

### Solution 3

1. **Failure Mode:**
   Because `self.running_mean` was assigned directly as a regular tensor rather than through `register_buffer()`, three critical bugs occur:
   - When calling `model.to("cuda")`, `self.running_mean` remains on the CPU, causing runtime device mismatch errors during forward evaluation.
   - `self.running_mean` is omitted from `model.state_dict()`. When restoring saved checkpoints, test-time running statistics are lost.
   - During DistributedDataParallel (DDP) synchronization across multiple GPUs, broadcast hooks fail to sync running statistics.

2. **Idiomatic Solution:**
   Register the tensor as a persistent buffer:
   ```python
   class Classifier(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc = nn.Linear(10, 10)
           self.register_buffer("running_mean", torch.zeros(10))
   ```
