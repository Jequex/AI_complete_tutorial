# Topic 13.1: Exercises & Solutions — PyTorch Tensors & Autograd

---

## Problem Set Overview

This exercise set explores low-level memory addressing in multidimensional tensors, stride computation across permutations, and reverse-mode automatic differentiation over dynamic computational DAGs.

---

### Problem 1: Memory Address Mapping in 4D Vision Tensors (NCHW to NHWC)

In deep learning for computer vision, image tensors are commonly formatted in **NCHW** order (Batch, Channels, Height, Width) or **NHWC** order (Batch, Height, Width, Channels).

Suppose an input tensor has batch size $N=4$, channels $C=3$, height $H=32$, and width $W=32$.
1. Derive the canonical C-contiguous strides vector $\mathbf{s}_{\text{NCHW}}$ for the shape $(4, 3, 32, 32)$.
2. Compute the exact physical 1D storage index for element $(n=2, c=1, h=10, w=15)$ assuming `storage_offset = 0`.
3. If the tensor is permuted via `x.permute(0, 2, 3, 1)` to $(N, H, W, C)$:
   - What are the resulting shape and strides?
   - Is this permuted tensor contiguous? Why or why not?
   - What is the physical storage index of element $(n=2, h=10, w=15, c=1)$ in the permuted tensor?

---

### Solution 1

1. **Stride Vector Derivation:**
   For shape $(d_0, d_1, d_2, d_3) = (4, 3, 32, 32)$:
   $$s_3 = 1$$
   $$s_2 = s_3 \cdot d_3 = 1 \cdot 32 = 32$$
   $$s_1 = s_2 \cdot d_2 = 32 \cdot 32 = 1024$$
   $$s_0 = s_1 \cdot d_1 = 1024 \cdot 3 = 3072$$
   Therefore:
   $$\mathbf{s}_{\text{NCHW}} = (3072, 1024, 32, 1)$$

2. **Storage Index Calculation:**
   $$\text{addr}(2, 1, 10, 15) = 2 \cdot 3072 + 1 \cdot 1024 + 10 \cdot 32 + 15 \cdot 1$$
   $$\text{addr} = 6144 + 1024 + 320 + 15 = 7503$$
   The element resides at offset $7503$ in the underlying 1D storage array.

3. **Permutation Analysis (`permute(0, 2, 3, 1)`):**
   - **New Shape:** $(N, H, W, C) = (4, 32, 32, 3)$
   - **New Strides:** The strides vector permutes by the same index mapping $(s_0, s_2, s_3, s_1)$:
     $$\mathbf{s}_{\text{permuted}} = (3072, 32, 1, 1024)$$
   - **Contiguity:** Standard C-contiguous strides for shape $(4, 32, 32, 3)$ would be $(32 \times 32 \times 3, 32 \times 3, 3, 1) = (3072, 96, 3, 1)$. Since $(3072, 32, 1, 1024) \ne (3072, 96, 3, 1)$, the tensor is **non-contiguous**.
   - **Index in Permuted Tensor:**
     $$\text{addr}_{\text{perm}}(2, 10, 15, 1) = 2 \cdot 3072 + 10 \cdot 32 + 15 \cdot 1 + 1 \cdot 1024 = 6144 + 320 + 15 + 1024 = 7503$$
     Notice that the calculated physical memory address is identical ($7503$), proving that permutation is a zero-copy operation that re-indexes the same physical byte!

---

### Problem 2: Dynamic DAG Gradient Accumulation with Diamond Dependencies

Consider a scalar dynamic graph where a single variable $x$ splits into two pathways and recombines:
$$u = x^2$$
$$v = 3x$$
$$y = u \cdot v + u$$

Let $x_0 = 2.0$.
1. Draw the computational DAG and identify each node's `grad_fn`.
2. Compute the analytical derivative $\frac{dy}{dx}$ evaluated at $x = 2.0$.
3. Trace the reverse-mode automatic differentiation step-by-step showing how gradients accumulate at node $u$ and node $x$.

---

### Solution 2

1. **Computational DAG:**
   - Input: $x$ ($x=2.0$)
   - Node $u = x^2$ (`grad_fn = <PowBackward>`)
   - Node $v = 3x$ (`grad_fn = <MulBackward>`)
   - Node $w = u \cdot v$ (`grad_fn = <MulBackward>`)
   - Root $y = w + u$ (`grad_fn = <AddBackward>`)

2. **Analytical Derivative:**
   Express $y$ purely in terms of $x$:
   $$y(x) = (x^2)(3x) + x^2 = 3x^3 + x^2$$
   $$\frac{dy}{dx} = 9x^2 + 2x$$
   Evaluating at $x = 2.0$:
   $$\left.\frac{dy}{dx}\right|_{x=2.0} = 9(2^2) + 2(2) = 9(4) + 4 = 40.0$$

3. **Step-by-Step Autograd Trace:**
   - **Forward values:**
     $x = 2.0$
     $u = 4.0$
     $v = 6.0$
     $w = 4.0 \times 6.0 = 24.0$
     $y = 24.0 + 4.0 = 28.0$

   - **Backward pass:**
     - Initialize root: $\bar{y} = \frac{\partial y}{\partial y} = 1.0$.
     - Reverse Node $y = w + u$:
       $$\bar{w} = \bar{y} \cdot 1.0 = 1.0$$
       $$\bar{u}_{\text{from } y} = \bar{y} \cdot 1.0 = 1.0$$
     - Reverse Node $w = u \cdot v$:
       $$\bar{u}_{\text{from } w} = \bar{w} \cdot v = 1.0 \times 6.0 = 6.0$$
       $$\bar{v} = \bar{w} \cdot u = 1.0 \times 4.0 = 4.0$$
     - **Gradient Accumulation at Node $u$:**
       $$\bar{u}_{\text{total}} = \bar{u}_{\text{from } y} + \bar{u}_{\text{from } w} = 1.0 + 6.0 = 7.0$$
     - Reverse Node $v = 3x$:
       $$\bar{x}_{\text{from } v} = \bar{v} \cdot 3 = 4.0 \times 3 = 12.0$$
     - Reverse Node $u = x^2$:
       $$\bar{x}_{\text{from } u} = \bar{u}_{\text{total}} \cdot (2x) = 7.0 \times (2 \times 2.0) = 7.0 \times 4.0 = 28.0$$
     - **Gradient Accumulation at Node $x$:**
       $$\bar{x}_{\text{total}} = \bar{x}_{\text{from } v} + \bar{x}_{\text{from } u} = 12.0 + 28.0 = 40.0$$

   The autograd trace yields exactly $40.0$, matching the analytical derivative.

---

### Problem 3: Practical PyTorch Memory Optimization

You are deploying a large vision transformer for inference on an edge device with strictly constrained GPU VRAM. An engineer notices that running 100 consecutive forward passes causes a `CUDA out of memory` crash.

Here is the problematic snippet:
```python
losses = []
for batch_x, batch_y in dataloader:
    out = model(batch_x)
    loss = criterion(out, batch_y)
    losses.append(loss)  # Line causing memory leak
```

**Questions:**
1. Why does `losses.append(loss)` cause a severe memory leak?
2. What two modifications must be made to eliminate the leak and optimize inference throughput?

---

### Solution 3

1. **Root Cause:**
   The variable `loss` is a PyTorch tensor attached to a live computational DAG via its `.grad_fn`. Appending `loss` directly to a Python list prevents the garbage collector from freeing the entire backward graph, including all intermediate layer activations saved for backpropagation across all previous batches.

2. **Remediations:**
   - **Extract scalar value:** If only logging the loss value, append a Python float using `loss.item()` or `float(loss)`.
   - **Disable Autograd:** During evaluation, wrap the loop in `torch.inference_mode()` or `torch.no_grad()`:
     ```python
     with torch.inference_mode():
         for batch_x, batch_y in dataloader:
             out = model(batch_x)
             loss = criterion(out, batch_y)
             losses.append(loss.item())
     ```
