# Topic 03: Attention Head Mechanisms & Induction Heads

## 1. Introduction & 12-Year-Old Intuition

Imagine you are playing a memory game with cards face down on a long table. 
The dealer lays down a sequence:
`[ Red Circle ] [ Blue Triangle ] [ Green Square ] ... [ Red Circle ]`

The dealer stops and asks: *"What card comes next?"*

Without knowing anything about geometry or colors, your brain instantly says: *"Blue Triangle!"* 
Why? Because you spotted that `Red Circle` appeared earlier in the sequence, and right after it came `Blue Triangle`. You didn't predict the card by memorizing an encyclopedia; you predicted it through **in-context pattern matching**.

How does a neural network do this mechanically?
It uses a two-step team of attention heads called an **Induction Head Circuit**:
1. **Player 1 (The Previous-Token Scout, Layer 0)**:
   Every time a card is placed, Player 1 whispers to it: *"Hey, write down on a sticky note the card that was right before you."*
   So when `Blue Triangle` arrives, its sticky note says: `Previous was Red Circle`.
2. **Player 2 (The Induction Head, Layer 1)**:
   When the dealer plays `Red Circle` a second time, Player 2 scans all the sticky notes on the table looking for one that says `Previous was Red Circle`.
   Player 2 spots `Blue Triangle`'s sticky note! Player 2 immediately points to `Blue Triangle` and shouts to the output: *"Next card is Blue Triangle!"*

This two-layer circuit is the fundamental mathematical engine powering in-context learning in modern Large Language Models (Elhage et al., 2021; Olsson et al., 2022).

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Entity | Domain | Operational Meaning & Interpretability Role |
| :--- | :--- | :--- | :--- |
| $\mathbf{h}_i \in \mathbb{R}^d$ | Residual Stream Vector | $d$-dim vector | Hidden activation at position $i$ before attention. |
| $W_Q, W_K$ | Query & Key Projection | $\mathbb{R}^{d_{\text{head}} \times d}$ | Determines the attention score between query position $i$ and key position $j$. |
| $W_V, W_O$ | Value & Output Projection | $W_V \in \mathbb{R}^{d_{\text{head}} \times d}, W_O \in \mathbb{R}^{d \times d_{\text{head}}}$ | Determines what representation is extracted from position $j$ and written to position $i$. |
| $W_{QK} = W_Q^\top W_K$ | The $QK$-Circuit Matrix | $\mathbb{R}^{d \times d}$ | Determines *where* an attention head moves information (bilinear form $x_i^\top W_{QK} x_j$). |
| $W_{OV} = W_O W_V$ | The $OV$-Circuit Matrix | $\mathbb{R}^{d \times d}$ | Determines *what* information an attention head writes into the residual stream. |
| $A \in \mathbb{R}^{T \times T}$ | Attention Pattern Matrix | Row-stochastic matrix | Softmax normalized attention probabilities: $A_{i, j} \ge 0, \sum_j A_{i, j} = 1$. |
| $W_E, W_U$ | Embedding & Unembedding | $W_E \in \mathbb{R}^{d \times V}, W_U \in \mathbb{R}^{V \times d}$ | Maps discrete tokens to residual stream and residual stream back to vocabulary logits. |
| $H_{0, \text{prev}}$ | Previous-Token Head | Layer 0 Attention Head | Attends to token $i-1$ and writes its token identity into position $i$. |
| $H_{1, \text{ind}}$ | Induction Head | Layer 1 Attention Head | Compares current token $i$ against previous-token features in keys to complete $[A][B] \dots [A] \to [B]$. |

---

## 3. The Transformer Circuits Mathematical Framework

In **A Mathematical Framework for Transformer Circuits** (Elhage et al., 2021), Anthropic researchers demonstrated that Multi-Head Attention can be decomposed into two independent, bilinear operations:

For a single attention head operating on residual activations $\mathbf{x}_i, \mathbf{x}_j \in \mathbb{R}^d$:

### 3.1 The $QK$-Circuit (Where to look)
The unnormalized attention logit between query position $i$ and key position $j$ is:

$$\alpha_{i, j} = \frac{(W_Q \mathbf{x}_i)^\top (W_K \mathbf{x}_j)}{\sqrt{d_{\text{head}}}} = \frac{\mathbf{x}_i^\top \left( W_Q^\top W_K \right) \mathbf{x}_j}{\sqrt{d_{\text{head}}}}$$

Defining the **$QK$-circuit matrix** $W_{QK} \triangleq W_Q^\top W_K \in \mathbb{R}^{d \times d}$:
$$\alpha_{i, j} = \frac{\mathbf{x}_i^\top W_{QK} \mathbf{x}_j}{\sqrt{d_{\text{head}}}}$$
$$A_{i, :} = \text{softmax}(\boldsymbol{\alpha}_{i, :})$$

The $QK$-circuit is a bilinear form on the residual stream that computes query-key affinity without depending on $W_V$ or $W_O$.

### 3.2 The $OV$-Circuit (What to copy)
Once attention weights $A_{i, j}$ are computed, the head extracts value vectors and projects them back into the residual stream:

$$\Delta \mathbf{x}_i = W_O \sum_{j} A_{i, j} W_V \mathbf{x}_j = \sum_j A_{i, j} \left( W_O W_V \right) \mathbf{x}_j$$

Defining the **$OV$-circuit matrix** $W_{OV} \triangleq W_O W_V \in \mathbb{R}^{d \times d}$:
$$\Delta \mathbf{x}_i = \sum_j A_{i, j} W_{OV} \mathbf{x}_j$$

The $OV$-circuit acts as a linear transformation mapping the representation at source position $j$ directly into destination position $i$.

---

## 4. Eigenvalue Spectrum of $W_E^\top W_U$ and $W_{OV}$

To understand how attention head outputs interact with vocabulary predictions, we analyze the end-to-end path from input tokens to output logits:

$$\text{Logits}_i = W_U \mathbf{x}_i = W_U \left( \sum_j A_{i, j} W_{OV} W_E \mathbf{t}_j \right)$$

Defining the end-to-end token-to-token map:
$$\mathbf{M} = W_U W_{OV} W_E \in \mathbb{R}^{V \times V}$$

### Direct Copy Heads vs Inversion Heads:
- **Positive Eigenvalues**: If $W_U W_{OV} W_E$ has large positive eigenvalues along the diagonal, the head is a **Direct Copy Head**: whenever token $T$ is attended to, the logit for token $T$ is boosted directly!
- **Negative Eigenvalues**: If the matrix has negative eigenvalues, the head acts as an **Inhibition Head** (e.g., suppressing duplicate tokens).

---

## 5. The Two-Layer Induction Head Mechanism

Why can't a single-layer Transformer implement general in-context repetition $[A][B] \dots [A] \to [B]$?

A 1-layer attention head can only attend based on tokens present at layer 0:
$$\alpha_{i, j} = (W_E \mathbf{t}_i)^\top W_{QK} (W_E \mathbf{t}_j)$$
If current token $\mathbf{t}_i = [A]$, the 1-layer head can search for $[A]$ (attending to position $j$ where $\mathbf{t}_j = [A]$). But if it attends to $[A]$, its $OV$-circuit can only copy $[A]$! It cannot attend to $[B]$ because $[B]$ does not have label $[A]$.

To predict $[B]$, the head must attend to the token that **followed** $[A]$! This requires composing **two layers**:

```
Layer 0: Previous-Token Head (H_{0})
Position j: [ B ] ─── Attends to j-1 [ A ] ───▶ Writes: "My previous was A" into residual stream

Layer 1: Induction Head (H_{1})
Position i: [ A ] (Second occurrence)
  Query at pos i:    "Search for token whose PREVIOUS was A"
  Key at pos j:      "My previous was A" (matches!)
  ──▶ Attention spikes: A_{i, j} ≈ 1.0 (Attends to B!)
  ──▶ OV-circuit copies [ B ] directly into output logits!
```

### Mathematical Proof of Composition:
Let $\mathbf{h}_0^{(j)} = W_E \mathbf{t}_j$.
1. **Layer 0 Output at position $j$**:
   $$\Delta \mathbf{h}_0^{(j)} = W_{OV}^{(0)} \mathbf{h}_0^{(j-1)} = W_{OV}^{(0)} W_E \mathbf{t}_{j-1}$$
   Residual stream entering Layer 1:
   $$\mathbf{h}_1^{(j)} = W_E \mathbf{t}_j + W_{OV}^{(0)} W_E \mathbf{t}_{j-1}$$
2. **Layer 1 $QK$-Circuit**:
   At position $i$ (where $\mathbf{t}_i = [A]$):
   $$\mathbf{q}_1^{(i)} = W_Q^{(1)} W_E \mathbf{t}_i = W_Q^{(1)} W_E [A]$$
   At past position $j$ (where $\mathbf{t}_j = [B]$ and $\mathbf{t}_{j-1} = [A]$):
   $$\mathbf{k}_1^{(j)} = W_K^{(1)} \mathbf{h}_1^{(j)} = W_K^{(1)} W_E [B] + W_K^{(1)} W_{OV}^{(0)} W_E [A]$$
3. **The Compositional Term**:
   The attention score contains the cross-term:
   $$\alpha_{i, j} \supset \left(W_Q^{(1)} W_E [A]\right)^\top \left(W_K^{(1)} W_{OV}^{(0)} W_E [A]\right) = [A]^\top \left( W_E^\top \left(W_Q^{(1)}\right)^\top W_K^{(1)} W_{OV}^{(0)} W_E \right) [A]$$
   When the composite matrix $\left(W_Q^{(1)}\right)^\top W_K^{(1)} W_{OV}^{(0)}$ is positive definite, $\alpha_{i, j}$ attains a massive positive spike whenever current token matches $\mathbf{t}_{j-1}$!
4. **The $OV$-Circuit**:
   $W_{OV}^{(1)}$ reads $W_E \mathbf{t}_j = [B]$ and projects it through $W_U$ to maximize $\text{logit}([B])$.

---

## 6. The Induction Bump & Phase Changes in Pretraining

In seminal empirical work across language models ranging from 10M to 50B parameters, **Olsson et al. (2022)** discovered the **Induction Bump**:
- During early training steps, the model has virtually zero in-context learning capability.
- At a specific critical step, thousands of induction heads form simultaneously across intermediate layers.
- Simultaneously with this formation, a sharp drop in training perplexity occurs, and few-shot in-context learning abruptly emerges!

Induction heads represent the macro-level transition from purely statistical n-gram memorization to active algorithmic in-context computation.
