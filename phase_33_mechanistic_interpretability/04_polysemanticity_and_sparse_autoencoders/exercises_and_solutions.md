# Topic 04: Exercises & Solutions — Polysemanticity & Sparse Autoencoders

## Exercise 1: High-Dimensional Almost-Orthogonal Packing (Johnson-Lindenstrauss)

1. In Euclidean space $\mathbb{R}^d$, what is the maximum number of strictly mutually orthogonal unit vectors that can exist?
2. Suppose we relax strict orthogonality: we allow any pair of unit vectors $\mathbf{u}_i, \mathbf{u}_j$ to have bounded inner product $|\langle \mathbf{u}_i, \mathbf{u}_j \rangle| \le \epsilon$ for some tolerance $\epsilon \in (0, 1)$. Using Kabatjanskii-Levenshtein / Johnson-Lindenstrauss spherical packing bounds, express the scaling of the maximum number of almost-orthogonal vectors $M(d, \epsilon)$ as a function of dimension $d$.
3. For $d = 4096$ and $\epsilon = 0.1$, explain why a Transformer residual stream can support orders of magnitude more semantic features than its physical dimension ($D \gg d$).

### Solution

#### 1. Strict Orthogonality Limit
In $\mathbb{R}^d$, any set of mutually orthogonal non-zero vectors is linearly independent. Therefore, the maximum number of strictly mutually orthogonal unit vectors is equal to the dimension of the space:
$$M_{\text{orth}} = d$$

#### 2. Almost-Orthogonal Packing Scaling
When inner product interference is bounded by $|\mathbf{u}_i^\top \mathbf{u}_j| \le \epsilon$, the problem is equivalent to spherical code packing on the unit sphere $\mathbb{S}^{d-1}$. By standard random coding arguments and the Johnson-Lindenstrauss lemma:
$$M(d, \epsilon) \ge \exp\left( c \cdot \epsilon^2 d \right)$$
for some constant $c > 0$.
The number of packable directions grows **exponentially** with the dimension $d$, rather than linearly!

#### 3. Implications for LLM Residual Streams
For $d = 4096$ and $\epsilon = 0.1$:
The exponent $\epsilon^2 d = (0.01)(4096) \approx 40.96$.
$e^{40.96} \approx 6 \times 10^{17}$ potential almost-orthogonal directions!
Even under conservative packing bounds, the residual stream can easily accommodate millions of sparse semantic concepts ($D \ge 100,000 \gg 4096$) with average pairwise cross-talk noise bounded below $0.10$. Because natural language features are sparse (only a handful active per token), the cumulative interference noise remains small and easily filtered by ReLU/threshold activations.

---

## Exercise 2: Scale-Invariance Degeneracy of the $L_1$ Penalty

Consider the unconstrained Sparse Autoencoder loss function:
$$\mathcal{L}(W_{\text{enc}}, W_{\text{dec}}, \mathbf{b}_{\text{enc}}) = \|\mathbf{x} - W_{\text{dec}} \mathbf{z}\|_2^2 + \lambda \sum_{j=1}^D |z_j|$$
where $\mathbf{z} = \text{ReLU}(W_{\text{enc}} \mathbf{x} + \mathbf{b}_{\text{enc}})$.

1. Suppose $(W_{\text{enc}}^*, W_{\text{dec}}^*, \mathbf{b}_{\text{enc}}^*)$ is an optimal parameter configuration with latent code $\mathbf{z}^*$. Consider a positive scalar rescaling $\alpha > 1$. Define transformed parameters:
   $$\widetilde{W}_{\text{enc}} = \frac{1}{\alpha} W_{\text{enc}}^*, \quad \widetilde{\mathbf{b}}_{\text{enc}} = \frac{1}{\alpha} \mathbf{b}_{\text{enc}}^*, \quad \widetilde{W}_{\text{dec}} = \alpha W_{\text{dec}}^*$$
   Show that the reconstruction error $\|\mathbf{x} - \widetilde{W}_{\text{dec}} \widetilde{\mathbf{z}}\|_2^2$ is completely unchanged.
2. What happens to the $L_1$ regularization penalty $\lambda \sum_j |\tilde{z}_j|$ as $\alpha \to \infty$?
3. How does constraining each decoder column $\|\mathbf{w}_j^{\text{dec}}\|_2 = 1$ definitively eliminate this degeneracy?

### Solution

#### 1. Invariance of Reconstruction
The scaled latent activations are:
$$\widetilde{\mathbf{z}} = \text{ReLU}\left( \widetilde{W}_{\text{enc}} \mathbf{x} + \widetilde{\mathbf{b}}_{\text{enc}} \right) = \text{ReLU}\left( \frac{1}{\alpha} (W_{\text{enc}}^* \mathbf{x} + \mathbf{b}_{\text{enc}}^*) \right) = \frac{1}{\alpha} \text{ReLU}\left( W_{\text{enc}}^* \mathbf{x} + \mathbf{b}_{\text{enc}}^* \right) = \frac{1}{\alpha} \mathbf{z}^*$$
where we used the non-negative homogeneity of the ReLU function $\text{ReLU}(c \cdot v) = c \cdot \text{ReLU}(v)$ for $c = \frac{1}{\alpha} > 0$.
The reconstructed vector is:
$$\hat{\mathbf{x}}_{\text{new}} = \widetilde{W}_{\text{dec}} \widetilde{\mathbf{z}} = (\alpha W_{\text{dec}}^*) \left( \frac{1}{\alpha} \mathbf{z}^* \right) = W_{\text{dec}}^* \mathbf{z}^* = \hat{\mathbf{x}}_{\text{orig}}$$
Thus, the reconstruction error $\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2$ is identically invariant for any $\alpha > 0$.

#### 2. Vanishing $L_1$ Penalty
The regularizer evaluates to:
$$\lambda \sum_{j=1}^D |\tilde{z}_j| = \lambda \sum_{j=1}^D \left| \frac{1}{\alpha} z_j^* \right| = \frac{1}{\alpha} \left( \lambda \sum_{j=1}^D |z_j^*| \right)$$
As $\alpha \to \infty$, the $L_1$ penalty approaches **zero**:
$$\lim_{\alpha \to \infty} \frac{1}{\alpha} \lambda \|\mathbf{z}^*\|_1 = 0$$
An unconstrained optimizer will simply drive $\|W_{\text{dec}}\| \to \infty$ and $\|\mathbf{z}\| \to 0$, trivially bypassing the sparsity objective without learning sparse features!

#### 3. Resolution via Column Normalization
By fixing $\|\mathbf{w}_j^{\text{dec}}\|_2 = 1$ for all dictionary columns $j \in \{1, \dots, D\}$, scaling the decoder matrix $W_{\text{dec}}$ by $\alpha \neq 1$ is explicitly forbidden. The scale of $\mathbf{z}$ is pinned to the physical scale of the input $\mathbf{x}$, ensuring that the $L_1$ penalty genuinely penalizes the number and magnitude of active features.

---

## Exercise 3: Dead Latents and Shrinkage Bias in Sparse Autoencoders

1. In training large Sparse Autoencoders (e.g., $D = 131,072$), a common failure mode is **dead latents** (neurons that never activate on any token in millions of forward passes). Explain how an excessively high $\lambda$ causes dead latents.
2. When a latent feature $j$ is active ($z_j > 0$), the $L_1$ penalty adds a constant derivative $\lambda$ to the backpropagated gradient: $\frac{\partial \mathcal{L}}{\partial z_j} = \dots + \lambda$. Explain how this causes **shrinkage bias** (underestimating true feature magnitudes).
3. State two modern architectural innovations that eliminate shrinkage bias (e.g., TopK SAEs, Gated SAEs).

### Solution

#### 1. Dead Latent Genesis
When $\lambda$ is too high, the cost of activating a latent exceeds the reconstruction error reduction it provides. If an encoder bias $\mathbf{b}_j^{\text{enc}}$ drifts negative during early training, $\mathbf{w}_j^\top \mathbf{x} + b_j \le 0$ for all samples in the batch. Because $\text{ReLU}'(u) = 0$ for $u \le 0$, the gradient $\frac{\partial \mathcal{L}}{\partial W_j^{\text{enc}}}$ becomes identically zero, permanently trapping the latent in an inactive, "dead" state.

#### 2. Shrinkage Bias
For any active feature $z_j > 0$, the $L_1$ subgradient is strictly positive ($+\lambda$). At the optimal balance between reconstruction and regularization:
$$\hat{z}_j \approx z_j^{\text{true}} - \frac{\lambda}{2}$$
The SAE systematically suppresses the magnitude of active features by a constant offset $\approx \lambda / 2$. This shrinkage distorts the downstream reconstruction of large-amplitude features.

#### 3. Modern Solutions
1. **TopK SAEs (Gao et al., 2024)**: Replace the $L_1$ penalty entirely with a hard TopK activation function $\text{TopK}(\mathbf{z})$, which retains only the $k$ largest pre-activations and zeroes the rest. This provides exact sparsity without shrinkage.
2. **Gated SAEs (Rajamanoharan et al., 2024)**: Decouple the gating decision from the magnitude prediction using separate gate ($z_{\text{gate}} = \text{Heaviside}(\dots)$) and magnitude branches, eliminating the linear gradient drag on active latents.
