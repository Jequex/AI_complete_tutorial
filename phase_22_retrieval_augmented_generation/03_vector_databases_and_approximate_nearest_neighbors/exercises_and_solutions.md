# Topic 22.3: Exercises & Solutions — Vector Databases & Approximate Nearest Neighbors (ANN)

---

### Problem 1: Distance Metric Equivalence & Query FLOP Scaling (Easy)

Let $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$ be two unit-normalized vectors ($\|\mathbf{u}\|_2 = \|\mathbf{v}\|_2 = 1.0$).

1. Prove algebraically that minimizing Euclidean distance $\|\mathbf{u} - \mathbf{v}\|_2$ produces the exact same rank-ordering as minimizing Cosine distance $d_{\text{cos}}(\mathbf{u}, \mathbf{v}) = 1 - \mathbf{u} \cdot \mathbf{v}$.
2. Consider a database of $N = 10^7$ vectors with dimension $d = 1536$.
   - Calculate the number of floating-point operations (FLOPs) required to execute one query using an exact brute-force linear scan.
   - For an HNSW graph index where the query explores an average of $300$ candidate nodes across layers, compute the speedup factor in distance operations.

<details>
<summary>View Complete Solution</summary>

**Step 1: Algebraic Equivalence Proof:**
Expand the squared Euclidean distance:
$$\|\mathbf{u} - \mathbf{v}\|_2^2 = \sum_{i=1}^d (u_i - v_i)^2 = \sum_{i=1}^d u_i^2 + \sum_{i=1}^d v_i^2 - 2 \sum_{i=1}^d u_i v_i$$
Since $\|\mathbf{u}\|_2^2 = \sum u_i^2 = 1$ and $\|\mathbf{v}\|_2^2 = \sum v_i^2 = 1$, and $\mathbf{u} \cdot \mathbf{v} = \sum u_i v_i$:
$$\|\mathbf{u} - \mathbf{v}\|_2^2 = 1 + 1 - 2(\mathbf{u} \cdot \mathbf{v}) = 2(1 - \mathbf{u} \cdot \mathbf{v}) = 2 \cdot d_{\text{cos}}(\mathbf{u}, \mathbf{v})$$
Taking the positive square root (since distance is non-negative):
$$\|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{2 \cdot d_{\text{cos}}(\mathbf{u}, \mathbf{v})}$$
Because $f(x) = \sqrt{2x}$ is a strictly monotonically increasing function for $x \ge 0$:
$$d_{\text{cos}}(\mathbf{u}, \mathbf{v}_a) < d_{\text{cos}}(\mathbf{u}, \mathbf{v}_b) \iff \|\mathbf{u} - \mathbf{v}_a\|_2 < \|\mathbf{u} - \mathbf{v}_b\|_2$$
Thus, rank orderings across all candidate vectors are mathematically invariant under normalized Euclidean distance, Cosine distance, and Dot product.

**Step 2: FLOP Scaling & HNSW Speedup:**
- **Exact Linear Scan:**
  Computing a dot product between two $d$-dimensional vectors takes $d$ multiplications and $d-1$ additions $\approx 2d$ FLOPs.
  For $N = 10^7$ and $d = 1536$:
  $$\text{FLOPs} = N \times (2d) = 10^7 \times (2 \times 1536) = 3.072 \times 10^{10} \text{ FLOPs} \quad (30.72 \text{ GFLOPs})$$
- **HNSW Index:**
  Evaluating 300 candidates:
  $$\text{FLOPs}_{\text{HNSW}} = 300 \times (2 \times 1536) = 9.216 \times 10^5 \text{ FLOPs} \quad (0.92 \text{ MFLOPs})$$
  $$\text{Speedup Factor} = \frac{3.072 \times 10^{10}}{9.216 \times 10^5} \approx 33,333\times$$
  HNSW delivers an astounding **$33,000\times$ computational reduction**, transforming a multi-second bottleneck into a sub-millisecond query.
</details>

---

### Problem 2: HNSW Hierarchy & Level Distribution Derivation (Medium)

In an HNSW graph with connection parameter $M = 16$, the maximum layer $l$ of a newly inserted node is drawn using:
$$l = \lfloor -\ln(U) \cdot m_L \rfloor, \quad \text{where } U \sim \text{Uniform}(0, 1] \text{ and } m_L = \frac{1}{\ln(M)}$$

1. Derive the cumulative distribution function $P(l \ge k)$ for any integer level $k \ge 0$.
2. In an index with $N = 10^6$ vectors, calculate the expected number of nodes present at Level 0, Level 1, Level 2, and Level 3.
3. Determine the expected maximum layer height $L_{\max}$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Derivation of $P(l \ge k)$:**
$$l \ge k \iff \lfloor -\ln(U) \cdot m_L \rfloor \ge k \iff -\ln(U) \cdot m_L \ge k$$
$$\iff -\ln(U) \ge \frac{k}{m_L} = k \ln(M) = \ln(M^k)$$
$$\iff \ln(U) \le -\ln(M^k) = \ln(M^{-k})$$
Exponentiating both sides (since $\exp$ is monotonic):
$$U \le M^{-k}$$
Since $U \sim \text{Uniform}(0, 1]$, the probability that $U \le x$ is simply $x$:
$$P(l \ge k) = M^{-k} = \left(\frac{1}{M}\right)^k$$
This proves that HNSW node assignment precisely follows a geometric distribution with ratio $\frac{1}{M}$, exactly mirroring an ideal skip-list!

**Step 2: Expected Nodes per Level for $N = 10^6, M = 16$:**
- **Level 0 ($k = 0$):** $P(l \ge 0) = 16^0 = 1 \implies \mathbb{E}[N_0] = 10^6 = 1,000,000$ nodes (100%).
- **Level 1 ($k = 1$):** $P(l \ge 1) = \frac{1}{16} \implies \mathbb{E}[N_1] = \frac{10^6}{16} = 62,500$ nodes.
- **Level 2 ($k = 2$):** $P(l \ge 2) = \frac{1}{256} \implies \mathbb{E}[N_2] = \frac{10^6}{256} \approx 3,906$ nodes.
- **Level 3 ($k = 3$):** $P(l \ge 3) = \frac{1}{4096} \implies \mathbb{E}[N_3] = \frac{10^6}{4096} \approx 244$ nodes.
- **Level 4 ($k = 4$):** $\mathbb{E}[N_4] = \frac{10^6}{65536} \approx 15$ nodes.

**Step 3: Expected Maximum Level $L_{\max}$:**
Setting the expected count to 1 node:
$$N \cdot M^{-L_{\max}} \approx 1 \implies M^{L_{\max}} \approx N \implies L_{\max} \approx \frac{\ln(N)}{\ln(M)} = \log_M(N)$$
For $N = 10^6, M = 16$:
$$L_{\max} \approx \frac{\ln(10^6)}{\ln(16)} = \frac{13.8155}{2.7726} \approx 4.98 \approx 5 \text{ levels}$$
The skip-list graph remains exceptionally shallow ($\sim 5$ levels) even over a million vectors.
</details>

---

### Problem 3: Product Quantization (PQ) Compression & Asymmetric Distance (Hard)

An enterprise vector store indexes $N = 5 \times 10^7$ vectors of dimension $d = 1024$ stored as 32-bit floats (`float32`).
To fit the index in memory, the team applies **Product Quantization (PQ)**:
- The vector is sliced into $m = 8$ equal sub-vectors of dimension $d^* = \frac{1024}{8} = 128$.
- For each sub-vector space, $k^* = 256$ centroids are learned via k-means.
- Each sub-vector is replaced by the 8-bit index ($1$ byte) of its nearest centroid code.

1. Calculate the raw uncompressed memory consumption of the base vectors in gigabytes.
2. Calculate the compressed memory consumption of the PQ-encoded database. What is the compression ratio?
3. In **Asymmetric Distance Computation (ADC)**, the query vector $\mathbf{q}$ is left unquantized, while database vectors are quantized. Explain how a precomputed lookup table of size $m \times k^*$ enables computing distances without decompressing the vectors.

<details>
<summary>View Complete Solution</summary>

**Step 1: Uncompressed Memory Consumption:**
Each `float32` takes 4 bytes.
$$\text{Bytes per vector} = 1024 \times 4 \text{ bytes} = 4096 \text{ bytes} = 4 \text{ KB}$$
For $N = 5 \times 10^7$ vectors:
$$\text{Total Memory} = 5 \times 10^7 \times 4096 \text{ bytes} = 2.048 \times 10^{11} \text{ bytes} \approx 204.8 \text{ GB}$$
This exceeds the RAM capacity of standard single-node cloud instances.

**Step 2: PQ Compressed Memory Consumption:**
Each vector is represented by $m = 8$ centroid indices.
Since $k^* = 256 = 2^8$, each index fits in $1$ byte:
$$\text{Compressed size per vector} = 8 \text{ bytes}$$
For $N = 5 \times 10^7$ vectors:
$$\text{Total PQ Memory} = 5 \times 10^7 \times 8 \text{ bytes} = 4 \times 10^8 \text{ bytes} = 400 \text{ MB}$$
The codebook overhead: $m = 8$ codebooks $\times 256$ centroids $\times 128$ dimensions $\times 4$ bytes $\approx 1 \text{ MB}$ (negligible).
$$\text{Compression Factor} = \frac{204.8 \text{ GB}}{0.401 \text{ GB}} \approx 512\times$$
The entire 50-million vector corpus is compressed from **$205 \text{ GB}$ down to just $400 \text{ MB}$**, easily fitting in an inexpensive laptop or mobile device RAM!

**Step 3: Asymmetric Distance Computation (ADC) Lookup Mechanics:**
Let query $\mathbf{q} = (\mathbf{q}_1, \mathbf{q}_2, \dots, \mathbf{q}_m)$ and let quantized vector $\mathbf{v} \approx (\mathbf{c}_{1, i_1}, \dots, \mathbf{c}_{m, i_m})$.
The squared Euclidean distance decomposes as:
$$\|\mathbf{q} - \mathbf{v}\|_2^2 \approx \sum_{j=1}^m \|\mathbf{q}_j - \mathbf{c}_{j, i_j}\|_2^2$$
Before scanning the database:
1. The system computes a distance matrix $\mathbf{D}_{\text{table}} \in \mathbb{R}^{m \times k^*}$, where:
   $$\mathbf{D}_{\text{table}}[j, k] = \|\mathbf{q}_j - \mathbf{c}_{j, k}\|_2^2$$
   This requires only $m \times k^* = 8 \times 256 = 2048$ small sub-vector distance calculations.
2. For each stored vector encoded as 8 bytes $(i_1, i_2, \dots, i_8)$, the distance is computed via $8$ array lookups and $7$ additions:
   $$d(\mathbf{q}, \mathbf{v}) = \sum_{j=1}^8 \mathbf{D}_{\text{table}}[j, i_j]$$
No decompression is ever required! Scanning vectors is transformed into blazing-fast memory lookups executed at cache-line speeds.
</details>
