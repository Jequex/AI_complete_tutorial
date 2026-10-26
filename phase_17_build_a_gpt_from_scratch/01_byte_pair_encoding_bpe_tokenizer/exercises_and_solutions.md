# Topic 17.1: Exercises & Solutions — Byte-Pair Encoding (BPE) Tokenizer

---

### Problem 1: Manual Step-by-Step BPE Merge Tracing (Easy)
Consider the following toy text corpus composed of individual character symbols with a special end-of-word marker `_`:
$$\mathcal{C} = \text{"low_ low_ lowest_ newer_ wider_"}$$

The character frequencies of the words in the corpus are:
- `l o w _` : 2 times
- `l o w e s t _` : 1 time
- `n e w e r _` : 1 time
- `w i d e r _` : 1 time

1. What is the initial base vocabulary $\mathcal{V}_0$?
2. Compute the frequencies of all adjacent character pairs across the corpus.
3. Identify the very first pair selected for merging.
4. Update the word representations and state the vocabulary after this merge.

<details>
<summary>View Complete Solution</summary>

**Step 1: Initial Vocabulary:**
The distinct characters present in the corpus are:
$$\mathcal{V}_0 = \{ \text{'d', 'e', 'i', 'l', 'o', 'r', 's', 't', 'w', '\_'} \}$$
Size $|\mathcal{V}_0| = 10$.

**Step 2: Adjacent pair frequencies:**
Let us count each adjacent pair across all word occurrences:
- `(l, o)`: occurs in `low_` (2x) and `lowest_` (1x) $\to 3$
- `(o, w)`: occurs in `low_` (2x) and `lowest_` (1x) $\to 3$
- `(w, _)`: occurs in `low_` (2x) $\to 2$
- `(w, e)`: occurs in `lowest_` (1x) and `newer_` (1x) $\to 2$
- `(e, s)`: occurs in `lowest_` (1x) $\to 1$
- `(s, t)`: occurs in `lowest_` (1x) $\to 1$
- `(t, _)`: occurs in `lowest_` (1x) $\to 1$
- `(n, e)`: occurs in `newer_` (1x) $\to 1$
- `(e, r)`: occurs in `newer_` (1x) and `wider_` (1x) $\to 2$
- `(r, _)`: occurs in `newer_` (1x) and `wider_` (1x) $\to 2$
- `(w, i)`: occurs in `wider_` (1x) $\to 1$
- `(i, d)`: occurs in `wider_` (1x) $\to 1$
- `(d, e)`: occurs in `wider_` (1x) $\to 1$

**Step 3: First pair selection:**
The maximal frequency is 3, shared by `(l, o)` and `(o, w)`. By alphabetical tie-breaking, we select `(l, o)`.

**Step 4: Update:**
- New token: `lo`
- Rule: `(l, o) -> lo`
- Updated vocabulary: $\mathcal{V}_1 = \mathcal{V}_0 \cup \{\text{'lo'}\}$
- Updated corpus:
  - `lo w _` : 2
  - `lo w e s t _` : 1
  - `n e w e r _` : 1
  - `w i d e r _` : 1
</details>

---

### Problem 2: Parameter Savings vs Quadratic Sequence Length Trade-off (Medium)
Suppose an LLM has embedding dimension $d_{\text{model}} = 4096$, and processes documents of average length $L = 2048$ characters.
Compare two architectural configurations:
- **System A (Character-level):** Vocabulary $V_A = 256$, average token count per document $T_A = 2048$.
- **System B (Subword BPE):** Vocabulary $V_B = 50{,}000$, average compression ratio $3.8$ chars per token, yielding $T_B = \lceil 2048 / 3.8 \rceil = 540$ tokens.

1. Compute the number of parameters in the token embedding table $W_e \in \mathbb{R}^{V \times d_{\text{model}}}$ for both systems.
2. In a single multi-head self-attention layer, computing the attention matrix $A = Q K^T$ requires $2 \times T^2 \times d_{\text{model}}$ floating-point operations (FLOPs). Compute and compare the FLOPs required for self-attention per document under System A vs System B.
3. Why does modern deep learning favor System B even though its embedding table is larger?

<details>
<summary>View Complete Solution</summary>

**Step 1: Embedding Table Parameters:**
$$\text{Params}(W_e) = V \times d_{\text{model}}$$
- System A: $256 \times 4096 = 1{,}048{,}576 \text{ parameters} \approx 1.05\text{ M params}$.
- System B: $50{,}000 \times 4096 = 204{,}800{,}000 \text{ parameters} \approx 204.8\text{ M params}$.

**Step 2: Self-Attention FLOPs ($2 \cdot T^2 \cdot d_{\text{model}}$):**
- System A ($T_A = 2048$):
  $$\text{FLOPs}_A = 2 \times (2048)^2 \times 4096 = 2 \times 4{,}194{,}304 \times 4096 = 34{,}359{,}738{,}368 \approx 34.36\text{ GFLOPs}$$
- System B ($T_B = 540$):
  $$\text{FLOPs}_B = 2 \times (540)^2 \times 4096 = 2 \times 291{,}600 \times 4096 = 2{,}388{,}787{,}200 \approx 2.39\text{ GFLOPs}$$

The speedup ratio in self-attention is:
$$\frac{\text{FLOPs}_A}{\text{FLOPs}_B} = \frac{34.36}{2.39} \approx 14.38\times$$

**Step 3: Architectural Insight:**
In modern 32-layer or 80-layer Transformers, self-attention and MLP blocks are evaluated at every single layer on every forward pass. Saving a factor of $14\times$ in compute across dozens of layers drastically outweighs the one-time memory cost of storing a 200M parameter lookup table. Moreover, subword tokens pack semantic concepts (words, roots, prefixes) into individual vectors, making gradient credit assignment much easier over long horizons.
</details>

---

### Problem 3: Handling Emoji & Foreign Scripts via Byte Fallback (Hard)
The UTF-8 encoding of the rocket emoji `🚀` (Unicode code point `U+1F680`) is represented by 4 hexadecimal bytes:
`0xF0 0x9F 0x9A 0x80`, which are decimal values `[240, 159, 154, 128]`.

1. If a BPE tokenizer was trained solely on pure English Wikipedia and has never observed `🚀`, how does a standard word-level tokenizer handle this input versus a byte-level BPE tokenizer?
2. Write down the sequence of token IDs produced by the byte-level BPE tokenizer for the string `"Go! 🚀"`, assuming standard ASCII byte IDs and no learned merges for rocket bytes.
3. Prove that byte-level decoding guarantees lossless reconstruction of the original string.

<details>
<summary>View Complete Solution</summary>

**Step 1: Word-level vs Byte-BPE:**
- A word-level tokenizer looks up `"🚀"` in its vocabulary, fails to find it, and replaces it with `<UNK>`. The original emoji cannot be reconstructed upon decoding.
- A byte-level BPE tokenizer starts with all 256 byte values ($0 \dots 255$) in its base vocabulary. Since none of the rocket bytes are merged, it emits the 4 individual byte tokens $[240, 159, 154, 128]$.

**Step 2: Token IDs for `"Go! 🚀"`:**
- `'G'` = ASCII 71
- `'o'` = ASCII 111
- `'!'` = ASCII 33
- `' '` = ASCII 32
- `'🚀'` = $[240, 159, 154, 128]$
Assuming no merges applied to these characters, the token list is:
$$[71, 111, 33, 32, 240, 159, 154, 128]$$

**Step 3: Lossless Invertibility:**
When decoding:
1. Each token ID is mapped back to its byte chunk:
   $$71 \mapsto \texttt{b'G'}, \quad 111 \mapsto \texttt{b'o'}, \quad 33 \mapsto \texttt{b'!'}, \quad 32 \mapsto \texttt{b' '}$$
   $$240 \mapsto \texttt{b'\textbackslash xf0'}, \quad 159 \mapsto \texttt{b'\textbackslash x9f'}, \quad 154 \mapsto \texttt{b'\textbackslash x9a'}, \quad 128 \mapsto \texttt{b'\textbackslash x80'}$$
2. The byte chunks are concatenated in order:
   $$\texttt{b'Go! \textbackslash xf0\textbackslash x9f\textbackslash x9a\textbackslash x80'}$$
3. Applying standard UTF-8 string decoding `.decode('utf-8')` validly decodes the 4-byte sequence back into `🚀`. The reconstructed string is `"Go! 🚀"`, identically matching the input.
</details>
