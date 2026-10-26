# Topic 17.1: Byte-Pair Encoding (BPE) Tokenizer

---

## 1. Executive Summary & The 12-Year-Old Analogy

A Large Language Model (LLM) such as GPT-4 does not process words, sentences, or English text directly. Computers operate exclusively on arrays of numerical values. Before a single neural network parameter or attention weight can be applied, raw text must be converted into a discrete sequence of integer IDs:
$$\text{"Artificial intelligence is fascinating"} \;\longrightarrow\; [38241, 4434, 318, 14925]$$

This foundational translation layer is the **Tokenizer**. How text is partitioned into tokens dictates the vocabulary size ($V$), the sequence length ($T$), the computational cost of self-attention ($\mathcal{O}(T^2)$), and whether the model can handle rare words, multilingual text, code, or emojis without failing on "Out-Of-Vocabulary" (OOV) tokens.

```
                         THE TOKENIZATION TAXONOMY
                         
  1. Character-Level           2. Word-Level                  3. Subword (BPE / Byte-BPE)
  ------------------           -------------                  --------------------------
  ['c', 'a', 't', 's']         ['cats']                       ['cat', 's']
  - Vocab: ~256 chars          - Vocab: >500,000 words        - Vocab: 32,000 to 100,000
  - Sequence length: Massive   - Sequence length: Short       - Sequence length: Optimal
  - No OOV, but weak semantics - Massive OOV problem (<UNK>)  - Zero OOV, rich semantics
```

Modern GPT models (from GPT-2 up through GPT-4) adopt **Byte-level Byte-Pair Encoding (BPE)** (Sennrich et al., 2016; Radford et al., 2019):
1. The base vocabulary begins with the 256 individual byte values ($0 \dots 255$) of raw UTF-8 encodings.
2. The tokenizer scans a massive text corpus, counts frequencies of adjacent token pairs, and greedily merges the single most frequent pair into a brand new subword token.
3. This process repeats for $K$ iterations, building a deterministic hierarchy of merge rules.
4. Because every string in the universe decomposes into UTF-8 bytes, the model **never encounters an unknown token `<UNK>`**, achieving 100% universal text coverage.

---

### The 12-Year-Old Analogy: The Secret Agent Word-Smushing Code

Imagine you and your best friend are secret agents passing handwritten notes across the classroom. 

- **Method 1 (Letter by Letter / Character-level):**
  You write out every single letter: `T - H - E - - C - A - T - - S - A - T`.
  It takes forever to write, your wrist hurts, and the teacher catches you because your note is three feet long!

- **Method 2 (Whole Words / Word-level):**
  You make a codebook where every English word gets a number: `the` = 1, `cat` = 2, `sat` = 3.
  It makes messages short! But your codebook is 500 pages thick. The moment someone writes an unfamiliar word, slang, or a typo like `supercalifragilistic` or `catzz`, your codebook throws its hands in the air and stamps it: `??? [UNKNOWN]`.

- **Method 3 (The Word-Smushing Secret Code / BPE):**
  You start with just the basic alphabet (26 letters or 256 bytes).
  You notice you write `t` followed by `h` constantly. So you make a rule: *Rule 1: whenever you see `t` and `h` together, smush them into a new symbol `th`!*
  Next, you see `th` followed by `e` constantly. *Rule 2: smush `th` and `e` into `the`!*
  Next, you notice `i` followed by `n` and then `g`. You smush `i` + `n` $\to$ `in`, and `in` + `g` $\to$ `ing`!

Now:
- Common words like `the`, `apple`, and `running` become single compact stamps.
- Weird or new words like `unbreakable` get broken into pieces you already know: `un` + `break` + `able`.
- Even alien emojis or strange symbols can simply fall back to their raw letter pieces!
Nothing is ever unknown, and notes stay short and sweet.

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $\mathcal{D}$ | Text Corpus | The raw training text dataset used to train tokenization merges. |
| $\mathcal{V}_0$ | Initial Base Vocab | Base alphabet, e.g., 256 raw bytes $\{0, 1, \dots, 255\}$. |
| $\mathcal{V}$ | Vocabulary Set | Set of all recognized subword tokens, $|\mathcal{V}| = V$. |
| $V$ | Integer ($\mathbb{N}$) | Vocabulary size (e.g., $50{,}257$ for GPT-2, $100{,}000$ for LLaMA 3). |
| $\mathbf{x} = (t_1, t_2, \dots, t_T)$ | Sequence of Tokens | Input sequence represented as a tuple of discrete token IDs $t_i \in \{0, \dots, V-1\}$. |
| $(p_a, p_b)$ | Token Pair | Adjacent pair of tokens considered for merging. |
| $\text{freq}(p_a, p_b)$ | Frequency Metric | Count of consecutive occurrences of pair $(p_a, p_b)$ across the corpus. |
| $\mathcal{R}$ | Merge Table | Ordered list or hash table of merge rules $(p_a, p_b) \mapsto t_{\text{new}}$. |
| $T$ | Integer ($\mathbb{N}$) | Sequence length (context window length) in tokens. |
| $\text{encode}(s)$ | Mapping $\Sigma^* \to \mathbb{Z}^T$ | Function converting string $s$ to list of token IDs. |
| $\text{decode}(\mathbf{x})$ | Mapping $\mathbb{Z}^T \to \Sigma^*$ | Function reconstructing string $s$ from token IDs. |

---

## 3. Mathematical Foundations of Byte-Pair Encoding

### 3.1 The Trade-Off: Entropy vs Sequence Length

Let $S$ be a text document containing $L_{\text{char}}$ characters.
If we tokenize using character-level tokens ($V_{\text{char}} \approx 256$), the sequence length is $T \approx L_{\text{char}}$. Because Transformer self-attention computes an attention matrix of size $T \times T$, the compute cost scales quadratically:
$$\text{Attention Cost} = \mathcal{O}(T^2 \cdot d_{\text{model}}) = \mathcal{O}(L_{\text{char}}^2 \cdot d_{\text{model}})$$

If we tokenize using word-level tokens ($V_{\text{word}} \approx 500{,}000$), sequence length decreases ($T \approx L_{\text{char}} / 5$), but the token embedding table $W_e \in \mathbb{R}^{V \times d_{\text{model}}}$ and output projection head explode in parameter count:
$$\text{Embedding Parameters} = 2 \times V \times d_{\text{model}} \approx 2 \times 500{,}000 \times 4096 \approx 4.1 \times 10^9 \text{ parameters!}$$

Subword tokenization (BPE) finds the **information-theoretic sweet spot**:
$$V \in [32{,}000, 100{,}000], \qquad T \approx \frac{L_{\text{char}}}{3.5 \dots 4.5}$$
It strikes an optimal balance between sequence brevity and parameter efficiency.

---

### 3.2 The Sennrich et al. (2016) BPE Algorithm

Given a corpus $\mathcal{D}$ decomposed into sequences of initial symbols:
1. Initialize vocabulary $\mathcal{V} \leftarrow \mathcal{V}_0$ (where $|\mathcal{V}_0| = 256$ byte tokens).
2. Represent corpus as a list of symbol sequences:
   $$\mathcal{C} = [s_1, s_2, \dots, s_N]$$
3. Repeat for $k = 1, 2, \dots, K$ target merges:
   - Compute adjacent pair counts across entire corpus:
     $$\text{freq}(u, v) = \sum_{c \in \mathcal{C}} \sum_{j=1}^{|c|-1} \mathbb{I}[c_j = u \land c_{j+1} = v]$$
   - Identify the globally maximal pair:
     $$(u^*, v^*) = \arg\max_{(u, v)} \text{freq}(u, v)$$
   - Allocate a fresh token ID:
     $$t_{\text{new}} = |\mathcal{V}_0| + k - 1$$
   - Record merge rule:
     $$\mathcal{R} \leftarrow \mathcal{R} \cup \{((u^*, v^*) \mapsto t_{\text{new}})\}$$
   - Update vocabulary:
     $$\mathcal{V} \leftarrow \mathcal{V} \cup \{t_{\text{new}}\}, \qquad \text{bytes}(t_{\text{new}}) = \text{bytes}(u^*) \mathbin{\Vert} \text{bytes}(v^*)$$
   - Replace every consecutive occurrence of $(u^*, v^*)$ in $\mathcal{C}$ with $t_{\text{new}}$.

---

### 3.3 Byte-Level Fallback & Zero OOV Guarantee

In classical NLP tokenizers, if an incoming character is not in $\mathcal{V}$, it is mapped to a special out-of-vocabulary token `<UNK>`. This destroys information:
$$\text{"The patient took 50mg of Azithromycin"} \;\longrightarrow\; \text{"The patient took 50mg of <UNK>"}$$

Byte-level BPE solves this completely by treating text as a stream of raw bytes.
Every valid or invalid UTF-8 string is fundamentally an array of integers $b \in [0, 255]$:
- ASCII characters (`'A'`, `'b'`, `'9'`) occupy 1 byte.
- Accents and European alphabets (`'é'`, `'ñ'`, `'λ'`) occupy 2 bytes.
- Asian characters, emojis (`'中'`, `'🚀'`) occupy 3 to 4 bytes.

Because $\{0, 1, \dots, 255\} \subset \mathcal{V}$, **any byte sequence can always be represented**. Even if an emoji was never seen during BPE merge training, it simply decomposes into its 4 constituent byte IDs. When decoded, those bytes reassemble into the exact original emoji!

---

### 3.4 Inference Encoding Algorithm (Priority Queue / Lowest-Index Rule)

During inference, how does the tokenizer encode a novel sentence?
Given text $s$, convert to byte tokens $T = [b_1, b_2, \dots, b_m]$.
1. While $\text{length}(T) \ge 2$:
   - Find all adjacent pairs in $T$: $\{(T_i, T_{i+1})\}$.
   - Check which pairs exist in the learned merge dictionary $\mathcal{R}$.
   - If no pairs are in $\mathcal{R}$, terminate.
   - Greedily choose the pair with the **highest priority** (i.e., the pair learned earliest during training, which has the lowest new token ID $t_{\text{new}}$).
   - Merge all occurrences of that pair in $T$.
2. Return final token list $T$.

---

## 4. Architectural Implementation Blueprint

The following diagram illustrates how raw text enters the BPE pipeline and returns lossless text:

```
[ Raw UTF-8 String ]  e.g., "AI 🚀"
         │
         ▼
[ UTF-8 Byte Stream ] [65, 73, 32, 240, 159, 154, 128]
         │
         ▼  Iterative Merge Application (according to learned rules R)
   Pair (65, 73) ('A', 'I') -> 260 ('AI')
         │
         ▼
[ Token IDs ]        [260, 32, 240, 159, 154, 128]
         │
         ▼ (Model Forward Pass / Storage / Transmission)
         │
         ▼ Lookup in Vocab Table: Id -> Bytes
[ Byte Chunks ]      [b'AI', b' ', b'\xf0', b'\x9f', b'\x9a', b'\x80']
         │
         ▼ b"".join() + .decode('utf-8')
[ Lossless Output ]  "AI 🚀"
```

---

## 5. Summary & Key Takeaways

1. **Tokenization is the Bridge:** It bridges human textual symbols and neural tensor computations.
2. **Subword BPE is Standard:** Character-level sequences are too long; word-level vocabularies are too vast and fragile. BPE balances both.
3. **Byte-Level Base Guarantees Robustness:** Starting with 256 byte tokens ensures $0\%$ OOV rate.
4. **Lossless Invertibility:** A production-grade tokenizer must satisfy $\text{decode}(\text{encode}(s)) \equiv s$ for all strings in all languages.
