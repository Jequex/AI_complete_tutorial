# Topic 18.2: Data Curation, Deduplication, and Filtering

---

## 🧭 Executive Overview

In the pretraining of state-of-the-art foundation models, the old adage *"garbage in, garbage out"* dominates all other factors. While modern architectures and compute-optimal scaling laws determine the *efficiency* of parameter learning, the **composition, diversity, cleanliness, and uniqueness of the training corpus dictate the intelligence ceiling of the resulting model**.

Raw web scrapes from repositories like **Common Crawl** contain tens of petabytes of text, but over $90\%$ of this raw data consists of machine-generated spam, SEO keyword stuffing, tracking scripts, duplicate boilerplate (cookie policies, navigation footers), and toxic or low-entropy gibberish. Training directly on unfiltered web data causes models to develop severe degeneration, catastrophic repetitive loops, high memorization rates, and poor downstream reasoning.

Modern data curation pipelines transform raw, noisy web dumps into gold-standard pretraining corpora (such as RefinedWeb, Dolma, FineWeb, and RedPajama) through a four-stage funnel:
1. **Document Extraction & Language Identification:** Parsing HTML into clean prose and isolating target natural languages using FastText classifiers.
2. **Heuristic Quality Filtering:** Discarding documents that violate linguistic structural priors (stopword density, symbol ratios, line lengths, KenLM perplexity).
3. **Fuzzy Near-Deduplication (MinHash + LSH):** Eliminating duplicated web pages, mirrored domains, and boilerplate using Locality-Sensitive Hashing on Jaccard similarities.
4. **Decontamination & Synthetic Data Augmentation:** Purging benchmark questions (MMLU, GSM8k) to prevent test-set leakage, and augmenting with textbook-quality synthetic reasoning data (Cosmopedia, UltraChat).

---

## 🌊 12-Year-Old Analogy: Panning for Gold in a Muddy River

> **Imagine you are standing beside a massive, raging, muddy river.**
>
> You are building the world's most intelligent robot, but it can only learn by eating pure, shining gold nuggets. If you scoop up a bucket of river water and dump the mud, rocks, twigs, plastic bottles, and dead fish straight into your robot, its mechanical gears will instantly grind to a screeching halt.
>
> **The Data Curation Funnel:**
> 1. **The Coarse Sieve (Document Extraction & Language ID):** First, you pour the river water through a wide wire mesh that catches tree branches, soda cans, and boots. You only keep the gravel that speaks your language.
> 2. **The Water Flume (Heuristic Quality Filters):** Next, you let the gravel tumble down a wooden flume. Lightweight mud, oily scum, and repetitive silt wash away down the sides, leaving behind only dense, readable stones.
> 3. **The Magnet Grid (MinHash & LSH Near-Deduplication):** You notice there are 5,000 identical plastic bottle caps with the exact same printed brand logo. Your robot doesn't need to read that logo 5,000 times! You sweep a magnetic sensor that groups similar items together and discards 4,999 identical copies.
> 4. **The Security Inspection (Decontamination):** Before feeding the gold to the robot, an inspector ensures no secret exam cheat-sheets fell into the bucket, so the robot passes its final school test fairly.
>
> What remains at the bottom of your pan is pure, glittering, 24-karat gold dust!

---

## 📖 Mathematical Symbol Decoder

| Symbol | Mathematical Definition | Concrete Physical Meaning | Typical Values / Units |
| :--- | :--- | :--- | :--- |
| $D$ | Corpus document | An individual text sample (article, paper, code file) | Strings of $10^2 - 10^5$ words |
| $S_D$ | $\{s_1, s_2, \dots, s_k\}$ | Set of $k$-shingles (word $k$-grams or character $n$-grams) | $k = 3$ words or $k = 5$ chars |
| $J(A, B)$ | $\frac{\|S_A \cap S_B\|}{\|S_A \cup S_B\|}$ | Jaccard Similarity between two documents' shingle sets | Real number $\in [0.0, 1.0]$ |
| $\pi$ | Random permutation | Permutation of the shingle hash space | $\pi: \mathbb{Z} \to \mathbb{Z}$ |
| $h_i(x)$ | $(a_i x + b_i) \bmod p$ | Universal hash function for MinHash signature construction | $p = 2^{31} - 1$ (Mersenne prime) |
| $M$ | Dimension of signature vector | Number of independent MinHash permutations | $64, 128, \text{ or } 256$ |
| $b$ | Number of LSH bands | Number of partitions of the MinHash signature | $10 - 32$ bands |
| $r$ | Rows per band | Number of signature rows per LSH band ($M = b \cdot r$) | $4 - 16$ rows |
| $s^*$ | $(1/b)^{1/r}$ | Theoretical 50% collision probability similarity threshold | Typically $0.6 - 0.8$ |
| $P_{\text{collision}}(s)$ | $1 - (1 - s^r)^b$ | Probability that two documents with similarity $s$ collide in LSH | S-curve probability $\in [0, 1]$ |

---

## 🔬 Deep-Dive: The Pretraining Data Funnel

```text
[ Raw Web Scrape: Common Crawl (Petabytes) ]
                     │
                     ▼
  [ Stage 1: Document Extraction & Language ID ]  --> Drop non-target languages & raw HTML markup
                     │
                     ▼
  [ Stage 2: Heuristic & Perplexity Filtering ]  --> Drop low-entropy spam, robotic loops, gibberish
                     │
                     ▼
  [ Stage 3: MinHash + LSH Near-Deduplication ]  --> Drop 80%+ identical mirrors & duplicated templates
                     │
                     ▼
  [ Stage 4: Decontamination & Synthetic Mix ]   --> Remove eval benchmarks & add high-reasoning text
                     │
                     ▼
[ Clean Gold Pretraining Corpus (FineWeb, Dolma) ]
```

---

### 1. Document Extraction & Language Identification
Raw Common Crawl data arrives as WARC (Web ARChive) files containing raw HTTP response headers and unparsed HTML. 
- **Boilerplate Extraction:** Libraries such as `trafilatura` and `resiliparse` strip out `<nav>`, `<aside>`, `<script>`, `<style>`, and advertising `<iframe>` tags, isolating the primary editorial text of the page.
- **Language Identification:** A FastText linear classifier trained on character n-grams produces a probability distribution over 176+ languages:
  $$P(\text{lang} \mid D) = \text{softmax}(W \cdot \bar{e}(D))$$
  Documents with $P(\text{en}) < 0.65$ (or target language threshold) are discarded or routed to multilingual pretraining buckets.

---

### 2. Heuristic Quality Filtering (The Gopher & C4 Rules)
Natural, human-authored text displays consistent statistical regularities across syntax and vocabulary. Automated crawlers and SEO spam violate these regularities:
- **Stopword Distribution:** In English, standard function words (*the, of, and, in, a, is*) consistently constitute at least $15\%$ to $30\%$ of any coherent paragraph. Machine keyword lists ("cheap flights hotel car rental book now") typically have stopword frequencies below $5\%$.
- **Symbol-to-Word Ratio:** Documents where symbols (`#`, `*`, `_`, `{`, `}`, `|`) exceed $15\%$ of total characters are typically unprocessed code dumps, binary data, or corrupted markup.
- **Line Length and Ellipses:** Web scrapers frequently catch truncated link directories where $> 30\%$ of lines end with ellipses (`...`) or average line lengths are under 3 words.
- **Perplexity Filtering with KenLM:** A lightweight 5-gram language model is trained on a clean anchor corpus (e.g., Wikipedia). Any web document with excessively high perplexity (unintelligible text) or excessively low perplexity (monotonously repeated words or robotic tables) is rejected:
  $$\text{Perplexity}(D) = \exp\left( -\frac{1}{|D|} \sum_{i=1}^{|D|} \log P(w_i \mid w_{i-4}, \dots, w_{i-1}) \right)$$

---

### 3. Deduplication: Exact Hashing vs MinHash + LSH

#### Why Deduplication is Crucial
Lee et al. (2022) demonstrated in *"Deduplicating Training Data Makes Language Models Better"* that:
1. Deduplication **reduces verbatim memorization by over an order of magnitude**, radically reducing privacy leakage.
2. Identical repeated text causes models to suffer from **repetition loops** at generation time (emitting the same sentence endlessly).
3. Eliminating duplicates saves **up to 30% of pretraining compute** without any drop in downstream accuracy.

#### The Failure of Exact Hashing (SHA-256)
If document $B$ is an exact copy of document $A$ except that a tracking footer *"Published on 2024-10-26"* is appended:
$$\text{SHA256}(A) = \text{e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855}$$
$$\text{SHA256}(B) = \text{a7c1b5298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b899}$$
Because cryptographic hashes exhibit the **avalanche effect** (a 1-bit change flips roughly $50\%$ of hash bits), exact hashing cannot detect that document $B$ is $99.5\%$ identical to $A$.

---

#### The MinHash Algorithm (Broder, 1997)
MinHash solves this by mapping arbitrary sets into fixed-size signature vectors such that the probability of signature collision **equals the exact Jaccard similarity**:
$$J(A, B) = \frac{|S_A \cap S_B|}{|S_A \cup S_B|}$$

**The MinHash Theorem:**
Let $\pi$ be a random permutation of the universal shingle vocabulary. For any two sets of shingles $S_A$ and $S_B$:
$$P\left( \min_{s \in S_A} \pi(s) = \min_{s \in S_B} \pi(s) \right) = \frac{|S_A \cap S_B|}{|S_A \cup S_B|} = J(A, B)$$

**Proof:**
Consider the union $S_A \cup S_B$. When applying a random permutation $\pi$, every element in $S_A \cup S_B$ is equally likely to be mapped to the minimum value. 
- An element falls in both $S_A$ and $S_B$ if and only if it belongs to $S_A \cap S_B$.
- Therefore, the minimum under $\pi$ belongs to both sets with probability $\frac{|S_A \cap S_B|}{|S_A \cup S_B|} = J(A, B)$. $\blacksquare$

To generate a signature vector of length $M$ (e.g. $M = 128$), we generate $M$ independent universal hash functions:
$$h_i(x) = (a_i x + b_i) \bmod p, \quad i \in \{1, \dots, M\}$$
where $p$ is a large prime ($2^{31} - 1$) and $a_i, b_i$ are random integers.
The $i$-th component of the signature is:
$$\text{sig}_D[i] = \min_{s \in S_D} h_i(\text{hash}(s))$$

The estimated Jaccard similarity is simply the fraction of matching slots:
$$\hat{J}(A, B) = \frac{1}{M} \sum_{i=1}^M \mathbb{I}(\text{sig}_A[i] == \text{sig}_B[i])$$
By the Central Limit Theorem, the standard error of this estimate is bounded by:
$$\sigma \approx \frac{1}{\sqrt{M}}$$
For $M = 128$, the standard error is $\frac{1}{\sqrt{128}} \approx 0.088$.

---

#### Locality-Sensitive Hashing (LSH) Banding
Comparing all pairs of signatures across a corpus of $N = 10^9$ documents would require $\frac{N(N-1)}{2} \approx 5 \times 10^{17}$ comparisons—computationally impossible.

LSH solves this by grouping the $M$ signature integers into **$b$ bands of $r$ rows** each ($M = b \cdot r$):
1. For each band $k \in \{1, \dots, b\}$, hash the $r$-tuple of integers $\text{sig}_D[k \cdot r : (k+1) \cdot r]$ into a hash table bucket.
2. Two documents become a **candidate duplicate pair** if they collide in **at least one band**.

**Derivation of the LSH S-Curve:**
Let two documents have true Jaccard similarity $s = J(A, B)$:
- Probability that all $r$ rows match in a specific band: $s^r$
- Probability that at least one row differs in that band: $1 - s^r$
- Probability that the documents fail to match across all $b$ independent bands: $(1 - s^r)^b$
- Probability that the documents collide in **at least one band**:
$$P_{\text{collision}}(s) = 1 - (1 - s^r)^b$$

This creates a sharp **S-shaped sigmoid curve**. The $50\%$ probability threshold $s^*$ occurs where $P_{\text{collision}}(s^*) = 0.5$:
$$1 - (1 - (s^*)^r)^b = 0.5 \implies (1 - (s^*)^r)^b = 0.5$$
$$1 - (s^*)^r = (0.5)^{\frac{1}{b}} \approx 1 - \frac{\ln 2}{b} \implies s^* \approx \left( \frac{1}{b} \right)^{\frac{1}{r}}$$

```text
Collision Probability P(s)
 1.0 ┤                      ╭──────────────────
     │                    ╭─╯
 0.8 ┤                   ╭╯
     │                  ╭╯
 0.5 ┤ ─ ─ ─ ─ ─ ─ ─ ─ ╭╯   (Threshold s* ~= (1/b)^(1/r))
     │                ╭╯
 0.2 ┤               ╭╯
     │             ╭─╯
 0.0 ┴─────────────┴────────────────────────────
     0.0          s*                          1.0  Jaccard Similarity s
```

By choosing $b$ and $r$, we tune the threshold precisely:
- $M = 128, b = 16, r = 8 \implies s^* = (1/16)^{1/8} \approx 0.707$ (standard for web near-deduplication).
- Pairs with $s > 0.85$ collide with probability $> 99.8\%$, while unrelated documents ($s < 0.2$) collide with probability $< 10^{-6}$.

---

### 4. Benchmark Decontamination & Synthetic Data
- **Contamination Auditing:** To ensure true zero-shot reasoning evaluation, modern pipelines scan pretraining text against test sets (MMLU, GSM8k, HumanEval). If any 13-gram or 8-gram substring of a test question appears in a pretraining document, that document is removed or decontaminated.
- **Synthetic Data (Cosmopedia, UltraChat):** Recent foundation models (e.g. Phi-3, Gemma-2, LLaMA-3) mix in hundreds of billions of synthetic tokens: rewriting raw web articles into structured, pedagogical textbooks and multi-turn student-teacher dialogues, vastly accelerating reasoning density per token.

---

## 💻 Python Implementation Architecture

The accompanying `experiment.py` script implements:
1. Pure Python word and character $k$-shingling.
2. Exact Jaccard similarity computation.
3. Universal hash function generation using prime $p = 2^{31} - 1$.
4. MinHash signature generation ($M = 128$) and LSH banding ($b = 16, r = 8$).
5. Comparison between exact SHA-256 and MinHash near-deduplication.
6. Gopher / C4 heuristic filters (stopword ratio, symbol ratio, repetition).
7. Benchmark decontamination n-gram containment check.

Run the experiment directly:
```bash
python3 phase_18_large_language_models_and_pretraining/02_data_curation_deduplication_and_filtering/experiment.py
```
