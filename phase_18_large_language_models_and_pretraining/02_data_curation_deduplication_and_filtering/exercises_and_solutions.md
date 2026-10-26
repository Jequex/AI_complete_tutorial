# Topic 18.2: Exercises & Solutions — Data Curation, Deduplication, and Filtering

---

## Exercise 1: LSH Banding Parameter Optimization & S-Curve Analysis

### Problem
A data engineering team has $M = 128$ MinHash hash permutations and wants to design an LSH indexing pipeline to deduplicate Common Crawl web documents.
1. The team tests three partitioning choices for $(b, r)$ such that $b \cdot r = 128$:
   - Configuration 1: $b = 32, r = 4$
   - Configuration 2: $b = 16, r = 8$
   - Configuration 3: $b = 8, r = 16$
   For each configuration, calculate the theoretical $50\%$ collision threshold $s^* \approx (1/b)^{1/r}$.
2. If the team defines "near-duplicate" as any document pair with true Jaccard similarity $s \ge 0.80$, and "unrelated" as $s \le 0.30$, compute the collision probability $P_{\text{collision}}(s) = 1 - (1 - s^r)^b$ for both $s = 0.80$ and $s = 0.30$ under all three configurations.
3. Which configuration provides the best balance between high recall for near-duplicates and low false-positive rate for unrelated documents?

---

### Solution

#### Step 1: Compute Theoretical Thresholds $s^* \approx (1/b)^{1/r}$
- **Configuration 1 ($b = 32, r = 4$):**
  $$s_1^* = \left(\frac{1}{32}\right)^{1/4} = (0.03125)^{0.25} \approx 0.4204$$
- **Configuration 2 ($b = 16, r = 8$):**
  $$s_2^* = \left(\frac{1}{16}\right)^{1/8} = (0.0625)^{0.125} \approx 0.7071$$
- **Configuration 3 ($b = 8, r = 16$):**
  $$s_3^* = \left(\frac{1}{8}\right)^{1/16} = (0.125)^{0.0625} \approx 0.8781$$

#### Step 2: Compute Collision Probabilities
We use the exact formula $P(s) = 1 - (1 - s^r)^b$:

- **For Configuration 1 ($b = 32, r = 4$):**
  - At $s = 0.80$:
    $$s^4 = 0.80^4 = 0.4096$$
    $$1 - s^4 = 0.5904$$
    $$(1 - s^4)^{32} = (0.5904)^{32} \approx 5.3 \times 10^{-8}$$
    $$P(0.80) = 1 - 5.3 \times 10^{-8} \approx \mathbf{0.99999995} \quad (99.999995\% \text{ recall})$$
  - At $s = 0.30$:
    $$s^4 = 0.30^4 = 0.0081$$
    $$1 - s^4 = 0.9919$$
    $$(1 - s^4)^{32} = (0.9919)^{32} \approx 0.7709$$
    $$P(0.30) = 1 - 0.7709 = \mathbf{0.2291} \quad (22.9\% \text{ false positives})$$

- **For Configuration 2 ($b = 16, r = 8$):**
  - At $s = 0.80$:
    $$s^8 = 0.80^8 \approx 0.16777$$
    $$1 - s^8 = 0.83223$$
    $$(1 - s^8)^{16} = (0.83223)^{16} \approx 0.05193$$
    $$P(0.80) = 1 - 0.05193 = \mathbf{0.9481} \quad (94.81\% \text{ recall})$$
  - At $s = 0.30$:
    $$s^8 = 0.30^8 = 0.00006561$$
    $$1 - s^8 = 0.99993439$$
    $$(1 - s^8)^{16} = (0.99993439)^{16} \approx 0.99895$$
    $$P(0.30) = 1 - 0.99895 = \mathbf{0.00105} \quad (0.105\% \text{ false positives})$$

- **For Configuration 3 ($b = 8, r = 16$):**
  - At $s = 0.80$:
    $$s^{16} = 0.80^{16} \approx 0.02815$$
    $$1 - s^{16} = 0.97185$$
    $$(1 - s^{16})^8 = (0.97185)^8 \approx 0.7937$$
    $$P(0.80) = 1 - 0.7937 = \mathbf{0.2063} \quad (20.63\% \text{ recall})$$
  - At $s = 0.30$:
    $$s^{16} = 0.30^{16} \approx 4.3 \times 10^{-9} \implies P(0.30) \approx 3.4 \times 10^{-8}$$

#### Step 3: Engineering Decision
- Configuration 1 has a severe false positive rate ($22.9\%$ of unrelated documents collide, creating billions of unnecessary pairwise Jaccard computations).
- Configuration 3 suffers from catastrophic false negatives ($79.4\%$ of true near-duplicates are missed).
- **Configuration 2 ($b = 16, r = 8$)** is optimal: it captures $94.8\%$ of true near-duplicates while allowing only $0.1\%$ false positives to enter the secondary verification stage.

---

## Exercise 2: Shannon Entropy & Repetition Quality Filtering

### Problem
Automated web crawlers frequently scrape infinite JavaScript loops or placeholder text such as:
- Document A (Repetitive): A sequence consisting of only the 4 words `"loading data please wait "` repeated 200 times (total 800 words).
- Document B (Natural Prose): An 800-word excerpt from an encyclopedia containing 320 unique words with unigram probabilities $p(w)$.
1. Formulate the unigram word entropy $H(W) = -\sum_{w \in V} p(w) \log_2 p(w)$.
2. Calculate the exact unigram word entropy $H(W_A)$ for Document A.
3. If an encyclopedia text has an empirical unigram entropy of $H(W_B) \approx 7.2 \text{ bits/word}$, how does this contrast with Document A?
4. Formulate the maximum 3-gram frequency heuristic ratio and evaluate it for Document A.

---

### Solution

#### Step 1: Unigram Entropy Formulation
$$H(W) = -\sum_{i=1}^{|V|} p(w_i) \log_2 p(w_i)$$

#### Step 2: Evaluate for Document A
Document A contains $|V| = 4$ unique words: `{"loading", "data", "please", "wait"}`. Each word appears exactly 200 times out of 800 words, so $p(w_i) = \frac{200}{800} = 0.25$.
$$H(W_A) = -\sum_{i=1}^4 0.25 \log_2(0.25) = -4 \cdot (0.25 \cdot (-2.0)) = 2.0 \text{ bits/word}$$

#### Step 3: Comparison
- $H(W_A) = 2.0 \text{ bits/word}$
- $H(W_B) = 7.2 \text{ bits/word}$
Document A exhibits an entropy deficit of over $5.2$ bits per word ($> 72\%$ reduction in informational diversity). A lightweight entropy filter rejecting any document with $H(W) < 4.0 \text{ bits/word}$ eliminates $100\%$ of such low-entropy crawler loops instantly.

#### Step 4: 3-Gram Repetition Ratio
In Document A, the 3-gram sequence `"loading data please"`, `"data please wait"`, `"please wait loading"`, and `"wait loading data"` each repeat roughly 200 times.
$$\text{Max 3-gram Ratio} = \frac{200}{800 - 2} \approx 25.06\%$$
Under standard Gopher / C4 rules, any document where a single 3-gram exceeds $20\%$ of all 3-grams is immediately discarded. Document A is rejected by both filters.

---

## Exercise 3: Benchmark Decontamination via n-gram Containment Probability

### Problem
Suppose a benchmark evaluation dataset contains $K = 5,000$ unique questions, each containing on average 30 words.
To prevent test set leakage, the team wants to verify that training text does not contain benchmark answers.
1. Why is an $n = 3$-gram check unsuitable for decontamination? (Consider how many common 3-word English phrases naturally occur).
2. Given an effective English vocabulary of $|V_{\text{eff}}| = 10,000$ common words, if 13 words were chosen randomly and independently, what would be the probability of a specific 13-gram appearing?
3. Explain why 13-gram exact matching creates an effective zero-false-positive boundary between genuine memorization/leakage and natural linguistic coincidence.

---

### Solution

#### Step 1: The Problem with Small $n$ ($n = 3$)
Common 3-grams like `"in order to"`, `"as well as"`, or `"according to the"` appear millions of times in any natural text corpus.
If a benchmark question contains `"in order to"`, an $n=3$ containment check would flag and discard virtually every high-quality document in Common Crawl, causing massive false positives and destroying valid training data.

#### Step 2: Random Collision Probability of 13-grams
For a random sequence of $n = 13$ words drawn from $|V_{\text{eff}}| = 10,000$:
$$P(\text{exact match}) = \left(\frac{1}{|V_{\text{eff}}|}\right)^{13} = (10^{-4})^{13} = 10^{-52}$$
Even accounting for natural language grammar where the effective branching factor (perplexity) is roughly $\approx 20$:
$$P(\text{grammatical 13-gram match}) \approx (20)^{-13} \approx (8.192 \times 10^{16})^{-1} \approx 1.2 \times 10^{-17}$$

In a 1-Trillion token corpus ($10^{12}$ tokens), the expected number of accidental matches with a specific 13-gram is:
$$\mathbb{E}[\text{accidental matches}] \approx 10^{12} \times 1.2 \times 10^{-17} \approx 1.2 \times 10^{-5} \ll 1$$

#### Step 3: Conclusion
An accidental collision of a 13-word contiguous sequence across a 1-Trillion token corpus has a probability of less than $0.001\%$. 
Therefore, if a 13-gram in the training corpus matches a 13-gram in a benchmark question, it is **statistically impossible to be a random coincidence**—it represents verbatim copying or data leakage with $> 99.999\%$ certainty.
