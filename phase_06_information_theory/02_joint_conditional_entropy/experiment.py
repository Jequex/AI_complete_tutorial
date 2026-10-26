"""
Topic 6.2: Joint & Conditional Entropy Experiment
Demonstrating that Conditioning Reduces Uncertainty on Natural Text
Pure Python 3 Standard Library implementation (Zero External Dependencies).
"""

import math
from collections import Counter, defaultdict

def compute_entropy(prob_dist):
    """Computes Shannon Entropy H(X) = -sum p * log2(p) in bits."""
    return -sum(p * math.log2(p) for p in prob_dist if p > 0)

def main():
    print("=" * 72)
    print("TOPIC 6.2: JOINT & CONDITIONAL ENTROPY IN NATURAL LANGUAGE")
    print("=" * 72)

    # 1. Natural Language Sample Corpus
    corpus = (
        "artificial intelligence and deep learning models rely on statistical patterns "
        "and information theory to minimize uncertainty in next token prediction. "
        "context allows neural networks to predict subsequent tokens with high confidence. "
        "as context window increases, conditional entropy drops significantly."
    )
    n = len(corpus)
    print(f"Corpus Length: {n} characters\n")

    # --- 1. Unigram Entropy H(X) ---
    unigram_counts = Counter(corpus)
    unigram_probs = [count / n for count in unigram_counts.values()]
    h_unigram = compute_entropy(unigram_probs)

    print("--- 1. Unigram Entropy (Zero Context) ---")
    print(f"Unique Characters (Alphabet Size |V|): {len(unigram_counts)}")
    print(f"Theoretical Uniform Entropy (log2 |V|): {math.log2(len(unigram_counts)):.4f} bits")
    print(f"Empirical Unigram Entropy H(X):          {h_unigram:.4f} bits / char\n")

    # --- 2. Bigram Joint Entropy H(X, Y) & Conditional Entropy H(Y | X) ---
    # X = char_t, Y = char_{t+1}
    bigram_counts = Counter()
    cond_counts = defaultdict(Counter)

    for i in range(n - 1):
        c1, c2 = corpus[i], corpus[i+1]
        bigram_counts[(c1, c2)] += 1
        cond_counts[c1][c2] += 1

    total_bigrams = n - 1
    bigram_probs = [count / total_bigrams for count in bigram_counts.values()]
    h_joint = compute_entropy(bigram_probs)

    # Compute Conditional Entropy H(Y | X) = sum P(x) * H(Y | X = x)
    h_conditional_bigram = 0.0
    for c1, c2_counter in cond_counts.items():
        total_from_c1 = sum(c2_counter.values())
        p_c1 = total_from_c1 / total_bigrams
        cond_dist = [cnt / total_from_c1 for cnt in c2_counter.values()]
        h_conditional_bigram += p_c1 * compute_entropy(cond_dist)

    print("--- 2. Bigram Joint & Conditional Entropy (1 Character Context) ---")
    print(f"Joint Entropy H(X, Y):                 {h_joint:.4f} bits / bigram")
    print(f"Conditional Entropy H(Y | X):          {h_conditional_bigram:.4f} bits / char")
    print(f"Chain Rule Verification: H(X) + H(Y|X) = {h_unigram + h_conditional_bigram:.4f} bits")
    chain_diff = abs(h_joint - (h_unigram + h_conditional_bigram))
    # Note slight boundary effect of 1 char difference between n and n-1
    print(f"Chain Rule Discrepancy:                {chain_diff:.4f} bits")
    assert chain_diff < 0.05, f"Chain rule check failed: diff={chain_diff}"
    print(">> SUCCESS: Chain Rule of Entropy verified!\n")

    # --- 3. Trigram Conditional Entropy H(Z | X, Y) (2 Characters Context) ---
    trigram_cond_counts = defaultdict(Counter)
    for i in range(n - 2):
        c1, c2, c3 = corpus[i], corpus[i+1], corpus[i+2]
        trigram_cond_counts[(c1, c2)][c3] += 1

    total_trigrams = n - 2
    h_conditional_trigram = 0.0
    for context, target_counter in trigram_cond_counts.items():
        total_from_context = sum(target_counter.values())
        p_context = total_from_context / total_trigrams
        cond_dist = [cnt / total_from_context for cnt in target_counter.values()]
        h_conditional_trigram += p_context * compute_entropy(cond_dist)

    print("--- 3. Trigram Conditional Entropy (2 Characters Context) ---")
    print(f"Conditional Entropy H(Z | X, Y):       {h_conditional_trigram:.4f} bits / char\n")

    # --- 4. Verifying Information Never Hurts (Conditioning Reduces Entropy) ---
    print("--- 4. Theorem Verification: Conditioning Never Increases Entropy ---")
    print(f"No Context (Unigram):   {h_unigram:.4f} bits")
    print(f"1-Char Context (Bigram): {h_conditional_bigram:.4f} bits (reduction: {h_unigram - h_conditional_bigram:.4f} bits)")
    print(f"2-Char Context (Trigram):{h_conditional_trigram:.4f} bits (reduction: {h_unigram - h_conditional_trigram:.4f} bits)")

    assert h_unigram > h_conditional_bigram > h_conditional_trigram, "Entropy did not strictly decrease with context!"
    print("\n>> VERIFIED: H(Z | X, Y) < H(Y | X) < H(X)")
    print(">> Proven empirically: Context collapses uncertainty in language modeling.")
    print("=" * 72)

if __name__ == "__main__":
    main()

