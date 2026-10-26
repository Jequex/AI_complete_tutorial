"""
Topic 6.3: Cross-Entropy & Perplexity Experiment
Simulating Next-Token Prediction, Loss Computation, and Temperature Scaling
Pure Python 3 Standard Library implementation (Zero External Dependencies).
"""

import math

def stable_softmax(logits, temperature=1.0):
    """Numerically stable softmax with temperature scaling."""
    scaled = [z / temperature for z in logits]
    max_val = max(scaled)
    exp_vals = [math.exp(z - max_val) for z in scaled]
    sum_exp = sum(exp_vals)
    return [e / sum_exp for e in exp_vals]

def compute_cross_entropy(probs, target_idx):
    """Negative log-likelihood of the target index (in nats)."""
    p_target = max(probs[target_idx], 1e-15)
    return -math.log(p_target)

def compute_entropy(probs):
    """Shannon entropy of distribution in bits."""
    return -sum(p * math.log2(p) for p in probs if p > 0)

def main():
    print("=" * 72)
    print("TOPIC 6.3: CROSS-ENTROPY, PERPLEXITY & TEMPERATURE SCALING")
    print("=" * 72)

    vocab = ["the", "cat", "sat", "on", "mat", "dog", "ate", "food"]
    vocab_size = len(vocab)
    target_idx = 1  # target token is "cat"
    print(f"Vocabulary Size: {vocab_size} tokens | Target Token: '{vocab[target_idx]}'\n")

    # --- 1. Comparing Three Model Checkpoints ---
    # Model A: Untrained random guesses (all logits equal)
    logits_untrained = [0.0] * vocab_size

    # Model B: Partially trained (cat has a modest advantage)
    logits_partially_trained = [1.2, 2.8, 0.5, 0.1, 0.2, 1.5, 0.3, 0.4]

    # Model C: Well-trained (cat has high logit)
    logits_well_trained = [0.1, 6.5, 0.2, -0.5, 0.0, 1.1, -0.2, 0.1]

    models = [
        ("Untrained (Uniform)", logits_untrained),
        ("Partially Trained", logits_partially_trained),
        ("Well Trained", logits_well_trained),
    ]

    print("--- 1. Model Checkpoint Evaluation ---")
    print(f"{'Model Checkpoint':<22} | {'P(target)':<10} | {'Cross-Entropy (nats)':<22} | {'Perplexity (PPL)':<16}")
    print("-" * 76)

    losses = []
    for name, logits in models:
        probs = stable_softmax(logits, temperature=1.0)
        p_target = probs[target_idx]
        ce_loss = compute_cross_entropy(probs, target_idx)
        ppl = math.exp(ce_loss)
        losses.append((ce_loss, ppl))
        print(f"{name:<22} | {p_target:<10.4f} | {ce_loss:<22.4f} | {ppl:<16.2f}")

    # Assert that training monotonically decreases cross-entropy and perplexity
    assert losses[0][0] > losses[1][0] > losses[2][0], "Loss did not decrease with training!"
    assert losses[0][1] > losses[1][1] > losses[2][1], "Perplexity did not decrease with training!"
    # Untrained perplexity should equal exactly vocabulary size
    assert abs(losses[0][1] - vocab_size) < 1e-4, f"Untrained PPL {losses[0][1]} != vocab size {vocab_size}"
    print(">> SUCCESS: Untrained PPL matches vocabulary size exactly (8.00)!\n")

    # --- 2. Temperature Scaling Analysis ---
    temperatures = [0.2, 0.5, 1.0, 2.0, 5.0]
    test_logits = logits_partially_trained

    print("--- 2. Temperature Scaling on Logits ---")
    print(f"{'Temperature T':<15} | {'Entropy (bits)':<16} | {'Target Prob P(cat)':<20} | {'Behavior':<20}")
    print("-" * 76)

    prev_entropy = -1.0
    for T in temperatures:
        probs = stable_softmax(test_logits, temperature=T)
        entropy = compute_entropy(probs)
        p_target = probs[target_idx]
        if T < 0.5:
            behavior = "Greedy / Argmax Mode"
        elif T <= 1.0:
            behavior = "Balanced / Calibrated"
        else:
            behavior = "High Diversity / Flattened"

        print(f"{T:<15.1f} | {entropy:<16.4f} | {p_target:<20.4f} | {behavior:<20}")
        if prev_entropy >= 0:
            assert entropy > prev_entropy, "Entropy must monotonically increase with temperature!"
        prev_entropy = entropy

    print("\n>> VERIFIED: Entropy strictly increases with Temperature T.")
    print(">> Low T drives predictions toward deterministic argmax (lowest entropy).")
    print(">> High T drives predictions toward maximum entropy uniform distribution.")
    print("=" * 72)

if __name__ == "__main__":
    main()

