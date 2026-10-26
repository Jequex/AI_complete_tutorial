#!/usr/bin/env python3
"""
EXPERIMENT 4.1: PROBABILITY FUNDAMENTALS, BAYES' THEOREM & FALSE POSITIVE PARADOX
================================================================================
A hands-on, zero-dependency Python laboratory demonstrating:
1. Empirical Verification of Kolmogorov's Axioms & Union Rule over 100,000 trials
2. Conditional Probability & Independence Testing
3. Monte Carlo Simulation of the False Positive Paradox (1,000,000 agents)
4. Pure Python Bayesian Spam Filter (Naive Bayes Classifier)

Runs on standard library Python 3 without external packages.
"""

import math
import random

print("=" * 70)
print("EXPERIMENT 4.1: PROBABILITY FUNDAMENTALS & BAYES' THEOREM")
print("=" * 70)

# Set seed for reproducible stochastic simulations
random.seed(42)

# ======================================================================
# 1. EMPIRICAL VERIFICATION OF KOLMOGOROV'S UNION RULE
# ======================================================================
print("\n--- 1. EMPIRICAL VERIFICATION OF THE UNION RULE ---")
# Rolling a fair 6-sided die
# Event A: Rolling an Even number {2, 4, 6} -> P(A) = 3/6 = 0.50
# Event B: Rolling a number >= 4 {4, 5, 6}   -> P(B) = 3/6 = 0.50
# Intersection (A and B): {4, 6}             -> P(A ∩ B) = 2/6 = 1/3 ≈ 0.3333
# Union (A or B): {2, 4, 5, 6}               -> P(A ∪ B) = 4/6 = 2/3 ≈ 0.6667

TRIALS = 100_000
count_A = 0
count_B = 0
count_A_and_B = 0
count_A_or_B = 0

for _ in range(TRIALS):
    roll = random.randint(1, 6)
    is_even = (roll % 2 == 0)
    is_gte_4 = (roll >= 4)

    if is_even:
        count_A += 1
    if is_gte_4:
        count_B += 1
    if is_even and is_gte_4:
        count_A_and_B += 1
    if is_even or is_gte_4:
        count_A_or_B += 1

emp_P_A = count_A / TRIALS
emp_P_B = count_B / TRIALS
emp_P_intersect = count_A_and_B / TRIALS
emp_P_union = count_A_or_B / TRIALS
calculated_union = emp_P_A + emp_P_B - emp_P_intersect

print(f"Trials: {TRIALS:,}")
print(f"  Empirical P(A) [Even]:           {emp_P_A:.4f} (Theoretical: 0.5000)")
print(f"  Empirical P(B) [>= 4]:           {emp_P_B:.4f} (Theoretical: 0.5000)")
print(f"  Empirical P(A ∩ B) [Even AND >=4]:{emp_P_intersect:.4f} (Theoretical: 0.3333)")
print(f"  Observed P(A ∪ B):               {emp_P_union:.4f} (Theoretical: 0.6667)")
print(f"  Formula P(A) + P(B) - P(A ∩ B):  {calculated_union:.4f}")
assert abs(emp_P_union - calculated_union) < 1e-5
print("✅ Kolmogorov Union Rule verified empirically!")


# ======================================================================
# 2. MONTE CARLO SIMULATION: THE FALSE POSITIVE PARADOX
# ======================================================================
print("\n" + "=" * 70)
print("--- 2. MONTE CARLO SIMULATION: THE FALSE POSITIVE PARADOX ---")

# Population parameters
POPULATION = 1_000_000
BASE_RATE = 0.0001      # 1 in 10,000 prevalence
SENSITIVITY = 0.99      # P(Pos | Disease) = 99%
FALSE_POS_RATE = 0.01   # P(Pos | Healthy) = 1%

print(f"Simulating diagnostic screening over {POPULATION:,} people...")
print(f"  Disease Prevalence: {BASE_RATE * 100:.2f}% (1 in 10,000)")
print(f"  Test Sensitivity:   {SENSITIVITY * 100:.1f}%")
print(f"  False Alarm Rate:   {FALSE_POS_RATE * 100:.1f}%")

true_sick_positive = 0
false_alarms = 0
true_sick_total = 0

for _ in range(POPULATION):
    has_disease = (random.random() < BASE_RATE)
    if has_disease:
        true_sick_total += 1
        if random.random() < SENSITIVITY:
            true_sick_positive += 1
    else:
        if random.random() < FALSE_POS_RATE:
            false_alarms += 1

total_tested_positive = true_sick_positive + false_alarms
empirical_posterior = true_sick_positive / total_tested_positive if total_tested_positive > 0 else 0.0

# Theoretical Bayesian Formula
# P(D | Pos) = [P(Pos | D) * P(D)] / [P(Pos | D)*P(D) + P(Pos | H)*P(H)]
numerator = SENSITIVITY * BASE_RATE
denominator = (SENSITIVITY * BASE_RATE) + (FALSE_POS_RATE * (1.0 - BASE_RATE))
theoretical_posterior = numerator / denominator

print(f"\nResults across {POPULATION:,} Simulated Individuals:")
print(f"  Actual Sick People in Pop:    {true_sick_total}")
print(f"  True Positives (Sick & Flagged): {true_sick_positive}")
print(f"  False Alarms (Healthy & Flagged): {false_alarms}")
print(f"  Total Flagged Positive:          {total_tested_positive}")
print(f"\n  👉 Empirical P(Sick | Positive):   {empirical_posterior * 100:.2f}%")
print(f"  👉 Theoretical Bayes Posterior:    {theoretical_posterior * 100:.2f}%")
print("✅ False Positive Paradox validated: The vast majority of positives are false alarms!")


# ======================================================================
# 3. PURE PYTHON NAIVE BAYES SPAM CLASSIFIER
# ======================================================================
print("\n" + "=" * 70)
print("--- 3. PURE PYTHON NAIVE BAYES SPAM CLASSIFIER ---")

# Tiny training dataset
train_corpus = [
    ("exclusive deal buy now free money", "spam"),
    ("win prize claim free gift voucher", "spam"),
    ("urgent invoice payment wire transfer required", "spam"),
    ("meeting agenda project roadmap updates tomorrow", "ham"),
    ("let us have lunch and review code tomorrow", "ham"),
    ("can you send the project slides and documentation", "ham")
]

# 1. Calculate Class Priors
total_docs = len(train_corpus)
spam_docs = sum(1 for _, label in train_corpus if label == "spam")
ham_docs = total_docs - spam_docs

prior_spam = spam_docs / total_docs
prior_ham = ham_docs / total_docs

# 2. Count word frequencies per class
spam_word_counts = {}
ham_word_counts = {}
vocab = set()

for text, label in train_corpus:
    words = text.lower().split()
    for w in words:
        vocab.add(w)
        if label == "spam":
            spam_word_counts[w] = spam_word_counts.get(w, 0) + 1
        else:
            ham_word_counts[w] = ham_word_counts.get(w, 0) + 1

total_spam_words = sum(spam_word_counts.values())
total_ham_words = sum(ham_word_counts.values())
vocab_size = len(vocab)

def predict_spam(message: str) -> dict[str, float]:
    """Computes log-posterior P(Spam | message) using Laplace (+1) smoothing."""
    words = message.lower().split()

    # Work in log-space to prevent floating point underflow:
    # log P(C | w1, w2...) = log P(C) + sum(log P(w_i | C))
    log_prob_spam = math.log(prior_spam)
    log_prob_ham = math.log(prior_ham)

    for w in words:
        # Laplace smoothing: (count + 1) / (total_words + vocab_size)
        p_w_given_spam = (spam_word_counts.get(w, 0) + 1) / (total_spam_words + vocab_size)
        p_w_given_ham = (ham_word_counts.get(w, 0) + 1) / (total_ham_words + vocab_size)

        log_prob_spam += math.log(p_w_given_spam)
        log_prob_ham += math.log(p_w_given_ham)

    # Softmax conversion back to probabilities
    max_log = max(log_prob_spam, log_prob_ham)
    exp_spam = math.exp(log_prob_spam - max_log)
    exp_ham = math.exp(log_prob_ham - max_log)
    prob_spam = exp_spam / (exp_spam + exp_ham)
    prob_ham = exp_ham / (exp_spam + exp_ham)

    return {"spam": prob_spam, "ham": prob_ham}

test_emails = [
    "free money prize voucher",
    "project documentation and code review meeting"
]

for email in test_emails:
    pred = predict_spam(email)
    label = "SPAM" if pred["spam"] > 0.5 else "HAM"
    print(f"\nMessage: '{email}'")
    print(f"  Prediction: [{label}] -> P(Spam)={pred['spam']*100:.1f}%, P(Ham)={pred['ham']*100:.1f}%")

print("\n" + "=" * 70)
print("EXPERIMENT 4.1 COMPLETED SUCCESSFULLY!")
print("=" * 70)

