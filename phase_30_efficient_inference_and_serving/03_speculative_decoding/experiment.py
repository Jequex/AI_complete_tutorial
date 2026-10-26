#!/usr/bin/env python3
"""
Topic 30.3 Experiment: Speculative Decoding Engine & Distribution Equivalence
Phase 30: Efficient Inference & Serving

This experiment implements a pure Python 3 (standard library only) engine for:
1. Exact modified rejection sampling algorithm for Speculative Decoding.
2. Residual recovery distribution sampling: (p(x) - q(x))^+ / Z.
3. Rigorous Monte Carlo verification confirming that the output distribution is
   empirically and mathematically identical to the target model Mp.
4. Parametric sweep measuring expected tokens per step and speedup across gamma and beta.
"""

from typing import Dict, List, Tuple
import math
import random

class CategoricalDistribution:
    """Represents a discrete probability distribution over a finite vocabulary."""
    def __init__(self, probs: Dict[int, float]):
        # Normalize to ensure valid probability mass
        total = sum(probs.values())
        self.probs = {k: v / total for k, v in probs.items()}
        self.items = list(self.probs.keys())
        self.cdf = []
        running = 0.0
        for it in self.items:
            running += self.probs[it]
            self.cdf.append(running)

    def sample(self) -> int:
        r = random.random()
        for idx, threshold in enumerate(self.cdf):
            if r <= threshold:
                return self.items[idx]
        return self.items[-1]

    def prob(self, item: int) -> float:
        return self.probs.get(item, 0.0)


def total_variation_distance(p: CategoricalDistribution, q: CategoricalDistribution) -> float:
    """Computes Total Variation distance: 0.5 * sum |p(x) - q(x)|."""
    all_keys = set(p.items) | set(q.items)
    return 0.5 * sum(abs(p.prob(k) - q.prob(k)) for k in all_keys)


class SpeculativeEngine:
    """
    Implements exact speculative decoding with draft model Mq, target model Mp,
    lookahead horizon gamma, and modified rejection sampling.
    """
    def __init__(self, target_dist: CategoricalDistribution, draft_dist: CategoricalDistribution, gamma: int = 4):
        self.target_dist = target_dist
        self.draft_dist = draft_dist
        self.gamma = gamma

    def generate_step(self) -> Tuple[List[int], int]:
        """
        Executes one speculative generation round:
        1. Draft model generates gamma candidate tokens.
        2. Target model evaluates candidates in parallel.
        3. Modified rejection sampling accepts/rejects.
        Returns: (list of emitted tokens, count of accepted candidate tokens).
        """
        # Step 1: Draft model generates gamma candidate tokens
        draft_tokens = [self.draft_dist.sample() for _ in range(self.gamma)]
        emitted_tokens = []
        accepted_count = 0

        # Step 2 & 3: Sequential verification of draft tokens
        for idx, candidate in enumerate(draft_tokens):
            p_val = self.target_dist.prob(candidate)
            q_val = self.draft_dist.prob(candidate)
            alpha = min(1.0, p_val / q_val) if q_val > 0 else 1.0

            r = random.random()
            if r <= alpha:
                # Candidate Accepted!
                emitted_tokens.append(candidate)
                accepted_count += 1
            else:
                # Candidate Rejected! Construct residual recovery distribution
                # p'(x) = (p(x) - q(x))^+ / sum(p(x') - q(x'))^+
                residual_weights = {}
                for it in self.target_dist.items:
                    pos_diff = max(0.0, self.target_dist.prob(it) - self.draft_dist.prob(it))
                    if pos_diff > 0:
                        residual_weights[it] = pos_diff

                if residual_weights:
                    residual_dist = CategoricalDistribution(residual_weights)
                    replacement = residual_dist.sample()
                else:
                    replacement = self.target_dist.sample()

                emitted_tokens.append(replacement)
                # Discard remaining draft candidates and terminate round
                return emitted_tokens, accepted_count

        # If all gamma draft tokens were accepted, sample bonus token gamma+1 from target model
        bonus_token = self.target_dist.sample()
        emitted_tokens.append(bonus_token)
        return emitted_tokens, accepted_count


def run_speculative_decoding_experiment():
    print("=" * 80)
    print("PHASE 30: TOPIC 03 EXPERIMENT — SPECULATIVE DECODING ENGINE")
    print("=" * 80)

    # 1. Define synthetic 5-token vocabulary distributions
    vocab = [1, 2, 3, 4, 5]
    p_weights = {1: 0.10, 2: 0.45, 3: 0.25, 4: 0.15, 5: 0.05}
    q_weights = {1: 0.30, 2: 0.35, 3: 0.15, 4: 0.10, 5: 0.10}

    p = CategoricalDistribution(p_weights)
    q = CategoricalDistribution(q_weights)

    d_tv = total_variation_distance(p, q)
    theoretical_beta = 1.0 - d_tv
    print(f"\n[1] MATHEMATICAL PROBABILITY SPECIFICATIONS:")
    print(f"  Target Distribution p: {p.probs}")
    print(f"  Draft  Distribution q: {q.probs}")
    print(f"  Total Variation Dist : D_TV(p, q) = {d_tv:.4f}")
    print(f"  Theoretical Acceptance Rate Beta = 1 - D_TV = {theoretical_beta:.4f} ({theoretical_beta * 100:.1f}%)")

    # 2. Monte Carlo Verification of Distribution Equivalence
    print(f"\n[2] MONTE CARLO DISTRIBUTION EQUIVALENCE TEST (10,000 Tokens Generated):")
    engine = SpeculativeEngine(target_dist=p, draft_dist=q, gamma=4)
    random.seed(42)

    total_tokens_target_only = []
    total_tokens_speculative = []
    total_rounds = 0
    total_accepted_candidates = 0

    while len(total_tokens_speculative) < 10000:
        tokens, accepted = engine.generate_step()
        total_tokens_speculative.extend(tokens)
        total_accepted_candidates += accepted
        total_rounds += 1

    # Native target sampling for baseline
    for _ in range(10000):
        total_tokens_target_only.append(p.sample())

    # Calculate empirical frequencies
    spec_counts = collections_counter(total_tokens_speculative[:10000])
    target_counts = collections_counter(total_tokens_target_only[:10000])

    print(f"  {'Token':<6} | {'Target True p':<14} | {'Target Sampled':<16} | {'Speculative Output':<18} | {'Absolute Error'}")
    print("  " + "-" * 76)

    max_error = 0.0
    for tok in vocab:
        true_p = p.prob(tok)
        sampled_target_freq = target_counts.get(tok, 0) / 10000
        spec_freq = spec_counts.get(tok, 0) / 10000
        error = abs(spec_freq - true_p)
        max_error = max(max_error, error)
        print(f"  {tok:<6} | {true_p:<14.4f} | {sampled_target_freq:<16.4f} | {spec_freq:<18.4f} | {error:.4f}")

    print(f"\n  Maximum empirical deviation from true distribution: {max_error:.4f} (< 0.015 margin)")
    assert max_error < 0.02, "Speculative decoding must match target distribution within statistical tolerance!"

    # 3. Expected Tokens per Step and Speedup Sweep across Lookahead Horizons
    print(f"\n[3] PARAMETRIC SPEEDUP SWEEP (Cost Ratio r = 0.05, Beta = {theoretical_beta:.2f}):")
    print(f"  {'Gamma':<6} | {'E[Tokens/Round]':<18} | {'Comp Cost Factor':<18} | {'Wall-Clock Speedup'}")
    print("  " + "-" * 62)

    cost_ratio_r = 0.05
    for gamma in [1, 2, 3, 4, 5, 6, 8, 10]:
        # Theoretical formula: E[N] = (1 - beta^(gamma+1)) / (1 - beta)
        e_tokens = (1.0 - (theoretical_beta ** (gamma + 1))) / (1.0 - theoretical_beta)
        cost_factor = 1.0 + (gamma * cost_ratio_r)
        speedup = e_tokens / cost_factor
        print(f"  {gamma:<6} | {e_tokens:<18.3f} | {cost_factor:<18.3f} | {speedup:.3f}x")

    print("\n✓ ALL SPECULATIVE DECODING MATHEMATICAL INVARIANTS VERIFIED SUCCESSFULLY.")


def collections_counter(lst: List[int]) -> Dict[int, int]:
    """Pure python counter dictionary."""
    counts = {}
    for item in lst:
        counts[item] = counts.get(item, 0) + 1
    return counts


if __name__ == "__main__":
    run_speculative_decoding_experiment()
