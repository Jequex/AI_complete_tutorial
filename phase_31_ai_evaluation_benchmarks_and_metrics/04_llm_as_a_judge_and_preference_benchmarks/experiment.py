#!/usr/bin/env python3
"""
Phase 31: AI Evaluation, Benchmarks & Metrics
Topic 04: LLM-as-a-Judge & Preference Benchmarks

Implementation of:
1. Tournament simulation of 6 LLM models under noisy pairwise judging.
2. Modeling of systematic judge biases: Position Bias (Order Effect) and Verbosity Bias.
3. Position-swapping bidirectional de-biasing protocol.
4. Online Bradley-Terry Elo rating update engine with K-factor adaptation.
5. Offline Maximum Likelihood Estimation (MLE) of Bradley-Terry latent strength parameters.
6. Empirical win-rate matrix and Elo convergence validation.

Pure Python 3 standard library only. No external dependencies.
"""

import math
import random
from typing import List, Tuple, Dict, Any


class LLMCompetitor:
    def __init__(self, name: str, true_elo: float, avg_verbosity: float):
        self.name = name
        self.true_elo = true_elo
        self.avg_verbosity = avg_verbosity  # Average response length in arbitrary units


def bradley_terry_expected(r_a: float, r_b: float) -> float:
    """Computes expected score of A against B: E_A = 1 / (1 + 10^((R_B - R_A)/400))."""
    return 1.0 / (1.0 + math.pow(10.0, (r_b - r_a) / 400.0))


def simulate_noisy_judgment(
    model_first: LLMCompetitor,
    model_second: LLMCompetitor,
    pos_bias_prob: float = 0.62,
    verbosity_weight: float = 0.05,
    rng: random.Random = None,
) -> int:
    """
    Simulates judge picking response in Position 1 (returns 1) or Position 2 (returns 2).
    Includes ground truth skill, position bias, and verbosity bias.
    """
    if rng is None:
        rng = random.Random()

    # Base Bradley-Terry probability of first beating second
    e_first = bradley_terry_expected(model_first.true_elo, model_second.true_elo)

    # Shift due to position bias (deviation from 0.5)
    bias_shift = pos_bias_prob - 0.50

    # Shift due to verbosity
    len_diff = model_first.avg_verbosity - model_second.avg_verbosity
    len_shift = verbosity_weight * (len_diff / 50.0)

    # Combined judge latent probability
    judge_p_first = max(0.01, min(0.99, e_first + bias_shift + len_shift))

    return 1 if rng.random() < judge_p_first else 2


def bidirectional_match(
    model_a: LLMCompetitor,
    model_b: LLMCompetitor,
    pos_bias_prob: float = 0.62,
    verbosity_weight: float = 0.05,
    rng: random.Random = None,
) -> Tuple[float, float, bool]:
    """
    Runs position-swapped match between A and B.
    Pass 1: A in Pos 1, B in Pos 2.
    Pass 2: B in Pos 1, A in Pos 2.
    Returns: (score_a, score_b, is_tie)
    """
    # Pass 1: A is Pos 1, B is Pos 2
    pass1_pick = simulate_noisy_judgment(model_a, model_b, pos_bias_prob, verbosity_weight, rng)
    a_won_pass1 = (pass1_pick == 1)

    # Pass 2: B is Pos 1, A is Pos 2
    pass2_pick = simulate_noisy_judgment(model_b, model_a, pos_bias_prob, verbosity_weight, rng)
    a_won_pass2 = (pass2_pick == 2)

    if a_won_pass1 and a_won_pass2:
        return 1.0, 0.0, False  # A wins decisively
    elif (not a_won_pass1) and (not a_won_pass2):
        return 0.0, 1.0, False  # B wins decisively
    else:
        return 0.5, 0.5, True   # Position flipped or tie &rarr; declared tie


def run_online_elo_tournament(
    competitors: List[LLMCompetitor],
    num_rounds: int = 1500,
    use_debiasing: bool = True,
    k_factor: float = 24.0,
    seed: int = 42,
) -> Tuple[Dict[str, float], List[List[float]]]:
    """
    Simulates round-robin / random matchmaking tournament and computes online Elo ratings.
    """
    rng = random.Random(seed)
    n = len(competitors)
    ratings = {c.name: 1200.0 for c in competitors}
    win_matrix = [[0.0] * n for _ in range(n)]

    for _ in range(num_rounds):
        # Pick 2 distinct models at random
        i, j = rng.sample(range(n), 2)
        mod_a = competitors[i]
        mod_b = competitors[j]

        if use_debiasing:
            s_a, s_b, _ = bidirectional_match(mod_a, mod_b, pos_bias_prob=0.62, rng=rng)
        else:
            # Naive unidirectional match (mod_a always in pos 1)
            pick = simulate_noisy_judgment(mod_a, mod_b, pos_bias_prob=0.62, rng=rng)
            s_a = 1.0 if pick == 1 else 0.0
            s_b = 1.0 - s_a

        # Record match results
        win_matrix[i][j] += s_a
        win_matrix[j][i] += s_b

        # Elo update
        e_a = bradley_terry_expected(ratings[mod_a.name], ratings[mod_b.name])
        e_b = 1.0 - e_a

        ratings[mod_a.name] += k_factor * (s_a - e_a)
        ratings[mod_b.name] += k_factor * (s_b - e_b)

    return ratings, win_matrix


def solve_bradley_terry_mle(
    competitors: List[LLMCompetitor],
    win_matrix: List[List[float]],
    max_iter: int = 200,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """
    Computes Maximum Likelihood Estimation of Bradley-Terry latent skill using Hunter's MM algorithm (2004).
    """
    n = len(competitors)
    # Total wins for each competitor
    w = [sum(win_matrix[i][j] for j in range(n) if i != j) for i in range(n)]
    
    # Initialize pi parameters uniformly
    pi = [1.0] * n

    for _ in range(max_iter):
        pi_prev = list(pi)
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i != j:
                    n_ij = win_matrix[i][j] + win_matrix[j][i]
                    if n_ij > 0:
                        denom += n_ij / (pi_prev[i] + pi_prev[j])
            if denom > 0:
                pi[i] = w[i] / denom

        # Normalize so geometric mean or sum is constant
        scale = sum(pi)
        pi = [p / scale * n for p in pi]

        # Convergence test
        diff = max(abs(pi[i] - pi_prev[i]) for i in range(n))
        if diff < tol:
            break

    # Anchor baseline (lowest true Elo model) to 1000
    base_pi = min(pi)
    mle_elos = {}
    for idx, c in enumerate(competitors):
        mle_elos[c.name] = 1000.0 + 400.0 * math.log10(pi[idx] / base_pi)

    return mle_elos


def run_experiment():
    print("=" * 80)
    print("PHASE 31 — TOPIC 04: LLM-AS-A-JUDGE & PREFERENCE BENCHMARKS")
    print("=" * 80)

    # 1. Define 6 competing LLM models with known latent capabilities
    models = [
        LLMCompetitor("Frontier-Titan-1", true_elo=1380.0, avg_verbosity=120.0),
        LLMCompetitor("Omni-Pro-v2", true_elo=1310.0, avg_verbosity=110.0),
        LLMCompetitor("Reasoning-Coder-70B", true_elo=1240.0, avg_verbosity=95.0),
        LLMCompetitor("Open-Instruct-70B", true_elo=1170.0, avg_verbosity=105.0),
        LLMCompetitor("Compact-Chat-8B", true_elo=1090.0, avg_verbosity=85.0),
        LLMCompetitor("Base-Draft-7B", true_elo=1000.0, avg_verbosity=80.0),
    ]

    print("\n[1] Benchmark Competitor Registry:")
    print(f"{'Model Name':<22} | {'True Latent Elo':<16} | {'Avg Verbosity':<14}")
    print("-" * 58)
    for m in models:
        print(f"{m.name:<22} | {m.true_elo:<16.1f} | {m.avg_verbosity:<14.1f}")

    # 2. Tournament A: Naive Unidirectional Judging (Severe Position Bias 62%)
    ratings_naive, wins_naive = run_online_elo_tournament(
        models, num_rounds=2000, use_debiasing=False, seed=101
    )

    # 3. Tournament B: Position-Swapped Bidirectional Judging (De-biased)
    ratings_debiased, wins_debiased = run_online_elo_tournament(
        models, num_rounds=2000, use_debiasing=True, seed=101
    )

    # 4. Bradley-Terry Offline MLE Solver
    mle_elos = solve_bradley_terry_mle(models, wins_debiased)

    print("\n[2] Tournament Results Comparison (2,000 matches):")
    print(f"{'Model Name':<22} | {'True Elo':<10} | {'Naive Online':<14} | {'Debiased Online':<16} | {'Offline MLE':<12}")
    print("-" * 80)
    for m in models:
        print(
            f"{m.name:<22} | {m.true_elo:<10.1f} | {ratings_naive[m.name]:<14.1f} | "
            f"{ratings_debiased[m.name]:<16.1f} | {mle_elos[m.name]:<12.1f}"
        )

    # 5. Check Rank Order Alignment
    true_order = [m.name for m in sorted(models, key=lambda x: x.true_elo, reverse=True)]
    debiased_order = [m.name for m in sorted(models, key=lambda x: ratings_debiased[x.name], reverse=True)]
    mle_order = [m.name for m in sorted(models, key=lambda x: mle_elos[x.name], reverse=True)]

    print("\n[3] Rank Order Verification:")
    print(f"  • True Rank Order     : {true_order}")
    print(f"  • Debiased Rank Order : {debiased_order}")
    print(f"  • Offline MLE Order   : {mle_order}")

    # 6. Empirical Win-Rate Matrix
    print("\n[4] Empirical Head-to-Head Win-Rate Matrix (% Wins of Row over Column):")
    header = f"{'':<20}" + "".join(f"{m.name[:8]:>9}" for m in models)
    print(header)
    print("-" * (20 + 9 * len(models)))
    for i, m_i in enumerate(models):
        row_str = f"{m_i.name[:18]:<20}"
        for j, m_j in enumerate(models):
            if i == j:
                row_str += f"{'--':>9}"
            else:
                total_played = wins_debiased[i][j] + wins_debiased[j][i]
                win_pct = (wins_debiased[i][j] / total_played * 100.0) if total_played > 0 else 50.0
                row_str += f"{win_pct:>8.1f}%"
        print(row_str)

    # Assertions
    assert debiased_order == true_order, "Debiased tournament must correctly rank all 6 models in descending order of true Elo"
    assert mle_order == true_order, "Offline Bradley-Terry MLE must perfectly match true rank ordering"
    assert mle_elos[models[0].name] > mle_elos[models[-1].name] + 300.0, "Elo spread must reflect genuine skill difference"

    print("\nAll LLM-as-a-Judge and preference tournament assertions passed successfully! (Exit Code 0)")


if __name__ == "__main__":
    run_experiment()
