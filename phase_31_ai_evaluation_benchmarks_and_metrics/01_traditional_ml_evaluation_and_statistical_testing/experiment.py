#!/usr/bin/env python3
"""
Phase 31: AI Evaluation, Benchmarks & Metrics
Topic 01: Traditional ML Evaluation and Statistical Testing

Implementation of:
1. Paired Student's t-test with two-tailed p-value calculation.
2. Wilcoxon Signed-Rank Test with rank tie-handling and continuity-corrected z-score.
3. Paired Monte Carlo Permutation Test (exact resampling under H0).
4. Bootstrap Confidence Intervals (Percentile & Empirical Basic).
5. Multiple Testing Adjustments: Bonferroni and Benjamini-Hochberg FDR.

Pure Python 3 standard library only. No external dependencies.
"""

import math
import random
from typing import List, Tuple, Dict, Any


def regularized_incomplete_beta(a: float, b: float, x: float, max_iter: int = 200, tol: float = 1e-12) -> float:
    """
    Computes regularized incomplete beta function I_x(a, b) using continued fraction (Betacf).
    Used to compute Student's t CDF accurately.
    """
    if x < 0.0 or x > 1.0:
        raise ValueError("x must be in [0, 1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    # Symmetry transformation
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - regularized_incomplete_beta(b, a, 1.0 - x, max_iter, tol)

    # Factor front
    lgamma_a = math.lgamma(a)
    lgamma_b = math.lgamma(b)
    lgamma_ab = math.lgamma(a + b)
    front = math.exp(lgamma_ab - lgamma_a - lgamma_b + a * math.log(x) + b * math.log(1.0 - x)) / a

    # Lentz continued fraction
    tiny = 1e-30
    f = 1.0
    c = 1.0
    d = 0.0

    for m in range(1, max_iter + 1):
        # Even step
        m2 = 2 * m
        # Numerator d_2m
        d2m = (m * (b - m) * x) / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 + d2m * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + d2m / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        f *= (c * d)

        # Odd step
        d2m_plus = -((a + m) * (a + b + m) * x) / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 + d2m_plus * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + d2m_plus / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        del_val = c * d
        f *= del_val

        if abs(del_val - 1.0) < tol:
            break

    return front * (f - 1.0) if f != 1.0 else front


def student_t_cdf(t: float, df: int) -> float:
    """
    Computes cumulative distribution function of Student's t-distribution with df degrees of freedom.
    """
    x = df / (df + t * t)
    prob = 0.5 * regularized_incomplete_beta(df / 2.0, 0.5, x)
    val = 1.0 - prob if t > 0 else prob
    return max(0.0, min(1.0, val))


def student_t_p_value_two_tailed(t: float, df: int) -> float:
    """Computes exact two-tailed p-value for Student's t-statistic."""
    x = df / (df + t * t)
    # The one-tailed upper tail is 0.5 * I_x(df/2, 0.5)
    tail = 0.5 * regularized_incomplete_beta(df / 2.0, 0.5, x)
    return max(0.0, min(1.0, 2.0 * tail))


def normal_cdf(z: float) -> float:
    """Standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def paired_t_test(scores_a: List[float], scores_b: List[float]) -> Dict[str, float]:
    """
    Computes Student's paired t-test for differences d_i = a_i - b_i.
    """
    n = len(scores_a)
    assert n == len(scores_b) and n > 1
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    std_d = math.sqrt(var_d)
    se_d = std_d / math.sqrt(n)
    t_stat = mean_d / se_d if se_d > 0 else 0.0
    df = n - 1
    p_value = student_t_p_value_two_tailed(abs(t_stat), df)
    return {
        "mean_diff": mean_d,
        "std_diff": std_d,
        "se_diff": se_d,
        "t_stat": t_stat,
        "df": float(df),
        "p_value": p_value,
    }


def wilcoxon_signed_rank_test(scores_a: List[float], scores_b: List[float]) -> Dict[str, float]:
    """
    Computes Wilcoxon Signed-Rank Test with tie-ranking and continuity-corrected normal approximation.
    """
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    non_zero = [(abs(d), 1 if d > 0 else -1) for d in diffs if abs(d) > 1e-12]
    n_r = len(non_zero)
    if n_r == 0:
        return {"w_stat": 0.0, "p_value": 1.0, "n_r": 0.0}

    # Sort by absolute difference
    sorted_items = sorted(enumerate(non_zero), key=lambda x: x[1][0])

    # Assign average ranks for ties
    ranks = [0.0] * n_r
    i = 0
    while i < n_r:
        j = i
        while j < n_r - 1 and abs(sorted_items[j][1][0] - sorted_items[j + 1][1][0]) < 1e-12:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            orig_idx = sorted_items[k][0]
            ranks[orig_idx] = avg_rank
        i = j + 1

    w_plus = sum(ranks[idx] for idx, (_, sign) in enumerate(non_zero) if sign > 0)
    w_minus = sum(ranks[idx] for idx, (_, sign) in enumerate(non_zero) if sign < 0)
    w_stat = min(w_plus, w_minus)

    # Large sample approximation
    mean_w = n_r * (n_r + 1) / 4.0
    var_w = n_r * (n_r + 1) * (2 * n_r + 1) / 24.0
    sigma_w = math.sqrt(var_w)

    z = (w_stat - mean_w + 0.5) / sigma_w if sigma_w > 0 else 0.0
    p_value = 2.0 * normal_cdf(z) if z < 0 else 2.0 * (1.0 - normal_cdf(z))

    return {
        "w_stat": w_stat,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "n_r": float(n_r),
        "z_stat": z,
        "p_value": min(1.0, p_value),
    }


def paired_permutation_test(scores_a: List[float], scores_b: List[float], num_resamples: int = 10000) -> Dict[str, float]:
    """
    Monte Carlo paired permutation test under H0 of sign exchangeability.
    """
    n = len(scores_a)
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    obs_t = abs(sum(diffs) / n)

    count_as_extreme = 0
    rng = random.Random(42)

    for _ in range(num_resamples):
        perm_sum = 0.0
        for d in diffs:
            sign = 1.0 if rng.random() > 0.5 else -1.0
            perm_sum += sign * d
        perm_t = abs(perm_sum / n)
        if perm_t >= obs_t - 1e-12:
            count_as_extreme += 1

    # Finite sample unbiased p-value
    p_value = (count_as_extreme + 1) / (num_resamples + 1)
    return {
        "obs_diff": sum(diffs) / n,
        "count_as_extreme": float(count_as_extreme),
        "num_resamples": float(num_resamples),
        "p_value": p_value,
    }


def bootstrap_confidence_interval(data: List[float], num_resamples: int = 2000, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Calculates Percentile and Empirical Basic bootstrap confidence intervals for the mean.
    """
    n = len(data)
    point_est = sum(data) / n
    rng = random.Random(1337)
    boot_means = []

    for _ in range(num_resamples):
        sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    idx_lower = int(math.floor(num_resamples * (alpha / 2.0)))
    idx_upper = int(math.ceil(num_resamples * (1.0 - alpha / 2.0))) - 1

    pct_lower = boot_means[idx_lower]
    pct_upper = boot_means[idx_upper]

    # Basic empirical bootstrap: 2*theta_hat - boot_upper, 2*theta_hat - boot_lower
    basic_lower = 2.0 * point_est - pct_upper
    basic_upper = 2.0 * point_est - pct_lower

    return {
        "point_estimate": point_est,
        "bootstrap_mean": sum(boot_means) / num_resamples,
        "percentile_ci": (pct_lower, pct_upper),
        "basic_ci": (basic_lower, basic_upper),
        "alpha": alpha,
    }


def multiple_testing_corrections(raw_p_values: List[Tuple[str, float]], alpha: float = 0.05, fdr_q: float = 0.05) -> Dict[str, Any]:
    """
    Applies Bonferroni FWER and Benjamini-Hochberg FDR adjustments to raw p-values.
    """
    m = len(raw_p_values)
    bonf_alpha = alpha / m

    # Bonferroni rejections
    bonf_discoveries = [name for name, p in raw_p_values if p <= bonf_alpha]

    # Benjamini-Hochberg procedure
    sorted_p = sorted(raw_p_values, key=lambda x: x[1])
    bh_thresholds = [(i + 1, (i + 1) / m * fdr_q) for i in range(m)]

    # Find largest k such that p_(k) <= (k/m)*Q
    max_k = -1
    for k in range(m - 1, -1, -1):
        if sorted_p[k][1] <= bh_thresholds[k][1]:
            max_k = k
            break

    bh_discoveries = [sorted_p[i][0] for i in range(max_k + 1)] if max_k >= 0 else []

    bh_table = []
    for rank, (name, p_val) in enumerate(sorted_p, start=1):
        thresh = (rank / m) * fdr_q
        bh_table.append({
            "rank": rank,
            "name": name,
            "p_value": p_val,
            "bh_threshold": thresh,
            "rejected": rank <= (max_k + 1),
        })

    return {
        "m": m,
        "alpha": alpha,
        "bonferroni_threshold": bonf_alpha,
        "bonferroni_discoveries": bonf_discoveries,
        "fdr_q": fdr_q,
        "bh_max_k": max_k + 1 if max_k >= 0 else 0,
        "bh_discoveries": bh_discoveries,
        "bh_table": bh_table,
    }


def run_experiment():
    print("=" * 80)
    print("PHASE 31 — TOPIC 01: TRADITIONAL ML EVALUATION & STATISTICAL TESTING")
    print("=" * 80)

    # 1. Simulate 20-fold cross-validation scores between Model A (Candidate) and Model B (Baseline)
    rng = random.Random(2026)
    n_folds = 20
    # True latent effect is +1.8% accuracy gain with fold variance
    baseline_scores = [round(82.0 + rng.gauss(0, 2.5), 2) for _ in range(n_folds)]
    candidate_scores = [round(b + 1.8 + rng.gauss(0, 1.2), 2) for b in baseline_scores]

    print("\n[1] Cross-Validation Fold Scores (20 Folds):")
    print(f"{'Fold':<6} | {'Baseline (B)':<14} | {'Candidate (A)':<14} | {'Difference (A-B)':<16}")
    print("-" * 56)
    for i in range(min(10, n_folds)):
        d = candidate_scores[i] - baseline_scores[i]
        print(f"{i+1:<6} | {baseline_scores[i]:<14.2f} | {candidate_scores[i]:<14.2f} | {d:+16.2f}")
    print("... (showing first 10 of 20 folds)")

    # 2. Paired Student's t-test
    t_res = paired_t_test(candidate_scores, baseline_scores)
    print("\n[2] Paired Student's t-Test:")
    print(f"  • Mean Difference (d_bar) : {t_res['mean_diff']:+.4f}%")
    print(f"  • Standard Deviation (s_d): {t_res['std_diff']:.4f}%")
    print(f"  • Standard Error (SE)     : {t_res['se_diff']:.4f}%")
    print(f"  • t-statistic (df={int(t_res['df'])}): {t_res['t_stat']:.4f}")
    print(f"  • Two-tailed p-value      : {t_res['p_value']:.6e} ({'p < 0.05 (Significant)' if t_res['p_value'] < 0.05 else 'Not significant'})")

    # 3. Wilcoxon Signed-Rank Test
    w_res = wilcoxon_signed_rank_test(candidate_scores, baseline_scores)
    print("\n[3] Wilcoxon Signed-Rank Test:")
    print(f"  • Non-zero pairs (n_r)    : {int(w_res['n_r'])}")
    print(f"  • W+ rank sum             : {w_res['w_plus']:.1f}")
    print(f"  • W- rank sum             : {w_res['w_minus']:.1f}")
    print(f"  • Test statistic W        : {w_res['w_stat']:.1f}")
    print(f"  • Asymptotic z-score      : {w_res['z_stat']:.4f}")
    print(f"  • Two-tailed p-value      : {w_res['p_value']:.6e} ({'p < 0.05 (Significant)' if w_res['p_value'] < 0.05 else 'Not significant'})")

    # 4. Monte Carlo Paired Permutation Test
    perm_res = paired_permutation_test(candidate_scores, baseline_scores, num_resamples=10000)
    print("\n[4] Paired Monte Carlo Permutation Test (10,000 resamples):")
    print(f"  • Observed Mean Diff      : {perm_res['obs_diff']:+.4f}%")
    print(f"  • Extreme Replicates      : {int(perm_res['count_as_extreme'])} / {int(perm_res['num_resamples'])}")
    print(f"  • Exact p-value           : {perm_res['p_value']:.6e}")

    # 5. Bootstrap Confidence Intervals on Candidate Accuracy
    boot_res = bootstrap_confidence_interval(candidate_scores, num_resamples=2000, alpha=0.05)
    print("\n[5] Bootstrap 95% Confidence Intervals (2,000 Replicates):")
    print(f"  • Point Estimate (Mean)   : {boot_res['point_estimate']:.3f}%")
    print(f"  • Percentile 95% CI       : [{boot_res['percentile_ci'][0]:.3f}%, {boot_res['percentile_ci'][1]:.3f}%]")
    print(f"  • Empirical Basic 95% CI  : [{boot_res['basic_ci'][0]:.3f}%, {boot_res['basic_ci'][1]:.3f}%]")

    # 6. Multiple Testing Corrections: Comparing Candidate against 8 competing baseline variants
    competing_p_values = [
        ("Architecture_Alpha", 0.0012),
        ("Architecture_Beta", 0.0041),
        ("Architecture_Gamma", 0.0120),
        ("Architecture_Delta", 0.0234),
        ("Architecture_Epsilon", 0.0489),
        ("Architecture_Zeta", 0.0712),
        ("Architecture_Eta", 0.1840),
        ("Architecture_Theta", 0.5420),
    ]
    mult_res = multiple_testing_corrections(competing_p_values, alpha=0.05, fdr_q=0.05)
    print("\n[6] Multiple Comparisons Testing (m = 8 models):")
    print(f"  • Bonferroni Alpha Cutoff : {mult_res['bonferroni_threshold']:.5f}")
    print(f"  • Bonferroni Discoveries  : {mult_res['bonferroni_discoveries']}")
    print(f"  • Benjamini-Hochberg (FDR Q = 0.05) Cutoff k*: {mult_res['bh_max_k']}")
    print(f"  • Benjamini-Hochberg Discoveries: {mult_res['bh_discoveries']}")

    print("\nBenjamini-Hochberg Step-by-Step Table:")
    print(f"{'Rank':<5} | {'Model Name':<22} | {'Raw p':<10} | {'BH Cutoff':<12} | {'Decision':<10}")
    print("-" * 65)
    for row in mult_res["bh_table"]:
        status = "REJECT H0" if row["rejected"] else "ACCEPT H0"
        print(f"{row['rank']:<5} | {row['name']:<22} | {row['p_value']:<10.4f} | {row['bh_threshold']:<12.5f} | {status:<10}")

    # Assertions
    assert t_res["p_value"] < 0.05, "Candidate model should be significantly superior via t-test"
    assert w_res["p_value"] < 0.05, "Candidate model should be significantly superior via Wilcoxon"
    assert perm_res["p_value"] < 0.05, "Candidate model should be significantly superior via Permutation"
    assert len(mult_res["bh_discoveries"]) >= len(mult_res["bonferroni_discoveries"]), "BH must be at least as powerful as Bonferroni"
    print("\nAll statistical test assertions verified successfully! (Exit Code 0)")


if __name__ == "__main__":
    run_experiment()
