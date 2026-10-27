"""
Topic 05: Scientific Discovery & Literature Synthesis
=====================================================
Pure Python 3 standard library implementation (no torch, no numpy).

Implements:
1. Research Paper Formal Decomposition Schema (Problem, Prior, Hypothesis, Method, Math, Ablations).
2. Factorial Ablation Design Engine:
   Full 2^3 factorial experiment evaluating:
   - Factor A: Attention Variant (Standard Softmax vs Gated Scaled Attention)
   - Factor B: Positional Encoding (Absolute vs Rotary RoPE)
   - Factor C: Normalization (Post-LN vs Pre-RMSNorm)
3. Non-Parametric Statistical Testing Suite:
   - Standardized Effect Size (Cohen's d with pooled sample variance)
   - Non-parametric Percentile Bootstrap Confidence Intervals (95% CI)
   - Exact Two-Sided Permutation Significance Test (Monte Carlo permutations)
4. Automated Hypothesis Decision System (Reject H0 with statistical rigor).
5. Comprehensive assertions ensuring exit code 0.
"""

import math
import random
import sys
from typing import List, Dict, Tuple, Optional, Any

random.seed(42)

# -----------------------------------------------------------------------------
# 1. Statistical Math Primitives (Pure Python)
# -----------------------------------------------------------------------------
def mean(data: List[float]) -> float:
    return sum(data) / len(data)

def variance(data: List[float], ddof: int = 1) -> float:
    m = mean(data)
    n = len(data)
    return sum((x - m) ** 2 for x in data) / max(1, n - ddof)

def std_dev(data: List[float], ddof: int = 1) -> float:
    return math.sqrt(variance(data, ddof))

def cohens_d(group_a: List[float], group_b: List[float]) -> float:
    """
    Computes Cohen's d effect size between group_b (treatment) and group_a (baseline):
    d = (mean_b - mean_a) / s_pooled
    """
    n_a, n_b = len(group_a), len(group_b)
    var_a, var_b = variance(group_a), variance(group_b)
    s_pooled = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if s_pooled < 1e-12:
        return 0.0
    return (mean(group_b) - mean(group_a)) / s_pooled

def bootstrap_ci(
    group_a: List[float],
    group_b: List[float],
    num_bootstrap: int = 2000,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Non-parametric bootstrap of the difference of means (mean_b - mean_a).
    Returns (observed_diff, ci_lower, ci_upper).
    """
    obs_diff = mean(group_b) - mean(group_a)
    n_a, n_b = len(group_a), len(group_b)
    diffs = []

    for _ in range(num_bootstrap):
        sample_a = [random.choice(group_a) for _ in range(n_a)]
        sample_b = [random.choice(group_b) for _ in range(n_b)]
        diffs.append(mean(sample_b) - mean(sample_a))

    diffs.sort()
    lower_idx = int((alpha / 2.0) * num_bootstrap)
    upper_idx = int((1.0 - alpha / 2.0) * num_bootstrap)
    return obs_diff, diffs[lower_idx], diffs[upper_idx]

def permutation_test(
    group_a: List[float],
    group_b: List[float],
    num_permutations: int = 2000
) -> Tuple[float, float]:
    """
    Two-sided permutation test for difference in means under H0: Group A and Group B
    are drawn from the same underlying distribution.
    Returns (observed_diff, p_value).
    """
    obs_diff = abs(mean(group_b) - mean(group_a))
    pooled = group_a + group_b
    n_a = len(group_a)
    n_total = len(pooled)

    count_extreme = 0
    for _ in range(num_permutations):
        # In-place Fisher-Yates shuffle
        shuffled = list(pooled)
        random.shuffle(shuffled)
        perm_a = shuffled[:n_a]
        perm_b = shuffled[n_a:]
        perm_diff = abs(mean(perm_b) - mean(perm_a))
        if perm_diff >= obs_diff:
            count_extreme += 1

    p_value = (count_extreme + 1) / (num_permutations + 1)
    return obs_diff, p_value

# -----------------------------------------------------------------------------
# 2. Research Paper Formal Decomposition Schema
# -----------------------------------------------------------------------------
class ResearchPaperDecomposition:
    def __init__(
        self,
        title: str,
        problem: str,
        prior_work: str,
        hypothesis: str,
        method: str,
        math_formulation: str,
        limitations: str
    ):
        self.title = title
        self.problem = problem
        self.prior_work = prior_work
        self.hypothesis = hypothesis
        self.method = method
        self.math_formulation = math_formulation
        self.limitations = limitations

    def summary_card(self) -> str:
        return (
            f"Paper: {self.title}\n"
            f"  [1. Problem]           {self.problem}\n"
            f"  [2. Prior Work]        {self.prior_work}\n"
            f"  [3. Hypothesis]        {self.hypothesis}\n"
            f"  [4. Method]            {self.method}\n"
            f"  [5. Math Formulation]  {self.math_formulation}\n"
            f"  [6. Limitations]       {self.limitations}\n"
        )

# -----------------------------------------------------------------------------
# 3. Factorial Ablation Experiment Engine
# -----------------------------------------------------------------------------
class FactorialAblationHarness:
    """
    Simulates a 2^3 factorial experiment on architectural components:
    - Factor A: Gated Attention (0 = Standard, 1 = Gated) [Expected true effect: +4.2 points]
    - Factor B: Positional Encoding (0 = Absolute, 1 = RoPE) [Expected true effect: +2.8 points]
    - Factor C: Layer Normalization (0 = Post-LN, 1 = Pre-RMSNorm) [Expected true effect: +1.5 points]
    - Baseline performance: 70.0 points
    - Noise std: 1.2 points
    """
    def __init__(self, runs_per_config: int = 25):
        self.runs_per_config = runs_per_config
        self.baseline_perf = 70.0
        self.effects = {
            "gated_attn": 4.2,
            "rope": 2.8,
            "rmsnorm": 1.5,
            # Synergistic interaction between RoPE and Gated Attention
            "gated_rope_interaction": 0.8
        }

    def evaluate_configuration(self, gated: bool, rope: bool, rmsnorm: bool) -> List[float]:
        results = []
        for _ in range(self.runs_per_config):
            score = self.baseline_perf
            if gated: score += self.effects["gated_attn"]
            if rope: score += self.effects["rope"]
            if rmsnorm: score += self.effects["rmsnorm"]
            if gated and rope: score += self.effects["gated_rope_interaction"]

            # Add empirical experimental measurement noise
            score += random.gauss(0.0, 1.2)
            results.append(score)
        return results

    def run_full_factorial(self) -> Dict[str, Dict[str, Any]]:
        configs = [
            ("Base (No Innovations)", False, False, False),
            ("+RMSNorm", False, False, True),
            ("+RoPE", False, True, False),
            ("+GatedAttn", True, False, False),
            ("+RMSNorm + RoPE", False, True, True),
            ("+GatedAttn + RMSNorm", True, False, True),
            ("+GatedAttn + RoPE", True, True, False),
            ("Full Proposed Architecture", True, True, True),
        ]

        table = {}
        for name, g, r, norm in configs:
            scores = self.evaluate_configuration(g, r, norm)
            table[name] = {
                "gated": g,
                "rope": r,
                "rmsnorm": norm,
                "scores": scores,
                "mean": mean(scores),
                "std": std_dev(scores)
            }
        return table

# -----------------------------------------------------------------------------
# 4. Main Experiment & Verification
# -----------------------------------------------------------------------------
def run_scientific_synthesis_experiment():
    print("=" * 70)
    print("Phase 35 - Topic 05: Scientific Discovery & Literature Synthesis")
    print("=" * 70)

    # 1. Formal Paper Decomposition
    paper = ResearchPaperDecomposition(
        title="Gated Rotary Transformers with Pre-Layer Normalization",
        problem="Standard Transformers experience attention saturation and context rot over long horizons.",
        prior_work="Vaswani et al. (2017) baseline, Su et al. (2021) RoPE, Shazeer (2020) GLU variants.",
        hypothesis="H1: Gating the post-softmax attention matrix coupled with rotary embeddings significantly improves multi-step associative reasoning over baseline.",
        method="Factorial integration of elementwise sigmoid gating, rotary position embeddings, and Pre-RMSNorm.",
        math_formulation="Y = (sigma(W_g X) odot Softmax(Q K^T / sqrt(d))) V",
        limitations="Introduces 8% additional parameter overhead in projection matrices; evaluated up to 8k context."
    )
    print("\n[Step 1] Research Paper Structured Decomposition:")
    print(paper.summary_card())

    # 2. Factorial Ablation Matrix
    print("[Step 2] Executing 2^3 Factorial Ablation Matrix (25 runs per configuration)...")
    harness = FactorialAblationHarness(runs_per_config=25)
    ablation_results = harness.run_full_factorial()

    print("\n" + "-" * 75)
    print(f"{'Configuration':<30} | {'Mean Score':<12} | {'Std Dev':<10} | {'Delta vs Base':<14}")
    print("-" * 75)

    base_mean = ablation_results["Base (No Innovations)"]["mean"]
    for config_name, data in ablation_results.items():
        delta = data["mean"] - base_mean
        print(f"{config_name:<30} | {data['mean']:>10.2f}   | {data['std']:>8.2f}   | {delta:>+12.2f}")
    print("-" * 75)

    # 3. Rigorous Non-Parametric Hypothesis Testing
    print("\n[Step 3] Statistical Significance & Effect Size Testing:")
    base_scores = ablation_results["Base (No Innovations)"]["scores"]
    full_scores = ablation_results["Full Proposed Architecture"]["scores"]

    # Cohen's d
    d_val = cohens_d(base_scores, full_scores)
    # Bootstrap CI
    obs_diff, ci_low, ci_high = bootstrap_ci(base_scores, full_scores, num_bootstrap=2500)
    # Permutation Test
    _, p_value = permutation_test(base_scores, full_scores, num_permutations=2500)

    print(f"  Observed Empirical Gain:       +{obs_diff:.3f} points")
    print(f"  Standardized Effect Size (d):   {d_val:.3f} (Cohen's d > 0.8 is 'Very Large')")
    print(f"  95% Bootstrap Confidence Int:  [{ci_low:+.3f}, {ci_high:+.3f}]")
    print(f"  Two-Sided Permutation p-value:  p = {p_value:.6f}")

    # 4. Component Contribution Decomposition (Main Effects)
    print("\n[Step 4] Component Marginal Contribution Analysis:")
    gated_isolated = ablation_results["+GatedAttn"]["mean"] - base_mean
    rope_isolated = ablation_results["+RoPE"]["mean"] - base_mean
    rmsnorm_isolated = ablation_results["+RMSNorm"]["mean"] - base_mean

    print(f"  Isolated Factor A (Gated Attention): +{gated_isolated:.2f} points")
    print(f"  Isolated Factor B (RoPE Embeddings): +{rope_isolated:.2f} points")
    print(f"  Isolated Factor C (Pre-RMSNorm):     +{rmsnorm_isolated:.2f} points")

    # 5. Scientific Assertions
    # (a) Hypothesis Test: Reject H0 with p < 0.001
    assert p_value < 0.001, f"Expected p < 0.001 to reject H0, got p = {p_value:.5f}"
    # (b) Effect Size: Extremely strong effect size d > 2.0
    assert d_val > 2.0, f"Expected Cohen's d > 2.0, got d = {d_val:.3f}"
    # (c) Bootstrap CI: Strictly strictly positive gain
    assert ci_low > 5.0, f"Expected 95% CI lower bound > +5.0, got {ci_low:.3f}"
    # (d) Full architecture outscores all isolated components
    assert ablation_results["Full Proposed Architecture"]["mean"] > ablation_results["+GatedAttn"]["mean"]
    assert ablation_results["Full Proposed Architecture"]["mean"] > ablation_results["+RoPE"]["mean"]

    print("\n" + "=" * 70)
    print("ALL SCIENTIFIC DISCOVERY & HYPOTHESIS TESTING ASSERTIONS PASSED (Exit Code 0)")
    print("=" * 70)

if __name__ == "__main__":
    run_scientific_synthesis_experiment()
