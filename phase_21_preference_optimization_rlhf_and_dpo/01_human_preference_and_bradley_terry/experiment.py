#!/usr/bin/env python3
"""
Phase 21 - Topic 01: Human Preference Modeling & The Bradley-Terry Framework
Experiment: Bradley-Terry Maximum Likelihood Estimator & Transitivity Verification

This experiment verifies from first principles:
1. Pure Python Bradley-Terry logistic link function P(y_w > y_l) = sigma(r_w - r_l).
2. Synthetic ground-truth latent reward generation across candidate LLM responses.
3. Stochastic comparison dataset generation simulating human pairwise judgments.
4. Numerical Maximum Likelihood Estimation (MLE) of latent rewards using gradient ascent.
5. Invariance to translation gauge freedom and score centering (mean = 0).
6. Rank recovery validation (Kendall's Tau / Spearman's rank correlation).
7. Formal verification of Strong Stochastic Transitivity (SST).
8. Condorcet cycle resolution (handling intransitive A > B > C > A loops).

Zero external dependencies: pure standard Python 3.
"""

import math
import random
import sys
from typing import List, Tuple, Dict


def sigmoid(z: float) -> float:
    """Numerically stable standard logistic sigmoid function."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)


def log_sigmoid(z: float) -> float:
    """Numerically stable log of logistic sigmoid: log(1 / (1 + exp(-z)))."""
    if z >= 0:
        return -math.log1p(math.exp(-z))
    else:
        return z - math.log1p(math.exp(z))


class BradleyTerryModel:
    """
    Bradley-Terry Maximum Likelihood Estimator for latent reward estimation
    from pairwise comparisons.
    """
    def __init__(self, item_names: List[str]):
        self.item_names = list(item_names)
        self.n_items = len(item_names)
        self.name_to_idx = {name: i for i, name in enumerate(item_names)}
        self.rewards = [0.0] * self.n_items

    def predict_prob(self, idx_a: int, idx_b: int) -> float:
        """Computes P(A > B) = sigma(r_a - r_b)."""
        delta = self.rewards[idx_a] - self.rewards[idx_b]
        return sigmoid(delta)

    def compute_nll(self, comparisons: List[Tuple[int, int]]) -> float:
        """
        Computes negative log-likelihood over pairs (winner, loser):
        NLL = - sum_{(w, l)} log sigma(r_w - r_l)
        """
        nll = 0.0
        for w, l in comparisons:
            delta = self.rewards[w] - self.rewards[l]
            nll -= log_sigmoid(delta)
        return nll

    def fit(self, comparisons: List[Tuple[int, int]], lr: float = 0.1, epochs: int = 500, verbose: bool = False) -> List[float]:
        """
        Fits latent rewards using gradient ascent on log-likelihood with zero-centering constraint.
        """
        loss_history = []
        for epoch in range(epochs):
            # Compute gradients: dL/dr_w = sigma(r_l - r_w), dL/dr_l = -sigma(r_l - r_w)
            grad = [0.0] * self.n_items
            for w, l in comparisons:
                delta = self.rewards[w] - self.rewards[l]
                prob_loser_beats_winner = sigmoid(-delta)  # 1 - sigma(delta)
                grad[w] += prob_loser_beats_winner
                grad[l] -= prob_loser_beats_winner

            # Gradient ascent step
            for i in range(self.n_items):
                self.rewards[i] += lr * grad[i]

            # Enforce centering constraint: mean(rewards) = 0 (gauge condition)
            mean_reward = sum(self.rewards) / self.n_items
            for i in range(self.n_items):
                self.rewards[i] -= mean_reward

            nll = self.compute_nll(comparisons)
            loss_history.append(nll)
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch:4d} | NLL: {nll:.4f} | Rewards: {[round(r, 3) for r in self.rewards]}")

        return loss_history


def spearman_rank_correlation(x: List[float], y: List[float]) -> float:
    """Computes Spearman's rank correlation between two scalar lists."""
    def get_ranks(seq: List[float]) -> List[float]:
        indexed = sorted(enumerate(seq), key=lambda item: item[1])
        ranks = [0.0] * len(seq)
        for rank, (original_idx, _) in enumerate(indexed):
            ranks[original_idx] = float(rank + 1)
        return ranks

    n = len(x)
    if n <= 1:
        return 1.0
    rx = get_ranks(x)
    ry = get_ranks(y)
    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - (6.0 * d_sq) / (n * (n ** 2 - 1))


def main():
    print("=" * 80)
    print("TOPIC 21.1 EXPERIMENT: BRADLEY-TERRY PREFERENCE MODEL & MLE REWARD SOLVER")
    print("=" * 80)
    random.seed(42)

    # 1. Define synthetic LLM completions with ground-truth latent rewards
    models = ["Llama-3-70B-Aligned", "Claude-3.5-Sonnet", "GPT-4o", "Mistral-Large", "Base-LLM-UnfineTuned"]
    ground_truth_rewards = [1.5, 2.6, 2.0, 0.4, -3.5]
    # Zero-center the true rewards for direct comparability
    mean_gt = sum(ground_truth_rewards) / len(ground_truth_rewards)
    ground_truth_rewards = [r - mean_gt for r in ground_truth_rewards]

    print("\n[Step 1] Ground-Truth Latent Rewards (Centered):")
    for name, r in zip(models, ground_truth_rewards):
        print(f"  - {name:<26}: {r:+.3f}")

    # 2. Simulate human pairwise comparisons according to Bradley-Terry probabilities
    print("\n[Step 2] Generating Pairwise Judgments via Bradley-Terry Distribution...")
    comparisons = []
    n_comparisons_per_pair = 300
    n = len(models)
    
    for i in range(n):
        for j in range(i + 1, n):
            r_i = ground_truth_rewards[i]
            r_j = ground_truth_rewards[j]
            p_i_beats_j = sigmoid(r_i - r_j)
            
            for _ in range(n_comparisons_per_pair):
                if random.random() < p_i_beats_j:
                    comparisons.append((i, j))  # i won
                else:
                    comparisons.append((j, i))  # j won

    print(f"  Generated {len(comparisons)} synthetic comparison matchups across {n * (n - 1) // 2} pairs.")

    # 3. Fit Bradley-Terry Model via Gradient Ascent on Log-Likelihood
    print("\n[Step 3] Optimizing Latent Rewards with Bradley-Terry MLE:")
    bt_model = BradleyTerryModel(models)
    initial_nll = bt_model.compute_nll(comparisons)
    print(f"  Initial NLL (all rewards = 0.0): {initial_nll:.4f}")
    
    losses = bt_model.fit(comparisons, lr=0.005, epochs=600, verbose=True)
    final_nll = losses[-1]
    print(f"  Final NLL after convergence: {final_nll:.4f} (Reduction: {initial_nll - final_nll:.4f})")
    assert final_nll < initial_nll, "Optimization must strictly decrease NLL."

    # 4. Compare Estimated Rewards vs Ground Truth
    print("\n[Step 4] Recovered Rewards vs. Ground Truth:")
    for name, gt, est in zip(models, ground_truth_rewards, bt_model.rewards):
        print(f"  - {name:<26} | True: {gt:+.3f} | Estimated: {est:+.3f} | Error: {abs(gt - est):.3f}")
        assert abs(gt - est) < 0.35, f"Estimated reward {est} diverges too far from true reward {gt}."

    rho = spearman_rank_correlation(ground_truth_rewards, bt_model.rewards)
    print(f"  Spearman's Rank Correlation: {rho:.4f}")
    assert rho > 0.95, f"Rank correlation must exceed 0.95, got {rho}."

    # 5. Formal Verification of Strong Stochastic Transitivity (SST)
    print("\n[Step 5] Verifying Strong Stochastic Transitivity (SST):")
    # For all triples (A, B, C) where P(A > B) >= 0.5 and P(B > C) >= 0.5:
    # SST requires: P(A > C) >= max(P(A > B), P(B > C))
    sst_verified = 0
    sst_checked = 0
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            for c in range(n):
                if c == a or c == b:
                    continue
                p_ab = bt_model.predict_prob(a, b)
                p_bc = bt_model.predict_prob(b, c)
                if p_ab >= 0.5 and p_bc >= 0.5:
                    p_ac = bt_model.predict_prob(a, c)
                    sst_checked += 1
                    max_prob = max(p_ab, p_bc)
                    assert p_ac >= max_prob - 1e-9, f"SST violated for triple {a}, {b}, {c}: {p_ac} < {max_prob}"
                    sst_verified += 1

    print(f"  Checked {sst_checked} transitive triples. SST satisfied in {sst_verified}/{sst_checked} instances (100.0%).")

    # 6. Condorcet Paradox & Cyclical Preference Handling
    print("\n[Step 6] Verifying Bradley-Terry Behavior on Condorcet Preference Cycles:")
    cycle_items = ["Response_A", "Response_B", "Response_C"]
    cycle_bt = BradleyTerryModel(cycle_items)
    # Perfectly symmetric cycle: A > B, B > C, C > A (100 matchups each)
    cycle_comparisons = (
        [(0, 1)] * 100 +  # A > B
        [(1, 2)] * 100 +  # B > C
        [(2, 0)] * 100    # C > A
    )
    cycle_bt.fit(cycle_comparisons, lr=0.01, epochs=300, verbose=False)
    print(f"  Recovered Rewards on Cyclic Dataset: {[round(r, 4) for r in cycle_bt.rewards]}")
    for r in cycle_bt.rewards:
        assert abs(r) < 1e-3, f"Expected reward ~ 0.0 for cyclic items, got {r}."
    print("  Successfully confirmed: Bradley-Terry resolves cyclical feedback with a neutral equilibrium tie!")

    print("\n" + "=" * 80)
    print("ALL TOPIC 21.1 PREFERENCE & BRADLEY-TERRY TESTS PASSED CLEANLY (EXIT CODE 0)")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
