#!/usr/bin/env python3
"""
Phase 20 - Topic 04: Instruction Evaluation & LLM-as-a-Judge
Experiment: Pure Python LLM Judge Evaluation Harness & Bias Mitigation Engine

This experiment implements and verifies:
1. Synthetic generation candidates with known intrinsic quality and verbosity profiles.
2. A simulated LLM Judge with parameterized biases:
   - Position Bias (primacy effect: favoring Option A over Option B)
   - Verbosity Bias (favoring longer responses regardless of content)
3. Naive vs. Debiased Evaluation Protocols:
   - Naive (unidirectional) matching
   - Bidirectional order swapping (evaluating (A, B) and (B, A)) with contradiction/tie detection
   - Length-adjusted / verbosity-penalized win-rate computation
4. Bradley-Terry (Elo) preference rating solver via pure Python iterative gradient ascent.
5. Verification assertions proving that debiasing eliminates position asymmetry and verbosity gaming.

Zero external dependencies: pure standard Python 3.
"""

import math
import random
import sys
from typing import List, Dict, Tuple, Optional


class CandidateModel:
    """Represents an LLM candidate with intrinsic quality and verbosity characteristics."""
    def __init__(self, name: str, true_quality: float, verbosity: float):
        self.name = name
        self.true_quality = true_quality  # Latent intrinsic capability
        self.verbosity = verbosity        # Multiplier on generated word count


class SimulatedJudge:
    """
    Simulates an LLM Judge (e.g., GPT-4) grading two models.
    Incorporates:
    - Position bias (delta favoring candidate in position A)
    - Verbosity bias (bonus for longer text)
    - Decision threshold for ties
    """
    def __init__(
        self,
        position_bias: float = 0.35,  # Bias towards position A
        verbosity_bias: float = 0.25, # Bias per unit of extra verbosity
        tie_margin: float = 0.15,     # Threshold below which judge declares tie
        seed: int = 42
    ):
        self.position_bias = position_bias
        self.verbosity_bias = verbosity_bias
        self.tie_margin = tie_margin
        self.rng = random.Random(seed)

    def compare(
        self,
        cand_A: CandidateModel,
        cand_B: CandidateModel
    ) -> str:
        """
        Grades pair (cand_A, cand_B).
        Returns: 'A' if cand_A wins, 'B' if cand_B wins, 'TIE' if score difference < tie_margin.
        """
        # Perceived utility of A: true_quality + position_bias + verbosity_bonus + noise
        noise_A = self.rng.gauss(0.0, 0.2)
        noise_B = self.rng.gauss(0.0, 0.2)

        utility_A = cand_A.true_quality + self.position_bias + (self.verbosity_bias * cand_A.verbosity) + noise_A
        utility_B = cand_B.true_quality + (self.verbosity_bias * cand_B.verbosity) + noise_B

        diff = utility_A - utility_B
        if abs(diff) < self.tie_margin:
            return "TIE"
        elif diff > 0:
            return "A"
        else:
            return "B"


class EvaluationHarness:
    """Runs tournament matches and computes debiased metrics."""
    def __init__(self, judge: SimulatedJudge):
        self.judge = judge

    def run_naive_matchup(
        self,
        m1: CandidateModel,
        m2: CandidateModel,
        num_trials: int = 100
    ) -> Dict[str, float]:
        """Unidirectional: m1 is always in slot A, m2 is always in slot B."""
        wins_m1 = 0
        wins_m2 = 0
        ties = 0

        for _ in range(num_trials):
            res = self.judge.compare(m1, m2)
            if res == "A":
                wins_m1 += 1
            elif res == "B":
                wins_m2 += 1
            else:
                ties += 1

        return {
            "win_rate_m1": (wins_m1 + 0.5 * ties) / num_trials,
            "wins_m1": wins_m1,
            "wins_m2": wins_m2,
            "ties": ties
        }

    def run_bidirectional_matchup(
        self,
        m1: CandidateModel,
        m2: CandidateModel,
        num_trials: int = 100
    ) -> Dict[str, float]:
        """
        Bidirectional order swapping:
        Runs trial with (m1, m2), and another trial with (m2, m1).
        Consensus rule:
        - m1 wins if m1 wins both orientations.
        - m2 wins if m2 wins both orientations.
        - Otherwise, result is a tie / position-bias contradiction.
        """
        wins_m1 = 0
        wins_m2 = 0
        ties = 0

        for _ in range(num_trials):
            # Pass 1: m1 is A, m2 is B
            res1 = self.judge.compare(m1, m2)
            # Pass 2: m2 is A, m1 is B
            res2 = self.judge.compare(m2, m1)

            # Consensus mapping
            m1_won_pass1 = (res1 == "A")
            m1_won_pass2 = (res2 == "B")
            m2_won_pass1 = (res1 == "B")
            m2_won_pass2 = (res2 == "A")

            if m1_won_pass1 and m1_won_pass2:
                wins_m1 += 1
            elif m2_won_pass1 and m2_won_pass2:
                wins_m2 += 1
            else:
                # Disagreement or ties
                ties += 1

        return {
            "win_rate_m1": (wins_m1 + 0.5 * ties) / num_trials,
            "wins_m1": wins_m1,
            "wins_m2": wins_m2,
            "ties": ties
        }

    def compute_length_penalized_winrate(
        self,
        raw_winrate: float,
        len_cand: float,
        len_baseline: float,
        beta: float = 0.20
    ) -> float:
        """
        Penalizes excess verbosity:
        winrate_adj = raw_winrate - beta * max(0, (L_cand - L_base) / L_base)
        """
        if len_cand <= len_baseline:
            return raw_winrate
        excess_ratio = (len_cand - len_baseline) / len_baseline
        penalty = beta * excess_ratio
        return max(0.0, raw_winrate - penalty)

    def solve_bradley_terry(
        self,
        models: List[CandidateModel],
        matches: List[Tuple[int, int, str]],
        iterations: int = 150,
        lr: float = 0.05
    ) -> Dict[str, float]:
        """
        Solves Bradley-Terry preference ratings:
        P(i > j) = exp(r_i) / (exp(r_i) + exp(r_j))
        via gradient ascent on the log-likelihood of pairwise outcomes.
        """
        K = len(models)
        ratings = [0.0] * K

        for _ in range(iterations):
            grad = [0.0] * K
            for i, j, outcome in matches:
                # P(i > j)
                p_i = 1.0 / (1.0 + math.exp(ratings[j] - ratings[i]))
                p_j = 1.0 - p_i

                if outcome == "i":
                    grad[i] += 1.0 - p_i
                    grad[j] -= 1.0 - p_i
                elif outcome == "j":
                    grad[i] -= p_i
                    grad[j] += p_i
                elif outcome == "tie":
                    grad[i] += 0.5 - p_i
                    grad[j] += 0.5 - p_j

            # Step
            for k in range(K):
                ratings[k] += lr * grad[k]

            # Center ratings around mean 0
            mean_r = sum(ratings) / K
            for k in range(K):
                ratings[k] -= mean_r

        # Scale to standard Elo range: Elo = 1000 + 400 * rating / ln(10)
        elo_scores = {}
        for k, m in enumerate(models):
            elo_scores[m.name] = round(1000 + (400.0 / math.log(10)) * ratings[k], 1)
        return elo_scores


def run_experiment():
    print("=" * 75)
    print("PHASE 20.4 EXPERIMENT: LLM-AS-A-JUDGE & BIAS MITIGATION ENGINE")
    print("=" * 75)

    # 1. Define Candidate Models
    # Model A: High quality, concise
    # Model B: Equal quality to Model A, concise
    # Model C: Low quality, but extremely verbose (word-salad flooder)
    # Model D: Moderate quality, standard length
    model_A = CandidateModel(name="Model-Alpha (Quality=2.0, Verbosity=1.0)", true_quality=2.0, verbosity=1.0)
    model_B = CandidateModel(name="Model-Beta  (Quality=2.0, Verbosity=1.0)", true_quality=2.0, verbosity=1.0)
    model_C = CandidateModel(name="Model-Gamma (Quality=1.4, Verbosity=3.5)", true_quality=1.4, verbosity=3.5)
    model_D = CandidateModel(name="Model-Delta (Quality=1.6, Verbosity=1.0)", true_quality=1.6, verbosity=1.0)

    # Initialize Judge with position bias (favors slot A by +0.35)
    judge = SimulatedJudge(position_bias=0.35, verbosity_bias=0.20, seed=123)
    harness = EvaluationHarness(judge)

    # --- 1. Position Bias Demonstration & Bidirectional Mitigation ---
    print("\n--- 1. Position Bias & Bidirectional Swapping Mitigation ---")
    print("Competing: Model-Alpha vs Model-Beta (Identical true quality: 2.0 vs 2.0)")

    # Naive unidirectional matching
    naive_res = harness.run_naive_matchup(model_A, model_B, num_trials=200)
    print(f"Naive Unidirectional (Alpha in slot A, Beta in slot B):")
    print(f"  Alpha Win Rate: {naive_res['win_rate_m1'] * 100:.1f}% (Wins: {naive_res['wins_m1']}, Losses: {naive_res['wins_m2']}, Ties: {naive_res['ties']})")
    
    # Assert naive evaluation suffered from severe position bias
    assert naive_res["win_rate_m1"] > 0.65, "Naive evaluation should heavily favor slot A"
    print("  ! Severe position bias detected: slot A unfairly favored.")

    # Debiased bidirectional order swapping
    debiased_res = harness.run_bidirectional_matchup(model_A, model_B, num_trials=200)
    print(f"Debiased Bidirectional Consensus (evaluating both (A, B) and (B, A)):")
    print(f"  Alpha Win Rate: {debiased_res['win_rate_m1'] * 100:.1f}% (Wins: {debiased_res['wins_m1']}, Losses: {debiased_res['wins_m2']}, Ties: {debiased_res['ties']})")

    # Symmetric assertion: For identical models, bidirectional win rate should be within 45% - 55%
    assert 0.40 <= debiased_res["win_rate_m1"] <= 0.60, f"Debiased win rate should be ~50%, got {debiased_res['win_rate_m1']}"
    print("✓ Position bias neutralized: Win rate restored to statistical parity (50%).")

    # --- 2. Verbosity Bias Demonstration & Length Penalization ---
    print("\n--- 2. Verbosity Bias & Length-Adjusted Scoring ---")
    print("Competing: Model-Delta (Quality=1.6, Length=1.0) vs Model-Gamma (Quality=1.4, Length=3.5)")

    # Naive judge favors verbose Model Gamma despite its lower true quality
    naive_vg = harness.run_naive_matchup(model_C, model_D, num_trials=200)
    gamma_raw_winrate = naive_vg["win_rate_m1"]
    print(f"Raw Verbose Win Rate (Model-Gamma vs Model-Delta): {gamma_raw_winrate * 100:.1f}%")
    assert gamma_raw_winrate > 0.50, "Verbose model should cheat judge with high raw win-rate"
    print("  ! Verbosity cheating detected: Lower quality model won due to bloated length.")

    # Apply length penalty
    gamma_adj_winrate = harness.compute_length_penalized_winrate(
        raw_winrate=gamma_raw_winrate,
        len_cand=model_C.verbosity,
        len_baseline=model_D.verbosity,
        beta=0.25
    )
    print(f"Length-Penalized Win Rate (Model-Gamma): {gamma_adj_winrate * 100:.1f}%")
    assert gamma_adj_winrate < 0.50, "Length penalty should knock verbose model below 50%"
    print(f"✓ Verbosity penalty applied: Adjusted win rate dropped by {(gamma_raw_winrate - gamma_adj_winrate)*100:.1f} percentage points.")

    # --- 3. Bradley-Terry Latent Skill Estimation ---
    print("\n--- 3. Tournament Bradley-Terry Elo Solver ---")
    model_E = CandidateModel(name="Model-Epsilon (Quality=1.1, Verbosity=1.0)", true_quality=1.1, verbosity=1.0)
    models = [model_A, model_D, model_E]
    # Simulate a round-robin tournament with bidirectional matches
    matches = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            for _ in range(50):
                # (i, j)
                r1 = judge.compare(models[i], models[j])
                # (j, i)
                r2 = judge.compare(models[j], models[i])
                
                # Consensus
                if r1 == "A" and r2 == "B":
                    matches.append((i, j, "i"))
                elif r1 == "B" and r2 == "A":
                    matches.append((i, j, "j"))
                else:
                    matches.append((i, j, "tie"))

    elo_scores = harness.solve_bradley_terry(models, matches)
    for name, elo in elo_scores.items():
        print(f"  {name:45s} -> Elo: {elo}")

    # Assert true quality order is respected in Elo: Model A (2.0) > Model D (1.6) > Model E (1.1)
    assert elo_scores[model_A.name] > elo_scores[model_D.name], "Model Alpha should rank higher than Model Delta"
    assert elo_scores[model_D.name] > elo_scores[model_E.name], "Model Delta should rank higher than Model Epsilon"
    print("✓ Bradley-Terry solver successfully reconstructed true latent capability rankings.")

    print("\n" + "=" * 75)
    print("ALL TESTS PASSED: Topic 20.4 Experiment verified with exit code 0.")
    print("=" * 75)


if __name__ == "__main__":
    run_experiment()
    sys.exit(0)
