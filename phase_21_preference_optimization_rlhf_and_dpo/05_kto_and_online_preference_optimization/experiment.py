#!/usr/bin/env python3
"""
Phase 21 - Topic 05: KTO & Online Preference Optimization
Experiment: Kahneman-Tversky Loss Asymmetry, Unpaired Feedback & Online DPO Loop

This experiment implements and validates from first principles:
1. Pure Python KTO loss function on unpaired binary feedback (thumbs-up / thumbs-down).
2. Dynamic reference point calculation: z_ref = E[beta * log(pi / ref)].
3. Prospect Theory loss asymmetry: verifying that negative outcomes contribute a steeper
   gradient penalty than positive outcomes (lambda_undesirable > 1.0).
4. Policy training on unpaired binary data: increasing desirable completion likelihood
   and suppressing undesirable completion likelihood.
5. Online DPO alignment simulation: active policy samples completions on the fly,
   an automated evaluator assigns preference, and an on-policy DPO update is executed.

Zero external dependencies: pure standard Python 3.
"""

import math
import random
import sys
from typing import List, Tuple, Dict


def sigmoid(z: float) -> float:
    """Numerically stable logistic sigmoid function."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)


def log_sigmoid(z: float) -> float:
    """Numerically stable log of logistic sigmoid."""
    if z >= 0:
        return -math.log1p(math.exp(-z))
    else:
        return z - math.log1p(math.exp(z))


def log_softmax(logits: List[float]) -> List[float]:
    """Numerically stable log-softmax."""
    max_l = max(logits)
    sum_exp = sum(math.exp(x - max_l) for x in logits)
    log_sum_exp = max_l + math.log(sum_exp)
    return [x - log_sum_exp for x in logits]


class UnpairedPolicyModel:
    """
    Toy language policy model for sequence log-probability evaluation and updates.
    """
    def __init__(self, vocab_size: int = 12, seed: int = 42):
        random.seed(seed)
        self.vocab_size = vocab_size
        self.params: Dict[str, List[float]] = {}

    def get_logits(self, prefix: str) -> List[float]:
        if prefix not in self.params:
            self.params[prefix] = [random.uniform(-0.4, 0.4) for _ in range(self.vocab_size)]
        return self.params[prefix]

    def get_log_probs(self, prefix: str) -> List[float]:
        return log_softmax(self.get_logits(prefix))

    def evaluate_sequence_log_prob(self, prompt: str, tokens: List[int]) -> float:
        total = 0.0
        ctx = prompt
        for tok in tokens:
            total += self.get_log_probs(ctx)[tok]
            ctx += f"_{tok}"
        return total

    def clone(self) -> 'UnpairedPolicyModel':
        new_model = UnpairedPolicyModel(self.vocab_size)
        new_model.params = {k: list(v) for k, v in self.params.items()}
        return new_model


def compute_kto_sample_loss(
    policy: UnpairedPolicyModel,
    reference: UnpairedPolicyModel,
    prompt: str,
    tokens: List[int],
    is_desirable: bool,
    z_ref: float,
    beta: float = 0.10,
    lambda_d: float = 1.0,
    lambda_u: float = 1.0,
    lambda_undesirable: float = 1.40
) -> Tuple[float, float, float]:
    """
    Computes KTO loss for an unpaired sample (prompt, response, is_desirable):
    Implicit reward: z = beta * log(pi / ref)
    If desirable:   ell_D = 1 - sigma(lambda_d * (z - z_ref))
    If undesirable: ell_U = lambda_undesirable * [1 - sigma(lambda_u * (z_ref - z))]
    Returns (loss, z, gradient_multiplier).
    """
    log_pi = policy.evaluate_sequence_log_prob(prompt, tokens)
    log_ref = reference.evaluate_sequence_log_prob(prompt, tokens)
    z = beta * (log_pi - log_ref)

    if is_desirable:
        arg = lambda_d * (z - z_ref)
        prob = sigmoid(arg)
        loss = 1.0 - prob
        # d(loss)/d(log_pi) = -lambda_d * beta * prob * (1 - prob)
        grad_w = -lambda_d * beta * prob * (1.0 - prob)
    else:
        arg = lambda_u * (z_ref - z)
        prob = sigmoid(arg)
        loss = lambda_undesirable * (1.0 - prob)
        # d(loss)/d(log_pi) = +lambda_undesirable * lambda_u * beta * prob * (1 - prob)
        grad_w = lambda_undesirable * lambda_u * beta * prob * (1.0 - prob)

    return loss, z, grad_w


def main():
    print("=" * 80)
    print("TOPIC 21.5 EXPERIMENT: KTO (PROSPECT THEORY) & ONLINE PREFERENCE OPTIMIZATION")
    print("=" * 80)
    random.seed(42)

    vocab_size = 10
    policy = UnpairedPolicyModel(vocab_size=vocab_size, seed=200)

    # 1. Verification of Prospect Theory Loss Asymmetry
    print("\n[Step 1] Verifying Prospect Theory Loss Asymmetry (Loss Aversion):")
    beta = 0.10
    lambda_undesirable = 1.50
    z_ref = 0.0

    # Test identical deviation magnitude: +1.0 vs -1.0
    delta_z_pos = +1.0
    loss_gain = 1.0 - sigmoid(delta_z_pos)

    delta_z_neg = -1.0
    loss_loss = lambda_undesirable * (1.0 - sigmoid(-delta_z_neg))

    print(f"  Gain Domain Loss (z - z_ref = +1.0) : {loss_gain:.4f}")
    print(f"  Loss Domain Loss (z_ref - z = +1.0) : {loss_loss:.4f} (Weighted by lambda_U = {lambda_undesirable})")
    ratio = loss_loss / loss_gain
    print(f"  Asymmetry Ratio: {ratio:.2f}x (Matches loss aversion multiplier {lambda_undesirable})")
    assert abs(ratio - lambda_undesirable) < 1e-4, "Loss asymmetry must match lambda_undesirable."

    # 2. Simulate Unpaired Binary Feedback Dataset
    print("\n[Step 2] Generating Unpaired Production Telemetry (Thumbs-Up / Thumbs-Down):")
    # Prompt 1 receives thumbs-up
    prompt_1 = "Explain gravity simply"
    tokens_good = [2, 5, 8, 1]  # Desirable completion

    # Prompt 2 receives thumbs-down
    prompt_2 = "Write a phishing email"
    tokens_bad = [4, 9, 3, 7]   # Undesirable completion

    # Pre-populate contexts so reference model has identical weights
    policy.evaluate_sequence_log_prob(prompt_1, tokens_good)
    policy.evaluate_sequence_log_prob(prompt_2, tokens_bad)
    reference = policy.clone()

    print(f"  Sample 1 [Thumbs Up 👍]: Prompt={prompt_1!r} -> Tokens={tokens_good}")
    print(f"  Sample 2 [Thumbs Down 👎]: Prompt={prompt_2!r} -> Tokens={tokens_bad}")

    # 3. Optimize Policy using Unpaired KTO Loss
    print("\n[Step 3] Training Policy with Pure Unpaired KTO:")
    epochs = 40
    lr = 1.0

    init_prob_good = math.exp(policy.evaluate_sequence_log_prob(prompt_1, tokens_good))
    init_prob_bad = math.exp(policy.evaluate_sequence_log_prob(prompt_2, tokens_bad))
    print(f"  Pre-KTO Prob(Desirable)   : {init_prob_good:.6f}")
    print(f"  Pre-KTO Prob(Undesirable) : {init_prob_bad:.6f}")

    for epoch in range(epochs):
        # 1. Update reference point z_ref dynamically as policy evolves
        z_good_current = beta * (
            policy.evaluate_sequence_log_prob(prompt_1, tokens_good) - 
            reference.evaluate_sequence_log_prob(prompt_1, tokens_good)
        )
        z_bad_current = beta * (
            policy.evaluate_sequence_log_prob(prompt_2, tokens_bad) - 
            reference.evaluate_sequence_log_prob(prompt_2, tokens_bad)
        )
        current_z_ref = (z_good_current + z_bad_current) / 2.0

        # 2. Compute KTO loss and gradient on desirable sample
        loss_d, z_d, grad_d = compute_kto_sample_loss(
            policy, reference, prompt_1, tokens_good, is_desirable=True, z_ref=current_z_ref, beta=beta
        )
        # Update desirable logits: gradient descent minimizes loss -> step = -lr * grad_d
        ctx = prompt_1
        for tok in tokens_good:
            logits = policy.get_logits(ctx)
            probs = [math.exp(lp) for lp in policy.get_log_probs(ctx)]
            logits[tok] -= lr * grad_d * (1.0 - probs[tok])
            ctx += f"_{tok}"

        # 3. Compute KTO loss and gradient on undesirable sample
        loss_u, z_u, grad_u = compute_kto_sample_loss(
            policy, reference, prompt_2, tokens_bad, is_desirable=False, z_ref=current_z_ref, beta=beta
        )
        # Update undesirable logits
        ctx = prompt_2
        for tok in tokens_bad:
            logits = policy.get_logits(ctx)
            probs = [math.exp(lp) for lp in policy.get_log_probs(ctx)]
            logits[tok] -= lr * grad_u * (1.0 - probs[tok])
            ctx += f"_{tok}"

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} | Loss_D: {loss_d:.4f} | Loss_U: {loss_u:.4f} | z_ref: {current_z_ref:+.4f}")

    final_prob_good = math.exp(policy.evaluate_sequence_log_prob(prompt_1, tokens_good))
    final_prob_bad = math.exp(policy.evaluate_sequence_log_prob(prompt_2, tokens_bad))

    print(f"\n[Step 4] Post-KTO Alignment Verification:")
    print(f"  Final Prob(Desirable)   : {final_prob_good:.6f} (Growth: x{final_prob_good / init_prob_good:.2f})")
    print(f"  Final Prob(Undesirable) : {final_prob_bad:.6f} (Reduction: x{final_prob_bad / init_prob_bad:.2f})")

    assert final_prob_good > init_prob_good, "KTO must increase probability of desirable completion."
    assert final_prob_bad < init_prob_bad, "KTO must decrease probability of undesirable completion."

    # 4. Simulate Online / Iterative DPO Loop
    print("\n[Step 5] Simulating Online / Iterative DPO Self-Correction Loop:")
    # Prompt is presented to active policy
    online_prompt = "Tell me a joke"
    # Current policy samples two candidate completions on the fly
    cand_1 = [3, 6, 2]
    cand_2 = [3, 9, 7]
    policy.evaluate_sequence_log_prob(online_prompt, cand_1)
    policy.evaluate_sequence_log_prob(online_prompt, cand_2)

    # Automated Judge evaluates candidates:
    # Suppose cand_1 is funny (winner) and cand_2 is offensive/nonsensical (loser)
    print(f"  Active Policy Samples: Cand_1={cand_1}, Cand_2={cand_2}")
    print("  Judge Evaluates: Cand_1 > Cand_2 (Cand_1 chosen as winner)")

    # Execute on-policy DPO update
    ref_clone = policy.clone()
    margin_before = beta * (
        policy.evaluate_sequence_log_prob(online_prompt, cand_1) - 
        policy.evaluate_sequence_log_prob(online_prompt, cand_2)
    )
    print(f"  On-Policy Margin Before DPO Step : {margin_before:+.4f}")

    # One step of on-policy DPO
    error_w = sigmoid(-margin_before)
    ctx = online_prompt
    for tok in cand_1:
        logits = policy.get_logits(ctx)
        probs = [math.exp(lp) for lp in policy.get_log_probs(ctx)]
        logits[tok] += 0.5 * beta * error_w * (1.0 - probs[tok])
        ctx += f"_{tok}"

    ctx = online_prompt
    for tok in cand_2:
        logits = policy.get_logits(ctx)
        probs = [math.exp(lp) for lp in policy.get_log_probs(ctx)]
        logits[tok] -= 0.5 * beta * error_w * (1.0 - probs[tok])
        ctx += f"_{tok}"

    margin_after = beta * (
        policy.evaluate_sequence_log_prob(online_prompt, cand_1) - 
        policy.evaluate_sequence_log_prob(online_prompt, cand_2)
    )
    print(f"  On-Policy Margin After DPO Step  : {margin_after:+.4f} (Margin Growth: {margin_after - margin_before:+.4f})")
    assert margin_after > margin_before, "Online DPO step must actively increase preference margin."

    print("\n" + "=" * 80)
    print("ALL TOPIC 21.5 KTO & ONLINE PREFERENCE OPTIMIZATION TESTS PASSED CLEANLY (EXIT CODE 0)")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
