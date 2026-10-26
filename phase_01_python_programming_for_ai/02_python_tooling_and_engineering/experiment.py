#!/usr/bin/env python3
"""
EXPERIMENT 1.2: AI SOFTWARE ENGINEERING, PROFILING & REPRODUCIBILITY
===================================================================
A hands-on, zero-dependency Python laboratory demonstrating:
1. Ironclad Seed Reproducibility (deterministic random initialization)
2. In-Memory Execution Profiling using cProfile & pstats
3. Professional Multi-Level Logging vs print()
4. Automated In-Script ML Unit Test Suite (unittest.TestCase)
5. Command-Line Argument Parsing (argparse) simulation

Runs on standard library Python 3 without external packages.
"""

import math
import random
import time
import io
import cProfile
import pstats
import logging
import unittest
import argparse

print("=" * 70)
print("EXPERIMENT 1.2: AI SOFTWARE ENGINEERING & REPRODUCIBILITY")
print("=" * 70)

# ======================================================================
# 1. SCIENTIFIC REPRODUCIBILITY & SEED CONTROL
# ======================================================================
print("\n--- 1. DETERMINISTIC RANDOM SEED CONTROL ---")

def generate_random_weights(seed: int, count: int = 5) -> list[float]:
    random.seed(seed)
    return [round(random.uniform(-1.0, 1.0), 4) for _ in range(count)]

weights_run1 = generate_random_weights(seed=1337)
weights_run2 = generate_random_weights(seed=1337)
weights_diff = generate_random_weights(seed=9999)

print(f"Run 1 (Seed 1337): {weights_run1}")
print(f"Run 2 (Seed 1337): {weights_run2}")
print(f"Run 3 (Seed 9999): {weights_diff}")

assert weights_run1 == weights_run2, "❌ Determinism failed! Seeds did not produce identical weights."
print("✅ Seed verification passed: Run 1 and Run 2 produced bit-for-bit identical weights!")


# ======================================================================
# 2. IN-MEMORY PROFILING WITH CPROFILE & PSTATS
# ======================================================================
print("\n" + "=" * 70)
print("--- 2. BOTTLENECK PROFILING WITH cProfile ---")

def slow_nested_lookup(n: int = 20_000):
    # Simulates an unoptimized, quadratic data search
    items = list(range(n))
    evens = [x for x in items if x % 2 == 0]
    # Set lookup is O(1) vs list lookup O(N)
    lookup_set = set(evens)
    hits = sum(1 for i in range(1000) if i in lookup_set)
    return hits

profiler = cProfile.Profile()
profiler.enable()

# Execute code to profile
result_hits = slow_nested_lookup()

profiler.disable()
stream = io.StringIO()
ps = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
ps.print_stats(5)

print(f"Function executed with {result_hits} hits. Top cumulative profile stats:")
print("\n".join(stream.getvalue().splitlines()[:12]))


# ======================================================================
# 3. PROFESSIONAL LOGGING CONFIGURATION
# ======================================================================
print("=" * 70)
print("--- 3. MULTI-LEVEL LOGGING ENGINE ---")

logger = logging.getLogger("AI_Engine")
logger.setLevel(logging.DEBUG)

# Create console handler with custom formatting
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("  %(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info("Initializing Transformer architecture: 12 layers, 768 hidden dim.")
logger.debug("Allocating cache buffers in GPU memory block 0x7fff400...")
logger.warning("Gradient norm exceeded threshold 5.0 (clipping applied).")
logger.info("Epoch 1/10 completed. Loss: 0.2841 | Val Acc: 91.4%")


# ======================================================================
# 4. IN-SCRIPT AUTOMATED ML UNIT TEST SUITE
# ======================================================================
print("\n" + "=" * 70)
print("--- 4. AUTOMATED ML INVARIANT UNIT TESTS ---")

class TestMachineLearningInvariants(unittest.TestCase):
    """Verifies mathematical invariants essential for neural training."""

    def test_float_arithmetic_tolerance(self):
        """Floating point calculations must use almost_equal."""
        val1 = sum([0.1] * 10)
        self.assertAlmostEqual(val1, 1.0, places=7)

    def test_softmax_probability_distribution(self):
        """Softmax outputs must sum to exactly 1.0 and each p_i in [0, 1]."""
        logits = [2.0, 1.0, 0.1]
        exp_logits = [math.exp(x) for x in logits]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]

        # Invariant 1: Sums to 1.0
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        # Invariant 2: All probabilities non-negative
        for p in probs:
            self.assertTrue(0.0 <= p <= 1.0)

    def test_random_cross_entropy_baseline(self):
        """A random model over K classes must have initial loss ≈ ln(K)."""
        num_classes = 10
        expected_random_loss = math.log(num_classes)
        # Simulated actual output from an uninitialized model
        observed_loss = 2.3025
        self.assertAlmostEqual(observed_loss, expected_random_loss, delta=0.05)

# Run test suite programmatically
suite = unittest.TestLoader().loadTestsFromTestCase(TestMachineLearningInvariants)
runner = unittest.TextTestRunner(verbosity=1)
test_output = runner.run(suite)

print(f"\nUnit Tests Ran: {test_output.testsRun} | Errors: {len(test_output.errors)} | Failures: {len(test_output.failures)}")
assert test_output.wasSuccessful(), "Unit tests failed!"
print("✅ All Machine Learning Invariant Tests Passed!")

# ======================================================================
# 5. COMMAND LINE ARGPARSE DEMONSTRATION
# ======================================================================
print("\n" + "=" * 70)
print("--- 5. ARGPARSE HYPERPARAMETER SIMULATION ---")

parser = argparse.ArgumentParser(description="AI Training Harness")
parser.add_argument("--model", type=str, default="gpt2-nano")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=3e-4)

# Parse simulated command line arguments
simulated_args = parser.parse_args(["--model", "custom-transformer", "--batch-size", "128"])
print(f"Parsed Hyperparameters: Model='{simulated_args.model}', Batch={simulated_args.batch_size}, LR={simulated_args.lr}")

print("\n" + "=" * 70)
print("EXPERIMENT 1.2 COMPLETED SUCCESSFULLY!")
print("=" * 70)

