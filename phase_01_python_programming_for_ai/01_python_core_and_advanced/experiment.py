#!/usr/bin/env python3
"""
EXPERIMENT 1.1: CORE & ADVANCED PYTHON FOR AI
=============================================
A hands-on, zero-dependency Python laboratory demonstrating:
1. Object Identity, Mutability, and Memory References (id, is, ==)
2. Custom PyTorch-style Dataset & Iterator Protocol (__getitem__, __len__, __iter__, __next__)
3. Mini-Batch Streaming Generator (yield) vs List Comprehension Memory
4. High-Performance Latency Profiler Decorator (@functools.wraps)
5. Custom AI Domain Exceptions (ShapeMismatchError)

Runs on standard library Python 3 without external packages.
"""

import sys
import time
import functools

print("=" * 70)
print("EXPERIMENT 1.1: CORE & ADVANCED PYTHON FOR AI")
print("=" * 70)

# ======================================================================
# 1. OBJECT IDENTITY, MUTABILITY, AND MEMORY REFERENCES
# ======================================================================
print("\n--- 1. OBJECT IDENTITY & MUTABILITY DEMONSTRATION ---")

# Sticky nametag mental model
a = [10, 20, 30]
b = a              # b is a second nametag on the same list object
c = [10, 20, 30]  # c is a distinct list object with identical contents

print(f"List 'a': {a} | id(a) = {hex(id(a))}")
print(f"List 'b': {b} | id(b) = {hex(id(b))}")
print(f"List 'c': {c} | id(c) = {hex(id(c))}")

print(f"\nEquality   'a == b': {a == b} (Same values)")
print(f"Identity   'a is b': {a is b} (Same physical memory address!)")
print(f"Equality   'a == c': {a == c} (Same values)")
print(f"Identity   'a is c': {a is c} (Different physical memory objects!)")

# Mutating 'b' alters 'a' because they are identical
b.append(40)
print(f"\nAfter 'b.append(40)':")
print(f"  List 'a' now contains: {a} (Mutated through 'b'!)")


# ======================================================================
# 2. CUSTOM PYTORCH-STYLE DATASET & DATALOADER
# ======================================================================
print("\n" + "=" * 70)
print("--- 2. PYTORCH-STYLE DATASET & DATALOADER ABSTRACTIONS ---")

class MockTextDataset:
    """Simulates a PyTorch Dataset for NLP token sequences."""
    def __init__(self, samples: list[str]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, any]:
        text = self.samples[idx]
        # Basic character-level tokenization simulation
        tokens = [ord(char) for char in text[:10]]
        return {"id": idx, "text": text, "tokens": tokens}

    def __repr__(self) -> str:
        return f"MockTextDataset(total_samples={len(self)})"

dataset = MockTextDataset([
    "The quick brown fox jumps over the lazy dog",
    "Artificial intelligence is mathematical modeling",
    "Deep learning relies on gradient descent",
    "Transformers utilize self-attention mechanisms",
    "Linear algebra powers modern machine learning"
])

print(dataset)
sample_0 = dataset[0]
print(f"Sample 0 -> Text: '{sample_0['text']}' | Tokens: {sample_0['tokens']}")


# ======================================================================
# 3. STREAMING GENERATORS WITH 'YIELD'
# ======================================================================
print("\n" + "=" * 70)
print("--- 3. BATCH GENERATOR FOR STREAMING BIG DATA ---")

def batch_streamer(dataset_obj, batch_size: int):
    """Yields batches sequentially without loading all data into RAM at once."""
    current_batch = []
    for i in range(len(dataset_obj)):
        current_batch.append(dataset_obj[i])
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch

print("Streaming mini-batches with batch_size=2:")
for batch_idx, batch in enumerate(batch_streamer(dataset, batch_size=2)):
    ids = [item["id"] for item in batch]
    print(f"  Batch {batch_idx + 1}: {len(batch)} items (Sample IDs: {ids})")


# ======================================================================
# 4. LATENCY PROFILER DECORATOR
# ======================================================================
print("\n" + "=" * 70)
print("--- 4. INFERENCE LATENCY PROFILER DECORATOR ---")

def measure_latency(func):
    """Measures function execution duration in milliseconds."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  ⏱️  [{func.__name__}] Executed in {elapsed_ms:.4f} ms")
        return result
    return wrapper

@measure_latency
def mock_transformer_forward(batch_size: int, hidden_dim: int):
    """Simulates a matrix multiplication feed-forward pass."""
    # Synthetic workload
    matrix = [[0.01 * (i + j) for j in range(hidden_dim)] for i in range(batch_size)]
    vector = [0.5 for _ in range(hidden_dim)]
    # Dot product
    result = [sum(row[k] * vector[k] for k in range(hidden_dim)) for row in matrix]
    return len(result)

output_count = mock_transformer_forward(batch_size=64, hidden_dim=128)
print(f"  Forward pass finished. Generated outputs: {output_count}")


# ======================================================================
# 5. DOMAIN-SPECIFIC AI EXCEPTIONS
# ======================================================================
print("\n" + "=" * 70)
print("--- 5. DOMAIN-SPECIFIC AI EXCEPTIONS (SHAPE CHECKING) ---")

class ShapeMismatchError(Exception):
    """Raised when tensor dimensions are incompatible for matrix multiplication."""
    def __init__(self, shape_a: tuple, shape_b: tuple, op: str):
        super().__init__(
            f"Shape Mismatch: Cannot execute '{op}' between shapes {shape_a} and {shape_b}. "
            f"Inner dimensions ({shape_a[1]} != {shape_b[0]}) must match!"
        )

def verify_matmul_shapes(shape_a: tuple[int, int], shape_b: tuple[int, int]):
    if shape_a[1] != shape_b[0]:
        raise ShapeMismatchError(shape_a, shape_b, op="A @ B")
    return (shape_a[0], shape_b[1])

# Test compatible
valid_out = verify_matmul_shapes((8, 64), (64, 32))
print(f"✅ Compatible: (8, 64) @ (64, 32) -> Output Shape: {valid_out}")

# Test incompatible with custom exception catching
try:
    verify_matmul_shapes((8, 64), (128, 32))
except ShapeMismatchError as e:
    print(f"❌ Handled Expected Error:\n   {e}")

print("\n" + "=" * 70)
print("EXPERIMENT 1.1 COMPLETED SUCCESSFULLY!")
print("=" * 70)

