#!/usr/bin/env python3
"""
Topic 6.1 Experiment: Shannon Entropy, Huffman Coding & Compression Bounds
Phase 06: Information Theory for Artificial Intelligence

This script provides a pure Python implementation (zero external dependencies) of:
1. Shannon Entropy calculator for discrete distributions and natural text.
2. Binary Entropy function verification across probabilities p in [0, 1].
3. Complete Huffman Tree construction and optimal prefix code generation.
4. Empirical verification of Shannon's Source Coding Theorem: H(X) <= L < H(X) + 1.
5. Compression ratio comparison: Raw 8-bit ASCII vs Optimal Huffman Coding.
"""

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Shannon Entropy Calculator
# ---------------------------------------------------------------------------

def compute_entropy_from_probs(probs: List[float], base: float = 2.0) -> float:
    """Computes Shannon Entropy H(X) = -sum(p * log(p))."""
    log_func = math.log2 if base == 2.0 else math.log
    entropy = 0.0
    for p in probs:
        if p > 1e-15:
            entropy -= p * log_func(p)
    return entropy


def compute_text_entropy(text: str) -> Tuple[float, Dict[str, float]]:
    """Calculates empirical character probabilities and Shannon entropy of text."""
    n = len(text)
    if n == 0:
        return 0.0, {}
    counts = Counter(text)
    probs = {char: count / n for char, count in counts.items()}
    h = compute_entropy_from_probs(list(probs.values()), base=2.0)
    return h, probs


# ---------------------------------------------------------------------------
# 2. Huffman Coding Tree Implementation (Pure Python)
# ---------------------------------------------------------------------------

class HuffmanNode:
    def __init__(self, char: Optional[str], freq: float):
        self.char = char
        self.freq = freq
        self.left: Optional[HuffmanNode] = None
        self.right: Optional[HuffmanNode] = None

    def __lt__(self, other: "HuffmanNode") -> bool:
        return self.freq < other.freq


def build_huffman_tree(probs: Dict[str, float]) -> Optional[HuffmanNode]:
    """Builds a binary Huffman tree from character probabilities."""
    nodes: List[HuffmanNode] = [HuffmanNode(char, p) for char, p in probs.items()]
    
    if len(nodes) == 0:
        return None
    if len(nodes) == 1:
        root = HuffmanNode(None, nodes[0].freq)
        root.left = nodes[0]
        return root
        
    while len(nodes) > 1:
        # Sort nodes by frequency ascending
        nodes.sort(key=lambda x: x.freq)
        # Pop the two lowest-frequency nodes
        left = nodes.pop(0)
        right = nodes.pop(0)
        
        # Merge into a parent node
        parent = HuffmanNode(None, left.freq + right.freq)
        parent.left = left
        parent.right = right
        nodes.append(parent)
        
    return nodes[0]


def generate_huffman_codes(
    node: Optional[HuffmanNode],
    current_code: str = "",
    codebook: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Recursively traverses the Huffman tree to generate binary codewords."""
    if codebook is None:
        codebook = {}
    if node is None:
        return codebook
        
    # Leaf node contains a character
    if node.char is not None:
        codebook[node.char] = current_code if current_code != "" else "0"
        return codebook
        
    generate_huffman_codes(node.left, current_code + "0", codebook)
    generate_huffman_codes(node.right, current_code + "1", codebook)
    return codebook


# ---------------------------------------------------------------------------
# 3. Main Execution Suite
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("TOPIC 6.1: SHANNON ENTROPY, HUFFMAN CODING & COMPRESSION LIMITS")
    print("=" * 72)

    # 1. Binary Entropy Function Verification
    print("\n--- 1. Binary Entropy Function H_b(p) ---")
    p_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    print(f"{'Probability p':>15} | {'Binary Entropy (Bits)':>24}")
    print("-" * 44)
    for p in p_values:
        probs = [p, 1.0 - p] if 0.0 < p < 1.0 else ([1.0] if p in (0.0, 1.0) else [0.0])
        h_val = compute_entropy_from_probs(probs, base=2.0)
        print(f"{p:>15.2f} | {h_val:>24.4f}")
        
    h_fair = compute_entropy_from_probs([0.5, 0.5], base=2.0)
    assert abs(h_fair - 1.0) < 1e-12, "Fair coin entropy must be exactly 1.0 bit!"
    print(">> SUCCESS: Binary entropy peaks at exactly 1.0 bit for p = 0.5!")

    # 2. Text Entropy & Huffman Compression
    sample_text = (
        "artificial intelligence machine learning neural network transformer "
        "entropy information theory data compression probability distribution"
    )
    print(f"\n--- 2. Shannon's Source Coding Theorem on Natural Text ---")
    print(f"Sample Text: \"{sample_text[:40]}...\" ({len(sample_text)} characters)")
    
    h_text, char_probs = compute_text_entropy(sample_text)
    print(f"Unique Characters (Alphabet Size): {len(char_probs)}")
    print(f"Shannon Entropy H(X):              {h_text:.4f} bits / character")

    # Build optimal Huffman codebook
    tree = build_huffman_tree(char_probs)
    codebook = generate_huffman_codes(tree)
    
    # Calculate average codeword length L = sum(p_i * length_i)
    avg_length = sum(char_probs[c] * len(codebook[c]) for c in char_probs)
    print(f"Huffman Average Code Length L:     {avg_length:.4f} bits / character")

    # Verify Shannon's Bound: H(X) <= L < H(X) + 1
    print(f"Shannon's Inequality Check:        {h_text:.4f} <= {avg_length:.4f} < {h_text + 1.0:.4f}")
    assert h_text <= avg_length < h_text + 1.0, "Shannon Source Coding Theorem violated!"
    print(">> SUCCESS: Shannon's Source Coding Theorem strictly verified!")

    # 3. Compression Benchmark: Raw ASCII vs Huffman
    raw_bits = len(sample_text) * 8  # 8 bits per ASCII byte
    compressed_bits = sum(len(codebook[c]) for c in sample_text)
    compression_ratio = raw_bits / compressed_bits
    
    print("\n--- 3. Compression Performance ---")
    print(f"Raw 8-bit ASCII Storage:     {raw_bits} bits")
    print(f"Optimal Huffman Storage:     {compressed_bits} bits")
    print(f"Compression Savings:         {100.0 * (1.0 - compressed_bits / raw_bits):.1f}% reduction")
    print(f"Effective Compression Ratio: {compression_ratio:.2f}x smaller")
    print("=" * 72)


if __name__ == "__main__":
    main()

