"""
Topic 6.5: Mutual Information & Information Bottleneck Experiment
Verifying Information Identities & Simulating Neural Representation Compression
Pure Python 3 Standard Library implementation (Zero External Dependencies).
"""

import math
from collections import Counter, defaultdict

def compute_entropy(probs):
    """Computes Shannon entropy in bits."""
    return -sum(p * math.log2(p) for p in probs if p > 0)

def compute_mutual_information(joint_table):
    """
    Computes I(X; Y) given a dict of {(x, y): count}.
    Returns (I_XY, H_X, H_Y, H_XY, H_X_given_Y, H_Y_given_X).
    """
    total = sum(joint_table.values())
    p_xy = {k: v / total for k, v in joint_table.items()}

    # Marginals
    p_x = defaultdict(float)
    p_y = defaultdict(float)
    for (x, y), p in p_xy.items():
        p_x[x] += p
        p_y[y] += p

    h_x = compute_entropy(list(p_x.values()))
    h_y = compute_entropy(list(p_y.values()))
    h_xy = compute_entropy(list(p_xy.values()))

    # Direct formula: sum P(x, y) * log2(P(x, y) / (P(x) * P(y)))
    mi = 0.0
    for (x, y), p in p_xy.items():
        if p > 0:
            ratio = p / (p_x[x] * p_y[y])
            mi += p * math.log2(ratio)

    h_y_given_x = h_xy - h_x
    h_x_given_y = h_xy - h_y

    return mi, h_x, h_y, h_xy, h_x_given_y, h_y_given_x

def main():
    print("=" * 72)
    print("TOPIC 6.5: MUTUAL INFORMATION & THE INFORMATION BOTTLENECK")
    print("=" * 72)

    # --- 1. Verifying the Fundamental Information Identities ---
    # Synthetic joint table between Weather X and Traffic Y
    # X in {Sunny: 0, Rainy: 1}, Y in {Fast: 0, Slow: 1}
    joint_data = {
        (0, 0): 40,  # Sunny & Fast
        (0, 1): 10,  # Sunny & Slow
        (1, 0): 10,  # Rainy & Fast
        (1, 1): 40,  # Rainy & Slow
    }

    mi, h_x, h_y, h_xy, h_x_given_y, h_y_given_x = compute_mutual_information(joint_data)

    print("--- 1. Information Venn Diagram Identities Verification ---")
    print(f"Marginal Entropy H(X):            {h_x:.4f} bits")
    print(f"Marginal Entropy H(Y):            {h_y:.4f} bits")
    print(f"Joint Entropy H(X, Y):            {h_xy:.4f} bits")
    print(f"Conditional Entropy H(X | Y):     {h_x_given_y:.4f} bits")
    print(f"Conditional Entropy H(Y | X):     {h_y_given_x:.4f} bits")
    print(f"Direct Mutual Information I(X; Y):{mi:.4f} bits\n")

    # Identity 1: I(X; Y) == H(X) - H(X | Y)
    id1 = h_x - h_x_given_y
    # Identity 2: I(X; Y) == H(Y) - H(Y | X)
    id2 = h_y - h_y_given_x
    # Identity 3: I(X; Y) == H(X) + H(Y) - H(X, Y)
    id3 = h_x + h_y - h_xy

    print(f"Identity 1 check (H(X) - H(X|Y)):    {id1:.4f} bits (diff: {abs(mi - id1):.2e})")
    print(f"Identity 2 check (H(Y) - H(Y|X)):    {id2:.4f} bits (diff: {abs(mi - id2):.2e})")
    print(f"Identity 3 check (H(X)+H(Y)-H(X,Y)): {id3:.4f} bits (diff: {abs(mi - id3):.2e})")

    assert abs(mi - id1) < 1e-10, "Identity 1 failed!"
    assert abs(mi - id2) < 1e-10, "Identity 2 failed!"
    assert abs(mi - id3) < 1e-10, "Identity 3 failed!"
    print(">> SUCCESS: All Mutual Information identities verified to machine precision!\n")

    # --- 2. Information Bottleneck Simulation Across Neural Network Layers ---
    print("--- 2. Information Bottleneck Across Network Layers ---")
    print("Task: 4-bit input X = (b1, b2, b3, b4). Target Y = Majority(b1, b2, b3).")
    print("Notice that bit 4 is completely useless noise.\n")

    # Generate all 16 4-bit patterns
    samples = []
    for b1 in [0, 1]:
        for b2 in [0, 1]:
            for b3 in [0, 1]:
                for b4 in [0, 1]:
                    x = (b1, b2, b3, b4)
                    y = 1 if (b1 + b2 + b3) >= 2 else 0
                    # T1: Input layer representation (stores all 4 bits)
                    t1 = x
                    # T2: Hidden layer representation (learns to drop noise bit 4)
                    t2 = (b1, b2, b3)
                    # T3: Output layer representation (collapses to target class)
                    t3 = y
                    samples.append((x, y, t1, t2, t3))

    # Evaluate I(X; T) and I(T; Y) for each layer T
    layers = [
        ("Layer 0 (Raw Input X)", [s[0] for s in samples]),
        ("Layer 1 (Internal T1)", [s[2] for s in samples]),
        ("Layer 2 (Internal T2)", [s[3] for s in samples]),
        ("Layer 3 (Output T3)",   [s[4] for s in samples]),
    ]

    targets = [s[1] for s in samples]
    raw_inputs = [s[0] for s in samples]

    print(f"{'Network Layer':<24} | {'I(X; T) [Input Info]':<22} | {'I(T; Y) [Label Info]':<22}")
    print("-" * 74)

    prev_i_xt = float("inf")
    for name, t_vals in layers:
        # Joint table for (X, T)
        xt_counts = Counter(zip(raw_inputs, t_vals))
        i_xt, _, _, _, _, _ = compute_mutual_information(xt_counts)

        # Joint table for (T, Y)
        ty_counts = Counter(zip(t_vals, targets))
        i_ty, _, _, _, _, _ = compute_mutual_information(ty_counts)

        print(f"{name:<24} | {i_xt:<22.4f} | {i_ty:<22.4f}")

        # Check compression: I(X; T) must monotonically decrease across successive layers
        assert i_xt <= prev_i_xt + 1e-10, f"Compression violated at {name}!"
        prev_i_xt = i_xt

        # Check preservation: I(T; Y) must remain 1.0 bit (perfect knowledge of target)
        assert abs(i_ty - 1.0) < 1e-10, f"Target information lost at {name}!"

    print("\n>> VERIFIED: Neural representations compress input information (4.0 -> 3.0 -> 1.0 bits)")
    print(">> while perfectly preserving label information I(T; Y) = 1.0000 bit.")
    print(">> Proven empirically: The Information Bottleneck principle in action!")
    print("=" * 72)

if __name__ == "__main__":
    main()

