#!/usr/bin/env python3
"""
Phase 17 - Topic 03: Full GPT Model Architecture & Weight Tying
Complete pure Python 3 implementation with ZERO external dependencies.

Implements:
1. Complete GPT model assembly:
   - Token Embedding Table W_token [V x d_model]
   - Learned Position Embedding Table W_pos [T_max x d_model]
   - Stack of N Pre-LN Transformer Blocks
   - Final LayerNorm and Language Model (LM) Head [d_model x V]
2. Weight Tying (Press & Wolf 2017): Reusing W_token as the projection matrix for W_head.
3. Cross-entropy loss calculation with numerically stable log-softmax.
4. Exact closed-form parameter counting formulas matching GPT-2 Small (124M), Medium (355M),
   Large (774M), and XL (1.5B).
5. Comprehensive test suite: forward pass, loss calculation, weight-tying assertion, and parameter audit.
"""

import math
import random
import sys
from typing import List, Tuple, Optional, Dict

# Set deterministic seed
random.seed(42)


# =====================================================================
# 1. Pure Python Math & Activation Primitives
# =====================================================================

def gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))


def softmax_1d(vec: List[float]) -> List[float]:
    max_val = max(vec)
    exp_vec = [math.exp(v - max_val) for v in vec]
    sum_exp = sum(exp_vec)
    return [v / sum_exp for v in exp_vec]


def log_softmax_1d(vec: List[float]) -> List[float]:
    """Numerically stable log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))."""
    max_val = max(vec)
    sum_exp = sum(math.exp(v - max_val) for v in vec)
    log_sum_exp = math.log(sum_exp)
    return [(v - max_val) - log_sum_exp for v in vec]


def matmul_2d(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    M = len(A)
    K = len(A[0])
    N = len(B[0])
    B_T = [[B[r][c] for r in range(K)] for c in range(N)]
    out = [[0.0] * N for _ in range(M)]
    for i in range(M):
        row_A = A[i]
        for j in range(N):
            row_B_T = B_T[j]
            s = 0.0
            for k in range(K):
                s += row_A[k] * row_B_T[k]
            out[i][j] = s
    return out


def transpose_2d(A: List[List[float]]) -> List[List[float]]:
    return [[A[r][c] for r in range(len(A))] for c in range(len(A[0]))]


# =====================================================================
# 2. Sub-Layer Modules
# =====================================================================

class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = [1.0] * d_model
        self.beta = [0.0] * d_model

    def forward(self, X: List[List[List[float]]]) -> List[List[List[float]]]:
        B = len(X)
        T = len(X[0])
        D = self.d_model
        out = [[[0.0] * D for _ in range(T)] for _ in range(B)]
        for b in range(B):
            for t in range(T):
                vec = X[b][t]
                mean = sum(vec) / D
                var = sum((v - mean) ** 2 for v in vec) / D
                std = math.sqrt(var + self.eps)
                for d in range(D):
                    out[b][t][d] = ((vec[d] - mean) / std) * self.gamma[d] + self.beta[d]
        return out


class CausalSelfAttention:
    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        init_scale = 1.0 / math.sqrt(d_model)
        self.W_q = [[random.gauss(0.0, init_scale) for _ in range(d_model)] for _ in range(d_model)]
        self.b_q = [0.0] * d_model
        self.W_k = [[random.gauss(0.0, init_scale) for _ in range(d_model)] for _ in range(d_model)]
        self.b_k = [0.0] * d_model
        self.W_v = [[random.gauss(0.0, init_scale) for _ in range(d_model)] for _ in range(d_model)]
        self.b_v = [0.0] * d_model
        self.W_o = [[random.gauss(0.0, init_scale) for _ in range(d_model)] for _ in range(d_model)]
        self.b_o = [0.0] * d_model

    def forward(self, X: List[List[List[float]]]) -> List[List[List[float]]]:
        B = len(X)
        T = len(X[0])
        D = self.d_model
        H = self.n_heads
        d_k = self.d_k
        out = [[[0.0] * D for _ in range(T)] for _ in range(B)]

        for b in range(B):
            seq = X[b]
            Q_all = matmul_2d(seq, self.W_q)
            K_all = matmul_2d(seq, self.W_k)
            V_all = matmul_2d(seq, self.W_v)
            for t in range(T):
                for d in range(D):
                    Q_all[t][d] += self.b_q[d]
                    K_all[t][d] += self.b_k[d]
                    V_all[t][d] += self.b_v[d]

            head_outputs = [[[0.0] * d_k for _ in range(T)] for _ in range(H)]
            for h in range(H):
                offset = h * d_k
                Q_h = [[Q_all[t][offset + i] for i in range(d_k)] for t in range(T)]
                K_h = [[K_all[t][offset + i] for i in range(d_k)] for t in range(T)]
                V_h = [[V_all[t][offset + i] for i in range(d_k)] for t in range(T)]

                scores = matmul_2d(Q_h, transpose_2d(K_h))
                attn_weights = [[0.0] * T for _ in range(T)]
                for i in range(T):
                    row = [0.0] * T
                    for j in range(T):
                        row[j] = scores[i][j] * self.scale if j <= i else -1e9
                    attn_weights[i] = softmax_1d(row)

                context_h = matmul_2d(attn_weights, V_h)
                head_outputs[h] = context_h

            concat_heads = [[0.0] * D for _ in range(T)]
            for t in range(T):
                for h in range(H):
                    offset = h * d_k
                    for i in range(d_k):
                        concat_heads[t][offset + i] = head_outputs[h][t][i]

            projected = matmul_2d(concat_heads, self.W_o)
            for t in range(T):
                for d in range(D):
                    out[b][t][d] = projected[t][d] + self.b_o[d]
        return out


class MLP:
    def __init__(self, d_model: int):
        self.d_model = d_model
        d_ff = 4 * d_model
        init_1 = 1.0 / math.sqrt(d_model)
        init_2 = 1.0 / math.sqrt(d_ff)
        self.W_1 = [[random.gauss(0.0, init_1) for _ in range(d_ff)] for _ in range(d_model)]
        self.b_1 = [0.0] * d_ff
        self.W_2 = [[random.gauss(0.0, init_2) for _ in range(d_model)] for _ in range(d_ff)]
        self.b_2 = [0.0] * d_model

    def forward(self, X: List[List[List[float]]]) -> List[List[List[float]]]:
        B = len(X)
        T = len(X[0])
        D = self.d_model
        d_ff = 4 * D
        out = [[[0.0] * D for _ in range(T)] for _ in range(B)]
        for b in range(B):
            h1 = matmul_2d(X[b], self.W_1)
            for t in range(T):
                for j in range(d_ff):
                    h1[t][j] = gelu(h1[t][j] + self.b_1[j])
            h2 = matmul_2d(h1, self.W_2)
            for t in range(T):
                for j in range(D):
                    out[b][t][j] = h2[t][j] + self.b_2[j]
        return out


class TransformerBlock:
    def __init__(self, d_model: int, n_heads: int):
        self.ln_1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, X: List[List[List[float]]]) -> List[List[List[float]]]:
        B = len(X)
        T = len(X[0])
        D = len(X[0][0])
        normed_1 = self.ln_1.forward(X)
        attn_out = self.attn.forward(normed_1)
        x_res1 = [[[X[b][t][d] + attn_out[b][t][d] for d in range(D)] for t in range(T)] for b in range(B)]
        normed_2 = self.ln_2.forward(x_res1)
        mlp_out = self.mlp.forward(normed_2)
        return [[[x_res1[b][t][d] + mlp_out[b][t][d] for d in range(D)] for t in range(T)] for b in range(B)]


# =====================================================================
# 3. Full GPT Model Architecture
# =====================================================================

class GPT:
    """
    Full Generative Pretrained Transformer (GPT) Architecture with Weight Tying.
    """
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        d_model: int,
        tie_weights: bool = True
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.tie_weights = tie_weights

        init_scale = 0.02
        # Token Embeddings: W_token [vocab_size, d_model]
        self.wte = [[random.gauss(0.0, init_scale) for _ in range(d_model)] for _ in range(vocab_size)]
        # Position Embeddings: W_pos [block_size, d_model]
        self.wpe = [[random.gauss(0.0, init_scale) for _ in range(d_model)] for _ in range(block_size)]

        # Stack of Transformer Blocks
        self.blocks = [TransformerBlock(d_model, n_head) for _ in range(n_layer)]

        # Final LayerNorm
        self.ln_f = LayerNorm(d_model)

        # Output LM Head: projects d_model -> vocab_size
        if not tie_weights:
            # Independent projection matrix
            self.lm_head_weight = [[random.gauss(0.0, init_scale) for _ in range(vocab_size)] for _ in range(d_model)]
        else:
            # Weight Tying: W_head is the exact transpose of W_token!
            # We reference self.wte directly, allocating ZERO extra weights.
            self.lm_head_weight = None

    def forward(
        self,
        idx: List[List[int]],
        targets: Optional[List[List[int]]] = None
    ) -> Tuple[List[List[List[float]]], Optional[float]]:
        """
        Input idx: [B, T] token IDs
        targets: [B, T] next-token ground truth IDs (optional)
        Returns:
            logits: [B, T, vocab_size]
            loss: scalar cross-entropy loss (if targets is provided)
        """
        B = len(idx)
        T = len(idx[0])
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"
        D = self.d_model

        # 1. Embeddings: Token Embedding + Position Embedding
        X = [[[0.0] * D for _ in range(T)] for _ in range(B)]
        for b in range(B):
            for t in range(T):
                token_id = idx[b][t]
                tok_emb = self.wte[token_id]
                pos_emb = self.wpe[t]
                for d in range(D):
                    X[b][t][d] = tok_emb[d] + pos_emb[d]

        # 2. Forward through N Transformer Blocks
        for block in self.blocks:
            X = block.forward(X)

        # 3. Final LayerNorm
        X = self.ln_f.forward(X)

        # 4. Output LM Head: [B, T, d_model] -> [B, T, vocab_size]
        logits = [[[0.0] * self.vocab_size for _ in range(T)] for _ in range(B)]

        if self.tie_weights:
            # W_token is [V, D]. Projection is X [T, D] * W_token^T [D, V]
            # Since wte is stored as [V, D], dot product between x[t] and wte[v] is: sum_d x[t][d] * wte[v][d]
            for b in range(B):
                for t in range(T):
                    h = X[b][t]
                    for v in range(self.vocab_size):
                        w_v = self.wte[v]
                        dot = 0.0
                        for d in range(D):
                            dot += h[d] * w_v[d]
                        logits[b][t][v] = dot
        else:
            # Independent LM head: X [T, D] * lm_head_weight [D, V]
            for b in range(B):
                logits[b] = matmul_2d(X[b], self.lm_head_weight)

        # 5. Cross-Entropy Loss (if targets provided)
        loss = None
        if targets is not None:
            total_nll = 0.0
            count = 0
            for b in range(B):
                for t in range(T):
                    log_probs = log_softmax_1d(logits[b][t])
                    target_id = targets[b][t]
                    total_nll += -log_probs[target_id]
                    count += 1
            loss = total_nll / count

        return logits, loss

    def count_parameters(self) -> Dict[str, int]:
        """Audit and count all unique learnable parameters."""
        wte_params = self.vocab_size * self.d_model
        wpe_params = self.block_size * self.d_model
        
        # Per block:
        # Attn: 4 * (d_model * d_model + d_model)
        # MLP: W1: d_model * 4d_model + 4d_model; W2: 4d_model * d_model + d_model
        # LN: 2 * (2 * d_model)
        d = self.d_model
        attn_params = 4 * (d * d + d)
        mlp_params = (d * 4 * d + 4 * d) + (4 * d * d + d)
        ln_params = 4 * d
        per_block = attn_params + mlp_params + ln_params
        total_blocks = self.n_layer * per_block

        ln_f_params = 2 * d
        lm_head_params = 0 if self.tie_weights else (d * self.vocab_size)

        total_unique = wte_params + wpe_params + total_blocks + ln_f_params + lm_head_params
        return {
            "wte": wte_params,
            "wpe": wpe_params,
            "blocks": total_blocks,
            "ln_f": ln_f_params,
            "lm_head": lm_head_params,
            "total_unique": total_unique
        }


# =====================================================================
# 4. Closed-Form Parameter Formulas for Standard GPT-2 Models
# =====================================================================

def theoretical_gpt2_params(
    vocab_size: int,
    block_size: int,
    n_layer: int,
    d_model: int,
    tied: bool = True
) -> int:
    """Exact theoretical closed-form parameter counter."""
    d = d_model
    # Embeddings
    wte = vocab_size * d
    wpe = block_size * d
    
    # Blocks:
    # Attn: Q, K, V, O projections (4 * d^2) + biases (4 * d)
    attn = 4 * d * d + 4 * d
    # MLP: W1 (4 * d^2) + b1 (4d) + W2 (4 * d^2) + b2 (d)
    mlp = 8 * d * d + 5 * d
    # LN: 2 LayerNorms per block, each 2 * d (gamma, beta)
    ln = 4 * d
    one_block = attn + mlp + ln
    
    # Final LN
    ln_f = 2 * d
    # LM Head
    lm_head = 0 if tied else (d * vocab_size)
    
    return wte + wpe + (n_layer * one_block) + ln_f + lm_head


# =====================================================================
# 5. Verification & Testing Suite
# =====================================================================

def run_experiment():
    print("=" * 70)
    print("Phase 17 - Topic 03: Full GPT Architecture & Weight Tying Lab")
    print("=" * 70)

    # 1. Instantiate Mini-GPT for fast forward-pass testing
    V = 50       # Vocabulary size
    T = 6        # Max sequence length
    N = 2        # Number of layers
    H = 2        # Attention heads
    D = 16       # Embedding dimension

    print(f"[*] Instantiating Mini-GPT (V={V}, T={T}, L={N}, H={H}, D={D}, Tied=True)")
    model = GPT(vocab_size=V, block_size=T, n_layer=N, n_head=H, d_model=D, tie_weights=True)

    # 2. Forward pass with Batch inputs & Targets
    batch_idx = [
        [1, 14, 25, 33, 42, 9],
        [7, 18, 2,  49, 11, 30]
    ]
    batch_targets = [
        [14, 25, 33, 42, 9,  0],
        [18, 2,  49, 11, 30, 0]
    ]

    logits, loss = model.forward(batch_idx, batch_targets)

    # Assert shape of output logits [B, T, V]
    assert len(logits) == 2
    assert len(logits[0]) == 6
    assert len(logits[0][0]) == V
    print(f"    [1] Forward pass logits shape: [2, 6, {V}] -> ✅ PASSED")
    print(f"    [2] Cross-Entropy loss computed: {loss:.4f} -> ✅ PASSED")

    # Initial random loss check: should be close to -log(1 / V) = log(V)
    expected_random_loss = math.log(V)
    print(f"        (Theoretical random guessing loss = ln({V}) = {expected_random_loss:.4f})")
    assert abs(loss - expected_random_loss) < 1.0, "Initial loss is abnormally far from uniform cross-entropy!"

    # 3. Verify Weight Tying Mechanics
    print("\n" + "=" * 70)
    print("[*] Verifying Weight Tying & Parameter Memory Footprint:")
    print("=" * 70)
    param_audit_tied = model.count_parameters()
    print(f"    Tied Model Parameters:")
    for k, v in param_audit_tied.items():
        print(f"      - {k:12s}: {v:8,d}")

    # Instantiate untied model
    untied_model = GPT(vocab_size=V, block_size=T, n_layer=N, n_head=H, d_model=D, tie_weights=False)
    param_audit_untied = untied_model.count_parameters()
    print(f"\n    Untied Model Parameters (LM Head separate):")
    for k, v in param_audit_untied.items():
        print(f"      - {k:12s}: {v:8,d}")

    diff = param_audit_untied["total_unique"] - param_audit_tied["total_unique"]
    assert diff == V * D, f"Expected parameter difference of {V*D}, got {diff}"
    print(f"\n    Weight tying saved exactly V * d_model = {V} * {D} = {diff} parameters! -> ✅ PASSED")

    # 4. Closed-form Formula Verification
    formula_tied = theoretical_gpt2_params(V, T, N, D, tied=True)
    assert formula_tied == param_audit_tied["total_unique"], "Formula mismatch on Mini-GPT!"
    print(f"    Formula prediction matches instantiated architecture exactly ({formula_tied}) -> ✅ PASSED")

    # 5. Standard GPT-2 Family Exact Parameter Audit
    print("\n" + "=" * 70)
    print("[*] Official GPT-2 Family Parameter Breakdown (V=50,257, T=1024, Tied):")
    print("=" * 70)
    gpt2_specs = [
        ("GPT-2 Small",  12, 768,  12),
        ("GPT-2 Medium", 24, 1024, 16),
        ("GPT-2 Large",  36, 1280, 20),
        ("GPT-2 XL",     48, 1600, 25),
    ]

    print(f"  {'Model':<15} | {'Layers':<7} | {'d_model':<8} | {'Tied Params':<15} | {'Untied Params':<15}")
    print("  " + "-" * 70)
    for name, L, d, h in gpt2_specs:
        tied_p = theoretical_gpt2_params(50257, 1024, L, d, tied=True)
        untied_p = theoretical_gpt2_params(50257, 1024, L, d, tied=False)
        print(f"  {name:<15} | {L:<7} | {d:<8} | {tied_p / 1e6:8.2f}M ({tied_p:,}) | {untied_p / 1e6:8.2f}M ({untied_p:,})")

    # Verify standard GPT-2 Small is ~124M
    gpt2_small_tied = theoretical_gpt2_params(50257, 1024, 12, 768, tied=True)
    assert 123_000_000 < gpt2_small_tied < 125_000_000, "GPT-2 Small count out of expected range!"

    print("\n" + "=" * 70)
    print("✅ All GPT Architecture & Weight Tying tests passed successfully!")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    run_experiment()
