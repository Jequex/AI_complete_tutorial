#!/usr/bin/env python3
"""
Topic 03: Attention Head Mechanisms & Induction Heads
-----------------------------------------------------
A pure Python 3 standard library implementation demonstrating:
1. Mathematical Transformer Circuits framework (Elhage et al. 2021):
   - QK-circuit: determines attention pattern A = softmax(x^T W_Q^T W_K x / sqrt(d_k))
   - OV-circuit: determines copied information W_O W_V
2. Two-Layer Induction Head Circuit composition:
   - Layer 0: Previous-Token Head (H_{0,0}) attending to position i-1 and copying its token identity.
   - Layer 1: Induction Head (H_{1,0}) matching current token against previous-token features,
     attending to the subsequent token, and boosting its logit for in-context completion:
     [... A B ... A] -> B.
3. Verification on repeated random sequences:
   - Measuring induction accuracy jump.
   - Attention pattern verification.
   - Knockout/ablation testing proving two-layer circuit necessity.

Zero external dependencies (pure Python standard library).
"""

import math
import random

random.seed(42)

# =====================================================================
# 1. Pure Python Math & Matrix Utilities
# =====================================================================

def dot(u, v):
    return sum(a * b for a, b in zip(u, v))

def vec_add(u, v):
    return [a + b for a, b in zip(u, v)]

def vec_scale(u, s):
    return [a * s for a in u]

def matmul_vec(W, x):
    return [dot(row, x) for row in W]

def matmul_mat(A, B):
    """Matrix multiplication: A (m x k), B (k x n) -> (m x n)"""
    m, k = len(A), len(A[0])
    n = len(B[0])
    res = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            res[i][j] = sum(A[i][p] * B[p][j] for p in range(k))
    return res

def transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def matrix_trace(A):
    """Trace of square matrix."""
    return sum(A[i][i] for i in range(len(A)))

def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]

def argmax(arr):
    return max(range(len(arr)), key=lambda i: arr[i])


# =====================================================================
# 2. Two-Layer Induction Transformer Implementation
# =====================================================================

class InductionTransformer:
    """
    Two-layer attention-only toy transformer with dimension d_model.
    Implements:
    - Layer 0 Head 0: Previous-Token Head
    - Layer 1 Head 0: Induction Head
    """
    def __init__(self, vocab_size=16, d_model=32, d_head=16):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_head = d_head
        
        # Embeddings: Token embeddings + Positional embeddings
        # We assign distinct orthogonal-like sub-bands in d_model:
        # dims 0..15: current token identity
        # dims 16..31: space reserved for previous-token signal written by Layer 0
        self.W_E = [[0.0 for _ in range(d_model)] for _ in range(vocab_size)]
        for v in range(vocab_size):
            # One-hot/sparse representation in dims 0..15
            self.W_E[v][v] = 2.0
            
        # Positional encodings (simple relative decay feature)
        self.W_pos = [[0.0 for _ in range(d_model)] for _ in range(128)]
        for pos in range(128):
            self.W_pos[pos][15] = pos * 0.1

        # Layer 0: Previous-Token Head
        # Q attends to current pos, K reads previous pos (pos - 1)
        # OV copies dims 0..15 (current token) into dims 16..31 (previous token memory)
        self.W_Q0 = [[0.0 for _ in range(d_model)] for _ in range(d_head)]
        self.W_K0 = [[0.0 for _ in range(d_model)] for _ in range(d_head)]
        self.W_V0 = [[0.0 for _ in range(d_model)] for _ in range(d_head)]
        self.W_O0 = [[0.0 for _ in range(d_head)] for _ in range(d_model)]
        
        # Positional attention: attend strongly to pos - 1
        self.W_Q0[0][15] = 5.0
        self.W_K0[0][15] = 5.0
        
        # Copy token identity (dims 0..15) into output dims 16..31
        for i in range(15):
            self.W_V0[i][i] = 1.0
            self.W_O0[16 + i][i] = 2.5

        # Layer 1: Induction Head
        # Q reads current token (dims 0..15)
        # K reads previous-token signal (dims 16..31) placed by Layer 0
        # When current token == token that came before token j, QK score spikes for position j!
        # OV copies token j's identity (dims 0..15) and adds it to the logits.
        self.W_Q1 = [[0.0 for _ in range(d_model)] for _ in range(d_head)]
        self.W_K1 = [[0.0 for _ in range(d_model)] for _ in range(d_head)]
        self.W_V1 = [[0.0 for _ in range(d_model)] for _ in range(d_head)]
        self.W_O1 = [[0.0 for _ in range(d_head)] for _ in range(d_model)]
        
        for i in range(15):
            self.W_Q1[i][i] = 2.5       # Q reads current token
            self.W_K1[i][16 + i] = 2.5  # K reads previous-token info at position j
            self.W_V1[i][i] = 1.0       # V reads token j
            self.W_O1[i][i] = 3.5       # O writes token j into residual stream

        # Unembedding matrix: reads dims 0..15
        self.W_U = [[0.0 for _ in range(vocab_size)] for _ in range(d_model)]
        for v in range(vocab_size):
            self.W_U[v][v] = 2.0

    def compute_attention(self, resid, W_Q, W_K, W_V, W_O, causal=True, prev_pos_bias=False):
        """Computes single-head self-attention with causal masking."""
        seq_len = len(resid)
        scale = 1.0 / math.sqrt(self.d_head)
        
        # Compute Q, K, V
        Q = [matmul_vec(W_Q, h) for h in resid]
        K = [matmul_vec(W_K, h) for h in resid]
        V = [matmul_vec(W_V, h) for h in resid]
        
        attn_matrix = []
        head_outputs = []
        
        for i in range(seq_len):
            scores = []
            if prev_pos_bias and i == 0:
                # Token 0 has no preceding token; output remains neutral
                attn_matrix.append([1.0] + [0.0] * (seq_len - 1))
                head_outputs.append([0.0] * self.d_model)
                continue

            for j in range(seq_len):
                if causal and j > i:
                    scores.append(-1e9)
                else:
                    score = dot(Q[i], K[j]) * scale
                    # For Layer 0, attend exclusively to token i-1
                    if prev_pos_bias:
                        if j == i - 1:
                            score += 10.0
                        else:
                            score -= 5.0
                    scores.append(score)
                    
            attn_weights = softmax(scores)
            attn_matrix.append(attn_weights)
            
            # Weighted sum of V
            context_v = [0.0] * self.d_head
            for j in range(seq_len):
                if attn_weights[j] > 1e-6:
                    for d in range(self.d_head):
                        context_v[d] += attn_weights[j] * V[j][d]
                        
            out = matmul_vec(W_O, context_v)
            head_outputs.append(out)
            
        return head_outputs, attn_matrix

    def forward(self, tokens, ablate_l0=False, ablate_l1=False):
        """
        Forward pass returning next-token predictions and attention maps.
        """
        seq_len = len(tokens)
        
        # 1. Embeddings: Residual Stream h_0
        resid = []
        for pos, tok in enumerate(tokens):
            h = vec_add(self.W_E[tok], self.W_pos[pos])
            resid.append(h)
            
        # 2. Layer 0 Attention (Previous-Token Head)
        if not ablate_l0:
            out_0, attn_0 = self.compute_attention(resid, self.W_Q0, self.W_K0, self.W_V0, self.W_O0, causal=True, prev_pos_bias=True)
            for pos in range(seq_len):
                resid[pos] = vec_add(resid[pos], out_0[pos])
        else:
            attn_0 = [[[1.0 / (i + 1) if j <= i else 0.0 for j in range(seq_len)] for _ in range(seq_len)] for i in range(seq_len)]

        # 3. Layer 1 Attention (Induction Head)
        if not ablate_l1:
            out_1, attn_1 = self.compute_attention(resid, self.W_Q1, self.W_K1, self.W_V1, self.W_O1, causal=True, prev_pos_bias=False)
            for pos in range(seq_len):
                resid[pos] = vec_add(resid[pos], out_1[pos])
        else:
            attn_1 = [[[1.0 / (i + 1) if j <= i else 0.0 for j in range(seq_len)] for _ in range(seq_len)] for i in range(seq_len)]

        # 4. Unembedding: Logits = resid * W_U
        all_logits = []
        W_U_T = transpose(self.W_U)
        for pos in range(seq_len):
            logits = matmul_vec(W_U_T, resid[pos])
            all_logits.append(logits)
            
        return all_logits, attn_0, attn_1


# =====================================================================
# 3. Experiment Execution: Repeated Random Sequence Benchmark
# =====================================================================

def run_experiment():
    print("=" * 78)
    print("PHASE 33 — TOPIC 03: ATTENTION HEAD CIRCUITS & INDUCTION HEADS")
    print("Transformer Circuits Framework: QK-Circuit and OV-Circuit Composition")
    print("=" * 78)

    model = InductionTransformer(vocab_size=12, d_model=32, d_head=16)
    
    # Generate repeated sequence: [A, B, C, D, E, A, B, C, D, E]
    pattern_len = 5
    vocab_tokens = [1, 3, 5, 2, 4]  # Distinct random tokens
    repeated_tokens = vocab_tokens + vocab_tokens  # Length 10
    
    token_str = " ".join(str(t) for t in repeated_tokens)
    print(f"\n[Test Sequence]: [ {token_str} ] (Pattern length = {pattern_len})")
    print("Tokens 0..4: First exposure (no previous repetition to copy)")
    print("Tokens 5..9: Repeated sequence (Induction Head should execute [A] -> [B] completion)\n")

    # Run Standard (Intact) Model
    logits, attn_l0, attn_l1 = model.forward(repeated_tokens)
    
    print(f"{'Position':<8} | {'Token':<7} | {'Target Next':<12} | {'Predicted':<10} | {'Prob(Target)':<12} | {'Mechanism'}")
    print("-" * 72)
    
    first_half_correct = 0
    second_half_correct = 0
    
    for pos in range(len(repeated_tokens) - 1):
        curr_token = repeated_tokens[pos]
        target_next = repeated_tokens[pos + 1]
        probs = softmax(logits[pos])
        pred = argmax(probs)
        target_prob = probs[target_next]
        
        is_repeat_phase = pos >= pattern_len
        mechanism = "Induction Head ([A]->[B])" if is_repeat_phase else "Prefix (Cold Start)"
        
        if pred == target_next:
            if is_repeat_phase:
                second_half_correct += 1
            else:
                first_half_correct += 1
                
        print(f"Pos {pos:<4} | Tok {curr_token:<3} | Target: {target_next:<4} | Pred: {pred:<5} | {target_prob*100:>8.2f}%    | {mechanism}")

    first_half_acc = first_half_correct / (pattern_len - 1)
    second_half_acc = second_half_correct / pattern_len
    
    print("-" * 72)
    print(f"Prefix Cold-Start Accuracy:     {first_half_acc*100:.1f}%")
    print(f"Repeated Phase Induction Acc:   {second_half_acc*100:.1f}%\n")

    # Inspect Attention Pattern of the Induction Head at pos 5 (token 'A' in second half)
    # Target is token 'B' at pos 1. The induction head at pos 5 must attend to pos 1!
    query_pos = pattern_len  # pos 5
    attn_weights_l1 = attn_l1[query_pos]
    
    print(f"[Attention Weights of Layer 1 Induction Head at Pos {query_pos} (Token {repeated_tokens[query_pos]})]")
    for j in range(query_pos + 1):
        bar = "█" * int(attn_weights_l1[j] * 25)
        print(f"  -> Attending to Pos {j} (Tok {repeated_tokens[j]}): {attn_weights_l1[j]*100:5.2f}% {bar}")

    # Ablation / Knockout Experiments
    print("\n[Ablation Experiments: Proving Circuit Composition]")
    probs_intact = softmax(logits[query_pos])
    logits_ablated_l0, _, _ = model.forward(repeated_tokens, ablate_l0=True)
    probs_ablated_l0 = softmax(logits_ablated_l0[query_pos])
    acc_ablated_l0 = 1.0 if argmax(probs_ablated_l0) == repeated_tokens[query_pos + 1] else 0.0

    logits_ablated_l1, _, _ = model.forward(repeated_tokens, ablate_l1=True)
    probs_ablated_l1 = softmax(logits_ablated_l1[query_pos])
    acc_ablated_l1 = 1.0 if argmax(probs_ablated_l1) == repeated_tokens[query_pos + 1] else 0.0

    print(f"1. Intact Model Target Probability at Pos {query_pos}:        {probs_intact[repeated_tokens[query_pos+1]]*100:.2f}%")
    print(f"2. Ablating Layer 0 (Previous-Token Head) Target Prob:  {probs_ablated_l0[repeated_tokens[query_pos+1]]*100:.2f}%")
    print(f"3. Ablating Layer 1 (Induction Head) Target Prob:       {probs_ablated_l1[repeated_tokens[query_pos+1]]*100:.2f}%")

    # Mathematical Assertions
    assert second_half_acc >= 0.80, f"Expected induction accuracy >= 80%, got {second_half_acc*100:.1f}%"
    assert attn_weights_l1[1] > 0.40, f"Expected Induction Head to attend to token after match (pos 1), got {attn_weights_l1[1]:.2f}"
    assert probs_ablated_l0[repeated_tokens[query_pos+1]] < 0.25, "Knocking out Layer 0 should break induction circuit!"

    print("\n>>> ALL TESTS PASSED: Induction head circuit explicitly verified!")
    print("=" * 78)

if __name__ == "__main__":
    run_experiment()
