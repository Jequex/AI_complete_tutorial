#!/usr/bin/env python3
"""
Topic 02: Activation Patching & Causal Tracing
----------------------------------------------
A pure Python 3 standard library implementation demonstrating:
1. A multi-layer Transformer model executing factual association:
   Clean input:     "The Eiffel Tower is located in [Paris]"
   Corrupted input: "The Colosseum is located in [Rome]"
2. Clean run, Corrupted run, and Causal Activation Patching.
3. Systematically patching residual stream, MLP, and Attention activations
   across all token positions and network layers.
4. Computing Average Treatment Effect (ATE) and Normalized Logit Difference
   Restoration:
     Restoration = (Logit_patched - Logit_corrupted) / (Logit_clean - Logit_corrupted)
5. Rendering an ASCII Causal Tracing Heatmap revealing:
   - Early/Mid MLP layers at the Subject token (storing/recalling factual memory).
   - Late Attention layers at the Last token (routing the answer to output logits).

Zero external dependencies (pure Python standard library).
"""

import math
import random

random.seed(1337)

# =====================================================================
# 1. Linear Algebra Primitives (Pure Python)
# =====================================================================

def dot(u, v):
    return sum(a * b for a, b in zip(u, v))

def vec_add(u, v):
    return [a + b for a, b in zip(u, v)]

def vec_scale(u, s):
    return [a * s for a in u]

def matmul_vec(W, x):
    return [dot(row, x) for row in W]

def softmax(logits):
    max_l = max(logits)
    exp_l = [math.exp(l - max_l) for l in logits]
    sum_exp = sum(exp_l)
    return [e / sum_exp for e in exp_l]

def relu(x):
    return [max(0.0, v) for v in x]

def random_matrix(rows, cols, scale=0.4):
    return [[random.gauss(0, scale / math.sqrt(cols)) for _ in range(cols)] for _ in range(rows)]


# =====================================================================
# 2. Toy Factual Recall Transformer
# =====================================================================

class ToyFactualTransformer:
    """
    A 4-layer Transformer with:
    - Token vocab: 0='The', 1='Eiffel', 2='Colosseum', 3='in', 4='Paris', 5='Rome'
    - Sequence length S = 3: [Subject, 'in', 'is']
    - Clean Prompt:     [1, 3] -> Target: 4 ('Paris')
    - Corrupted Prompt: [2, 3] -> Target: 5 ('Rome')
    """
    def __init__(self, vocab_size=8, d_model=16, num_layers=4):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Embeddings
        self.token_embeddings = {
            i: [random.gauss(0, 0.5) for _ in range(d_model)] for i in range(vocab_size)
        }
        
        # Embed distinctive subject features into Eiffel (1) and Colosseum (2)
        self.fact_vector = [0.0] * d_model
        self.fact_vector[2] = 2.0  # Key dimension for Paris vs Rome
        self.token_embeddings[1][2] = +2.5 # Eiffel -> Paris
        self.token_embeddings[2][2] = -2.5 # Colosseum -> Rome

        # Layers: Each layer has an Attention block and an MLP block
        self.attn_weights = []
        self.mlp_w1 = []
        self.mlp_w2 = []
        
        for l in range(num_layers):
            self.attn_weights.append(random_matrix(d_model, d_model, scale=0.3))
            self.mlp_w1.append(random_matrix(d_model * 2, d_model, scale=0.4))
            self.mlp_w2.append(random_matrix(d_model, d_model * 2, scale=0.4))

        # Designate Layer 1 MLP as the "Factual Knowledge Recall" module
        # It expands the subject feature into the city feature along dimension 5
        self.mlp_w1[1][5] = [0.0] * d_model
        self.mlp_w1[1][5][2] = 2.0  # Reads subject feature at dim 2
        self.mlp_w2[1][5] = [0.0] * (d_model * 2)
        self.mlp_w2[1][5][5] = 3.0  # Writes city representation at dim 5
        
        # Designate Layer 3 Attention as the "Information Mover Head"
        # It copies the factual signal from token 0 (Subject) to token 1 (Last token)
        for i in range(d_model):
            self.attn_weights[3][i][i] = 1.2

        # Final unembedding projection W_U
        self.W_unembed = random_matrix(vocab_size, d_model, scale=0.3)
        self.W_unembed[4] = [0.0] * d_model
        self.W_unembed[4][5] = +4.0  # Paris logit reads dim 5
        self.W_unembed[5] = [0.0] * d_model
        self.W_unembed[5][5] = -4.0  # Rome logit reads -dim 5

    def forward(self, token_ids, patch_spec=None, cached_activations=None):
        """
        Forward pass with optional activation patching.
        patch_spec: dict with keys:
            'layer': int
            'pos': int
            'component': 'resid', 'mlp', or 'attn'
            'clean_act': vector to inject
        """
        seq_len = len(token_ids)
        # Residual stream activations: list of vectors per token position
        resid = [list(self.token_embeddings[tid]) for tid in token_ids]
        
        cache = {
            'resid': [ [list(r) for r in resid] ],
            'attn': [],
            'mlp': []
        }
        
        for l in range(self.num_layers):
            # Check patching on residual stream input at layer l
            if patch_spec and patch_spec['component'] == 'resid' and patch_spec['layer'] == l:
                pos = patch_spec['pos']
                resid[pos] = list(patch_spec['clean_act'])

            # 1. Attention Block: token 1 attends back to token 0
            layer_attn_out = []
            for pos in range(seq_len):
                if pos == 0:
                    # Token 0 attends only to itself
                    attn_vec = matmul_vec(self.attn_weights[l], resid[0])
                else:
                    # Token 1 attends 70% to token 0 (subject) and 30% to itself
                    attn_0 = matmul_vec(self.attn_weights[l], resid[0])
                    attn_1 = matmul_vec(self.attn_weights[l], resid[pos])
                    attn_vec = vec_add(vec_scale(attn_0, 0.7), vec_scale(attn_1, 0.3))
                layer_attn_out.append(attn_vec)
                
            # Check patching on Attention output
            if patch_spec and patch_spec['component'] == 'attn' and patch_spec['layer'] == l:
                pos = patch_spec['pos']
                layer_attn_out[pos] = list(patch_spec['clean_act'])
                
            cache['attn'].append([list(a) for a in layer_attn_out])
            
            # Add Attention to residual stream
            for pos in range(seq_len):
                resid[pos] = vec_add(resid[pos], vec_scale(layer_attn_out[pos], 0.6))
                
            # 2. MLP Block (Position-wise feed-forward)
            layer_mlp_out = []
            for pos in range(seq_len):
                h1 = relu(matmul_vec(self.mlp_w1[l], resid[pos]))
                h2 = matmul_vec(self.mlp_w2[l], h1)
                layer_mlp_out.append(h2)
                
            # Check patching on MLP output
            if patch_spec and patch_spec['component'] == 'mlp' and patch_spec['layer'] == l:
                pos = patch_spec['pos']
                layer_mlp_out[pos] = list(patch_spec['clean_act'])
                
            cache['mlp'].append([list(m) for m in layer_mlp_out])
            
            # Add MLP to residual stream
            for pos in range(seq_len):
                resid[pos] = vec_add(resid[pos], vec_scale(layer_mlp_out[pos], 0.7))
                
            cache['resid'].append([list(r) for r in resid])

        # Final Logits at the last token position
        last_resid = resid[-1]
        logits = matmul_vec(self.W_unembed, last_resid)
        return logits, cache


# =====================================================================
# 3. Causal Tracing Pipeline
# =====================================================================

def run_experiment():
    print("=" * 78)
    print("PHASE 33 — TOPIC 02: ACTIVATION PATCHING & CAUSAL TRACING")
    print("Isolating Factual Recall and Information Flow Circuits")
    print("=" * 78)

    model = ToyFactualTransformer(vocab_size=8, d_model=16, num_layers=4)
    
    clean_tokens = [1, 3]      # ["Eiffel", "in"]
    corrupted_tokens = [2, 3]  # ["Colosseum", "in"]
    
    clean_target = 4      # Paris
    corrupted_target = 5  # Rome
    
    # 1. Clean Run
    clean_logits, clean_cache = model.forward(clean_tokens)
    clean_prob = softmax(clean_logits)
    clean_diff = clean_logits[clean_target] - clean_logits[corrupted_target]
    
    # 2. Corrupted Run
    corrupt_logits, corrupt_cache = model.forward(corrupted_tokens)
    corrupt_prob = softmax(corrupt_logits)
    corrupt_diff = corrupt_logits[clean_target] - corrupt_logits[corrupted_target]
    
    print("\n[Baseline Model Executions]")
    print(f"Clean Run:     P('Paris') = {clean_prob[clean_target]*100:.2f}%, P('Rome') = {clean_prob[corrupted_target]*100:.2f}% | Logit Diff = {clean_diff:+.3f}")
    print(f"Corrupted Run: P('Paris') = {corrupt_prob[clean_target]*100:.2f}%, P('Rome') = {corrupt_prob[corrupted_target]*100:.2f}% | Logit Diff = {corrupt_diff:+.3f}")

    total_effect = clean_diff - corrupt_diff
    print(f"Total Causal Effect (Clean - Corrupted Logit Diff) = {total_effect:.3f}\n")

    # 3. Activation Patching Grid (Causal Tracing)
    # Positions: 0 = Subject ("Eiffel"), 1 = Last Token ("in")
    token_labels = ["Subject [Eiffel]", "Last Token [in]"]
    components = ['mlp', 'attn', 'resid']
    
    trace_results = {comp: [[0.0 for _ in range(len(clean_tokens))] for _ in range(model.num_layers)] for comp in components}
    
    for comp in components:
        for l in range(model.num_layers):
            for pos in range(len(clean_tokens)):
                clean_act = clean_cache[comp][l][pos] if comp != 'resid' else clean_cache['resid'][l][pos]
                patch_spec = {
                    'layer': l,
                    'pos': pos,
                    'component': comp,
                    'clean_act': clean_act
                }
                patched_logits, _ = model.forward(corrupted_tokens, patch_spec=patch_spec)
                patched_diff = patched_logits[clean_target] - patched_logits[corrupted_target]
                
                # Normalized restoration metric
                restoration = (patched_diff - corrupt_diff) / total_effect
                trace_results[comp][l][pos] = max(0.0, restoration)

    # 4. Display Heatmaps
    for comp in ['mlp', 'attn']:
        comp_name = "MLP Layer Activations (Knowledge Storage)" if comp == 'mlp' else "Attention Head Activations (Information Routing)"
        print(f"\n--- Causal Tracing Heatmap: {comp_name} ---")
        print(f"{'Layer':<8} | " + " | ".join(f"{lbl:^22}" for lbl in token_labels))
        print("-" * 60)
        for l in range(model.num_layers):
            row_str = f"Layer {l:<2} | "
            for pos in range(len(clean_tokens)):
                val = trace_results[comp][l][pos]
                # ASCII visual bar
                bar_len = int(val * 15)
                bar = "█" * bar_len + "░" * (15 - bar_len)
                row_str += f"{bar} ({val*100:5.1f}%) | "
            print(row_str)

    # 5. Scientific Verifications and Assertions
    mlp_subj_l1 = trace_results['mlp'][1][0]   # Early/mid MLP at subject
    attn_last_l3 = trace_results['attn'][3][1] # Late Attention at last token
    
    print("\n[Mechanistic Interpretability Findings]")
    print(f"1. Early/Mid MLP at Subject token (Layer 1, pos 0) restoration: {mlp_subj_l1*100:.1f}%.")
    print(f"   -> Verifies localized factual recall in intermediate feed-forward weights (ROME hypothesis).")
    print(f"2. Late Attention at Last token (Layer 3, pos 1) restoration: {attn_last_l3*100:.1f}%.")
    print(f"   -> Verifies information mover routing from subject to prompt terminus.")

    assert mlp_subj_l1 > 0.40, f"Expected high causal effect in Layer 1 MLP at subject (got {mlp_subj_l1:.2f})"
    assert attn_last_l3 > 0.50, f"Expected high causal effect in Layer 3 Attention at last token (got {attn_last_l3:.2f})"
    
    print("\n>>> ALL TESTS PASSED: Causal tracing simulation executed and verified successfully!")
    print("=" * 78)

if __name__ == "__main__":
    run_experiment()
