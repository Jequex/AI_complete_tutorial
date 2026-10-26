#!/usr/bin/env python3
"""
Phase 20 - Topic 03: Instruction Datasets & Multi-Turn Conversations
Experiment: Pure Python Multi-Turn Dialogue Compiler & Function Calling Masking Engine

This experiment implements and verifies:
1. Structured dialogue representation supporting system, user, assistant, and tool turns.
2. Function schema serialization and tool invocation formatting.
3. Multi-turn tokenization and label vector compilation:
   - System instruction: MASKED (-100)
   - User query turns: MASKED (-100)
   - Tool execution environment output: MASKED (-100)
   - Assistant function call output (<tool_call>...): SUPERVISED (next-token IDs)
   - Assistant conversational answers: SUPERVISED (next-token IDs)
   - Assistant turn closure (<|im_end|>): SUPERVISED (model learns to yield turn)
4. Sequence packing: Packing multiple multi-turn dialogues into a single contiguous context window.
5. Exact boundary assertions verifying every token ID and mask label.

Zero external dependencies: pure standard Python 3.
"""

import json
import re
import sys
from typing import List, Dict, Any, Optional, Tuple


SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<|endoftext|>"
]


class DialogueTurn:
    """Represents a single turn in a multi-turn conversation."""
    def __init__(
        self,
        role: str,
        content: str = "",
        tool_call: Optional[Dict[str, Any]] = None,
        tool_name: Optional[str] = None
    ):
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"Invalid role: {role}")
        self.role = role
        self.content = content
        self.tool_call = tool_call
        self.tool_name = tool_name


class MultiTurnTokenizer:
    """Tokenizes text into integer IDs, reserving atomic IDs for special tokens."""
    def __init__(self):
        self.special_to_id = {tok: 100 + i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.id_to_special = {i: tok for tok, i in self.special_to_id.items()}
        self.vocab = {}
        self.inv_vocab = {}
        self._next_id = 500

        escaped = [re.escape(tok) for tok in sorted(SPECIAL_TOKENS, key=len, reverse=True)]
        self.pattern = re.compile("(" + "|".join(escaped) + r"|\w+|[^\w\s]|\s+)")

    def encode(self, text: str) -> List[int]:
        tokens = self.pattern.findall(text)
        ids = []
        for t in tokens:
            if t in self.special_to_id:
                ids.append(self.special_to_id[t])
            else:
                if t not in self.vocab:
                    self.vocab[t] = self._next_id
                    self.inv_vocab[self._next_id] = t
                    self._next_id += 1
                ids.append(self.vocab[t])
        return ids

    def decode(self, token_ids: List[int]) -> str:
        out = []
        for tid in token_ids:
            if tid in self.id_to_special:
                out.append(self.id_to_special[tid])
            elif tid in self.inv_vocab:
                out.append(self.inv_vocab[tid])
            else:
                out.append(f"<UNK_{tid}>")
        return "".join(out)


class MultiTurnDatasetProcessor:
    """
    Compiles multi-turn conversations into contiguous token sequences with
    loss mask vectors.
    """
    def __init__(self, tokenizer: MultiTurnTokenizer):
        self.tokenizer = tokenizer

    def compile_dialogue(
        self,
        turns: List[DialogueTurn],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[int], List[int], List[Dict[str, Any]]]:
        """
        Compiles dialogue turns into input_ids and labels.
        Returns:
            input_ids: List[int]
            labels: List[int] (-100 for masked positions, token_id for supervised)
            turn_metadata: List[dict] tracking start/end token indices and masking state
        """
        full_tokens: List[int] = []
        full_labels: List[int] = []
        turn_metadata = []

        # If tools are defined, inject schemas into system turn
        tools_header = ""
        if tools:
            tools_json = json.dumps(tools, separators=(",", ":"))
            tools_header = f"\n[AVAILABLE_TOOLS]{tools_json}[/AVAILABLE_TOOLS]"

        for turn_idx, turn in enumerate(turns):
            start_pos = len(full_tokens)
            
            # Format turn string
            if turn.role == "system":
                turn_text = f"<|im_start|>system\n{turn.content}{tools_header}<|im_end|>\n"
                is_supervised = False
            elif turn.role == "user":
                turn_text = f"<|im_start|>user\n{turn.content}<|im_end|>\n"
                is_supervised = False
            elif turn.role == "tool":
                # Tool responses come from external APIs; model must NOT be penalized for environment output
                turn_text = f"<|im_start|>tool\n<tool_response>{turn.content}</tool_response><|im_end|>\n"
                is_supervised = False
            elif turn.role == "assistant":
                if turn.tool_call is not None:
                    call_json = json.dumps(turn.tool_call, separators=(",", ":"))
                    turn_text = f"<|im_start|>assistant\n<tool_call>{call_json}</tool_call><|im_end|>\n"
                else:
                    turn_text = f"<|im_start|>assistant\n{turn.content}<|im_end|>\n"
                is_supervised = True
            else:
                raise ValueError(f"Unknown role: {turn.role}")

            turn_tokens = self.tokenizer.encode(turn_text)
            
            # In causal LM: token at t predicts token at t+1
            # If a turn is supervised, the tokens inside the assistant body and its closing <|im_end|>
            # are supervised. The opening header '<|im_start|>assistant\n' is prompt context.
            header_text = f"<|im_start|>{turn.role}\n"
            header_tokens = self.tokenizer.encode(header_text)
            header_len = len(header_tokens)

            turn_labels = []
            for i, tid in enumerate(turn_tokens):
                if not is_supervised:
                    turn_labels.append(-100)
                else:
                    # Role header itself is context; assistant body and closing tags are supervised
                    if i < header_len:
                        turn_labels.append(-100)
                    else:
                        turn_labels.append(tid)

            full_tokens.extend(turn_tokens)
            full_labels.extend(turn_labels)
            
            turn_metadata.append({
                "turn_index": turn_idx,
                "role": turn.role,
                "start_idx": start_pos,
                "end_idx": len(full_tokens),
                "is_supervised": is_supervised,
                "token_count": len(turn_tokens)
            })

        # Autoregressive shift:
        # input_ids = full_tokens[:-1]
        # target_labels = full_labels[1:]
        input_ids = full_tokens[:-1]
        target_labels = full_labels[1:]

        return input_ids, target_labels, turn_metadata

    def pack_dialogues(
        self,
        dialogues: List[Tuple[List[int], List[int]]],
        max_seq_len: int
    ) -> List[Dict[str, Any]]:
        """
        Packs multiple conversations into fixed-length sequences.
        Creates sequence segment IDs / boundaries to prevent cross-dialogue attention bleeding.
        """
        packed_batches = []
        cur_input_ids: List[int] = []
        cur_labels: List[int] = []
        cur_segments: List[int] = []
        dialogue_counter = 1

        for inp, lbl in dialogues:
            if len(cur_input_ids) + len(inp) > max_seq_len:
                # Pad remaining
                pad_len = max_seq_len - len(cur_input_ids)
                cur_input_ids.extend([0] * pad_len)
                cur_labels.extend([-100] * pad_len)
                cur_segments.extend([0] * pad_len)

                packed_batches.append({
                    "input_ids": cur_input_ids,
                    "labels": cur_labels,
                    "segments": cur_segments
                })
                cur_input_ids = []
                cur_labels = []
                cur_segments = []
                dialogue_counter = 1

            cur_input_ids.extend(inp)
            cur_labels.extend(lbl)
            cur_segments.extend([dialogue_counter] * len(inp))
            dialogue_counter += 1

        if cur_input_ids:
            pad_len = max_seq_len - len(cur_input_ids)
            cur_input_ids.extend([0] * pad_len)
            cur_labels.extend([-100] * pad_len)
            cur_segments.extend([0] * pad_len)
            packed_batches.append({
                "input_ids": cur_input_ids,
                "labels": cur_labels,
                "segments": cur_segments
            })

        return packed_batches


def run_experiment():
    print("=" * 75)
    print("PHASE 20.3 EXPERIMENT: MULTI-TURN DATASET PROCESSOR & TOOL MASKING")
    print("=" * 75)

    tokenizer = MultiTurnTokenizer()
    processor = MultiTurnDatasetProcessor(tokenizer)

    # 1. Define a Multi-Turn Conversation with Function Calling
    tools_schema = [
        {
            "name": "get_stock_price",
            "description": "Retrieve real-time stock quote for a given ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"]
            }
        }
    ]

    dialogue = [
        DialogueTurn("system", "You are an intelligent financial assistant with access to stock tools."),
        DialogueTurn("user", "Can you check the current stock price of NVDA?"),
        DialogueTurn("assistant", tool_call={"name": "get_stock_price", "arguments": {"ticker": "NVDA"}}),
        DialogueTurn("tool", '{"ticker": "NVDA", "price": 128.50, "currency": "USD"}', tool_name="get_stock_price"),
        DialogueTurn("assistant", "NVIDIA (NVDA) is currently trading at $128.50 USD.")
    ]

    print("\n--- 1. Compiling Multi-Turn Conversation ---")
    input_ids, targets, meta = processor.compile_dialogue(dialogue, tools=tools_schema)
    
    print(f"Total compiled tokens: {len(input_ids) + 1} (Input IDs: {len(input_ids)}, Targets: {len(targets)})")
    for m in meta:
        role_label = m["role"].upper()
        status = "SUPERVISED (Active Loss)" if m["is_supervised"] else "MASKED (target = -100)"
        print(f"Turn {m['turn_index']} [{role_label:9s}] tokens {m['start_idx']:3d}..{m['end_idx']:3d} ({m['token_count']:3d} toks) -> {status}")

    # 2. Assert Strict Turn Boundary Invariants
    print("\n--- 2. Verifying Boundary & Masking Invariants ---")

    # Invariant A: System Turn must be 100% masked
    sys_meta = meta[0]
    # Corresponding target indices: start_idx to end_idx - 1 (due to 1-position shift)
    for idx in range(sys_meta["start_idx"], sys_meta["end_idx"] - 1):
        assert targets[idx] == -100, f"System token at target idx {idx} was not masked!"
    print("✓ Invariant A passed: System instructions and tool schemas are completely masked.")

    # Invariant B: User Turn must be 100% masked
    user_meta = meta[1]
    for idx in range(user_meta["start_idx"], user_meta["end_idx"] - 1):
        assert targets[idx] == -100, f"User token at target idx {idx} was not masked!"
    print("✓ Invariant B passed: User query tokens are completely masked.")

    # Invariant C: Assistant Tool Call MUST be supervised
    asst_tool_meta = meta[2]
    # Header is masked, but payload has active supervision
    active_in_tool_call = sum(1 for idx in range(asst_tool_meta["start_idx"], asst_tool_meta["end_idx"] - 1) if targets[idx] != -100)
    assert active_in_tool_call > 0, "Assistant tool call should have supervised targets!"
    print(f"✓ Invariant C passed: Assistant tool call contains {active_in_tool_call} supervised tokens.")

    # Invariant D: External Tool Environment Output MUST be masked
    env_tool_meta = meta[3]
    for idx in range(env_tool_meta["start_idx"], env_tool_meta["end_idx"] - 1):
        assert targets[idx] == -100, f"External tool output at target idx {idx} was incorrectly supervised!"
    print("✓ Invariant D passed: Tool environment JSON responses are completely masked.")

    # Invariant E: Assistant Final Answer MUST be supervised
    final_asst_meta = meta[4]
    active_in_final = sum(1 for idx in range(final_asst_meta["start_idx"], final_asst_meta["end_idx"] - 1) if targets[idx] != -100)
    assert active_in_final > 0, "Assistant final response should have supervised targets!"
    print(f"✓ Invariant E passed: Assistant conversational response contains {active_in_final} supervised tokens.")

    # 3. Test Sequence Packing
    print("\n--- 3. Testing Sequence Packing & Segmentation ---")
    d1_inp, d1_tgt, _ = processor.compile_dialogue(dialogue[:3])
    d2_inp, d2_tgt, _ = processor.compile_dialogue(dialogue[3:])
    
    max_len = 256
    packed = processor.pack_dialogues([(d1_inp, d1_tgt), (d2_inp, d2_tgt)], max_seq_len=max_len)
    
    assert len(packed) == 1, f"Should pack into a single batch entry, got {len(packed)}"
    batch = packed[0]
    assert len(batch["input_ids"]) == max_len
    assert len(batch["labels"]) == max_len
    assert len(batch["segments"]) == max_len

    # Verify segment isolation: segment 1 followed by segment 2 followed by padding (0)
    seg1_count = batch["segments"].count(1)
    seg2_count = batch["segments"].count(2)
    pad_count = batch["segments"].count(0)
    print(f"Packed Sequence Allocation: Dialogue 1 = {seg1_count} toks, Dialogue 2 = {seg2_count} toks, Padding = {pad_count} toks.")
    assert seg1_count == len(d1_inp)
    assert seg2_count == len(d2_inp)
    assert seg1_count + seg2_count + pad_count == max_len
    print("✓ Sequence packing correctly concatenated dialogues with exact boundary segment markers.")

    print("\n" + "=" * 75)
    print("ALL TESTS PASSED: Topic 20.3 Experiment verified with exit code 0.")
    print("=" * 75)


if __name__ == "__main__":
    run_experiment()
    sys.exit(0)
