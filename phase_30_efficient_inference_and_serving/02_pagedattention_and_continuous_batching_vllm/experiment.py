#!/usr/bin/env python3
"""
Topic 30.2 Experiment: PagedAttention Block Manager & Continuous Batching Scheduler
Phase 30: Efficient Inference & Serving

This experiment implements a pure Python 3 (standard library only) engine for:
1. PagedAttention Physical Block Pool and Logical Block Table mapping.
2. Dynamic on-demand block allocation and Copy-on-Write (CoW) fork semantics.
3. Continuous / Iteration-Level Batching scheduler (Orca/vLLM style).
4. Memory fragmentation benchmark comparing Contiguous pre-allocation vs PagedAttention.
"""

from typing import Dict, List, Optional, Set, Tuple
import collections
import random

class PhysicalBlock:
    """Represents a single physical memory block in GPU VRAM."""
    def __init__(self, block_id: int, block_size: int):
        self.block_id = block_id
        self.block_size = block_size
        self.ref_count = 0
        self.tokens: List[int] = []

    def is_full(self) -> bool:
        return len(self.tokens) >= self.block_size

    def num_free_slots(self) -> int:
        return self.block_size - len(self.tokens)

    def reset(self):
        self.ref_count = 0
        self.tokens.clear()


class BlockManager:
    """
    Manages the physical block pool, allocating, freeing, and performing
    Copy-on-Write (CoW) on physical KV blocks.
    """
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.blocks = [PhysicalBlock(i, block_size) for i in range(num_blocks)]
        self.free_blocks = collections.deque(range(num_blocks))

    def allocate(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("Out of Physical Blocks: GPU VRAM exhausted!")
        bid = self.free_blocks.popleft()
        block = self.blocks[bid]
        block.reset()
        block.ref_count = 1
        return bid

    def free(self, block_id: int):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            block.reset()
            self.free_blocks.append(block_id)

    def fork(self, block_id: int):
        """Increments reference counter for sharing during parallel sampling."""
        self.blocks[block_id].ref_count += 1

    def append_token(self, block_table: List[int], token_id: int) -> List[int]:
        """
        Appends a token to the sequence's logical block table.
        Allocates new blocks or triggers Copy-on-Write if current block is shared.
        """
        if not block_table:
            new_bid = self.allocate()
            block_table.append(new_bid)

        last_bid = block_table[-1]
        last_block = self.blocks[last_bid]

        # Case 1: Block is full -> Allocate new block
        if last_block.is_full():
            new_bid = self.allocate()
            self.blocks[new_bid].tokens.append(token_id)
            block_table.append(new_bid)
            return block_table

        # Case 2: Block has room but is shared (ref_count > 1) -> Copy-on-Write!
        if last_block.ref_count > 1:
            new_bid = self.allocate()
            # Copy contents from shared block to new private block
            self.blocks[new_bid].tokens = list(last_block.tokens)
            self.blocks[new_bid].tokens.append(token_id)
            # Decrement reference count on shared block
            self.free(last_bid)
            block_table[-1] = new_bid
            return block_table

        # Case 3: Block has room and is private -> directly append
        last_block.tokens.append(token_id)
        return block_table


class Request:
    """Represents a generative inference query with prompt and generation budget."""
    def __init__(self, req_id: int, prompt_len: int, max_gen_tokens: int):
        self.req_id = req_id
        self.prompt_len = prompt_len
        self.max_gen_tokens = max_gen_tokens
        self.generated_tokens = 0
        self.block_table: List[int] = []
        self.is_finished = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_len + self.generated_tokens


class ContinuousBatchingEngine:
    """
    Iteration-Level Continuous Batching scheduler using PagedAttention.
    """
    def __init__(self, block_manager: BlockManager, max_batch_size: int = 16):
        self.bm = block_manager
        self.max_batch_size = max_batch_size
        self.waiting_queue: collections.deque[Request] = collections.deque()
        self.running_batch: List[Request] = []
        self.completed_requests: List[Request] = []
        self.iteration = 0

    def add_request(self, req: Request):
        self.waiting_queue.append(req)

    def step(self) -> Tuple[int, int]:
        """
        Executes one forward decode step across the running batch.
        Admits new requests, generates 1 token per sequence, and evicts completed sequences.
        """
        self.iteration += 1

        # 1. Admit new requests if slots and memory are available
        while len(self.running_batch) < self.max_batch_size and self.waiting_queue:
            candidate = self.waiting_queue[0]
            blocks_needed = (candidate.prompt_len + self.bm.block_size - 1) // self.bm.block_size
            if len(self.bm.free_blocks) >= blocks_needed:
                req = self.waiting_queue.popleft()
                # Prefill allocation: populate logical blocks for prompt
                for _ in range(blocks_needed):
                    bid = self.bm.allocate()
                    req.block_table.append(bid)
                # Fill prompt tokens
                curr_block_idx = 0
                for tok in range(req.prompt_len):
                    bid = req.block_table[tok // self.bm.block_size]
                    self.bm.blocks[bid].tokens.append(100 + tok)
                self.running_batch.append(req)
            else:
                break  # Memory full, wait for active requests to finish

        # 2. Forward step: generate 1 token per running sequence
        tokens_generated = len(self.running_batch)
        finished_this_step: List[Request] = []

        for req in self.running_batch:
            # Generate next token
            next_token = 1000 + req.generated_tokens
            req.block_table = self.bm.append_token(req.block_table, next_token)
            req.generated_tokens += 1

            if req.generated_tokens >= req.max_gen_tokens:
                req.is_finished = True
                finished_this_step.append(req)

        # 3. Evict completed requests and return physical blocks to pool
        for req in finished_this_step:
            self.running_batch.remove(req)
            for bid in req.block_table:
                self.bm.free(bid)
            self.completed_requests.append(req)

        return tokens_generated, len(self.running_batch)


def run_pagedattention_simulation():
    print("=" * 80)
    print("PHASE 30: TOPIC 02 EXPERIMENT — PAGEDATTENTION & CONTINUOUS BATCHING")
    print("=" * 80)

    # 1. Verify Copy-on-Write Mechanics
    print("\n[1] DEMONSTRATING COPY-ON-WRITE (CoW) FOR PARALLEL SAMPLING:")
    bm = BlockManager(num_blocks=32, block_size=4)
    table_parent = []
    # Append 3 prompt tokens (fits in 1 block of size 4)
    for tok in [10, 20, 30]:
        table_parent = bm.append_token(table_parent, tok)

    parent_block = table_parent[0]
    print(f"  Parent sequence allocated Block ID: {parent_block}, Tokens: {bm.blocks[parent_block].tokens}, Ref Count: {bm.blocks[parent_block].ref_count}")

    # Fork to child sequence
    table_child = list(table_parent)
    bm.fork(parent_block)
    print(f"  Forked to child! Parent block ref_count is now: {bm.blocks[parent_block].ref_count}")

    # Child appends a new token -> Triggers Copy-on-Write
    table_child = bm.append_token(table_child, token_id=99)
    child_block = table_child[0]
    print(f"  Child generated token 99! CoW triggered!")
    print(f"  Parent Block ID {parent_block}: Tokens: {bm.blocks[parent_block].tokens}, Ref Count: {bm.blocks[parent_block].ref_count}")
    print(f"  Child  Block ID {child_block}: Tokens: {bm.blocks[child_block].tokens}, Ref Count: {bm.blocks[child_block].ref_count}")
    assert parent_block != child_block, "CoW must allocate distinct physical block for child!"
    assert bm.blocks[parent_block].ref_count == 1
    assert bm.blocks[child_block].ref_count == 1

    # 2. Continuous Batching Simulation
    print("\n[2] SIMULATING CONTINUOUS BATCHING SCHEDULER:")
    pool_blocks = 128
    block_size = 16
    bm_serving = BlockManager(num_blocks=pool_blocks, block_size=block_size)
    engine = ContinuousBatchingEngine(bm_serving, max_batch_size=4)

    # Seed 10 requests with variable generation lengths
    random.seed(42)
    requests = [
        Request(req_id=i, prompt_len=random.randint(10, 30), max_gen_tokens=random.randint(5, 25))
        for i in range(10)
    ]
    for r in requests:
        engine.add_request(r)

    print(f"  Enqueued {len(requests)} requests. Total physical blocks: {pool_blocks} (Block size = {block_size}).")
    step_log = []
    while engine.running_batch or engine.waiting_queue:
        toks_gen, running_count = engine.step()
        free_blocks = len(engine.bm.free_blocks)
        if engine.iteration % 5 == 0 or not (engine.running_batch or engine.waiting_queue):
            print(f"  Step {engine.iteration:3d} | Active Running: {running_count} | Queued: {len(engine.waiting_queue):2d} | Free Blocks: {free_blocks:3d} | Finished: {len(engine.completed_requests):2d}")

    print(f"  All {len(engine.completed_requests)} requests completed successfully across {engine.iteration} steps!")

    # 3. Memory Fragmentation Benchmark: Contiguous vs PagedAttention
    print("\n[3] MEMORY FRAGMENTATION COMPARISON BENCHMARK:")
    num_test_requests = 100
    s_max = 2048
    random.seed(1337)
    sampled_lengths = [random.randint(100, 800) for _ in range(num_test_requests)]

    bytes_per_token = 320 * 1024  # 320 KB (LLaMA-70B GQA)
    total_tokens_used = sum(sampled_lengths)

    # Contiguous Allocation (Pre-allocates s_max for every request)
    contiguous_allocated_tokens = num_test_requests * s_max
    contiguous_waste_pct = (1.0 - (total_tokens_used / contiguous_allocated_tokens)) * 100.0

    # PagedAttention (Block size = 16)
    paged_blocks = sum((length + 15) // 16 for length in sampled_lengths)
    paged_allocated_tokens = paged_blocks * 16
    paged_waste_pct = (1.0 - (total_tokens_used / paged_allocated_tokens)) * 100.0

    print(f"  Workload: {num_test_requests} requests (Average Context: {total_tokens_used / num_test_requests:.1f} tokens, Max Context: {s_max})")
    print(f"  • Contiguous Pre-allocation:")
    print(f"      Total Allocated : {contiguous_allocated_tokens * bytes_per_token / 1e9:.2f} GB")
    print(f"      Actual Used     : {total_tokens_used * bytes_per_token / 1e9:.2f} GB")
    print(f"      Memory Waste    : {contiguous_waste_pct:.2f}% (Internal Fragmentation)")
    print(f"  • PagedAttention (Block Size = 16):")
    print(f"      Total Allocated : {paged_allocated_tokens * bytes_per_token / 1e9:.2f} GB")
    print(f"      Actual Used     : {total_tokens_used * bytes_per_token / 1e9:.2f} GB")
    print(f"      Memory Waste    : {paged_waste_pct:.2f}%")
    print(f"  • Memory Savings Factor: {contiguous_allocated_tokens / paged_allocated_tokens:.2f}x reduction in required VRAM!")

    assert paged_waste_pct < 3.0, "PagedAttention memory waste must be under 3%!"
    assert contiguous_waste_pct > 70.0, "Contiguous pre-allocation waste must exceed 70%!"
    print("\n✓ ALL PAGEDATTENTION & CONTINUOUS BATCHING INVARIANTS VERIFIED.")

if __name__ == "__main__":
    run_pagedattention_simulation()
