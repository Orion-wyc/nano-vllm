import unittest
from types import SimpleNamespace

import torch

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.model_runner import ModelRunner, _stack_kv_blocks
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class FakeModelRunner:
    def __init__(self):
        self.calls = []

    def call(self, name, *args):
        self.calls.append((name, args))
        if name == "run":
            return [7] * len(args[0])
        if name == "export_kv":
            return torch.zeros(2, 1, len(args[0]), 4, 1, 2)
        if name == "export_state":
            return ["state", args[0]]
        return None


def make_engine():
    engine = object.__new__(LLMEngine)
    engine.config = SimpleNamespace(
        kvcache_block_size=4,
        max_num_seqs=2,
        max_num_batched_tokens=4,
        max_model_len=64,
        num_kvcache_blocks=8,
        enable_prefix_caching=False,
        eos=-1,
    )
    engine.scheduler = Scheduler(engine.config)
    engine.model_runner = FakeModelRunner()
    return engine


class PrefixImportTests(unittest.TestCase):
    def setUp(self):
        self.block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.block_size

    def test_import_prefix_blocks_allocates_blocks_and_state_slot(self):
        manager = BlockManager(num_blocks=8, block_size=4, max_num_seqs=2)
        seq = Sequence([1, 2, 3, 4, 5, 6], SamplingParams())
        seq.num_cached_tokens = 3

        block_ids = manager.import_prefix_blocks(seq, 2)

        self.assertEqual(len(block_ids), 2)
        self.assertEqual(seq.block_table, block_ids)
        self.assertEqual(seq.state_slot, 0)
        self.assertEqual(manager.used_state_slots, {0})
        self.assertEqual(manager.used_block_ids, set(block_ids))
        self.assertEqual(len(manager.free_block_ids), 6)
        self.assertEqual(seq.num_cached_tokens, 3)

    def test_import_prefix_blocks_uses_free_state_slots(self):
        manager = BlockManager(num_blocks=8, block_size=4, max_num_seqs=2)
        first = Sequence([1] * 6, SamplingParams())
        second = Sequence([2] * 6, SamplingParams())

        first_ids = manager.import_prefix_blocks(first, 1)
        second_ids = manager.import_prefix_blocks(second, 2)

        self.assertEqual(first.state_slot, 0)
        self.assertEqual(second.state_slot, 1)
        self.assertEqual(manager.used_state_slots, {0, 1})
        self.assertFalse(set(first_ids) & set(second_ids))

        manager.deallocate(first)
        third = Sequence([3] * 6, SamplingParams())
        manager.import_prefix_blocks(third, 1)
        self.assertEqual(third.state_slot, 0)

    def test_import_prefix_blocks_with_zero_blocks(self):
        manager = BlockManager(num_blocks=8, block_size=4, max_num_seqs=2)
        seq = Sequence([1, 2, 3], SamplingParams())

        block_ids = manager.import_prefix_blocks(seq, 0)

        self.assertEqual(block_ids, [])
        self.assertEqual(seq.block_table, [])
        self.assertEqual(seq.state_slot, 0)
        self.assertEqual(seq.num_cached_tokens, 0)

    def test_stack_kv_blocks_accepts_5d_and_6d(self):
        six_d = torch.zeros(2, 3, 1, 4, 5, 6)
        five_d = torch.ones(2, 3, 4, 5, 6)

        stacked = _stack_kv_blocks([six_d, five_d])

        self.assertEqual(tuple(stacked.shape), (2, 3, 2, 4, 5, 6))
        self.assertTrue(torch.equal(stacked[:, :, 0], six_d.squeeze(2)))
        self.assertTrue(torch.equal(stacked[:, :, 1], five_d))

    def test_stack_kv_blocks_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            _stack_kv_blocks([])
        with self.assertRaises(ValueError):
            _stack_kv_blocks([torch.zeros(2, 3, 4)])
        with self.assertRaises(ValueError):
            _stack_kv_blocks([torch.zeros(2, 3, 2, 4, 5, 6)])

    def test_import_kv_blocks_delegates_stacked_tensor(self):
        runner = object.__new__(ModelRunner)
        captured = {}
        runner.import_kv = lambda block_ids, kv: captured.update(block_ids=block_ids, kv=kv)

        blocks = [torch.zeros(2, 1, 4, 1, 2), torch.ones(2, 1, 4, 1, 2)]
        runner.import_kv_blocks([3, 5], blocks)

        self.assertEqual(captured["block_ids"], [3, 5])
        self.assertEqual(tuple(captured["kv"].shape), (2, 1, 2, 4, 1, 2))
        with self.assertRaises(ValueError):
            runner.import_kv_blocks([3], blocks)


class PrefillWithPrefixTests(unittest.TestCase):
    def setUp(self):
        self.block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.block_size

    def test_imports_prefix_and_emits_block_snapshot(self):
        engine = make_engine()
        boundaries = []

        def on_block_complete(seq):
            boundaries.append((seq.num_cached_tokens, seq.state_snapshot))

        prompt = list(range(10))
        result = engine.prefill_with_prefix(
            prompt,
            SamplingParams(max_tokens=8),
            prefix={
                "num_blocks": 1,
                "num_tokens": 4,
                "block_tensors": [torch.zeros(2, 1, 1, 4, 1, 2)],
                "state": ["prefix-state"],
            },
            on_block_complete=on_block_complete,
            state_snapshot=True,
        )

        names = [name for name, _ in engine.model_runner.calls]
        self.assertEqual(names[:3], ["import_kv_blocks", "import_state", "run"])
        self.assertEqual(engine.model_runner.calls[1][1][1], ["prefix-state"])
        self.assertEqual(result["prefix_blocks"], 1)
        self.assertEqual(result["prefix_tokens"], 4)
        self.assertEqual(len(result["block_table"]), 3)
        self.assertEqual(boundaries, [(8, ["state", 0])])
        self.assertEqual(result["token_ids"], prompt + [7])
        self.assertFalse(result["finished"])

    def test_reprocesses_last_token_when_prefix_covers_prompt(self):
        engine = make_engine()
        result = engine.prefill_with_prefix(
            [1, 2, 3, 4],
            SamplingParams(max_tokens=4),
            prefix=SimpleNamespace(
                num_blocks=1,
                num_tokens=4,
                block_tensors=[torch.zeros(2, 1, 1, 4, 1, 2)],
            ),
            hold_blocks=False,
        )

        self.assertEqual(result["prefix_tokens"], 4)
        self.assertEqual(len(result["block_table"]), 1)
        self.assertEqual(result["token_ids"], [1, 2, 3, 4, 7])

    def test_rejects_misaligned_or_oversized_prefix(self):
        engine = make_engine()
        with self.assertRaises(ValueError):
            engine.prefill_with_prefix(
                [1] * 8,
                SamplingParams(max_tokens=4),
                prefix={"num_blocks": 1, "num_tokens": 3, "block_tensors": []},
            )
        with self.assertRaises(ValueError):
            engine.prefill_with_prefix(
                [1] * 3,
                SamplingParams(max_tokens=4),
                prefix={"num_blocks": 1, "num_tokens": 4, "block_tensors": []},
            )


if __name__ == "__main__":
    unittest.main()
