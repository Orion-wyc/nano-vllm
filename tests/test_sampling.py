import unittest

import torch

from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams
from nanovllm.models.qwen3_5 import RMSNormGated


class SamplingTests(unittest.TestCase):
    def test_greedy_and_repetition_penalty(self):
        logits = torch.tensor([[5.0, 4.0, -2.0], [5.0, 4.0, -2.0]])
        seqs = [Sequence([0], SamplingParams(temperature=0)),
                Sequence([0], SamplingParams(temperature=0, repetition_penalty=2))]
        self.assertEqual(Sampler()(logits, seqs).tolist(), [0, 1])

    def test_top_k_and_top_p_keep_highest_token(self):
        logits = torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
        seqs = [Sequence([2], SamplingParams(top_k=1)),
                Sequence([2], SamplingParams(top_p=0.1))]
        self.assertEqual(Sampler()(logits, seqs).tolist(), [0, 0])

    def test_invalid_sampling_params(self):
        for kwargs in ({"top_p": 0}, {"top_k": 0}, {"repetition_penalty": 0}, {"temperature": -1}):
            with self.assertRaises(ValueError):
                SamplingParams(**kwargs)

    def test_gated_norm_matches_reference_cast_order(self):
        layer = RMSNormGated(4)
        x = torch.tensor([[0.2, -0.9, 1.7, 0.03]], dtype=torch.bfloat16)
        gate = torch.tensor([[0.7, -0.3, 1.2, 0.5]], dtype=torch.bfloat16)
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + layer.eps)
        expected = (layer.weight * normalized.to(x.dtype) * torch.nn.functional.silu(gate.float())).to(x.dtype)
        torch.testing.assert_close(layer(x, gate), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
