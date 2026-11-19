import unittest

import torch

from nanovllm.serving.pd_mooncake import (
    load_manifest,
    load_prefill,
    publish_prefill,
    request_kv_key,
    request_state_key,
)
from nanovllm.serving.prefix_cache import PrefixCacheKeyBuilder


class FakeStore:

    def __init__(self):
        self.data = {}

    def put(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key)

    def get_batch(self, keys):
        return [self.data.get(key, b"") for key in keys]


class PDMooncakeTests(unittest.TestCase):

    def test_publish_and_load_manifest(self):
        store = FakeStore()
        builder = PrefixCacheKeyBuilder("scope", 4)
        token_ids = list(range(9))
        blocks = [
            torch.zeros(2, 2, 4, 1, 2),
            torch.zeros(2, 2, 4, 1, 2),
            torch.zeros(2, 2, 4, 1, 2),
        ]
        state = [(torch.ones(2), torch.zeros(3))]
        manifest = publish_prefill(
            store,
            builder,
            "xfer-test",
            "engine-1",
            token_ids,
            8,
            blocks,
            state,
            7,
            99,
            4,
            [2, 2, 3, 4, 1, 2],
            1,
            "block",
            1,
        )
        self.assertEqual(manifest.block_keys[0], builder.kv_key(manifest.prefix_hashes[0]))
        self.assertEqual(manifest.block_keys[1], builder.kv_key(manifest.prefix_hashes[1]))
        self.assertEqual(manifest.block_keys[2], request_kv_key("xfer-test", 2))
        self.assertEqual(manifest.state_key, request_state_key("xfer-test"))
        self.assertIn(builder.state_key(manifest.prefix_hashes[-1]), store.data)

        loaded = load_manifest(store, builder, "xfer-test")
        self.assertEqual(loaded.token_ids, token_ids)
        self.assertEqual(loaded.num_prompt_tokens, 8)
        kv, loaded_state = load_prefill(store, loaded)
        self.assertEqual(tuple(kv.shape), (2, 2, 3, 4, 1, 2))
        self.assertTrue(torch.equal(loaded_state[0][0], state[0][0]))

    def test_missing_block_raises(self):
        store = FakeStore()
        builder = PrefixCacheKeyBuilder("scope", 4)
        blocks = [torch.zeros(2, 2, 4, 1, 2)]
        manifest = publish_prefill(
            store,
            builder,
            "xfer-missing",
            "engine-1",
            list(range(5)),
            4,
            blocks,
            None,
            1,
            99,
            4,
            [2, 2, 1, 4, 1, 2],
        )
        store.data.pop(manifest.block_keys[0])
        with self.assertRaises(ValueError):
            load_prefill(store, manifest)


if __name__ == "__main__":
    unittest.main()
