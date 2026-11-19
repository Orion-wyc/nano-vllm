import unittest
from unittest import mock

from nanovllm.serving import mooncake_store
from nanovllm.serving.mooncake_store import (
    MooncakeStoreClient,
    MooncakeStoreConfig,
    create_store,
)
from nanovllm.serving.prefix_cache import (
    PrefixCacheKeyBuilder,
    compute_scope,
    find_prefix,
)


class FakePrefixStore:
    def __init__(self, keys=()):
        self.objects = {key: b"payload" for key in keys}
        self.exist_calls: list[list[str]] = []

    def batch_is_exist(self, keys):
        self.exist_calls.append(list(keys))
        return [1 if key in self.objects else 0 for key in keys]


class FakeMooncakeDistributedStore:
    instances: list["FakeMooncakeDistributedStore"] = []

    def __init__(self):
        self.data: dict[str, bytes] = {}
        self.setup_args = None
        self.setup_kwargs = None
        self.setup_return = 0
        self.put_return = 0
        self.put_batch_return = 0
        self.remove_return = 0
        self.closed = False
        self.instances.append(self)

    def setup(self, *args, **kwargs):
        self.setup_args = args
        self.setup_kwargs = kwargs
        return self.setup_return

    def put(self, key, value):
        if self.put_return == 0:
            self.data[key] = bytes(value)
        return self.put_return

    def get(self, key):
        return self.data.get(key, b"")

    def put_batch(self, keys, values):
        if self.put_batch_return == 0:
            for key, value in zip(keys, values):
                self.data[key] = bytes(value)
        return self.put_batch_return

    def get_batch(self, keys):
        return [self.data.get(key, b"") for key in keys]

    def batch_is_exist(self, keys):
        return [1 if key in self.data else 0 for key in keys]

    def remove(self, key):
        if self.remove_return == 0:
            self.data.pop(key, None)
        return self.remove_return

    def close(self):
        self.closed = True
        return 0


class PrefixCacheKeyTests(unittest.TestCase):
    def test_block_hashes_are_stable_and_ignore_partial_tail(self):
        builder = PrefixCacheKeyBuilder("scope-a", 256)
        tokens = list(range(600))

        first = builder.block_hashes(tokens)
        second = builder.block_hashes(list(tokens))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(first[0]), 64)
        self.assertEqual(len(first[1]), 64)

        # 600 tokens only fill two 256-token blocks; the 88-token tail is
        # neither hashed nor allowed to influence earlier hashes.
        self.assertEqual(builder.block_hashes(tokens + list(range(88))), first)
        self.assertEqual(builder.block_hashes(tokens[:512]), first)
        self.assertEqual(len(builder.block_hashes(tokens[:511])), 1)
        self.assertEqual(builder.block_hashes(tokens[:255]), [])

        # A different boundary size changes the block partition.
        self.assertEqual(len(PrefixCacheKeyBuilder("scope-a", 128).block_hashes(tokens)), 4)

    def test_block_hashes_are_chained(self):
        builder = PrefixCacheKeyBuilder("scope-a", 4)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        baseline = builder.block_hashes(tokens)

        changed_first = list(tokens)
        changed_first[0] = 99
        first_hashes = builder.block_hashes(changed_first)
        self.assertNotEqual(first_hashes[0], baseline[0])
        self.assertNotEqual(first_hashes[1], baseline[1])

        changed_second = list(tokens)
        changed_second[4] = 99
        second_hashes = builder.block_hashes(changed_second)
        self.assertEqual(second_hashes[0], baseline[0])
        self.assertNotEqual(second_hashes[1], baseline[1])

        other_scope = PrefixCacheKeyBuilder("scope-b", 4)
        self.assertNotEqual(other_scope.block_hashes(tokens)[0], baseline[0])

    def test_key_formats(self):
        builder = PrefixCacheKeyBuilder("qwen-test", 256)
        prefix_hash = builder.block_hashes(list(range(256)))[0]

        self.assertEqual(
            builder.kv_key(prefix_hash), f"nano:kv:v1:qwen-test:{prefix_hash}"
        )
        self.assertEqual(
            builder.state_key(prefix_hash),
            f"nano:state:v1:qwen-test:{prefix_hash}",
        )
        self.assertEqual(
            builder.manifest_key("xfer-1234"), "nano:manifest:v1:xfer-1234"
        )

    def test_compute_scope_is_deterministic(self):
        kwargs = dict(
            model_path="models/Qwen3.5-0.8B",
            dtype="bfloat16",
            block_size=256,
            num_kv_layers=6,
            num_kv_heads=2,
            head_dim=256,
        )
        scope = compute_scope(**kwargs)
        self.assertEqual(scope, compute_scope(**kwargs))
        self.assertTrue(scope.startswith("v1-"))
        self.assertIn("qwen3-5-0-8b", scope)
        self.assertIn("bfloat16", scope)
        self.assertIn("-l6-h2-d256-", scope)
        self.assertEqual(
            scope,
            compute_scope(
                "models/./Qwen3.5-0.8B", "torch.bfloat16", 256, 6, 2, 256
            ),
        )

        self.assertNotEqual(scope, compute_scope(**{**kwargs, "block_size": 128}))
        self.assertNotEqual(scope, compute_scope(**{**kwargs, "head_dim": 128}))
        self.assertNotEqual(
            scope, compute_scope(**{**kwargs, "model_path": "models/Qwen3.5-4B"})
        )

    def test_prefix_match_stops_at_first_missing_kv_block(self):
        builder = PrefixCacheKeyBuilder("scope-a", 256)
        tokens = list(range(768))
        hashes = builder.block_hashes(tokens)
        self.assertEqual(len(hashes), 3)

        store = FakePrefixStore([builder.kv_key(hashes[0]), builder.kv_key(hashes[2])])
        match = find_prefix(store, builder, tokens, require_state=False)
        self.assertEqual(match.num_blocks, 1)
        self.assertEqual(match.num_tokens, 256)
        self.assertEqual(match.hashes, hashes[:1])
        self.assertEqual(match.kv_keys, [builder.kv_key(hashes[0])])
        self.assertIsNone(match.state_key)

        # Blocks 1 and 2 exist, but block 0 is missing: no prefix matches.
        store = FakePrefixStore([builder.kv_key(hashes[1]), builder.kv_key(hashes[2])])
        empty = find_prefix(store, builder, tokens, require_state=False)
        self.assertEqual((empty.num_blocks, empty.num_tokens), (0, 0))
        self.assertEqual(empty.hashes, [])
        self.assertEqual(empty.kv_keys, [])
        self.assertIsNone(empty.state_key)

        # Short prompts have no complete block and never touch the store.
        short_store = FakePrefixStore()
        empty = find_prefix(short_store, builder, tokens[:255])
        self.assertEqual(empty.num_blocks, 0)
        self.assertEqual(short_store.exist_calls, [])

    def test_prefix_match_without_state_uses_all_kv_blocks(self):
        builder = PrefixCacheKeyBuilder("scope-a", 256)
        tokens = list(range(768))
        hashes = builder.block_hashes(tokens)
        store = FakePrefixStore([builder.kv_key(prefix_hash) for prefix_hash in hashes])

        match = find_prefix(store, builder, tokens, require_state=False)
        self.assertEqual(match.num_blocks, 3)
        self.assertEqual(match.num_tokens, 768)
        self.assertEqual(match.kv_keys, [builder.kv_key(h) for h in hashes])
        self.assertEqual(len(store.exist_calls), 1)

    def test_state_gating_falls_back_to_previous_boundary(self):
        builder = PrefixCacheKeyBuilder("scope-a", 256)
        tokens = list(range(768))
        hashes = builder.block_hashes(tokens)
        kv_keys = [builder.kv_key(prefix_hash) for prefix_hash in hashes]
        state_keys = [builder.state_key(prefix_hash) for prefix_hash in hashes]

        # All KV blocks exist, but the state for the second block is missing.
        store = FakePrefixStore(kv_keys + [state_keys[0], state_keys[2]])
        match = find_prefix(store, builder, tokens, require_state=True)
        self.assertEqual(match.num_blocks, 1)
        self.assertEqual(match.num_tokens, 256)
        self.assertEqual(match.hashes, hashes[:1])
        self.assertEqual(match.kv_keys, kv_keys[:1])
        self.assertEqual(match.state_key, state_keys[0])
        self.assertEqual(len(store.exist_calls), 2)

        # The missing boundary is restored: the full prefix matches.
        store = FakePrefixStore(kv_keys + state_keys)
        match = find_prefix(store, builder, tokens, require_state=True)
        self.assertEqual(match.num_blocks, 3)
        self.assertEqual(match.state_key, state_keys[2])

        # No state at block 0 means no reusable prefix at all.
        store = FakePrefixStore(kv_keys + [state_keys[1], state_keys[2]])
        match = find_prefix(store, builder, tokens, require_state=True)
        self.assertEqual(match.num_blocks, 0)
        self.assertIsNone(match.state_key)


class MooncakeStoreClientTests(unittest.TestCase):
    def setUp(self):
        FakeMooncakeDistributedStore.instances.clear()

    def _client(self, **overrides):
        config = MooncakeStoreConfig(enabled=True, **overrides)
        return MooncakeStoreClient(config)

    def test_setup_uses_design_doc_arguments(self):
        client = self._client(
            master="10.0.0.5:50051",
            metadata="P2PHANDSHAKE",
            protocol="tcp",
            global_segment_size=1234,
            local_buffer_size=5678,
            tenant="tenant-a",
        )
        with mock.patch.object(
            mooncake_store,
            "_load_store_class",
            return_value=FakeMooncakeDistributedStore,
        ):
            client.setup()

        engine = FakeMooncakeDistributedStore.instances[-1]
        self.assertEqual(
            engine.setup_args,
            (
                "127.0.0.1",
                "P2PHANDSHAKE",
                1234,
                5678,
                "tcp",
                "",
                "10.0.0.5:50051",
            ),
        )
        self.assertEqual(engine.setup_kwargs.get("tenant_id"), "tenant-a")

    def test_setup_import_failure_and_non_zero_return(self):
        client = self._client()
        with mock.patch.object(
            mooncake_store, "_load_store_class", side_effect=ImportError("missing")
        ):
            with self.assertRaisesRegex(RuntimeError, "[Mm]ooncake"):
                client.setup()

        class FailingStore(FakeMooncakeDistributedStore):
            def setup(self, *args, **kwargs):
                return 3

        with mock.patch.object(
            mooncake_store, "_load_store_class", return_value=FailingStore
        ):
            with self.assertRaisesRegex(RuntimeError, "50051"):
                self._client(master="127.0.0.1:50051").setup()

    def test_operations_require_setup(self):
        client = self._client()
        with self.assertRaisesRegex(RuntimeError, "setup"):
            client.put("k", b"v")

    def test_put_get_and_missing_objects(self):
        client = self._client()
        with mock.patch.object(
            mooncake_store,
            "_load_store_class",
            return_value=FakeMooncakeDistributedStore,
        ):
            client.setup()
            client.put("alpha", b"one")
            self.assertEqual(client.get("alpha"), b"one")
            self.assertIsNone(client.get("missing"))
            client.put("empty", b"")
            self.assertIsNone(client.get("empty"))

        self.assertRegex(client.engine_id, r"^nano-[0-9a-f]{8}$")
        self.assertEqual(client.engine_id, client.engine_id)
        self.assertIs(client.config.enabled, True)

    def test_batch_operations_normalize_missing_entries(self):
        client = self._client()
        with mock.patch.object(
            mooncake_store,
            "_load_store_class",
            return_value=FakeMooncakeDistributedStore,
        ):
            client.setup()
            client.put_batch(["a", "b"], [b"A", b"B"])
            self.assertEqual(client.get_batch(["a", "missing", "b"]), [b"A", b"", b"B"])
            self.assertEqual(client.batch_is_exist(["a", "missing", "b"]), [1, 0, 1])
            self.assertEqual(client.batch_is_exist([]), [])
            self.assertEqual(client.get_batch([]), [])

            client.remove("a")
            self.assertEqual(client.batch_is_exist(["a", "b"]), [0, 1])

            client.close()
            engine = FakeMooncakeDistributedStore.instances[-1]
            self.assertTrue(engine.closed)
            with self.assertRaisesRegex(RuntimeError, "setup"):
                client.get("b")

    def test_non_zero_put_batch_and_remove_raise(self):
        client = self._client()
        with mock.patch.object(
            mooncake_store,
            "_load_store_class",
            return_value=FakeMooncakeDistributedStore,
        ):
            client.setup()
            engine = FakeMooncakeDistributedStore.instances[-1]
            engine.put_return = 7
            with self.assertRaisesRegex(RuntimeError, "code 7"):
                client.put("k", b"v")
            engine.put_return = 0
            engine.put_batch_return = 5
            with self.assertRaisesRegex(RuntimeError, "code 5"):
                client.put_batch(["k"], [b"v"])
            engine.put_batch_return = 0
            engine.remove_return = 2
            with self.assertRaisesRegex(RuntimeError, "code 2"):
                client.remove("k")

            with self.assertRaises(ValueError):
                client.put_batch(["a"], [])

    def test_create_store_respects_enabled_flag(self):
        self.assertIsNone(create_store(MooncakeStoreConfig()))
        client = create_store(MooncakeStoreConfig(enabled=True))
        self.assertIsInstance(client, MooncakeStoreClient)


if __name__ == "__main__":
    unittest.main()
