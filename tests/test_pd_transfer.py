import time
import unittest

import torch

from nanovllm.serving.pd_transfer import (
    PDTransferStore,
    decode_payload,
    encode_payload,
    extract_metadata,
    transfer_metadata,
)


class PDTransferTests(unittest.TestCase):
    def test_payload_round_trip(self):
        kv = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4)
        state = [(torch.ones(2), torch.zeros(3))]
        payload = decode_payload(encode_payload(kv, state))
        self.assertTrue(torch.equal(payload["kv"], kv))
        self.assertTrue(torch.equal(payload["state"][0][0], state[0][0]))
        self.assertTrue(torch.equal(payload["state"][0][1], state[0][1]))

    def test_store_expires_entries(self):
        store = PDTransferStore(ttl_seconds=0.05)
        transfer_id = store.put(b"payload")
        self.assertEqual(store.get(transfer_id), b"payload")
        time.sleep(0.1)
        self.assertIsNone(store.get(transfer_id))

    def test_metadata_round_trip(self):
        metadata = transfer_metadata(
            transfer_id="abc",
            prefill_url="http://127.0.0.1:8101/",
            token_ids=[1, 2, 3],
            num_prompt_tokens=2,
            num_blocks=1,
            first_token=3,
            eos_token_id=9,
            has_state=True,
        )
        self.assertEqual(extract_metadata({"nano_pd": metadata})["transfer_id"], "abc")
        self.assertIsNone(extract_metadata({"kv_transfer_params": metadata}))
        self.assertIsNone(extract_metadata(None))


if __name__ == "__main__":
    unittest.main()
