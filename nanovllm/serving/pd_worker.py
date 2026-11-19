from __future__ import annotations

import asyncio
import uuid
from typing import AsyncIterator

import httpx

from nanovllm.engine.llm_engine import LLMEngine

from .async_engine import AsyncLLMEngine
from .mooncake_bootstrap import MooncakeBootstrapServer
from .mooncake_store import MooncakeStoreConfig, create_store
from .pd_mooncake import (
    load_manifest,
    load_prefill,
    load_prefix,
    publish_block_snapshot,
    publish_prefill,
)
from .pd_transfer import (
    PDTransferStore,
    decode_payload,
    encode_payload,
    extract_metadata,
    transfer_metadata,
)
from .prefix_cache import PrefixCacheKeyBuilder, compute_scope, find_prefix
from .router import BaseRouter


def _decode_text(tokenizer, token_ids) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _find_stop(text: str, stops: list[str] | None) -> int | None:
    if not stops:
        return None
    best = None
    for stop in stops:
        if not stop:
            continue
        index = text.find(stop)
        if index != -1 and (best is None or index < best):
            best = index
    return best


def _completion_event(
    request_id: str,
    token_ids: list[int],
    text: str,
    finished: bool,
    finish_reason: str | None,
    num_prompt_tokens: int,
    role: str,
    pd_transfer: bool,
    **extra,
) -> dict:
    event = {
        "request_id": request_id,
        "seq_id": -1,
        "token_ids": list(token_ids),
        "text": text,
        "delta_text": text,
        "finished": finished,
        "num_prompt_tokens": num_prompt_tokens,
        "num_generated_tokens": len(token_ids),
        "finish_reason": finish_reason,
        "worker_role": role,
        "pd_transfer": pd_transfer,
    }
    event.update(extra)
    return event


def _prefix_builder(engine: LLMEngine) -> PrefixCacheKeyBuilder:
    config = engine.config
    text_config = config.text_config
    num_kv_layers = sum(
        1
        for module in engine.model_runner.model.modules()
        if hasattr(module, "k_cache") and hasattr(module, "v_cache")
    )
    num_kv_heads = text_config.num_key_value_heads // config.tensor_parallel_size
    head_dim = getattr(
        text_config,
        "head_dim",
        text_config.hidden_size // text_config.num_attention_heads,
    )
    scope = compute_scope(
        config.model,
        str(text_config.dtype),
        config.kvcache_block_size,
        num_kv_layers,
        num_kv_heads,
        head_dim,
    )
    return PrefixCacheKeyBuilder(scope, config.kvcache_block_size)


class PrefillWorkerRouter(BaseRouter):
    """Prefill-only worker speaking the vLLM PD handoff protocol.

    The router sends ``max_tokens=1`` to this worker and forwards the
    ``kv_transfer_params`` object from the response to a decode worker. With the
    HTTP backend, KV and Qwen3.5 state are staged in this process; with the
    Mooncake backend they are published to the shared store and the decode
    worker pulls them by transfer ID.
    """

    worker_role = "prefill"

    def __init__(
        self,
        engine: LLMEngine,
        advertise_url: str,
        ttl_seconds: float = 300.0,
        mooncake_config: MooncakeStoreConfig | None = None,
    ):
        self.engine = engine
        self.advertise_url = advertise_url.rstrip("/")
        self.worker_id = None
        self.transfer_store = PDTransferStore(ttl_seconds=ttl_seconds)
        self._lock = asyncio.Lock()
        self.mooncake = create_store(mooncake_config) if mooncake_config else None
        self.prefix_builder = None
        self.bootstrap = None
        self._state_snapshots = "none"
        self._has_state = hasattr(engine.model_runner.model, "export_state")
        if self.mooncake is not None:
            self.mooncake.setup()
            self.prefix_builder = _prefix_builder(engine)
            self._state_snapshots = mooncake_config.state_snapshots
            self.bootstrap = MooncakeBootstrapServer(
                self.mooncake.engine_id,
                mooncake_config.bootstrap_port,
            )
            self.bootstrap.start()

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    def _on_block_complete(self, prompt_tokens, seq):
        block_size = self.engine.config.kvcache_block_size
        boundary = seq.num_cached_tokens
        if boundary <= 0 or boundary % block_size != 0:
            return
        hashes = self.prefix_builder.block_hashes(prompt_tokens[:boundary])
        if not hashes:
            return
        block_index = boundary // block_size - 1
        block_id = seq.block_table[block_index]
        kv = self.engine.model_runner.call("export_kv", [block_id])
        state = getattr(seq, "state_snapshot", None)
        if state is None and self._has_state:
            state = self.engine.model_runner.call("export_state", seq.state_slot)
        publish_block_snapshot(
            self.mooncake,
            self.prefix_builder,
            hashes[-1],
            kv.squeeze(2),
            state,
        )

    async def _generate_mooncake(
        self,
        prompt,
        sampling_params,
        request_id: str,
        stop: list[str] | None,
        kv_transfer_params: dict | None,
    ) -> AsyncIterator[dict]:
        prompt_tokens = (
            self.engine.tokenizer.encode(prompt)
            if isinstance(prompt, str)
            else list(prompt)
        )
        transfer_id = None
        if isinstance(kv_transfer_params, dict):
            transfer_id = kv_transfer_params.get("transfer_id")
        if not transfer_id:
            transfer_id = f"xfer-{uuid.uuid4()}"

        match = await asyncio.to_thread(
            find_prefix,
            self.mooncake,
            self.prefix_builder,
            prompt_tokens,
            self._has_state,
        )
        prefix = None
        if match.num_blocks:
            block_tensors, state = await asyncio.to_thread(
                load_prefix,
                self.mooncake,
                match,
            )
            prefix = {
                "num_blocks": match.num_blocks,
                "num_tokens": match.num_tokens,
                "block_tensors": block_tensors,
                "state": state,
            }

        callback = None
        if self._state_snapshots == "block":
            callback = lambda seq: self._on_block_complete(prompt_tokens, seq)

        result = await asyncio.to_thread(
            self.engine.prefill_with_prefix,
            prompt,
            sampling_params,
            prefix,
            callback,
            self._state_snapshots == "block",
            True,
        )
        token_ids = list(result["token_ids"])
        num_prompt_tokens = int(result["num_prompt_tokens"])
        completion = token_ids[num_prompt_tokens:]
        first_token = completion[0] if completion else -1
        text = _decode_text(self.tokenizer, completion)

        kv = result.get("kv")
        block_tensors = []
        kv_shape = []
        if kv is not None:
            kv_shape = list(kv.shape)
            block_tensors = [kv[:, :, index] for index in range(kv.shape[2])]
        manifest = await asyncio.to_thread(
            publish_prefill,
            self.mooncake,
            self.prefix_builder,
            transfer_id,
            self.mooncake.engine_id,
            token_ids,
            num_prompt_tokens,
            block_tensors,
            result.get("state"),
            first_token,
            int(self.engine.config.eos),
            self.engine.config.kvcache_block_size,
            kv_shape,
            match.num_blocks,
            self._state_snapshots,
            match.num_blocks,
        )
        yield _completion_event(
            request_id=request_id,
            token_ids=completion,
            text=text,
            finished=True,
            finish_reason="length",
            num_prompt_tokens=num_prompt_tokens,
            role="prefill",
            pd_transfer=True,
            worker_id=self.worker_id,
            kv_store="mooncake",
            transfer_id=manifest.transfer_id,
            prefix_hit_blocks=match.num_blocks,
        )

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params,
        request_id: str | None = None,
        stop: list[str] | None = None,
        kv_transfer_params: dict | None = None,
    ) -> AsyncIterator[dict]:
        if request_id is None:
            request_id = f"cmpl-{uuid.uuid4().hex}"
        if self.mooncake is not None:
            async with self._lock:
                async for event in self._generate_mooncake(
                    prompt,
                    sampling_params,
                    request_id,
                    stop,
                    kv_transfer_params,
                ):
                    yield event
            return
        async with self._lock:
            result = await asyncio.to_thread(
                self.engine.prefill,
                prompt,
                sampling_params,
                True,
            )
        token_ids = list(result["token_ids"])
        num_prompt_tokens = int(result["num_prompt_tokens"])
        completion = token_ids[num_prompt_tokens:]
        first_token = completion[0] if completion else -1
        text = _decode_text(self.tokenizer, completion)
        payload = await asyncio.to_thread(
            encode_payload,
            result.get("kv"),
            result.get("state"),
        )
        transfer_id = self.transfer_store.put(payload)
        metadata = transfer_metadata(
            transfer_id=transfer_id,
            prefill_url=self.advertise_url,
            token_ids=token_ids,
            num_prompt_tokens=num_prompt_tokens,
            num_blocks=len(result.get("block_table") or []),
            first_token=first_token,
            eos_token_id=int(self.engine.config.eos),
            has_state=result.get("state") is not None,
        )
        yield _completion_event(
            request_id=request_id,
            token_ids=completion,
            text=text,
            finished=True,
            finish_reason="length",
            num_prompt_tokens=num_prompt_tokens,
            role="prefill",
            pd_transfer=True,
            kv_transfer_params={"nano_pd": metadata},
            worker_id=self.worker_id,
            kv_store="http",
            prefix_hit_blocks=0,
        )

    async def shutdown(self):
        if self.bootstrap is not None:
            self.bootstrap.stop()
        if self.mooncake is not None:
            self.mooncake.close()
        self.transfer_store.clear()
        await asyncio.to_thread(self.engine.exit)


class DecodeWorkerRouter(BaseRouter):

    worker_role = "decode"

    def __init__(
        self,
        engine: AsyncLLMEngine,
        fetch_timeout: float = 120.0,
        mooncake_config: MooncakeStoreConfig | None = None,
    ):
        self.engine = engine
        self.worker_id = None
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(fetch_timeout, connect=10.0),
            trust_env=False,
        )
        self._lock = asyncio.Lock()
        self._inflight = 0
        self.mooncake = create_store(mooncake_config) if mooncake_config else None
        self.prefix_builder = None
        self._missing_policy = "error"
        if self.mooncake is not None:
            self.mooncake.setup()
            self.prefix_builder = _prefix_builder(engine.engine)
            self._missing_policy = mooncake_config.missing_policy

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    @property
    def inflight(self) -> int:
        return self._inflight

    async def _fetch_payload(self, metadata: dict) -> dict:
        prefill_url = str(metadata["prefill_url"]).rstrip("/")
        transfer_id = metadata["transfer_id"]
        url = f"{prefill_url}/internal/pd/transfer/{transfer_id}"
        response = await self._client.get(url)
        response.raise_for_status()
        return await asyncio.to_thread(decode_payload, response.content)

    async def _generate_from_imported(
        self,
        token_ids: list[int],
        num_prompt_tokens: int,
        kv,
        state,
        sampling_params,
        request_id: str,
        stop: list[str] | None,
        prefix_hit_blocks: int,
        transfer_id: str | None,
        kv_store: str,
    ) -> AsyncIterator[dict]:
        config = self.engine.engine.config
        if num_prompt_tokens + sampling_params.max_tokens > config.max_model_len:
            raise ValueError("prompt length plus max_tokens exceeds max_model_len")
        completion = token_ids[num_prompt_tokens:]
        first_token = int(token_ids[num_prompt_tokens]) if completion else -1
        eos_token_id = int(config.eos)
        is_eos = (
            first_token == eos_token_id
            and first_token >= 0
            and not sampling_params.ignore_eos
        )
        text = _decode_text(self.tokenizer, completion)
        stop_index = _find_stop(text, stop)
        extras = {
            "kv_store": kv_store,
            "prefix_hit_blocks": int(prefix_hit_blocks),
        }
        if transfer_id is not None:
            extras["transfer_id"] = transfer_id
        if is_eos or sampling_params.max_tokens <= len(completion) or stop_index is not None:
            if stop_index is not None:
                text = text[:stop_index]
            finish_reason = "stop" if (is_eos or stop_index is not None) else "length"
            yield _completion_event(
                request_id=request_id,
                token_ids=completion,
                text=text,
                finished=True,
                finish_reason=finish_reason,
                num_prompt_tokens=num_prompt_tokens,
                role="decode",
                pd_transfer=True,
                worker_id=self.worker_id,
                **extras,
            )
            return

        prefill_result = {
            "token_ids": token_ids,
            "prompt_token_ids": token_ids[:num_prompt_tokens],
            "num_prompt_tokens": num_prompt_tokens,
            "block_table": [0] * ((len(token_ids) + config.kvcache_block_size - 1) // config.kvcache_block_size),
            "kv": kv,
            "state": state,
            "sampling_params": sampling_params,
            "finished": False,
        }
        seq_id, initial = await self.engine.add_prefilled(
            prefill_result,
            request_id,
            stop=stop,
        )
        initial = list(initial or [])
        text = _decode_text(self.tokenizer, initial)
        if seq_id is None:
            yield _completion_event(
                request_id=request_id,
                token_ids=initial,
                text=text,
                finished=True,
                finish_reason="length",
                num_prompt_tokens=num_prompt_tokens,
                role="decode",
                pd_transfer=True,
                worker_id=self.worker_id,
                **extras,
            )
            return
        yield _completion_event(
            request_id=request_id,
            token_ids=initial,
            text=text,
            finished=False,
            finish_reason=None,
            num_prompt_tokens=num_prompt_tokens,
            role="decode",
            pd_transfer=True,
            worker_id=self.worker_id,
            seq_id=seq_id,
            **extras,
        )
        async for event in self.engine.stream(request_id):
            event["worker_role"] = "decode"
            event["worker_id"] = self.worker_id
            event["pd_transfer"] = True
            event.update(extras)
            yield event

    async def _generate_from_mooncake(
        self,
        prompt,
        transfer_id: str,
        sampling_params,
        request_id: str,
        stop: list[str] | None,
    ) -> AsyncIterator[dict]:
        manifest = await asyncio.to_thread(
            load_manifest,
            self.mooncake,
            self.prefix_builder,
            transfer_id,
        )
        if manifest is None:
            if self._missing_policy != "recompute":
                raise ValueError(f"Mooncake manifest missing: {transfer_id}")
            async for event in self.engine.generate(
                prompt,
                sampling_params,
                request_id=request_id,
                stop=stop,
            ):
                event["worker_role"] = "decode"
                event["worker_id"] = self.worker_id
                event["pd_transfer"] = False
                event["kv_store"] = "mooncake"
                event["fallback"] = True
                yield event
            return
        kv, state = await asyncio.to_thread(
            load_prefill,
            self.mooncake,
            manifest,
        )
        async for event in self._generate_from_imported(
            manifest.token_ids,
            manifest.num_prompt_tokens,
            kv,
            state,
            sampling_params,
            request_id,
            stop,
            manifest.prefix_hit_blocks,
            manifest.transfer_id,
            "mooncake",
        ):
            yield event

    async def _generate_from_prefill(
        self,
        metadata: dict,
        sampling_params,
        request_id: str,
        stop: list[str] | None,
    ) -> AsyncIterator[dict]:
        payload = await self._fetch_payload(metadata)
        token_ids = list(metadata["token_ids"])
        num_prompt_tokens = int(metadata["num_prompt_tokens"])
        async for event in self._generate_from_imported(
            token_ids,
            num_prompt_tokens,
            payload.get("kv"),
            payload.get("state"),
            sampling_params,
            request_id,
            stop,
            0,
            None,
            "http",
        ):
            yield event

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params,
        request_id: str | None = None,
        stop: list[str] | None = None,
        kv_transfer_params: dict | None = None,
    ) -> AsyncIterator[dict]:
        if request_id is None:
            request_id = f"cmpl-{uuid.uuid4().hex}"
        mooncake_transfer_id = None
        if isinstance(kv_transfer_params, dict) and kv_transfer_params.get("do_remote_prefill"):
            mooncake_transfer_id = kv_transfer_params.get("transfer_id")
        metadata = extract_metadata(kv_transfer_params)
        if self.mooncake is not None and mooncake_transfer_id:
            async with self._lock:
                self._inflight += 1
            try:
                async for event in self._generate_from_mooncake(
                    prompt,
                    mooncake_transfer_id,
                    sampling_params,
                    request_id,
                    stop,
                ):
                    yield event
            finally:
                async with self._lock:
                    self._inflight -= 1
            return
        if metadata is None:
            async for event in self.engine.generate(
                prompt,
                sampling_params,
                request_id=request_id,
                stop=stop,
            ):
                event["worker_role"] = "decode"
                event["worker_id"] = self.worker_id
                event["pd_transfer"] = False
                event["kv_store"] = "mooncake" if self.mooncake is not None else "http"
                yield event
            return

        async with self._lock:
            self._inflight += 1
        try:
            async for event in self._generate_from_prefill(
                metadata,
                sampling_params,
                request_id,
                stop,
            ):
                yield event
        finally:
            async with self._lock:
                self._inflight -= 1

    async def shutdown(self):
        if self.mooncake is not None:
            self.mooncake.close()
        await self._client.aclose()
        await self.engine.shutdown()
