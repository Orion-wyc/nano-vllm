from __future__ import annotations

import asyncio
import uuid
from typing import AsyncIterator

import httpx

from nanovllm.engine.llm_engine import LLMEngine

from .async_engine import AsyncLLMEngine
from .pd_transfer import (
    PDTransferStore,
    decode_payload,
    encode_payload,
    extract_metadata,
    transfer_metadata,
)
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


class PrefillWorkerRouter(BaseRouter):
    """Prefill-only worker speaking the vLLM PD handoff protocol.

    The router sends ``max_tokens=1`` to this worker and forwards the
    ``kv_transfer_params`` object from the response to a decode worker. The KV
    cache and Qwen3.5 recurrent state are staged in this worker and fetched by
    the decode worker over HTTP.
    """

    worker_role = "prefill"

    def __init__(
        self,
        engine: LLMEngine,
        advertise_url: str,
        ttl_seconds: float = 300.0,
    ):
        self.engine = engine
        self.advertise_url = advertise_url.rstrip("/")
        self.worker_id = None
        self.transfer_store = PDTransferStore(ttl_seconds=ttl_seconds)
        self._lock = asyncio.Lock()

    @property
    def tokenizer(self):
        return self.engine.tokenizer

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
        )

    async def shutdown(self):
        self.transfer_store.clear()
        await asyncio.to_thread(self.engine.exit)


class DecodeWorkerRouter(BaseRouter):

    worker_role = "decode"

    def __init__(
        self,
        engine: AsyncLLMEngine,
        fetch_timeout: float = 120.0,
    ):
        self.engine = engine
        self.worker_id = None
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(fetch_timeout, connect=10.0),
            trust_env=False,
        )
        self._lock = asyncio.Lock()
        self._inflight = 0

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
        metadata = extract_metadata(kv_transfer_params)
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

    async def _generate_from_prefill(
        self,
        metadata: dict,
        sampling_params,
        request_id: str,
        stop: list[str] | None,
    ) -> AsyncIterator[dict]:
        token_ids = list(metadata["token_ids"])
        num_prompt_tokens = int(metadata["num_prompt_tokens"])
        config = self.engine.engine.config
        if num_prompt_tokens + sampling_params.max_tokens > config.max_model_len:
            raise ValueError("prompt length plus max_tokens exceeds max_model_len")
        payload = await self._fetch_payload(metadata)
        completion = token_ids[num_prompt_tokens:]
        first_token = int(metadata.get("first_token", -1))
        eos_token_id = int(metadata.get("eos_token_id", -1))
        is_eos = (
            first_token == eos_token_id
            and first_token >= 0
            and not sampling_params.ignore_eos
        )
        text = _decode_text(self.tokenizer, completion)
        stop_index = _find_stop(text, stop)
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
            )
            return

        prefill_result = {
            "token_ids": token_ids,
            "prompt_token_ids": token_ids[:num_prompt_tokens],
            "num_prompt_tokens": num_prompt_tokens,
            "block_table": [0] * int(metadata["num_blocks"]),
            "kv": payload.get("kv"),
            "state": payload.get("state"),
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
        )
        async for event in self.engine.stream(request_id):
            event["worker_role"] = "decode"
            event["worker_id"] = self.worker_id
            event["pd_transfer"] = True
            yield event

    async def shutdown(self):
        await self._client.aclose()
        await self.engine.shutdown()
