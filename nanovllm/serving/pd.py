from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, AsyncIterator

from .router import BaseRouter

if TYPE_CHECKING:
    from .async_engine import AsyncLLMEngine
    from nanovllm.engine.llm_engine import LLMEngine
    from nanovllm.sampling_params import SamplingParams


class PDRouter(BaseRouter):

    def __init__(self, prefill_engine: LLMEngine, decode_engine: AsyncLLMEngine):
        self.prefill_engine = prefill_engine
        self.decode_engine = decode_engine
        self._prefill_lock = asyncio.Lock()

    @property
    def tokenizer(self):
        return self.decode_engine.tokenizer

    def _decode_text(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
        stop: list[str] | None = None,
        kv_transfer_params: dict | None = None,
    ) -> AsyncIterator[dict]:
        if request_id is None:
            request_id = str(uuid.uuid4())
        async with self._prefill_lock:
            result = await asyncio.to_thread(
                self.prefill_engine.prefill,
                prompt,
                sampling_params,
            )
        if result["finished"]:
            initial = result["token_ids"][result["num_prompt_tokens"]:]
            text = self._decode_text(initial)
            yield {
                "request_id": request_id,
                "seq_id": -1,
                "token_ids": initial,
                "text": text,
                "delta_text": text,
                "finished": True,
                "num_prompt_tokens": result["num_prompt_tokens"],
                "num_generated_tokens": len(initial),
                "finish_reason": "length",
            }
            return
        try:
            seq_id, initial = await self.decode_engine.add_prefilled(
                result,
                request_id,
                stop=stop,
            )
            text = self._decode_text(initial)
            yield {
                "request_id": request_id,
                "seq_id": seq_id,
                "token_ids": initial,
                "text": text,
                "delta_text": text,
                "finished": False,
                "num_prompt_tokens": result["num_prompt_tokens"],
                "num_generated_tokens": len(initial),
                "finish_reason": None,
            }
            async for event in self.decode_engine.stream(request_id):
                yield event
        except Exception:
            await self.decode_engine.abort(request_id)
            raise

    async def shutdown(self):
        await self.decode_engine.shutdown()
        await asyncio.to_thread(self.prefill_engine.exit)
