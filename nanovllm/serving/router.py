from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, AsyncIterator

from nanovllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from .async_engine import AsyncLLMEngine


class BaseRouter:

    @property
    def tokenizer(self):
        raise NotImplementedError

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
        stop: list[str] | None = None,
    ) -> AsyncIterator[dict]:
        raise NotImplementedError
        yield

    async def shutdown(self):
        raise NotImplementedError


class SingleRouter(BaseRouter):

    def __init__(self, engine: AsyncLLMEngine):
        self.engine = engine

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
        stop: list[str] | None = None,
    ) -> AsyncIterator[dict]:
        async for event in self.engine.generate(prompt, sampling_params, request_id=request_id, stop=stop):
            yield event

    async def shutdown(self):
        await self.engine.shutdown()


class RoundRobinRouter(BaseRouter):

    def __init__(self, engines: list[AsyncLLMEngine]):
        self.engines = engines
        self._lock = asyncio.Lock()
        self._next = 0

    @property
    def tokenizer(self):
        return self.engines[0].tokenizer

    async def _select(self) -> AsyncLLMEngine:
        async with self._lock:
            engine = self.engines[self._next % len(self.engines)]
            self._next += 1
            return engine

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
        stop: list[str] | None = None,
    ) -> AsyncIterator[dict]:
        engine = await self._select()
        async for event in engine.generate(prompt, sampling_params, request_id=request_id, stop=stop):
            yield event

    async def shutdown(self):
        for engine in self.engines:
            await engine.shutdown()


def _arg(args, name, default=None):
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def _engine_kwargs(args, device_id: int) -> dict:
    kwargs = {
        "device_id": device_id,
        "max_model_len": _arg(args, "max_model_len", 4096),
        "max_num_seqs": _arg(args, "max_num_seqs", 512),
        "max_num_batched_tokens": _arg(args, "max_num_batched_tokens", 16384),
        "gpu_memory_utilization": _arg(args, "gpu_memory_utilization", 0.9),
        "enforce_eager": _arg(args, "enforce_eager", False),
        "tensor_parallel_size": _arg(args, "tensor_parallel_size", 1),
    }
    return {k: v for k, v in kwargs.items() if v is not None}


async def create_router(args) -> BaseRouter:
    from .async_engine import AsyncLLMEngine

    model = _arg(args, "model")
    device_id = _arg(args, "device_id", 0)
    tp = _arg(args, "tensor_parallel_size", 1)
    workers = _arg(args, "workers", 1)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if tp != 1 and (_arg(args, "pd", False) or workers > 1):
        raise ValueError("PD and multiple workers currently require tensor_parallel_size=1")
    if _arg(args, "pd", False):
        from nanovllm.engine.llm_engine import LLMEngine

        from .pd import PDRouter

        prefill_device = _arg(args, "prefill_device", device_id)
        decode_device = _arg(args, "decode_device", device_id)
        gpu_memory_utilization = _arg(args, "gpu_memory_utilization", 0.9)
        if prefill_device == decode_device:
            gpu_memory_utilization = gpu_memory_utilization / 2
        prefill_kwargs = _engine_kwargs(args, prefill_device)
        decode_kwargs = _engine_kwargs(args, decode_device)
        prefill_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
        decode_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
        prefill_engine = LLMEngine(model, **prefill_kwargs)
        decode_engine = AsyncLLMEngine(model, **decode_kwargs)
        return PDRouter(prefill_engine, decode_engine)
    if workers > 1:
        engines = [
            AsyncLLMEngine(model, **_engine_kwargs(args, device_id + i))
            for i in range(workers)
        ]
        return RoundRobinRouter(engines)
    return SingleRouter(AsyncLLMEngine(model, **_engine_kwargs(args, device_id)))
