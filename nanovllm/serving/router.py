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
        kv_transfer_params: dict | None = None,
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
        kv_transfer_params: dict | None = None,
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
        kv_transfer_params: dict | None = None,
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
    role = _arg(args, "role", "regular")
    if role not in ("regular", "prefill", "decode"):
        raise ValueError(f"unsupported worker role: {role}")
    if role != "regular" and _arg(args, "pd", False):
        raise ValueError("--role cannot be combined with in-process --pd")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if role != "regular" and workers != 1:
        raise ValueError("PD worker roles require --workers 1")
    if tp != 1 and (_arg(args, "pd", False) or workers > 1 or role != "regular"):
        raise ValueError("PD and multiple workers currently require tensor_parallel_size=1")
    if role == "prefill":
        from nanovllm.engine.llm_engine import LLMEngine

        from .pd_worker import PrefillWorkerRouter

        advertise_url = _arg(args, "advertise_url", None)
        if not advertise_url:
            host = _arg(args, "host", "127.0.0.1")
            port = _arg(args, "port", 8000)
            advertise_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
            advertise_url = f"http://{advertise_host}:{port}"
        engine = LLMEngine(model, **_engine_kwargs(args, device_id))
        router = PrefillWorkerRouter(
            engine,
            advertise_url,
            _arg(args, "kv_transfer_ttl", 300.0),
        )
        router.worker_id = _arg(args, "worker_id", None)
        return router
    if role == "decode":
        from .pd_worker import DecodeWorkerRouter

        engine = AsyncLLMEngine(model, **_engine_kwargs(args, device_id))
        router = DecodeWorkerRouter(engine, _arg(args, "kv_fetch_timeout", 120.0))
        router.worker_id = _arg(args, "worker_id", None)
        return router
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
