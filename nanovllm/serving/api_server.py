import argparse
import json
import time
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from nanovllm.sampling_params import SamplingParams
from nanovllm.serving.protocol import ChatCompletionRequest, CompletionRequest
from nanovllm.serving.router import BaseRouter, create_router


def _normalize_stop(stop) -> list[str] | None:
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop] if stop else None
    stops = [item for item in stop if item]
    return stops or None


def _usage(event) -> tuple[int, int, int]:
    prompt_tokens = int(event.get("num_prompt_tokens") or 0)
    completion_tokens = int(event.get("num_generated_tokens") or 0)
    return prompt_tokens, completion_tokens, prompt_tokens + completion_tokens


def _register_routes(app: FastAPI, model_name: str) -> None:

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        engine_router = app.state.router
        request_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model = req.model or model_name
        stop = _normalize_stop(req.stop)
        try:
            sampling_params = SamplingParams(
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                ignore_eos=req.ignore_eos,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
            )
        except (AssertionError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if req.stream:
            async def event_stream():
                try:
                    async for event in engine_router.generate(
                        req.prompt,
                        sampling_params,
                        request_id=request_id,
                        stop=stop,
                    ):
                        chunk = {
                            "id": request_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "text": event.get("delta_text", ""),
                                    "finish_reason": event.get("finish_reason"),
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                except Exception as exc:
                    error = {"error": {"message": str(exc), "type": "internal_error"}}
                    yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        last_event = None
        try:
            async for event in engine_router.generate(
                req.prompt,
                sampling_params,
                request_id=request_id,
                stop=stop,
            ):
                last_event = event
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        if last_event is None:
            raise HTTPException(status_code=500, detail="generation produced no output")
        prompt_tokens, completion_tokens, total_tokens = _usage(last_event)
        return {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": last_event.get("text", ""),
                    "finish_reason": last_event.get("finish_reason"),
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        engine_router = app.state.router
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model = req.model or model_name
        stop = _normalize_stop(req.stop)
        messages = [{"role": message.role, "content": message.content} for message in req.messages]
        try:
            prompt = engine_router.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        try:
            sampling_params = SamplingParams(
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                ignore_eos=req.ignore_eos,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
            )
        except (AssertionError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if req.stream:
            async def event_stream():
                try:
                    async for event in engine_router.generate(
                        prompt,
                        sampling_params,
                        request_id=request_id,
                        stop=stop,
                    ):
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": event.get("delta_text", "")},
                                    "finish_reason": event.get("finish_reason"),
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                except Exception as exc:
                    error = {"error": {"message": str(exc), "type": "internal_error"}}
                    yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        last_event = None
        try:
            async for event in engine_router.generate(
                prompt,
                sampling_params,
                request_id=request_id,
                stop=stop,
            ):
                last_event = event
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        if last_event is None:
            raise HTTPException(status_code=500, detail="generation produced no output")
        prompt_tokens, completion_tokens, total_tokens = _usage(last_event)
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": last_event.get("text", ""),
                    },
                    "finish_reason": last_event.get("finish_reason"),
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }


def create_app(router: BaseRouter, model_name: str) -> FastAPI:
    app = FastAPI()
    app.state.router = router
    app.state.model_name = model_name
    _register_routes(app, model_name)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--tp", dest="tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", dest="enforce_eager", action="store_true", default=True)
    parser.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pd", action="store_true")
    parser.add_argument("--prefill-device", type=int, default=0)
    parser.add_argument("--decode-device", type=int, default=1)
    args = parser.parse_args()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        router = await create_router(args)
        app.state.router = router
        app.state.model_name = args.model
        try:
            yield
        finally:
            await router.shutdown()

    app = FastAPI(lifespan=lifespan)
    _register_routes(app, args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
