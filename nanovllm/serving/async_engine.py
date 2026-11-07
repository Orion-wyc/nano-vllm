import asyncio
import queue
import threading
import traceback
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import AsyncIterator

from nanovllm.engine.llm_engine import LLMEngine


@dataclass(slots=True)
class RequestState:
    request_id: str
    seq_id: int | None
    queue: asyncio.Queue
    stop: list[str]
    prompt_tokens: list[int]
    prev_text: str = ""
    finished: bool = False
    finish_reason: str | None = None
    token_ids: list[int] = field(default_factory=list)
    text: str = ""


class AsyncLLMEngine:

    def __init__(self, model, **kwargs):
        self.loop = asyncio.get_running_loop()
        self.engine = LLMEngine(model, **kwargs)
        self._commands: queue.Queue = queue.Queue()
        self._seq_to_state: dict[int, RequestState] = {}
        self._request_to_state: dict[str, RequestState] = {}
        self._request_to_seq: dict[str, int] = {}
        self._shutdown_requested = False
        self._fatal = False
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="AsyncLLMEngine")
        self._thread.start()

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    async def generate(self, prompt, sampling_params, request_id=None, stop=None) -> AsyncIterator[dict]:
        if request_id is None:
            request_id = f"cmpl-{uuid.uuid4().hex}"
        event_queue = asyncio.Queue()
        future = self._submit(self._do_add_request, prompt, sampling_params, request_id, stop, event_queue)
        await asyncio.wrap_future(future, loop=self.loop)
        finished = False
        try:
            async for event in self.stream(request_id):
                if event.get("finished"):
                    finished = True
                yield event
        finally:
            if not finished:
                await self.abort(request_id)

    async def add_prefilled(self, prefill_result, request_id, stop=None) -> tuple[int | None, list[int]]:
        if request_id is None:
            request_id = f"cmpl-{uuid.uuid4().hex}"
        event_queue = asyncio.Queue()
        future = self._submit(self._do_add_prefilled, prefill_result, request_id, stop, event_queue)
        return await asyncio.wrap_future(future, loop=self.loop)

    async def stream(self, request_id) -> AsyncIterator[dict]:
        state = self._request_to_state.get(request_id)
        if state is None:
            raise KeyError(f"unknown request_id: {request_id}")
        while True:
            event = await state.queue.get()
            yield event
            if event.get("finished"):
                self._cleanup_state(state)
                break

    async def abort(self, request_id) -> None:
        future = self._submit(self._do_abort, request_id)
        await asyncio.wrap_future(future, loop=self.loop)

    async def shutdown(self) -> None:
        if self._thread is None:
            return
        if self._thread.is_alive():
            future = self._submit(self._do_shutdown)
            await asyncio.wrap_future(future, loop=self.loop)
            self._thread.join()
        self._thread = None
        self.engine.exit()

    def _submit(self, fn, *args) -> Future:
        future: Future = Future()
        self._commands.put((fn, args, future))
        return future

    def _execute(self, command) -> None:
        fn, args, future = command
        try:
            result = fn(*args)
        except BaseException as exc:
            if not future.cancelled():
                future.set_exception(exc)
            return
        if not future.cancelled():
            future.set_result(result)

    def _worker_loop(self) -> None:
        while True:
            command = None
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                pass
            if command is not None:
                self._execute(command)
            if self._shutdown_requested:
                break
            has_work = False
            if not self._fatal:
                try:
                    has_work = self.engine.has_work()
                except BaseException:
                    traceback.print_exc()
                    self._fatal = True
                    self._fail_all()
            if has_work:
                try:
                    updates, _ = self.engine.step_updates()
                    self._dispatch(updates)
                except BaseException:
                    traceback.print_exc()
                    self._fatal = True
                    self._fail_all()
                continue
            if command is None:
                command = self._commands.get()
                self._execute(command)
                if self._shutdown_requested:
                    break

    def _do_add_request(self, prompt, sampling_params, request_id, stop, event_queue) -> int:
        if request_id in self._request_to_state:
            raise ValueError(f"duplicate request_id: {request_id}")
        if isinstance(prompt, str):
            prompt_tokens = self.engine.tokenizer.encode(prompt)
        else:
            prompt_tokens = list(prompt)
        seq_id = self.engine.add_request(prompt, sampling_params)
        state = RequestState(
            request_id=request_id,
            seq_id=seq_id,
            queue=event_queue,
            stop=self._normalize_stop(stop),
            prompt_tokens=prompt_tokens,
        )
        self._seq_to_state[seq_id] = state
        self._request_to_state[request_id] = state
        self._request_to_seq[request_id] = seq_id
        return seq_id

    def _do_add_prefilled(self, prefill_result, request_id, stop, event_queue) -> tuple[int | None, list[int]]:
        if request_id in self._request_to_state:
            raise ValueError(f"duplicate request_id: {request_id}")
        seq_id, initial_tokens = self.engine.import_prefilled(prefill_result)
        initial_tokens = list(initial_tokens or [])
        prompt_tokens = self._extract_prompt_tokens(prefill_result)
        text = self._decode(initial_tokens)
        finished = bool(prefill_result.get("finished", False)) if isinstance(prefill_result, dict) else False
        if seq_id is None:
            finished = True
        finish_reason = prefill_result.get("finish_reason") if isinstance(prefill_result, dict) else None
        if finished and finish_reason is None:
            finish_reason = "length"
        state = RequestState(
            request_id=request_id,
            seq_id=seq_id,
            queue=event_queue,
            stop=self._normalize_stop(stop),
            prompt_tokens=prompt_tokens,
            prev_text=text,
            finished=finished,
            finish_reason=finish_reason,
            token_ids=initial_tokens,
            text=text,
        )
        if seq_id is not None:
            self._seq_to_state[seq_id] = state
            self._request_to_seq[request_id] = seq_id
        self._request_to_state[request_id] = state
        if finished:
            event = self._build_event(state, initial_tokens, text, "", True, state.finish_reason)
            self._emit(state, event)
            self._remove_active(state)
            return None, initial_tokens
        return seq_id, initial_tokens

    def _do_abort(self, request_id) -> bool:
        state = self._request_to_state.get(request_id)
        if state is None:
            seq_id = self._request_to_seq.get(request_id)
            if seq_id is None:
                return False
            return bool(self.engine.abort_request(seq_id))
        if state.finished:
            return True
        if state.seq_id is not None:
            try:
                self.engine.abort_request(state.seq_id)
            except BaseException:
                traceback.print_exc()
        state.finished = True
        if state.finish_reason is None:
            state.finish_reason = "stop"
        event = self._build_event(state, state.token_ids, state.text, "", True, state.finish_reason)
        self._emit(state, event)
        self._cleanup_state(state)
        return True

    def _do_shutdown(self) -> None:
        self._shutdown_requested = True

    def _dispatch(self, updates) -> None:
        for seq_id, completion_token_ids, is_finished in updates:
            state = self._seq_to_state.get(seq_id)
            if state is None or state.finished:
                continue
            token_ids = list(completion_token_ids)
            text = self._decode(token_ids)
            stop_index = self._find_stop(text, state.stop)
            if stop_index is not None:
                text = text[:stop_index]
                delta_text = self._delta_text(text, state.prev_text)
                state.prev_text = text
                state.text = text
                state.token_ids = token_ids
                state.finished = True
                state.finish_reason = "stop"
                try:
                    self.engine.abort_request(seq_id)
                except BaseException:
                    traceback.print_exc()
                event = self._build_event(state, token_ids, text, delta_text, True, "stop")
                self._emit(state, event)
                self._remove_active(state)
                continue
            delta_text = self._delta_text(text, state.prev_text)
            state.prev_text = text
            state.text = text
            state.token_ids = token_ids
            if is_finished:
                state.finished = True
                state.finish_reason = "length"
                event = self._build_event(state, token_ids, text, delta_text, True, "length")
                self._emit(state, event)
                self._remove_active(state)
            else:
                event = self._build_event(state, token_ids, text, delta_text, False, None)
                self._emit(state, event)

    def _fail_all(self) -> None:
        for state in list(self._request_to_state.values()):
            if state.finished:
                continue
            state.finished = True
            state.finish_reason = "stop"
            event = self._build_event(state, state.token_ids, state.text, "", True, "stop")
            self._emit(state, event)
            self._remove_active(state)

    def _build_event(self, state, token_ids, text, delta_text, finished, finish_reason) -> dict:
        return {
            "request_id": state.request_id,
            "seq_id": state.seq_id if state.seq_id is not None else -1,
            "token_ids": list(token_ids),
            "text": text,
            "delta_text": delta_text,
            "finished": finished,
            "num_prompt_tokens": len(state.prompt_tokens),
            "num_generated_tokens": len(token_ids),
            "finish_reason": finish_reason,
        }

    def _emit(self, state, event) -> None:
        if self.loop.is_closed():
            return
        try:
            self.loop.call_soon_threadsafe(state.queue.put_nowait, event)
        except RuntimeError:
            pass

    def _decode(self, token_ids) -> str:
        return self.engine.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def _delta_text(self, text, prev_text) -> str:
        if text.startswith(prev_text):
            return text[len(prev_text):]
        if prev_text.startswith(text):
            return ""
        return text

    def _find_stop(self, text, stops) -> int | None:
        best_index = None
        for stop in stops:
            if not stop:
                continue
            index = text.find(stop)
            if index != -1 and (best_index is None or index < best_index):
                best_index = index
        return best_index

    def _extract_prompt_tokens(self, result) -> list[int]:
        if not isinstance(result, dict):
            return []
        for key in ("prompt_token_ids", "prompt_tokens", "prompt", "input_ids"):
            if key not in result:
                continue
            value = result[key]
            if isinstance(value, str):
                return self.engine.tokenizer.encode(value)
            if isinstance(value, (list, tuple)):
                return list(value)
        value = result.get("num_prompt_tokens")
        if isinstance(value, int) and value >= 0:
            return [0] * value
        return []

    def _normalize_stop(self, stop) -> list[str]:
        if stop is None:
            return []
        if isinstance(stop, str):
            return [stop] if stop else []
        return [item for item in stop if item]

    def _remove_active(self, state) -> None:
        if state.seq_id is not None and self._seq_to_state.get(state.seq_id) is state:
            self._seq_to_state.pop(state.seq_id, None)
        if self._request_to_seq.get(state.request_id) == state.seq_id:
            self._request_to_seq.pop(state.request_id, None)

    def _cleanup_state(self, state) -> None:
        self._remove_active(state)
        if self._request_to_state.get(state.request_id) is state:
            self._request_to_state.pop(state.request_id, None)
