import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        runner = getattr(self, "model_runner", None)
        if runner is None:
            return
        runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> int:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        self._validate_request(prompt, sampling_params)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def _validate_request(self, prompt, sampling_params):
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        if sampling_params.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if len(prompt) + sampling_params.max_tokens > self.config.max_model_len:
            raise ValueError("prompt length plus max_tokens exceeds max_model_len")
        if (len(prompt) + self.config.kvcache_block_size - 1) // self.config.kvcache_block_size > self.config.num_kvcache_blocks:
            raise ValueError("prompt exceeds available KV cache capacity")

    def step_updates(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        updates = [(seq.seq_id, list(seq.completion_token_ids), seq.is_finished) for seq in seqs]
        return updates, num_tokens

    def step(self):
        updates, num_tokens = self.step_updates()
        outputs = [(seq_id, token_ids) for seq_id, token_ids, finished in updates if finished]
        return outputs, num_tokens

    def abort_request(self, seq_id: int) -> bool:
        return self.scheduler.abort(seq_id)

    def has_work(self) -> bool:
        return not self.scheduler.is_finished()

    def prefill(self, prompt, sampling_params, hold_blocks: bool = False):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        self._validate_request(prompt, sampling_params)
        seq = Sequence(prompt, sampling_params)
        if hold_blocks:
            seq.ignore_eos = True
            seq.max_tokens = max(seq.max_tokens, seq.num_prompt_tokens + 2)
        self.scheduler.add(seq)
        while not seq.is_finished and seq.num_cached_tokens < seq.num_prompt_tokens:
            seqs, is_prefill = self.scheduler.schedule()
            assert is_prefill and len(seqs) == 1 and seqs[0] is seq
            token_ids = self.model_runner.call("run", seqs, True)
            self.scheduler.postprocess(seqs, token_ids, True)
        block_ids = list(seq.block_table)
        kv = self.model_runner.call("export_kv", block_ids) if block_ids else None
        state = self.model_runner.call("export_state", seq.state_slot) if seq.state_slot >= 0 else None
        result = {
            "token_ids": list(seq.token_ids),
            "prompt_token_ids": list(seq.token_ids[:seq.num_prompt_tokens]),
            "num_prompt_tokens": seq.num_prompt_tokens,
            "block_table": block_ids,
            "state_slot": seq.state_slot,
            "kv": kv,
            "state": state,
            "sampling_params": sampling_params,
            "finished": seq.is_finished,
        }
        self.scheduler.abort(seq.seq_id)
        return result

    def import_prefilled(self, result):
        if result.get("finished"):
            return None, list(result["token_ids"][result["num_prompt_tokens"]:])
        sp = result["sampling_params"]
        seq = Sequence(result["token_ids"], sp)
        seq.num_prompt_tokens = result["num_prompt_tokens"]
        seq.num_cached_tokens = result["num_prompt_tokens"]
        seq.num_scheduled_tokens = 0
        seq.is_prefill = True
        seq.status = SequenceStatus.WAITING
        block_ids = self.scheduler.block_manager.import_sequence(seq, len(result["block_table"]))
        self.model_runner.call("import_kv", block_ids, result["kv"])
        while len(seq.block_table) < seq.num_blocks:
            seq.block_table.append(self.scheduler.block_manager._allocate_block())
        if result.get("state") is not None:
            self.model_runner.call("import_state", seq.state_slot, result["state"])
        self.scheduler.add(seq)
        return seq.seq_id, list(seq.completion_token_ids)

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
