import datetime
import os
import pickle
import socket
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.device import (
    device_type,
    is_npu,
    device,
    set_device,
    synchronize,
    empty_cache,
    reset_peak_memory_stats,
    mem_get_info,
    memory_stats,
    memory_allocated,
    dist_backend,
    to_device,
    tensor_from_list,
)
from nanovllm.utils.loader import load_model


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        set_device(rank)
        if self.world_size == 1:
            backend = "gloo"
            port = _free_port()
        else:
            backend = dist_backend()
            port = int(os.environ.get("NANO_VLLM_DIST_PORT", "2333"))
        timeout = datetime.timedelta(seconds=int(os.environ.get("NANO_VLLM_DIST_TIMEOUT", "300")))
        dist.init_process_group(backend, f"tcp://127.0.0.1:{port}", world_size=self.world_size, rank=rank, timeout=timeout)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.text_config.dtype)
        torch.set_default_device(device(rank).type)
        if config.model_type.startswith("qwen3_5"):
            from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
            self.model = Qwen3_5ForCausalLM(config.text_config)
        else:
            self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.use_paged_attention = is_npu() or device_type() == "cpu" or config.model_type.startswith("qwen3_5")
        self.enforce_eager = config.enforce_eager or device_type() != "cuda"
        if hasattr(self.model, "allocate_state_cache"):
            self.model.allocate_state_cache(config.max_num_seqs)
        self.warmup_model()
        self.allocate_kv_cache()
        if torch.cuda.is_available() and not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        empty_cache()
        reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
            seq.state_slot = 0
        self.run(seqs, True)
        empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        text_config = config.text_config
        free, total = mem_get_info()
        used = total - free
        stats = memory_stats()
        peak = stats.get("allocated_bytes.all.peak", memory_allocated())
        current = stats.get("allocated_bytes.all.current", memory_allocated())
        num_kv_heads = text_config.num_key_value_heads // self.world_size
        head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)
        num_kv_layers = sum(1 for module in self.model.modules() if hasattr(module, "k_cache") and hasattr(module, "v_cache"))
        block_bytes = 2 * num_kv_layers * self.block_size * num_kv_heads * head_dim * text_config.dtype.itemsize
        if total == 0:
            config.num_kvcache_blocks = 256
        else:
            config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, num_kv_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32)
        return to_device(block_tables, self.rank)

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        context_lens = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            context_lens.append(end)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if (self.use_paged_attention and all(seq.block_table for seq in seqs)) or cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        input_ids = tensor_from_list(input_ids, torch.int64, self.rank)
        positions = tensor_from_list(positions, torch.int64, self.rank)
        cu_seqlens_q = tensor_from_list(cu_seqlens_q, torch.int32, self.rank)
        cu_seqlens_k = tensor_from_list(cu_seqlens_k, torch.int32, self.rank)
        slot_mapping = tensor_from_list(slot_mapping, torch.int32, self.rank)
        context_lens = tensor_from_list(context_lens, torch.int32, self.rank)
        state_slots = [seq.state_slot if seq.state_slot >= 0 else 0 for seq in seqs]
        reset_slots = [slot for seq, slot in zip(seqs, state_slots) if seq.num_cached_tokens == 0]
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, state_slots, reset_slots)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = tensor_from_list(input_ids, torch.int64, self.rank)
        positions = tensor_from_list(positions, torch.int64, self.rank)
        slot_mapping = tensor_from_list(slot_mapping, torch.int32, self.rank)
        context_lens = tensor_from_list(context_lens, torch.int32, self.rank)
        block_tables = self.prepare_block_tables(seqs)
        state_slots = [seq.state_slot if seq.state_slot >= 0 else 0 for seq in seqs]
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, state_slots=state_slots, reset_slots=None)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32)
        return to_device(temperatures, self.rank)

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512 or device_type() != "cuda":
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
