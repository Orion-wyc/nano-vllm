import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.linear import ReplicatedLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.utils.context import get_context


class RMSNormShifted(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        input_dtype = x.dtype
        x = x.float()
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.to(input_dtype)


class RMSNormGated(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))

    def forward(self, x, gate):
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * self.weight.float()
        x = x * F.silu(gate.float())
        return x.to(input_dtype)


def build_rotary_emb(config, head_dim):
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    rotary_dim = int(head_dim * rope_parameters.get("partial_rotary_factor", 1.0))
    base = rope_parameters.get("rope_theta", 1000000)
    try:
        rotary_emb = get_rope(
            head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=base,
        )
    except AssertionError:
        rotary_emb = get_rope(
            rotary_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=base,
        )
    return rotary_emb, rotary_dim


def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim, device=value.device, dtype=value.dtype)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen3_5GatedDeltaNet(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.eps = config.rms_norm_eps

        self.in_proj_qkv = ReplicatedLinear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = ReplicatedLinear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = ReplicatedLinear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = ReplicatedLinear(self.hidden_size, self.num_v_heads, bias=False)
        self.out_proj = RowParallelLinear(self.value_dim, self.hidden_size, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
            padding=self.conv_kernel_size - 1,
        )
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads))
        self.norm = RMSNormGated(self.head_v_dim, eps=self.eps)

        self.conv_state = None
        self.recurrent_state = None

    def allocate_state_cache(self, num_slots):
        self.conv_state = torch.zeros(
            num_slots,
            self.conv_dim,
            self.conv_kernel_size - 1,
            dtype=self.conv1d.weight.dtype,
        )
        self.recurrent_state = torch.zeros(
            num_slots,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            dtype=torch.float32,
        )

    def _clear_states(self, slots):
        if slots:
            index = torch.tensor(slots, dtype=torch.long, device=self.conv_state.device)
            self.conv_state.index_fill_(0, index, 0)
            self.recurrent_state.index_fill_(0, index, 0)

    def _conv_prefill(self, x, conv_state):
        length = x.shape[0]
        conv_input = torch.cat([conv_state, x.transpose(0, 1)], dim=-1)
        out = F.conv1d(conv_input.unsqueeze(0), self.conv1d.weight, None, groups=self.conv_dim)[0]
        out = F.silu(out[:, -length:])
        return out.transpose(0, 1), conv_input[:, -(self.conv_kernel_size - 1):]

    def _conv_decode(self, x, conv_state):
        conv_input = torch.cat([conv_state, x.unsqueeze(-1)], dim=-1)
        out = F.conv1d(conv_input, self.conv1d.weight, None, groups=self.conv_dim)
        out = F.silu(out)
        return out[:, :, 0], conv_input[:, :, -(self.conv_kernel_size - 1):]

    def _split_qkv(self, mixed, shape):
        query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(*shape, self.num_k_heads, self.head_k_dim)
        key = key.reshape(*shape, self.num_k_heads, self.head_k_dim)
        value = value.reshape(*shape, self.num_v_heads, self.head_v_dim)
        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            query = query.repeat_interleave(repeat, dim=-2)
            key = key.repeat_interleave(repeat, dim=-2)
        return query, key, value

    def _project(self, hidden):
        mixed = self.in_proj_qkv(hidden)
        z = self.in_proj_z(hidden)
        b = self.in_proj_b(hidden)
        a = self.in_proj_a(hidden)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        return mixed, z, beta, g

    def _forward_prefill(self, hidden):
        context = get_context()
        self._clear_states(context.reset_slots)
        mixed, z, beta, g = self._project(hidden)
        boundaries = context.cu_seqlens_q.tolist()
        slots = context.state_slots
        outputs = []
        for i in range(len(boundaries) - 1):
            start = int(boundaries[i])
            end = int(boundaries[i + 1])
            length = end - start
            slot = int(slots[i])
            mixed_i, conv_state = self._conv_prefill(mixed[start:end], self.conv_state[slot])
            self.conv_state[slot] = conv_state
            query, key, value = self._split_qkv(mixed_i, (length,))
            out, final_state = chunk_gated_delta_rule(
                query.unsqueeze(0),
                key.unsqueeze(0),
                value.unsqueeze(0),
                g=g[start:end].unsqueeze(0),
                beta=beta[start:end].unsqueeze(0),
                chunk_size=64,
                initial_state=self.recurrent_state[slot].unsqueeze(0),
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            self.recurrent_state[slot] = final_state[0]
            z_i = z[start:end].reshape(-1, self.head_v_dim)
            core = self.norm(out[0].reshape(-1, self.head_v_dim), z_i)
            outputs.append(core.reshape(length, -1))
        return self.out_proj(torch.cat(outputs, dim=0))

    def _forward_decode(self, hidden):
        context = get_context()
        slots = torch.tensor(context.state_slots, dtype=torch.long, device=hidden.device)
        mixed, z, beta, g = self._project(hidden)
        mixed, conv_state = self._conv_decode(mixed, self.conv_state[slots])
        self.conv_state[slots] = conv_state
        batch_size = hidden.shape[0]
        query, key, value = self._split_qkv(mixed, (batch_size, 1))
        out, final_state = recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g.unsqueeze(1),
            beta=beta.unsqueeze(1),
            initial_state=self.recurrent_state[slots],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        self.recurrent_state[slots] = final_state
        core = self.norm(out[:, 0].reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(core.reshape(batch_size, -1))

    def forward(self, hidden):
        if get_context().is_prefill:
            return self._forward_prefill(hidden)
        return self._forward_decode(hidden)


class Qwen3_5FullAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert dist.get_world_size() == 1
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.q_proj = ReplicatedLinear(config.hidden_size, self.q_size * 2, bias=False)
        self.k_proj = ReplicatedLinear(config.hidden_size, self.kv_size, bias=False)
        self.v_proj = ReplicatedLinear(config.hidden_size, self.kv_size, bias=False)
        self.o_proj = ReplicatedLinear(self.q_size, config.hidden_size, bias=False)
        self.q_norm = RMSNormShifted(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNormShifted(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb, self.rotary_dim = build_rotary_emb(config, self.head_dim)
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)

    def forward(self, positions, hidden):
        qg = self.q_proj(hidden).view(-1, self.num_heads, self.head_dim * 2)
        query, gate = qg.chunk(2, dim=-1)
        query = self.q_norm(query)
        key = self.k_norm(self.k_proj(hidden).view(-1, self.num_kv_heads, self.head_dim))
        value = self.v_proj(hidden).view(-1, self.num_kv_heads, self.head_dim)
        if self.rotary_dim == self.head_dim:
            query, key = self.rotary_emb(positions, query, key)
        else:
            query_rot, key_rot = self.rotary_emb(positions, query[..., :self.rotary_dim], key[..., :self.rotary_dim])
            query = torch.cat([query_rot, query[..., self.rotary_dim:]], dim=-1)
            key = torch.cat([key_rot, key[..., self.rotary_dim:]], dim=-1)
        out = self.attn(query, key, value)
        out = out.flatten(1, -1) * torch.sigmoid(gate.flatten(1, -1))
        return self.o_proj(out)


class Qwen3_5MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(config.hidden_size, [config.intermediate_size] * 2, bias=False)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False)
        assert config.hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen3_5DecoderLayer(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5FullAttention(config)
        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = RMSNormShifted(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormShifted(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden):
        residual = hidden
        hidden = self.input_layernorm(hidden)
        if self.layer_type == "linear_attention":
            hidden = self.linear_attn(hidden)
        else:
            hidden = self.self_attn(positions, hidden)
        hidden = residual + hidden
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        return residual + hidden


class Qwen3_5Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.rotary_emb, self.rotary_dim = build_rotary_emb(config, head_dim)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        for layer in self.layers:
            if layer.layer_type == "full_attention":
                layer.self_attn.rotary_emb = self.rotary_emb
                layer.self_attn.rotary_dim = self.rotary_dim
        self.norm = RMSNormShifted(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(positions, hidden)
        return self.norm(hidden)


class Qwen3_5ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.language_model.embed_tokens.weight.data

    def forward(self, input_ids, positions):
        return self.model.language_model(input_ids, positions)

    def compute_logits(self, hidden):
        return self.lm_head(hidden)

    def allocate_state_cache(self, num_slots):
        for layer in self.model.language_model.layers:
            if layer.layer_type == "linear_attention":
                layer.linear_attn.allocate_state_cache(num_slots)
