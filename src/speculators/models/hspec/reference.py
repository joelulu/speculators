"""Correctness-oriented H-Spec model with an explicit target-context interface.

This module is intentionally independent of vLLM's paged KV allocator. Prefix
K/V tensors are borrowed from the target, never registered as model buffers or
persisted in a drafter cache. The Mamba-2 state update below uses a PyTorch
associative scan, not the paper's fused CUDA scan; benchmark only after adding
an optimized initial-state-aware SSD kernel and wiring in-place paged KV access.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

_HIDDEN_LAYERS = 5
_KV_LAYERS = 3
_BLOCK_RANK = 2
_SCAN_RANK = 3


@dataclass(frozen=True)
class HSpecReferenceConfig:
    hidden_size: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    target_hidden_layers: tuple[int, ...] = (1, 9, 17, 25, 34)
    target_kv_layers: tuple[int, ...] = (17, 25, 34)
    mamba_heads: int = 48
    mamba_head_dim: int = 64
    mamba_state_dim: int = 4
    mamba_groups: int = 4
    conv_kernel: int = 4
    sliding_window: int = 2048
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    markov_rank: int = 256

    def __post_init__(self) -> None:
        if (
            len(self.target_hidden_layers) != _HIDDEN_LAYERS
            or len(self.target_kv_layers) != _KV_LAYERS
        ):
            raise ValueError("H-Spec needs five hidden layers and three KV layers")
        if any(i not in self.target_hidden_layers for i in self.target_kv_layers):
            raise ValueError("KV layers must be among the selected hidden layers")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("attention heads must be divisible by KV heads")
        if self.mamba_heads % self.mamba_groups:
            raise ValueError("Mamba heads must be divisible by groups")
        if self.head_dim % 2 or self.sliding_window < 1:
            raise ValueError(
                "RoPE head dimension must be even; window must be positive"
            )


@dataclass(frozen=True)
class TargetContext:
    """Inputs from a frozen target model; layer numbers are one-based.

    last_hidden: [batch, five layers, hidden]. Each prefix KV pair is
    [batch, kv_heads, prefix_length, head_dim], with K already RoPE rotated.
    All examples in a batch currently share one prefix length. No gradients
    should flow into target tensors.
    """

    last_hidden: torch.Tensor
    prefix_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    # Absolute confirmed length when prefix_kv contains only a borrowed window.
    prefix_length: int | None = None


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + self.eps)).to(
            x.dtype
        ) * self.weight


def _rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Apply target-compatible half-split RoPE to [B, heads, time, dim]."""
    dim = x.shape[-1]
    inv = theta ** (-torch.arange(0, dim, 2, device=x.device).float() / dim)
    phase = positions.float()[:, None] * inv[None, :]
    cosine = torch.cat((phase.cos(), phase.cos()), dim=-1)[None, None]
    sine = torch.cat((phase.sin(), phase.sin()), dim=-1)[None, None]
    a, b = x.chunk(2, dim=-1)
    return x * cosine + torch.cat((-b, a), dim=-1) * sine


class SelectiveMamba2Reference(nn.Module):
    """Mamba-2 grouped selective recurrence with an injected initial SSD state.

    The convolution has zero block history. H-Spec supplies prefix context
    through the initial SSD state and attention; it does not keep conv history
    between verified blocks. A logarithmic-depth associative scan evaluates
    positions in parallel, but separate PyTorch launches are not a speed
    replacement for a fused SSD kernel.
    """

    def __init__(self, c: HSpecReferenceConfig) -> None:
        super().__init__()
        self.heads = c.mamba_heads
        self.head_dim = c.mamba_head_dim
        self.state_dim = c.mamba_state_dim
        self.groups = c.mamba_groups
        inner = self.heads * self.head_dim
        conv_width = inner + 2 * self.groups * self.state_dim
        self.in_proj = nn.Linear(
            c.hidden_size, 2 * inner + 2 * self.groups * self.state_dim + self.heads
        )
        self.conv = nn.Conv1d(conv_width, conv_width, c.conv_kernel, groups=conv_width)
        self.a_log = nn.Parameter(torch.zeros(self.heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.heads))
        self.d_skip = nn.Parameter(torch.ones(self.heads))
        self.out_proj = nn.Linear(inner, c.hidden_size)

    def forward(self, x: torch.Tensor, initial_state: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        heads, width, state_dim = self.heads, self.head_dim, self.state_dim
        inner = heads * width
        z, xbc, dt = torch.split(
            self.in_proj(x), [inner, inner + 2 * self.groups * state_dim, heads], dim=-1
        )
        # Left padding makes the depthwise convolution causal inside the block.
        xbc = self.conv(
            functional.pad(xbc.transpose(1, 2), (self.conv.kernel_size[0] - 1, 0))
        )
        xbc = functional.silu(xbc.transpose(1, 2))
        values, b, c = torch.split(
            xbc, [inner, self.groups * state_dim, self.groups * state_dim], dim=-1
        )
        values = values.reshape(batch, length, heads, width)
        repeat = heads // self.groups
        b = b.reshape(batch, length, self.groups, state_dim).repeat_interleave(
            repeat, dim=2
        )
        c = c.reshape(batch, length, self.groups, state_dim).repeat_interleave(
            repeat, dim=2
        )
        delta = functional.softplus(dt.float() + self.dt_bias.float())
        decay = torch.exp(-delta * self.a_log.float().exp())
        if initial_state.shape != (batch, heads, width, state_dim):
            raise ValueError(
                "initial_state has wrong [batch, head, width, state] shape"
            )
        updates = (
            delta[..., None, None] * values.float()[..., None] * b.float()[..., None, :]
        )
        states = _parallel_state_scan(decay, updates, initial_state.float())
        result = (states * c.float()[..., None, :]).sum(-1)
        result = result + self.d_skip.float()[None, None, :, None] * values.float()
        result = result.reshape(batch, length, inner)
        return self.out_proj((result * functional.silu(z.float())).to(x.dtype))


def _parallel_state_scan(
    decay: torch.Tensor, updates: torch.Tensor, initial_state: torch.Tensor
) -> torch.Tensor:
    """Compute S[t] = decay[t] * S[t-1] + updates[t] in log2(T) stages.

    A segment is represented by (a, u): applying it to an incoming state
    yields a * state + u. Composition of adjacent segments is associative.
    Keeping the initial state separate makes gradients reach the shared
    hidden-state projection, including for a one-position block.
    """
    if decay.ndim != _SCAN_RANK or updates.shape[:_SCAN_RANK] != decay.shape:
        raise ValueError("scan inputs must have matching [batch, time, heads]")
    if initial_state.shape != updates.shape[:1] + updates.shape[2:]:
        raise ValueError("initial state must have [batch, heads, width, state]")
    products = decay
    sums = updates
    offset = 1
    while offset < decay.shape[1]:
        right_products = products[:, offset:]
        products = torch.cat(
            (products[:, :offset], right_products * products[:, :-offset]), dim=1
        )
        sums = torch.cat(
            (
                sums[:, :offset],
                sums[:, offset:] + right_products[..., None, None] * sums[:, :-offset],
            ),
            dim=1,
        )
        offset *= 2
    return sums + products[..., None, None] * initial_state[:, None]


class TargetKVAttention(nn.Module):
    """Attend to a borrowed target KV prefix and causal in-block K/V."""

    def __init__(self, c: HSpecReferenceConfig) -> None:
        super().__init__()
        self.c = c
        self.q_proj = nn.Linear(c.hidden_size, c.num_attention_heads * c.head_dim)
        self.k_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * c.head_dim)
        self.v_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * c.head_dim)
        self.o_proj = nn.Linear(c.num_attention_heads * c.head_dim, c.hidden_size)
        self.q_norm = RMSNorm(c.head_dim, c.rms_norm_eps)
        self.k_norm = RMSNorm(c.head_dim, c.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        prefix: tuple[torch.Tensor, torch.Tensor],
        absolute_prefix_length: int | None = None,
    ) -> torch.Tensor:
        c = self.c
        batch, length, _ = x.shape
        key, value = prefix
        expected = (batch, c.num_key_value_heads, key.shape[-2], c.head_dim)
        if key.shape != expected or value.shape != expected:
            raise ValueError(
                "target KV shape or head layout does not match the drafter"
            )
        if key.device != x.device or value.device != x.device:
            raise ValueError("target KV and drafter must share a device")
        prefix_len = key.shape[-2]
        absolute_prefix_length = (
            prefix_len if absolute_prefix_length is None else absolute_prefix_length
        )
        if absolute_prefix_length < prefix_len:
            raise ValueError("borrowed KV window exceeds the confirmed prefix")
        # Slice before concatenation so temporary KV storage is O(window + block).
        offset = max(0, prefix_len - c.sliding_window)
        start = absolute_prefix_length - prefix_len + offset
        key, value = key[:, :, offset:], value[:, :, offset:]
        positions = torch.arange(
            absolute_prefix_length - 1,
            absolute_prefix_length + length - 1,
            device=x.device,
        )
        q = self.q_norm(
            self.q_proj(x).reshape(batch, length, c.num_attention_heads, c.head_dim)
        )
        k = self.k_norm(
            self.k_proj(x).reshape(batch, length, c.num_key_value_heads, c.head_dim)
        )
        v = self.v_proj(x).reshape(batch, length, c.num_key_value_heads, c.head_dim)
        q = _rope(q.transpose(1, 2), positions, c.rope_theta)
        k = _rope(k.transpose(1, 2), positions, c.rope_theta)
        k = torch.cat((key, k), dim=-2)
        v = torch.cat((value, v.transpose(1, 2)), dim=-2)
        group = c.num_attention_heads // c.num_key_value_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        key_positions = torch.arange(
            start, absolute_prefix_length + length, device=x.device
        )
        allowed = (key_positions[None] <= positions[:, None]) & (
            key_positions[None] > positions[:, None] - c.sliding_window
        )
        attended = functional.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        return self.o_proj(attended.transpose(1, 2).reshape(batch, length, -1))


class HSpecLayer(nn.Module):
    def __init__(self, c: HSpecReferenceConfig, with_attention: bool) -> None:
        super().__init__()
        self.mamba_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.mamba = SelectiveMamba2Reference(c)
        self.attn_norm = (
            RMSNorm(c.hidden_size, c.rms_norm_eps) if with_attention else None
        )
        self.attn = TargetKVAttention(c) if with_attention else None
        self.mlp_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        initial_state: torch.Tensor,
        prefix: tuple[torch.Tensor, torch.Tensor] | None,
        absolute_prefix_length: int | None = None,
    ) -> torch.Tensor:
        x = x + self.mamba(self.mamba_norm(x), initial_state)
        if self.attn is not None:
            if prefix is None:
                raise ValueError("attention layer needs its mapped target KV")
            x = x + self.attn(self.attn_norm(x), prefix, absolute_prefix_length)
        normalized = self.mlp_norm(x)
        return x + self.down_proj(
            functional.silu(self.gate_proj(normalized)) * self.up_proj(normalized)
        )


class HSpecReference(nn.Module):
    """Three Mamba/attention/MLP layers and one Mamba/MLP layer.

    Pass a verifier's frozen embedding weight to initialize ``embed_tokens``.
    ``previous_token_ids`` supplies teacher-forced previous tokens for Markov
    correction. The returned logits remain *draft* proposals; a target verifier
    must still check and accept them using a valid speculative sampler.
    """

    def __init__(
        self,
        config: HSpecReferenceConfig,
        embedding_weight: torch.Tensor | None = None,
        lm_head_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if embedding_weight is not None:
            if embedding_weight.shape != self.embed_tokens.weight.shape:
                raise ValueError("target embedding dimensions do not match H-Spec")
            with torch.no_grad():
                self.embed_tokens.weight.copy_(embedding_weight)
        self.embed_tokens.weight.requires_grad_(False)
        self.mask_embedding = nn.Parameter(torch.empty(config.hidden_size))
        nn.init.normal_(self.mask_embedding, std=0.02)
        self.fuse = nn.Linear(
            _HIDDEN_LAYERS * config.hidden_size, config.hidden_size, bias=False
        )
        state_size = config.mamba_heads * config.mamba_head_dim * config.mamba_state_dim
        self.state_proj = nn.Linear(config.hidden_size, state_size, bias=False)
        self.layers = nn.ModuleList(
            [HSpecLayer(config, i < _KV_LAYERS) for i in range(4)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if lm_head_weight is not None:
            if lm_head_weight.shape != self.lm_head.weight.shape:
                raise ValueError("target LM head dimensions do not match H-Spec")
            with torch.no_grad():
                self.lm_head.weight.copy_(lm_head_weight)
        self.lm_head.weight.requires_grad_(False)
        self.markov_w1 = nn.Embedding(config.vocab_size, config.markov_rank)
        self.markov_w2 = nn.Linear(config.markov_rank, config.vocab_size, bias=False)

    def forward(
        self,
        block_input_ids: torch.Tensor,
        target: TargetContext,
        previous_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        if (
            block_input_ids.ndim != _BLOCK_RANK
            or block_input_ids.shape[1] < _BLOCK_RANK
        ):
            raise ValueError(
                "block_input_ids must include an anchor and at least one mask"
            )
        batch = block_input_ids.shape[0]
        if (
            target.last_hidden.shape != (batch, _HIDDEN_LAYERS, c.hidden_size)
            or len(target.prefix_kv) != _KV_LAYERS
        ):
            raise ValueError(
                "target context does not match the configured layer mapping"
            )
        if any(
            key.shape[-2] != target.prefix_kv[0][0].shape[-2]
            for key, _ in target.prefix_kv
        ):
            raise ValueError("selected KV layers must have the same prefix length")
        fused = self.fuse(target.last_hidden.reshape(batch, -1))
        initial_state = self.state_proj(fused).reshape(
            batch, c.mamba_heads, c.mamba_head_dim, c.mamba_state_dim
        )
        x = self.embed_tokens(block_input_ids)
        x = torch.cat(
            (x[:, :1], self.mask_embedding.expand(batch, x.shape[1] - 1, -1)), dim=1
        )
        for index, layer in enumerate(self.layers):
            x = layer(
                x,
                initial_state,
                target.prefix_kv[index] if index < _KV_LAYERS else None,
                target.prefix_length,
            )
        logits = self.lm_head(self.norm(x))
        if previous_token_ids is not None:
            if previous_token_ids.shape != block_input_ids.shape:
                raise ValueError("previous_token_ids must match block shape")
            logits = logits + self.markov_w2(self.markov_w1(previous_token_ids))
        return logits

    @torch.no_grad()
    def draft_block(
        self,
        block_input_ids: torch.Tensor,
        target: TargetContext,
        *,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample Markov-corrected proposals after one parallel backbone call.

        Returns ``(draft_ids, proposal_logits)`` for the masked slots only.
        A speculative verifier must use the *same* proposal distribution
        (including its temperature and any truncation) for exact acceptance.
        This reference method is not wired into a serving sampler.
        """
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        base = self.forward(block_input_ids, target)
        previous = block_input_ids[:, 0]
        tokens = []
        proposals = []
        for slot in range(1, block_input_ids.shape[1]):
            logits = base[:, slot] + self.markov_w2(self.markov_w1(previous))
            proposals.append(logits)
            if temperature == 0:
                previous = logits.argmax(dim=-1)
            else:
                previous = torch.multinomial(
                    (logits / temperature).softmax(-1), 1
                ).squeeze(-1)
            tokens.append(previous)
        return torch.stack(tokens, dim=1), torch.stack(proposals, dim=1)
