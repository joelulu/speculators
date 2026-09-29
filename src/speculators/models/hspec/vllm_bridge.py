"""Build reference H-Spec context from a vLLM FlashAttention target cache.

This correctness bridge gathers a bounded dense window. It does not replace
the fused, allocation-free paged attention needed for a serving-speed drafter.
"""

from __future__ import annotations

from importlib import import_module

import torch

from .reference import HSpecReferenceConfig, TargetContext


def context_from_vllm_pages(
    last_hidden: torch.Tensor,
    caches: dict[str, torch.Tensor],
    block_tables: dict[str, torch.Tensor],
    prefix_length: int,
    config: HSpecReferenceConfig,
) -> TargetContext:
    """Borrow verified target pages for one request without modifying the cache.

    ``last_hidden`` is the five selected target layers at the final confirmed
    position, shaped [1, 5, hidden_size]. Block tables must describe the same
    request and the same confirmed prefix as these hidden states.
    """
    borrow_target_kv_layers = import_module(
        "vllm.v1.spec_decode.hspec_kv"
    ).borrow_target_kv_layers

    if last_hidden.shape != (1, 5, config.hidden_size):
        raise ValueError("expected five last-token hidden states for one request")
    kv = borrow_target_kv_layers(
        caches,
        block_tables,
        prefix_length,
        layer_numbers=config.target_kv_layers,
        head_dim=config.head_dim,
        window=config.sliding_window,
    )
    if any(
        key.shape[1] != config.num_key_value_heads or key.device != last_hidden.device
        for key, _ in kv
    ):
        raise ValueError("target KV heads or device do not match the drafter")
    return TargetContext(
        last_hidden=last_hidden.detach(),
        prefix_kv=kv,
        prefix_length=prefix_length,
    )
