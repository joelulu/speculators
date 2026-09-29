"""Single-block teacher-forced training utilities for the H-Spec reference.

Layer numbers in the paper are one-based. Hugging Face cache layers are
zero-based; hidden_states[0] is the embedding output. Prefix KV comes from
the *same* frozen target forward as the teacher logits, then is sliced to the
confirmed prefix. This avoids leaking future-token KV into the drafter.
"""

from __future__ import annotations

import torch
from torch.nn import functional

from .reference import HSpecReferenceConfig, TargetContext

_TOKEN_RANK = 2


def _layer_kv(cache, layer_number: int) -> tuple[torch.Tensor, torch.Tensor]:
    index = layer_number - 1
    if hasattr(cache, "layers"):
        layer = cache.layers[index]
        return layer.keys, layer.values
    # Older transformers expose a legacy tuple of (key, value) pairs.
    if isinstance(cache, (list, tuple)):
        return cache[index][:2]
    raise TypeError("Unsupported target cache layout; expected DynamicCache.layers")


def extract_context(
    target_output,
    prefix_length: int,
    config: HSpecReferenceConfig,
) -> TargetContext:
    """Take read-only prefix views from a target forward with output_hidden_states.

    The target must be a Qwen3-style model that stores post-RoPE keys. Cache
    layout and 1-based target-layer mapping must be verified for other families.
    """
    if prefix_length < 1 or target_output.past_key_values is None:
        raise ValueError("need a nonempty prefix and target use_cache=True")
    states = target_output.hidden_states
    if states is None or max(config.target_hidden_layers) >= len(states):
        raise ValueError("target did not return the requested hidden layers")
    last_hidden = torch.stack(
        [states[layer][:, prefix_length - 1] for layer in config.target_hidden_layers],
        dim=1,
    ).detach()
    kv = []
    for layer in config.target_kv_layers:
        key, value = _layer_kv(target_output.past_key_values, layer)
        if key.shape[-2] < prefix_length or value.shape[-2] < prefix_length:
            raise ValueError("target KV shorter than the confirmed prefix")
        kv.append(
            (key[:, :, :prefix_length].detach(), value[:, :, :prefix_length].detach())
        )
    return TargetContext(
        last_hidden=last_hidden, prefix_kv=tuple(kv), prefix_length=prefix_length
    )


def training_block(
    input_ids: torch.Tensor,
    target_output,
    prefix_length: int,
    mask_token_id: int,
    config: HSpecReferenceConfig,
    *,
    draft_tokens: int = 7,
) -> tuple[torch.Tensor, TargetContext, torch.Tensor, torch.Tensor]:
    """Build [anchor, MASK x k], target context, previous ids, teacher logits.

    Targets are logits at positions prefix-1 through prefix+k-2. They predict
    ground-truth tokens at positions prefix through prefix+k-1. The anchor's
    draft logit is excluded from the loss.
    """
    if input_ids.ndim != _TOKEN_RANK or prefix_length < 1 or draft_tokens < 1:
        raise ValueError("expected [batch, time], a prefix, and draft tokens")
    if prefix_length + draft_tokens > input_ids.shape[1]:
        raise ValueError("training sample needs k ground-truth continuation tokens")
    anchor = input_ids[:, prefix_length - 1 : prefix_length]
    masks = torch.full(
        (input_ids.shape[0], draft_tokens),
        mask_token_id,
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    block = torch.cat((anchor, masks), dim=1)
    previous = torch.cat(
        (
            anchor,
            anchor,
            input_ids[:, prefix_length : prefix_length + draft_tokens - 1],
        ),
        dim=1,
    )
    teacher = target_output.logits[
        :, prefix_length - 1 : prefix_length + draft_tokens - 1
    ].detach()
    return (
        block,
        extract_context(target_output, prefix_length, config),
        previous,
        teacher,
    )


def ce_tv_loss(
    draft_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    ce_weight: float = 0.1,
    decay_gamma: float = 4.0,
) -> torch.Tensor:
    """Position-weighted CE + TV loss, matching DFlash's reduction.

    ``draft_logits`` has the unused anchor logit at slot zero. ``labels`` and
    ``teacher_logits`` contain only k actual continuation positions.
    """
    predicted = draft_logits[:, 1:].float()
    if predicted.shape != teacher_logits.shape or labels.shape != predicted.shape[:2]:
        raise ValueError("draft, teacher, and labels must align on k positions")
    if decay_gamma <= 0 or not 0 <= ce_weight <= 1:
        raise ValueError("decay_gamma must be positive and ce_weight in [0, 1]")
    ce = functional.cross_entropy(predicted.transpose(1, 2), labels, reduction="none")
    tv = 0.5 * (predicted.softmax(-1) - teacher_logits.float().softmax(-1)).abs().sum(
        -1
    )
    positions = torch.arange(predicted.shape[1], device=predicted.device)
    weights = torch.exp(-positions.float() / decay_gamma)
    return ((ce_weight * ce + (1.0 - ce_weight) * tv) * weights).mean()
