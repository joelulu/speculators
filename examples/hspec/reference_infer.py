"""Offline, greedy H-Spec proposal and exact target verification.

This recomputes the target prefix and candidate sequence every block. It is a
correctness demonstration, not a vLLM serving path or a speed benchmark.
"""

import argparse
from dataclasses import fields
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from speculators.models.hspec import HSpecReference, HSpecReferenceConfig
from speculators.models.hspec.training import extract_context


def greedy_verify_block(
    target_logits: torch.Tensor,
    proposals: torch.Tensor,
    limit: int,
) -> tuple[torch.Tensor, int]:
    """Return verified tokens, plus the number of accepted draft tokens.

    ``target_logits`` contains the last k+1 positions from a target forward
    over prefix + k proposals. The first predicts proposal 0; the last
    predicts the additional token after all k proposals are accepted.
    At the first rejection, emit the target's greedy choice at that position.
    When all proposals pass, emit one extra target token if the limit permits.
    """
    if proposals.ndim != 2 or proposals.shape[0] != 1 or proposals.shape[1] < 1:
        raise ValueError("expected at least one proposal for one request")
    if limit < 1:
        raise ValueError("limit must be positive")
    if target_logits.ndim != 3 or target_logits.shape[:2] != (
        1,
        proposals.shape[1] + 1,
    ):
        raise ValueError("target logits must cover every proposal and an extra token")
    verified = []
    accepted = 0
    for index in range(min(proposals.shape[1], limit)):
        choice = target_logits[:, index].argmax(-1)
        if not torch.equal(choice, proposals[:, index]):
            verified.append(choice)
            return torch.stack(verified, dim=1), accepted
        verified.append(choice)
        accepted += 1
    if accepted == proposals.shape[1] and len(verified) < limit:
        verified.append(target_logits[:, -1].argmax(-1))
    return torch.stack(verified, dim=1), accepted


def load_drafter(
    checkpoint_path: Path, target: torch.nn.Module, device: torch.device
) -> tuple[HSpecReference, HSpecReferenceConfig]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = HSpecReferenceConfig(**checkpoint["config"])
    hf_config = target.config
    if hf_config.model_type != "qwen3" or hf_config.num_hidden_layers != 36:
        raise ValueError("reference inference requires a 36-layer Qwen3 target")
    expected = {
        "hidden_size": hf_config.hidden_size,
        "vocab_size": hf_config.vocab_size,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_key_value_heads": hf_config.num_key_value_heads,
        "head_dim": hf_config.head_dim,
        "intermediate_size": hf_config.intermediate_size,
    }
    if any(getattr(config, field) != value for field, value in expected.items()):
        raise ValueError("checkpoint dimensions do not match the selected target")
    if set(checkpoint["config"]) != {field.name for field in fields(config)}:
        raise ValueError("checkpoint configuration is incomplete")
    rope = getattr(hf_config, "rope_parameters", None) or {}
    rope_theta = (
        rope.get("rope_theta", getattr(hf_config, "rope_theta", 1_000_000.0))
        if isinstance(rope, dict)
        else getattr(hf_config, "rope_theta", 1_000_000.0)
    )
    if config.rope_theta != rope_theta:
        raise ValueError("checkpoint RoPE theta does not match the target")
    drafter = HSpecReference(config).to(device=device, dtype=torch.bfloat16).eval()
    drafter.load_state_dict(checkpoint["model"])
    return drafter, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--draft-tokens", type=int, default=7)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.draft_tokens < 1:
        parser.error("token counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("the Qwen3 target and drafter require a GPU")
    device = torch.device("cuda")
    target = (
        AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    target.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    if tokenizer.eos_token_id is None:
        raise ValueError("target tokenizer must have an EOS token")
    drafter, config = load_drafter(args.checkpoint, target, device)
    ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)
    if ids.shape[1] < 1:
        raise ValueError("prompt must contain at least one token")
    initial_length = ids.shape[1]
    accepted_total = 0
    blocks = 0
    with torch.inference_mode():
        while ids.shape[1] - initial_length < args.max_new_tokens:
            prefix_length = ids.shape[1]
            remaining = args.max_new_tokens - (prefix_length - initial_length)
            draft_count = min(args.draft_tokens, remaining)
            prefix_output = target(
                ids, use_cache=True, output_hidden_states=True, logits_to_keep=1
            )
            context = extract_context(prefix_output, prefix_length, config)
            block = torch.cat(
                (
                    ids[:, -1:],
                    torch.full(
                        (1, draft_count),
                        tokenizer.eos_token_id,
                        device=device,
                        dtype=ids.dtype,
                    ),
                ),
                dim=1,
            )
            proposals, _ = drafter.draft_block(block, context)
            del context, prefix_output
            verification = target(
                torch.cat((ids, proposals), dim=1),
                use_cache=False,
                logits_to_keep=draft_count + 1,
            ).logits
            verified, accepted = greedy_verify_block(verification, proposals, remaining)
            del verification
            eos_positions = (verified[0] == tokenizer.eos_token_id).nonzero()
            if eos_positions.numel():
                verified = verified[:, : eos_positions[0, 0].item() + 1]
            ids = torch.cat((ids, verified), dim=1)
            accepted_total += min(accepted, verified.shape[1])
            blocks += 1
            if eos_positions.numel():
                break
    print(tokenizer.decode(ids[0, initial_length:], skip_special_tokens=True))
    print(f"verified_blocks={blocks} accepted_draft_tokens={accepted_total}")


if __name__ == "__main__":
    main()
