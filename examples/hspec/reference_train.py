"""Run a small H-Spec Qwen3 correctness/training experiment.

Example (on a GPU with enough memory for the frozen target):
    python examples/hspec/reference_train.py \
      --target Qwen/Qwen3-8B --texts ./train_texts.txt --steps 10 \
      --out ./hspec_reference.pt

This is not the 100K-sample packed training recipe or the vLLM serving path.
"""

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from speculators.models.hspec import HSpecReference, HSpecReferenceConfig
from speculators.models.hspec.training import ce_tv_loss, training_block


def config_for_target(target_config) -> HSpecReferenceConfig:
    if target_config.model_type != "qwen3":
        raise ValueError(
            "reference extraction currently supports Qwen3 full-attention targets"
        )
    depth = target_config.num_hidden_layers
    if depth != 36:
        raise ValueError("paper's Qwen3-4B/8B mapping requires 36 target layers")
    heads = 44 if target_config.hidden_size == 2560 else 48
    rope = getattr(target_config, "rope_parameters", None) or {}
    if isinstance(rope, dict):
        rope_theta = rope.get(
            "rope_theta", getattr(target_config, "rope_theta", 1_000_000.0)
        )
    else:
        rope_theta = getattr(target_config, "rope_theta", 1_000_000.0)
    return HSpecReferenceConfig(
        hidden_size=target_config.hidden_size,
        vocab_size=target_config.vocab_size,
        num_attention_heads=target_config.num_attention_heads,
        num_key_value_heads=target_config.num_key_value_heads,
        head_dim=target_config.head_dim,
        intermediate_size=target_config.intermediate_size,
        mamba_heads=heads,
        rope_theta=rope_theta,
        rms_norm_eps=target_config.rms_norm_eps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--texts",
        type=Path,
        required=True,
        help="one already-prepared training text per line",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--draft-tokens", type=int, default=7)
    parser.add_argument("--max-prefix", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("this script needs a GPU for the frozen Qwen3 target")
    device = torch.device("cuda")
    target = (
        AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    target.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    config = config_for_target(target.config)
    embedding_weight = target.get_input_embeddings().weight
    output_layer = target.get_output_embeddings()
    output_weight = (
        output_layer.weight if output_layer is not None else embedding_weight
    )
    drafter = HSpecReference(config, embedding_weight, output_weight).to(
        device=device, dtype=torch.bfloat16
    )
    optimizer = torch.optim.AdamW(
        (p for p in drafter.parameters() if p.requires_grad), lr=6e-4, weight_decay=0.01
    )

    texts = [
        line.strip() for line in args.texts.read_text().splitlines() if line.strip()
    ]
    if not texts:
        raise ValueError("text file contains no training examples")
    for step in range(args.steps):
        text = texts[step % len(texts)]
        ids = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_prefix + args.draft_tokens,
        )["input_ids"].to(device)
        if ids.shape[1] <= args.draft_tokens + 1:
            raise ValueError(
                f"line {step % len(texts) + 1} too short for the requested draft length"
            )
        prefix_length = min(ids.shape[1] - args.draft_tokens, args.max_prefix)
        with torch.no_grad():
            output = target(
                input_ids=ids[:, : prefix_length + args.draft_tokens],
                use_cache=True,
                output_hidden_states=True,
            )
        block, context, previous, teacher = training_block(
            ids,
            output,
            prefix_length,
            tokenizer.eos_token_id,
            config,
            draft_tokens=args.draft_tokens,
        )
        logits = drafter(block, context, previous)
        loss = ce_tv_loss(
            logits, teacher, ids[:, prefix_length : prefix_length + args.draft_tokens]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(drafter.parameters(), max_norm=1.0)
        optimizer.step()
        print(
            f"step={step + 1} loss={loss.item():.4f} prefix={prefix_length}", flush=True
        )
        del output, context

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": drafter.state_dict(), "config": vars(config)}, args.out)
    git_executable = shutil.which("git")
    sha = (
        subprocess.run(  # noqa: S603 - resolved git executable and fixed arguments
            [git_executable, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if git_executable
        else "unknown"
    )
    args.out.with_suffix(args.out.suffix + ".json").write_text(
        json.dumps(
            {
                "git_sha": sha,
                "target": args.target,
                "steps": args.steps,
                "seed": args.seed,
                "draft_tokens": args.draft_tokens,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
