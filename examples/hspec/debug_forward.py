"""Single-process, single-block inference inspection of DFlash or H-Spec.

Calls the existing decoder modules, without a training loss or vLLM worker.
DFlash uses eager attention for stepping through Python; this is not a runtime
benchmark or a replacement for the serving engine's cache lifecycle.
"""

import argparse
import pdb
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from speculators import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.model_definitions import Qwen3DFlashAttention
from speculators.models.hspec import HSpecReference
from speculators.models.hspec.reference import SelectiveMamba2Reference, TargetKVAttention
from speculators.models.hspec.training import extract_context

from reference_infer import load_drafter


def inspect_dflash(args, target_output, ids, tokenizer, device):
    draft = SpeculatorModel.from_pretrained(
        args.dflash_draft,
        verifier=args.target,
        cache_dir=str(args.conversion_cache),
        dtype=torch.bfloat16,
    ).to(device).eval()
    if not isinstance(draft, DFlashDraftModel):
        raise ValueError("Expected an original DFlash checkpoint")
    if draft.config.sample_from_anchor:
        raise ValueError("This example expects sample_from_anchor=False")
    if args.draft_tokens + 1 > draft.block_size:
        raise ValueError("Requested block exceeds the DFlash checkpoint block size")
    # Use the checkpoint's configured layer mapping, not H-Spec's mapping.
    target_hidden = torch.cat(
        [target_output.hidden_states[i] for i in draft.target_layer_ids], dim=-1
    )
    if target_hidden.shape[-1] != draft.fc.in_features:
        raise ValueError("Checkpoint layer mapping does not match its fc weight")
    block = torch.full(
        (1, args.draft_tokens + 1), draft.mask_token_id,
        dtype=ids.dtype, device=device,
    )
    block[:, 0] = ids[:, -1]
    prefix_length = ids.shape[1]
    positions = torch.cat((
        torch.arange(prefix_length, device=device),
        torch.arange(prefix_length - 1, prefix_length + args.draft_tokens, device=device),
    )).unsqueeze(0)
    # DFlash full attention reads all confirmed context and the entire mask block.
    if draft.uses_sliding_window_attn:
        raise ValueError("This debug example supports full-attention DFlash only")
    attention_mask = torch.zeros(
        (1, 1, block.shape[1], prefix_length + block.shape[1]),
        dtype=torch.bfloat16, device=device,
    )
    past_key_values = DynamicCache()
    for layer in draft.layers:
        layer.self_attn.config._attn_implementation = "eager"
    print("DFLASH target layers (HF hidden_states indices):", draft.target_layer_ids)
    print("target_hidden:", tuple(target_hidden.shape), "block:", block.tolist())
    print("Breakpoint commands: b Qwen3DFlashAttention.forward; c")
    if args.debug:
        pdb.set_trace()
    # Original fusion and decoder modules; no loss or synthetic training labels.
    fc_output = draft.hidden_norm(draft.fc(target_hidden))
    hidden_states = draft.embed_tokens(block)
    position_embeddings = draft.rotary_emb(hidden_states, positions)
    for layer in draft.layers:
        hidden_states = layer(
            hidden_states=hidden_states, target_hidden=fc_output,
            attention_mask=attention_mask, position_embeddings=position_embeddings,
            past_key_value=past_key_values, use_cache=True,
        )
    logits = draft.lm_head(draft.norm(hidden_states))[:, 1:]
    proposals = logits.argmax(-1)
    if draft.use_draft_vocab:
        proposals = draft.d2t[proposals]
    print("draft-owned KV layer 0:", tuple(past_key_values.layers[0].keys.shape))
    show_output(tokenizer, proposals, logits)


def inspect_hspec(args, target, target_output, ids, tokenizer, device):
    drafter, config = load_drafter(args.hspec_checkpoint, target, device)
    context = extract_context(target_output, ids.shape[1], config)
    block = torch.full(
        (1, args.draft_tokens + 1), tokenizer.eos_token_id,
        dtype=ids.dtype, device=device,
    )
    block[:, 0] = ids[:, -1]
    print("HSPEC hidden layers (one-based):", config.target_hidden_layers)
    print("HSPEC KV layers (one-based):", config.target_kv_layers)
    print("last_hidden:", tuple(context.last_hidden.shape), "block:", block.tolist())
    print("prefix KV:", [(tuple(k.shape), tuple(v.shape)) for k, v in context.prefix_kv])
    print("Breakpoint commands: b HSpecReference.forward; b TargetKVAttention.forward; c")
    if args.debug:
        pdb.set_trace()
    proposals, logits = drafter.draft_block(block, context)
    show_output(tokenizer, proposals, logits)


def show_output(tokenizer, proposals, logits):
    print("proposal logits:", tuple(logits.shape))
    print("proposal IDs:", proposals.tolist())
    print("UNVERIFIED draft text:", tokenizer.decode(proposals[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dflash", "hspec"), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--dflash-draft")
    parser.add_argument("--hspec-checkpoint", type=Path)
    parser.add_argument("--conversion-cache", type=Path, default=Path(".debug-dflash-cache"))
    parser.add_argument("--prompt", default="Explain speculative decoding in one sentence.")
    parser.add_argument("--draft-tokens", type=int, default=7)
    parser.add_argument("--debug", action="store_true", help="Enter pdb before the draft forward")
    args = parser.parse_args()
    if args.draft_tokens < 1:
        parser.error("--draft-tokens must be positive")
    if args.mode == "dflash" and not args.dflash_draft:
        parser.error("dflash mode requires --dflash-draft")
    if args.mode == "hspec" and not args.hspec_checkpoint:
        parser.error("hspec mode requires --hspec-checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16,
    ).to(device).eval()
    target.requires_grad_(False)
    ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)
    if ids.shape[1] < 1:
        raise ValueError("Prompt must contain at least one token")
    # Both modes use the same raw prompt and frozen target prefix forward.
    with torch.no_grad():
        target_output = target(ids, use_cache=True, output_hidden_states=True, logits_to_keep=1)
        print("prefix IDs:", ids.tolist(), "prefix length:", ids.shape[1])
        if args.mode == "dflash":
            inspect_dflash(args, target_output, ids, tokenizer, device)
        else:
            inspect_hspec(args, target, target_output, ids, tokenizer, device)


if __name__ == "__main__":
    main()
