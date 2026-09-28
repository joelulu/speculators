# H-Spec reproduction: reference stage

Source: Jiang et al., [H-Spec](https://arxiv.org/abs/2609.24197), v1. This
branch implements a **correctness-oriented PyTorch reference**, not an
end-to-end reproduction of the paper's throughput or acceptance results.

## Implemented

- Qwen3 4B/8B layer mapping: final-position hidden from layers 1, 9, 17, 25,
  34; direct post-RoPE target K/V from layers 17, 25, 34. Paper layer numbers
  are one-based; `DynamicCache.layers` indexes are zero-based.
- One shared fusion/state projection produces a `[B, 44|48, 64, 4]` initial
  state; four selective state-space blocks run with that same initial state.
  The first three contain attention against the target KV and current block;
  the fourth has no attention. Attention is causal, position aligned, and
  limited to a 2,048-token window. Prefix K/V are borrowed tensors, not a
  persistent drafter-owned cache.
- Block state evolution uses a logarithmic-depth associative scan with an
  explicit, differentiable initial state. Tests compare its outputs and
  gradients with the serial recurrence for different block lengths. This
  PyTorch alignment path launches multiple operations; it is not a fused
  Mamba-2/SSD inference kernel or a throughput result.
- DSpark-style low-rank Markov logit bias conditioned on the previous **known**
  token during teacher-forced training. The reference includes a one-block
  training path with frozen target output and 0.1 CE + 0.9 TV loss.

## Run a small alignment/training check

Install this repository and its dependencies in a GPU environment, then:

```bash
pytest -q tests/unit/models/test_hspec_reference.py
python examples/hspec/reference_train.py \
  --target Qwen/Qwen3-8B \
  --texts /path/to/target_generated_texts.txt \
  --steps 10 --out /path/to/hspec_reference.pt
```

The text file contains one preprocessed sequence per line. The reference
trains one contiguous block from each line; it does not implement 8,192-token
multipacking or the paper's 100K samples and five epochs. `reference_train.py`
records the git SHA and run arguments next to the checkpoint. Save the
prepared texts and target revision for a comparable run.

## Required before claiming a full H-Spec reproduction

1. Replace the PyTorch associative scan in `reference.py` with an optimized
   initial-state-aware **Mamba-2/SSD kernel**. Check its outputs and gradients
   against the scan and serial recurrence. Check whether the paper retains
   convolution state across blocks; this reference starts block convolution
   with zero history.
2. Use the existing packed Speculators trainer with the selected target K/V
   and the *last hidden at each chosen anchor*. Current DFlash data files
   contain hidden states but do not supply target K/V to the drafter. Do not
   use a full-sequence last hidden for every anchor or include future K/V.
3. Match the published training corpus, packed sampling, target-generated
   responses, 32K vocabulary pruning, target LM-head weights, optimizer
   schedule and 0.1 CE + 0.9 TV loss with position weighting.
4. Extend vLLM's speculative drafter/target interface so the three H-Spec
   attention layers read the verifier's **paged** KV blocks in place. The
   Qwen3 query layout, RoPE positions, TP shards, window, request block table
   and verifier KV lifetime must all match. Passing dense target K/V to the
   Python reference materializes temporary windows and is **not** this
   service integration. The verified target must still perform lossless
   speculative acceptance; do not treat draft logits alone as final output.
5. Compare matched DFlash, DSpark and H-Spec checkpoints on mean accepted
   length and first rejection. Profile drafter/verify latency, separately
   allocated target and draft KV, and output throughput at concurrency
   1/8/32/64/128 and context lengths up to 32K. Record model and framework
   revisions and pass `--provenance-dir` for `scripts/launch_vllm.py` runs.

Only after steps 1–5 are executed on a GPU can this branch be used to assess
whether the paper's claimed performance carries over to the chosen vLLM build.
