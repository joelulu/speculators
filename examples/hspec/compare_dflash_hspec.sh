#!/usr/bin/env bash
# Ordered Qwen3-4B smoke test: pretrained DFlash inference, fresh DFlash
# training + inference, H-Spec reference training + offline greedy inference.
# Run from anywhere. Set DFLASH_DRAFT to a Qwen3-4B DFlash checkpoint.
set -euo pipefail

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

MODEL=${MODEL:-/home/jovyan/LMM/lmm_model/Qwen3-4B}
MODEL_ROOT=${MODEL_ROOT:-/home/jovyan/LMM/lmm_model}
SPEC_PYTHON=${SPEC_PYTHON:-$REPO/.venv-hspec/bin/python}
VLLM_PYTHON=${VLLM_PYTHON:-$SPEC_PYTHON}
GPU=${GPU:-0}
TRAIN_GPU=${TRAIN_GPU:-1}
PORT=${PORT:-18731}
RUN_ROOT=${RUN_ROOT:-$REPO/hspec_comparison_$(date +%Y%m%d_%H%M%S)}
PROMPT=${PROMPT:-'Question: Explain speculative decoding in one sentence. Answer:'}
DFLASH_DRAFT=${DFLASH_DRAFT:-}

die() { echo "ERROR: $*" >&2; exit 1; }
[[ -f "$MODEL/config.json" ]] || die "target missing: $MODEL/config.json"
[[ -x "$SPEC_PYTHON" ]] || die "set SPEC_PYTHON to the Speculators environment Python"
[[ -x "$VLLM_PYTHON" ]] || die "set VLLM_PYTHON to the vLLM environment Python"
[[ "$GPU" != "$TRAIN_GPU" ]] || die "DFlash online training requires two distinct GPUs (GPU and TRAIN_GPU)"
[[ -n "$DFLASH_DRAFT" ]] || {
  echo "Set DFLASH_DRAFT to the matching local Qwen3-4B DFlash checkpoint." >&2
  find "$MODEL_ROOT" -maxdepth 1 -type d -iname '*Qwen3*4B*DFlash*' -print >&2 || true
  exit 1
}
[[ -f "$DFLASH_DRAFT/config.json" ]] || die "DFlash draft missing config.json: $DFLASH_DRAFT"
"$SPEC_PYTHON" -c 'import torch, transformers, speculators; assert torch.cuda.is_available()'
"$VLLM_PYTHON" -c 'import vllm, hs_connectors; print("vllm:", vllm.__version__)' || die "vLLM Python must import vllm and hs_connectors"

mkdir -p "$RUN_ROOT" "$RUN_ROOT/dflash" "$RUN_ROOT/hspec"
RUN_ROOT=$(cd "$RUN_ROOT" && pwd)
printf 'MODEL=%s\nDFLASH_DRAFT=%s\nSPEC_PYTHON=%s\nVLLM_PYTHON=%s\nGPU=%s TRAIN_GPU=%s\n' \
  "$MODEL" "$DFLASH_DRAFT" "$SPEC_PYTHON" "$VLLM_PYTHON" "$GPU" "$TRAIN_GPU" | tee "$RUN_ROOT/settings.txt"
git rev-parse HEAD >> "$RUN_ROOT/settings.txt"

SERVER_PID=
stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
  fi
}
trap stop_server EXIT INT TERM

start_server() {
  local log=$1; shift
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    die "port $PORT already serves another process; choose another PORT"
  fi
  setsid env CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_PYTHON" scripts/launch_vllm.py "$@" \
    >"$log" 2>&1 &
  SERVER_PID=$!
  local tries
  for ((tries=0; tries<240; tries++)); do
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "vLLM ready (pid=$SERVER_PID, log=$log)"
      return
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -60 "$log" >&2
      die "vLLM exited; see $log"
    fi
    sleep 2
  done
  tail -60 "$log" >&2
  die "vLLM startup timed out; see $log"
}

infer_dflash() {
  local draft=$1 name=$2
  echo "=== DFlash inference: $name ==="
  start_server "$RUN_ROOT/dflash/${name}_server.log" \
    eval "$MODEL" --spec-model "$draft" --spec-method dflash --spec-tokens 7 \
    --provenance-dir "$RUN_ROOT/dflash/${name}_provenance" \
    --no-hash-checkpoints -- \
    --port "$PORT" --served-model-name hspec-target --max-model-len 512 \
    --gpu-memory-utilization 0.75 --enforce-eager
  PROMPT="$PROMPT" "$SPEC_PYTHON" - "$PORT" >"$RUN_ROOT/dflash/${name}_infer.txt" <<'PY'
import json, os, sys, urllib.request
port = int(sys.argv[1])
payload = json.dumps({"model": "hspec-target", "prompt": os.environ["PROMPT"],
                      "temperature": 0, "max_tokens": 32}).encode()
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions", data=payload,
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=180) as response:
    result = json.load(response)
print("text:", result["choices"][0]["text"])
print("usage:", result.get("usage"))
PY
  cat "$RUN_ROOT/dflash/${name}_infer.txt"
  curl -fsS "http://127.0.0.1:$PORT/metrics" \
    | grep -Ei 'spec_decode|draft_token|accepted_token' \
    >"$RUN_ROOT/dflash/${name}_metrics.txt" || true
  stop_server
}

# 1. Run the existing DFlash checkpoint before touching either training path.
infer_dflash "$DFLASH_DRAFT" pretrained

# 2. Make a tiny, target-generated dataset on GPU 0. The JSONL records the
# exact token IDs and supervised continuation; the text is shared with H-Spec.
echo '=== Generate on-policy smoke data ==='
MODEL="$MODEL" RUN_ROOT="$RUN_ROOT" CUDA_VISIBLE_DEVICES="$GPU" \
  "$SPEC_PYTHON" - >"$RUN_ROOT/data_generation.log" 2>&1 <<'PY'
import json, os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

root = Path(os.environ["RUN_ROOT"])
model_name = os.environ["MODEL"]
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).cuda().eval()
prompts = [
    "Question: Explain speculative decoding in one sentence. Answer:",
    "Question: Why must a target model verify draft tokens? Answer:",
    "Question: What happens when a speculative draft token is rejected? Answer:",
    "Question: Describe the difference between drafting and verification. Answer:",
    "Question: Why can speculative decoding preserve greedy output? Answer:",
    "Question: What does the acceptance length measure? Answer:",
    "Question: What is the purpose of a draft model? Answer:",
    "Question: Explain target model KV cache reuse. Answer:",
]
with (root / "dflash" / "on_policy.jsonl").open("w") as data, \
     (root / "hspec" / "texts.txt").open("w") as texts:
    for prompt in prompts:
        prefix = tok(prompt, return_tensors="pt")["input_ids"].cuda()
        with torch.inference_mode():
            full = model.generate(prefix, max_new_tokens=48, do_sample=False,
                                  pad_token_id=tok.eos_token_id)[0]
        ids = full.tolist()
        mask = [0] * prefix.shape[1] + [1] * (len(ids) - prefix.shape[1])
        if sum(mask) < 8:
            raise RuntimeError("generated continuation is shorter than 8 tokens")
        data.write(json.dumps({"input_ids": ids, "loss_mask": mask}) + "\n")
        texts.write(tok.decode(full, skip_special_tokens=True).replace("\n", " ") + "\n")
print("wrote 8 target-generated examples")
PY
tail -3 "$RUN_ROOT/data_generation.log"

# 3. Train the repository's original DFlash implementation. This uses its own
# Speculators trainer and a vLLM target server for hidden-state extraction.
echo '=== DFlash train: prepare data ==='
"$SPEC_PYTHON" -m speculators prepare-data --model "$MODEL" \
  --data "$RUN_ROOT/dflash/on_policy.jsonl" \
  --output "$RUN_ROOT/dflash/data" --max-samples 8 --seq-length 256 \
  >"$RUN_ROOT/dflash/prepare.log" 2>&1
echo '=== DFlash train: start target extractor ==='
start_server "$RUN_ROOT/dflash/extractor.log" train "$MODEL" \
  --target-layer-ids 2 18 33 \
  --hidden-states-path "$RUN_ROOT/dflash/hidden_states" \
  --provenance-dir "$RUN_ROOT/dflash/extractor_provenance" \
  --no-hash-checkpoints -- \
  --port "$PORT" --max-model-len 512 --gpu-memory-utilization 0.75 --enforce-eager
echo '=== DFlash train: 2 optimizer steps ==='
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$SPEC_PYTHON" -m speculators.train \
  --verifier-name-or-path "$MODEL" \
  --data-path "$RUN_ROOT/dflash/data" \
  --save-path "$RUN_ROOT/dflash/checkpoints" \
  --vllm-endpoint "http://127.0.0.1:$PORT/v1" \
  --hidden-states-path "$RUN_ROOT/dflash/hidden_states" \
  --speculator-type dflash --num-layers 5 --block-size 8 \
  --target-layer-ids 2 18 33 --max-anchors 32 \
  --total-seq-len 256 --epochs 1 --max-steps 2 \
  --num-workers 1 --optimizer adamw --lr 3e-4 \
  --on-missing generate --on-generate delete --log-freq 1 \
  >"$RUN_ROOT/dflash/train.log" 2>&1 || {
    tail -80 "$RUN_ROOT/dflash/train.log" >&2
    die "DFlash training failed; see $RUN_ROOT/dflash/train.log"
  }
stop_server
[[ -f "$RUN_ROOT/dflash/checkpoints/0/model.safetensors" ]] || \
  die "DFlash training ended without checkpoint; inspect train.log"
infer_dflash "$RUN_ROOT/dflash/checkpoints/0" trained_2step

# 4. Train and infer with the H-Spec correctness reference on GPU 0.
echo '=== H-Spec reference train: 2 optimizer steps ==='
CUDA_VISIBLE_DEVICES="$GPU" "$SPEC_PYTHON" examples/hspec/reference_train.py \
  --target "$MODEL" --texts "$RUN_ROOT/hspec/texts.txt" \
  --steps 2 --max-prefix 128 --draft-tokens 7 \
  --out "$RUN_ROOT/hspec/hspec_smoke.pt" \
  2>&1 | tee "$RUN_ROOT/hspec/train.log"
echo '=== H-Spec offline greedy inference ==='
CUDA_VISIBLE_DEVICES="$GPU" "$SPEC_PYTHON" examples/hspec/reference_infer.py \
  --target "$MODEL" --checkpoint "$RUN_ROOT/hspec/hspec_smoke.pt" \
  --prompt "$PROMPT" --max-new-tokens 32 \
  2>&1 | tee "$RUN_ROOT/hspec/infer.log"
echo "Done. Logs and checkpoints: $RUN_ROOT"
