#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
MODE=${1:?Usage: bash examples/hspec/debug_forward.sh dflash|hspec}
PYTHON=${SPEC_PYTHON:-$ROOT/.venv-hspec/bin/python}
MODEL=${MODEL:-/home/jovyan/LMM/lmm_model/Qwen3-4B}
PROMPT=${PROMPT:-Explain speculative decoding in one sentence.}
COMMON=(--mode "$MODE" --target "$MODEL" --prompt "$PROMPT" --draft-tokens 7)
case "$MODE" in
  dflash)
    EXTRA=(--dflash-draft "${DFLASH_DRAFT:-/home/jovyan/LMM/lmm_model/Qwen3-4B-DFlash-b16}") ;;
  hspec)
    EXTRA=(--hspec-checkpoint "${HSPEC_CHECKPOINT:-$ROOT/hspec_smoke.pt}") ;;
  *) echo "Mode must be dflash or hspec" >&2; exit 2 ;;
esac
if [[ ${DEBUG:-1} == 1 ]]; then COMMON+=(--debug); fi
export CUDA_VISIBLE_DEVICES=${GPU:-0}
exec "$PYTHON" examples/hspec/debug_forward.py "${COMMON[@]}" "${EXTRA[@]}"
