#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?usage: start_vllm_server.sh <model_path> <port> [served_model_name] [parser_path]}"
PORT="${2:?usage: start_vllm_server.sh <model_path> <port> [served_model_name] [parser_path]}"
SERVED_MODEL_NAME="${3:-$(basename "$MODEL_PATH")}"
PARSER_PATH="${4:-}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

CMD=(
  vllm serve "$MODEL_PATH"
  --host 127.0.0.1
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype auto
  --max-model-len 32768
)

if [[ -n "$PARSER_PATH" ]]; then
  CMD+=(
    --enable-auto-tool-choice
    --tool-parser-plugin "$PARSER_PATH"
    --tool-call-parser xlam
  )
fi

printf 'Starting vLLM server on port %s for %s\n' "$PORT" "$MODEL_PATH"
exec "${CMD[@]}"
