#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

QWEN_MODEL="../autodl-tmp/Qwen2.5-3B-Instruct"
XLAM_MODEL="../autodl-tmp/xLAM-2-3b-fc-r"
XLAM_PARSER="../autodl-tmp/xLAM-2-3b-fc-r/xlam_tool_call_parser.py"

WHEN2CALL_TEST="data/processed/when2call/when2call_test_mcq.jsonl"
BFCL_JSON="data/processed/bfcl/bfcl_v3_strict.json"

QWEN_PORT=8000
XLAM_PORT=8001

mkdir -p eval_results/baseline logs

cleanup() {
  pkill -f "vllm serve $QWEN_MODEL" >/dev/null 2>&1 || true
  pkill -f "vllm serve $XLAM_MODEL" >/dev/null 2>&1 || true
}

wait_for_server() {
  local url="$1"
  local name="$2"
  for _ in $(seq 1 180); do
    if curl -sf "$url" >/dev/null; then
      echo "$name is healthy at $url"
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for $name at $url" >&2
  return 1
}

trap cleanup EXIT

echo "Preparing datasets..."
# python scripts/data/prepare_when2call.py
# python scripts/data/download_bfcl.py --source modelscope_parquet --repo-id AI-ModelScope/bfcl_v3
# python scripts/data/convert_bfcl_parquet.py

echo "Starting Qwen vLLM server..."
nohup bash scripts/inference/start_vllm_server.sh \
  "$QWEN_MODEL" \
  "$QWEN_PORT" \
  Qwen2.5-3B-Instruct \
  > logs/qwen_vllm_full.log 2>&1 &

wait_for_server "http://127.0.0.1:${QWEN_PORT}/health" "Qwen vLLM server"

echo "Running Qwen + When2Call..."
python scripts/inference/run_baseline.py \
  --backend vllm \
  --api-base "http://127.0.0.1:${QWEN_PORT}/v1" \
  --served-model-name Qwen2.5-3B-Instruct \
  --dataset when2call \
  --dataset-path "$WHEN2CALL_TEST" \
  --model-path "$QWEN_MODEL" \
  --output-path eval_results/baseline/qwen_when2call_full_vllm.json

echo "Scoring Qwen + When2Call..."
python scripts/eval/score_predictions.py \
  --dataset when2call \
  --predictions-path eval_results/baseline/qwen_when2call_full_vllm.json \
  | tee eval_results/baseline/qwen_when2call_full_vllm.score.json

# echo "Running Qwen + BFCL..."
# python scripts/inference/run_baseline.py \
#   --backend vllm \
#   --api-base "http://127.0.0.1:${QWEN_PORT}/v1" \
#   --served-model-name Qwen2.5-3B-Instruct \
#   --dataset bfcl \
#   --dataset-path "$BFCL_JSON" \
#   --model-path "$QWEN_MODEL" \
#   --output-path eval_results/baseline/qwen_bfcl_full_vllm.json

# echo "Scoring Qwen + BFCL..."
# python scripts/eval/score_predictions.py \
#   --dataset bfcl \
#   --predictions-path eval_results/baseline/qwen_bfcl_full_vllm.json \
#   | tee eval_results/baseline/qwen_bfcl_full_vllm.score.json

echo "Stopping Qwen vLLM server..."
pkill -f "vllm serve $QWEN_MODEL" || true
sleep 5

echo "Starting xLAM vLLM server..."
nohup bash scripts/inference/start_vllm_server.sh \
  "$XLAM_MODEL" \
  "$XLAM_PORT" \
  xLAM-2-3b-fc-r \
  "$XLAM_PARSER" \
  > logs/xlam_vllm_full.log 2>&1 &

wait_for_server "http://127.0.0.1:${XLAM_PORT}/health" "xLAM vLLM server"

echo "Running xLAM + When2Call..."
python scripts/inference/run_baseline.py \
  --backend vllm \
  --api-base "http://127.0.0.1:${XLAM_PORT}/v1" \
  --served-model-name xLAM-2-3b-fc-r \
  --dataset when2call \
  --dataset-path "$WHEN2CALL_TEST" \
  --model-path "$XLAM_MODEL" \
  --output-path eval_results/baseline/xlam_when2call_full_vllm.json

echo "Scoring xLAM + When2Call..."
python scripts/eval/score_predictions.py \
  --dataset when2call \
  --predictions-path eval_results/baseline/xlam_when2call_full_vllm.json \
  | tee eval_results/baseline/xlam_when2call_full_vllm.score.json

# echo "Running xLAM + BFCL..."
# python scripts/inference/run_baseline.py \
#   --backend vllm \
#   --api-base "http://127.0.0.1:${XLAM_PORT}/v1" \
#   --served-model-name xLAM-2-3b-fc-r \
#   --dataset bfcl \
#   --dataset-path "$BFCL_JSON" \
#   --model-path "$XLAM_MODEL" \
#   --output-path eval_results/baseline/xlam_bfcl_full_vllm.json

# echo "Scoring xLAM + BFCL..."
# python scripts/eval/score_predictions.py \
#   --dataset bfcl \
#   --predictions-path eval_results/baseline/xlam_bfcl_full_vllm.json \
#   | tee eval_results/baseline/xlam_bfcl_full_vllm.score.json

echo "Full baseline run completed."
