#!/usr/bin/env bash

set -euo pipefail

# Run from repo root (resolve relative to this script).
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Override these via: sbatch --export=ALL,CONTEXT_LENGTH=256,MEASURE_STEPS=20 ...
CONTEXT_LENGTH="${CONTEXT_LENGTH:-128}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
MEASURE_STEPS="${MEASURE_STEPS:-10}"
MODE="${MODE:-forward-backward}"
DEVICE="${DEVICE:-cuda}"
MIXED_PRECISION="${MIXED_PRECISION:-0}"   # 1 to enable --mixed-precision
MP_DTYPE="${MP_DTYPE:-bf16}"              # bf16 or fp16

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="outputs/benchmark/${TIMESTAMP}"
mkdir -p "${OUTDIR}"

MODEL_SIZES=(small medium large xl 2.7b)

echo "Output directory: ${OUTDIR}"
echo "mode=${MODE} device=${DEVICE} context=${CONTEXT_LENGTH} batch=${BATCH_SIZE} warmup=${WARMUP_STEPS} measure=${MEASURE_STEPS}"

failed_models=()

for size in "${MODEL_SIZES[@]}"; do
  echo "=============================="
  echo "Running model size: ${size}"
  echo "=============================="

  cmd=(
    uv run python benchmark.py
    --device "${DEVICE}"
    --mode "${MODE}"
    --model-size "${size}"
    --context-length "${CONTEXT_LENGTH}"
    --batch-size "${BATCH_SIZE}"
    --warmup-steps "${WARMUP_STEPS}"
    --measure-steps "${MEASURE_STEPS}"
    --json-output "${OUTDIR}/${size}.json"
  )

  if [[ "${MIXED_PRECISION}" == "1" ]]; then
    cmd+=(--mixed-precision --mp-dtype "${MP_DTYPE}")
  fi

  set +e
  "${cmd[@]}"
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "FAILED: ${size} (exit code ${rc})"
    failed_models+=("${size}")
  else
    echo "DONE: ${size} -> ${OUTDIR}/${size}.json"
  fi
done

echo "=============================="
if [[ ${#failed_models[@]} -eq 0 ]]; then
  echo "All model sizes finished successfully."
else
  echo "Some model sizes failed: ${failed_models[*]}"
  echo "This can happen due to OOM on larger models/contexts."
fi
echo "Results are in: ${OUTDIR}"
