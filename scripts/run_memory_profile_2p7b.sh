#!/usr/bin/env bash

set -euo pipefail

# Run from repo root.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Tunables (override via env).
MODEL_SIZE="${MODEL_SIZE:-2.7b}"
CONTEXTS="${CONTEXTS:-128 256 512}"
MODES="${MODES:-forward train-step}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
MEASURE_STEPS="${MEASURE_STEPS:-1}"
MEMORY_MAX_ENTRIES="${MEMORY_MAX_ENTRIES:-1000000}"

RUN_FP32="${RUN_FP32:-1}"
RUN_BF16="${RUN_BF16:-1}"

# If OUTDIR is not provided, create a timestamped directory.
OUTDIR="${OUTDIR:-}"
if [[ -z "${OUTDIR}" ]]; then
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  BASE_OUTDIR="outputs/memory_profiles/${TIMESTAMP}"
else
  BASE_OUTDIR="${OUTDIR}"
fi
mkdir -p "${BASE_OUTDIR}"

echo "Output directory: ${BASE_OUTDIR}"
echo "model=${MODEL_SIZE} device=${DEVICE} batch=${BATCH_SIZE} warmup=${WARMUP_STEPS} measure=${MEASURE_STEPS}"
echo "contexts=${CONTEXTS} modes=${MODES} run_fp32=${RUN_FP32} run_bf16=${RUN_BF16}"

failed=()

run_one() {
  local precision="$1"  # fp32 | bf16
  local ctx="$2"
  local mode="$3"

  local outdir="${BASE_OUTDIR}/${precision}"
  mkdir -p "${outdir}"

  local base="ctx${ctx}_${mode}"

  cmd=(
    uv run python benchmark.py
    --device "${DEVICE}"
    --model-size "${MODEL_SIZE}"
    --batch-size "${BATCH_SIZE}"
    --context-length "${ctx}"
    --warmup-steps "${WARMUP_STEPS}"
    --measure-steps "${MEASURE_STEPS}"
    --mode "${mode}"
    --memory-max-entries "${MEMORY_MAX_ENTRIES}"
    --memory-profile "${outdir}/${base}.pickle"
    --json-output "${outdir}/${base}.json"
  )

  if [[ "${precision}" == "bf16" ]]; then
    cmd+=(--mixed-precision --mp-dtype bf16)
  fi

  echo "=============================="
  echo "Running ${precision} | ctx=${ctx} | mode=${mode}"
  echo "=============================="

  set +e
  "${cmd[@]}"
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "FAILED: ${precision} ctx=${ctx} mode=${mode} (rc=${rc})"
    failed+=("${precision}:ctx${ctx}:${mode}")
  else
    echo "DONE: ${precision} ctx=${ctx} mode=${mode}"
  fi
}

if [[ "${RUN_FP32}" == "1" ]]; then
  for ctx in ${CONTEXTS}; do
    for mode in ${MODES}; do
      run_one fp32 "${ctx}" "${mode}"
    done
  done
fi

if [[ "${RUN_BF16}" == "1" ]]; then
  for ctx in ${CONTEXTS}; do
    for mode in ${MODES}; do
      run_one bf16 "${ctx}" "${mode}"
    done
  done
fi

echo "=============================="
if [[ ${#failed[@]} -eq 0 ]]; then
  echo "All memory profiling runs finished successfully."
else
  echo "Some runs failed: ${failed[*]}"
fi
echo "Results are in: ${BASE_OUTDIR}"

# Generate a markdown + terminal summary from the produced pickles.
echo
echo "Generating peak-memory summary..."
uv run python scripts/summarize_memory_profiles.py \
  --base-dir "${BASE_OUTDIR}" \
  --markdown-output "${BASE_OUTDIR}/summary.md"
echo "Summary markdown: ${BASE_OUTDIR}/summary.md"
