#!/usr/bin/env bash

set -euo pipefail

# Run from repo root (resolve relative to this script).
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Override via environment when needed.
CONTEXT_LENGTH="${CONTEXT_LENGTH:-128}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
MEASURE_STEPS="${MEASURE_STEPS:-10}"
MODE="${MODE:-forward-backward}"
DEVICE="${DEVICE:-cuda}"
MP_DTYPE="${MP_DTYPE:-bf16}"    # bf16 or fp16

MODEL_SIZES=(small medium large xl 2.7b)
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BASE_OUTDIR="outputs/benchmark_mixed_precision/${TIMESTAMP}"
FP32_OUTDIR="${BASE_OUTDIR}/fp32"
MP_OUTDIR="${BASE_OUTDIR}/mixed_${MP_DTYPE}"
mkdir -p "${FP32_OUTDIR}" "${MP_OUTDIR}"

echo "Output directory: ${BASE_OUTDIR}"
echo "mode=${MODE} device=${DEVICE} context=${CONTEXT_LENGTH} batch=${BATCH_SIZE} warmup=${WARMUP_STEPS} measure=${MEASURE_STEPS}"

declare -a failed_fp32=()
declare -a failed_mp=()

run_for_precision() {
  local mixed_flag="$1"      # 0 or 1
  local outdir="$2"
  local label="$3"

  for size in "${MODEL_SIZES[@]}"; do
    echo "=============================="
    echo "Running ${label} | model size: ${size}"
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
      --json-output "${outdir}/${size}.json"
    )

    if [[ "${mixed_flag}" == "1" ]]; then
      cmd+=(--mixed-precision --mp-dtype "${MP_DTYPE}")
    fi

    set +e
    "${cmd[@]}"
    rc=$?
    set -e

    if [[ $rc -ne 0 ]]; then
      echo "FAILED: ${label} ${size} (exit code ${rc})"
      if [[ "${mixed_flag}" == "1" ]]; then
        failed_mp+=("${size}")
      else
        failed_fp32+=("${size}")
      fi
    else
      echo "DONE: ${label} ${size} -> ${outdir}/${size}.json"
    fi
  done
}

run_for_precision 0 "${FP32_OUTDIR}" "FP32"
run_for_precision 1 "${MP_OUTDIR}" "MIXED-${MP_DTYPE}"

python - "${FP32_OUTDIR}" "${MP_OUTDIR}" <<'PY'
import json
import sys
from pathlib import Path

fp32_dir = Path(sys.argv[1])
mp_dir = Path(sys.argv[2])
model_sizes = ["small", "medium", "large", "xl", "2.7b"]

def read(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())

def fmt(v):
    return "NA" if v is None else f"{v*1000:.2f}"

def speedup(base, new):
    if base is None or new is None or new == 0:
        return None
    return base / new

print("\n=== Mixed Precision Summary (ms; speedup = FP32 / Mixed) ===")
print(f"{'model':<8} {'fwd_fp32':>11} {'fwd_mp':>11} {'fwd_x':>8} {'bwd_fp32':>11} {'bwd_mp':>11} {'bwd_x':>8}")

for size in model_sizes:
    a = read(fp32_dir / f"{size}.json")
    b = read(mp_dir / f"{size}.json")
    if a is None or b is None:
        print(f"{size:<8} {'NA':>11} {'NA':>11} {'NA':>8} {'NA':>11} {'NA':>11} {'NA':>8}")
        continue

    sa = a.get("summary", {})
    sb = b.get("summary", {})

    fwd_a = sa.get("forward_mean_s")
    fwd_b = sb.get("forward_mean_s")
    bwd_a = sa.get("backward_mean_s")
    bwd_b = sb.get("backward_mean_s")

    fwd_x = speedup(fwd_a, fwd_b)
    bwd_x = speedup(bwd_a, bwd_b)

    fwd_x_s = "NA" if fwd_x is None else f"{fwd_x:.2f}x"
    bwd_x_s = "NA" if bwd_x is None else f"{bwd_x:.2f}x"

    print(
        f"{size:<8} {fmt(fwd_a):>11} {fmt(fwd_b):>11} {fwd_x_s:>8} "
        f"{fmt(bwd_a):>11} {fmt(bwd_b):>11} {bwd_x_s:>8}"
    )
PY

echo
echo "FP32 JSONs : ${FP32_OUTDIR}"
echo "Mixed JSONs: ${MP_OUTDIR}"

echo "failed_fp32: ${failed_fp32[*]:-none}"
echo "failed_mp  : ${failed_mp[*]:-none}"
