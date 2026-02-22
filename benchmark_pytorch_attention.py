#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from cs336_basics.model import scaled_dot_product_attention


BATCH_SIZE = 8
DEFAULT_D_MODEL_VALUES = [16, 32, 64, 128]
DEFAULT_SEQUENCE_LENGTH_VALUES = [256, 1024, 4096, 8192, 16384]
AttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


@dataclass
class BenchmarkResult:
    d_model: int
    sequence_length: int
    status: str
    forward_mean_ms: float | None
    forward_std_ms: float | None
    backward_mean_ms: float | None
    backward_std_ms: float | None
    memory_before_backward_mib: float | None
    peak_memory_mib: float | None
    error: str | None = None


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def sync_cuda_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def maybe_build_causal_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=device))


def timed_forward_passes(
    forward_attention_fn: AttentionFn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    warmup_iters: int,
    measure_iters: int,
    device: torch.device,
) -> list[float]:
    for _ in range(warmup_iters):
        with torch.no_grad():
            _ = forward_attention_fn(q, k, v, mask)
        sync_cuda_if_needed(device)

    times_ms: list[float] = []
    for _ in range(measure_iters):
        start = time.perf_counter()
        with torch.no_grad():
            _ = forward_attention_fn(q, k, v, mask)
        sync_cuda_if_needed(device)
        times_ms.append((time.perf_counter() - start) * 1_000)
    return times_ms


def timed_backward_passes(
    backward_attention_fn: AttentionFn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    warmup_iters: int,
    measure_iters: int,
    device: torch.device,
) -> tuple[list[float], float | None]:
    for _ in range(warmup_iters):
        q.grad = None
        k.grad = None
        v.grad = None
        output = backward_attention_fn(q, k, v, mask)
        output.sum().backward()
        sync_cuda_if_needed(device)

    times_ms: list[float] = []
    memory_before_backward_mib: float | None = None

    for step in range(measure_iters):
        q.grad = None
        k.grad = None
        v.grad = None

        output = backward_attention_fn(q, k, v, mask)
        loss = output.sum()

        if step == 0 and device.type == "cuda":
            sync_cuda_if_needed(device)
            memory_before_backward_mib = torch.cuda.memory_allocated(device=device) / (1024**2)

        start = time.perf_counter()
        loss.backward()
        sync_cuda_if_needed(device)
        times_ms.append((time.perf_counter() - start) * 1_000)

    return times_ms, memory_before_backward_mib


def summarize(times_ms: list[float]) -> tuple[float, float]:
    if len(times_ms) == 1:
        return times_ms[0], 0.0
    return statistics.fmean(times_ms), statistics.pstdev(times_ms)


def benchmark_one_configuration(
    forward_attention_fn: AttentionFn,
    backward_attention_fn: AttentionFn,
    d_model: int,
    sequence_length: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup_iters: int,
    measure_iters: int,
    use_causal_mask: bool,
) -> BenchmarkResult:
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device=device)
            sync_cuda_if_needed(device)

        q = torch.randn(
            (BATCH_SIZE, sequence_length, d_model),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        k = torch.randn(
            (BATCH_SIZE, sequence_length, d_model),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        v = torch.randn(
            (BATCH_SIZE, sequence_length, d_model),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        mask = maybe_build_causal_mask(sequence_length=sequence_length, device=device) if use_causal_mask else None

        forward_times_ms = timed_forward_passes(
            forward_attention_fn=forward_attention_fn,
            q=q,
            k=k,
            v=v,
            mask=mask,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            device=device,
        )
        backward_times_ms, memory_before_backward_mib = timed_backward_passes(
            backward_attention_fn=backward_attention_fn,
            q=q,
            k=k,
            v=v,
            mask=mask,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            device=device,
        )

        forward_mean_ms, forward_std_ms = summarize(forward_times_ms)
        backward_mean_ms, backward_std_ms = summarize(backward_times_ms)

        peak_memory_mib = None
        if device.type == "cuda":
            peak_memory_mib = torch.cuda.max_memory_allocated(device=device) / (1024**2)

        return BenchmarkResult(
            d_model=d_model,
            sequence_length=sequence_length,
            status="ok",
            forward_mean_ms=forward_mean_ms,
            forward_std_ms=forward_std_ms,
            backward_mean_ms=backward_mean_ms,
            backward_std_ms=backward_std_ms,
            memory_before_backward_mib=memory_before_backward_mib,
            peak_memory_mib=peak_memory_mib,
        )
    except RuntimeError as error:
        error_text = str(error)
        if "out of memory" in error_text.lower():
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return BenchmarkResult(
                d_model=d_model,
                sequence_length=sequence_length,
                status="oom",
                forward_mean_ms=None,
                forward_std_ms=None,
                backward_mean_ms=None,
                backward_std_ms=None,
                memory_before_backward_mib=None,
                peak_memory_mib=None,
                error=error_text.splitlines()[0],
            )
        raise


def print_results_table(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'d_model':>7} {'seq_len':>8} {'status':>7} {'fwd(ms)':>14} "
        f"{'bwd(ms)':>14} {'mem_before_bwd(MiB)':>20} {'peak_mem(MiB)':>14}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        forward = (
            f"{result.forward_mean_ms:.3f} +/- {result.forward_std_ms:.3f}"
            if result.forward_mean_ms is not None and result.forward_std_ms is not None
            else "-"
        )
        backward = (
            f"{result.backward_mean_ms:.3f} +/- {result.backward_std_ms:.3f}"
            if result.backward_mean_ms is not None and result.backward_std_ms is not None
            else "-"
        )
        memory_before_bwd = f"{result.memory_before_backward_mib:.2f}" if result.memory_before_backward_mib is not None else "-"
        peak_mem = f"{result.peak_memory_mib:.2f}" if result.peak_memory_mib is not None else "-"
        print(
            f"{result.d_model:7d} {result.sequence_length:8d} {result.status:>7} "
            f"{forward:>14} {backward:>14} {memory_before_bwd:>20} {peak_mem:>14}"
        )
        if result.error:
            print(f"  error: {result.error}")


def write_json(path: Path, results: list[BenchmarkResult], args: argparse.Namespace) -> None:
    payload = {
        "batch_size": BATCH_SIZE,
        "device": args.device,
        "dtype": args.dtype,
        "compile": args.compile,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "use_causal_mask": args.use_causal_mask,
        "results": [asdict(result) for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark cs336_basics.model.scaled_dot_product_attention across sequence lengths and d_model values."
        )
    )
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--measure-iters", type=int, default=100)
    parser.add_argument("--d-model-values", type=str, default="16,32,64,128")
    parser.add_argument("--sequence-length-values", type=str, default="256,1024,4096,8192,16384")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-causal-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser.parse_args()


def eager_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    return scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    d_model_values = parse_csv_ints(args.d_model_values)
    sequence_length_values = parse_csv_ints(args.sequence_length_values)

    print("PyTorch attention benchmark")
    print(f"  batch_size={BATCH_SIZE}")
    print(f"  device={device}")
    print(f"  dtype={dtype}")
    print(f"  compile={args.compile}")
    print(f"  warmup_iters={args.warmup_iters}")
    print(f"  measure_iters={args.measure_iters}")
    print(f"  use_causal_mask={args.use_causal_mask}")
    print(f"  d_model_values={d_model_values}")
    print(f"  sequence_length_values={sequence_length_values}")
    print()

    results: list[BenchmarkResult] = []
    total = len(d_model_values) * len(sequence_length_values)
    count = 0
    for d_model in d_model_values:
        for sequence_length in sequence_length_values:
            forward_attention_fn = eager_attention
            backward_attention_fn = eager_attention
            if args.compile:
                # Compile per configuration so each compiled graph only sees one shape.
                # This avoids hitting global recompile limits across the whole grid sweep.
                forward_attention_fn = torch.compile(eager_attention, dynamic=True)
                backward_attention_fn = torch.compile(eager_attention, dynamic=True)
            count += 1
            print(f"[{count}/{total}] d_model={d_model}, sequence_length={sequence_length}")
            result = benchmark_one_configuration(
                forward_attention_fn=forward_attention_fn,
                backward_attention_fn=backward_attention_fn,
                d_model=d_model,
                sequence_length=sequence_length,
                dtype=dtype,
                device=device,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
                use_causal_mask=args.use_causal_mask,
            )
            results.append(result)
            if result.status == "ok":
                print(
                    f"  forward={result.forward_mean_ms:.3f} ms, "
                    f"backward={result.backward_mean_ms:.3f} ms"
                )
            else:
                print(f"  status={result.status}: {result.error}")

    print()
    print_results_table(results)

    if args.json_output is not None:
        write_json(args.json_output, results, args)
        print(f"\nWrote JSON results to {args.json_output}")
    if args.csv_output is not None:
        write_csv(args.csv_output, results)
        print(f"Wrote CSV results to {args.csv_output}")


if __name__ == "__main__":
    main()
