from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import triton

from cs336_systems.flashattention2triton import FlashAttention2Triton


NEG_INF_MASK_VALUE = -1e6


@dataclass
class TimingResult:
    forward_ms: float
    backward_ms: float
    end_to_end_ms: float


@dataclass
class BenchmarkRow:
    dtype: str
    sequence_length: int
    d_model: int
    torch_status: str
    flash_status: str
    torch_forward_ms: float | None
    flash_forward_ms: float | None
    forward_speedup: float | None
    torch_backward_ms: float | None
    flash_backward_ms: float | None
    backward_speedup: float | None
    torch_end_to_end_ms: float | None
    flash_end_to_end_ms: float | None
    end_to_end_speedup: float | None
    torch_error: str | None
    flash_error: str | None


def parse_int_list(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_dtype_list(raw: str) -> list[torch.dtype]:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    out: list[torch.dtype] = []
    for token in raw.split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"Unsupported dtype {token!r}. Choose from: float32,bfloat16")
        out.append(mapping[key])
    if not out:
        raise ValueError("Expected at least one dtype.")
    return out


def dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.bfloat16:
        return "bfloat16"
    raise ValueError(f"Unsupported dtype: {dtype}")


def is_oom_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def maybe_speedup(torch_ms: float | None, flash_ms: float | None) -> float | None:
    if torch_ms is None or flash_ms is None:
        return None
    return torch_ms / flash_ms


def regular_pytorch_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = True,
) -> torch.Tensor:
    scale = Q.shape[-1] ** -0.5
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if is_causal:
        n_queries = S.shape[-2]
        n_keys = S.shape[-1]
        q_idx = torch.arange(n_queries, device=S.device)[:, None]
        k_idx = torch.arange(n_keys, device=S.device)[None, :]
        S = torch.where(q_idx >= k_idx, S, torch.full_like(S, NEG_INF_MASK_VALUE))
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, V)


def make_impls(
    compile_flash: bool,
    compile_pytorch: bool,
) -> tuple[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor], Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]:
    def flash_impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return FlashAttention2Triton.apply(q, k, v, True)

    def torch_impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return regular_pytorch_attention(q, k, v, True)

    if compile_flash:
        flash_impl = torch.compile(flash_impl, dynamic=True)
    if compile_pytorch:
        torch_impl = torch.compile(torch_impl, dynamic=True)
    return flash_impl, torch_impl


def bench_impl(
    impl: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q_base: torch.Tensor,
    k_base: torch.Tensor,
    v_base: torch.Tensor,
    warmup_ms: int,
    rep_ms: int,
) -> tuple[str, TimingResult | None, str | None]:
    q = q_base.detach().clone().requires_grad_(True)
    k = k_base.detach().clone().requires_grad_(True)
    v = v_base.detach().clone().requires_grad_(True)

    try:
        def run_forward() -> None:
            with torch.no_grad():
                _ = impl(q, k, v)

        forward_ms = float(triton.testing.do_bench(run_forward, warmup=warmup_ms, rep=rep_ms))

        output = impl(q, k, v)
        grad_output = torch.randn_like(output)

        def run_backward() -> None:
            q.grad = None
            k.grad = None
            v.grad = None
            output.backward(grad_output, retain_graph=True)

        backward_ms = float(triton.testing.do_bench(run_backward, warmup=warmup_ms, rep=rep_ms))

        def run_end_to_end() -> None:
            q.grad = None
            k.grad = None
            v.grad = None
            out = impl(q, k, v)
            out.backward(torch.ones_like(out))

        end_to_end_ms = float(triton.testing.do_bench(run_end_to_end, warmup=warmup_ms, rep=rep_ms))

        return "ok", TimingResult(forward_ms, backward_ms, end_to_end_ms), None
    except RuntimeError as exc:
        if is_oom_error(exc):
            if q.is_cuda:
                torch.cuda.empty_cache()
            return "oom", None, str(exc).splitlines()[0]
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FlashAttention-2 Triton implementation vs regular PyTorch attention.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--rep-ms", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-lengths", type=str, default="128,256,512,1024,2048,4096,8192,16384,32768,65536")
    parser.add_argument("--d-models", type=str, default="16,32,64,128")
    parser.add_argument("--dtypes", type=str, default="bfloat16,float32")
    parser.add_argument("--compile-flash", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-pytorch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def print_table(rows: list[BenchmarkRow]) -> None:
    header = (
        f"{'dtype':>9} {'seq':>7} {'d':>5} {'status(torch/flash)':>20} "
        f"{'fwd torch':>10} {'fwd flash':>10} {'fwd x':>8} "
        f"{'bwd torch':>10} {'bwd flash':>10} {'bwd x':>8} "
        f"{'e2e torch':>10} {'e2e flash':>10} {'e2e x':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        status = f"{row.torch_status}/{row.flash_status}"
        fwd_t = f"{row.torch_forward_ms:.3f}" if row.torch_forward_ms is not None else "-"
        fwd_f = f"{row.flash_forward_ms:.3f}" if row.flash_forward_ms is not None else "-"
        fwd_x = f"{row.forward_speedup:.3f}" if row.forward_speedup is not None else "-"
        bwd_t = f"{row.torch_backward_ms:.3f}" if row.torch_backward_ms is not None else "-"
        bwd_f = f"{row.flash_backward_ms:.3f}" if row.flash_backward_ms is not None else "-"
        bwd_x = f"{row.backward_speedup:.3f}" if row.backward_speedup is not None else "-"
        e2e_t = f"{row.torch_end_to_end_ms:.3f}" if row.torch_end_to_end_ms is not None else "-"
        e2e_f = f"{row.flash_end_to_end_ms:.3f}" if row.flash_end_to_end_ms is not None else "-"
        e2e_x = f"{row.end_to_end_speedup:.3f}" if row.end_to_end_speedup is not None else "-"

        print(
            f"{row.dtype:>9} {row.sequence_length:7d} {row.d_model:5d} {status:>20} "
            f"{fwd_t:>10} {fwd_f:>10} {fwd_x:>8} "
            f"{bwd_t:>10} {bwd_f:>10} {bwd_x:>8} "
            f"{e2e_t:>10} {e2e_f:>10} {e2e_x:>8}"
        )
        if row.torch_error:
            print(f"  torch_error: {row.torch_error}")
        if row.flash_error:
            print(f"  flash_error: {row.flash_error}")


def write_csv(path: Path, rows: list[BenchmarkRow]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(path: Path, rows: list[BenchmarkRow], args: argparse.Namespace) -> None:
    payload = {
        "device": args.device,
        "batch_size": args.batch_size,
        "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms,
        "compile_flash": args.compile_flash,
        "compile_pytorch": args.compile_pytorch,
        "rows": [asdict(row) for row in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("PDF requirement for flash_benchmarking is batch_size=1.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    seq_lengths = parse_int_list(args.sequence_lengths)
    d_models = parse_int_list(args.d_models)
    dtypes = parse_dtype_list(args.dtypes)

    rows: list[BenchmarkRow] = []
    total = len(dtypes) * len(d_models) * len(seq_lengths)
    current = 0
    for dtype in dtypes:
        for d_model in d_models:
            for seq_len in seq_lengths:
                # Reset compilation state between benchmark points to avoid
                # cross-shape / cross-dtype recompile-limit interference.
                torch._dynamo.reset()
                FlashAttention2Triton._compiled_backward = None
                flash_impl, torch_impl = make_impls(
                    compile_flash=args.compile_flash,
                    compile_pytorch=args.compile_pytorch,
                )

                current += 1
                dtype_str = dtype_name(dtype)
                print(f"[{current}/{total}] dtype={dtype_str}, seq_len={seq_len}, d_model={d_model}")

                q_base = torch.randn((args.batch_size, seq_len, d_model), device=device, dtype=dtype)
                k_base = torch.randn((args.batch_size, seq_len, d_model), device=device, dtype=dtype)
                v_base = torch.randn((args.batch_size, seq_len, d_model), device=device, dtype=dtype)

                torch_status, torch_timing, torch_error = bench_impl(
                    torch_impl, q_base, k_base, v_base, args.warmup_ms, args.rep_ms
                )
                flash_status, flash_timing, flash_error = bench_impl(
                    flash_impl, q_base, k_base, v_base, args.warmup_ms, args.rep_ms
                )

                row = BenchmarkRow(
                    dtype=dtype_str,
                    sequence_length=seq_len,
                    d_model=d_model,
                    torch_status=torch_status,
                    flash_status=flash_status,
                    torch_forward_ms=torch_timing.forward_ms if torch_timing else None,
                    flash_forward_ms=flash_timing.forward_ms if flash_timing else None,
                    forward_speedup=maybe_speedup(
                        torch_timing.forward_ms if torch_timing else None,
                        flash_timing.forward_ms if flash_timing else None,
                    ),
                    torch_backward_ms=torch_timing.backward_ms if torch_timing else None,
                    flash_backward_ms=flash_timing.backward_ms if flash_timing else None,
                    backward_speedup=maybe_speedup(
                        torch_timing.backward_ms if torch_timing else None,
                        flash_timing.backward_ms if flash_timing else None,
                    ),
                    torch_end_to_end_ms=torch_timing.end_to_end_ms if torch_timing else None,
                    flash_end_to_end_ms=flash_timing.end_to_end_ms if flash_timing else None,
                    end_to_end_speedup=maybe_speedup(
                        torch_timing.end_to_end_ms if torch_timing else None,
                        flash_timing.end_to_end_ms if flash_timing else None,
                    ),
                    torch_error=torch_error,
                    flash_error=flash_error,
                )
                rows.append(row)

    print()
    print_table(rows)

    if args.csv_output is not None:
        write_csv(args.csv_output, rows)
        print(f"\nWrote CSV results to {args.csv_output}")
    if args.json_output is not None:
        write_json(args.json_output, rows, args)
        print(f"Wrote JSON results to {args.json_output}")


if __name__ == "__main__":
    main()
