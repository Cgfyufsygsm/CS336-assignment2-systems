#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


MB = 1_000_000
DEFAULT_BACKENDS = ["gloo", "nccl"]
DEFAULT_SIZES_MB = [1, 10, 100, 1000]
DEFAULT_WORLD_SIZES = [2, 4, 6]


@dataclass
class BenchmarkResult:
    backend: str
    device_type: str
    world_size: int
    tensor_size_mb: int
    tensor_numel: int
    latency_mean_ms: float
    latency_median_ms: float
    latency_std_ms: float
    latency_min_ms: float
    latency_max_ms: float
    payload_bandwidth_gbps: float


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def parse_csv_strs(raw: str) -> list[str]:
    values = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one backend.")
    invalid = [x for x in values if x not in {"gloo", "nccl"}]
    if invalid:
        raise ValueError(f"Unsupported backend(s): {invalid}. Allowed: gloo,nccl")
    return values


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def setup_process_group(rank: int, world_size: int, backend: str, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def benchmark_worker(
    rank: int,
    world_size: int,
    backend: str,
    tensor_size_mb: int,
    warmup_iters: int,
    measure_iters: int,
    master_port: int,
    queue: mp.SimpleQueue,
) -> None:
    setup_process_group(rank=rank, world_size=world_size, backend=backend, master_port=master_port)

    if backend == "nccl":
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    tensor_bytes = tensor_size_mb * MB
    tensor_numel = tensor_bytes // 4  # float32
    tensor = torch.ones(tensor_numel, dtype=torch.float32, device=device)

    for _ in range(warmup_iters):
        dist.barrier()
        synchronize_if_cuda(device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
        synchronize_if_cuda(device)

    latencies_ms: list[float] = []
    for _ in range(measure_iters):
        dist.barrier()
        synchronize_if_cuda(device)
        t0 = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
        synchronize_if_cuda(device)
        latencies_ms.append((time.perf_counter() - t0) * 1_000)

    gathered: list[list[float] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, latencies_ms)

    if rank == 0:
        all_latencies = [x for rank_lat in gathered for x in (rank_lat or [])]
        mean_ms = statistics.fmean(all_latencies)
        median_ms = statistics.median(all_latencies)
        std_ms = statistics.pstdev(all_latencies) if len(all_latencies) > 1 else 0.0
        min_ms = min(all_latencies)
        max_ms = max(all_latencies)
        payload_bandwidth_gbps = (tensor_bytes / 1e9) / (mean_ms / 1_000.0)
        queue.put(
            BenchmarkResult(
                backend=backend,
                device_type=device.type,
                world_size=world_size,
                tensor_size_mb=tensor_size_mb,
                tensor_numel=tensor_numel,
                latency_mean_ms=mean_ms,
                latency_median_ms=median_ms,
                latency_std_ms=std_ms,
                latency_min_ms=min_ms,
                latency_max_ms=max_ms,
                payload_bandwidth_gbps=payload_bandwidth_gbps,
            )
        )

    if dist.is_initialized():
        dist.destroy_process_group()


def print_results_table(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'backend':>7} {'device':>6} {'procs':>5} {'size(MB)':>8} "
        f"{'mean(ms)':>10} {'median(ms)':>11} {'std(ms)':>9} {'bw(GB/s)':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        mean = f"{r.latency_mean_ms:.3f}"
        median = f"{r.latency_median_ms:.3f}"
        std = f"{r.latency_std_ms:.3f}"
        bw = f"{r.payload_bandwidth_gbps:.3f}"
        print(
            f"{r.backend:>7} {r.device_type:>6} {r.world_size:5d} {r.tensor_size_mb:8d} "
            f"{mean:>10} {median:>11} {std:>9} {bw:>9}"
        )


def write_csv(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark single-node all-reduce latency for combinations of "
            "backend/device, tensor size, and number of processes."
        )
    )
    parser.add_argument("--backends", type=parse_csv_strs, default=DEFAULT_BACKENDS)
    parser.add_argument("--sizes-mb", type=parse_csv_ints, default=DEFAULT_SIZES_MB)
    parser.add_argument("--world-sizes", type=parse_csv_ints, default=DEFAULT_WORLD_SIZES)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--measure-iters", type=int, default=30)
    parser.add_argument("--master-port-base", type=int, default=29500)
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    ctx = mp.get_context("spawn")

    combos = [
        (backend, world_size, size_mb)
        for backend in args.backends
        for world_size in args.world_sizes
        for size_mb in args.sizes_mb
    ]

    for combo_index, (backend, world_size, size_mb) in enumerate(combos):
        queue = ctx.SimpleQueue()
        master_port = args.master_port_base + combo_index
        mp.spawn(
            benchmark_worker,
            args=(
                world_size,
                backend,
                size_mb,
                args.warmup_iters,
                args.measure_iters,
                master_port,
                queue,
            ),
            nprocs=world_size,
            join=True,
        )
        results.append(queue.get())
    return results


def main() -> None:
    args = parse_args()
    results = run_benchmark(args)
    print_results_table(results)

    if results and args.csv_output is not None:
        write_csv(args.csv_output, results)
        print(f"Wrote CSV results to: {args.csv_output}")


if __name__ == "__main__":
    main()
