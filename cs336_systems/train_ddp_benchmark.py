from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_systems.ddp import DDPOverlapBucketed, DDPOverlapIndividualParameters, MinimalDDPFlat, NaiveDDP


MODEL_SPECS: dict[str, dict[str, int]] = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7b": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

DDP_IMPLS = {
    "naive": NaiveDDP,
    "flat": MinimalDDPFlat,
    "overlap": DDPOverlapIndividualParameters,
    "bucketed": DDPOverlapBucketed,
}


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def setup_process_group(rank: int, world_size: int, backend: str, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_process_group() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def canonical_model_size(name: str) -> str:
    key = name.strip().lower()
    aliases = {
        "2.7b": "2.7b",
        "2.7B": "2.7b",
        "2_7b": "2.7b",
        "2-7b": "2.7b",
    }
    return aliases.get(key, key)


def resolve_model_hparams(args: argparse.Namespace) -> dict[str, Any]:
    model_size = canonical_model_size(args.model_size)
    if model_size not in MODEL_SPECS:
        raise ValueError(f"Unknown --model-size={args.model_size!r}. Choices: {', '.join(MODEL_SPECS.keys())}")

    spec: dict[str, Any] = dict(MODEL_SPECS[model_size])
    if args.d_model is not None:
        spec["d_model"] = args.d_model
    if args.d_ff is not None:
        spec["d_ff"] = args.d_ff
    if args.num_layers is not None:
        spec["num_layers"] = args.num_layers
    if args.num_heads is not None:
        spec["num_heads"] = args.num_heads

    spec["vocab_size"] = args.vocab_size
    spec["context_length"] = args.context_length
    spec["rope_theta"] = args.rope_theta
    return spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark DDP training and gradient communication overhead for "
            "naive (per-parameter sync all-reduce), flat (single flattened all-reduce), "
            "overlap (per-parameter async all-reduce with backward overlap), "
            "and bucketed overlap implementations."
        )
    )
    parser.add_argument("--model-size", type=str, default="xl", help="small, medium, large, xl, 2.7b")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4, help="Global batch size (across all ranks).")
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)

    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--master-port", type=int, default=29592)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--ddp-impl", choices=["naive", "flat", "overlap", "bucketed"], default="naive")
    parser.add_argument("--bucket-size-mb", type=float, default=10.0)
    return parser.parse_args()


def build_local_batches(
    *,
    rank: int,
    world_size: int,
    total_steps: int,
    global_batch_size: int,
    context_length: int,
    vocab_size: int,
    seed: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if global_batch_size % world_size != 0:
        raise ValueError("--batch-size (global) must be divisible by --world-size.")

    local_batch_size = global_batch_size // world_size
    local_inputs: list[torch.Tensor] = []
    local_labels: list[torch.Tensor] = []
    for step in range(total_steps):
        generator = torch.Generator().manual_seed(seed + step)
        global_input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(global_batch_size, context_length),
            generator=generator,
            dtype=torch.long,
        )
        global_labels = torch.randint(
            low=0,
            high=vocab_size,
            size=(global_batch_size, context_length),
            generator=generator,
            dtype=torch.long,
        )
        offset = rank * local_batch_size
        local_input_ids = global_input_ids[offset : offset + local_batch_size].to(device)
        local_step_labels = global_labels[offset : offset + local_batch_size].to(device)
        local_inputs.append(local_input_ids)
        local_labels.append(local_step_labels)
    return local_inputs, local_labels


def run_one_training_step(
    *,
    ddp_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    synchronize_if_cuda(device)
    step_start = time.perf_counter()

    if hasattr(ddp_model, "on_train_batch_start"):
        ddp_model.on_train_batch_start()

    optimizer.zero_grad(set_to_none=True)
    logits = ddp_model(input_ids)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()

    synchronize_if_cuda(device)
    comm_start = time.perf_counter()
    ddp_model.finish_gradient_synchronization()
    synchronize_if_cuda(device)
    communication_time_s = time.perf_counter() - comm_start

    optimizer.step()

    synchronize_if_cuda(device)
    step_time_s = time.perf_counter() - step_start
    return step_time_s, communication_time_s


def benchmark_worker(rank: int, args: argparse.Namespace, queue: mp.SimpleQueue) -> None:
    setup_process_group(rank=rank, world_size=args.world_size, backend=args.backend, master_port=args.master_port)

    if args.backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requested but CUDA is unavailable.")
        device_count = torch.cuda.device_count()
        if args.world_size > device_count:
            raise RuntimeError(
                f"--world-size={args.world_size} but only {device_count} CUDA device(s) are available."
            )
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)

    hparams = resolve_model_hparams(args)
    model = BasicsTransformerLM(**hparams).to(device)
    model.train()
    if args.ddp_impl == "bucketed":
        ddp_model = DDP_IMPLS[args.ddp_impl](model, bucket_size_mb=args.bucket_size_mb)
    else:
        ddp_model = DDP_IMPLS[args.ddp_impl](model)
    optimizer = AdamW(ddp_model.parameters(), lr=args.lr)

    total_steps = args.warmup_steps + args.measure_steps
    local_inputs, local_labels = build_local_batches(
        rank=rank,
        world_size=args.world_size,
        total_steps=total_steps,
        global_batch_size=args.batch_size,
        context_length=hparams["context_length"],
        vocab_size=hparams["vocab_size"],
        seed=args.seed,
        device=device,
    )

    for step in range(args.warmup_steps):
        dist.barrier()
        run_one_training_step(
            ddp_model=ddp_model,
            optimizer=optimizer,
            input_ids=local_inputs[step],
            labels=local_labels[step],
            device=device,
        )

    step_times_s: list[float] = []
    communication_times_s: list[float] = []
    for step in range(args.measure_steps):
        dist.barrier()
        idx = args.warmup_steps + step
        step_time_s, comm_time_s = run_one_training_step(
            ddp_model=ddp_model,
            optimizer=optimizer,
            input_ids=local_inputs[idx],
            labels=local_labels[idx],
            device=device,
        )
        step_times_s.append(step_time_s)
        communication_times_s.append(comm_time_s)

    gathered_step_times: list[list[float] | None] = [None for _ in range(args.world_size)]
    gathered_comm_times: list[list[float] | None] = [None for _ in range(args.world_size)]
    dist.all_gather_object(gathered_step_times, step_times_s)
    dist.all_gather_object(gathered_comm_times, communication_times_s)

    if rank == 0:
        step_times_by_rank = [per_rank for per_rank in gathered_step_times if per_rank is not None]
        comm_times_by_rank = [per_rank for per_rank in gathered_comm_times if per_rank is not None]
        if len(step_times_by_rank) != args.world_size or len(comm_times_by_rank) != args.world_size:
            raise RuntimeError("Failed to gather timing data from all ranks.")

        # End-to-end training throughput is bottlenecked by the slowest rank each step.
        step_times_s = [max(per_rank[i] for per_rank in step_times_by_rank) for i in range(args.measure_steps)]
        communication_times_s = [max(per_rank[i] for per_rank in comm_times_by_rank) for i in range(args.measure_steps)]
        comm_fraction = sum(communication_times_s) / sum(step_times_s)
        local_batch_size = args.batch_size // args.world_size
        result = {
            "backend": args.backend,
            "ddp_impl": args.ddp_impl,
            "world_size": args.world_size,
            "model_size": canonical_model_size(args.model_size),
            "hparams": hparams,
            "global_batch_size": args.batch_size,
            "local_batch_size": local_batch_size,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "bucket_size_mb": args.bucket_size_mb if args.ddp_impl == "bucketed" else None,
            "summary": {
                "step_times_s": step_times_s,
                "communication_times_s": communication_times_s,
                "step_mean_s": statistics.fmean(step_times_s),
                "step_std_s": statistics.pstdev(step_times_s) if len(step_times_s) > 1 else 0.0,
                "communication_mean_s": statistics.fmean(communication_times_s),
                "communication_std_s": statistics.pstdev(communication_times_s) if len(communication_times_s) > 1 else 0.0,
                "communication_fraction": comm_fraction,
                "communication_percent": comm_fraction * 100.0,
            },
        }
        queue.put(result)

    cleanup_process_group()


def main() -> None:
    args = parse_args()
    if args.world_size < 2:
        raise ValueError("--world-size must be at least 2 for DDP benchmarking.")

    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    mp.spawn(benchmark_worker, args=(args, queue), nprocs=args.world_size, join=True)
    result = queue.get()
    print(json.dumps(result, indent=2))

    if args.json_output is not None:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
