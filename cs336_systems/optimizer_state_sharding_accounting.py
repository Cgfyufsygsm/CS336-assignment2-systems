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
from cs336_systems.ddp import DDPOverlapBucketed, DDPOverlapIndividualParameters, MinimalDDPFlat, NaiveDDP
from cs336_systems.sharded_optimizer import ShardedOptimizer


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
            "Profile memory and step-time impact of optimizer state sharding "
            "under distributed training."
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
    parser.add_argument("--ddp-impl", choices=["naive", "flat", "overlap", "bucketed"], default="naive")
    parser.add_argument("--bucket-size-mb", type=float, default=10.0)

    parser.add_argument("--optimizer-sharding", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--master-port", type=int, default=29650)

    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--profile-memory", action="store_true", default=True)
    parser.add_argument("--no-profile-memory", dest="profile_memory", action="store_false")
    parser.add_argument("--profile-time", action="store_true", default=True)
    parser.add_argument("--no-profile-time", dest="profile_time", action="store_false")

    parser.add_argument("--json-output", type=str, default=None)
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
        local_inputs.append(global_input_ids[offset : offset + local_batch_size].to(device))
        local_labels.append(global_labels[offset : offset + local_batch_size].to(device))
    return local_inputs, local_labels


def bytes_of_params(module: torch.nn.Module) -> int:
    return sum(param.numel() * param.element_size() for param in module.parameters())


def bytes_of_grads(module: torch.nn.Module) -> int:
    total = 0
    for param in module.parameters():
        if param.grad is not None:
            total += param.grad.numel() * param.grad.element_size()
    return total


def bytes_of_optimizer_state(optimizer: torch.optim.Optimizer) -> int:
    target_optimizer = optimizer
    if isinstance(optimizer, ShardedOptimizer):
        target_optimizer = optimizer._local_optimizer  # noqa: SLF001
        if target_optimizer is None:
            return 0

    total = 0
    for state in target_optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


def run_one_step(
    *,
    ddp_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    profile_memory: bool = False,
) -> dict[str, float | int]:
    if hasattr(ddp_model, "on_train_batch_start"):
        ddp_model.on_train_batch_start()

    if profile_memory and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    synchronize_if_cuda(device)
    start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    logits = ddp_model(input_ids)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()
    ddp_model.finish_gradient_synchronization()
    synchronize_if_cuda(device)

    if profile_memory and device.type == "cuda":
        peak_before_step = int(torch.cuda.max_memory_allocated(device=device))
    else:
        peak_before_step = -1

    optimizer.step()
    synchronize_if_cuda(device)
    end = time.perf_counter()

    if profile_memory and device.type == "cuda":
        peak_after_step = int(torch.cuda.max_memory_allocated(device=device))
    else:
        peak_after_step = -1

    return {
        "step_time_s": end - start,
        "peak_before_optimizer_step_bytes": peak_before_step,
        "peak_after_optimizer_step_bytes": peak_after_step,
    }


def accounting_worker(rank: int, args: argparse.Namespace, result_path: str) -> None:
    setup_process_group(rank=rank, world_size=args.world_size, backend=args.backend, master_port=args.master_port)
    try:
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
        if args.ddp_impl == "bucketed":
            ddp_model = DDP_IMPLS[args.ddp_impl](model, bucket_size_mb=args.bucket_size_mb)
        else:
            ddp_model = DDP_IMPLS[args.ddp_impl](model)
        ddp_model.train()

        optimizer_kwargs = {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "betas": (args.beta1, args.beta2),
            "eps": args.eps,
        }
        if args.optimizer_sharding:
            optimizer = ShardedOptimizer(ddp_model.parameters(), torch.optim.AdamW, **optimizer_kwargs)
        else:
            optimizer = torch.optim.AdamW(ddp_model.parameters(), **optimizer_kwargs)

        peak_after_model_init = -1
        local_memory_result: dict[str, int] = {}
        if args.profile_memory and device.type == "cuda":
            synchronize_if_cuda(device)
            torch.cuda.reset_peak_memory_stats(device=device)
            synchronize_if_cuda(device)
            peak_after_model_init = int(torch.cuda.max_memory_allocated(device=device))

        total_steps = max(1, args.warmup_steps + args.measure_steps)
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

        if args.profile_memory:
            dist.barrier()
            profiled = run_one_step(
                ddp_model=ddp_model,
                optimizer=optimizer,
                input_ids=local_inputs[0],
                labels=local_labels[0],
                device=device,
                profile_memory=True,
            )
            local_memory_result = {
                "peak_after_model_init_bytes": peak_after_model_init,
                "peak_before_optimizer_step_bytes": int(profiled["peak_before_optimizer_step_bytes"]),
                "peak_after_optimizer_step_bytes": int(profiled["peak_after_optimizer_step_bytes"]),
                "parameter_bytes": bytes_of_params(ddp_model.module),
                "gradient_bytes_after_backward": bytes_of_grads(ddp_model.module),
                "optimizer_state_bytes_local": bytes_of_optimizer_state(optimizer),
            }

        local_step_times: list[float] = []
        if args.profile_time:
            # Warmup
            for step in range(args.warmup_steps):
                dist.barrier()
                run_one_step(
                    ddp_model=ddp_model,
                    optimizer=optimizer,
                    input_ids=local_inputs[step % total_steps],
                    labels=local_labels[step % total_steps],
                    device=device,
                    profile_memory=False,
                )

            # Measured
            for step in range(args.measure_steps):
                dist.barrier()
                result = run_one_step(
                    ddp_model=ddp_model,
                    optimizer=optimizer,
                    input_ids=local_inputs[(args.warmup_steps + step) % total_steps],
                    labels=local_labels[(args.warmup_steps + step) % total_steps],
                    device=device,
                    profile_memory=False,
                )
                local_step_times.append(float(result["step_time_s"]))

        gathered_memory: list[dict[str, int] | None] = [None for _ in range(args.world_size)]
        gathered_step_times: list[list[float] | None] = [None for _ in range(args.world_size)]
        dist.all_gather_object(gathered_memory, local_memory_result if args.profile_memory else {})
        dist.all_gather_object(gathered_step_times, local_step_times if args.profile_time else [])

        if rank == 0:
            result: dict[str, Any] = {
                "optimizer_sharding": args.optimizer_sharding,
                "optimizer_name": "torch.optim.AdamW",
                "backend": args.backend,
                "world_size": args.world_size,
                "ddp_impl": args.ddp_impl,
                "bucket_size_mb": args.bucket_size_mb if args.ddp_impl == "bucketed" else None,
                "model_size": canonical_model_size(args.model_size),
                "hparams": hparams,
                "global_batch_size": args.batch_size,
                "local_batch_size": args.batch_size // args.world_size,
                "warmup_steps": args.warmup_steps,
                "measure_steps": args.measure_steps,
            }

            if args.profile_memory:
                memory_entries = [entry for entry in gathered_memory if entry]
                result["memory"] = {
                    "peak_after_model_init_bytes": max(entry["peak_after_model_init_bytes"] for entry in memory_entries),
                    "peak_before_optimizer_step_bytes": max(
                        entry["peak_before_optimizer_step_bytes"] for entry in memory_entries
                    ),
                    "peak_after_optimizer_step_bytes": max(
                        entry["peak_after_optimizer_step_bytes"] for entry in memory_entries
                    ),
                    "parameter_bytes_per_rank": max(entry["parameter_bytes"] for entry in memory_entries),
                    "gradient_bytes_after_backward_per_rank": max(
                        entry["gradient_bytes_after_backward"] for entry in memory_entries
                    ),
                    "optimizer_state_bytes_by_rank": [entry["optimizer_state_bytes_local"] for entry in memory_entries],
                }

            if args.profile_time:
                per_rank_times = [times for times in gathered_step_times if times]
                step_times = [max(times[i] for times in per_rank_times) for i in range(args.measure_steps)]
                result["time"] = {
                    "step_times_s": step_times,
                    "step_mean_s": statistics.fmean(step_times),
                    "step_std_s": statistics.pstdev(step_times) if len(step_times) > 1 else 0.0,
                    "step_min_s": min(step_times),
                    "step_max_s": max(step_times),
                }

            result_file = Path(result_path)
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.write_text(json.dumps(result, indent=2))
    finally:
        cleanup_process_group()


def main() -> None:
    args = parse_args()
    if args.world_size < 2:
        raise ValueError("--world-size must be at least 2 for distributed accounting.")
    if not args.profile_memory and not args.profile_time:
        raise ValueError("At least one of --profile-memory / --profile-time must be enabled.")

    result_path = Path(f"/tmp/optimizer_state_sharding_accounting_{os.getpid()}_{time.time_ns()}.json")
    mp.spawn(accounting_worker, args=(args, str(result_path)), nprocs=args.world_size, join=True)

    result = json.loads(result_path.read_text())
    print(json.dumps(result, indent=2))

    if args.json_output is not None:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
