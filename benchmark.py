from __future__ import annotations

import argparse
import json
import statistics
import timeit
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_systems.annotated_scaled_dot_product_attention import install_annotated_scaled_dot_product_attention


MODEL_SPECS: dict[str, dict[str, int]] = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7b": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


@dataclass
class BenchmarkSummary:
    step_times_s: list[float]
    mean_s: float
    std_s: float
    min_s: float
    max_s: float
    forward_times_s: list[float]
    forward_mean_s: float | None
    forward_std_s: float | None
    backward_times_s: list[float]
    backward_mean_s: float | None
    backward_std_s: float | None
    optimizer_step_times_s: list[float]
    optimizer_step_mean_s: float | None
    optimizer_step_std_s: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_times_s": self.step_times_s,
            "mean_s": self.mean_s,
            "std_s": self.std_s,
            "min_s": self.min_s,
            "max_s": self.max_s,
            "forward_times_s": self.forward_times_s,
            "forward_mean_s": self.forward_mean_s,
            "forward_std_s": self.forward_std_s,
            "backward_times_s": self.backward_times_s,
            "backward_mean_s": self.backward_mean_s,
            "backward_std_s": self.backward_std_s,
            "optimizer_step_times_s": self.optimizer_step_times_s,
            "optimizer_step_mean_s": self.optimizer_step_mean_s,
            "optimizer_step_std_s": self.optimizer_step_std_s,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark forward/backward/training steps for CS336 models.")

    parser.add_argument("--model-size", type=str, default="small", help="small, medium, large, xl, 2.7b")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)

    parser.add_argument("--mode", choices=["forward", "forward-backward", "train-step"], default="forward")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--mp-dtype", choices=["fp16", "bf16"], default="bf16")

    parser.add_argument("--enable-nvtx", action="store_true")
    parser.add_argument("--memory-profile", type=str, default=None, help="Path to write memory snapshot pickle.")
    parser.add_argument("--memory-max-entries", type=int, default=1_000_000)

    parser.add_argument("--json-output", type=str, default=None)

    return parser.parse_args()


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

    spec = dict(MODEL_SPECS[model_size])
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


def parse_autocast_dtype(mp_dtype: str) -> torch.dtype:
    if mp_dtype == "fp16":
        return torch.float16
    if mp_dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed precision dtype: {mp_dtype}")


def cuda_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def maybe_nvtx_range(enabled: bool, name: str):
    if not enabled:
        return nullcontext()
    return torch.cuda.nvtx.range(name)


def run_one_step(
    mode: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    autocast_context,
    enable_nvtx: bool,
    device: torch.device,
    collect_stage_timings: bool = False,
) -> dict[str, float]:
    stage_times_s: dict[str, float] = {}

    def timed_stage(name: str, fn) -> None:
        if collect_stage_timings:
            cuda_sync_if_needed(device)
            start = timeit.default_timer()
            fn()
            cuda_sync_if_needed(device)
            stage_times_s[name] = timeit.default_timer() - start
            return
        fn()

    if mode == "forward":
        def do_forward() -> None:
            with maybe_nvtx_range(enable_nvtx, "forward"):
                with autocast_context():
                    _ = model(input_ids)

        timed_stage("forward_s", do_forward)
        return stage_times_s

    optimizer.zero_grad(set_to_none=True)

    loss = None

    def do_forward() -> None:
        nonlocal loss
        with maybe_nvtx_range(enable_nvtx, "forward"):
            with autocast_context():
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

    timed_stage("forward_s", do_forward)

    def do_backward() -> None:
        with maybe_nvtx_range(enable_nvtx, "backward"):
            assert loss is not None
            loss.backward()

    timed_stage("backward_s", do_backward)

    if mode == "train-step":
        def do_optimizer_step() -> None:
            with maybe_nvtx_range(enable_nvtx, "optimizer-step"):
                optimizer.step()

        timed_stage("optimizer_step_s", do_optimizer_step)

    return stage_times_s


def run_benchmark(args: argparse.Namespace) -> tuple[BenchmarkSummary, dict[str, Any]]:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    hparams = resolve_model_hparams(args)

    model = BasicsTransformerLM(**hparams).to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr)

    input_ids = torch.randint(
        low=0,
        high=hparams["vocab_size"],
        size=(args.batch_size, hparams["context_length"]),
        device=device,
    )
    labels = torch.randint(
        low=0,
        high=hparams["vocab_size"],
        size=(args.batch_size, hparams["context_length"]),
        device=device,
    )

    if args.mixed_precision:
        if device.type != "cuda":
            raise RuntimeError("--mixed-precision currently requires a CUDA device.")
        dtype = parse_autocast_dtype(args.mp_dtype)
        autocast_context = lambda: torch.autocast(device_type="cuda", dtype=dtype)  # noqa: E731
    else:
        autocast_context = nullcontext

    if args.enable_nvtx and device.type != "cuda":
        raise RuntimeError("--enable-nvtx requires CUDA.")
    if args.enable_nvtx:
        install_annotated_scaled_dot_product_attention()

    with maybe_nvtx_range(args.enable_nvtx, "warmup"):
        for _ in range(args.warmup_steps):
            run_one_step(
                mode=args.mode,
                model=model,
                optimizer=optimizer,
                input_ids=input_ids,
                labels=labels,
                autocast_context=autocast_context,
                enable_nvtx=args.enable_nvtx,
                device=device,
            )
            cuda_sync_if_needed(device)

    if args.memory_profile is not None:
        if device.type != "cuda":
            raise RuntimeError("--memory-profile requires CUDA.")
        snapshot_path = Path(args.memory_profile)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._record_memory_history(max_entries=args.memory_max_entries)

    step_times_s: list[float] = []
    forward_times_s: list[float] = []
    backward_times_s: list[float] = []
    optimizer_step_times_s: list[float] = []
    with maybe_nvtx_range(args.enable_nvtx, "measured"):
        for _ in range(args.measure_steps):
            start = timeit.default_timer()
            stage_times_s = run_one_step(
                mode=args.mode,
                model=model,
                optimizer=optimizer,
                input_ids=input_ids,
                labels=labels,
                autocast_context=autocast_context,
                enable_nvtx=args.enable_nvtx,
                device=device,
                collect_stage_timings=True,
            )
            cuda_sync_if_needed(device)
            end = timeit.default_timer()
            step_times_s.append(end - start)
            if "forward_s" in stage_times_s:
                forward_times_s.append(stage_times_s["forward_s"])
            if "backward_s" in stage_times_s:
                backward_times_s.append(stage_times_s["backward_s"])
            if "optimizer_step_s" in stage_times_s:
                optimizer_step_times_s.append(stage_times_s["optimizer_step_s"])

    if args.memory_profile is not None:
        torch.cuda.memory._dump_snapshot(str(snapshot_path))
        torch.cuda.memory._record_memory_history(enabled=None)

    mean_s = statistics.mean(step_times_s) if step_times_s else float("nan")
    std_s = statistics.stdev(step_times_s) if len(step_times_s) > 1 else 0.0
    min_s = min(step_times_s) if step_times_s else float("nan")
    max_s = max(step_times_s) if step_times_s else float("nan")
    forward_mean_s = statistics.mean(forward_times_s) if forward_times_s else None
    forward_std_s = statistics.stdev(forward_times_s) if len(forward_times_s) > 1 else (0.0 if forward_times_s else None)
    backward_mean_s = statistics.mean(backward_times_s) if backward_times_s else None
    backward_std_s = statistics.stdev(backward_times_s) if len(backward_times_s) > 1 else (0.0 if backward_times_s else None)
    optimizer_step_mean_s = statistics.mean(optimizer_step_times_s) if optimizer_step_times_s else None
    optimizer_step_std_s = (
        statistics.stdev(optimizer_step_times_s)
        if len(optimizer_step_times_s) > 1
        else (0.0 if optimizer_step_times_s else None)
    )

    summary = BenchmarkSummary(
        step_times_s=step_times_s,
        mean_s=mean_s,
        std_s=std_s,
        min_s=min_s,
        max_s=max_s,
        forward_times_s=forward_times_s,
        forward_mean_s=forward_mean_s,
        forward_std_s=forward_std_s,
        backward_times_s=backward_times_s,
        backward_mean_s=backward_mean_s,
        backward_std_s=backward_std_s,
        optimizer_step_times_s=optimizer_step_times_s,
        optimizer_step_mean_s=optimizer_step_mean_s,
        optimizer_step_std_s=optimizer_step_std_s,
    )
    return summary, hparams


def main() -> None:
    args = parse_args()
    summary, hparams = run_benchmark(args)

    result = {
        "mode": args.mode,
        "device": args.device,
        "model_size": canonical_model_size(args.model_size),
        "hparams": hparams,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "mixed_precision": args.mixed_precision,
        "mp_dtype": args.mp_dtype if args.mixed_precision else None,
        "summary": summary.to_dict(),
    }

    print(json.dumps(result, indent=2))

    if args.json_output is not None:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
