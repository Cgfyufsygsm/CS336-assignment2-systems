#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Any


FILE_RE = re.compile(r"^ctx(?P<context>\d+)_(?P<mode>forward|train-step)\.pickle$")


def _end_active_by_device(snapshot: dict[str, Any]) -> dict[int, int]:
    out: dict[int, int] = {}
    for seg in snapshot.get("segments", []):
        dev = int(seg.get("device", 0))
        out[dev] = out.get(dev, 0) + int(seg.get("active_size", 0))
    return out


def _trace_stats(end_active_bytes: int, trace: list[dict[str, Any]]) -> tuple[int, int, int]:
    # Replay alloc/free events relative to profiling start.
    prefix = 0
    max_prefix = 0
    min_prefix = 0
    for event in trace:
        action = event.get("action")
        size = int(event.get("size", 0))
        if action == "alloc":
            prefix += size
        elif action == "free_completed":
            prefix -= size
        if prefix > max_prefix:
            max_prefix = prefix
        if prefix < min_prefix:
            min_prefix = prefix

    # end = start + final_prefix  =>  start = end - final_prefix
    start_active = end_active_bytes - prefix
    peak_active = start_active + max_prefix
    trough_active = start_active + min_prefix
    return start_active, peak_active, trough_active


def gib(nbytes: int) -> float:
    return nbytes / (1024 * 1024 * 1024)


def collect(base_dir: Path) -> dict[str, dict[tuple[int, str], dict[str, int]]]:
    result: dict[str, dict[tuple[int, str], dict[str, int]]] = {}
    for precision_dir in sorted(base_dir.iterdir()):
        if not precision_dir.is_dir():
            continue
        precision = precision_dir.name
        result[precision] = {}
        for pkl_path in sorted(precision_dir.glob("ctx*_*.pickle")):
            m = FILE_RE.match(pkl_path.name)
            if not m:
                continue
            context = int(m.group("context"))
            mode = m.group("mode")
            snapshot = pickle.loads(pkl_path.read_bytes())

            end_by_device = _end_active_by_device(snapshot)
            traces = snapshot.get("device_traces", [])

            start_total = 0
            peak_total = 0
            trough_total = 0

            for device_idx, trace in enumerate(traces):
                end_active = end_by_device.get(device_idx, 0)
                start_active, peak_active, trough_active = _trace_stats(end_active, trace)
                start_total += start_active
                peak_total += peak_active
                trough_total += trough_active

            end_total = sum(end_by_device.values())
            result[precision][(context, mode)] = {
                "start_active_bytes": start_total,
                "peak_active_bytes": peak_total,
                "end_active_bytes": end_total,
                "trough_active_bytes": trough_total,
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize peak active memory from PyTorch memory snapshot pickles.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/memory_profiles"),
        help="Directory containing precision subdirectories (e.g., fp32/, bf16/).",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Optional path to write the same summary as a Markdown table.",
    )
    args = parser.parse_args()

    data = collect(args.base_dir)
    precisions = sorted(data.keys())
    contexts = sorted({ctx for values in data.values() for (ctx, _) in values.keys()})
    modes = ["forward", "train-step"]

    if not precisions:
        raise SystemExit(f"No precision directories found under {args.base_dir}")

    lines: list[str] = []
    lines.append("Peak active memory from memory snapshot pickles")
    lines.append(f"Base dir: {args.base_dir}")
    lines.append("")
    lines.append("| Precision | Context | Forward Peak (GiB) | Train-step Peak (GiB) |")
    lines.append("| --------- | ------: | -----------------: | --------------------: |")
    for precision in precisions:
        for ctx in contexts:
            row: list[str] = []
            for mode in modes:
                entry = data[precision].get((ctx, mode))
                if entry is None:
                    row.append("NA")
                else:
                    row.append(f"{gib(entry['peak_active_bytes']):.2f}")
            lines.append(f"| {precision} | {ctx} | {row[0]} | {row[1]} |")

    lines.append("")
    lines.append("| Precision | Context | Mode | Start Active (GiB) | Peak Active (GiB) | End Active (GiB) |")
    lines.append("| --------- | ------: | ---- | -----------------: | ----------------: | ---------------: |")
    for precision in precisions:
        for ctx in contexts:
            for mode in modes:
                entry = data[precision].get((ctx, mode))
                if entry is None:
                    continue
                lines.append(
                    f"| {precision} | {ctx} | {mode} | "
                    f"{gib(entry['start_active_bytes']):.2f} | "
                    f"{gib(entry['peak_active_bytes']):.2f} | "
                    f"{gib(entry['end_active_bytes']):.2f} |"
                )

    print("\n".join(lines))

    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
