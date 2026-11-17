#!/usr/bin/env python3
"""Replace elapsed_seconds with trajectory-based agent_seconds under traj_results/opencode."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_cli_swebench.py"
DEFAULT_ROOT = ROOT / "traj_results" / "opencode"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_cli_swebench", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def ordered_result(payload: dict, agent_seconds: float) -> dict:
    out = {
        "instance_id": payload["instance_id"],
        "status": payload.get("status"),
        "detail": payload.get("detail"),
        "agent_seconds": agent_seconds,
    }
    for key, value in payload.items():
        if key in out or key == "elapsed_seconds":
            continue
        out[key] = value
    return out


def backfill_instance(result_path: Path) -> tuple[str, str, float | None]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    instance_id = payload.get("instance_id") or result_path.parent.name
    traj = result_path.with_name("trajectory.jsonl")
    if payload.get("agent_seconds") is not None and "elapsed_seconds" not in payload:
        parsed = runner.trajectory_agent_seconds(traj)
        if parsed is not None and abs(parsed - float(payload["agent_seconds"])) < 1e-6:
            return instance_id, "already", parsed
    parsed = runner.trajectory_agent_seconds(traj)
    if parsed is None:
        reason = "missing trajectory" if not traj.is_file() else "no timestamps"
        return instance_id, reason, None
    runner.atomic_write(
        result_path,
        json.dumps(ordered_result(payload, parsed), ensure_ascii=False, indent=2) + "\n",
    )
    return instance_id, "updated", parsed


def backfill_summary(summary_path: Path) -> tuple[int, int]:
    items = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"{summary_path} is not a JSON list")
    updated = skipped = 0
    changed = False
    for item in items:
        instance_id = item.get("instance_id")
        if not instance_id:
            skipped += 1
            continue
        result_path = summary_path.parent / instance_id / "result.json"
        if not result_path.is_file():
            skipped += 1
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if "agent_seconds" not in result:
            skipped += 1
            continue
        new_item = ordered_result(item, float(result["agent_seconds"]))
        if new_item != item:
            item.clear()
            item.update(new_item)
            changed = True
            updated += 1
        else:
            skipped += 1
    if changed:
        runner.atomic_write(
            summary_path,
            json.dumps(items, ensure_ascii=False, indent=2) + "\n",
        )
    return updated, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="directory of OpenCode run folders",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    # <model>/<run>/<instance>/result.json
    result_paths = sorted(root.glob("*/*/*/result.json"))
    print(f"found {len(result_paths)} result.json under {root}", flush=True)

    counts: dict[str, int] = {}
    failures: list[str] = []
    for index, path in enumerate(result_paths, 1):
        instance_id, status, value = backfill_instance(path)
        counts[status] = counts.get(status, 0) + 1
        if status not in {"updated", "already"}:
            rel = path.relative_to(root)
            failures.append(f"{rel}: {status}")
        if index % 500 == 0:
            print(f"  results {index}/{len(result_paths)}", flush=True)

    summary_paths = sorted(root.glob("*/*/summary.json"))
    summary_updated = summary_skipped = 0
    for path in summary_paths:
        updated, skipped = backfill_summary(path)
        summary_updated += updated
        summary_skipped += skipped

    print("result.json:", dict(sorted(counts.items())))
    print(f"summary.json files={len(summary_paths)} entries_updated={summary_updated} unchanged={summary_skipped}")
    if failures:
        print(f"skipped {len(failures)} results:")
        for line in failures:
            print(f"  {line}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
