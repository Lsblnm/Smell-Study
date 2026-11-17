#!/usr/bin/env python3
"""Copy instance runs into other task dirs, and rebuild summary.json from result.json."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT_SKIP_DIRS = {".git"}


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _load_json(path: Path, default):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _instance_dirs(src: Path) -> list[Path]:
    dirs = []
    for child in sorted(src.iterdir()):
        if not child.is_dir() or child.name in ROOT_SKIP_DIRS:
            continue
        dirs.append(child)
    return dirs


def copy_instances(src: str | Path, targets: list[str | Path]) -> None:
    """把 src 下所有实例子目录复制到每个 target，并把 src/preds.json 合并进 target/preds.json。

    已存在的实例目录、已存在的 preds 键不会被覆盖。
    """
    src_dir = _as_path(src)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"源目录不存在: {src_dir}")

    src_preds = _load_json(src_dir / "preds.json", {})
    if not isinstance(src_preds, dict):
        raise ValueError(f"{src_dir / 'preds.json'} 不是 JSON 对象")

    instances = _instance_dirs(src_dir)
    if not instances:
        print(f"[copy] {src_dir} 下没有实例子目录")

    for target in targets:
        dst_dir = _as_path(target)
        dst_dir.mkdir(parents=True, exist_ok=True)

        copied = skipped = 0
        for inst in instances:
            dest = dst_dir / inst.name
            if dest.exists():
                skipped += 1
                continue
            shutil.copytree(inst, dest)
            copied += 1

        dst_preds_path = dst_dir / "preds.json"
        dst_preds = _load_json(dst_preds_path, {})
        if not isinstance(dst_preds, dict):
            raise ValueError(f"{dst_preds_path} 不是 JSON 对象")

        added = overwritten_skipped = 0
        for instance_id, record in src_preds.items():
            if instance_id in dst_preds:
                overwritten_skipped += 1
                continue
            dst_preds[instance_id] = record
            added += 1
        _dump_json(dst_preds_path, dst_preds)

        print(
            f"[copy] {src_dir} -> {dst_dir}: "
            f"dirs copied={copied} skipped={skipped}; "
            f"preds added={added} skipped={overwritten_skipped}"
        )


def rebuild_summaries(root: str | Path) -> None:
    """清空并重建 root 下每个任务子目录的 summary.json。

    每个任务目录（如 traj_results/opencode/glm/GC-S2）的 summary.json
    由该目录下各实例的 result.json 重新生成。
    """
    root_dir = _as_path(root)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"根目录不存在: {root_dir}")

    task_dirs = [
        child
        for child in sorted(root_dir.iterdir())
        if child.is_dir() and child.name not in ROOT_SKIP_DIRS
    ]
    if not task_dirs:
        print(f"[summary] {root_dir} 下没有任务子目录")
        return

    for task_dir in task_dirs:
        entries = []
        missing = 0
        for inst_dir in sorted(task_dir.iterdir()):
            if not inst_dir.is_dir():
                continue
            result_path = inst_dir / "result.json"
            if not result_path.is_file():
                missing += 1
                continue
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            entries.append(payload)

        _dump_json(task_dir / "summary.json", entries)
        print(
            f"[summary] {task_dir}: wrote {len(entries)} results"
            + (f", skipped {missing} dirs without result.json" if missing else "")
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    copy_p = sub.add_parser("copy", help="复制实例子目录并合并 preds.json")
    copy_p.add_argument("src", help="源任务目录，例如 traj_results/opencode/glm/None")
    copy_p.add_argument("targets", nargs="+", help="一个或多个目标任务目录")

    sum_p = sub.add_parser("summary", help="重建根目录下各任务子目录的 summary.json")
    sum_p.add_argument("root", help="例如 traj_results/opencode/glm")

    args = parser.parse_args()
    if args.cmd == "copy":
        copy_instances(args.src, args.targets)
    else:
        rebuild_summaries(args.root)
    return 0


if __name__ == "__main__":
    # 处理None，复制实例
    # copy_instances(
    # "traj_results/opencode/glm/None",
    # [
    #     "traj_results/opencode/glm/None-FE",
    #     "traj_results/opencode/glm/None-GC",
    #     "traj_results/opencode/glm/None-LM"
    # ],
    # )
    # 完善summary.json
    rebuild_summaries("traj_results/opencode/glm")