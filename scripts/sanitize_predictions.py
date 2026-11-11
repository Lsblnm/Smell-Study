#!/usr/bin/env python3
"""Strip build artifacts from predictions produced before the runner filtered them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inject_smells import atomic_write
from run_cli_swebench import strip_artifact_sections

PATCH_FILES = ("patch.diff", "patch.partial.diff")


def sanitize_predictions(path: Path, dry_run: bool) -> list[str]:
    predictions = json.loads(path.read_text(encoding="utf-8"))
    notes = []
    for instance_id, record in predictions.items():
        patch, dropped = strip_artifact_sections(record.get("model_patch") or "")
        if not dropped:
            continue
        record["model_patch"] = patch
        notes.append(f"  {instance_id}: dropped {' '.join(dropped)}")
    if notes and not dry_run:
        atomic_write(path, json.dumps(predictions, ensure_ascii=False, indent=2))
    return notes


def sanitize_patch_file(path: Path, dry_run: bool) -> list[str]:
    patch, dropped = strip_artifact_sections(path.read_text(encoding="utf-8"))
    if not dropped:
        return []
    if not dry_run:
        atomic_write(path, patch)
    return [f"  {path}: dropped {' '.join(dropped)}"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("traj_results"))
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)

    notes = []
    for path in sorted(arguments.root.rglob("preds.json")):
        found = sanitize_predictions(path, arguments.dry_run)
        if found:
            notes += [str(path)] + found
    for name in PATCH_FILES:
        for path in sorted(arguments.root.rglob(name)):
            notes += sanitize_patch_file(path, arguments.dry_run)

    prefix = "would clean" if arguments.dry_run else "cleaned"
    print("\n".join(notes) if notes else "nothing to clean")
    if notes:
        print(f"{prefix} {len([n for n in notes if n.startswith('  ')])} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
