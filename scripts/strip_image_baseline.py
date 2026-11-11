#!/usr/bin/env python3
"""Drop image build fixes from predictions taken against a plain HEAD baseline.

SWE-bench images ship /testbed with uncommitted build fixes. Runs that diffed
against HEAD folded those into every prediction, and since the evaluator's
container already carries them, `git apply` of such a patch fails and the
instance is scored unresolved through no fault of the agent.

Only sections that match the image's own diff byte for byte are removed, so a
file the agent genuinely edited is never touched.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inject_smells import atomic_write, load_images

UTF8 = "utf-8"
DOCKER_TIMEOUT = 300


def image_diff(image: str) -> str:
    """The uncommitted changes an image ships in /testbed."""
    result = subprocess.run(
        [
            "docker", "run", "--rm", "--entrypoint", "/bin/sh", image, "-lc",
            "git config --global --add safe.directory /testbed >/dev/null 2>&1; "
            "cd /testbed && git diff --no-ext-diff",
        ],
        capture_output=True,
        text=True,
        timeout=DOCKER_TIMEOUT,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout


def section(patch: str, path: str) -> str | None:
    match = re.search(
        r"^diff --git a/%s b/.*?(?=^diff --git |\Z)" % re.escape(path),
        patch,
        re.M | re.S,
    )
    return match.group() if match else None


def comparable(text: str) -> str:
    """Blob hashes differ between baselines; the hunks are what must match."""
    return "\n".join(
        line for line in text.splitlines() if not line.startswith("index ")
    ).rstrip()


def strip_image_sections(patch: str, reference: str) -> tuple[str, list[str], list[str]]:
    """Remove sections identical to the image's diff; report those that differ."""
    dropped: list[str] = []
    kept_but_dirty: list[str] = []
    output: list[str] = []
    for match in re.finditer(r"^diff --git a/(\S+) b/.*?(?=^diff --git |\Z)",
                             patch, re.M | re.S):
        path, whole = match.group(1), match.group()
        mirror = section(reference, path)
        if mirror is None:
            output.append(whole)
        elif comparable(mirror) == comparable(whole):
            dropped.append(path)
        else:
            kept_but_dirty.append(path)
            output.append(whole)
    return "".join(output), dropped, kept_but_dirty


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory holding preds.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)

    path = arguments.output_dir / "preds.json"
    predictions = json.loads(path.read_text(encoding=UTF8))
    images = load_images()

    def work(instance_id: str):
        try:
            reference = image_diff(images[instance_id])
        except (KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
            return instance_id, None, [], [], str(error)[:100]
        patch, dropped, dirty = strip_image_sections(
            predictions[instance_id]["model_patch"] or "", reference
        )
        return instance_id, patch, dropped, dirty, None

    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        results = list(pool.map(work, predictions))

    changed = 0
    for instance_id, patch, dropped, dirty, error in sorted(results):
        if error:
            print(f"  [skipped] {instance_id}: {error}")
            continue
        for path_name in dirty:
            print(f"  [check]   {instance_id}: {path_name} overlaps the image "
                  f"but differs, left in place")
        if not dropped:
            continue
        changed += 1
        print(f"  {instance_id}: dropped {' '.join(dropped)}")
        if arguments.dry_run:
            continue
        predictions[instance_id]["model_patch"] = patch
        instance_patch = arguments.output_dir / instance_id / "patch.diff"
        if instance_patch.is_file():
            atomic_write(instance_patch, patch)

    if changed and not arguments.dry_run:
        atomic_write(path, json.dumps(predictions, ensure_ascii=False, indent=2))
    verb = "would clean" if arguments.dry_run else "cleaned"
    print(f"{verb} {changed} of {len(predictions)} predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
