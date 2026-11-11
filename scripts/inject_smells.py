#!/usr/bin/env python3
"""Inject code smells into SWE-bench instances with isolated Codex workers."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
IMAGE_LIST = ROOT / "multilingual_300_images.txt"
PATCH_ROOT = ROOT / "smell-patch"
INTRO_ROOT = ROOT / "smell-patch-intro"
DATASET_CACHE = ROOT / ".cache" / "swebench_multilingual_test.json"
ENV_FILE = ROOT / ".env"


def smell_tmp_root() -> Path:
    """Keep worker copies on the project disk (/data1), not the small root /tmp."""
    configured = os.environ.get("SMELL_INJECT_TMPDIR", "").strip()
    return Path(configured).expanduser() if configured else ROOT / ".tmp"


def configure_temp_root() -> Path:
    global LOCK_ROOT
    root = smell_tmp_root()
    root.mkdir(parents=True, exist_ok=True)
    LOCK_ROOT = root / "inject-locks"
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(root)
    os.environ["TEMP"] = str(root)
    os.environ["TMP"] = str(root)
    return root


LOCK_ROOT = smell_tmp_root() / "inject-locks"
DATASET_NAME = "SWE-bench/SWE-bench_Multilingual"
DATASET_SPLIT = "test"
AGENTS = ("codex",)
CODEX_PROVIDER_ENV_KEYS = {
    "ikuncode": "IKUNCODE_API_KEY",
    "ikuncode_grok": "IKUNCODE_GROK_API_KEY",
}
SHARED_CODEX_HOME_FILES = ("config.toml", "auth.json", "installation_id")
# Keep the local Codex runtime off ChatGPT-hosted Apps/MCP/model catalog side
# channels. Those hang on chatgpt.com and never reach the ikuncode provider.
CODEX_RUNTIME_OVERRIDES = (
    "features.apps=false",
    "features.plugins=false",
    "features.remote_models=false",
    "apps._default.enabled=false",
    "orchestrator.mcp.enabled=false",
    "check_for_update_on_startup=false",
    'web_search="disabled"',
)

SMELLS = {
    "LongMethod.md": ("LongMethod", "LM"),
    "GodClass.md": ("GodClass", "GC"),
    "FeatureEnvy.md": ("FeatureEnvy", "FE"),
}
FINAL_STATUSES = {"SUCCESS", "FAILED", "NOT_INJECTABLE"}
PATCH_PATH_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)
PRINT_LOCK = threading.Lock()
MODEL_EFFORTS = {"low", "medium", "high"}
AGENT_OUTPUT_DIRS = (".smell-output", "smell-patch", "smell-patch-intro")
MAX_TURNS = 3
CONTINUE_PROMPT = """Continue executing the task in the current workspace.
Do not merely describe what you will do. Inspect and modify the code, run the
required validations, and finish the task. Your final response must be a single
JSON object whose "Status" field is exactly one of `SUCCESS`, `NOT_INJECTABLE`,
or `FAILED`."""
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass(frozen=True)
class Smell:
    name: str
    abbreviation: str
    prompt_path: Path


@dataclass(frozen=True)
class Paths:
    patch: Path
    intro: Path
    gold: Path
    test: Path


@dataclass(frozen=True)
class Result:
    instance_id: str
    status: str
    detail: str = ""


class CommandError(RuntimeError):
    """Raised when an external command fails."""


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def instance_log(instance_id: str, message: str) -> None:
    log(f"[{instance_id}] {message}")


def parse_instances(value: str, cwd: Path | None = None) -> list[str]:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (cwd or Path.cwd()) / candidate

    if candidate.is_file():
        raw = [
            line.strip()
            for line in candidate.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        raw = [item.strip() for item in value.split(",") if item.strip()]

    instances = list(dict.fromkeys(raw))
    if not instances:
        raise ValueError("instance list is empty")
    if any("/" in item or any(char.isspace() for char in item) for item in instances):
        raise ValueError("instance IDs must not contain slashes or whitespace")
    return instances


def parse_ids_file(value: str | Path, cwd: Path | None = None) -> list[str]:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (cwd or Path.cwd()) / candidate
    if not candidate.is_file():
        raise ValueError(f"ids file does not exist: {candidate}")
    return parse_instances(str(candidate), cwd=cwd)


def restrict_instances(instances: Sequence[str], ids: Sequence[str]) -> list[str]:
    allowed = set(instances)
    selected = [item for item in ids if item in allowed]
    outside = [item for item in ids if item not in allowed]
    if outside:
        log(f"IDS_NOT_IN_INSTANCES count={len(outside)}")
    if not selected:
        raise ValueError("no instances from --ids are present in the instance list")
    return selected


def resolve_smell(prompt_value: str, cwd: Path | None = None) -> Smell:
    prompt_path = Path(prompt_value)
    if not prompt_path.is_absolute():
        prompt_path = (cwd or Path.cwd()) / prompt_path
    prompt_path = prompt_path.resolve()
    if not prompt_path.is_file():
        raise ValueError(f"prompt does not exist: {prompt_path}")
    if prompt_path.name not in SMELLS:
        supported = ", ".join(SMELLS)
        raise ValueError(
            f"unsupported prompt {prompt_path.name!r}; expected one of: {supported}"
        )
    name, abbreviation = SMELLS[prompt_path.name]
    return Smell(name, abbreviation, prompt_path)


def load_images(path: Path = IMAGE_LIST) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"image list does not exist: {path}")
    images: dict[str, str] = {}
    prefix = "swebench/sweb.eval.x86_64."
    suffix = ":latest"
    for line in path.read_text(encoding="utf-8").splitlines():
        image = line.strip()
        if not image:
            continue
        if not image.startswith(prefix) or not image.endswith(suffix):
            raise ValueError(f"invalid SWE-bench image name: {image}")
        encoded_id = image[len(prefix) : -len(suffix)]
        if "_1776_" not in encoded_id:
            raise ValueError(f"cannot recover instance ID from image name: {image}")
        instance_id = encoded_id.replace("_1776_", "__", 1)
        images[instance_id] = image
    return images


def load_all_dataset_records() -> list[dict]:
    if DATASET_CACHE.is_file():
        log(f"LOAD_DATASET_CACHE {DATASET_CACHE}")
        return json.loads(DATASET_CACHE.read_text(encoding="utf-8"))

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ValueError(
            "datasets is not installed; install project dependencies before running"
        ) from error

    if os.environ.get("SMELL_INJECT_DATASET_ONLINE", "").lower() not in {"1", "true", "yes"}:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        log("LOAD_DATASET_OFFLINE enabled; set SMELL_INJECT_DATASET_ONLINE=1 to refresh")

    records = [dict(item) for item in load_dataset(DATASET_NAME, split=DATASET_SPLIT)]
    atomic_write(DATASET_CACHE, json.dumps(records, ensure_ascii=False, indent=2))
    log(f"WRITE_DATASET_CACHE {DATASET_CACHE}")
    return records


def load_instances(instance_ids: Sequence[str]) -> dict[str, dict]:
    wanted = set(instance_ids)
    instances = {
        item["instance_id"]: dict(item)
        for item in load_all_dataset_records()
        if item["instance_id"] in wanted
    }
    missing = [instance_id for instance_id in instance_ids if instance_id not in instances]
    if missing:
        raise ValueError(
            f"instances missing from {DATASET_NAME} {DATASET_SPLIT}: {', '.join(missing)}"
        )
    return instances


def configure_output_roots(
    patch_root: Path | None = None,
    intro_root: Path | None = None,
) -> tuple[Path, Path]:
    """Override smell-patch and smell-patch-intro independently; omitted paths keep defaults."""
    global PATCH_ROOT, INTRO_ROOT
    if patch_root is not None:
        PATCH_ROOT = patch_root.expanduser().resolve()
    else:
        PATCH_ROOT = ROOT / "smell-patch"
    if intro_root is not None:
        INTRO_ROOT = intro_root.expanduser().resolve()
    else:
        INTRO_ROOT = ROOT / "smell-patch-intro"
    return PATCH_ROOT, INTRO_ROOT


def output_paths(smell: Smell, instance_id: str) -> Paths:
    stem = f"{instance_id}-{smell.abbreviation}"
    return Paths(
        patch=PATCH_ROOT / smell.name / f"{stem}.patch",
        intro=INTRO_ROOT / smell.name / f"{stem}.json",
        gold=PATCH_ROOT / smell.name / f"{stem}-gold.patch",
        test=PATCH_ROOT / smell.name / f"{stem}-test.patch",
    )


def load_env_file(path: Path | None = None) -> None:
    """Read KEY=VALUE lines so Codex credentials do not depend on the caller's shell."""
    if path is None:
        path = ENV_FILE
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def default_codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".codex"


def prepare_worker_codex_home(destination: Path, source: Path | None = None) -> Path:
    """Give each Codex process its own sqlite dir; copy only config/auth files."""
    source = (source or default_codex_home()).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    for name in SHARED_CODEX_HOME_FILES:
        origin = source / name
        if origin.is_file():
            shutil.copy2(origin, destination / name)
    return destination


def find_codex_api_key(provider: str | None = None) -> str:
    load_env_file()
    if provider is None:
        _, _, provider = codex_model_selection()
    variable = CODEX_PROVIDER_ENV_KEYS.get(provider, "IKUNCODE_API_KEY")
    api_key = os.environ.get(variable, "").strip()
    if not api_key:
        raise ValueError(
            f"{variable} is not set; add it to .env or export it for Codex injection "
            f"(provider={provider})"
        )
    return api_key


def codex_model_selection() -> tuple[str, str, str]:
    model = os.environ.get("CODEX_MODEL", "").strip() or "gpt-5.6-sol"
    effort = os.environ.get("CODEX_MODEL_EFFORT", "").strip().lower() or "high"
    provider = os.environ.get("CODEX_MODEL_PROVIDER", "").strip() or "ikuncode"
    if effort not in MODEL_EFFORTS:
        raise ValueError(
            "CODEX_MODEL_EFFORT must be one of: "
            f"{', '.join(sorted(MODEL_EFFORTS))}"
        )
    return model, effort, provider


def codex_model_description() -> str:
    model, effort, provider = codex_model_selection()
    return f"{provider}/{model}(effort={effort})"


def run_command(
    command: Sequence[str],
    *,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        input=input_text,
        text=True,
        capture_output=True,
    )
    if check and result.returncode != 0:
        output = (result.stderr or result.stdout).strip()
        raise CommandError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{output}"
        )
    return result


def make_temporary_directory(*, prefix: str, directory: str) -> tempfile.TemporaryDirectory:
    kwargs: dict[str, object] = {"prefix": prefix, "dir": directory}
    if "ignore_cleanup_errors" in tempfile.TemporaryDirectory.__init__.__code__.co_varnames:
        kwargs["ignore_cleanup_errors"] = True
    return tempfile.TemporaryDirectory(**kwargs)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_prompt(
    smell: Smell, instance_id: str, instance: dict, container: str
) -> str:
    smell_prompt = smell.prompt_path.read_text(encoding="utf-8")
    return f"""{smell_prompt}

# Automation contract (takes precedence for paths and final output)

You are processing only SWE-bench instance `{instance_id}` in the current working
directory. Its complete metadata is included at the end of this prompt.

- Work only in the current directory. Modify only existing production source files allowed by
  the smell prompt. Do not add/delete files, modify tests, commit, or stage changes.
- Run repository commands and tests inside the isolated container with
  `docker exec {container} <command>`. Its `/testbed` is the current directory.
- When validation temporarily applies patches, finish with `/testbed` containing
  only the smell source changes so the scheduler can export `git diff HEAD`.
- If adapting gold or test patches is necessary, write them respectively to
  `.smell-output/refactored-gold.patch` and
  `.smell-output/refactored-test.patch`. These are control files; do not write the
  final smell patch because the scheduler extracts it from the source tree.
- Ignore any earlier storage rule that asks you to create `smell-patch/` or
  `smell-patch-intro/` inside the repository. Do not create those directories in
  the current workspace; the scheduler writes final artifacts outside `/testbed`.
- Your final response is the smell.json report: a single JSON object matching the
  template in the smell prompt. The "Status" field MUST be exactly one of
  `SUCCESS`, `NOT_INJECTABLE`, or `FAILED`.
- The report must include the injection location/strategy, quality-factor evidence,
  validation outcomes, or a precise failure/non-injectable reason.
- Return SUCCESS only when a non-empty smell diff remains and all required checks
  pass. A normal inability to inject or validate must still return a complete JSON
  report.

# Instance metadata

```json
{json.dumps(instance, ensure_ascii=False, indent=2)}
```
"""


def run_codex_agent(
    prompt: str, workspace: Path, instance_id: str | None = None
) -> str:
    model, effort, provider = codex_model_selection()
    try:
        from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox
    except ImportError as error:
        raise ValueError(
            "openai-codex is not installed; run `python -m pip install openai-codex`"
        ) from error

    worker_root = smell_tmp_root()
    worker_root.mkdir(parents=True, exist_ok=True)
    try:
        with make_temporary_directory(
            prefix="codex-home-",
            directory=str(worker_root),
        ) as home_name:
            env = {"CODEX_HOME": str(prepare_worker_codex_home(Path(home_name)))}
            with Codex(
                CodexConfig(config_overrides=CODEX_RUNTIME_OVERRIDES, env=env)
            ) as codex:
                thread = codex.thread_start(
                    cwd=str(workspace),
                    model=model,
                    model_provider=provider,
                    sandbox=Sandbox.full_access,
                    approval_mode=ApprovalMode.deny_all,
                    ephemeral=True,
                    config={"model_reasoning_effort": effort},
                )
                message = prompt
                reports = []
                for turn in range(1, MAX_TURNS + 1):
                    result = thread.run(message)
                    status = getattr(result.status, "value", result.status)
                    if status != "completed":
                        raise CommandError(
                            f"Codex turn ended with status {status}: {result.error}"
                        )
                    report = result.final_response or ""
                    reports.append(report)
                    if parse_status(report) is not None:
                        return report
                    if instance_id:
                        instance_log(
                            instance_id,
                            f"CODEX_CONTINUE turn={turn}/{MAX_TURNS} "
                            "reason=missing Status field",
                        )
                    message = CONTINUE_PROMPT
                raise CommandError(
                    f"Codex did not return a final JSON Status after {MAX_TURNS} turns. "
                    f"Last response: {reports[-1].strip() if reports else '<empty>'}"
                )
    except CommandError:
        raise
    except Exception as error:
        raise CommandError(
            f"Codex SDK error ({type(error).__name__}): {error}\n"
            f"{traceback.format_exc()}"
        ) from error


def prepare_workspace(image: str, workspace: Path) -> None:
    workspace.mkdir(parents=True)
    run_command(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            "root",
            "--entrypoint",
            "/bin/sh",
            "-v",
            f"{workspace}:/smell-work",
            image,
            "-lc",
            (
                "cp -a /testbed/. /smell-work/ && "
                f"chown -R {os.getuid()}:{os.getgid()} /smell-work"
            ),
        ]
    )


def start_validation_container(image: str, workspace: Path, container: str) -> None:
    run_command(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            container,
            "--user",
            "root",
            "--entrypoint",
            "/bin/sh",
            "-v",
            f"{workspace}:/testbed",
            image,
            "-lc",
            "git config --global --add safe.directory /testbed && "
            "while :; do sleep 3600; done",
        ]
    )


def restore_path_ownership(container: str, path: str, instance_id: str) -> None:
    instance_log(instance_id, f"RESTORE_OWNERSHIP {path}")
    run_command(
        [
            "docker",
            "exec",
            container,
            "sh",
            "-lc",
            (
                # chmod first so chown -R can traverse 000 directories created
                # by root-owned test processes inside the container.
                f"if [ -e {path!r} ]; then "
                f"chmod -R u+rwX {path!r} 2>/dev/null || true; "
                f"chown -R {os.getuid()}:{os.getgid()} {path!r}; "
                "fi"
            ),
        ],
        check=False,
    )


def restore_workspace_ownership(container: str, instance_id: str) -> None:
    restore_path_ownership(container, "/testbed", instance_id)


def cleanup_agent_output_dirs(workspace: Path) -> None:
    for relative in AGENT_OUTPUT_DIRS:
        path = workspace / relative
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def read_first_existing(paths: Sequence[Path]) -> str | None:
    for path in paths:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return None


def refactored_patch_candidates(
    workspace: Path, output_dir: Path, smell: Smell, instance_id: str, suffix: str
) -> list[Path]:
    stem = f"{instance_id}-{smell.abbreviation}"
    return [
        output_dir / f"refactored-{suffix}.patch",
        workspace / "smell-patch" / smell.name / f"{stem}-{suffix}.patch",
    ]


def extract_json_object(report: str) -> dict | None:
    text = report.strip()
    if not text:
        return None
    candidates: list[str] = []
    candidates.extend(match.group(1) for match in JSON_FENCE_RE.finditer(text))
    candidates.append(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_status(report: str) -> str | None:
    payload = extract_json_object(report)
    if payload is None:
        return None
    status = payload.get("Status")
    return status if isinstance(status, str) and status in FINAL_STATUSES else None


def patch_files(patch: str) -> set[str]:
    return {path for path in PATCH_PATH_RE.findall(patch) if path != "/dev/null"}


def validate_diff(
    container: str,
    diff: str,
    instance: dict,
    gold_patch: str,
    test_patch: str,
) -> None:
    if not diff.strip():
        raise ValueError("Agent reported SUCCESS but git diff is empty")

    run_command(
        ["docker", "exec", container, "git", "-C", "/testbed", "diff", "HEAD", "--check"]
    )
    names = run_command(
        [
            "docker",
            "exec",
            container,
            "git",
            "-C",
            "/testbed",
            "diff",
            "HEAD",
            "--name-status",
            "--no-renames",
        ]
    ).stdout.splitlines()
    invalid = [line for line in names if not line.startswith("M\t")]
    if invalid:
        raise ValueError(f"smell patch may only modify existing files: {invalid}")

    changed = {line.split("\t", 1)[1] for line in names}
    original_test_files = patch_files(instance.get("test_patch", ""))
    changed_tests = sorted(changed & original_test_files)
    if changed_tests:
        raise ValueError(f"smell patch modifies test files: {changed_tests}")

    status = run_command(
        [
            "docker",
            "exec",
            container,
            "git",
            "-C",
            "/testbed",
            "status",
            "--porcelain",
            "--untracked-files=all",
        ]
    ).stdout.splitlines()
    untracked = [line for line in status if line.startswith("?? ")]
    if untracked:
        raise ValueError(f"Agent left untracked files in /testbed: {untracked}")

    if gold_patch.strip():
        run_command(
            ["docker", "exec", "-i", container, "git", "-C", "/testbed", "apply", "--check", "-"],
            input_text=gold_patch,
        )
        run_command(
            ["docker", "exec", "-i", container, "git", "-C", "/testbed", "apply", "-"],
            input_text=gold_patch,
        )
    if test_patch.strip():
        run_command(
            ["docker", "exec", "-i", container, "git", "-C", "/testbed", "apply", "--check", "-"],
            input_text=test_patch,
        )


def failure_report(instance_id: str, reason: str, agent_report: str = "") -> str:
    detail = reason.strip()
    if agent_report.strip():
        detail = f"{detail}\n\n## Agent report\n\n{agent_report.strip()}"
    return json.dumps(
        {
            "Status": "FAILED",
            "实例名称": instance_id,
            "注入位置": {
                "气味类型": None,
                "注入策略": None,
                "目标文件": None,
                "目标方法": None,
                "来源文件": None,
                "暴露等级": None,
                "注入说明": None,
                "失败原因": detail,
            },
            "质量因子": {},
            "安全验证": "scheduler rejected the result",
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n"


def process_instance(
    instance: dict,
    smell: Smell,
    image: str,
) -> Result:
    instance_id = instance["instance_id"]
    paths = output_paths(smell, instance_id)
    if paths.intro.exists():
        return Result(instance_id, "SKIPPED", "smell.json exists")

    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = LOCK_ROOT / f"{smell.name}-{instance_id}.lock"
    instance_log(instance_id, f"WAIT_LOCK {lock_path}")
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return Result(instance_id, "BUSY", "another process holds the lock")
        instance_log(instance_id, "LOCK_ACQUIRED")

        if paths.intro.exists():
            return Result(instance_id, "SKIPPED", "smell.json created by another process")

        container = (
            f"smell-{smell.abbreviation.lower()}-"
            f"{re.sub(r'[^a-zA-Z0-9_.-]', '-', instance_id).lower()}-{uuid.uuid4().hex[:8]}"
        )

        worker_root = smell_tmp_root()
        worker_root.mkdir(parents=True, exist_ok=True)
        with make_temporary_directory(
            prefix="smell-worker-",
            directory=str(worker_root),
        ) as temporary_name:
            workspace = Path(temporary_name) / "testbed"
            started = False
            try:
                instance_log(instance_id, f"CHECK_IMAGE {image}")
                run_command(["docker", "image", "inspect", image])
                instance_log(instance_id, f"PREPARE_WORKSPACE {workspace}")
                prepare_workspace(image, workspace)
                output_dir = workspace / ".smell-output"
                output_dir.mkdir()
                instance_log(instance_id, f"START_CONTAINER {container}")
                start_validation_container(image, workspace, container)
                started = True
                prompt = build_prompt(smell, instance_id, instance, container)
                instance_log(
                    instance_id,
                    f"START_CODEX model={codex_model_description()}",
                )
                report = run_codex_agent(prompt, workspace, instance_id)
                instance_log(instance_id, "CODEX_FINISHED")
                invalid_detail = "invalid Codex report"

                status = parse_status(report)
                if status is None:
                    instance_log(instance_id, "INVALID_REPORT missing Status field")
                    atomic_write(
                        paths.intro,
                        failure_report(instance_id, "invalid or missing Status field", report),
                    )
                    return Result(instance_id, "FAILED", invalid_detail)
                instance_log(instance_id, f"REPORT_STATUS {status}")

                if status != "SUCCESS":
                    instance_log(instance_id, f"WRITE_JSON {paths.intro}")
                    atomic_write(paths.intro, report)
                    return Result(instance_id, status)

                restore_path_ownership(container, "/testbed/.smell-output", instance_id)
                restore_path_ownership(container, "/testbed/smell-patch", instance_id)
                restore_path_ownership(container, "/testbed/smell-patch-intro", instance_id)

                refactored_gold = read_first_existing(
                    refactored_patch_candidates(
                        workspace, output_dir, smell, instance_id, "gold"
                    )
                )
                refactored_test = read_first_existing(
                    refactored_patch_candidates(
                        workspace, output_dir, smell, instance_id, "test"
                    )
                )
                gold_patch = refactored_gold if refactored_gold is not None else instance.get("patch", "")
                test_patch = refactored_test if refactored_test is not None else instance.get("test_patch", "")
                had_refactored_gold = refactored_gold is not None
                had_refactored_test = refactored_test is not None
                cleanup_agent_output_dirs(workspace)

                instance_log(instance_id, "EXPORT_DIFF")
                diff = run_command(
                    [
                        "docker",
                        "exec",
                        container,
                        "git",
                        "-C",
                        "/testbed",
                        "diff",
                        "HEAD",
                        "--binary",
                        "--no-ext-diff",
                    ]
                ).stdout
                instance_log(instance_id, f"DIFF_BYTES {len(diff.encode('utf-8'))}")

                try:
                    instance_log(instance_id, "VALIDATE_DIFF")
                    validate_diff(container, diff, instance, gold_patch, test_patch)
                except (CommandError, ValueError) as error:
                    instance_log(instance_id, f"VALIDATION_FAILED {type(error).__name__}: {error}")
                    atomic_write(paths.intro, failure_report(instance_id, str(error), report))
                    return Result(instance_id, "FAILED", str(error))
                instance_log(instance_id, "VALIDATION_PASSED")

                instance_log(instance_id, f"WRITE_PATCH {paths.patch}")
                atomic_write(paths.patch, diff)
                if had_refactored_gold:
                    instance_log(instance_id, f"WRITE_REFACTORED_GOLD {paths.gold}")
                    atomic_write(paths.gold, gold_patch)
                if had_refactored_test:
                    instance_log(instance_id, f"WRITE_REFACTORED_TEST {paths.test}")
                    atomic_write(paths.test, test_patch)
                instance_log(instance_id, f"WRITE_JSON {paths.intro}")
                atomic_write(paths.intro, report)
                return Result(instance_id, "SUCCESS")
            except (CommandError, OSError, ValueError) as error:
                # Infrastructure/agent interruption deliberately has no final smell.json.
                instance_log(instance_id, f"ERROR_DETAIL {type(error).__name__}: {error}")
                return Result(instance_id, "ERROR", str(error))
            finally:
                if started:
                    restore_workspace_ownership(container, instance_id)
                    instance_log(instance_id, f"STOP_CONTAINER {container}")
                    run_command(["docker", "rm", "-f", container], check=False)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt_md", help="smell prompt, for example prompts/GodClass.md")
    parser.add_argument("instances", help="ID file or comma-separated instance IDs")
    parser.add_argument("parallelism", type=positive_int, help="maximum parallel instances")
    parser.add_argument(
        "--agent",
        choices=AGENTS,
        default="codex",
        help="injection SDK; only the Codex SDK is supported",
    )
    parser.add_argument(
        "--patch-root",
        type=Path,
        default=None,
        help="directory for smell patches; default is <project>/smell-patch",
    )
    parser.add_argument(
        "--intro-root",
        type=Path,
        default=None,
        help="directory for smell.json files; default is <project>/smell-patch-intro",
    )
    parser.add_argument(
        "--ids",
        type=Path,
        default=None,
        help="only inject IDs listed in this file; skip IDs whose smell.json already exists",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        smell = resolve_smell(args.prompt_md)
        instances = parse_instances(args.instances)
        if args.ids is not None:
            instances = restrict_instances(instances, parse_ids_file(args.ids))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    patch_root, intro_root = configure_output_roots(args.patch_root, args.intro_root)
    temp_root = configure_temp_root()
    ids_desc = str(args.ids) if args.ids is not None else "-"
    log(
        f"START smell={smell.name} prompt={smell.prompt_path} "
        f"agent={args.agent} instances={len(instances)} ids={ids_desc} "
        f"parallelism={args.parallelism} patch_root={patch_root} "
        f"intro_root={intro_root} temp_root={temp_root}"
    )

    pending: list[str] = []
    results: list[Result] = []
    for instance_id in instances:
        if output_paths(smell, instance_id).intro.exists():
            result = Result(instance_id, "SKIPPED", "smell.json exists")
            results.append(result)
            log(f"[{instance_id}] SKIPPED: smell.json exists")
        else:
            pending.append(instance_id)

    if not pending:
        log(f"Done: SKIPPED={len(results)}")
        return 0
    log(f"PENDING count={len(pending)}")

    try:
        log(f"LOAD_IMAGES {IMAGE_LIST}")
        images = load_images()
        log(f"LOAD_IMAGES_DONE count={len(images)}")
        _, _, provider = codex_model_selection()
        find_codex_api_key(provider)
        log(f"CODEX configured model={codex_model_description()}")
        missing_images = [instance for instance in pending if instance not in images]
        if missing_images:
            raise ValueError(f"instances missing from image list: {', '.join(missing_images)}")
        log(f"LOAD_DATASET {DATASET_NAME} split={DATASET_SPLIT}")
        instance_records = load_instances(pending)
        log(f"LOAD_DATASET_DONE count={len(instance_records)}")
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        log(f"LAUNCH_WORKERS max_workers={args.parallelism}")
        with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
            futures = {
                executor.submit(
                    process_instance,
                    instance_records[instance_id],
                    smell,
                    images[instance_id],
                ): instance_id
                for instance_id in pending
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as error:
                    instance_id = futures[future]
                    detail = f"unhandled worker exception ({type(error).__name__}): {error}"
                    instance_log(instance_id, detail)
                    instance_log(instance_id, traceback.format_exc())
                    result = Result(instance_id, "ERROR", detail)
                results.append(result)
                suffix = f": {result.detail}" if result.detail else ""
                log(f"[{result.instance_id}] {result.status}{suffix}")
    except KeyboardInterrupt:
        log("Interrupted; unfinished instances have no smell.json and will retry next run.")
        return 130

    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    log(f"Done: {summary or 'no instances'}")
    return 1 if any(result.status in {"ERROR", "FAILED"} for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
