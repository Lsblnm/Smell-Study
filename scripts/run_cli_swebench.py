#!/usr/bin/env python3
"""Run Codex or OpenCode on SWE-bench Multilingual instances."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inject_smells import (
    DATASET_NAME,
    DATASET_SPLIT,
    atomic_write,
    load_all_dataset_records,
    load_images,
    parse_instances,
    positive_int,
    prepare_worker_codex_home,
    prepare_workspace,
    restore_workspace_ownership,
    run_command,
    smell_tmp_root,
    start_validation_container,
)


ROOT = Path(__file__).resolve().parents[1]
UTF8 = "utf-8"
ENV_FILE = ROOT / ".env"
ENV_PLACEHOLDER = re.compile(r"^\{env:([A-Za-z_][A-Za-z0-9_]*)\}$")
OUTPUT_LOCK = threading.Lock()
PRINT_LOCK = threading.Lock()
CONTAINER_READY_TIMEOUT = 180
CONTAINER_POLL_SECONDS = 2
PREFLIGHT_TIMEOUT = 30
RETRY_BACKOFF_SECONDS = 60
REASONING_EFFORT = "high"
ARTIFACT_SIZE_LIMIT = 1 << 20
DIFF_SECTION = re.compile(r"^diff --git .*?(?=^diff --git |\Z)", re.M | re.S)
TESTBED_WIPE_SCRIPT = "if [ -d /testbed ]; then find /testbed -mindepth 1 -delete; fi"
INSTANCE_ARTIFACTS = (
    "prompt.txt",
    "trajectory.jsonl",
    "stderr.log",
    "patch.diff",
    "patch.partial.diff",
    "error.log",
    "result.json",
)


@dataclass(frozen=True)
class Result:
    instance_id: str
    status: str
    detail: str
    elapsed_seconds: float


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def load_env_file(path: Path = ENV_FILE) -> None:
    """Read KEY=VALUE lines so credentials do not depend on the caller's shell."""
    if not path.is_file():
        return
    for line in path.read_text(encoding=UTF8).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def read_agent_config(path: Path, agent: str) -> str:
    try:
        return path.read_text(encoding=UTF8)
    except OSError as error:
        raise ValueError(
            f"cannot read the {agent} config at {path}: {error}. "
            f"HOME is {Path.home()}; point it at the account owning the CLI config."
        ) from error


def codex_provider() -> tuple[str, str, Path]:
    """Return (base_url, api key variable, config path) for the active provider."""
    path = Path.home() / ".codex" / "config.toml"
    text = read_agent_config(path, "codex")
    section = re.search(r"\[model_providers\.[^\]]+\](.*?)(?=\n\[|\Z)", text, re.S)
    base = re.search(r'base_url\s*=\s*"([^"]+)"', section.group(1)) if section else None
    name = re.search(r'env_key\s*=\s*"([^"]+)"', section.group(1)) if section else None
    if base is None or name is None:
        raise ValueError(f"no usable [model_providers.*] entry in {path}")
    return base.group(1), name.group(1), path


def opencode_provider() -> tuple[str, str, Path]:
    path = Path.home() / ".config" / "opencode" / "opencode.json"
    providers = json.loads(read_agent_config(path, "opencode")).get("provider", {})
    for provider in providers.values():
        options = provider.get("options", {})
        placeholder = ENV_PLACEHOLDER.match(str(options.get("apiKey", "")))
        if options.get("baseURL") and placeholder:
            return options["baseURL"], placeholder.group(1), path
    raise ValueError(f"no provider with baseURL and a {{env:VAR}} apiKey in {path}")


def available_models(base_url: str, api_key: str) -> set[str]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=PREFLIGHT_TIMEOUT) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        raise ValueError(
            f"{base_url} rejected the credentials ({error.code} {error.reason})"
        ) from error
    except Exception as error:
        raise ValueError(f"cannot reach {base_url}: {error}") from error
    return {str(item.get("id")) for item in payload.get("data", [])}


def preflight(agent: str, model: str) -> None:
    """Fail before spending an hour discovering the credentials never arrived."""
    base_url, variable, config = (
        opencode_provider() if agent == "opencode" else codex_provider()
    )
    api_key = os.environ.get(variable)
    if not api_key:
        raise ValueError(
            f"{config} reads the API key from ${variable}, which is not set. "
            f"Add '{variable}=<key>' to {ENV_FILE} or export it."
        )
    models = available_models(base_url, api_key)
    wanted = model.rsplit("/", 1)[-1]
    if models and wanted not in models:
        raise ValueError(
            f"{base_url} does not serve model '{wanted}'. "
            f"Available: {', '.join(sorted(models))}"
        )
    log(f"PREFLIGHT ok agent={agent} base_url={base_url} model={wanted}")


def reset_instance_dir(instance_dir: Path) -> None:
    """Drop artifacts from earlier attempts so a rerun never mixes generations."""
    instance_dir.mkdir(parents=True, exist_ok=True)
    for name in INSTANCE_ARTIFACTS:
        (instance_dir / name).unlink(missing_ok=True)


def resolve_smell_patch(smell_dir: Path, instance_id: str) -> Path:
    exact = smell_dir / f"{instance_id}.patch"
    matches = (
        [exact]
        if exact.is_file()
        else [
            patch
            for patch in sorted(smell_dir.glob(f"{instance_id}-*.patch"))
            if not patch.name.endswith(("-gold.patch", "-test.patch"))
        ]
    )
    if len(matches) != 1:
        raise ValueError(
            f"expected one smell patch for {instance_id} in {smell_dir}, "
            f"found {len(matches)}"
        )
    return matches[0].resolve()


def materialize_test_patch(
    instance: dict,
    dest: Path,
    smell_patch: Path | None = None,
) -> Path | None:
    """Prefer a smell-adapted ``*-test.patch``; otherwise use the dataset test patch."""
    if smell_patch is not None:
        adapted = smell_patch.with_name(f"{smell_patch.stem}-test.patch")
        if adapted.is_file():
            return adapted.resolve()
    text = instance.get("test_patch") or ""
    if not str(text).strip():
        return None
    path = dest / "eval-test.patch"
    path.write_text(str(text), encoding=UTF8)
    return path


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
            f"instances missing from {DATASET_NAME} {DATASET_SPLIT}: "
            f"{', '.join(missing)}"
        )
    return instances


def build_prompt(instance: dict, container: str) -> str:
    return f"""Fix the issue described below in the current repository.

Work directly in the current directory. Inspect and edit the implementation,
and run the tests that are already in the working tree. Those tests include the
evaluation FAIL_TO_PASS cases, which currently fail, and PASS_TO_PASS cases,
which should still pass. The repository is also mounted at `/testbed` in the
isolated container `{container}`. Run repository commands that require its
dependencies with:

    docker exec {container} sh -lc 'cd /testbed && <command>'

Do not inspect or request the benchmark gold patch. Do not modify test files.
Do not commit or stage changes. Do not merely describe a solution: implement it
in the working tree. Finish after making the best validated fix you can.

Issue:

{instance["problem_statement"]}
"""


def default_opencode_data_home() -> Path:
    configured = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(configured).expanduser() if configured else Path.home() / ".local/share"
    return base / "opencode"


def prepare_worker_opencode_env(destination: Path) -> dict[str, str]:
    """Give each OpenCode process its own sqlite dir; copy only auth.json."""
    share = destination / "share"
    state = destination / "state"
    data = share / "opencode"
    data.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    auth = default_opencode_data_home() / "auth.json"
    if auth.is_file():
        shutil.copy2(auth, data / "auth.json")
    return {
        "XDG_DATA_HOME": str(share),
        "XDG_STATE_HOME": str(state),
    }


def agent_environment(agent: str, isolated_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    isolated_root.mkdir(parents=True, exist_ok=True)
    if agent == "codex":
        env["CODEX_HOME"] = str(prepare_worker_codex_home(isolated_root / "codex"))
    else:
        env.update(prepare_worker_opencode_env(isolated_root / "opencode"))
    return env


def agent_command(agent: str, model: str, workspace: Path, prompt: str) -> tuple[list[str], str | None]:
    if agent == "codex":
        return (
            [
                "codex",
                "exec",
                "--config",
                f"model_reasoning_effort={REASONING_EFFORT}",
                "--model",
                model,
                "--sandbox",
                "danger-full-access",
                "--json",
                "-",
            ],
            prompt,
        )
    return (
        [
            "opencode",
            "run",
            "--model",
            model,
            "--variant",
            REASONING_EFFORT,
            "--format",
            "json",
            "--auto",
            "--dir",
            str(workspace),
            prompt,
        ],
        None,
    )


def terminate_process_group(process: subprocess.Popen) -> None:
    for number in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, number)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def run_agent(
    agent: str,
    model: str,
    workspace: Path,
    prompt: str,
    timeout: int,
    instance_dir: Path,
    isolated_root: Path | None = None,
) -> int:
    """Stream agent output to disk so a timeout still leaves a usable trajectory."""
    command, input_text = agent_command(agent, model, workspace, prompt)
    env = agent_environment(agent, isolated_root or (instance_dir / ".agent-runtime"))
    prompt_path = instance_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding=UTF8)
    stdin = prompt_path.open(encoding=UTF8) if input_text is not None else subprocess.DEVNULL
    try:
        with (instance_dir / "trajectory.jsonl").open("w", encoding=UTF8) as stdout, (
            instance_dir / "stderr.log"
        ).open("w", encoding=UTF8) as stderr:
            process = subprocess.Popen(
                command,
                cwd=workspace,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                env=env,
                text=True,
                start_new_session=True,
            )
            try:
                return process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                terminate_process_group(process)
                raise TimeoutError(f"{agent} exceeded {timeout} seconds") from None
    finally:
        if stdin is not subprocess.DEVNULL:
            stdin.close()


def wait_for_container(container: str) -> None:
    """Block until /testbed is usable as a git work tree inside the container.

    The container marks the bind mount as a safe directory from its entrypoint, so
    execing before that finishes makes git reject /testbed for dubious ownership
    and silently degrade to --no-index behaviour.
    """
    deadline = time.monotonic() + CONTAINER_READY_TIMEOUT
    while True:
        run_command(
            [
                "docker",
                "exec",
                container,
                "git",
                "config",
                "--global",
                "safe.directory",
                "/testbed",
            ],
            check=False,
        )
        probe = run_command(
            [
                "docker",
                "exec",
                container,
                "git",
                "-C",
                "/testbed",
                "rev-parse",
                "--is-inside-work-tree",
            ],
            check=False,
        )
        if probe.returncode == 0 and probe.stdout.strip() == "true":
            return
        if time.monotonic() >= deadline:
            reason = probe.stderr.strip() or probe.stdout.strip() or "unknown error"
            raise RuntimeError(
                f"/testbed in {container} is not a git work tree after "
                f"{CONTAINER_READY_TIMEOUT}s: {reason}"
            )
        time.sleep(CONTAINER_POLL_SECONDS)


def _apply_index_clause(container_patch: str) -> str:
    return (
        f"{{ git apply --index {container_patch} || "
        f"git apply --3way --index {container_patch}; }} && "
        "test -z \"$(git ls-files -u)\" && "
    )


def prepare_baseline(
    container: str,
    smell_patch: Path | None = None,
    test_patch: Path | None = None,
) -> str:
    """Commit /testbed as the agent will find it and return that commit.

    Images ship /testbed with uncommitted build fixes: a pinned dependency in
    pom.xml, an extra ant target, a regenerated parser.c. Diffing against HEAD
    would bundle those into every prediction, and the evaluator's container
    already has them, so `git apply` of such a patch fails and costs the
    instance. Folding them into the baseline keeps predictions to the agent's
    own work, with or without a smell.

    Evaluation tests are also folded in (dataset test.patch, or a smell-adapted
    *-test.patch when present) so the agent can run FAIL_TO_PASS / PASS_TO_PASS
    without those hunks leaking into the exported prediction.
    """
    apply_extra = ""
    if smell_patch is not None:
        container_patch = "/tmp/smell.patch"
        run_command(["docker", "cp", str(smell_patch), f"{container}:{container_patch}"])
        apply_extra += _apply_index_clause(container_patch)
    if test_patch is not None:
        container_patch = "/tmp/eval-test.patch"
        run_command(["docker", "cp", str(test_patch), f"{container}:{container_patch}"])
        apply_extra += _apply_index_clause(container_patch)
    command = (
        "cd /testbed && git add -A && " + apply_extra + "tree=$(git write-tree) && "
        "parent=$(git rev-parse HEAD) && "
        "baseline=$(printf '%s\\n' 'Experiment baseline' | "
        "git -c user.name='Experiment Baseline' "
        "-c user.email='experiment@example.invalid' "
        "-c commit.gpgsign=false commit-tree \"$tree\" -p \"$parent\") && "
        "git reset --hard \"$baseline\" >/dev/null && printf '%s\\n' \"$baseline\""
    )
    return run_command(
        ["docker", "exec", container, "sh", "-lc", command]
    ).stdout.strip().splitlines()[-1]


def strip_artifact_sections(patch: str) -> tuple[str, list[str]]:
    """Return the patch without binary or oversized file sections, plus their paths.

    `git add -A` also stages whatever the agent's build and test runs left in
    /testbed. Shipping those costs more than disk: GNU patch cannot read a
    `GIT binary patch`, so a single compiled artifact strips the evaluator of
    its two apply fallbacks and turns any hiccup into an unresolved instance.
    """
    kept: list[str] = []
    dropped: list[str] = []
    for match in DIFF_SECTION.finditer(patch):
        section = match.group()
        if "\nGIT binary patch\n" in section or len(section) > ARTIFACT_SIZE_LIMIT:
            dropped.append(section.split("\n", 1)[0].split(" b/", 1)[-1])
        else:
            kept.append(section)
    return "".join(kept), dropped


def export_prediction(container: str, baseline_commit: str, instance_id: str) -> str:
    run_command(
        [
            "docker",
            "exec",
            container,
            "sh",
            "-lc",
            "cd /testbed && git add -A",
        ]
    )
    raw = run_command(
        [
            "docker",
            "exec",
            container,
            "git",
            "-C",
            "/testbed",
            "diff",
            "--cached",
            "--binary",
            "--no-ext-diff",
            baseline_commit,
        ]
    ).stdout
    patch, dropped = strip_artifact_sections(raw)
    if dropped:
        log(f"[{instance_id}] DROPPED_ARTIFACTS {' '.join(dropped)}")
    return patch


def save_partial_patch(
    container: str, baseline_commit: str, instance_dir: Path
) -> None:
    """Keep whatever a failed run changed, for diagnosis only, never as a prediction."""
    try:
        patch = export_prediction(container, baseline_commit, instance_dir.name)
    except Exception:
        return
    if patch.strip():
        atomic_write(instance_dir / "patch.partial.diff", patch)


def write_prediction(
    output_dir: Path,
    instance_id: str,
    model_name: str,
    patch: str,
) -> None:
    with OUTPUT_LOCK:
        path = output_dir / "preds.json"
        predictions = json.loads(path.read_text()) if path.is_file() else {}
        predictions[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": patch,
        }
        atomic_write(path, json.dumps(predictions, ensure_ascii=False, indent=2))


def nonempty_predictions(output_dir: Path) -> set[str]:
    path = output_dir / "preds.json"
    if not path.is_file():
        return set()
    records = json.loads(path.read_text())
    return {
        instance_id
        for instance_id, record in records.items()
        if str(record.get("model_patch") or "").strip()
    }


def testbed_has_leftovers(workspace: Path) -> bool:
    """True when the bind mount still has files the host may not be able to delete."""
    try:
        return workspace.exists() and (
            not workspace.is_dir() or any(workspace.iterdir())
        )
    except OSError:
        return True


def wipe_testbed(
    workspace: Path,
    *,
    image: str,
    container: str | None,
    instance_id: str,
) -> None:
    """Delete bind-mounted files as root so TemporaryDirectory cleanup cannot fail.

    Host-user rmtree cannot unlink root-owned test artifacts (for example Vue's
    ``test/tmp/tail/tail.pos``). Python 3.11's TemporaryDirectory then raises
    during chmod even with ignore_cleanup_errors=True, which would abort the
    whole worker pool after a successful prediction.
    """
    if container:
        log(f"[{instance_id}] WIPE_TESTBED")
        run_command(
            ["docker", "exec", container, "sh", "-lc", TESTBED_WIPE_SCRIPT],
            check=False,
        )
        return
    if not testbed_has_leftovers(workspace):
        return
    log(f"[{instance_id}] WIPE_TESTBED_FALLBACK")
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
            f"{workspace.resolve()}:/testbed",
            image,
            "-lc",
            TESTBED_WIPE_SCRIPT,
        ],
        check=False,
    )


def cleanup_temporary_directory(temporary_dir: tempfile.TemporaryDirectory, instance_id: str) -> None:
    """Never let host-side rmtree of root leftovers kill a finished instance."""
    try:
        temporary_dir.cleanup()
    except OSError as error:
        log(f"[{instance_id}] CLEANUP_WARN {type(error).__name__}: {error}")


def process_instance(
    *,
    agent: str,
    model: str,
    instance: dict,
    image: str,
    smell_patch: Path | None,
    output_dir: Path,
    timeout: int,
) -> Result:
    instance_id = instance["instance_id"]
    started_at = time.monotonic()
    container = (
        f"cli-eval-{agent}-{re.sub(r'[^a-zA-Z0-9_.-]', '-', instance_id).lower()}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    instance_dir = output_dir / instance_id
    reset_instance_dir(instance_dir)
    started = False
    baseline_commit = None
    status = "ERROR"
    detail = ""

    worker_root = smell_tmp_root()
    worker_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = tempfile.TemporaryDirectory(
        prefix=f"{agent}-swebench-",
        dir=str(worker_root),
        ignore_cleanup_errors=True,
    )
    try:
        workspace = Path(temporary_dir.name) / "testbed"
        try:
            log(f"[{instance_id}] preparing {image}")
            run_command(["docker", "image", "inspect", image])
            prepare_workspace(image, workspace)
            start_validation_container(image, workspace, container)
            started = True
            wait_for_container(container)
            test_patch = materialize_test_patch(
                instance, Path(temporary_dir.name), smell_patch
            )
            if test_patch is not None:
                log(f"[{instance_id}] APPLY_TEST {test_patch.name}")
            baseline_commit = prepare_baseline(container, smell_patch, test_patch)
            # The baseline is written by root inside the container; hand the bind
            # mount back before the agent edits it from the host.
            restore_workspace_ownership(container, instance_id)

            log(f"[{instance_id}] running {agent} model={model}")
            returncode = run_agent(
                agent,
                model,
                workspace,
                build_prompt(instance, container),
                timeout,
                instance_dir,
                isolated_root=Path(temporary_dir.name) / "agent-runtime",
            )
            if returncode != 0:
                raise RuntimeError(f"{agent} exited with code {returncode}")

            patch = export_prediction(container, baseline_commit, instance_id)
            atomic_write(instance_dir / "patch.diff", patch)
            write_prediction(output_dir, instance_id, f"{agent}/{model}", patch)
            status = "COMPLETED"
            detail = f"patch_bytes={len(patch.encode(UTF8))}"
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            atomic_write(instance_dir / "error.log", detail + "\n")
            if started and baseline_commit is not None:
                save_partial_patch(container, baseline_commit, instance_dir)
        finally:
            if started:
                restore_workspace_ownership(container, instance_id)
                wipe_testbed(
                    workspace, image=image, container=container, instance_id=instance_id
                )
                run_command(["docker", "rm", "-f", container], check=False)
            wipe_testbed(
                workspace, image=image, container=None, instance_id=instance_id
            )
    finally:
        cleanup_temporary_directory(temporary_dir, instance_id)

    result = Result(instance_id, status, detail, time.monotonic() - started_at)
    atomic_write(
        instance_dir / "result.json",
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
    )
    log(f"[{instance_id}] {status} {detail}")
    return result


def process_instance_with_retries(*, retries: int, **kwargs) -> Result:
    """Retry whole instances, since upstream API outages fail every concurrent run."""
    instance_id = kwargs["instance"]["instance_id"]
    for attempt in range(retries + 1):
        result = process_instance(**kwargs)
        if result.status == "COMPLETED" or attempt == retries:
            return result
        delay = RETRY_BACKOFF_SECONDS * (attempt + 1)
        log(
            f"[{instance_id}] retry {attempt + 1}/{retries} in {delay}s "
            f"after {result.detail}"
        )
        time.sleep(delay)
    return result


def write_summary(output_dir: Path, results: Sequence[Result]) -> None:
    """Merge into any previous summary so partial reruns keep the full picture."""
    path = output_dir / "summary.json"
    merged = {}
    if path.is_file():
        merged = {item["instance_id"]: item for item in json.loads(path.read_text())}
    merged.update({result.instance_id: asdict(result) for result in results})
    atomic_write(path, json.dumps(list(merged.values()), ensure_ascii=False, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, choices=["codex", "opencode"])
    parser.add_argument("--model", required=True, help="CLI model identifier")
    parser.add_argument("--instances", required=True, help="ID file or comma-separated IDs")
    parser.add_argument("--smell", type=Path, help="Optional directory of smell patches")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=positive_int, default=1)
    parser.add_argument("--timeout", type=positive_int, default=2_700)
    parser.add_argument(
        "--retries",
        type=non_negative_int,
        default=0,
        help="extra attempts per instance after a failure",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="rerun every ID in --instances, including nonempty preds.json patches",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip the credential and connectivity check",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if shutil.which(args.agent) is None:
        raise ValueError(f"{args.agent} is not installed or not on PATH")

    load_env_file()
    if not args.skip_preflight:
        preflight(args.agent, args.model)

    smell_dir = args.smell.resolve() if args.smell else None
    if smell_dir is not None and not smell_dir.is_dir():
        raise ValueError(f"smell directory does not exist: {smell_dir}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    instance_ids = parse_instances(args.instances)
    if not args.redo:
        completed = nonempty_predictions(output_dir)
        instance_ids = [item for item in instance_ids if item not in completed]
    if not instance_ids:
        log("No instances to run.")
        return 0

    instances = load_instances(instance_ids)
    images = load_images()
    missing_images = [item for item in instance_ids if item not in images]
    if missing_images:
        raise ValueError(f"images missing from image list: {', '.join(missing_images)}")
    smell_patches = (
        {
            instance_id: resolve_smell_patch(smell_dir, instance_id)
            for instance_id in instance_ids
        }
        if smell_dir is not None
        else {}
    )

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_instance_with_retries,
                retries=args.retries,
                agent=args.agent,
                model=args.model,
                instance=instances[instance_id],
                image=images[instance_id],
                smell_patch=smell_patches.get(instance_id),
                output_dir=output_dir,
                timeout=args.timeout,
            ): instance_id
            for instance_id in instance_ids
        }
        for future in as_completed(futures):
            results.append(future.result())

    write_summary(output_dir, results)
    failures = sum(result.status != "COMPLETED" for result in results)
    log(f"Finished {len(results)} instances: {len(results) - failures} completed, {failures} errors.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
