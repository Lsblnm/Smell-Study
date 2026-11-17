import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from swebench.harness.constants import KEY_PREDICTION
from swebench.harness.run_evaluation import (
    apply_refactored_gold_patches,
    apply_refactored_test_patches,
    resolve_refactored_patch,
    resolve_smell_patch,
)


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_cli_swebench.py"
SPEC = importlib.util.spec_from_file_location("run_cli_swebench", SCRIPT)
run_cli_swebench = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = run_cli_swebench
SPEC.loader.exec_module(run_cli_swebench)


def test_evaluator_resolves_base_patch_without_matching_variants(tmp_path):
    base = tmp_path / "repo__name-1-FE.patch"
    base.write_text("smell")
    (tmp_path / "repo__name-1-FE-gold.patch").write_text("gold")
    (tmp_path / "repo__name-1-FE-test.patch").write_text("test")

    assert resolve_smell_patch(tmp_path, "repo__name-1") == base.resolve()
    assert (
        resolve_refactored_patch(tmp_path, "repo__name-1", "gold")
        == (tmp_path / "repo__name-1-FE-gold.patch").resolve()
    )


def test_evaluator_rejects_multiple_base_smell_patches(tmp_path):
    (tmp_path / "repo__name-1-FE.patch").write_text("feature envy")
    (tmp_path / "repo__name-1-GC.patch").write_text("god class")

    with pytest.raises(ValueError, match="found 2"):
        resolve_smell_patch(tmp_path, "repo__name-1")


def test_evaluator_applies_optional_refactored_patches(tmp_path):
    (tmp_path / "repo__name-1-FE.patch").write_text("smell")
    (tmp_path / "repo__name-1-FE-gold.patch").write_text("adapted gold")
    (tmp_path / "repo__name-1-FE-test.patch").write_text("adapted test")
    instances = [{"instance_id": "repo__name-1", "test_patch": "original test"}]
    predictions = {
        "repo__name-1": {
            "instance_id": "repo__name-1",
            KEY_PREDICTION: "original gold",
        }
    }

    adapted = apply_refactored_test_patches(instances, tmp_path)
    apply_refactored_gold_patches(predictions, tmp_path)

    assert adapted[0]["test_patch"] == "adapted test"
    assert instances[0]["test_patch"] == "original test"
    assert predictions["repo__name-1"][KEY_PREDICTION] == "adapted gold"


def test_cli_runner_folds_image_build_fixes_into_the_baseline_without_smell(monkeypatch):
    commands = []

    def fake_run_command(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="baseline-commit\n")

    monkeypatch.setattr(run_cli_swebench, "run_command", fake_run_command)

    assert run_cli_swebench.prepare_baseline("container", None) == "baseline-commit"
    baseline_command = commands[-1][-1]
    assert baseline_command.startswith("cd /testbed && git add -A && ")
    assert "commit-tree" in baseline_command
    assert "git apply" not in baseline_command


def test_cli_runner_applies_smell_when_requested(tmp_path, monkeypatch):
    smell_patch = tmp_path / "instance-FE.patch"
    smell_patch.write_text("smell")
    commands = []

    def fake_run_command(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="baseline-commit\n")

    monkeypatch.setattr(run_cli_swebench, "run_command", fake_run_command)

    assert (
        run_cli_swebench.prepare_baseline("container", smell_patch) == "baseline-commit"
    )
    assert commands[0][:2] == ["docker", "cp"]
    baseline_command = commands[-1][-1]
    assert "git apply --index /tmp/smell.patch" in baseline_command


def test_cli_runner_applies_dataset_test_patch_without_smell(tmp_path, monkeypatch):
    test_patch = tmp_path / "eval-test.patch"
    test_patch.write_text("original test")
    commands = []

    def fake_run_command(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="baseline-commit\n")

    monkeypatch.setattr(run_cli_swebench, "run_command", fake_run_command)
    run_cli_swebench.prepare_baseline("container", None, test_patch)

    assert commands[0][:2] == ["docker", "cp"]
    assert str(test_patch) in commands[0]
    baseline_command = commands[-1][-1]
    assert "git apply --index /tmp/eval-test.patch" in baseline_command
    assert "/tmp/smell.patch" not in baseline_command


def test_cli_runner_applies_smell_before_test_patch(tmp_path, monkeypatch):
    smell_patch = tmp_path / "instance-FE.patch"
    smell_patch.write_text("smell")
    test_patch = tmp_path / "instance-FE-test.patch"
    test_patch.write_text("adapted test")
    commands = []

    def fake_run_command(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="baseline-commit\n")

    monkeypatch.setattr(run_cli_swebench, "run_command", fake_run_command)
    run_cli_swebench.prepare_baseline("container", smell_patch, test_patch)

    copies = [command for command in commands if command[:2] == ["docker", "cp"]]
    assert copies[0][2] == str(smell_patch)
    assert copies[1][2] == str(test_patch)
    baseline_command = commands[-1][-1]
    assert baseline_command.index("/tmp/smell.patch") < baseline_command.index(
        "/tmp/eval-test.patch"
    )


def test_materialize_test_patch_prefers_adapted_file(tmp_path):
    smell_patch = tmp_path / "repo__name-1-FE.patch"
    smell_patch.write_text("smell")
    adapted = tmp_path / "repo__name-1-FE-test.patch"
    adapted.write_text("adapted test")
    instance = {"instance_id": "repo__name-1", "test_patch": "original test"}

    assert (
        run_cli_swebench.materialize_test_patch(instance, tmp_path / "out", smell_patch)
        == adapted.resolve()
    )


def test_materialize_test_patch_falls_back_to_dataset(tmp_path):
    smell_patch = tmp_path / "repo__name-1-FE.patch"
    smell_patch.write_text("smell")
    instance = {"instance_id": "repo__name-1", "test_patch": "original test"}
    dest = tmp_path / "out"
    dest.mkdir()

    path = run_cli_swebench.materialize_test_patch(instance, dest, smell_patch)
    assert path == dest / "eval-test.patch"
    assert path.read_text() == "original test"


def test_materialize_test_patch_skips_empty_dataset_patch(tmp_path):
    instance = {"instance_id": "repo__name-1", "test_patch": ""}
    assert (
        run_cli_swebench.materialize_test_patch(instance, tmp_path, None) is None
    )


def test_cli_runner_prompt_allows_evaluation_tests_but_not_gold():
    prompt = run_cli_swebench.build_prompt(
        {"problem_statement": "bug"}, "container"
    )
    assert "FAIL_TO_PASS" in prompt
    assert "Do not modify test files" in prompt
    assert "gold patch" in prompt
    assert "or test patch" not in prompt


def test_cli_runner_creates_baseline_without_hooks(tmp_path, monkeypatch):
    smell_patch = tmp_path / "instance-FE.patch"
    smell_patch.write_text("smell")
    commands = []

    def fake_run_command(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="baseline-commit\n")

    monkeypatch.setattr(run_cli_swebench, "run_command", fake_run_command)

    assert (
        run_cli_swebench.prepare_baseline("container", smell_patch) == "baseline-commit"
    )
    baseline_command = commands[-1][-1]
    assert "commit-tree" in baseline_command
    assert "git reset --hard" in baseline_command
    assert "commit --amend" not in baseline_command


def test_cli_runner_reconciles_smell_hunks_already_in_workspace(tmp_path, monkeypatch):
    smell_patch = tmp_path / "instance-FE.patch"
    smell_patch.write_text("smell")
    commands = []

    def fake_run_command(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="baseline-commit\n")

    monkeypatch.setattr(run_cli_swebench, "run_command", fake_run_command)
    run_cli_swebench.prepare_baseline("container", smell_patch)

    baseline_command = commands[-1][-1]
    assert baseline_command.startswith("cd /testbed && git add -A && ")
    assert "git apply --3way --index" in baseline_command
    assert "git ls-files -u" in baseline_command
    assert "git diff --check" not in baseline_command


def test_cli_runner_waits_until_testbed_is_a_git_work_tree(monkeypatch):
    probes = iter(
        [
            SimpleNamespace(returncode=128, stdout="", stderr="dubious ownership"),
            SimpleNamespace(returncode=0, stdout="true\n", stderr=""),
        ]
    )

    def fake_run_command(command, **kwargs):
        if "rev-parse" in command:
            return next(probes)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_cli_swebench, "run_command", fake_run_command)
    monkeypatch.setattr(run_cli_swebench.time, "sleep", lambda seconds: None)

    run_cli_swebench.wait_for_container("container")
    assert next(probes, None) is None


def test_cli_runner_reports_container_that_never_becomes_a_repo(monkeypatch):
    monkeypatch.setattr(
        run_cli_swebench,
        "run_command",
        lambda command, **kwargs: SimpleNamespace(
            returncode=128, stdout="", stderr="not a git repository"
        ),
    )
    monkeypatch.setattr(run_cli_swebench.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(run_cli_swebench, "CONTAINER_READY_TIMEOUT", 0)

    with pytest.raises(RuntimeError, match="not a git work tree"):
        run_cli_swebench.wait_for_container("container")


def test_cli_runner_keeps_trajectory_when_the_agent_times_out(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_cli_swebench,
        "agent_command",
        lambda agent, model, workspace, prompt: (
            ["sh", "-c", "echo streamed; sleep 30"],
            None,
        ),
    )

    with pytest.raises(TimeoutError, match="exceeded 1 seconds"):
        run_cli_swebench.run_agent("codex", "model", tmp_path, "fix it", 1, tmp_path)

    assert (tmp_path / "trajectory.jsonl").read_text() == "streamed\n"
    assert (tmp_path / "prompt.txt").read_text() == "fix it"


def test_cli_runner_clears_artifacts_from_earlier_attempts(tmp_path):
    (tmp_path / "patch.diff").write_text("stale patch")
    (tmp_path / "trajectory.jsonl").write_text("stale trajectory")
    (tmp_path / "keep.txt").write_text("unrelated")

    run_cli_swebench.reset_instance_dir(tmp_path)

    assert not (tmp_path / "patch.diff").exists()
    assert not (tmp_path / "trajectory.jsonl").exists()
    assert (tmp_path / "keep.txt").exists()


def test_cli_runner_drops_binary_and_oversized_sections_from_a_patch():
    fix = (
        "diff --git a/router.go b/router.go\n"
        "--- a/router.go\n+++ b/router.go\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    binary = (
        "diff --git a/gin.test b/gin.test\n"
        "new file mode 100755\n"
        "index 0000000..ca8c3af\n"
        "GIT binary patch\nliteral 12\nzabc\n\n"
    )
    huge = (
        "diff --git a/generated.json b/generated.json\n"
        "--- a/generated.json\n+++ b/generated.json\n"
        "@@ -1 +1 @@\n" + "+x" * run_cli_swebench.ARTIFACT_SIZE_LIMIT + "\n"
    )

    patch, dropped = run_cli_swebench.strip_artifact_sections(binary + fix + huge)

    assert patch == fix
    assert dropped == ["gin.test", "generated.json"]


def test_cli_runner_keeps_a_patch_without_artifacts_byte_for_byte():
    fix = (
        "diff --git a/a.go b/a.go\n--- a/a.go\n+++ b/a.go\n@@ -1 +1 @@\n-x\n+y\n"
        "diff --git a/b.go b/b.go\n--- a/b.go\n+++ b/b.go\n@@ -1 +1 @@\n-p\n+q\n"
    )

    assert run_cli_swebench.strip_artifact_sections(fix) == (fix, [])


def test_trajectory_agent_seconds_uses_first_and_last_event_stamps(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    path.write_text(
        '{"type":"step_start","timestamp":1794186576362}\n'
        '{"type":"tool_use","timestamp":1794187000000}\n'
        '{"type":"step_finish","timestamp":1794187583517}\n'
    )

    assert run_cli_swebench.trajectory_agent_seconds(path) == pytest.approx(1007.155)
    assert run_cli_swebench.trajectory_agent_seconds(tmp_path / "missing.jsonl") is None


def test_cli_runner_merges_summary_across_reruns(tmp_path):
    (tmp_path / "summary.json").write_text(
        json.dumps(
            [
                {
                    "instance_id": "kept",
                    "status": "COMPLETED",
                    "detail": "patch_bytes=1",
                    "agent_seconds": 1.0,
                },
                {
                    "instance_id": "retried",
                    "status": "ERROR",
                    "detail": "TimeoutError",
                    "agent_seconds": 2.0,
                },
            ]
        )
    )

    run_cli_swebench.write_summary(
        tmp_path,
        [run_cli_swebench.Result("retried", "COMPLETED", "patch_bytes=3", 3.0)],
    )

    summary = {
        item["instance_id"]: item
        for item in json.loads((tmp_path / "summary.json").read_text())
    }
    assert summary["kept"]["status"] == "COMPLETED"
    assert summary["retried"]["status"] == "COMPLETED"


def test_cli_runner_loads_credentials_from_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nIKUNCODE_API_KEY='from-file'\nEMPTY\n")
    monkeypatch.delenv("IKUNCODE_API_KEY", raising=False)

    run_cli_swebench.load_env_file(env_file)

    assert os.environ["IKUNCODE_API_KEY"] == "from-file"


def test_cli_runner_keeps_an_exported_key_over_the_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("IKUNCODE_API_KEY=from-file\n")
    monkeypatch.setenv("IKUNCODE_API_KEY", "from-shell")

    run_cli_swebench.load_env_file(env_file)

    assert os.environ["IKUNCODE_API_KEY"] == "from-shell"


def test_cli_runner_reads_provider_details_from_agent_configs(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    (codex_dir / "config.toml").write_text(
        '[model_providers.ikuncode]\n'
        'base_url = "https://relay.example/v1"\n'
        'env_key = "RELAY_KEY"\n'
        '\n[projects."/tmp/x"]\ntrust_level = "trusted"\n'
    )
    opencode_dir = tmp_path / ".config" / "opencode"
    opencode_dir.mkdir(parents=True)
    (opencode_dir / "opencode.json").write_text(
        json.dumps(
            {
                "provider": {
                    "ikuncode": {
                        "options": {
                            "baseURL": "https://relay.example/v1",
                            "apiKey": "{env:RELAY_KEY}",
                        }
                    }
                }
            }
        )
    )

    assert run_cli_swebench.codex_provider()[:2] == (
        "https://relay.example/v1",
        "RELAY_KEY",
    )
    assert run_cli_swebench.opencode_provider()[:2] == (
        "https://relay.example/v1",
        "RELAY_KEY",
    )


def test_cli_runner_reads_v2_opencode_providers(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    opencode_dir = tmp_path / ".config" / "opencode"
    opencode_dir.mkdir(parents=True)
    (opencode_dir / "opencode.json").write_text(
        json.dumps(
            {
                "providers": {
                    "ikuncode": {
                        "package": "@opencode/ai/providers/openai-compatible",
                        "settings": {
                            "baseURL": "https://relay.example/v1",
                            "apiKey": "{env:RELAY_KEY}",
                        },
                    }
                }
            }
        )
    )

    assert run_cli_swebench.opencode_provider()[:2] == (
        "https://relay.example/v1",
        "RELAY_KEY",
    )


def test_cli_runner_preflight_rejects_a_missing_key(monkeypatch):
    monkeypatch.setattr(
        run_cli_swebench,
        "opencode_provider",
        lambda: ("https://relay.example/v1", "RELAY_KEY", Path("/cfg")),
    )
    monkeypatch.delenv("RELAY_KEY", raising=False)

    with pytest.raises(ValueError, match="which is not set"):
        run_cli_swebench.preflight("opencode", "ikuncode/model-a")


def test_cli_runner_preflight_rejects_an_unavailable_model(monkeypatch):
    monkeypatch.setattr(
        run_cli_swebench,
        "opencode_provider",
        lambda: ("https://relay.example/v1", "RELAY_KEY", Path("/cfg")),
    )
    monkeypatch.setenv("RELAY_KEY", "secret")
    monkeypatch.setattr(
        run_cli_swebench, "available_models", lambda url, key: {"model-b"}
    )

    with pytest.raises(ValueError, match="does not serve model 'model-a'"):
        run_cli_swebench.preflight("opencode", "ikuncode/model-a")


def test_cli_runner_preflight_accepts_a_working_relay(monkeypatch):
    monkeypatch.setattr(
        run_cli_swebench,
        "codex_provider",
        lambda: ("https://relay.example/v1", "RELAY_KEY", Path("/cfg")),
    )
    monkeypatch.setenv("RELAY_KEY", "secret")
    monkeypatch.setattr(
        run_cli_swebench, "available_models", lambda url, key: {"model-a", "model-b"}
    )

    run_cli_swebench.preflight("codex", "model-a")


def test_cli_runner_wipes_root_testbed_before_tempdir_cleanup(tmp_path, monkeypatch):
    commands = []

    def fake_run_command(command, **kwargs):
        commands.append(list(command))
        return SimpleNamespace(stdout="baseline\n", returncode=0)

    def fake_prepare_workspace(image, workspace):
        workspace.mkdir(parents=True)
        (workspace / "tail.pos").write_text("root leftover")

    monkeypatch.setattr(run_cli_swebench, "smell_tmp_root", lambda: tmp_path / "tmp")
    monkeypatch.setattr(run_cli_swebench, "run_command", fake_run_command)
    monkeypatch.setattr(run_cli_swebench, "prepare_workspace", fake_prepare_workspace)
    monkeypatch.setattr(run_cli_swebench, "start_validation_container", lambda *a, **k: None)
    monkeypatch.setattr(run_cli_swebench, "wait_for_container", lambda *a, **k: None)
    monkeypatch.setattr(run_cli_swebench, "restore_workspace_ownership", lambda *a, **k: None)
    monkeypatch.setattr(run_cli_swebench, "run_agent", lambda *a, **k: 0)
    monkeypatch.setattr(run_cli_swebench, "export_prediction", lambda *a, **k: "diff")
    monkeypatch.setattr(run_cli_swebench, "prepare_baseline", lambda *a, **k: "abc123")

    result = run_cli_swebench.process_instance(
        agent="opencode",
        model="model-a",
        instance={
            "instance_id": "vuejs__core-11739",
            "problem_statement": "fix it",
        },
        image="swebench/vuejs",
        smell_patch=None,
        output_dir=tmp_path / "out",
        timeout=10,
    )

    assert result.status == "COMPLETED"
    wipe_indexes = [
        index
        for index, command in enumerate(commands)
        if command[:2] == ["docker", "exec"]
        and command[-1] == run_cli_swebench.TESTBED_WIPE_SCRIPT
    ]
    rm_indexes = [
        index
        for index, command in enumerate(commands)
        if command[:3] == ["docker", "rm", "-f"]
    ]
    fallback_indexes = [
        index
        for index, command in enumerate(commands)
        if command[:2] == ["docker", "run"]
        and command[-1] == run_cli_swebench.TESTBED_WIPE_SCRIPT
    ]
    assert wipe_indexes, "expected docker exec wipe while the container is alive"
    assert rm_indexes
    assert wipe_indexes[0] < rm_indexes[0]
    assert fallback_indexes
    assert rm_indexes[0] < fallback_indexes[0]


def test_cli_runner_keeps_completed_result_if_tempdir_cleanup_fails(
    tmp_path, monkeypatch
):
    real_temporary_directory = run_cli_swebench.tempfile.TemporaryDirectory

    class BoomTemporaryDirectory(real_temporary_directory):
        def cleanup(self):
            try:
                super().cleanup()
            except OSError:
                pass
            raise PermissionError(
                "[Errno 1] Operation not permitted: "
                "'/tmp/opencode-swebench/testbed/test/tmp/tail'"
            )

    monkeypatch.setattr(
        run_cli_swebench.tempfile, "TemporaryDirectory", BoomTemporaryDirectory
    )
    monkeypatch.setattr(run_cli_swebench, "smell_tmp_root", lambda: tmp_path / "tmp")
    monkeypatch.setattr(
        run_cli_swebench,
        "run_command",
        lambda *a, **k: SimpleNamespace(stdout="", returncode=0),
    )
    monkeypatch.setattr(
        run_cli_swebench, "prepare_workspace", lambda image, workspace: workspace.mkdir()
    )
    monkeypatch.setattr(run_cli_swebench, "start_validation_container", lambda *a, **k: None)
    monkeypatch.setattr(run_cli_swebench, "wait_for_container", lambda *a, **k: None)
    monkeypatch.setattr(run_cli_swebench, "restore_workspace_ownership", lambda *a, **k: None)

    def fake_run_agent(agent, model, workspace, prompt, timeout, instance_dir, isolated_root=None):
        (instance_dir / "trajectory.jsonl").write_text(
            '{"type":"step_start","timestamp":1000000000000}\n'
            '{"type":"step_finish","timestamp":1000000123500}\n'
        )
        return 0

    monkeypatch.setattr(run_cli_swebench, "run_agent", fake_run_agent)
    monkeypatch.setattr(run_cli_swebench, "export_prediction", lambda *a, **k: "diff")
    monkeypatch.setattr(run_cli_swebench, "prepare_baseline", lambda *a, **k: "abc123")
    monkeypatch.setattr(run_cli_swebench, "wipe_testbed", lambda *a, **k: None)

    result = run_cli_swebench.process_instance(
        agent="opencode",
        model="model-a",
        instance={
            "instance_id": "vuejs__core-11739",
            "problem_statement": "fix it",
        },
        image="swebench/vuejs",
        smell_patch=None,
        output_dir=tmp_path / "out",
        timeout=10,
    )

    assert result.status == "COMPLETED"
    assert result.agent_seconds == pytest.approx(123.5)
    saved = json.loads(
        (tmp_path / "out" / "vuejs__core-11739" / "result.json").read_text()
    )
    assert saved["agent_seconds"] == pytest.approx(123.5)
    assert "elapsed_seconds" not in saved
    assert (tmp_path / "out" / "preds.json").is_file()


def test_cli_runner_retries_until_an_attempt_completes(monkeypatch):
    statuses = iter(["ERROR", "COMPLETED"])
    attempts = []

    def fake_process_instance(**kwargs):
        attempts.append(kwargs["instance"]["instance_id"])
        return run_cli_swebench.Result("repo__name-1", next(statuses), "", 1.0)

    monkeypatch.setattr(run_cli_swebench, "process_instance", fake_process_instance)
    monkeypatch.setattr(run_cli_swebench.time, "sleep", lambda seconds: None)

    result = run_cli_swebench.process_instance_with_retries(
        retries=2, instance={"instance_id": "repo__name-1"}
    )

    assert result.status == "COMPLETED"
    assert len(attempts) == 2


def test_cli_runner_requires_instances_and_makes_smell_optional(tmp_path):
    args = run_cli_swebench.make_parser().parse_args(
        [
            "--agent",
            "codex",
            "--model",
            "model",
            "--instances",
            "ids-fe.txt",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert args.instances == "ids-fe.txt"
    assert args.smell is None


def test_prepare_worker_opencode_env_copies_auth_not_sqlite(tmp_path, monkeypatch):
    source = tmp_path / "share" / "opencode"
    source.mkdir(parents=True)
    (source / "auth.json").write_text('{"token": "x"}')
    (source / "opencode.db").write_text("db")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))

    env = run_cli_swebench.prepare_worker_opencode_env(tmp_path / "isolated")

    data = Path(env["XDG_DATA_HOME"]) / "opencode"
    assert (data / "auth.json").read_text() == '{"token": "x"}'
    assert not (data / "opencode.db").exists()
    assert env["XDG_STATE_HOME"] == str(tmp_path / "isolated" / "state")


def test_agent_environment_isolates_codex_and_opencode_homes(tmp_path, monkeypatch):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(
        run_cli_swebench,
        "prepare_worker_codex_home",
        lambda destination, source=None: destination,
    )

    codex_env = run_cli_swebench.agent_environment("codex", tmp_path / "codex-root")
    opencode_env = run_cli_swebench.agent_environment("opencode", tmp_path / "oc-root")

    assert Path(codex_env["CODEX_HOME"]) == tmp_path / "codex-root" / "codex"
    assert Path(opencode_env["XDG_DATA_HOME"]) == tmp_path / "oc-root" / "opencode" / "share"
    assert "CODEX_HOME" not in opencode_env or opencode_env.get("CODEX_HOME") == os.environ.get(
        "CODEX_HOME"
    )


def test_cli_runner_builds_noninteractive_agent_commands(tmp_path):
    codex, codex_input = run_cli_swebench.agent_command(
        "codex", "gpt-codex", tmp_path, "fix it"
    )
    opencode, opencode_input = run_cli_swebench.agent_command(
        "opencode", "openai/gpt", tmp_path, "fix it"
    )

    assert codex[:2] == ["codex", "exec"]
    assert "--json" in codex
    assert "--ask-for-approval" not in codex
    assert codex[codex.index("--config") + 1] == "model_reasoning_effort=high"
    assert codex_input == "fix it"
    assert opencode[:2] == ["opencode", "run"]
    assert opencode[opencode.index("--variant") + 1] == "high"
    assert opencode[-1] == "fix it"
    assert opencode_input is None
