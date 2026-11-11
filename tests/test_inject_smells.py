import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "inject_smells.py"
SPEC = importlib.util.spec_from_file_location("inject_smells", SCRIPT)
inject_smells = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = inject_smells
SPEC.loader.exec_module(inject_smells)


def test_parse_instances_from_file_and_deduplicate(tmp_path):
    instances = tmp_path / "instances.txt"
    instances.write_text("# batch\na__repo-1\nb__repo-2\na__repo-1\n")

    assert inject_smells.parse_instances(str(instances)) == ["a__repo-1", "b__repo-2"]


def test_parse_comma_separated_instances():
    assert inject_smells.parse_instances("a__repo-1, b__repo-2,a__repo-1") == [
        "a__repo-1",
        "b__repo-2",
    ]


def test_smell_tmp_root_defaults_to_project_tmp(monkeypatch):
    monkeypatch.delenv("SMELL_INJECT_TMPDIR", raising=False)
    assert inject_smells.smell_tmp_root() == inject_smells.ROOT / ".tmp"


def test_smell_tmp_root_respects_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SMELL_INJECT_TMPDIR", str(tmp_path / "custom"))
    assert inject_smells.smell_tmp_root() == tmp_path / "custom"


def test_configure_temp_root_points_python_tmpdir_at_big_disk(monkeypatch, tmp_path):
    monkeypatch.setenv("SMELL_INJECT_TMPDIR", str(tmp_path / "custom"))
    root = inject_smells.configure_temp_root()
    assert root == tmp_path / "custom"
    assert root.is_dir()
    assert inject_smells.LOCK_ROOT == root / "inject-locks"
    assert os.environ["TMPDIR"] == str(root)


@pytest.mark.parametrize(
    ("filename", "name", "abbreviation"),
    [
        ("LongMethod.md", "LongMethod", "LM"),
        ("GodClass.md", "GodClass", "GC"),
        ("FeatureEnvy.md", "FeatureEnvy", "FE"),
    ],
)
def test_resolve_smell(tmp_path, filename, name, abbreviation):
    prompt = tmp_path / filename
    prompt.write_text("prompt")

    smell = inject_smells.resolve_smell(str(prompt))

    assert smell.name == name
    assert smell.abbreviation == abbreviation


def test_positive_parallelism():
    assert inject_smells.positive_int("5") == 5
    with pytest.raises(Exception):
        inject_smells.positive_int("0")


def test_parser_defaults_to_codex():
    parser = inject_smells.make_parser()
    default = parser.parse_args(["prompts/FeatureEnvy.md", "ids-fe.txt", "5"])
    assert default.agent == "codex"
    assert default.patch_root is None
    assert default.intro_root is None
    assert default.ids is None
    selected = parser.parse_args(
        ["--agent", "codex", "prompts/FeatureEnvy.md", "ids-fe.txt", "5"]
    )
    assert selected.agent == "codex"


def test_parser_accepts_separate_patch_and_intro_roots(tmp_path):
    args = inject_smells.make_parser().parse_args(
        [
            "--agent",
            "codex",
            "--patch-root",
            str(tmp_path / "patches"),
            "--intro-root",
            str(tmp_path / "intros"),
            "prompts/LongMethod.md",
            "idfs-300.txt",
            "20",
        ]
    )
    assert args.patch_root == tmp_path / "patches"
    assert args.intro_root == tmp_path / "intros"


def test_parser_accepts_ids_file(tmp_path):
    ids = tmp_path / "ids-fe.txt"
    args = inject_smells.make_parser().parse_args(
        [
            "--ids",
            str(ids),
            "prompts/FeatureEnvy.md",
            "idfs-300.txt",
            "10",
        ]
    )
    assert args.ids == ids


def test_parse_ids_file_and_restrict_instances(tmp_path):
    ids = tmp_path / "ids-fe.txt"
    ids.write_text("# subset\nb__repo-2\nc__repo-3\na__repo-1\n")

    assert inject_smells.parse_ids_file(ids) == [
        "b__repo-2",
        "c__repo-3",
        "a__repo-1",
    ]
    assert inject_smells.restrict_instances(
        ["a__repo-1", "b__repo-2", "d__repo-4"],
        ["b__repo-2", "c__repo-3", "a__repo-1"],
    ) == ["b__repo-2", "a__repo-1"]
    with pytest.raises(ValueError, match="ids file does not exist"):
        inject_smells.parse_ids_file(tmp_path / "missing.txt")
    with pytest.raises(ValueError, match="no instances from --ids"):
        inject_smells.restrict_instances(["a__repo-1"], ["c__repo-3"])


def test_configure_output_roots_keeps_project_layout_by_default(monkeypatch):
    monkeypatch.setattr(inject_smells, "PATCH_ROOT", Path("/old/smell-patch"))
    monkeypatch.setattr(inject_smells, "INTRO_ROOT", Path("/old/smell-patch-intro"))
    patch_root, intro_root = inject_smells.configure_output_roots()
    assert patch_root == inject_smells.ROOT / "smell-patch"
    assert intro_root == inject_smells.ROOT / "smell-patch-intro"


def test_configure_output_roots_sets_patch_and_intro_independently(tmp_path, monkeypatch):
    monkeypatch.setattr(inject_smells, "PATCH_ROOT", inject_smells.PATCH_ROOT)
    monkeypatch.setattr(inject_smells, "INTRO_ROOT", inject_smells.INTRO_ROOT)
    patch_root, intro_root = inject_smells.configure_output_roots(
        tmp_path / "patches", None
    )
    assert patch_root == (tmp_path / "patches").resolve()
    assert intro_root == inject_smells.ROOT / "smell-patch-intro"

    patch_root, intro_root = inject_smells.configure_output_roots(
        None, tmp_path / "intros"
    )
    assert patch_root == inject_smells.ROOT / "smell-patch"
    assert intro_root == (tmp_path / "intros").resolve()

    patch_root, intro_root = inject_smells.configure_output_roots(
        tmp_path / "patches", tmp_path / "intros"
    )
    smell = inject_smells.Smell("LongMethod", "LM", tmp_path / "LongMethod.md")
    paths = inject_smells.output_paths(smell, "a__repo-1")
    assert paths.patch == patch_root / "LongMethod" / "a__repo-1-LM.patch"
    assert paths.intro == intro_root / "LongMethod" / "a__repo-1-LM.json"


def test_prepare_worker_codex_home_copies_config_not_sqlite(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "config.toml").write_text("model = 'x'\n")
    (source / "auth.json").write_text("{}\n")
    (source / "state_5.sqlite").write_text("db")
    (source / "logs_2.sqlite").write_text("logs")
    destination = tmp_path / "dest"

    home = inject_smells.prepare_worker_codex_home(destination, source)

    assert home == destination
    assert (home / "config.toml").read_text() == "model = 'x'\n"
    assert (home / "auth.json").read_text() == "{}\n"
    assert not (home / "state_5.sqlite").exists()
    assert not (home / "logs_2.sqlite").exists()


def test_codex_api_key_reads_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("IKUNCODE_API_KEY=from-file\n")
    monkeypatch.setattr(inject_smells, "ENV_FILE", env_file)
    monkeypatch.delenv("IKUNCODE_API_KEY", raising=False)
    assert inject_smells.find_codex_api_key() == "from-file"

    monkeypatch.delenv("IKUNCODE_API_KEY", raising=False)
    monkeypatch.setattr(inject_smells, "ENV_FILE", tmp_path / "missing.env")
    with pytest.raises(ValueError, match="IKUNCODE_API_KEY"):
        inject_smells.find_codex_api_key()


def test_codex_grok_provider_reads_grok_key(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("IKUNCODE_GROK_API_KEY=grok-from-file\n")
    monkeypatch.setattr(inject_smells, "ENV_FILE", env_file)
    monkeypatch.delenv("IKUNCODE_GROK_API_KEY", raising=False)
    monkeypatch.delenv("IKUNCODE_API_KEY", raising=False)
    monkeypatch.setenv("CODEX_MODEL_PROVIDER", "ikuncode_grok")

    assert inject_smells.find_codex_api_key("ikuncode_grok") == "grok-from-file"
    with pytest.raises(ValueError, match="IKUNCODE_API_KEY"):
        inject_smells.find_codex_api_key("ikuncode")


def test_codex_model_defaults_and_rejects_invalid_effort(monkeypatch):
    monkeypatch.delenv("CODEX_MODEL", raising=False)
    monkeypatch.delenv("CODEX_MODEL_EFFORT", raising=False)
    monkeypatch.delenv("CODEX_MODEL_PROVIDER", raising=False)
    assert inject_smells.codex_model_selection() == (
        "gpt-5.6-sol",
        "high",
        "ikuncode",
    )
    assert inject_smells.codex_model_description() == "ikuncode/gpt-5.6-sol(effort=high)"

    monkeypatch.setenv("CODEX_MODEL", "gpt-5.6-terra")
    monkeypatch.setenv("CODEX_MODEL_EFFORT", "medium")
    assert inject_smells.codex_model_selection() == (
        "gpt-5.6-terra",
        "medium",
        "ikuncode",
    )

    monkeypatch.setenv("CODEX_MODEL_EFFORT", "max")
    with pytest.raises(ValueError, match="CODEX_MODEL_EFFORT"):
        inject_smells.codex_model_selection()


def _install_fake_codex(monkeypatch, captured, thread):
    class FakeCodex:
        def __init__(self, config=None):
            captured["codex_config"] = config

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def thread_start(self, **kwargs):
            captured["start"] = kwargs
            return thread

    class FakeCodexConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "openai_codex",
        types.SimpleNamespace(
            Codex=FakeCodex,
            CodexConfig=FakeCodexConfig,
            ApprovalMode=types.SimpleNamespace(deny_all="deny_all"),
            Sandbox=types.SimpleNamespace(full_access="full-access"),
        ),
    )


def test_codex_agent_uses_requested_workspace(tmp_path, monkeypatch):
    captured = {}
    messages = []

    class FakeThread:
        def run(self, prompt, **kwargs):
            messages.append(prompt)
            return types.SimpleNamespace(
                status="completed",
                error=None,
                final_response='{"Status": "SUCCESS"}',
            )

    _install_fake_codex(monkeypatch, captured, FakeThread())
    monkeypatch.setenv("SMELL_INJECT_TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.delenv("CODEX_MODEL", raising=False)
    monkeypatch.delenv("CODEX_MODEL_EFFORT", raising=False)
    monkeypatch.delenv("CODEX_MODEL_PROVIDER", raising=False)

    assert inject_smells.run_codex_agent("inject", tmp_path) == '{"Status": "SUCCESS"}'
    assert messages == ["inject"]
    assert captured["start"]["cwd"] == str(tmp_path)
    assert captured["start"]["model"] == "gpt-5.6-sol"
    assert captured["start"]["model_provider"] == "ikuncode"
    assert captured["start"]["sandbox"] == "full-access"
    assert captured["start"]["approval_mode"] == "deny_all"
    assert captured["start"]["ephemeral"] is True
    assert captured["start"]["config"] == {"model_reasoning_effort": "high"}
    assert captured["codex_config"].config_overrides == inject_smells.CODEX_RUNTIME_OVERRIDES
    isolated_home = Path(captured["codex_config"].env["CODEX_HOME"])
    assert isolated_home.parent == tmp_path / "tmp"
    assert isolated_home.resolve() != inject_smells.default_codex_home().resolve()


def test_codex_agent_continues_until_final_status(tmp_path, monkeypatch):
    messages = []
    reports = iter(["先分析代码结构。", '{"Status": "SUCCESS"}'])

    class FakeThread:
        def run(self, prompt, **kwargs):
            messages.append(prompt)
            return types.SimpleNamespace(
                status="completed",
                error=None,
                final_response=next(reports),
            )

    _install_fake_codex(monkeypatch, {}, FakeThread())
    monkeypatch.setenv("SMELL_INJECT_TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.delenv("CODEX_MODEL", raising=False)
    monkeypatch.delenv("CODEX_MODEL_EFFORT", raising=False)
    monkeypatch.delenv("CODEX_MODEL_PROVIDER", raising=False)

    assert (
        inject_smells.run_codex_agent("inject", tmp_path, "a__repo-1")
        == '{"Status": "SUCCESS"}'
    )
    assert messages == ["inject", inject_smells.CONTINUE_PROMPT]


def test_codex_agent_rejects_incomplete_reports(tmp_path, monkeypatch):
    class FakeThread:
        def run(self, prompt, **kwargs):
            return types.SimpleNamespace(
                status="completed",
                error=None,
                final_response="继续分析。",
            )

    _install_fake_codex(monkeypatch, {}, FakeThread())
    monkeypatch.setenv("SMELL_INJECT_TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.delenv("CODEX_MODEL", raising=False)
    monkeypatch.delenv("CODEX_MODEL_EFFORT", raising=False)
    monkeypatch.delenv("CODEX_MODEL_PROVIDER", raising=False)

    with pytest.raises(inject_smells.CommandError, match="after 3 turns"):
        inject_smells.run_codex_agent("inject", tmp_path)


def test_validation_container_allows_mounted_git_workspace(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(inject_smells, "run_command", fake_run)
    inject_smells.start_validation_container(
        "image:latest", tmp_path, "validation-container"
    )

    assert "git config --global --add safe.directory /testbed" in calls[0][-1]
    assert f"{tmp_path}:/testbed" in calls[0]


def test_load_instances_filters_dataset(tmp_path, monkeypatch):
    def fake_load_dataset(dataset_name, split):
        assert dataset_name == inject_smells.DATASET_NAME
        assert split == inject_smells.DATASET_SPLIT
        return [
            {"instance_id": "a__repo-1", "patch": "a"},
            {"instance_id": "b__repo-2", "patch": "b"},
        ]

    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset)
    )
    monkeypatch.setattr(inject_smells, "DATASET_CACHE", tmp_path / "dataset.json")

    assert inject_smells.load_instances(["b__repo-2"]) == {
        "b__repo-2": {"instance_id": "b__repo-2", "patch": "b"}
    }


def test_load_instances_uses_project_cache(tmp_path, monkeypatch):
    cache = tmp_path / "dataset.json"
    cache.write_text('[{"instance_id": "cached__repo-1", "patch": "cached"}]')
    monkeypatch.setattr(inject_smells, "DATASET_CACHE", cache)
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            load_dataset=lambda dataset_name, split: pytest.fail("must use cache")
        ),
    )

    assert inject_smells.load_instances(["cached__repo-1"]) == {
        "cached__repo-1": {"instance_id": "cached__repo-1", "patch": "cached"}
    }


def test_load_instances_reports_missing_ids(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=lambda dataset_name, split: []),
    )
    monkeypatch.setattr(inject_smells, "DATASET_CACHE", tmp_path / "dataset.json")

    with pytest.raises(ValueError, match="missing"):
        inject_smells.load_instances(["missing__repo-1"])


def test_restore_path_ownership_chmods_before_chown(monkeypatch):
    commands = []

    def fake_run(command, **kwargs):
        commands.append((list(command), kwargs.get("check", True)))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(inject_smells, "run_command", fake_run)
    monkeypatch.setattr(inject_smells, "instance_log", lambda *args, **kwargs: None)

    inject_smells.restore_path_ownership("container", "/testbed", "vuejs__core-11739")

    command, check = commands[0]
    script = command[-1]
    assert check is False
    assert "chmod -R u+rwX '/testbed'" in script
    assert script.index("chmod") < script.index("chown")
    assert f"chown -R {os.getuid()}:{os.getgid()} '/testbed'" in script


def test_cleanup_agent_output_dirs_removes_agent_artifacts(tmp_path):
    for relative in ("smell-patch", "smell-patch-intro", ".smell-output"):
        path = tmp_path / relative
        path.mkdir()
        (path / "artifact.patch").write_text("generated")

    inject_smells.cleanup_agent_output_dirs(tmp_path)

    assert not (tmp_path / "smell-patch").exists()
    assert not (tmp_path / "smell-patch-intro").exists()
    assert not (tmp_path / ".smell-output").exists()


def test_smell_json_is_the_only_completion_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(inject_smells, "PATCH_ROOT", tmp_path / "patches")
    monkeypatch.setattr(inject_smells, "INTRO_ROOT", tmp_path / "intros")
    smell = inject_smells.Smell("FeatureEnvy", "FE", tmp_path / "FeatureEnvy.md")
    paths = inject_smells.output_paths(smell, "a__repo-1")
    paths.patch.parent.mkdir(parents=True)
    paths.patch.write_text("existing patch")

    assert not paths.intro.exists()

    paths.intro.parent.mkdir(parents=True)
    paths.intro.write_text('{"Status": "FAILED"}')
    assert paths.intro.exists()


def test_process_instance_skips_existing_smell_json(tmp_path, monkeypatch):
    monkeypatch.setattr(inject_smells, "PATCH_ROOT", tmp_path / "patches")
    monkeypatch.setattr(inject_smells, "INTRO_ROOT", tmp_path / "intros")
    smell = inject_smells.Smell("GodClass", "GC", tmp_path / "GodClass.md")
    paths = inject_smells.output_paths(smell, "a__repo-1")
    paths.intro.parent.mkdir(parents=True)
    paths.intro.write_text('{"Status": "NOT_INJECTABLE"}')

    result = inject_smells.process_instance(
        {"instance_id": "a__repo-1"}, smell, "image:latest"
    )

    assert result.status == "SKIPPED"


def test_atomic_write_replaces_content(tmp_path):
    destination = tmp_path / "FeatureEnvy" / "instance-FE.json"
    inject_smells.atomic_write(destination, '{"Status": "SUCCESS"}\n')
    inject_smells.atomic_write(destination, '{"Status": "FAILED"}\n')
    assert destination.read_text() == '{"Status": "FAILED"}\n'


@pytest.mark.parametrize(
    ("report", "status"),
    [
        ('{"Status": "SUCCESS"}', "SUCCESS"),
        ('{"Status": "FAILED", "实例名称": "x"}', "FAILED"),
        ('```json\n{"Status": "NOT_INJECTABLE"}\n```', "NOT_INJECTABLE"),
        ("preface\n{\"Status\": \"SUCCESS\"}\n", "SUCCESS"),
        ("Status: SUCCESS\n", None),
        ("success", None),
    ],
)
def test_parse_status(report, status):
    assert inject_smells.parse_status(report) == status


def prepare_worker(tmp_path, monkeypatch, status):
    patch_root = tmp_path / "patches"
    intro_root = tmp_path / "intros"
    lock_root = tmp_path / "locks"
    instance = {"instance_id": "a__repo-1", "patch": "", "test_patch": ""}
    prompt = tmp_path / "FeatureEnvy.md"
    prompt.write_text("inject smell")
    smell = inject_smells.Smell("FeatureEnvy", "FE", prompt)
    calls = []

    monkeypatch.setattr(inject_smells, "PATCH_ROOT", patch_root)
    monkeypatch.setattr(inject_smells, "INTRO_ROOT", intro_root)
    monkeypatch.setattr(inject_smells, "LOCK_ROOT", lock_root)
    monkeypatch.setenv("SMELL_INJECT_TMPDIR", str(tmp_path / "workers"))

    def fake_run(command, *, input_text=None, check=True):
        command = list(command)
        calls.append(command)
        stdout = ""
        if "--binary" in command:
            stdout = (
                "diff --git a/src/Main.java b/src/Main.java\n"
                "--- a/src/Main.java\n+++ b/src/Main.java\n"
                "@@ -1 +1 @@\n-old\n+new\n"
            )
        elif "--name-status" in command:
            stdout = "M\tsrc/Main.java\n"
        elif "--porcelain" in command:
            stdout = " M src/Main.java\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    def fake_codex(prompt, workspace, instance_id=None):
        assert "a__repo-1" in prompt
        assert instance_id == "a__repo-1"
        if status == "INTERRUPTED":
            raise inject_smells.CommandError("Codex connection interrupted")
        return json.dumps({"Status": status, "实例名称": "a__repo-1"}, ensure_ascii=False)

    monkeypatch.setattr(inject_smells, "run_command", fake_run)
    monkeypatch.setattr(inject_smells, "run_codex_agent", fake_codex)
    return instance, smell, calls


def test_success_publishes_patch_and_completion_json(tmp_path, monkeypatch):
    instance, smell, calls = prepare_worker(tmp_path, monkeypatch, "SUCCESS")

    result = inject_smells.process_instance(
        instance, smell, "image:latest"
    )

    paths = inject_smells.output_paths(smell, instance["instance_id"])
    assert result.status == "SUCCESS"
    assert paths.patch.read_text().startswith("diff --git")
    assert json.loads(paths.intro.read_text())["Status"] == "SUCCESS"
    assert any(command[:3] == ["docker", "rm", "-f"] for command in calls)


@pytest.mark.parametrize("status", ["FAILED", "NOT_INJECTABLE"])
def test_normal_unsuccessful_result_publishes_only_json(tmp_path, monkeypatch, status):
    instance, smell, _ = prepare_worker(tmp_path, monkeypatch, status)

    result = inject_smells.process_instance(
        instance, smell, "image:latest"
    )

    paths = inject_smells.output_paths(smell, instance["instance_id"])
    assert result.status == status
    assert paths.intro.exists()
    assert json.loads(paths.intro.read_text())["Status"] == status
    assert not paths.patch.exists()


def test_interrupted_agent_does_not_publish_json(tmp_path, monkeypatch):
    instance, smell, calls = prepare_worker(tmp_path, monkeypatch, "INTERRUPTED")

    result = inject_smells.process_instance(
        instance, smell, "image:latest"
    )

    assert result.status == "ERROR"
    assert not inject_smells.output_paths(smell, instance["instance_id"]).intro.exists()
    assert any(command[:3] == ["docker", "rm", "-f"] for command in calls)


def test_lock_contention_does_not_start_worker(tmp_path, monkeypatch):
    instance, smell, calls = prepare_worker(tmp_path, monkeypatch, "SUCCESS")
    inject_smells.LOCK_ROOT.mkdir(parents=True)
    lock_path = inject_smells.LOCK_ROOT / f"{smell.name}-{instance['instance_id']}.lock"

    with lock_path.open("w") as lock:
        inject_smells.fcntl.flock(lock, inject_smells.fcntl.LOCK_EX)
        result = inject_smells.process_instance(
            instance, smell, "image:latest"
        )

    assert result.status == "BUSY"
    assert calls == []
