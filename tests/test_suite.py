import json
from pathlib import Path
from types import SimpleNamespace

from atlas_posthoc_plus import suite
from atlas_posthoc_plus.config import load_config
from atlas_posthoc_plus.io.chain_sweep import SWEEP_DIR_ENV, SWEEP_LIMIT_ENV
from atlas_posthoc_plus.registry import Binding, WorkflowSpec
from atlas_posthoc_plus.suite import run_suite


def test_suite_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "blocks.shp").touch()
    (project / "precincts.shp").touch()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[project]
root = "project"

[inputs]
blocks = "blocks.shp"
precincts = "precincts.shp"

[outputs]
root = "results"
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "atlas_posthoc_plus.suite.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    config = load_config(config_path)
    code = run_suite("test", ["build-crosswalk"], config)

    assert code == 0
    manifest_path = project / "results/suite-test-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["success"] is True
    assert manifest["results"][0]["workflow"] == "build-crosswalk"
    assert "-m atlas_posthoc_plus.workflows.setup.crosswalk" in (
        manifest["results"][0]["command"]
    )


def test_suite_materializes_each_chain_once_and_shares_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    chains = [project / "chain-a.jsonl", project / "chain-b.jsonl"]
    for chain in chains:
        chain.touch()
    graph = project / "graph.json"
    graph.touch()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[project]
root = "project"

[chains]
files = ["chain-a.jsonl", "chain-b.jsonl"]

[inputs]
graph_json = "graph.json"

[analysis]
max_maps_per_chain = 25

[outputs]
root = "results"

[workflow.cache-test]
extra_args = ["--thin-step", "3"]
""",
        encoding="utf-8",
    )
    spec = WorkflowSpec(
        "cache-test",
        "example.cache_test",
        "test",
        "Exercise suite chain caching.",
        (Binding("--chains", "chains.files", "paths", required=True),),
    )
    monkeypatch.setitem(suite.WORKFLOWS, spec.name, spec)

    cached: list[tuple[Path, Path, Path, int | None]] = []

    def fake_materialize(path, graph_path, cache_dir, **kwargs):
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cached.append(
            (Path(path), Path(graph_path), Path(cache_dir), kwargs.get("max_records"))
        )
        return False

    environments = []
    monkeypatch.setattr(suite, "materialize_chain_sweep", fake_materialize)

    def fake_run(*args, **kwargs):
        assert Path(kwargs["env"][SWEEP_DIR_ENV]).is_dir()
        environments.append(kwargs["env"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        suite.subprocess,
        "run",
        fake_run,
    )

    code = run_suite("test", [spec.name], load_config(config_path))

    assert code == 0
    assert [item[0] for item in cached] == [path.resolve() for path in chains]
    assert all(item[1] == graph.resolve() for item in cached)
    assert all(item[3] == 73 for item in cached)
    assert environments[0][SWEEP_LIMIT_ENV] == "73"
    assert environments[0][SWEEP_DIR_ENV].endswith("results/.atlas-chain-cache")
    assert not (project / "results/.atlas-chain-cache").exists()


def test_suite_removes_chain_cache_after_workflow_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "chain.jsonl").touch()
    (project / "graph.json").touch()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[project]
root = "project"

[chains]
files = ["chain.jsonl"]

[inputs]
graph_json = "graph.json"

[outputs]
root = "results"
""",
        encoding="utf-8",
    )
    spec = WorkflowSpec(
        "cache-failure-test",
        "example.cache_failure_test",
        "test",
        "Exercise cleanup after a failed workflow.",
        (Binding("--chains", "chains.files", "paths", required=True),),
    )
    monkeypatch.setitem(suite.WORKFLOWS, spec.name, spec)

    def fake_materialize(path, graph_path, cache_dir, **kwargs):
        destination = Path(cache_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "large-cache.pickle").touch()
        return False

    monkeypatch.setattr(suite, "materialize_chain_sweep", fake_materialize)
    monkeypatch.setattr(
        suite.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=9),
    )

    code = run_suite("test", [spec.name], load_config(config_path))

    assert code == 9
    assert not (project / "results/.atlas-chain-cache").exists()
